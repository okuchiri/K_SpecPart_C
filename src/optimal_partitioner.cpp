#include "kspecpart/optimal_partitioner.hpp"

#include "kspecpart/external_tools.hpp"
#include "kspecpart/golden_evaluator.hpp"
#include "kspecpart/io.hpp"
#include "kspecpart/tree_partition.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

namespace kspecpart {

namespace {

namespace fs = std::filesystem;

bool has_prefix(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

std::string unique_token(const std::string& label) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<std::uint64_t>(now) ^ static_cast<std::uint64_t>(::getpid()));
    std::uniform_int_distribution<unsigned long long> dist;
    std::ostringstream oss;
    oss << label << '-' << ::getpid() << '-' << now << '-' << dist(rng);
    return oss.str();
}

fs::path make_base_path(const std::string& work_prefix, const std::string& label) {
    fs::path base;
    if (work_prefix.empty()) {
        base = fs::temp_directory_path() / unique_token(label);
    } else {
        fs::path base_dir(work_prefix);
        std::error_code ec;
        fs::create_directories(base_dir, ec);
        if (label == "ilp" || label == "hmetis") {
            return base_dir / "coarse.hgr";
        }
        const std::string parallel_prefix = "hmetis-par-";
        if (has_prefix(label, parallel_prefix)) {
            const int run_id = std::stoi(label.substr(parallel_prefix.size()));
            return base_dir / ("coarse.hgr." + std::to_string(run_id + 1));
        }
        return base_dir / label;
    }
    std::error_code ec;
    fs::create_directories(base.parent_path(), ec);
    return base;
}

std::optional<fs::path> resolve_debug_external_dir() {
    const char* raw = std::getenv("K_SPECPART_DEBUG_EXTERNAL_DIR");
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    fs::path dir(raw);
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        return std::nullopt;
    }
    return dir;
}

void persist_debug_artifact(const fs::path& source, const std::string& label) {
    const std::optional<fs::path> debug_dir = resolve_debug_external_dir();
    if (!debug_dir.has_value()) {
        return;
    }
    std::error_code ec;
    if (!fs::exists(source, ec) || ec) {
        return;
    }
    fs::path destination =
        *debug_dir / (unique_token(label) + source.extension().string());
    fs::copy_file(source, destination, fs::copy_options::overwrite_existing, ec);
}

bool julia_small_ilp_edge_case(const Hypergraph& hypergraph, int num_parts) {
    return (hypergraph.num_hyperedges < 1500 && num_parts == 2) ||
           (hypergraph.num_hyperedges < 300 && num_parts > 2);
}

bool structurally_degenerate_overlay_case(const Hypergraph& hypergraph, int num_parts) {
    const int tiny_edge_budget = std::max(1, num_parts - 1);
    return hypergraph.num_hyperedges <= tiny_edge_budget &&
           hypergraph.num_vertices > 10000;
}

bool is_balanced_partition(const Hypergraph& hypergraph,
                           const std::vector<int>& partition,
                           int num_parts,
                           int imb) {
    if (static_cast<int>(partition.size()) != hypergraph.num_vertices) {
        return false;
    }
    const BalanceLimits limits = compute_balance_limits(hypergraph, num_parts, imb);
    const PartitionResult metrics = evaluate_partition(hypergraph, num_parts, partition);
    return balance_penalty(metrics.balance, limits) == 0;
}

std::string read_method_marker(const std::string& file_name) {
    std::ifstream input(file_name);
    if (!input) {
        return "";
    }
    std::string line;
    std::getline(input, line);
    return trim_copy(line);
}

std::optional<std::vector<int>> read_partition_if_valid(const std::string& file_name, int expected_vertices) {
    std::error_code ec;
    if (!fs::exists(file_name, ec)) {
        return std::nullopt;
    }
    std::vector<int> partition = read_partition_file(file_name);
    if (static_cast<int>(partition.size()) != expected_vertices) {
        return std::nullopt;
    }
    return partition;
}

struct ExternalPartitionAttempt {
    std::vector<int> partition;
    std::string method;
};

std::optional<ExternalPartitionAttempt> run_ilp_partitioner_once(const Hypergraph& hypergraph,
                                                                 const std::string& executable,
                                                                 int num_parts,
                                                                 int imb,
                                                                 const fs::path& hypergraph_file_path) {
    const std::string hypergraph_file = hypergraph_file_path.string();
    const std::string partition_file = hypergraph_file + ".part." + std::to_string(num_parts);
    const std::string method_file = partition_file + ".method";
    write_hypergraph_file(hypergraph_file, hypergraph);

    ExternalCommand command;
    command.argv = {executable, hypergraph_file, std::to_string(num_parts), std::to_string(imb)};
    const std::vector<std::string> ortools_paths = discover_ortools_library_paths(executable);
    const std::string merged_library_path = merge_path_environment("LD_LIBRARY_PATH", ortools_paths);
    if (!merged_library_path.empty()) {
        command.env.push_back({"LD_LIBRARY_PATH", merged_library_path});
    }
    command.stdout_file = "/dev/null";
    command.redirect_stderr_to_stdout = true;
    const bool ok = run_external_command(command);
    std::optional<std::vector<int>> partition =
        ok ? read_partition_if_valid(partition_file, hypergraph.num_vertices) : std::nullopt;
    const std::string method_marker = read_method_marker(method_file);

    persist_debug_artifact(hypergraph_file, "overlay-ilp-input");
    persist_debug_artifact(partition_file, "overlay-ilp-output");
    persist_debug_artifact(method_file, "overlay-ilp-method");

    std::error_code ec;
    fs::remove(hypergraph_file, ec);
    fs::remove(partition_file, ec);
    fs::remove(method_file, ec);
    if (!partition.has_value()) {
        return std::nullopt;
    }
    return ExternalPartitionAttempt{*partition, method_marker.empty() ? "ilp" : method_marker};
}

std::optional<std::vector<int>> run_hmetis_once(const Hypergraph& hypergraph,
                                                const std::string& executable,
                                                int num_parts,
                                                int imb,
                                                int runs,
                                                const fs::path& hypergraph_file);

std::optional<std::vector<int>> run_hmetis_with_existing_input(const std::string& executable,
                                                               int expected_vertices,
                                                               int num_parts,
                                                               int imb,
                                                               int runs,
                                                               const fs::path& hypergraph_file) {
    const std::string partition_file = hypergraph_file.string() + ".part." + std::to_string(num_parts);
    ExternalCommand command;
    if (is_executable_file(executable)) {
        command.argv.push_back(executable);
    } else {
        const std::optional<std::string> loader = resolve_linux32_loader();
        if (!loader.has_value()) {
            return std::nullopt;
        }
        command.argv.push_back(*loader);
        command.argv.push_back(executable);
    }
    command.argv.push_back(hypergraph_file.string());
    command.argv.push_back(std::to_string(num_parts));
    command.argv.push_back(std::to_string(imb));
    command.argv.push_back(std::to_string(runs));
    command.argv.push_back("1");
    command.argv.push_back("1");
    command.argv.push_back("1");
    command.argv.push_back("0");
    command.argv.push_back("0");
    // hMETIS is an old 32-bit external binary. When it crashes, it may emit
    // raw stderr such as "Bad system call (core dumped)" even though we already
    // detect failure from the exit status / missing partition file.
    command.stdout_file = "/dev/null";
    command.redirect_stderr_to_stdout = true;
    const bool ok = run_external_command(command);
    std::optional<std::vector<int>> partition =
        ok ? read_partition_if_valid(partition_file, expected_vertices) : std::nullopt;

    persist_debug_artifact(hypergraph_file, "overlay-hmetis-input");
    persist_debug_artifact(partition_file, "overlay-hmetis-output");

    std::error_code ec;
    fs::remove(hypergraph_file, ec);
    fs::remove(partition_file, ec);
    return partition;
}

std::optional<std::vector<int>> run_hmetis_once(const Hypergraph& hypergraph,
                                                const std::string& executable,
                                                int num_parts,
                                                int imb,
                                                int runs,
                                                const fs::path& hypergraph_file) {
    write_hypergraph_file(hypergraph_file.string(), hypergraph);
    return run_hmetis_with_existing_input(
        executable, hypergraph.num_vertices, num_parts, imb, runs, hypergraph_file);
}

}  // namespace

std::optional<std::string> resolve_hmetis_executable(const std::string& configured_path) {
    const std::optional<std::string> executable = resolve_tool_path(configured_path,
                                                                    "hmetis",
                                                                    {"/home/norising/K_SpecPart_C/scripts/hmetis_wrapper.sh",
                                                                     "/home/norising/hmetis-1.5-linux/hmetis"},
                                                                    ToolPathMode::kRegularFile);
    if (!executable.has_value()) {
        return std::nullopt;
    }
    if (is_executable_file(*executable) || resolve_linux32_loader().has_value()) {
        return executable;
    }
    return std::nullopt;
}

std::optional<std::string> resolve_ilp_partitioner_executable(const std::string& configured_path) {
    return resolve_tool_path(configured_path,
                             "ilp_part",
                             {"/home/norising/K_SpecPart_C/scripts/julia_ilp_wrapper.sh",
                              "/home/norising/K_SpecPart/ilp_partitioner/build/ilp_part",
                              "/home/norising/K_SpecPart/ilp_partitioner/ilp_partitioner"},
                             ToolPathMode::kExecutable);
}

std::optional<std::vector<int>> run_hmetis_initial_partition(const Hypergraph& hypergraph,
                                                             const std::string& executable,
                                                             int num_parts,
                                                             int imb,
                                                             const std::string& work_prefix,
                                                             int runs) {
    if (hypergraph.num_vertices == 0) {
        return std::vector<int>{};
    }
    const fs::path base_path = make_base_path(work_prefix, "initial-hint");
    return run_hmetis_once(hypergraph,
                           executable,
                           num_parts,
                           imb,
                           std::max(1, runs),
                           base_path);
}

std::optional<std::string> optimal_partitioner_skip_reason(const Hypergraph& hypergraph,
                                                           const OptimalPartitionerOptions& options) {
    if (structurally_degenerate_overlay_case(hypergraph, options.num_parts) &&
        julia_small_ilp_edge_case(hypergraph, options.num_parts)) {
        return std::string("skipped degenerate contracted overlay")
             + " (very few hyperedges but many vertices)";
    }
    return std::nullopt;
}

std::optional<OptimalPartitionerResult> run_optimal_partitioner(const Hypergraph& hypergraph,
                                                                const OptimalPartitionerOptions& options,
                                                                const std::string& work_prefix) {
    if (hypergraph.num_vertices == 0) {
        return OptimalPartitionerResult{{}, "empty"};
    }

    const std::optional<std::string> ilp_executable =
        options.enable_ilp ? resolve_ilp_partitioner_executable(options.ilp_partitioner_executable) : std::nullopt;
    if (options.enable_ilp && options.ilp_partitioner_explicit && !ilp_executable.has_value()) {
        throw std::runtime_error("failed to resolve ilp partitioner executable: " +
                                 options.ilp_partitioner_executable);
    }

    const std::optional<std::string> hmetis_executable =
        options.enable_hmetis ? resolve_hmetis_executable(options.hmetis_executable) : std::nullopt;
    if (options.enable_hmetis && options.hmetis_explicit && !hmetis_executable.has_value()) {
        throw std::runtime_error("failed to resolve hmetis executable: " + options.hmetis_executable);
    }

    const bool small_ilp_case =
        julia_small_ilp_edge_case(hypergraph, options.num_parts);

    if (small_ilp_case && ilp_executable.has_value()) {
        const fs::path base_path = make_base_path(work_prefix, "ilp");
        std::optional<ExternalPartitionAttempt> ilp_partition =
            run_ilp_partitioner_once(hypergraph, *ilp_executable, options.num_parts, options.imb, base_path);
        if (ilp_partition.has_value()) {
            const PartitionResult metrics =
                evaluate_partition(hypergraph, options.num_parts, ilp_partition->partition);
            if (is_balanced_partition(hypergraph, ilp_partition->partition, options.num_parts, options.imb) &&
                metrics.cutsize > 0) {
                return OptimalPartitionerResult{ilp_partition->partition, ilp_partition->method};
            }
        }
    }

    if (small_ilp_case) {
        if (!hmetis_executable.has_value()) {
            return std::nullopt;
        }
        const fs::path base_path = make_base_path(work_prefix, "hmetis");
        std::optional<std::vector<int>> partition =
            run_hmetis_once(hypergraph,
                            *hmetis_executable,
                            options.num_parts,
                            options.imb,
                            options.hmetis_runs,
                            base_path);
        if (partition.has_value()) {
            return OptimalPartitionerResult{*partition, "hmetis"};
        }
        return std::nullopt;
    }

    if (!hmetis_executable.has_value()) {
        return std::nullopt;
    }

    const int parallel_runs = std::max(1, options.parallel_runs);
    const fs::path shared_hypergraph_file = make_base_path(work_prefix, "hmetis");
    write_hypergraph_file(shared_hypergraph_file.string(), hypergraph);
    int best_cutsize = std::numeric_limits<int>::max();
    std::vector<int> best_partition;
    for (int run = 0; run < parallel_runs; ++run) {
        const fs::path local_hypergraph_file =
            make_base_path(work_prefix, "hmetis-par-" + std::to_string(run));
        std::error_code ec;
        fs::copy_file(shared_hypergraph_file,
                      local_hypergraph_file,
                      fs::copy_options::overwrite_existing,
                      ec);
        if (ec) {
            continue;
        }
        // Match the current Julia reference environment more closely. There,
        // `Threads.@threads` is effectively serial because `Threads.nthreads()`
        // is 1, so the 10 overlay hMETIS attempts are launched one after
        // another rather than all at once.
        const std::optional<std::vector<int>> partition =
            run_hmetis_with_existing_input(*hmetis_executable,
                                           hypergraph.num_vertices,
                                           options.num_parts,
                                           options.imb,
                                           options.hmetis_runs,
                                           local_hypergraph_file);
        if (!partition.has_value()) {
            continue;
        }
        const PartitionResult metrics =
            evaluate_partition(hypergraph, options.num_parts, *partition);
        if (metrics.cutsize < best_cutsize) {
            best_cutsize = metrics.cutsize;
            best_partition = *partition;
        }
    }

    if (!best_partition.empty()) {
        std::error_code ec;
        fs::remove(shared_hypergraph_file, ec);
        return OptimalPartitionerResult{best_partition, "hmetis"};
    }
    std::error_code ec;
    fs::remove(shared_hypergraph_file, ec);
    return std::nullopt;
}

}  // namespace kspecpart
