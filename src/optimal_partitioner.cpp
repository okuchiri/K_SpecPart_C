#include "kspecpart/optimal_partitioner.hpp"

#include "kspecpart/external_tools.hpp"
#include "kspecpart/golden_evaluator.hpp"
#include "kspecpart/io.hpp"
#include "kspecpart/tree_partition.hpp"

#include <chrono>
#include <filesystem>
#include <future>
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
        base = fs::path(work_prefix + "." + unique_token(label));
    }
    std::error_code ec;
    fs::create_directories(base.parent_path(), ec);
    return base;
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

std::optional<std::vector<int>> run_ilp_partitioner_once(const Hypergraph& hypergraph,
                                                         const std::string& executable,
                                                         int num_parts,
                                                         int imb,
                                                         const fs::path& base_path) {
    const std::string hypergraph_file = base_path.string() + ".hgr";
    const std::string partition_file = hypergraph_file + ".part." + std::to_string(num_parts);
    write_hypergraph_file(hypergraph_file, hypergraph);

    ExternalCommand command;
    command.argv = {executable, hypergraph_file, std::to_string(num_parts), std::to_string(imb)};
    const std::vector<std::string> ortools_paths = discover_ortools_library_paths(executable);
    const std::string merged_library_path = merge_path_environment("LD_LIBRARY_PATH", ortools_paths);
    if (!merged_library_path.empty()) {
        command.env.push_back({"LD_LIBRARY_PATH", merged_library_path});
    }
    const bool ok = run_external_command(command);
    std::optional<std::vector<int>> partition =
        ok ? read_partition_if_valid(partition_file, hypergraph.num_vertices) : std::nullopt;

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
    const bool ok = run_external_command(command);
    std::optional<std::vector<int>> partition =
        ok ? read_partition_if_valid(partition_file, hypergraph.num_vertices) : std::nullopt;

    std::error_code ec;
    fs::remove(hypergraph_file, ec);
    fs::remove(partition_file, ec);
    return partition;
}

}  // namespace

std::optional<std::string> resolve_hmetis_executable(const std::string& configured_path) {
    const std::optional<std::string> executable = resolve_tool_path(configured_path,
                                                                    "hmetis",
                                                                    {"/home/norising/hmetis-1.5-linux/hmetis"},
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
                             {"/home/norising/K_SpecPart/ilp_partitioner/build/ilp_part",
                              "/home/norising/K_SpecPart/ilp_partitioner/ilp_partitioner"},
                             ToolPathMode::kExecutable);
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
        (hypergraph.num_hyperedges < 1500 && options.num_parts == 2) ||
        (hypergraph.num_hyperedges < 300 && options.num_parts > 2);

    if (small_ilp_case && ilp_executable.has_value()) {
        const fs::path base_path = make_base_path(work_prefix, "ilp");
        std::optional<std::vector<int>> ilp_partition =
            run_ilp_partitioner_once(hypergraph, *ilp_executable, options.num_parts, options.imb, base_path);
        if (ilp_partition.has_value()) {
            const PartitionResult metrics =
                evaluate_partition(hypergraph, options.num_parts, *ilp_partition);
            if (is_balanced_partition(hypergraph, *ilp_partition, options.num_parts, options.imb) &&
                metrics.cutsize > 0) {
                return OptimalPartitionerResult{*ilp_partition, "ilp"};
            }
        }
    }

    if (!hmetis_executable.has_value()) {
        return std::nullopt;
    }

    if (small_ilp_case) {
        const fs::path base_path = make_base_path(work_prefix, "hmetis");
        std::optional<std::vector<int>> partition =
            run_hmetis_once(hypergraph,
                            *hmetis_executable,
                            options.num_parts,
                            options.imb,
                            options.hmetis_runs,
                            fs::path(base_path.string() + ".hgr"));
        if (partition.has_value()) {
            return OptimalPartitionerResult{*partition, "hmetis"};
        }
        return std::nullopt;
    }

    const int parallel_runs = std::max(1, options.parallel_runs);
    std::vector<std::future<std::optional<std::vector<int>>>> futures;
    futures.reserve(parallel_runs);
    for (int run = 0; run < parallel_runs; ++run) {
        futures.push_back(std::async(std::launch::async,
                                     [&, run]() -> std::optional<std::vector<int>> {
                                         const fs::path base_path = make_base_path(
                                             work_prefix, "hmetis-par-" + std::to_string(run));
                                         return run_hmetis_once(hypergraph,
                                                                *hmetis_executable,
                                                                options.num_parts,
                                                                options.imb,
                                                                options.hmetis_runs,
                                                                fs::path(base_path.string() + ".hgr"));
                                     }));
    }

    int best_cutsize = std::numeric_limits<int>::max();
    std::vector<int> best_partition;
    for (auto& future : futures) {
        const std::optional<std::vector<int>> partition = future.get();
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
        return OptimalPartitionerResult{best_partition, "hmetis"};
    }
    return std::nullopt;
}

}  // namespace kspecpart
