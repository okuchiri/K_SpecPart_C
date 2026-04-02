#include "kspecpart/triton_refiner.hpp"

#include "kspecpart/external_tools.hpp"
#include "kspecpart/io.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
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

std::string unique_token(const std::string& label, int run_id) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<std::uint64_t>(now) ^
                        (static_cast<std::uint64_t>(::getpid()) << 16U) ^
                        static_cast<std::uint64_t>(run_id));
    std::uniform_int_distribution<unsigned long long> dist;
    std::ostringstream oss;
    oss << label << '-' << run_id << '-' << ::getpid() << '-' << now << '-' << dist(rng);
    return oss.str();
}

fs::path make_base_path(const std::string& work_prefix, int run_id) {
    fs::path base;
    if (work_prefix.empty()) {
        base = fs::temp_directory_path() / unique_token("triton", run_id);
    } else {
        base = fs::path(work_prefix + "." + unique_token("triton", run_id));
    }
    std::error_code ec;
    fs::create_directories(base.parent_path(), ec);
    return base;
}

}  // namespace

std::optional<std::string> resolve_triton_refiner_executable(const std::string& configured_path) {
    return resolve_tool_path(configured_path,
                             "openroad",
                             {"/home/norising/TritonPart_OpenROAD/build/src/openroad",
                              "/home/norising/OpenROAD/build/src/openroad"},
                             ToolPathMode::kExecutable);
}

std::optional<std::vector<int>> run_triton_refiner(const std::string& hypergraph_file,
                                                   const std::vector<int>& partition,
                                                   const TritonRefinerOptions& options,
                                                   const std::string& work_prefix,
                                                   int run_id) {
    const std::optional<std::string> executable =
        resolve_triton_refiner_executable(options.refiner_executable);
    if (options.explicit_path && !executable.has_value()) {
        throw std::runtime_error("failed to resolve triton refiner executable: " +
                                 options.refiner_executable);
    }
    if (!executable.has_value()) {
        return std::nullopt;
    }

    const fs::path base_path = make_base_path(work_prefix, run_id);
    const std::string partition_file =
        base_path.string() + ".part." + std::to_string(options.num_parts);
    const std::string tcl_file = base_path.string() + ".tcl";
    const std::string log_file = base_path.string() + ".log";

    write_partition_file(partition_file, partition);
    {
        std::ofstream tcl_output(tcl_file);
        if (!tcl_output) {
            throw std::runtime_error("failed to open triton tcl file: " + tcl_file);
        }
        tcl_output << "triton_part_refine"
                   << " -hypergraph_file " << hypergraph_file
                   << " -partition_file " << partition_file
                   << " -num_parts " << options.num_parts
                   << " -ub_factor " << options.imb
                   << " -seed " << options.seed << '\n';
        tcl_output << "exit\n";
    }

    ExternalCommand command;
    command.argv = {*executable, tcl_file};
    command.stdout_file = log_file;
    command.redirect_stderr_to_stdout = true;
    const bool ok = run_external_command(command);
    std::optional<std::vector<int>> refined =
        ok ? std::optional<std::vector<int>>(read_partition_file(partition_file)) : std::nullopt;

    std::error_code ec;
    fs::remove(partition_file, ec);
    fs::remove(tcl_file, ec);
    fs::remove(log_file, ec);

    if (!refined.has_value() ||
        static_cast<int>(refined->size()) != static_cast<int>(partition.size())) {
        return std::nullopt;
    }
    return refined;
}

}  // namespace kspecpart
