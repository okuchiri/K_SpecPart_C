#include "kspecpart/metis.hpp"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

namespace kspecpart {

namespace {

bool is_executable(const std::filesystem::path& path) {
    std::error_code ec;
    const auto status = std::filesystem::status(path, ec);
    if (ec || !std::filesystem::exists(status) || std::filesystem::is_directory(status)) {
        return false;
    }
    return ::access(path.c_str(), X_OK) == 0;
}

std::optional<std::filesystem::path> search_path_for_executable(const std::string& executable) {
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) {
        return std::nullopt;
    }

    std::stringstream path_stream(path_env);
    std::string entry;
    while (std::getline(path_stream, entry, ':')) {
        if (entry.empty()) {
            continue;
        }
        std::filesystem::path candidate = std::filesystem::path(entry) / executable;
        if (is_executable(candidate)) {
            return candidate;
        }
    }
    return std::nullopt;
}

int count_undirected_edges(const WeightedGraph& graph) {
    int edges = 0;
    for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
        for (const auto& [neighbor, weight] : graph.adjacency[vertex]) {
            (void)weight;
            if (vertex < neighbor) {
                ++edges;
            }
        }
    }
    return edges;
}

int metis_integer_weight(double weight) {
    if (!std::isfinite(weight) || weight <= 0.0) {
        return 1;
    }
    const double floored = std::floor(weight);
    if (floored < 1.0) {
        return 1;
    }
    if (floored > static_cast<double>(INT_MAX)) {
        return INT_MAX;
    }
    return static_cast<int>(floored);
}

std::filesystem::path make_temp_graph_path() {
    const auto temp_dir = std::filesystem::temp_directory_path();
    const auto pid = static_cast<unsigned long long>(::getpid());
    for (unsigned long long suffix = 0; suffix < 10000; ++suffix) {
        auto candidate = temp_dir / ("kspecpart_metis_" + std::to_string(pid) + "_" + std::to_string(suffix) + ".gr");
        if (!std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("failed to allocate a temporary METIS graph path");
}

void write_metis_graph_file(const WeightedGraph& graph, const std::filesystem::path& path) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open METIS graph file: " + path.string());
    }

    output << graph.num_vertices << ' ' << count_undirected_edges(graph) << " 001\n";
    for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
        bool first = true;
        for (const auto& [neighbor, weight] : graph.adjacency[vertex]) {
            if (neighbor == vertex) {
                continue;
            }
            if (!first) {
                output << ' ';
            }
            first = false;
            output << (neighbor + 1) << ' ' << metis_integer_weight(weight);
        }
        output << '\n';
    }
}

bool run_process(const std::string& executable, const std::vector<std::string>& args) {
    std::vector<char*> argv;
    argv.reserve(args.size() + 2);
    argv.push_back(const_cast<char*>(executable.c_str()));
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid < 0) {
        return false;
    }
    if (pid == 0) {
        const int null_fd = ::open("/dev/null", O_WRONLY);
        if (null_fd >= 0) {
            ::dup2(null_fd, STDOUT_FILENO);
            ::dup2(null_fd, STDERR_FILENO);
            ::close(null_fd);
        }
        ::execv(executable.c_str(), argv.data());
        _exit(errno == ENOENT ? 127 : 126);
    }

    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

std::optional<std::vector<int>> read_metis_partition_file(const std::filesystem::path& path, int expected_vertices) {
    std::ifstream input(path);
    if (!input) {
        return std::nullopt;
    }

    std::vector<int> partition;
    partition.reserve(expected_vertices);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        partition.push_back(std::stoi(line));
    }
    if (static_cast<int>(partition.size()) != expected_vertices) {
        return std::nullopt;
    }
    return partition;
}

}  // namespace

std::optional<std::string> resolve_gpmetis_executable(const std::string& configured_path) {
    if (!configured_path.empty()) {
        std::filesystem::path candidate(configured_path);
        std::error_code ec;
        if (std::filesystem::is_directory(candidate, ec)) {
            candidate /= "gpmetis";
        }
        if (is_executable(candidate)) {
            return candidate.string();
        }
        return std::nullopt;
    }

    const auto discovered = search_path_for_executable("gpmetis");
    if (!discovered.has_value()) {
        return std::nullopt;
    }
    return discovered->string();
}

std::optional<std::vector<int>> run_gpmetis_partition(const WeightedGraph& graph,
                                                      int num_parts,
                                                      const std::string& gpmetis_executable,
                                                      int ufactor,
                                                      int seed) {
    if (graph.num_vertices <= 0 || num_parts <= 0) {
        return std::nullopt;
    }

    const std::filesystem::path graph_file = make_temp_graph_path();
    const std::filesystem::path partition_file =
        std::filesystem::path(graph_file.string() + ".part." + std::to_string(num_parts));

    try {
        write_metis_graph_file(graph, graph_file);
        const bool success = run_process(
            gpmetis_executable,
            {graph_file.string(),
             std::to_string(num_parts),
             "-ptype=rb",
             "-ufactor=" + std::to_string(std::max(0, ufactor)),
             "-seed=" + std::to_string(seed),
             "-dbglvl=0"});
        if (!success) {
            std::error_code ec;
            std::filesystem::remove(graph_file, ec);
            std::filesystem::remove(partition_file, ec);
            return std::nullopt;
        }

        auto partition = read_metis_partition_file(partition_file, graph.num_vertices);
        std::error_code ec;
        std::filesystem::remove(graph_file, ec);
        std::filesystem::remove(partition_file, ec);
        return partition;
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(graph_file, ec);
        std::filesystem::remove(partition_file, ec);
        throw;
    }
}

}  // namespace kspecpart
