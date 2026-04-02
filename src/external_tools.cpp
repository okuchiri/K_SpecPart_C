#include "kspecpart/external_tools.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <unistd.h>

namespace kspecpart {

namespace {

namespace fs = std::filesystem;

std::vector<fs::path> path_search_candidates(const std::string& executable_name) {
    std::vector<fs::path> candidates;
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) {
        return candidates;
    }

    std::stringstream ss(path_env);
    std::string item;
    while (std::getline(ss, item, ':')) {
        if (!item.empty()) {
            candidates.push_back(fs::path(item) / executable_name);
        }
    }
    return candidates;
}

void append_unique_path(std::vector<std::string>& paths,
                        std::unordered_set<std::string>& seen,
                        const fs::path& path) {
    std::error_code ec;
    if (!fs::exists(path, ec) || !fs::is_directory(path, ec)) {
        return;
    }
    const std::string value = path.string();
    if (!seen.insert(value).second) {
        return;
    }
    paths.push_back(value);
}

void append_unique_entry(std::vector<std::string>& paths,
                         std::unordered_set<std::string>& seen,
                         const std::string& entry) {
    if (entry.empty()) {
        return;
    }
    if (!seen.insert(entry).second) {
        return;
    }
    paths.push_back(entry);
}

void append_if_contains_ortools(std::vector<std::string>& paths,
                                std::unordered_set<std::string>& seen,
                                const fs::path& dir) {
    std::error_code ec;
    if (!fs::exists(dir / "libortools.so.9", ec) &&
        !fs::exists(dir / "libortools.so", ec)) {
        return;
    }
    append_unique_path(paths, seen, dir);
}

void append_from_cache_file(std::vector<std::string>& paths,
                            std::unordered_set<std::string>& seen,
                            const fs::path& cache_file) {
    std::ifstream input(cache_file);
    if (!input) {
        return;
    }

    std::string line;
    while (std::getline(input, line)) {
        const std::string prefix = "ortools_DIR:PATH=";
        if (line.rfind(prefix, 0) != 0) {
            continue;
        }

        const fs::path ortools_dir = trim_copy(line.substr(prefix.size()));
        if (ortools_dir.empty()) {
            continue;
        }

        append_if_contains_ortools(paths, seen, ortools_dir.parent_path().parent_path());
        append_if_contains_ortools(paths, seen, ortools_dir.parent_path().parent_path().parent_path() / "lib");
        append_if_contains_ortools(paths, seen, ortools_dir.parent_path().parent_path().parent_path() / "lib64");
    }
}

std::vector<std::string> split_path_environment(const char* value) {
    std::vector<std::string> entries;
    if (value == nullptr) {
        return entries;
    }

    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ':')) {
        if (!item.empty()) {
            entries.push_back(item);
        }
    }
    return entries;
}

}  // namespace

std::string trim_copy(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

bool is_regular_file(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec) && fs::is_regular_file(path, ec);
}

bool is_executable_file(const fs::path& path) {
    return kspecpart::is_regular_file(path) && ::access(path.c_str(), X_OK) == 0;
}

std::optional<std::string> resolve_tool_path(const std::string& configured_path,
                                             const std::string& executable_name,
                                             const std::vector<fs::path>& defaults,
                                             ToolPathMode mode) {
    const std::string trimmed = trim_copy(configured_path);
    std::vector<fs::path> candidates;
    if (!trimmed.empty()) {
        const fs::path configured(trimmed);
        if (fs::is_directory(configured)) {
            candidates.push_back(configured / executable_name);
        }
        candidates.push_back(configured);
    } else {
        candidates.insert(candidates.end(), defaults.begin(), defaults.end());
        const std::vector<fs::path> path_candidates = path_search_candidates(executable_name);
        candidates.insert(candidates.end(), path_candidates.begin(), path_candidates.end());
    }

    for (const fs::path& candidate : candidates) {
        const bool ok = mode == ToolPathMode::kExecutable ? is_executable_file(candidate)
                                                          : kspecpart::is_regular_file(candidate);
        if (ok) {
            return candidate.string();
        }
    }
    return std::nullopt;
}

std::optional<std::string> resolve_linux32_loader() {
    for (const fs::path& candidate : {fs::path("/lib/ld-linux.so.2"),
                                      fs::path("/lib32/ld-linux.so.2"),
                                      fs::path("/lib/i386-linux-gnu/ld-linux.so.2")}) {
        if (kspecpart::is_regular_file(candidate)) {
            return candidate.string();
        }
    }
    return std::nullopt;
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

bool run_external_command(const ExternalCommand& command) {
    if (command.argv.empty()) {
        return false;
    }

    std::string shell_command;
    for (const auto& [name, value] : command.env) {
        if (!name.empty() && !value.empty()) {
            shell_command += name + "=" + shell_quote(value) + " ";
        }
    }
    for (std::size_t i = 0; i < command.argv.size(); ++i) {
        if (i > 0) {
            shell_command += ' ';
        }
        shell_command += shell_quote(command.argv[i]);
    }
    if (command.stdout_file.has_value()) {
        shell_command += " > " + shell_quote(*command.stdout_file);
        if (command.redirect_stderr_to_stdout) {
            shell_command += " 2>&1";
        }
    } else if (command.redirect_stderr_to_stdout) {
        shell_command += " 2>&1";
    }
    return std::system(shell_command.c_str()) == 0;
}

std::vector<std::string> discover_ortools_library_paths(const std::string& executable_path) {
    std::vector<std::string> paths;
    std::unordered_set<std::string> seen;

    for (const std::string& entry : split_path_environment(std::getenv("LD_LIBRARY_PATH"))) {
        append_if_contains_ortools(paths, seen, fs::path(entry));
    }

    for (const fs::path& candidate : {
             fs::path("/home/norising/or-tools-9.4/install_make/lib"),
             fs::path("/home/norising/or-tools-9.4/build_make/lib"),
             fs::path("/home/norising/or-tools-stable/install_make/lib"),
             fs::path("/home/norising/or-tools-stable/build_make/lib"),
             fs::path("/home/tool/ortools/install/CentOS7-gcc9/lib64"),
             fs::path("/home/tool/ortools/install/CentOS7-gcc9/lib")}) {
        append_if_contains_ortools(paths, seen, candidate);
    }

    const fs::path executable(executable_path);
    for (const fs::path& cache_file : {
             executable.parent_path() / "CMakeCache.txt",
             executable.parent_path().parent_path() / "build/CMakeCache.txt",
             executable.parent_path().parent_path() / "ilp_partitioner/build/CMakeCache.txt"}) {
        append_from_cache_file(paths, seen, cache_file);
    }

    return paths;
}

std::string merge_path_environment(const std::string& env_name,
                                   const std::vector<std::string>& preferred_paths) {
    std::vector<std::string> merged;
    std::unordered_set<std::string> seen;
    for (const std::string& path : preferred_paths) {
        append_unique_entry(merged, seen, path);
    }
    for (const std::string& path : split_path_environment(std::getenv(env_name.c_str()))) {
        append_unique_entry(merged, seen, path);
    }

    std::string joined;
    for (std::size_t i = 0; i < merged.size(); ++i) {
        if (i > 0) {
            joined.push_back(':');
        }
        joined += merged[i];
    }
    return joined;
}

}  // namespace kspecpart
