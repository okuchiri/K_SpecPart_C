#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace kspecpart {

enum class ToolPathMode {
    kExecutable,
    kRegularFile,
};

struct ExternalCommand {
    std::vector<std::string> argv;
    std::vector<std::pair<std::string, std::string>> env;
    std::optional<std::string> stdout_file;
    bool redirect_stderr_to_stdout = false;
};

std::string trim_copy(std::string value);
bool is_regular_file(const std::filesystem::path& path);
bool is_executable_file(const std::filesystem::path& path);
std::optional<std::string> resolve_tool_path(const std::string& configured_path,
                                             const std::string& executable_name,
                                             const std::vector<std::filesystem::path>& defaults,
                                             ToolPathMode mode = ToolPathMode::kExecutable);
std::optional<std::string> resolve_linux32_loader();
std::string shell_quote(const std::string& value);
bool run_external_command(const ExternalCommand& command);
std::vector<std::string> discover_ortools_library_paths(const std::string& executable_path);
std::string merge_path_environment(const std::string& env_name,
                                   const std::vector<std::string>& preferred_paths);

}  // namespace kspecpart
