#pragma once

#include "kspecpart/definitions.hpp"

#include <optional>
#include <string>
#include <vector>

namespace kspecpart {

std::optional<std::string> resolve_gpmetis_executable(const std::string& configured_path);

std::optional<std::vector<int>> run_gpmetis_partition(const WeightedGraph& graph,
                                                      int num_parts,
                                                      const std::string& gpmetis_executable,
                                                      int ufactor,
                                                      int seed);

}  // namespace kspecpart
