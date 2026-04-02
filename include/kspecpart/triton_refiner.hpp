#pragma once

#include <optional>
#include <string>
#include <vector>

namespace kspecpart {

struct TritonRefinerOptions {
    int num_parts = 2;
    int imb = 2;
    int seed = 0;
    std::string refiner_executable;
    bool explicit_path = false;
};

std::optional<std::string> resolve_triton_refiner_executable(const std::string& configured_path);
std::optional<std::vector<int>> run_triton_refiner(const std::string& hypergraph_file,
                                                   const std::vector<int>& partition,
                                                   const TritonRefinerOptions& options,
                                                   const std::string& work_prefix,
                                                   int run_id);

}  // namespace kspecpart
