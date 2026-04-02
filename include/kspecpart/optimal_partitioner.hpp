#pragma once

#include "kspecpart/definitions.hpp"

#include <optional>
#include <string>
#include <vector>

namespace kspecpart {

struct OptimalPartitionerOptions {
    int num_parts = 2;
    int imb = 2;
    int seed = 0;
    int hmetis_runs = 10;
    int parallel_runs = 10;
    std::string hmetis_executable;
    std::string ilp_partitioner_executable;
    bool enable_hmetis = true;
    bool enable_ilp = true;
    bool hmetis_explicit = false;
    bool ilp_partitioner_explicit = false;
};

struct OptimalPartitionerResult {
    std::vector<int> partition;
    std::string method;
};

std::optional<std::string> resolve_hmetis_executable(const std::string& configured_path);
std::optional<std::string> resolve_ilp_partitioner_executable(const std::string& configured_path);
std::optional<OptimalPartitionerResult> run_optimal_partitioner(const Hypergraph& hypergraph,
                                                                const OptimalPartitionerOptions& options,
                                                                const std::string& work_prefix);

}  // namespace kspecpart
