#pragma once

#include "kspecpart/definitions.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace kspecpart {

struct ExactPartitionerOptions {
    int num_parts = 2;
    int imb = 2;
    std::size_t max_search_nodes = 4000000;
};

bool should_try_exact_partitioner(const Hypergraph& hypergraph,
                                  const ExactPartitionerOptions& options);

std::optional<std::vector<int>> run_exact_partitioner(const Hypergraph& hypergraph,
                                                      const ExactPartitionerOptions& options);

}  // namespace kspecpart
