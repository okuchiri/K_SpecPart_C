#pragma once

#include "kspecpart/definitions.hpp"

#include <vector>

namespace kspecpart {

PartitionResult evaluate_partition(const Hypergraph& hypergraph, int num_parts, const std::vector<int>& partition);

}  // namespace kspecpart
