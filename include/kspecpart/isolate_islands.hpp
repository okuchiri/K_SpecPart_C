#pragma once

#include "kspecpart/definitions.hpp"

#include <unordered_set>
#include <utility>
#include <vector>

namespace kspecpart {

struct IsolateResult {
    Hypergraph hypergraph;
    std::vector<int> original_indices;
    std::vector<int> new_indices;
    std::vector<int> component_labels;
    std::vector<int> component_sizes;
    int main_component = -1;
};

std::pair<std::vector<int>, std::vector<int>> island_removal(
    const Hypergraph& hypergraph,
    const std::unordered_set<int>& excluded_hyperedges);

IsolateResult isolate_islands(const Hypergraph& hypergraph);

}  // namespace kspecpart
