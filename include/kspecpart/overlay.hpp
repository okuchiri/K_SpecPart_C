#pragma once

#include "kspecpart/definitions.hpp"

#include <vector>

namespace kspecpart {

struct OverlayResult {
    Hypergraph hypergraph;
    std::vector<int> clusters;
};

Hypergraph contract_hypergraph(const Hypergraph& hypergraph, const std::vector<int>& clusters);
OverlayResult overlay_partitions(const std::vector<std::vector<int>>& partitions, const Hypergraph& hypergraph);

}  // namespace kspecpart
