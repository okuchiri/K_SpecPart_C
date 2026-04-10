#pragma once

#include "kspecpart/definitions.hpp"
#include "kspecpart/julia_random.hpp"

#include <vector>

namespace kspecpart {

struct OverlayResult {
    Hypergraph hypergraph;
    std::vector<int> clusters;
};

Hypergraph contract_hypergraph(const Hypergraph& hypergraph,
                               const std::vector<int>& clusters,
                               AlgorithmRng& rng);
OverlayResult overlay_partitions(const std::vector<std::vector<int>>& partitions,
                                 const Hypergraph& hypergraph,
                                 AlgorithmRng& rng);

}  // namespace kspecpart
