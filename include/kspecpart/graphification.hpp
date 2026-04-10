#pragma once

#include "kspecpart/definitions.hpp"
#include "kspecpart/julia_random.hpp"

namespace kspecpart {

WeightedGraph hypergraph_to_graph(const Hypergraph& hypergraph,
                                  int cycles,
                                  int seed,
                                  AlgorithmRng* shared_rng = nullptr);

}  // namespace kspecpart
