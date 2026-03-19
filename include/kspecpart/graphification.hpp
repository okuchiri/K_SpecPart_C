#pragma once

#include "kspecpart/definitions.hpp"

#include <random>

namespace kspecpart {

WeightedGraph hypergraph_to_graph(const Hypergraph& hypergraph, int cycles, std::mt19937& rng);

}  // namespace kspecpart
