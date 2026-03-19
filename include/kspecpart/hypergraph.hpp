#pragma once

#include "kspecpart/definitions.hpp"

#include <vector>

namespace kspecpart {

Hypergraph build_hypergraph(int num_vertices,
                            int num_hyperedges,
                            const std::vector<int>& eptr,
                            const std::vector<int>& eind,
                            const std::vector<int>& fixed,
                            const std::vector<int>& vwts,
                            const std::vector<int>& hwts);

Hypergraph remove_single_hyperedges(const Hypergraph& hypergraph);

}  // namespace kspecpart
