#pragma once

#include "kspecpart/definitions.hpp"

#include <vector>

namespace kspecpart {

struct PartitionIndex {
    std::vector<int> p1;
    std::vector<int> p2;
};

struct LeastCommonAncestor {
    std::vector<std::vector<int>> rmq_sparse_table;
    std::vector<int> euler_level;
    std::vector<int> child;
    std::vector<int> parents;
    std::vector<int> euler_tour;
    std::vector<int> level_vec;
    std::vector<int> fts;
    std::vector<int> ifts;
};

struct CutProfile {
    std::vector<int> vtx_cuts;
    std::vector<int> edge_cuts;
    std::vector<int> edge_diff;
    std::vector<int> pred;
    std::vector<int> edge_terminators;
    PartitionIndex p;
    std::vector<int> forced_type;
    std::vector<int> forced_0;
    std::vector<int> forced_1;
    std::vector<int> forced_01;
    std::vector<int> FB0;
    std::vector<int> FB1;
    std::vector<int> edge_cuts_0;
    std::vector<int> edge_cuts_1;
};

std::vector<int> incident_edges(const Hypergraph& hypergraph, const std::vector<int>& vertices);
std::vector<int> incident_nodes(const Hypergraph& hypergraph, const std::vector<int>& edges);

LeastCommonAncestor build_lca(const WeightedGraph& tree, int root = 0);
int lca_query(int u, int v, const LeastCommonAncestor& lca);

std::pair<std::vector<int>, std::vector<int>> compute_edge_cuts(const Hypergraph& hypergraph,
                                                                const LeastCommonAncestor& lca,
                                                                const std::vector<int>& edges);

std::pair<std::vector<int>, std::vector<int>> compute_edge_cuts_fixed(const Hypergraph& hypergraph,
                                                                      const PartitionIndex& pindex,
                                                                      const LeastCommonAncestor& lca,
                                                                      const std::vector<int>& edges);

CutProfile distill_cuts_on_tree(const Hypergraph& hypergraph,
                                const PartitionIndex& pindex,
                                const WeightedGraph& tree,
                                int root = 0);

}  // namespace kspecpart
