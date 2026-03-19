#include "kspecpart/overlay.hpp"

#include "kspecpart/hypergraph.hpp"
#include "kspecpart/isolate_islands.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kspecpart {

namespace {

struct VectorHasher {
    std::size_t operator()(const std::vector<int>& value) const noexcept {
        std::size_t seed = value.size();
        for (int item : value) {
            seed ^= static_cast<std::size_t>(item + 0x9e3779b9) + (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }
};

std::vector<int> cut_edges_for_partition(const Hypergraph& hypergraph, const std::vector<int>& partition) {
    std::vector<int> cut_edges;
    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        const int start = hypergraph.eptr[edge];
        const int end = hypergraph.eptr[edge + 1];
        const int base_part = partition[hypergraph.eind[start]];
        bool cut = false;
        for (int idx = start + 1; idx < end; ++idx) {
            if (partition[hypergraph.eind[idx]] != base_part) {
                cut = true;
                break;
            }
        }
        if (cut) {
            cut_edges.push_back(edge);
        }
    }
    return cut_edges;
}

}  // namespace

Hypergraph contract_hypergraph(const Hypergraph& hypergraph, const std::vector<int>& clusters) {
    const int contracted_vertices = *std::max_element(clusters.begin(), clusters.end()) + 1;
    std::vector<int> vwts(contracted_vertices, 0);
    std::vector<int> fixed(contracted_vertices, -1);
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        const int cluster = clusters[vertex];
        vwts[cluster] += hypergraph.vwts[vertex];
        if (hypergraph.fixed[vertex] >= 0) {
            if (fixed[cluster] == -1) {
                fixed[cluster] = hypergraph.fixed[vertex];
            } else if (fixed[cluster] != hypergraph.fixed[vertex]) {
                fixed[cluster] = hypergraph.fixed[vertex];
            }
        }
    }

    std::unordered_map<std::vector<int>, int, VectorHasher> edge_weights;
    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        std::vector<int> pins;
        pins.reserve(hypergraph.eptr[edge + 1] - hypergraph.eptr[edge]);
        for (int idx = hypergraph.eptr[edge]; idx < hypergraph.eptr[edge + 1]; ++idx) {
            pins.push_back(clusters[hypergraph.eind[idx]]);
        }
        std::sort(pins.begin(), pins.end());
        pins.erase(std::unique(pins.begin(), pins.end()), pins.end());
        if (pins.size() <= 1) {
            continue;
        }
        edge_weights[pins] += hypergraph.hwts[edge];
    }

    std::vector<int> eptr = {0};
    std::vector<int> eind;
    std::vector<int> hwts;
    eind.reserve(hypergraph.eind.size());
    hwts.reserve(edge_weights.size());
    for (const auto& [pins, weight] : edge_weights) {
        eind.insert(eind.end(), pins.begin(), pins.end());
        eptr.push_back(static_cast<int>(eind.size()));
        hwts.push_back(weight);
    }

    return build_hypergraph(contracted_vertices,
                            static_cast<int>(hwts.size()),
                            eptr,
                            eind,
                            fixed,
                            vwts,
                            hwts);
}

OverlayResult overlay_partitions(const std::vector<std::vector<int>>& partitions, const Hypergraph& hypergraph) {
    std::unordered_set<int> union_cut;
    for (const auto& partition : partitions) {
        for (int edge : cut_edges_for_partition(hypergraph, partition)) {
            union_cut.insert(edge);
        }
    }

    const auto [clusters, sizes] = island_removal(hypergraph, union_cut);
    OverlayResult result;
    result.hypergraph = contract_hypergraph(hypergraph, clusters);
    result.clusters = clusters;
    (void)sizes;
    return result;
}

}  // namespace kspecpart
