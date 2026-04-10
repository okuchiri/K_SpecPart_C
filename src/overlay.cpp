#include "kspecpart/overlay.hpp"

#include "kspecpart/hypergraph.hpp"
#include "kspecpart/isolate_islands.hpp"

#include <algorithm>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kspecpart {

namespace {

struct RawContractedEdge {
    std::vector<int> pins;
    int weight = 0;
    double hash = 0.0;
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

std::vector<RawContractedEdge> build_raw_contracted_edges(const Hypergraph& hypergraph,
                                                          const std::vector<int>& clusters) {
    std::vector<RawContractedEdge> edges;
    edges.reserve(hypergraph.num_hyperedges);
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
        edges.push_back({std::move(pins), hypergraph.hwts[edge], 0.0});
    }
    return edges;
}

}  // namespace

Hypergraph contract_hypergraph(const Hypergraph& hypergraph,
                               const std::vector<int>& clusters,
                               AlgorithmRng& rng) {
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

    std::vector<RawContractedEdge> raw_edges =
        build_raw_contracted_edges(hypergraph, clusters);

    std::vector<double> random_row(contracted_vertices, 0.0);
    rng.fill_float64_bulk(random_row.data(), contracted_vertices);
    for (RawContractedEdge& edge : raw_edges) {
        double hash = 0.0;
        for (int pin : edge.pins) {
            hash += random_row[pin];
        }
        edge.hash = hash;
    }

    std::vector<int> permutation(raw_edges.size(), 0);
    for (int index = 0; index < static_cast<int>(raw_edges.size()); ++index) {
        permutation[static_cast<std::size_t>(index)] = index;
    }
    std::stable_sort(permutation.begin(),
                     permutation.end(),
                     [&](int lhs, int rhs) {
                         return raw_edges[static_cast<std::size_t>(lhs)].hash <
                                raw_edges[static_cast<std::size_t>(rhs)].hash;
                     });

    std::vector<std::vector<int>> unique_edges;
    std::vector<int> edge_weights;
    unique_edges.reserve(raw_edges.size());
    edge_weights.reserve(raw_edges.size());
    bool have_last_hash = false;
    double last_hash = 0.0;
    for (int index : permutation) {
        const RawContractedEdge& edge = raw_edges[static_cast<std::size_t>(index)];
        if (have_last_hash && edge.hash == last_hash) {
            edge_weights.back() += edge.weight;
            continue;
        }
        unique_edges.push_back(edge.pins);
        edge_weights.push_back(edge.weight);
        last_hash = edge.hash;
        have_last_hash = true;
    }

    std::vector<int> eptr = {0};
    std::vector<int> eind;
    std::vector<int> hwts;
    eind.reserve(hypergraph.eind.size());
    hwts.reserve(unique_edges.size());
    for (int edge = 0; edge < static_cast<int>(unique_edges.size()); ++edge) {
        const std::vector<int>& pins = unique_edges[edge];
        const int weight = edge_weights[edge];
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

OverlayResult overlay_partitions(const std::vector<std::vector<int>>& partitions,
                                 const Hypergraph& hypergraph,
                                 AlgorithmRng& rng) {
    std::unordered_set<int> union_cut;
    for (const auto& partition : partitions) {
        for (int edge : cut_edges_for_partition(hypergraph, partition)) {
            union_cut.insert(edge);
        }
    }

    const auto [clusters, sizes] = island_removal(hypergraph, union_cut);
    OverlayResult result;
    result.hypergraph = contract_hypergraph(hypergraph, clusters, rng);
    result.clusters = clusters;
    (void)sizes;
    return result;
}

}  // namespace kspecpart
