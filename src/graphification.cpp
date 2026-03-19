#include "kspecpart/graphification.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace kspecpart {

namespace {

void add_undirected_edge(std::vector<std::unordered_map<int, double>>& accum,
                         int u,
                         int v,
                         double weight) {
    if (u == v || weight == 0.0) {
        return;
    }
    accum[u][v] += weight;
    accum[v][u] += weight;
}

}  // namespace

WeightedGraph hypergraph_to_graph(const Hypergraph& hypergraph, int cycles, std::mt19937& rng) {
    const int n = hypergraph.num_vertices;
    std::vector<std::unordered_map<int, double>> accum(n);
    cycles = std::max(1, cycles);

    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        const int start = hypergraph.eptr[edge];
        const int end = hypergraph.eptr[edge + 1];
        const int size = end - start;
        if (size <= 1) {
            continue;
        }

        const double edge_weight = static_cast<double>(hypergraph.hwts[edge]);
        if (size == 2) {
            add_undirected_edge(accum, hypergraph.eind[start], hypergraph.eind[start + 1], edge_weight);
            continue;
        }

        if (size == 3) {
            const double weight = edge_weight / 2.0;
            add_undirected_edge(accum, hypergraph.eind[start], hypergraph.eind[start + 1], weight);
            add_undirected_edge(accum, hypergraph.eind[start + 1], hypergraph.eind[start + 2], weight);
            add_undirected_edge(accum, hypergraph.eind[start + 2], hypergraph.eind[start], weight);
            continue;
        }

        std::vector<int> pins(hypergraph.eind.begin() + start, hypergraph.eind.begin() + end);
        const double scale = (std::floor(size / 2.0) * std::ceil(size / 2.0)) / static_cast<double>(size - 1);
        const double cycle_weight = edge_weight / (static_cast<double>(cycles) * 2.0 * scale);
        for (int cycle = 0; cycle < cycles; ++cycle) {
            std::shuffle(pins.begin(), pins.end(), rng);
            for (int i = 0; i + 1 < size; ++i) {
                add_undirected_edge(accum, pins[i], pins[i + 1], cycle_weight);
            }
            add_undirected_edge(accum, pins.back(), pins.front(), cycle_weight);
        }
    }

    WeightedGraph graph;
    graph.num_vertices = n;
    graph.adjacency.resize(n);
    graph.degrees.assign(n, 0.0);
    for (int vertex = 0; vertex < n; ++vertex) {
        graph.adjacency[vertex].reserve(accum[vertex].size());
        for (const auto& [neighbor, weight] : accum[vertex]) {
            graph.adjacency[vertex].push_back({neighbor, weight});
            graph.degrees[vertex] += weight;
        }
        std::sort(graph.adjacency[vertex].begin(),
                  graph.adjacency[vertex].end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    }
    return graph;
}

}  // namespace kspecpart
