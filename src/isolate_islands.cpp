#include "kspecpart/isolate_islands.hpp"

#include "kspecpart/hypergraph.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kspecpart {

namespace {

struct UnionFind {
    explicit UnionFind(int n) : parent(n), size(n, 1) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }
        if (size[a] < size[b]) {
            std::swap(a, b);
        }
        parent[b] = a;
        size[a] += size[b];
    }

    std::vector<int> parent;
    std::vector<int> size;
};

}  // namespace

std::pair<std::vector<int>, std::vector<int>> island_removal(
    const Hypergraph& hypergraph,
    const std::unordered_set<int>& excluded_hyperedges) {
    UnionFind uf(hypergraph.num_vertices);
    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        if (excluded_hyperedges.count(edge) > 0) {
            continue;
        }
        const int start = hypergraph.eptr[edge];
        const int end = hypergraph.eptr[edge + 1];
        for (int idx = start + 1; idx < end; ++idx) {
            uf.unite(hypergraph.eind[start], hypergraph.eind[idx]);
        }
    }

    std::vector<int> root_sizes(hypergraph.num_vertices, 0);
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        const int root = uf.find(vertex);
        root_sizes[root] += 1;
    }

    std::vector<int> root_to_cluster(hypergraph.num_vertices, -1);
    std::vector<int> sizes;
    sizes.reserve(hypergraph.num_vertices);
    for (int root = 0; root < hypergraph.num_vertices; ++root) {
        if (root_sizes[root] <= 0) {
            continue;
        }
        root_to_cluster[root] = static_cast<int>(sizes.size());
        sizes.push_back(root_sizes[root]);
    }

    std::vector<int> clusters(hypergraph.num_vertices, -1);
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        const int root = uf.find(vertex);
        clusters[vertex] = root_to_cluster[root];
    }

    return {clusters, sizes};
}

IsolateResult isolate_islands(const Hypergraph& hypergraph) {
    const auto [clusters, cluster_sizes] = island_removal(hypergraph, {});
    const int main_component = static_cast<int>(std::distance(
        cluster_sizes.begin(),
        std::max_element(cluster_sizes.begin(), cluster_sizes.end())));

    std::vector<int> original_indices;
    std::vector<int> new_indices(hypergraph.num_vertices, -1);
    std::vector<int> vwts_processed;
    std::vector<int> fixed_processed;
    original_indices.reserve(hypergraph.num_vertices);
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        if (clusters[vertex] != main_component) {
            continue;
        }
        new_indices[vertex] = static_cast<int>(original_indices.size());
        original_indices.push_back(vertex);
        vwts_processed.push_back(hypergraph.vwts[vertex]);
        fixed_processed.push_back(hypergraph.fixed[vertex]);
    }

    std::vector<int> eptr = {0};
    std::vector<int> eind;
    std::vector<int> hwts;
    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        std::vector<int> pins;
        for (int idx = hypergraph.eptr[edge]; idx < hypergraph.eptr[edge + 1]; ++idx) {
            const int mapped = new_indices[hypergraph.eind[idx]];
            if (mapped >= 0) {
                pins.push_back(mapped);
            }
        }
        if (pins.empty()) {
            continue;
        }
        eind.insert(eind.end(), pins.begin(), pins.end());
        eptr.push_back(static_cast<int>(eind.size()));
        hwts.push_back(hypergraph.hwts[edge]);
    }

    IsolateResult result;
    result.hypergraph = build_hypergraph(static_cast<int>(original_indices.size()),
                                         static_cast<int>(hwts.size()),
                                         eptr,
                                         eind,
                                         fixed_processed,
                                         vwts_processed,
                                         hwts);
    result.original_indices = std::move(original_indices);
    result.new_indices = std::move(new_indices);
    result.component_labels = clusters;
    result.component_sizes = cluster_sizes;
    result.main_component = main_component;
    return result;
}

}  // namespace kspecpart
