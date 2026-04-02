#include "kspecpart/cut_distillation.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kspecpart {

namespace {

std::vector<int> stable_unique_preserve_order(const std::vector<int>& values, int domain_size) {
    if (domain_size <= 0) {
        return {};
    }

    std::vector<char> seen(domain_size, 0);
    std::vector<int> unique_values;
    unique_values.reserve(values.size());
    for (int value : values) {
        if (value < 0 || value >= domain_size) {
            continue;
        }
        if (seen[value]) {
            continue;
        }
        seen[value] = 1;
        unique_values.push_back(value);
    }
    return unique_values;
}

std::vector<int> intersection_sorted(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    std::vector<int> result;
    std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(result));
    return result;
}

std::vector<int> difference_sorted(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    std::vector<int> result;
    std::set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(result));
    return result;
}

struct JuliaEulerVectors {
    std::vector<int> euler_tour;
    std::vector<int> finish_order;
    std::vector<int> parent_of;
};

JuliaEulerVectors build_julia_euler_vectors(const WeightedGraph& tree, int root) {
    const int n = tree.num_vertices;
    std::vector<char> seen(n, 0);
    std::vector<int> finish_order(n, -1);
    std::vector<int> parent_of(n, -1);
    std::vector<int> stack = {root};
    std::vector<int> euler_tour = {root};

    seen[root] = 1;
    parent_of[root] = root;

    int finish_index = 0;
    int iterations = 0;
    while (!stack.empty()) {
        ++iterations;
        if (iterations > std::max(1, 4 * n)) {
            throw std::runtime_error("build_julia_euler_vectors exceeded the expected iteration bound");
        }
        const int vertex = stack.back();
        int next_vertex = -1;
        for (const auto& [neighbor, weight] : tree.adjacency[vertex]) {
            (void)weight;
            if (!seen[neighbor]) {
                next_vertex = neighbor;
                break;
            }
        }

        if (next_vertex < 0) {
            finish_order[finish_index++] = vertex;
            euler_tour.push_back(parent_of[vertex]);
            stack.pop_back();
            continue;
        }

        seen[next_vertex] = 1;
        euler_tour.push_back(next_vertex);
        stack.push_back(next_vertex);
        parent_of[next_vertex] = vertex;
    }

    if (!euler_tour.empty()) {
        euler_tour.pop_back();
    }
    finish_order.resize(finish_index);
    return {std::move(euler_tour),
            std::move(finish_order),
            std::move(parent_of)};
}

std::vector<int> build_julia_node_levels(const std::vector<int>& euler_tour, int num_vertices) {
    int level = -1;
    std::vector<int> level_vec(num_vertices, 0);
    std::vector<char> seen(num_vertices, 0);

    for (int vertex : euler_tour) {
        if (!seen[vertex]) {
            ++level;
            level_vec[vertex] = level;
            seen[vertex] = 1;
        } else {
            --level;
        }
    }
    return level_vec;
}

std::vector<std::vector<int>> build_rmq_sparse_table(const std::vector<int>& euler_level) {
    const int n = static_cast<int>(euler_level.size());
    const int cols = n <= 1 ? 1 : static_cast<int>(std::ceil(std::log2(static_cast<double>(n)))) + 1;
    std::vector<std::vector<int>> table(n, std::vector<int>(cols, 0));
    for (int i = 0; i < n; ++i) {
        table[i][0] = i;
    }
    for (int j = 1; j < cols; ++j) {
        const int offset = 1 << (j - 1);
        for (int i = 0; i < n; ++i) {
            const int rhs = std::min(i + offset, n - 1);
            table[i][j] = euler_level[table[i][j - 1]] <= euler_level[table[rhs][j - 1]]
                ? table[i][j - 1]
                : table[rhs][j - 1];
        }
    }
    return table;
}

int rmq_query(int left, int right, const LeastCommonAncestor& lca) {
    if (left > right) {
        std::swap(left, right);
    }
    const int length = right - left + 1;
    const int k = length <= 1 ? 0 : static_cast<int>(std::floor(std::log2(static_cast<double>(length))));
    const int left_idx = lca.rmq_sparse_table[left][k];
    const int right_idx = lca.rmq_sparse_table[right - (1 << k) + 1][k];
    return lca.euler_level[left_idx] <= lca.euler_level[right_idx] ? left_idx : right_idx;
}

std::vector<int> filter_mobile_pins(const Hypergraph& hypergraph,
                                    int edge,
                                    const std::vector<int>& is_fixed) {
    std::vector<int> pins;
    for (int idx = hypergraph.eptr[edge]; idx < hypergraph.eptr[edge + 1]; ++idx) {
        const int vertex = hypergraph.eind[idx];
        if (!is_fixed[vertex]) {
            pins.push_back(vertex);
        }
    }
    return pins;
}

int max_hyperedge_size(const Hypergraph& hypergraph) {
    int max_size = 1;
    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        max_size = std::max(max_size, hypergraph.eptr[edge + 1] - hypergraph.eptr[edge]);
    }
    return max_size;
}

void heap_push_min(std::vector<int>& heap, int& heap_size, int value) {
    ++heap_size;
    int index = heap_size;
    while (index > 1) {
        const int parent = index / 2;
        if (heap[parent] > value) {
            heap[index] = heap[parent];
            index = parent;
        } else {
            break;
        }
    }
    heap[index] = value;
}

int heap_pop_min(std::vector<int>& heap, int& heap_size) {
    const int top = heap[1];
    heap[1] = heap[heap_size];
    --heap_size;

    const int value = heap[1];
    int index = 1;
    while (true) {
        const int left = 2 * index;
        const int right = left + 1;
        if (left > heap_size) {
            break;
        }

        int child = left;
        if (right <= heap_size && heap[right] < heap[left]) {
            child = right;
        }

        if (heap[child] < value) {
            heap[index] = heap[child];
            index = child;
        } else {
            break;
        }
    }
    heap[index] = value;
    return top;
}

void accumulate_heap_lcas_literal(const std::vector<int>& pins,
                                  const LeastCommonAncestor& lca,
                                  int edge_weight,
                                  std::vector<int>& edge_diff,
                                  int& terminator,
                                  bool final_subtract,
                                  std::vector<int>& heap) {
    int heap_size = 0;
    for (int vertex : pins) {
        edge_diff[vertex] += edge_weight;
        heap_push_min(heap, heap_size, lca.fts[vertex]);
    }

    while (heap_size > 1) {
        const int u_pos = heap_pop_min(heap, heap_size);
        const int u = lca.ifts[u_pos];
        const int v = lca.ifts[heap[1]];
        if (u == v) {
            edge_diff[u] -= edge_weight;
            continue;
        }
        const int w = lca_query(u, v, lca);
        heap_push_min(heap, heap_size, lca.fts[w]);
    }

    if (heap_size > 0) {
        terminator = lca.ifts[heap[1]];
        if (final_subtract) {
            edge_diff[terminator] -= edge_weight;
        }
    }
}

}  // namespace

std::vector<int> incident_edges(const Hypergraph& hypergraph, const std::vector<int>& vertices) {
    std::vector<int> edges;
    for (int vertex : vertices) {
        if (vertex < 0 || vertex >= hypergraph.num_vertices) {
            continue;
        }
        for (int idx = hypergraph.vptr[vertex]; idx < hypergraph.vptr[vertex + 1]; ++idx) {
            edges.push_back(hypergraph.vind[idx]);
        }
    }
    return stable_unique_preserve_order(edges, hypergraph.num_hyperedges);
}

std::vector<int> incident_nodes(const Hypergraph& hypergraph, const std::vector<int>& edges) {
    std::vector<int> vertices;
    for (int edge : edges) {
        if (edge < 0 || edge >= hypergraph.num_hyperedges) {
            continue;
        }
        for (int idx = hypergraph.eptr[edge]; idx < hypergraph.eptr[edge + 1]; ++idx) {
            vertices.push_back(hypergraph.eind[idx]);
        }
    }
    return stable_unique_preserve_order(vertices, hypergraph.num_vertices);
}

LeastCommonAncestor build_lca(const WeightedGraph& tree, int root) {
    const int n = tree.num_vertices;
    if (n == 0) {
        return {};
    }
    if (root < 0 || root >= n) {
        throw std::out_of_range("tree root is out of range");
    }

    JuliaEulerVectors euler_vectors = build_julia_euler_vectors(tree, root);
    if (static_cast<int>(euler_vectors.finish_order.size()) != n) {
        throw std::runtime_error("build_lca expects a connected tree-like graph");
    }

    const std::vector<int> node_level = build_julia_node_levels(euler_vectors.euler_tour, n);
    std::vector<int> euler_level(euler_vectors.euler_tour.size(), 0);
    for (int i = 0; i < static_cast<int>(euler_vectors.euler_tour.size()); ++i) {
        euler_level[i] = node_level[euler_vectors.euler_tour[i]];
    }

    std::vector<int> fts(n, -1);
    for (int idx = 0; idx < static_cast<int>(euler_vectors.euler_tour.size()); ++idx) {
        fts[euler_vectors.euler_tour[idx]] = idx;
    }

    std::vector<int> ifts(euler_vectors.euler_tour.size(), -1);
    for (int vertex = 0; vertex < n; ++vertex) {
        if (fts[vertex] >= 0) {
            ifts[fts[vertex]] = vertex;
        }
    }

    LeastCommonAncestor lca;
    lca.rmq_sparse_table = build_rmq_sparse_table(euler_level);
    lca.euler_level = std::move(euler_level);
    lca.child = std::move(euler_vectors.finish_order);
    lca.parents = std::move(euler_vectors.parent_of);
    lca.euler_tour = std::move(euler_vectors.euler_tour);
    lca.level_vec = lca.euler_level;
    lca.fts = std::move(fts);
    lca.ifts = std::move(ifts);
    return lca;
}

int lca_query(int u, int v, const LeastCommonAncestor& lca) {
    if (u < 0 || u >= static_cast<int>(lca.fts.size()) || v < 0 || v >= static_cast<int>(lca.fts.size())) {
        throw std::out_of_range("lca_query vertex is out of range");
    }
    const int u_idx = lca.fts[u];
    const int v_idx = lca.fts[v];
    if (u_idx < 0 || v_idx < 0) {
        throw std::runtime_error("lca_query vertex is not present in the Euler tour");
    }
    const int lca_idx = rmq_query(u_idx, v_idx, lca);
    return lca.euler_tour[lca_idx];
}

std::pair<std::vector<int>, std::vector<int>> compute_edge_cuts(const Hypergraph& hypergraph,
                                                                const LeastCommonAncestor& lca,
                                                                const std::vector<int>& edges) {
    std::vector<int> edge_diff(hypergraph.num_vertices, 0);
    std::vector<int> edge_terminators(hypergraph.num_hyperedges, -1);
    std::vector<int> heap(max_hyperedge_size(hypergraph) + 1, 0);

    for (int edge : edges) {
        if (edge < 0 || edge >= hypergraph.num_hyperedges) {
            continue;
        }
        const int edge_weight = hypergraph.hwts[edge];
        const int start = hypergraph.eptr[edge];
        const int end = hypergraph.eptr[edge + 1];
        const int pin_count = end - start;
        if (pin_count <= 0) {
            continue;
        }
        if (pin_count == 1) {
            const int u = hypergraph.eind[start];
            edge_terminators[edge] = u;
            edge_diff[u] += edge_weight;
            continue;
        }
        if (pin_count == 2) {
            const int u = hypergraph.eind[start];
            const int v = hypergraph.eind[start + 1];
            const int w = lca_query(u, v, lca);
            edge_terminators[edge] = w;
            if (w == u) {
                edge_diff[u] -= edge_weight;
                edge_diff[v] += edge_weight;
            } else if (w == v) {
                edge_diff[v] -= edge_weight;
                edge_diff[u] += edge_weight;
            } else {
                edge_diff[u] += edge_weight;
                edge_diff[v] += edge_weight;
                edge_diff[w] -= 2 * edge_weight;
            }
            continue;
        }

        std::vector<int> pins;
        pins.reserve(pin_count);
        for (int idx = start; idx < end; ++idx) {
            pins.push_back(hypergraph.eind[idx]);
        }
        accumulate_heap_lcas_literal(pins, lca, edge_weight, edge_diff, edge_terminators[edge], true, heap);
    }

    return {edge_diff, edge_terminators};
}

std::pair<std::vector<int>, std::vector<int>> compute_edge_cuts_fixed(const Hypergraph& hypergraph,
                                                                      const PartitionIndex& pindex,
                                                                      const LeastCommonAncestor& lca,
                                                                      const std::vector<int>& edges) {
    std::vector<int> is_fixed(hypergraph.num_vertices, 0);
    for (int vertex : pindex.p1) {
        if (vertex >= 0 && vertex < hypergraph.num_vertices) {
            is_fixed[vertex] = 1;
        }
    }
    for (int vertex : pindex.p2) {
        if (vertex >= 0 && vertex < hypergraph.num_vertices) {
            is_fixed[vertex] = 1;
        }
    }

    std::vector<int> edge_diff(hypergraph.num_vertices, 0);
    std::vector<int> edge_terminators(hypergraph.num_hyperedges, -1);
    std::vector<int> heap(2 * max_hyperedge_size(hypergraph) + 1, 0);

    for (int edge : edges) {
        if (edge < 0 || edge >= hypergraph.num_hyperedges) {
            continue;
        }
        const int edge_weight = hypergraph.hwts[edge];
        const std::vector<int> pins = filter_mobile_pins(hypergraph, edge, is_fixed);
        const int pin_count = static_cast<int>(pins.size());
        if (pin_count <= 0) {
            continue;
        }
        if (pin_count == 1) {
            edge_terminators[edge] = pins[0];
            edge_diff[pins[0]] += edge_weight;
            continue;
        }
        if (pin_count == 2) {
            const int u = pins[0];
            const int v = pins[1];
            const int w = lca_query(u, v, lca);
            edge_terminators[edge] = w;
            if (w == u) {
                edge_diff[v] += edge_weight;
            } else if (w == v) {
                edge_diff[u] += edge_weight;
            } else {
                edge_diff[u] += edge_weight;
                edge_diff[v] += edge_weight;
                edge_diff[w] -= edge_weight;
            }
            continue;
        }

        accumulate_heap_lcas_literal(pins, lca, edge_weight, edge_diff, edge_terminators[edge], false, heap);
    }

    return {edge_diff, edge_terminators};
}

CutProfile distill_cuts_on_tree(const Hypergraph& hypergraph,
                                const PartitionIndex& pindex,
                                const WeightedGraph& tree,
                                int root) {
    if (hypergraph.num_vertices != tree.num_vertices) {
        throw std::invalid_argument("distill_cuts_on_tree expects the tree and hypergraph to share the same vertex set");
    }

    std::vector<int> forced_0 = incident_edges(hypergraph, pindex.p1);
    std::vector<int> forced_1 = incident_edges(hypergraph, pindex.p2);
    std::vector<int> forced_01 = intersection_sorted(forced_0, forced_1);
    forced_0 = difference_sorted(forced_0, forced_01);
    forced_1 = difference_sorted(forced_1, forced_01);

    std::vector<int> forced_type(hypergraph.num_hyperedges, -1);
    for (int edge : forced_0) {
        forced_type[edge] = 0;
    }
    for (int edge : forced_1) {
        forced_type[edge] = 1;
    }
    for (int edge : forced_01) {
        forced_type[edge] = 2;
    }

    const LeastCommonAncestor lca = build_lca(tree, root);
    std::vector<int> mobile_edges;
    mobile_edges.reserve(hypergraph.num_hyperedges);
    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        if (forced_type[edge] == -1) {
            mobile_edges.push_back(edge);
        }
    }

    auto [edge_diff, edge_terminators] = compute_edge_cuts(hypergraph, lca, mobile_edges);
    auto [edge_diff_0, edge_terminators_0] =
        compute_edge_cuts_fixed(hypergraph, pindex, lca, forced_0);
    auto [edge_diff_1, edge_terminators_1] =
        compute_edge_cuts_fixed(hypergraph, pindex, lca, forced_1);

    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        if (edge_terminators[edge] < 0 && edge_terminators_0[edge] >= 0) {
            edge_terminators[edge] = edge_terminators_0[edge];
        }
        if (edge_terminators[edge] < 0 && edge_terminators_1[edge] >= 0) {
            edge_terminators[edge] = edge_terminators_1[edge];
        }
    }

    std::vector<int> vtx_cuts(hypergraph.num_vertices, 0);
    std::vector<int> edge_cuts(hypergraph.num_vertices, 0);
    std::vector<int> edge_cuts_0(hypergraph.num_vertices, 0);
    std::vector<int> edge_cuts_1(hypergraph.num_vertices, 0);
    std::vector<int> fb0(hypergraph.num_vertices, 0);
    std::vector<int> fb1(hypergraph.num_vertices, 0);

    for (int edge : forced_0) {
        if (edge >= 0 && edge < hypergraph.num_hyperedges && edge_terminators[edge] >= 0) {
            fb0[edge_terminators[edge]] += hypergraph.hwts[edge];
        }
    }
    for (int edge : forced_1) {
        if (edge >= 0 && edge < hypergraph.num_hyperedges && edge_terminators[edge] >= 0) {
            fb1[edge_terminators[edge]] += hypergraph.hwts[edge];
        }
    }

    for (int idx = 0; idx + 1 < static_cast<int>(lca.child.size()); ++idx) {
        const int node = lca.child[idx];
        const int parent = lca.parents[node];
        edge_cuts[node] += edge_diff[node];
        edge_cuts[parent] += edge_cuts[node];
        edge_cuts_0[node] += edge_diff_0[node];
        edge_cuts_0[parent] += edge_cuts_0[node];
        edge_cuts_1[node] += edge_diff_1[node];
        edge_cuts_1[parent] += edge_cuts_1[node];
        vtx_cuts[node] += hypergraph.vwts[node];
        vtx_cuts[parent] += vtx_cuts[node];
        fb0[parent] += fb0[node];
        fb1[parent] += fb1[node];
    }

    CutProfile profile;
    profile.vtx_cuts = std::move(vtx_cuts);
    profile.edge_cuts = std::move(edge_cuts);
    profile.edge_diff = std::move(edge_diff);
    profile.pred = lca.parents;
    profile.edge_terminators = std::move(edge_terminators);
    profile.p = pindex;
    profile.forced_type = std::move(forced_type);
    profile.forced_0 = std::move(forced_0);
    profile.forced_1 = std::move(forced_1);
    profile.forced_01 = std::move(forced_01);
    profile.FB0 = std::move(fb0);
    profile.FB1 = std::move(fb1);
    profile.edge_cuts_0 = std::move(edge_cuts_0);
    profile.edge_cuts_1 = std::move(edge_cuts_1);
    return profile;
}

}  // namespace kspecpart
