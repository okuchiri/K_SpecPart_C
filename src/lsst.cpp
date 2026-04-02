#include "kspecpart/lsst.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace kspecpart {

namespace {

struct DisjointSet {
    explicit DisjointSet(int size) : parent(size), rank(size, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int vertex) {
        if (parent[vertex] != vertex) {
            parent[vertex] = find(parent[vertex]);
        }
        return parent[vertex];
    }

    bool unite(int lhs, int rhs) {
        lhs = find(lhs);
        rhs = find(rhs);
        if (lhs == rhs) {
            return false;
        }
        if (rank[lhs] < rank[rhs]) {
            std::swap(lhs, rhs);
        }
        parent[rhs] = lhs;
        if (rank[lhs] == rank[rhs]) {
            ++rank[lhs];
        }
        return true;
    }

    std::vector<int> parent;
    std::vector<int> rank;
};

struct AkpwEdge {
    int i = -1;
    int j = -1;
    double weight = 0.0;
    int original_u = -1;
    int original_v = -1;
    int original_index = -1;
};

struct IJVindGraph {
    std::vector<AkpwEdge> list;
    std::vector<int> colptr;
};

struct AkpwInputState {
    std::vector<AkpwEdge> findnz_edges;
    std::vector<AkpwEdge> orig_list;
};

std::vector<AkpwEdge> sort_ijvind_like_julia(const std::vector<AkpwEdge>& edges);

struct HeapEntry {
    int node = -1;
    int original_index = -1;
    double distance = 0.0;
};

struct HeapEntryGreater {
    bool operator()(const HeapEntry& lhs, const HeapEntry& rhs) const {
        return lhs.distance > rhs.distance;
    }
};

WeightedGraph make_empty_graph(int num_vertices) {
    WeightedGraph graph;
    graph.num_vertices = num_vertices;
    graph.adjacency.assign(num_vertices, {});
    graph.degrees.assign(num_vertices, 0.0);
    return graph;
}

void add_undirected_edge(WeightedGraph& graph, int u, int v, double weight) {
    if (u == v || weight <= 0.0) {
        return;
    }
    graph.adjacency[u].push_back({v, weight});
    graph.adjacency[v].push_back({u, weight});
    graph.degrees[u] += weight;
    graph.degrees[v] += weight;
}

void sort_adjacency(WeightedGraph& graph) {
    for (auto& neighbors : graph.adjacency) {
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });
    }
}

std::vector<int> spectral_order(const Eigen::MatrixXd& embedding, int num_vertices) {
    std::vector<int> order(num_vertices);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        const double lv = embedding.cols() > 0 ? embedding(lhs, 0) : static_cast<double>(lhs);
        const double rv = embedding.cols() > 0 ? embedding(rhs, 0) : static_cast<double>(rhs);
        if (lv != rv) {
            return lv < rv;
        }
        return lhs < rhs;
    });
    return order;
}

std::vector<int> inverse_permutation(const std::vector<int>& permutation) {
    std::vector<int> inverse(permutation.size(), -1);
    for (int index = 0; index < static_cast<int>(permutation.size()); ++index) {
        inverse[permutation[index]] = index;
    }
    return inverse;
}

std::vector<AkpwEdge> make_findnz_edge_list_like_julia(const WeightedGraph& graph,
                                                       const std::vector<int>& old_to_new) {
    std::vector<AkpwEdge> edges;
    for (int old_j = 0; old_j < graph.num_vertices; ++old_j) {
        for (const auto& [old_i, weight] : graph.adjacency[old_j]) {
            edges.push_back({old_to_new[old_i],
                             old_to_new[old_j],
                             weight,
                             old_to_new[old_i],
                             old_to_new[old_j],
                             -1});
        }
    }

    // `findnz(graph)` in Julia returns the sparse nonzeros in CSC order: column `j`
    // first, then row `i` within each column.
    edges = sort_ijvind_like_julia(edges);
    for (int index = 0; index < static_cast<int>(edges.size()); ++index) {
        edges[index].original_index = index;
    }
    return edges;
}

std::vector<AkpwEdge> make_orig_list_like_julia(const std::vector<AkpwEdge>& findnz_edges) {
    std::vector<AkpwEdge> orig_list = findnz_edges;
    // This mirrors `sort(IJVindList(ai,aj,av), rev=true)`: same entries, same `ind`,
    // re-ordered by descending weight.
    std::stable_sort(orig_list.begin(), orig_list.end(), [](const AkpwEdge& lhs, const AkpwEdge& rhs) {
        return lhs.weight > rhs.weight;
    });
    return orig_list;
}

AkpwInputState make_akpw_input_state_like_julia(const WeightedGraph& graph,
                                                const std::vector<int>& old_to_new) {
    AkpwInputState state;
    state.findnz_edges = make_findnz_edge_list_like_julia(graph, old_to_new);
    state.orig_list = make_orig_list_like_julia(state.findnz_edges);
    return state;
}

int max_vertex_id(const std::vector<AkpwEdge>& edges) {
    int maximum = -1;
    for (const auto& edge : edges) {
        maximum = std::max(maximum, std::max(edge.i, edge.j));
    }
    return maximum;
}

int max_i_vertex_id_like_julia(const std::vector<AkpwEdge>& edges) {
    int maximum = -1;
    for (const auto& edge : edges) {
        maximum = std::max(maximum, edge.i);
    }
    return maximum;
}

int max_j_vertex_id_like_julia(const std::vector<AkpwEdge>& edges) {
    int maximum = -1;
    for (const auto& edge : edges) {
        maximum = std::max(maximum, edge.j);
    }
    return maximum;
}

template <typename KeyFn>
std::vector<AkpwEdge> stable_counting_sort_from_cumdeg(const std::vector<AkpwEdge>& edges,
                                                       const std::vector<int>& cumdeg,
                                                       KeyFn key_fn) {
    if (edges.empty()) {
        return {};
    }

    std::vector<int> positions = cumdeg;
    std::vector<AkpwEdge> output(edges.size());
    for (int index = static_cast<int>(edges.size()) - 1; index >= 0; --index) {
        const int key = key_fn(edges[index]);
        assert(key >= 0);
        assert(key < static_cast<int>(positions.size()));
        const int position = --positions[key];
        output[position] = edges[index];
    }

    return output;
}

IJVindGraph build_ijvind_graph_like_julia(const std::vector<AkpwEdge>& edges) {
    if (edges.empty()) {
        return {};
    }

    const int num_vertices = max_i_vertex_id_like_julia(edges) + 1;
    const int max_j = max_j_vertex_id_like_julia(edges) + 1;
    assert(num_vertices == max_j);

    std::vector<int> degree(num_vertices, 0);
    for (const auto& edge : edges) {
        assert(edge.i >= 0);
        assert(edge.i < num_vertices);
        ++degree[edge.i];
    }

    std::vector<int> cumdeg(num_vertices, 0);
    if (!degree.empty()) {
        std::partial_sum(degree.begin(), degree.end(), cumdeg.begin());
    }

    std::vector<int> colptr(num_vertices + 1, 0);
    for (int vertex = 0; vertex < num_vertices; ++vertex) {
        colptr[vertex + 1] = cumdeg[vertex];
    }

    const std::vector<AkpwEdge> list1 =
        stable_counting_sort_from_cumdeg(edges, cumdeg, [](const AkpwEdge& edge) { return edge.i; });
    const std::vector<AkpwEdge> list2 =
        stable_counting_sort_from_cumdeg(list1, cumdeg, [](const AkpwEdge& edge) { return edge.j; });

    return {list2, colptr};
}

std::vector<AkpwEdge> sort_ijvind_like_julia(const std::vector<AkpwEdge>& edges) {
    return build_ijvind_graph_like_julia(edges).list;
}

void compress_sorted_edges_like_julia(std::vector<AkpwEdge>& edges) {
    if (edges.empty()) {
        return;
    }

    std::vector<AkpwEdge> compressed;
    compressed.reserve(edges.size());

    AkpwEdge current = edges.front();
    for (std::size_t index = 1; index < edges.size(); ++index) {
        const AkpwEdge& edge = edges[index];
        if (edge.i == current.i && edge.j == current.j) {
            if (edge.weight > current.weight) {
                current = edge;
            }
            continue;
        }
        compressed.push_back(current);
        current = edge;
    }

    compressed.push_back(current);
    edges.swap(compressed);
}

int compress_indices_like_julia(std::vector<AkpwEdge>& edges) {
    if (edges.empty()) {
        return 0;
    }

    const int max_vertex = max_vertex_id(edges);
    std::vector<int> remap(max_vertex + 1, -1);

    int next_index = 0;
    for (const auto& edge : edges) {
        if (remap[edge.j] < 0) {
            remap[edge.j] = next_index++;
        }
    }

    for (auto& edge : edges) {
        if (remap[edge.i] < 0) {
            // Julia's sparse traversal should make every active vertex appear as some `j`.
            assert(false && "compress_indices_like_julia expected every active vertex to appear as a column");
            remap[edge.i] = next_index++;
        }
        edge.i = remap[edge.i];
        edge.j = remap[edge.j];
    }

    return next_index;
}

void dijkstra_from_seed(const IJVindGraph& graph,
                        int seed,
                        int component_id,
                        double expansion_factor,
                        std::vector<int>& component,
                        std::vector<int>& tree_edges) {
    double boundary = 0.0;
    double volume = 0.0;
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryGreater> heap;

    component[seed] = component_id;

    for (int index = graph.colptr[seed]; index < graph.colptr[seed + 1]; ++index) {
        const AkpwEdge& edge = graph.list[index];
        const int neighbor = edge.i;
        if (component[neighbor] != -1) {
            continue;
        }

        const double inv_weight =
            edge.weight > 0.0 ? 1.0 / edge.weight : std::numeric_limits<double>::infinity();
        heap.push({neighbor, edge.original_index, inv_weight});
        boundary += edge.weight;
        volume += edge.weight;
    }

    while (boundary > expansion_factor * volume && !heap.empty()) {
        const HeapEntry best = heap.top();
        heap.pop();
        if (component[best.node] != -1) {
            continue;
        }

        component[best.node] = component_id;
        tree_edges.push_back(best.original_index);

        for (int index = graph.colptr[best.node]; index < graph.colptr[best.node + 1]; ++index) {
            const AkpwEdge& edge = graph.list[index];
            const int neighbor = edge.i;
            if (component[neighbor] == component_id) {
                boundary -= edge.weight;
                volume += edge.weight;
            } else if (component[neighbor] == -1) {
                const double inv_weight =
                    edge.weight > 0.0 ? 1.0 / edge.weight : std::numeric_limits<double>::infinity();
                heap.push({neighbor, edge.original_index, best.distance + inv_weight});
                boundary += edge.weight;
                volume += edge.weight;
            }
        }
    }
}

void cluster_phase_like_julia(const std::vector<AkpwEdge>& current_edges,
                              double expansion_factor,
                              std::vector<int>& tree_edges) {
    if (current_edges.empty()) {
        return;
    }

    const int num_vertices = current_edges.back().j + 1;
    assert(current_edges.back().j >= current_edges.back().i);
    assert(static_cast<int>(current_edges.size()) >= num_vertices);
    if (num_vertices <= 1) {
        return;
    }

    const IJVindGraph ijv_graph = build_ijvind_graph_like_julia(current_edges);
    assert(static_cast<int>(ijv_graph.colptr.size()) == num_vertices + 1);
    std::vector<int> component(num_vertices, -1);
    int next_component = 0;

    for (int seed_index = 0; seed_index < num_vertices; ++seed_index) {
        const AkpwEdge& ijvind = current_edges[seed_index];
        const int edge_u = ijvind.i;
        const int edge_v = ijvind.j;

        if (component[edge_u] == -1 && component[edge_v] == -1) {
            dijkstra_from_seed(ijv_graph, edge_u, next_component++, expansion_factor, component, tree_edges);
        }
    }
}

std::vector<AkpwEdge> contract_edges_like_julia(const std::vector<AkpwEdge>& current_edges,
                                                DisjointSet& name_map,
                                                double& max_weight) {
    std::vector<AkpwEdge> new_edges;
    new_edges.reserve(current_edges.size());
    max_weight = 0.0;

    for (const auto& edge : current_edges) {
        const int name_i = name_map.find(edge.original_u);
        const int name_j = name_map.find(edge.original_v);
        if (name_i == name_j) {
            continue;
        }
        if (edge.weight > max_weight) {
            max_weight = edge.weight;
        }
        new_edges.push_back({name_i, name_j, edge.weight, edge.original_u, edge.original_v, edge.original_index});
    }

    return new_edges;
}

void expose_more_edges_like_julia(const std::vector<AkpwEdge>& original_edges,
                                  int& last,
                                  double expansion_factor,
                                  double& max_weight,
                                  DisjointSet& name_map,
                                  std::vector<AkpwEdge>& new_edges) {
    const int m = static_cast<int>(original_edges.size());

    ++last;
    while (last < m && original_edges[last].weight > expansion_factor * max_weight) {
        const AkpwEdge& edge = original_edges[last];
        const int name_i = name_map.find(edge.original_u);
        const int name_j = name_map.find(edge.original_v);
        if (name_i != name_j) {
            if (max_weight == 0.0) {
                max_weight = edge.weight;
            }
            new_edges.push_back({name_i, name_j, edge.weight, edge.original_u, edge.original_v, edge.original_index});
        }
        ++last;
    }
    --last;
}

double akpw_expansion_factor(int nleft) {
    if (nleft <= 2) {
        return 1.0;
    }
    return 1.0 / (2.0 * std::log(static_cast<double>(nleft)));
}

void add_fallback_spectral_chain(WeightedGraph& tree,
                                 const std::vector<int>& new_to_old,
                                 const Eigen::MatrixXd& embedding) {
    const std::vector<int> order = spectral_order(embedding, static_cast<int>(new_to_old.size()));
    for (int index = 1; index < static_cast<int>(order.size()); ++index) {
        const int prev_new = order[index - 1];
        const int curr_new = order[index];
        const int prev_old = new_to_old[prev_new];
        const int curr_old = new_to_old[curr_new];
        double weight =
            embedding.cols() > 0 ? std::abs(embedding(curr_new, 0) - embedding(prev_new, 0)) : 1.0;
        if (weight <= 0.0) {
            weight = 1e-6;
        }
        add_undirected_edge(tree, prev_old, curr_old, weight);
    }
}

}  // namespace

WeightedGraph construct_akpw_lsst_tree(const WeightedGraph& graph, const Eigen::MatrixXd& embedding) {
    WeightedGraph tree = make_empty_graph(graph.num_vertices);
    if (graph.num_vertices <= 1) {
        return tree;
    }

    const std::vector<int> new_to_old = spectral_order(embedding, graph.num_vertices);
    const std::vector<int> old_to_new = inverse_permutation(new_to_old);
    const AkpwInputState input_state = make_akpw_input_state_like_julia(graph, old_to_new);
    const std::vector<AkpwEdge>& findnz_edges = input_state.findnz_edges;
    const std::vector<AkpwEdge>& orig_list = input_state.orig_list;
    if (orig_list.empty()) {
        add_fallback_spectral_chain(tree, new_to_old, embedding);
        sort_adjacency(tree);
        return tree;
    }

    const int n = graph.num_vertices;
    const int m = static_cast<int>(orig_list.size());

    std::vector<int> selected_tree_edges;
    selected_tree_edges.reserve(std::max(0, n - 1));

    DisjointSet name_map(n);
    double expansion_factor = akpw_expansion_factor(n);

    const double max_weight = orig_list.front().weight;
    int last = 1;
    const double target = expansion_factor * max_weight;
    while (last < m && orig_list[last].weight > target) {
        ++last;
    }
    --last;

    std::vector<AkpwEdge> current_edges(orig_list.begin(), orig_list.begin() + last + 1);
    int num_vertices_left = compress_indices_like_julia(current_edges);
    current_edges = sort_ijvind_like_julia(current_edges);

    // This follows Laplacians.jl's akpwSub5 loop shape:
    // 1. keep a descending-weight prefix `1:last`,
    // 2. cluster that prefix,
    // 3. contract with the name map,
    // 4. expose more edges while their weight stays above xf * maxv.
    while (num_vertices_left > 1 && !current_edges.empty()) {
        const int previous_tree_size = static_cast<int>(selected_tree_edges.size());
        const int previous_last = last;

        const int nleft = std::max(2, n - previous_tree_size);
        expansion_factor = akpw_expansion_factor(nleft);

        cluster_phase_like_julia(current_edges, expansion_factor, selected_tree_edges);

        for (int index = previous_tree_size; index < static_cast<int>(selected_tree_edges.size()); ++index) {
            const AkpwEdge& edge = findnz_edges[selected_tree_edges[index]];
            const int name_u = name_map.find(edge.original_u);
            const int name_v = name_map.find(edge.original_v);
            if (name_u < name_v) {
                name_map.unite(name_u, name_v);
            } else {
                name_map.unite(name_v, name_u);
            }
        }

        double current_max_weight = 0.0;
        std::vector<AkpwEdge> next_edges =
            contract_edges_like_julia(current_edges, name_map, current_max_weight);
        expose_more_edges_like_julia(
            orig_list, last, expansion_factor, current_max_weight, name_map, next_edges);

        num_vertices_left = compress_indices_like_julia(next_edges);
        if (num_vertices_left > 1) {
            // `sortIJVind + compress` in akpwSub5 uses the same stable two-pass bucket
            // ordering pattern as `combineMultiG`, so we mirror that structure here.
            next_edges = sort_ijvind_like_julia(next_edges);
            compress_sorted_edges_like_julia(next_edges);
        }

        if (selected_tree_edges.size() == static_cast<std::size_t>(previous_tree_size) &&
            last == previous_last) {
            break;
        }

        current_edges = std::move(next_edges);
    }

    DisjointSet tree_components(n);
    for (int edge_index : selected_tree_edges) {
        const AkpwEdge& edge = findnz_edges[edge_index];
        if (tree_components.unite(edge.original_u, edge.original_v)) {
            add_undirected_edge(tree,
                                new_to_old[edge.original_u],
                                new_to_old[edge.original_v],
                                std::max(edge.weight, 1e-6));
        }
    }

    for (const auto& edge : findnz_edges) {
        if (tree_components.unite(edge.original_u, edge.original_v)) {
            add_undirected_edge(tree,
                                new_to_old[edge.original_u],
                                new_to_old[edge.original_v],
                                std::max(edge.weight, 1e-6));
        }
    }

    sort_adjacency(tree);
    return tree;
}

}  // namespace kspecpart
