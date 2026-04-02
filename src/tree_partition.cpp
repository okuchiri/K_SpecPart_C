#include "kspecpart/tree_partition.hpp"

#include "kspecpart/cut_distillation.hpp"
#include "kspecpart/embedding.hpp"
#include "kspecpart/golden_evaluator.hpp"
#include "kspecpart/graphification.hpp"
#include "kspecpart/lsst.hpp"
#include "kspecpart/metis.hpp"
#include "kspecpart/projection.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

struct ScoredPartition {
    std::vector<int> partition;
    PartitionResult metrics;
    long long penalty = 0;
};

struct TreeSweepResult {
    std::vector<int> partition;
    int cutsize = std::numeric_limits<int>::max();
    int cut_point = -1;
};

std::uint64_t make_edge_key(int u, int v) {
    const std::uint32_t a = static_cast<std::uint32_t>(std::min(u, v));
    const std::uint32_t b = static_cast<std::uint32_t>(std::max(u, v));
    return (static_cast<std::uint64_t>(a) << 32U) | b;
}

int total_vertex_weight(const Hypergraph& hypergraph) {
    return std::accumulate(hypergraph.vwts.begin(), hypergraph.vwts.end(), 0);
}

bool partition_complete(const std::vector<int>& partition, int num_vertices) {
    return static_cast<int>(partition.size()) == num_vertices &&
           std::all_of(partition.begin(), partition.end(), [](int part) { return part >= 0; });
}

bool partition_has_all_parts(const std::vector<int>& partition, int num_parts) {
    if (!partition_complete(partition, static_cast<int>(partition.size()))) {
        return false;
    }
    std::vector<char> seen(num_parts, 0);
    for (int part : partition) {
        if (part < 0 || part >= num_parts) {
            return false;
        }
        seen[part] = 1;
    }
    return std::all_of(seen.begin(), seen.end(), [](char value) { return value != 0; });
}

bool fixed_vertices_satisfied(const Hypergraph& hypergraph, const std::vector<int>& partition, int num_parts) {
    if (!partition_complete(partition, hypergraph.num_vertices)) {
        return false;
    }
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        if (hypergraph.fixed[vertex] >= 0) {
            if (hypergraph.fixed[vertex] >= num_parts || partition[vertex] != hypergraph.fixed[vertex]) {
                return false;
            }
        }
    }
    return true;
}

bool edge_is_cut_after_move(const Hypergraph& hypergraph,
                            const std::vector<int>& partition,
                            int edge,
                            int moved_vertex,
                            int new_part) {
    const int start = hypergraph.eptr[edge];
    const int end = hypergraph.eptr[edge + 1];
    const int first_vertex = hypergraph.eind[start];
    const int base_part = first_vertex == moved_vertex ? new_part : partition[first_vertex];
    for (int idx = start + 1; idx < end; ++idx) {
        const int vertex = hypergraph.eind[idx];
        const int part = vertex == moved_vertex ? new_part : partition[vertex];
        if (part != base_part) {
            return true;
        }
    }
    return false;
}

int delta_cut_for_move(const Hypergraph& hypergraph,
                       const std::vector<int>& partition,
                       int vertex,
                       int to_part) {
    const int from_part = partition[vertex];
    if (from_part == to_part) {
        return 0;
    }
    int delta = 0;
    for (int idx = hypergraph.vptr[vertex]; idx < hypergraph.vptr[vertex + 1]; ++idx) {
        const int edge = hypergraph.vind[idx];
        const bool before = edge_is_cut_after_move(hypergraph, partition, edge, -1, -1);
        const bool after = edge_is_cut_after_move(hypergraph, partition, edge, vertex, to_part);
        delta += (after - before) * hypergraph.hwts[edge];
    }
    return delta;
}

std::vector<int> compute_balance(const Hypergraph& hypergraph, const std::vector<int>& partition, int num_parts) {
    std::vector<int> balance(num_parts, 0);
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        const int part = partition[vertex];
        if (part >= 0 && part < num_parts) {
            balance[part] += hypergraph.vwts[vertex];
        }
    }
    return balance;
}

void greedy_assign_unset_vertices(const Hypergraph& hypergraph,
                                  std::vector<int>& partition,
                                  const BalanceLimits& limits,
                                  int num_parts) {
    std::vector<int> balance = compute_balance(hypergraph, partition, num_parts);
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        if (partition[vertex] >= 0) {
            continue;
        }
        int best_part = 0;
        int best_weight = std::numeric_limits<int>::max();
        for (int part = 0; part < num_parts; ++part) {
            const int projected = balance[part] + hypergraph.vwts[vertex];
            if (projected <= limits.max_capacity && balance[part] < best_weight) {
                best_part = part;
                best_weight = balance[part];
            }
        }
        if (best_weight == std::numeric_limits<int>::max()) {
            best_part = static_cast<int>(std::distance(balance.begin(), std::min_element(balance.begin(), balance.end())));
        }
        partition[vertex] = best_part;
        balance[best_part] += hypergraph.vwts[vertex];
    }
}

double coordinate_distance(const Eigen::MatrixXd& embedding, int lhs, int rhs, bool lst) {
    if (embedding.cols() == 0) {
        return 1.0;
    }
    double distance = 0.0;
    for (int col = 0; col < embedding.cols(); ++col) {
        const double span = embedding(lhs, col) - embedding(rhs, col);
        if (lst) {
            distance += span == 0.0 ? 1e9 : 1.0 / (span * span);
        } else {
            distance += std::abs(span);
        }
    }
    return distance > 0.0 ? distance : 1e-6;
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

WeightedGraph make_empty_graph(int num_vertices) {
    WeightedGraph graph;
    graph.num_vertices = num_vertices;
    graph.adjacency.assign(num_vertices, {});
    graph.degrees.assign(num_vertices, 0.0);
    return graph;
}

void sort_graph_adjacency(WeightedGraph& graph) {
    for (auto& neighbors : graph.adjacency) {
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });
    }
}

std::vector<std::vector<int>> vertices_by_part(const std::vector<int>& partition, int num_parts) {
    std::vector<std::vector<int>> result(num_parts);
    for (int vertex = 0; vertex < static_cast<int>(partition.size()); ++vertex) {
        const int part = partition[vertex];
        if (part >= 0 && part < num_parts) {
            result[part].push_back(vertex);
        }
    }
    return result;
}

PartitionIndex one_vs_rest_index(const std::vector<std::vector<int>>& part_vertices, int focus_part) {
    PartitionIndex pindex;
    if (focus_part < 0 || focus_part >= static_cast<int>(part_vertices.size())) {
        return pindex;
    }
    pindex.p1 = part_vertices[focus_part];
    for (int part = 0; part < static_cast<int>(part_vertices.size()); ++part) {
        if (part == focus_part) {
            continue;
        }
        pindex.p2.insert(pindex.p2.end(), part_vertices[part].begin(), part_vertices[part].end());
    }
    return pindex;
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

WeightedGraph reweight_graph(const WeightedGraph& graph, const Eigen::MatrixXd& embedding, bool lst) {
    WeightedGraph reweighted = make_empty_graph(graph.num_vertices);
    for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
        for (const auto& [neighbor, weight] : graph.adjacency[vertex]) {
            if (vertex >= neighbor) {
                continue;
            }
            (void)weight;
            add_undirected_edge(reweighted, vertex, neighbor, coordinate_distance(embedding, vertex, neighbor, lst));
        }
    }
    sort_graph_adjacency(reweighted);
    return reweighted;
}

WeightedGraph construct_mst_tree(const WeightedGraph& graph, const Eigen::MatrixXd& embedding) {
    const int n = graph.num_vertices;
    WeightedGraph tree = make_empty_graph(n);
    if (n <= 1) {
        return tree;
    }

    struct QueueVertex {
        double weight;
        int vertex;
    };

    auto cmp = [](const QueueVertex& lhs, const QueueVertex& rhs) {
        if (lhs.weight != rhs.weight) {
            return lhs.weight > rhs.weight;
        }
        return lhs.vertex > rhs.vertex;
    };
    std::priority_queue<QueueVertex, std::vector<QueueVertex>, decltype(cmp)> queue(cmp);

    const int degree_threshold = 10;
    std::vector<int> finished(n, 0);
    std::vector<double> best_weight(n, std::numeric_limits<double>::infinity());
    std::vector<int> parent(n, -1);
    // Keep the same control structure as Julia's `degrees_aware_prim_mst`.
    // That helper currently checks degree limits but never updates `degrees`.
    std::vector<int> degrees(n, 0);
    std::vector<int> roots;
    double max_weight = 1.0;

    for (int start = 0; start < n; ++start) {
        if (finished[start]) {
            continue;
        }

        roots.push_back(start);
        best_weight[start] = -1.0;
        queue.push({best_weight[start], start});

        while (!queue.empty()) {
            const QueueVertex current = queue.top();
            queue.pop();
            const int vertex = current.vertex;
            if (finished[vertex]) {
                continue;
            }
            if (current.weight > best_weight[vertex]) {
                continue;
            }

            finished[vertex] = 1;
            bool connection_flag = false;

            for (const auto& [neighbor, weight] : graph.adjacency[vertex]) {
                if (degrees[neighbor] + 1 > degree_threshold && degrees[vertex] + 1 > degree_threshold) {
                    continue;
                }
                if (finished[neighbor]) {
                    continue;
                }
                if (best_weight[neighbor] > weight) {
                    best_weight[neighbor] = weight;
                    parent[neighbor] = vertex;
                    queue.push({weight, neighbor});
                    connection_flag = true;
                }
            }

            if (!connection_flag) {
                for (const auto& [neighbor, weight] : graph.adjacency[vertex]) {
                    if (finished[neighbor]) {
                        continue;
                    }
                    if (best_weight[neighbor] > weight) {
                        best_weight[neighbor] = weight;
                        parent[neighbor] = vertex;
                        queue.push({weight, neighbor});
                    }
                }
            }
        }
    }

    for (int vertex = 0; vertex < n; ++vertex) {
        if (parent[vertex] < 0) {
            continue;
        }
        const int source = parent[vertex];
        double edge_weight = 1e-6;
        for (const auto& [neighbor, weight] : graph.adjacency[vertex]) {
            if (neighbor == source) {
                edge_weight = weight;
                break;
            }
        }
        max_weight = std::max(max_weight, edge_weight);
        add_undirected_edge(tree, source, vertex, edge_weight);
    }

    if (roots.size() > 1) {
        std::sort(roots.begin(), roots.end(), [&](int lhs, int rhs) {
            const double lv = embedding.cols() > 0 ? embedding(lhs, 0) : static_cast<double>(lhs);
            const double rv = embedding.cols() > 0 ? embedding(rhs, 0) : static_cast<double>(rhs);
            if (lv != rv) {
                return lv < rv;
            }
            return lhs < rhs;
        });
        for (std::size_t i = 1; i < roots.size(); ++i) {
            const int prev = roots[i - 1];
            const int curr = roots[i];
            const double bridge = max_weight * 2.0 + coordinate_distance(embedding, prev, curr, false);
            add_undirected_edge(tree, prev, curr, bridge);
        }
    }

    sort_graph_adjacency(tree);
    return tree;
}

WeightedGraph construct_path_tree(const Eigen::MatrixXd& embedding) {
    const int n = static_cast<int>(embedding.rows());
    WeightedGraph tree = make_empty_graph(n);
    if (n <= 1) {
        return tree;
    }

    const std::vector<int> order = spectral_order(embedding, n);
    for (int index = 1; index < n; ++index) {
        const int prev = order[index - 1];
        const int curr = order[index];
        double weight = embedding.cols() > 0 ? std::abs(embedding(curr, 0) - embedding(prev, 0)) : 1.0;
        if (weight == 0.0) {
            weight = 1e-6;
        }
        add_undirected_edge(tree, prev, curr, weight);
    }
    sort_graph_adjacency(tree);
    return tree;
}

WeightedGraph construct_tree(const WeightedGraph& graph, const Eigen::MatrixXd& embedding, int tree_type) {
    if (tree_type == 1) {
        return construct_akpw_lsst_tree(graph, embedding);
    }
    if (tree_type == 2) {
        return construct_mst_tree(graph, embedding);
    }
    if (tree_type == 3) {
        return construct_path_tree(embedding);
    }
    return construct_mst_tree(graph, embedding);
}

std::vector<int> compact_partition_labels(const std::vector<int>& components) {
    std::unordered_map<int, int> remap;
    std::vector<int> partition(components.size(), -1);
    int next_label = 0;
    for (std::size_t i = 0; i < components.size(); ++i) {
        const int component = components[i];
        auto [it, inserted] = remap.emplace(component, next_label);
        if (inserted) {
            ++next_label;
        }
        partition[i] = it->second;
    }
    return partition;
}

std::vector<int> connected_components(const WeightedGraph& graph,
                                      const std::unordered_set<std::uint64_t>& removed_edges) {
    std::vector<int> labels(graph.num_vertices, -1);
    int next_label = 0;

    for (int start = 0; start < graph.num_vertices; ++start) {
        if (labels[start] >= 0) {
            continue;
        }
        std::vector<int> stack = {start};
        labels[start] = next_label;
        while (!stack.empty()) {
            const int vertex = stack.back();
            stack.pop_back();
            for (const auto& [neighbor, weight] : graph.adjacency[vertex]) {
                (void)weight;
                if (removed_edges.find(make_edge_key(vertex, neighbor)) != removed_edges.end()) {
                    continue;
                }
                if (labels[neighbor] >= 0) {
                    continue;
                }
                labels[neighbor] = next_label;
                stack.push_back(neighbor);
            }
        }
        ++next_label;
    }
    return labels;
}

int count_parts(const std::vector<int>& partition) {
    int max_part = -1;
    for (int part : partition) {
        max_part = std::max(max_part, part);
    }
    return max_part + 1;
}

int sum_hyperedge_weights(const Hypergraph& hypergraph, const std::vector<int>& edges) {
    int total = 0;
    for (int edge : edges) {
        if (edge >= 0 && edge < hypergraph.num_hyperedges) {
            total += hypergraph.hwts[edge];
        }
    }
    return total;
}

WeightedGraph build_metis_cost_tree(const WeightedGraph& tree,
                                    const CutProfile& distilled_cuts,
                                    const Hypergraph& hypergraph) {
    WeightedGraph weighted_tree = make_empty_graph(tree.num_vertices);
    const int nforced_0 = sum_hyperedge_weights(hypergraph, distilled_cuts.forced_0);
    const int nforced_1 = sum_hyperedge_weights(hypergraph, distilled_cuts.forced_1);
    const int nforced_01 = sum_hyperedge_weights(hypergraph, distilled_cuts.forced_01);

    for (int vertex = 0; vertex < tree.num_vertices; ++vertex) {
        const int parent = distilled_cuts.pred[vertex];
        if (parent < 0 || parent == vertex) {
            continue;
        }

        const int exc_0 = distilled_cuts.edge_cuts[vertex] + nforced_0 - distilled_cuts.FB0[vertex] +
                          distilled_cuts.edge_cuts_1[vertex] + nforced_01;
        const int exc_1 = distilled_cuts.edge_cuts[vertex] + nforced_1 - distilled_cuts.FB1[vertex] +
                          distilled_cuts.edge_cuts_0[vertex] + nforced_01;
        add_undirected_edge(weighted_tree, vertex, parent, std::max(1, std::min(exc_0, exc_1)));
    }

    sort_graph_adjacency(weighted_tree);
    return weighted_tree;
}

std::optional<std::vector<int>> metis_tree_partition_candidate(const WeightedGraph& tree,
                                                               const CutProfile& distilled_cuts,
                                                               const Hypergraph& hypergraph,
                                                               int num_parts,
                                                               const TreePartitionOptions& options,
                                                               const std::optional<std::string>& gpmetis_executable) {
    if (!gpmetis_executable.has_value()) {
        return std::nullopt;
    }

    const WeightedGraph metis_tree = build_metis_cost_tree(tree, distilled_cuts, hypergraph);
    return run_gpmetis_partition(metis_tree, num_parts, *gpmetis_executable, options.imb, options.seed);
}

PartitionIndex partition_index_from_fixed(const Hypergraph& hypergraph) {
    PartitionIndex pindex;
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        if (hypergraph.fixed[vertex] == 0) {
            pindex.p1.push_back(vertex);
        } else if (hypergraph.fixed[vertex] == 1) {
            pindex.p2.push_back(vertex);
        }
    }
    return pindex;
}

Eigen::MatrixXd build_k_way_embedding_from_partition(const Hypergraph& hypergraph,
                                                     const WeightedGraph& graph,
                                                     const std::vector<int>& partition,
                                                     const TreePartitionOptions& options,
                                                     const Eigen::MatrixXd& constraint_basis) {
    const auto part_vertices = vertices_by_part(partition, options.num_parts);
    std::vector<Eigen::MatrixXd> embeddings;
    embeddings.reserve(options.num_parts);
    const int dims = std::max(1, options.eigvecs);
    const int epsilon = std::max(1, options.num_parts - 1);

    for (int part = 0; part < options.num_parts; ++part) {
        const PartitionIndex pindex = one_vs_rest_index(part_vertices, part);
        embeddings.push_back(
            solve_eigs(hypergraph, graph, pindex, false, dims, options.solver_iters, epsilon,
                       options.seed, constraint_basis));
    }

    Eigen::MatrixXd concatenated = concatenate_embeddings(embeddings, hypergraph.num_vertices);
    if (concatenated.cols() == 0) {
        return concatenated;
    }
    return reduce_embedding_for_tree_partition(concatenated,
                                               partition,
                                               options.eigvecs,
                                               options.seed,
                                               options.projection_strategy);
}

int tree_root(const CutProfile& distilled_cuts) {
    for (int vertex = 0; vertex < static_cast<int>(distilled_cuts.pred.size()); ++vertex) {
        if (distilled_cuts.pred[vertex] == vertex) {
            return vertex;
        }
    }
    return 0;
}

TreeSweepResult evaluate_removed_edges(const WeightedGraph& tree,
                                       const Hypergraph& hypergraph,
                                       const std::unordered_set<std::uint64_t>& removed_edges,
                                       int num_parts,
                                       int cut_point) {
    TreeSweepResult result;
    result.cut_point = cut_point;
    result.partition = compact_partition_labels(connected_components(tree, removed_edges));
    if (count_parts(result.partition) != num_parts ||
        !fixed_vertices_satisfied(hypergraph, result.partition, num_parts)) {
        result.partition.clear();
        return result;
    }
    const PartitionResult metrics = evaluate_partition(hypergraph, num_parts, result.partition);
    result.cutsize = metrics.cutsize;
    return result;
}

std::optional<TreeSweepResult> two_way_linear_tree_sweep(const WeightedGraph& tree,
                                                         const CutProfile& distilled_cuts,
                                                         const Hypergraph& hypergraph,
                                                         const BalanceLimits& limits) {
    const int n = hypergraph.num_vertices;
    const int root = tree_root(distilled_cuts);
    const int nforced_0 = sum_hyperedge_weights(hypergraph, distilled_cuts.forced_0);
    const int nforced_1 = sum_hyperedge_weights(hypergraph, distilled_cuts.forced_1);
    const int nforced_01 = sum_hyperedge_weights(hypergraph, distilled_cuts.forced_01);
    const int total_weight = total_vertex_weight(hypergraph);
    const double huge_cost = 1e9;

    std::vector<int> edge_cuts = distilled_cuts.edge_cuts;
    std::vector<int> vtx_cuts = distilled_cuts.vtx_cuts;
    if (root >= 0 && root < n) {
        edge_cuts[root] = hypergraph.num_hyperedges;
        vtx_cuts[root] = 0;
    }

    std::vector<int> exc_0(n, 0);
    std::vector<int> exc_1(n, 0);
    std::vector<double> cut_cost_0(n, huge_cost);
    std::vector<double> cut_cost_1(n, huge_cost);
    std::vector<double> cut_cost(n, huge_cost);
    std::vector<double> total_cost(n, huge_cost);
    std::vector<double> ratio_cost(n, huge_cost);
    std::vector<int> area_util_0(n, 0);
    std::vector<int> area_util_1(n, 0);

    for (int vertex = 0; vertex < n; ++vertex) {
        exc_0[vertex] = edge_cuts[vertex] + nforced_0 - distilled_cuts.FB0[vertex] +
                        distilled_cuts.edge_cuts_1[vertex] + nforced_01;
        exc_1[vertex] = edge_cuts[vertex] + nforced_1 - distilled_cuts.FB1[vertex] +
                        distilled_cuts.edge_cuts_0[vertex] + nforced_01;
        cut_cost_0[vertex] = exc_0[vertex];
        cut_cost_1[vertex] = exc_1[vertex];
        if (cut_cost_0[vertex] <= cut_cost_1[vertex]) {
            cut_cost[vertex] = cut_cost_0[vertex];
            area_util_0[vertex] = vtx_cuts[vertex];
            area_util_1[vertex] = total_weight - vtx_cuts[vertex];
        } else {
            cut_cost[vertex] = cut_cost_1[vertex];
            area_util_1[vertex] = vtx_cuts[vertex];
            area_util_0[vertex] = total_weight - vtx_cuts[vertex];
        }
        double area_cost = 0.0;
        if (area_util_0[vertex] > limits.max_capacity || area_util_1[vertex] > limits.max_capacity) {
            area_cost = huge_cost;
        }
        total_cost[vertex] = cut_cost[vertex] + area_cost;
        if (area_util_0[vertex] > 0 && area_util_1[vertex] > 0) {
            ratio_cost[vertex] = cut_cost[vertex] /
                                 static_cast<double>(area_util_0[vertex] * area_util_1[vertex]);
        }
    }

    std::vector<int> cut_indices(n);
    std::iota(cut_indices.begin(), cut_indices.end(), 0);
    auto total_cost_less = [&](int lhs, int rhs) {
        if (total_cost[lhs] != total_cost[rhs]) {
            return total_cost[lhs] < total_cost[rhs];
        }
        return lhs < rhs;
    };
    std::sort(cut_indices.begin(), cut_indices.end(), total_cost_less);

    int cut_point = -1;
    for (int vertex : cut_indices) {
        if (distilled_cuts.pred[vertex] != vertex) {
            cut_point = vertex;
            break;
        }
    }

    if (cut_point >= 0 && total_cost[cut_point] >= huge_cost) {
        cut_point = -1;
        auto ratio_cost_less = [&](int lhs, int rhs) {
            if (ratio_cost[lhs] != ratio_cost[rhs]) {
                return ratio_cost[lhs] < ratio_cost[rhs];
            }
            return lhs < rhs;
        };
        std::sort(cut_indices.begin(), cut_indices.end(), ratio_cost_less);
        for (int vertex : cut_indices) {
            if (distilled_cuts.pred[vertex] == vertex) {
                continue;
            }
            if (area_util_0[vertex] > limits.max_capacity || area_util_1[vertex] > limits.max_capacity) {
                continue;
            }
            cut_point = vertex;
            break;
        }
    }

    if (cut_point < 0) {
        return std::nullopt;
    }

    const int parent = distilled_cuts.pred[cut_point];
    if (parent < 0 || parent == cut_point) {
        return std::nullopt;
    }

    std::unordered_set<std::uint64_t> removed_edges = {make_edge_key(cut_point, parent)};
    TreeSweepResult result = evaluate_removed_edges(tree, hypergraph, removed_edges, 2, cut_point);
    if (result.partition.empty()) {
        return std::nullopt;
    }
    return result;
}

std::optional<std::vector<int>> k_way_linear_tree_sweep(const WeightedGraph& tree,
                                                        const PartitionIndex& fixed_vertices,
                                                        const Hypergraph& hypergraph,
                                                        const BalanceLimits& limits,
                                                        int num_parts) {
    const CutProfile base_profile = distill_cuts_on_tree(hypergraph, fixed_vertices, tree, 0);
    Hypergraph recursive_hypergraph = hypergraph;
    CutProfile recursive_profile = base_profile;
    std::vector<int> cut_points;
    BalanceLimits recursive_limits{limits.min_capacity, total_vertex_weight(hypergraph) - limits.min_capacity};

    for (int level = 0; level < num_parts - 1; ++level) {
        std::optional<TreeSweepResult> result = two_way_linear_tree_sweep(tree, recursive_profile, recursive_hypergraph, recursive_limits);
        if (!result.has_value() || result->cut_point < 0) {
            return std::nullopt;
        }

        const std::vector<int> blocks = compute_balance(recursive_hypergraph, result->partition, 2);
        const int smaller_side = blocks[0] <= blocks[1] ? 0 : 1;
        int recursive_total = 0;
        for (int vertex = 0; vertex < recursive_hypergraph.num_vertices; ++vertex) {
            if (result->partition[vertex] == smaller_side) {
                recursive_hypergraph.vwts[vertex] = 0;
            } else {
                recursive_total += recursive_hypergraph.vwts[vertex];
            }
        }

        cut_points.push_back(result->cut_point);
        recursive_limits.max_capacity = recursive_total - limits.min_capacity;
        if (level + 1 < num_parts - 1) {
            if (recursive_limits.max_capacity < recursive_limits.min_capacity) {
                return std::nullopt;
            }
            recursive_profile = distill_cuts_on_tree(recursive_hypergraph, fixed_vertices, tree, 0);
        }
    }

    std::unordered_set<std::uint64_t> removed_edges;
    for (int cut_point : cut_points) {
        const int parent = base_profile.pred[cut_point];
        if (parent < 0 || parent == cut_point) {
            return std::nullopt;
        }
        removed_edges.insert(make_edge_key(cut_point, parent));
    }

    TreeSweepResult result = evaluate_removed_edges(tree, hypergraph, removed_edges, num_parts, cut_points.empty() ? -1 : cut_points.back());
    if (result.partition.empty()) {
        return std::nullopt;
    }
    return result.partition;
}

std::vector<int> select_columns(const Eigen::MatrixXd& embedding, const std::vector<int>& cols) {
    std::vector<int> normalized = cols;
    normalized.erase(std::remove_if(normalized.begin(), normalized.end(), [&](int col) {
        return col < 0 || col >= embedding.cols();
    }), normalized.end());
    return normalized;
}

Eigen::MatrixXd slice_embedding(const Eigen::MatrixXd& embedding, const std::vector<int>& cols) {
    const std::vector<int> normalized = select_columns(embedding, cols);
    Eigen::MatrixXd sliced(embedding.rows(), static_cast<int>(normalized.size()));
    for (int index = 0; index < static_cast<int>(normalized.size()); ++index) {
        sliced.col(index) = embedding.col(normalized[index]);
    }
    return sliced;
}

void enumerate_subsets_recursive(int dims,
                                 int start,
                                 std::vector<int>& current,
                                 std::vector<std::vector<int>>& subsets) {
    for (int index = start; index < dims; ++index) {
        current.push_back(index);
        subsets.push_back(current);
        enumerate_subsets_recursive(dims, index + 1, current, subsets);
        current.pop_back();
    }
}

std::vector<std::vector<int>> enumerate_embedding_subsets(int dims) {
    if (dims <= 0) {
        return {};
    }
    std::vector<std::vector<int>> subsets;
    std::vector<int> current;
    enumerate_subsets_recursive(dims, 0, current, subsets);
    return subsets;
}

std::vector<ScoredPartition> score_candidates(const Hypergraph& hypergraph,
                                              const std::vector<std::vector<int>>& candidates,
                                              int num_parts,
                                              const BalanceLimits& limits) {
    std::unordered_set<std::vector<int>, VectorHasher> seen;
    std::vector<ScoredPartition> scored;
    for (const auto& candidate : candidates) {
        if (!partition_complete(candidate, hypergraph.num_vertices)) {
            continue;
        }
        if (!fixed_vertices_satisfied(hypergraph, candidate, num_parts)) {
            continue;
        }
        if (!seen.insert(candidate).second) {
            continue;
        }
        PartitionResult metrics = evaluate_partition(hypergraph, num_parts, candidate);
        scored.push_back({candidate, metrics, balance_penalty(metrics.balance, limits)});
    }

    std::sort(scored.begin(), scored.end(), [](const ScoredPartition& lhs, const ScoredPartition& rhs) {
        if ((lhs.penalty == 0) != (rhs.penalty == 0)) {
            return lhs.penalty == 0;
        }
        if (lhs.penalty != rhs.penalty) {
            return lhs.penalty < rhs.penalty;
        }
        return lhs.metrics.cutsize < rhs.metrics.cutsize;
    });
    return scored;
}

std::vector<std::vector<int>> generate_tree_candidates(const Hypergraph& hypergraph,
                                                       const WeightedGraph& graph,
                                                       const Eigen::MatrixXd& embedding,
                                                       const PartitionIndex& fixed_vertices,
                                                       int num_parts,
                                                       const BalanceLimits& limits,
                                                       const TreePartitionOptions& options,
                                                       const std::optional<std::string>& gpmetis_executable) {
    std::vector<std::vector<int>> candidates;
    for (const auto& dims : enumerate_embedding_subsets(embedding.cols())) {
        Eigen::MatrixXd sliced = slice_embedding(embedding, dims);
        if (sliced.cols() == 0) {
            continue;
        }
        for (int tree_type = 1; tree_type <= 2; ++tree_type) {
            const bool lst = tree_type == 1;
            WeightedGraph reweighted = reweight_graph(graph, sliced, lst);
            WeightedGraph tree = construct_tree(reweighted, sliced, tree_type);
            const CutProfile distilled_cuts = distill_cuts_on_tree(hypergraph, fixed_vertices, tree, 0);
            std::optional<std::vector<int>> metis_partition =
                metis_tree_partition_candidate(tree, distilled_cuts, hypergraph, num_parts, options, gpmetis_executable);
            if (metis_partition.has_value()) {
                candidates.push_back(std::move(*metis_partition));
            }
            if (num_parts == 2) {
                std::optional<TreeSweepResult> result = two_way_linear_tree_sweep(tree, distilled_cuts, hypergraph, limits);
                if (result.has_value()) {
                    candidates.push_back(std::move(result->partition));
                }
            } else {
                std::optional<std::vector<int>> partition =
                    k_way_linear_tree_sweep(tree, fixed_vertices, hypergraph, limits, num_parts);
                if (partition.has_value()) {
                    candidates.push_back(std::move(partition.value()));
                }
            }
        }
    }
    return candidates;
}

std::vector<std::vector<int>> generate_two_way_candidates(const Hypergraph& hypergraph,
                                                          const WeightedGraph& graph,
                                                          const Eigen::MatrixXd& embedding,
                                                          const PartitionIndex& fixed_vertices,
                                                          const std::vector<int>& base_partition,
                                                          const BalanceLimits& limits,
                                                          const TreePartitionOptions& options,
                                                          const std::optional<std::string>& gpmetis_executable,
                                                          int best_solns,
                                                          std::mt19937& rng) {
    (void)rng;
    (void)best_solns;
    std::vector<std::vector<int>> candidates =
        generate_tree_candidates(hypergraph, graph, embedding, fixed_vertices, 2, limits, options, gpmetis_executable);
    if (partition_complete(base_partition, hypergraph.num_vertices) &&
        fixed_vertices_satisfied(hypergraph, base_partition, 2)) {
        candidates.push_back(base_partition);
    }
    return candidates;
}

std::vector<std::vector<int>> generate_k_way_candidates(const Hypergraph& hypergraph,
                                                        const WeightedGraph& graph,
                                                        const Eigen::MatrixXd& embedding,
                                                        const PartitionIndex& fixed_vertices,
                                                        const std::vector<int>& base_partition,
                                                        int num_parts,
                                                        const BalanceLimits& limits,
                                                        const TreePartitionOptions& options,
                                                        const std::optional<std::string>& gpmetis_executable,
                                                        int best_solns,
                                                        std::mt19937& rng) {
    (void)rng;
    (void)best_solns;
    std::vector<std::vector<int>> candidates = generate_tree_candidates(
        hypergraph, graph, embedding, fixed_vertices, num_parts, limits, options, gpmetis_executable);
    if (partition_complete(base_partition, hypergraph.num_vertices) &&
        fixed_vertices_satisfied(hypergraph, base_partition, num_parts)) {
        candidates.push_back(base_partition);
    }
    return candidates;
}

std::vector<int> fallback_partition(const Hypergraph& hypergraph, int num_parts) {
    std::vector<int> fallback(hypergraph.num_vertices, 0);
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        if (hypergraph.fixed[vertex] >= 0) {
            fallback[vertex] = hypergraph.fixed[vertex];
        } else {
            fallback[vertex] = vertex % num_parts;
        }
    }
    return fallback;
}

}  // namespace

BalanceLimits compute_balance_limits(const Hypergraph& hypergraph, int num_parts, int imb) {
    const int total = total_vertex_weight(hypergraph);
    const double avg = static_cast<double>(total) / static_cast<double>(num_parts);
    BalanceLimits limits;
    limits.min_capacity = static_cast<int>(std::floor(avg * (100.0 - imb * num_parts) / 100.0));
    limits.max_capacity = static_cast<int>(std::ceil(avg * (100.0 + imb * num_parts) / 100.0));

    const int julia_max = static_cast<int>(std::ceil(total * ((100.0 / num_parts) + imb) / 100.0));
    const int julia_min = static_cast<int>(std::floor(total * ((100.0 / num_parts) - imb) / 100.0));
    limits.min_capacity = std::max(0, julia_min);
    limits.max_capacity = std::max(limits.min_capacity, julia_max);
    return limits;
}

long long balance_penalty(const std::vector<int>& balance, const BalanceLimits& limits) {
    long long penalty = 0;
    for (int weight : balance) {
        if (weight < limits.min_capacity) {
            penalty += static_cast<long long>(limits.min_capacity - weight) * (limits.min_capacity - weight);
        }
        if (weight > limits.max_capacity) {
            penalty += static_cast<long long>(weight - limits.max_capacity) * (weight - limits.max_capacity);
        }
    }
    return penalty;
}

std::vector<int> local_refine_partition(const Hypergraph& hypergraph,
                                        std::vector<int> partition,
                                        int num_parts,
                                        const BalanceLimits& limits,
                                        std::mt19937& rng) {
    greedy_assign_unset_vertices(hypergraph, partition, limits, num_parts);
    std::vector<int> balance = compute_balance(hypergraph, partition, num_parts);

    for (int round = 0; round < hypergraph.num_vertices * 8; ++round) {
        bool changed = false;

        for (int src = 0; src < num_parts; ++src) {
            while (balance[src] > limits.max_capacity) {
                int best_vertex = -1;
                int best_dst = -1;
                int best_delta = std::numeric_limits<int>::max();
                for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
                    if (partition[vertex] != src || hypergraph.fixed[vertex] >= 0) {
                        continue;
                    }
                    for (int dst = 0; dst < num_parts; ++dst) {
                        if (dst == src) {
                            continue;
                        }
                        if (balance[dst] + hypergraph.vwts[vertex] > limits.max_capacity) {
                            continue;
                        }
                        const int delta = delta_cut_for_move(hypergraph, partition, vertex, dst);
                        if (delta < best_delta ||
                            (delta == best_delta && balance[dst] < (best_dst >= 0 ? balance[best_dst] : std::numeric_limits<int>::max()))) {
                            best_vertex = vertex;
                            best_dst = dst;
                            best_delta = delta;
                        }
                    }
                }
                if (best_vertex < 0) {
                    break;
                }
                const int weight = hypergraph.vwts[best_vertex];
                partition[best_vertex] = best_dst;
                balance[src] -= weight;
                balance[best_dst] += weight;
                changed = true;
            }
        }

        for (int dst = 0; dst < num_parts; ++dst) {
            while (balance[dst] < limits.min_capacity) {
                int best_vertex = -1;
                int best_src = -1;
                int best_delta = std::numeric_limits<int>::max();
                for (int src = 0; src < num_parts; ++src) {
                    if (src == dst || balance[src] <= limits.min_capacity) {
                        continue;
                    }
                    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
                        if (partition[vertex] != src || hypergraph.fixed[vertex] >= 0) {
                            continue;
                        }
                        const int weight = hypergraph.vwts[vertex];
                        if (balance[src] - weight < limits.min_capacity ||
                            balance[dst] + weight > limits.max_capacity) {
                            continue;
                        }
                        const int delta = delta_cut_for_move(hypergraph, partition, vertex, dst);
                        if (delta < best_delta) {
                            best_vertex = vertex;
                            best_src = src;
                            best_delta = delta;
                        }
                    }
                }
                if (best_vertex < 0) {
                    break;
                }
                const int weight = hypergraph.vwts[best_vertex];
                partition[best_vertex] = dst;
                balance[best_src] -= weight;
                balance[dst] += weight;
                changed = true;
            }
        }

        if (!changed) {
            break;
        }
    }

    (void)rng;
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        if (hypergraph.fixed[vertex] >= 0) {
            partition[vertex] = hypergraph.fixed[vertex];
        }
    }
    return partition;
}

std::vector<TreePartitionCandidate> tree_partition_with_embedding(const Hypergraph& hypergraph,
                                                                  const WeightedGraph& graph,
                                                                  const Eigen::MatrixXd& embedding,
                                                                  const PartitionIndex& fixed_vertices,
                                                                  const TreePartitionOptions& options,
                                                                  const std::vector<int>& base_partition,
                                                                  std::mt19937& rng) {
    std::vector<TreePartitionCandidate> result;
    if (hypergraph.num_vertices == 0) {
        return result;
    }
    if (hypergraph.num_vertices == 1) {
        const int part = hypergraph.fixed[0] >= 0 ? hypergraph.fixed[0] : 0;
        result.push_back({{part}, 0, {hypergraph.vwts[0]}});
        return result;
    }

    const BalanceLimits limits = compute_balance_limits(hypergraph, options.num_parts, options.imb);
    const std::optional<std::string> gpmetis_executable =
        options.enable_metis ? resolve_gpmetis_executable(options.gpmetis_executable) : std::nullopt;
    if (options.enable_metis && options.gpmetis_explicit && !gpmetis_executable.has_value()) {
        throw std::runtime_error("failed to resolve gpmetis executable: " + options.gpmetis_executable);
    }
    std::vector<std::vector<int>> candidates = options.num_parts == 2
        ? generate_two_way_candidates(hypergraph,
                                      graph,
                                      embedding,
                                      fixed_vertices,
                                      base_partition,
                                      limits,
                                      options,
                                      gpmetis_executable,
                                      options.best_solns,
                                      rng)
        : generate_k_way_candidates(hypergraph,
                                    graph,
                                    embedding,
                                    fixed_vertices,
                                    base_partition,
                                    options.num_parts,
                                    limits,
                                    options,
                                    gpmetis_executable,
                                    options.best_solns,
                                    rng);

    auto scored = score_candidates(hypergraph, candidates, options.num_parts, limits);
    if (options.best_solns > 0 && static_cast<int>(scored.size()) > options.best_solns) {
        scored.resize(options.best_solns);
    }
    result.reserve(scored.size());
    for (const auto& candidate : scored) {
        result.push_back({candidate.partition, candidate.metrics.cutsize, candidate.metrics.balance});
    }
    return result;
}

std::vector<TreePartitionCandidate> tree_partition(const Hypergraph& hypergraph,
                                                   const TreePartitionOptions& options,
                                                   const std::vector<int>& base_partition,
                                                   std::mt19937& rng,
                                                   const Eigen::MatrixXd& constraint_basis) {
    if (hypergraph.num_vertices == 0) {
        return {};
    }
    if (hypergraph.num_vertices == 1) {
        const int part = hypergraph.fixed[0] >= 0 ? hypergraph.fixed[0] : 0;
        return {{{part}, 0, {hypergraph.vwts[0]}}};
    }

    WeightedGraph graph = hypergraph_to_graph(hypergraph, options.cycles, rng);
    const PartitionIndex fixed_vertices = partition_index_from_fixed(hypergraph);
    Eigen::MatrixXd embedding;
    if (options.num_parts > 2 && partition_has_all_parts(base_partition, options.num_parts)) {
        embedding = build_k_way_embedding_from_partition(
            hypergraph, graph, base_partition, options, constraint_basis);
    } else {
        const int dims = options.num_parts == 2
            ? std::max(1, std::min(hypergraph.num_vertices - 1, std::max(1, options.eigvecs)))
            : std::max(1, std::min(hypergraph.num_vertices - 1, std::max(options.eigvecs, options.num_parts - 1)));
        const int epsilon = options.num_parts == 2 ? 1 : std::max(1, options.num_parts - 1);
        embedding =
            solve_eigs(hypergraph, graph, fixed_vertices, false, dims, options.solver_iters, epsilon,
                       options.seed, constraint_basis);
    }
    return tree_partition_with_embedding(hypergraph, graph, embedding, fixed_vertices, options, base_partition, rng);
}

std::vector<int> tree_partition_best_with_embedding(const Hypergraph& hypergraph,
                                                    const WeightedGraph& graph,
                                                    const Eigen::MatrixXd& embedding,
                                                    const PartitionIndex& fixed_vertices,
                                                    const TreePartitionOptions& options,
                                                    const std::vector<int>& base_partition,
                                                    std::mt19937& rng) {
    const BalanceLimits limits = compute_balance_limits(hypergraph, options.num_parts, options.imb);
    std::vector<TreePartitionCandidate> candidates = tree_partition_with_embedding(hypergraph, graph, embedding, fixed_vertices, options, base_partition, rng);

    std::vector<int> best_partition;
    long long best_penalty = std::numeric_limits<long long>::max();
    int best_cut = std::numeric_limits<int>::max();

    for (const auto& candidate : candidates) {
        std::vector<int> refined = local_refine_partition(hypergraph, candidate.partition, options.num_parts, limits, rng);
        PartitionResult metrics = evaluate_partition(hypergraph, options.num_parts, refined);
        const long long penalty = balance_penalty(metrics.balance, limits);
        if (penalty < best_penalty || (penalty == best_penalty && metrics.cutsize < best_cut)) {
            best_partition = std::move(refined);
            best_penalty = penalty;
            best_cut = metrics.cutsize;
        }
    }

    if (!best_partition.empty()) {
        return best_partition;
    }
    if (partition_complete(base_partition, hypergraph.num_vertices) &&
        fixed_vertices_satisfied(hypergraph, base_partition, options.num_parts)) {
        return local_refine_partition(hypergraph, base_partition, options.num_parts, limits, rng);
    }
    return local_refine_partition(hypergraph, fallback_partition(hypergraph, options.num_parts), options.num_parts, limits, rng);
}

std::vector<int> tree_partition_best(const Hypergraph& hypergraph,
                                     const TreePartitionOptions& options,
                                     const std::vector<int>& base_partition,
                                     std::mt19937& rng,
                                     const Eigen::MatrixXd& constraint_basis) {
    const BalanceLimits limits = compute_balance_limits(hypergraph, options.num_parts, options.imb);
    std::vector<TreePartitionCandidate> candidates =
        tree_partition(hypergraph, options, base_partition, rng, constraint_basis);

    std::vector<int> best_partition;
    long long best_penalty = std::numeric_limits<long long>::max();
    int best_cut = std::numeric_limits<int>::max();

    for (const auto& candidate : candidates) {
        std::vector<int> refined = local_refine_partition(hypergraph, candidate.partition, options.num_parts, limits, rng);
        PartitionResult metrics = evaluate_partition(hypergraph, options.num_parts, refined);
        const long long penalty = balance_penalty(metrics.balance, limits);
        if (penalty < best_penalty || (penalty == best_penalty && metrics.cutsize < best_cut)) {
            best_partition = std::move(refined);
            best_penalty = penalty;
            best_cut = metrics.cutsize;
        }
    }

    if (!best_partition.empty()) {
        return best_partition;
    }
    if (partition_complete(base_partition, hypergraph.num_vertices) &&
        fixed_vertices_satisfied(hypergraph, base_partition, options.num_parts)) {
        return local_refine_partition(hypergraph, base_partition, options.num_parts, limits, rng);
    }
    return local_refine_partition(hypergraph, fallback_partition(hypergraph, options.num_parts), options.num_parts, limits, rng);
}

std::vector<int> partition_two_way_hypergraph(const Hypergraph& hypergraph,
                                              int imb,
                                              int eigvecs,
                                              int solver_iters,
                                              int cycles,
                                              int best_solns,
                                              const std::vector<int>& base_partition,
                                              std::mt19937& rng,
                                              const Eigen::MatrixXd& constraint_basis) {
    TreePartitionOptions options;
    options.num_parts = 2;
    options.imb = imb;
    options.eigvecs = eigvecs;
    options.solver_iters = solver_iters;
    options.cycles = cycles;
    options.best_solns = best_solns;
    return tree_partition_best(hypergraph, options, base_partition, rng, constraint_basis);
}

std::vector<int> partition_k_way_hypergraph(const Hypergraph& hypergraph,
                                            int num_parts,
                                            int imb,
                                            int eigvecs,
                                            int solver_iters,
                                            int cycles,
                                            int best_solns,
                                            const std::vector<int>& base_partition,
                                            std::mt19937& rng,
                                            ProjectionStrategy projection_strategy,
                                            const Eigen::MatrixXd& constraint_basis) {
    TreePartitionOptions options;
    options.num_parts = num_parts;
    options.imb = imb;
    options.eigvecs = eigvecs;
    options.solver_iters = solver_iters;
    options.cycles = cycles;
    options.best_solns = best_solns;
    options.projection_strategy = projection_strategy;
    return tree_partition_best(hypergraph, options, base_partition, rng, constraint_basis);
}

}  // namespace kspecpart
