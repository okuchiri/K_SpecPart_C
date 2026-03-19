#include "kspecpart/specpart.hpp"

#include "kspecpart/embedding.hpp"
#include "kspecpart/golden_evaluator.hpp"
#include "kspecpart/graphification.hpp"
#include "kspecpart/hypergraph.hpp"
#include "kspecpart/io.hpp"
#include "kspecpart/isolate_islands.hpp"
#include "kspecpart/overlay.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
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

struct BalanceLimits {
    int min_capacity = 0;
    int max_capacity = 0;
};

struct ScoredPartition {
    std::vector<int> partition;
    PartitionResult metrics;
    long long penalty = 0;
};

struct ProcessedRefineResult {
    std::vector<int> partition;
    PartitionResult metrics;
};

int total_vertex_weight(const Hypergraph& hypergraph) {
    return std::accumulate(hypergraph.vwts.begin(), hypergraph.vwts.end(), 0);
}

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

bool partition_complete(const std::vector<int>& partition, int num_vertices) {
    return static_cast<int>(partition.size()) == num_vertices &&
           std::all_of(partition.begin(), partition.end(), [](int part) { return part >= 0; });
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

void repair_balance(const Hypergraph& hypergraph,
                    std::vector<int>& partition,
                    const BalanceLimits& limits,
                    int num_parts,
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

    std::vector<int> order(hypergraph.num_vertices);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    for (int vertex : order) {
        if (hypergraph.fixed[vertex] >= 0) {
            partition[vertex] = hypergraph.fixed[vertex];
        }
    }
}

template <typename DerivedA, typename DerivedB>
double squared_distance(const Eigen::MatrixBase<DerivedA>& lhs, const Eigen::MatrixBase<DerivedB>& rhs) {
    return (lhs - rhs).squaredNorm();
}

std::vector<int> assign_by_centers(const Eigen::MatrixXd& embedding,
                                   const Hypergraph& hypergraph,
                                   const Eigen::MatrixXd& centers,
                                   const BalanceLimits& limits,
                                   int num_parts,
                                   std::mt19937& rng) {
    std::vector<int> partition(hypergraph.num_vertices, -1);
    std::vector<int> balance(num_parts, 0);

    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        if (hypergraph.fixed[vertex] >= 0 && hypergraph.fixed[vertex] < num_parts) {
            partition[vertex] = hypergraph.fixed[vertex];
            balance[partition[vertex]] += hypergraph.vwts[vertex];
        }
    }

    std::vector<int> order;
    order.reserve(hypergraph.num_vertices);
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        if (partition[vertex] < 0) {
            order.push_back(vertex);
        }
    }
    std::shuffle(order.begin(), order.end(), rng);

    for (int vertex : order) {
        std::vector<std::pair<double, int>> scores;
        scores.reserve(num_parts);
        for (int part = 0; part < num_parts; ++part) {
            scores.push_back({squared_distance(embedding.row(vertex), centers.row(part)), part});
        }
        std::sort(scores.begin(), scores.end());
        int assigned_part = scores.front().second;
        for (const auto& [distance, part] : scores) {
            (void)distance;
            if (balance[part] + hypergraph.vwts[vertex] <= limits.max_capacity) {
                assigned_part = part;
                break;
            }
        }
        partition[vertex] = assigned_part;
        balance[assigned_part] += hypergraph.vwts[vertex];
    }

    repair_balance(hypergraph, partition, limits, num_parts, rng);
    return partition;
}

Eigen::MatrixXd initialize_centers(const Eigen::MatrixXd& embedding,
                                   const Hypergraph& hypergraph,
                                   int num_parts,
                                   std::mt19937& rng) {
    Eigen::MatrixXd centers = Eigen::MatrixXd::Zero(num_parts, embedding.cols());
    std::vector<int> initialized(num_parts, 0);

    for (int part = 0; part < num_parts; ++part) {
        double total_weight = 0.0;
        for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
            if (hypergraph.fixed[vertex] == part) {
                centers.row(part) += static_cast<double>(hypergraph.vwts[vertex]) * embedding.row(vertex);
                total_weight += hypergraph.vwts[vertex];
            }
        }
        if (total_weight > 0.0) {
            centers.row(part) /= total_weight;
            initialized[part] = 1;
        }
    }

    std::uniform_int_distribution<int> pick_any(0, hypergraph.num_vertices - 1);
    int first = pick_any(rng);
    for (int part = 0; part < num_parts; ++part) {
        if (initialized[part]) {
            continue;
        }
        if (std::all_of(initialized.begin(), initialized.end(), [](int flag) { return flag == 0; })) {
            centers.row(part) = embedding.row(first);
            initialized[part] = 1;
            continue;
        }
        std::vector<double> distances(hypergraph.num_vertices, 0.0);
        double total = 0.0;
        for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
            double best = std::numeric_limits<double>::max();
            for (int existing = 0; existing < num_parts; ++existing) {
                if (!initialized[existing]) {
                    continue;
                }
                best = std::min(best, squared_distance(embedding.row(vertex), centers.row(existing)));
            }
            distances[vertex] = best;
            total += best;
        }
        if (total <= 0.0) {
            centers.row(part) = embedding.row(pick_any(rng));
            initialized[part] = 1;
            continue;
        }
        std::uniform_real_distribution<double> pick(0.0, total);
        double target = pick(rng);
        double prefix = 0.0;
        int chosen = 0;
        for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
            prefix += distances[vertex];
            if (prefix >= target) {
                chosen = vertex;
                break;
            }
        }
        centers.row(part) = embedding.row(chosen);
        initialized[part] = 1;
    }
    return centers;
}

std::vector<int> balanced_kmeans_partition(const Eigen::MatrixXd& embedding,
                                           const Hypergraph& hypergraph,
                                           int num_parts,
                                           const BalanceLimits& limits,
                                           std::mt19937& rng) {
    if (hypergraph.num_vertices == 0) {
        return {};
    }
    if (hypergraph.num_vertices <= num_parts) {
        std::vector<int> partition(hypergraph.num_vertices, 0);
        for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
            partition[vertex] = std::min(vertex, num_parts - 1);
            if (hypergraph.fixed[vertex] >= 0) {
                partition[vertex] = hypergraph.fixed[vertex];
            }
        }
        repair_balance(hypergraph, partition, limits, num_parts, rng);
        return partition;
    }

    Eigen::MatrixXd centers = initialize_centers(embedding, hypergraph, num_parts, rng);
    std::vector<int> partition(hypergraph.num_vertices, 0);

    for (int iter = 0; iter < 12; ++iter) {
        partition = assign_by_centers(embedding, hypergraph, centers, limits, num_parts, rng);
        Eigen::MatrixXd updated = Eigen::MatrixXd::Zero(num_parts, embedding.cols());
        std::vector<double> weight_sum(num_parts, 0.0);
        for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
            const int part = partition[vertex];
            updated.row(part) += static_cast<double>(hypergraph.vwts[vertex]) * embedding.row(vertex);
            weight_sum[part] += hypergraph.vwts[vertex];
        }
        for (int part = 0; part < num_parts; ++part) {
            if (weight_sum[part] > 0.0) {
                updated.row(part) /= weight_sum[part];
            } else {
                updated.row(part) = embedding.row(std::uniform_int_distribution<int>(0, hypergraph.num_vertices - 1)(rng));
            }
        }
        centers = updated;
    }
    repair_balance(hypergraph, partition, limits, num_parts, rng);
    return partition;
}

std::vector<int> sweep_two_way_partition(const Hypergraph& hypergraph,
                                         const Eigen::VectorXd& scores,
                                         const BalanceLimits& limits) {
    std::vector<int> order(hypergraph.num_vertices);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) { return scores[lhs] < scores[rhs]; });

    std::vector<int> best_partition;
    int best_cut = std::numeric_limits<int>::max();
    int prefix_weight = 0;
    const int total_weight = total_vertex_weight(hypergraph);

    for (int split = 1; split < hypergraph.num_vertices; ++split) {
        prefix_weight += hypergraph.vwts[order[split - 1]];
        const int suffix_weight = total_weight - prefix_weight;
        if (prefix_weight < limits.min_capacity || prefix_weight > limits.max_capacity ||
            suffix_weight < limits.min_capacity || suffix_weight > limits.max_capacity) {
            continue;
        }

        std::vector<int> partition(hypergraph.num_vertices, 1);
        for (int i = 0; i < split; ++i) {
            partition[order[i]] = 0;
        }

        bool fixed_ok = true;
        for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
            if (hypergraph.fixed[vertex] >= 0 && hypergraph.fixed[vertex] != partition[vertex]) {
                fixed_ok = false;
                break;
            }
        }
        if (!fixed_ok) {
            continue;
        }

        const PartitionResult metrics = evaluate_partition(hypergraph, 2, partition);
        if (metrics.cutsize < best_cut) {
            best_cut = metrics.cutsize;
            best_partition = std::move(partition);
        }
    }
    return best_partition;
}

std::vector<int> local_refine_partition(const Hypergraph& hypergraph,
                                        std::vector<int> partition,
                                        int num_parts,
                                        const BalanceLimits& limits,
                                        std::mt19937& rng) {
    repair_balance(hypergraph, partition, limits, num_parts, rng);
    std::vector<int> balance = compute_balance(hypergraph, partition, num_parts);

    for (int pass = 0; pass < 4; ++pass) {
        std::vector<int> order(hypergraph.num_vertices);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        bool changed = false;

        for (int vertex : order) {
            if (hypergraph.fixed[vertex] >= 0) {
                partition[vertex] = hypergraph.fixed[vertex];
                continue;
            }

            const int from = partition[vertex];
            int best_to = from;
            int best_delta = 0;
            for (int to = 0; to < num_parts; ++to) {
                if (to == from) {
                    continue;
                }
                const int weight = hypergraph.vwts[vertex];
                if (balance[to] + weight > limits.max_capacity) {
                    continue;
                }
                if (balance[from] - weight < limits.min_capacity) {
                    continue;
                }
                const int delta = delta_cut_for_move(hypergraph, partition, vertex, to);
                if (delta < best_delta) {
                    best_delta = delta;
                    best_to = to;
                }
            }

            if (best_to != from) {
                const int weight = hypergraph.vwts[vertex];
                partition[vertex] = best_to;
                balance[from] -= weight;
                balance[best_to] += weight;
                changed = true;
            }
        }

        if (!changed) {
            break;
        }
    }
    repair_balance(hypergraph, partition, limits, num_parts, rng);
    return partition;
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

std::vector<std::vector<int>> generate_two_way_candidates(const Hypergraph& hypergraph,
                                                          const Eigen::MatrixXd& embedding,
                                                          const std::vector<int>& base_partition,
                                                          const BalanceLimits& limits,
                                                          int best_solns,
                                                          std::mt19937& rng) {
    std::vector<std::vector<int>> candidates;
    if (partition_complete(base_partition, hypergraph.num_vertices)) {
        candidates.push_back(base_partition);
    }

    if (embedding.cols() > 0) {
        for (int col = 0; col < embedding.cols(); ++col) {
            auto sweep = sweep_two_way_partition(hypergraph, embedding.col(col), limits);
            if (!sweep.empty()) {
                candidates.push_back(std::move(sweep));
            }
            auto reverse_sweep = sweep_two_way_partition(hypergraph, -embedding.col(col), limits);
            if (!reverse_sweep.empty()) {
                candidates.push_back(std::move(reverse_sweep));
            }
        }
    }

    const int runs = std::max(4, best_solns * 3);
    for (int run = 0; run < runs; ++run) {
        candidates.push_back(balanced_kmeans_partition(embedding, hypergraph, 2, limits, rng));
    }
    return candidates;
}

std::vector<std::vector<int>> generate_k_way_candidates(const Hypergraph& hypergraph,
                                                        const Eigen::MatrixXd& embedding,
                                                        const std::vector<int>& base_partition,
                                                        int num_parts,
                                                        const BalanceLimits& limits,
                                                        int best_solns,
                                                        std::mt19937& rng) {
    std::vector<std::vector<int>> candidates;
    if (partition_complete(base_partition, hypergraph.num_vertices)) {
        candidates.push_back(base_partition);
    }

    const int runs = std::max(6, best_solns * 4);
    for (int run = 0; run < runs; ++run) {
        candidates.push_back(balanced_kmeans_partition(embedding, hypergraph, num_parts, limits, rng));
    }
    return candidates;
}

std::vector<int> partition_two_way_hypergraph(const Hypergraph& hypergraph,
                                              int imb,
                                              int eigvecs,
                                              int solver_iters,
                                              int cycles,
                                              int best_solns,
                                              const std::vector<int>& base_partition,
                                              std::mt19937& rng) {
    if (hypergraph.num_vertices == 0) {
        return {};
    }
    if (hypergraph.num_vertices == 1) {
        return {hypergraph.fixed[0] >= 0 ? hypergraph.fixed[0] : 0};
    }

    const int num_parts = 2;
    const BalanceLimits limits = compute_balance_limits(hypergraph, num_parts, imb);
    WeightedGraph graph = hypergraph_to_graph(hypergraph, cycles, rng);
    const int dims = std::max(1, std::min(hypergraph.num_vertices - 1, std::max(1, eigvecs)));
    Eigen::MatrixXd embedding = leading_eigenvectors(graph, dims, solver_iters, rng);
    auto candidates = generate_two_way_candidates(hypergraph, embedding, base_partition, limits, best_solns, rng);
    auto scored = score_candidates(hypergraph, candidates, num_parts, limits);
    if (scored.empty()) {
        std::vector<int> fallback(hypergraph.num_vertices, 0);
        for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
            if (hypergraph.fixed[vertex] >= 0) {
                fallback[vertex] = hypergraph.fixed[vertex];
            } else {
                fallback[vertex] = vertex % num_parts;
            }
        }
        return local_refine_partition(hypergraph, fallback, num_parts, limits, rng);
    }
    return local_refine_partition(hypergraph, scored.front().partition, num_parts, limits, rng);
}

std::vector<int> partition_k_way_hypergraph(const Hypergraph& hypergraph,
                                            int num_parts,
                                            int imb,
                                            int eigvecs,
                                            int solver_iters,
                                            int cycles,
                                            int best_solns,
                                            const std::vector<int>& base_partition,
                                            std::mt19937& rng) {
    if (hypergraph.num_vertices == 0) {
        return {};
    }
    if (hypergraph.num_vertices == 1) {
        return {hypergraph.fixed[0] >= 0 ? hypergraph.fixed[0] : 0};
    }

    const BalanceLimits limits = compute_balance_limits(hypergraph, num_parts, imb);
    WeightedGraph graph = hypergraph_to_graph(hypergraph, cycles, rng);
    const int dims = std::max(1, std::min(hypergraph.num_vertices - 1, std::max(eigvecs, num_parts - 1)));
    Eigen::MatrixXd embedding = leading_eigenvectors(graph, dims, solver_iters, rng);
    auto candidates = generate_k_way_candidates(hypergraph, embedding, base_partition, num_parts, limits, best_solns, rng);
    auto scored = score_candidates(hypergraph, candidates, num_parts, limits);
    if (scored.empty()) {
        std::vector<int> fallback(hypergraph.num_vertices, 0);
        for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
            if (hypergraph.fixed[vertex] >= 0) {
                fallback[vertex] = hypergraph.fixed[vertex];
            } else {
                fallback[vertex] = vertex % num_parts;
            }
        }
        return local_refine_partition(hypergraph, fallback, num_parts, limits, rng);
    }
    return local_refine_partition(hypergraph, scored.front().partition, num_parts, limits, rng);
}

std::vector<int> project_partition(const std::vector<int>& clusters,
                                   const std::vector<int>& contracted_partition,
                                   int num_vertices) {
    std::vector<int> projected(num_vertices, 0);
    for (int vertex = 0; vertex < num_vertices; ++vertex) {
        projected[vertex] = contracted_partition[clusters[vertex]];
    }
    return projected;
}

std::vector<int> initial_partition_for_processed(const Hypergraph& hypergraph,
                                                 const SpecPartOptions& options,
                                                 const std::vector<int>& hint,
                                                 std::mt19937& rng) {
    if (partition_complete(hint, hypergraph.num_vertices)) {
        return hint;
    }
    if (options.num_parts == 2) {
        return partition_two_way_hypergraph(hypergraph,
                                            options.imb,
                                            options.eigvecs,
                                            options.solver_iters,
                                            options.ncycles,
                                            options.best_solns,
                                            {},
                                            rng);
    }
    return partition_k_way_hypergraph(hypergraph,
                                      options.num_parts,
                                      options.imb,
                                      options.eigvecs,
                                      options.solver_iters,
                                      options.ncycles,
                                      options.best_solns,
                                      {},
                                      rng);
}

std::vector<int> lift_partition_to_original(const Hypergraph& original_hypergraph,
                                            const IsolateResult& isolate,
                                            const std::vector<int>& processed_partition,
                                            int num_parts) {
    std::vector<int> partition(original_hypergraph.num_vertices, -1);
    std::vector<int> balance(num_parts, 0);

    for (int old_vertex = 0; old_vertex < original_hypergraph.num_vertices; ++old_vertex) {
        const int new_vertex = isolate.new_indices[old_vertex];
        if (new_vertex >= 0) {
            partition[old_vertex] = processed_partition[new_vertex];
            balance[partition[old_vertex]] += original_hypergraph.vwts[old_vertex];
        }
    }

    for (int component = 0; component < static_cast<int>(isolate.component_sizes.size()); ++component) {
        if (component == isolate.main_component) {
            continue;
        }

        std::vector<int> vertices;
        vertices.reserve(isolate.component_sizes[component]);
        std::unordered_map<int, int> fixed_weights;
        for (int vertex = 0; vertex < original_hypergraph.num_vertices; ++vertex) {
            if (isolate.component_labels[vertex] != component) {
                continue;
            }
            vertices.push_back(vertex);
            if (original_hypergraph.fixed[vertex] >= 0) {
                fixed_weights[original_hypergraph.fixed[vertex]] += original_hypergraph.vwts[vertex];
            }
        }

        int assigned_part = -1;
        if (fixed_weights.size() == 1) {
            assigned_part = fixed_weights.begin()->first;
        } else if (fixed_weights.empty()) {
            assigned_part = static_cast<int>(std::distance(balance.begin(), std::min_element(balance.begin(), balance.end())));
        } else {
            assigned_part = fixed_weights.begin()->first;
            int best_weight = fixed_weights.begin()->second;
            for (const auto& [part, weight] : fixed_weights) {
                if (weight > best_weight) {
                    assigned_part = part;
                    best_weight = weight;
                }
            }
        }

        for (int vertex : vertices) {
            if (original_hypergraph.fixed[vertex] >= 0) {
                partition[vertex] = original_hypergraph.fixed[vertex];
                balance[partition[vertex]] += original_hypergraph.vwts[vertex];
            } else {
                partition[vertex] = assigned_part;
                balance[assigned_part] += original_hypergraph.vwts[vertex];
            }
        }
    }

    for (int vertex = 0; vertex < original_hypergraph.num_vertices; ++vertex) {
        if (partition[vertex] < 0) {
            partition[vertex] = original_hypergraph.fixed[vertex] >= 0 ? original_hypergraph.fixed[vertex] : 0;
        }
    }
    return partition;
}

std::vector<int> remap_hint_to_processed(const std::vector<int>& hint, const IsolateResult& isolate) {
    std::vector<int> processed(isolate.hypergraph.num_vertices, -1);
    if (static_cast<int>(hint.size()) != static_cast<int>(isolate.new_indices.size())) {
        return {};
    }
    for (int old_vertex = 0; old_vertex < static_cast<int>(isolate.new_indices.size()); ++old_vertex) {
        const int new_vertex = isolate.new_indices[old_vertex];
        if (new_vertex >= 0) {
            processed[new_vertex] = hint[old_vertex];
        }
    }
    return processed;
}

void log_partition_metrics(const std::string& label, const PartitionResult& metrics) {
    std::cout << label << " cutsize=" << metrics.cutsize << " balance=[";
    for (std::size_t i = 0; i < metrics.balance.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << metrics.balance[i];
    }
    std::cout << "]\n";
}

ProcessedRefineResult two_way_spectral_refine(const Hypergraph& processed,
                                              std::vector<int> current,
                                              const SpecPartOptions& options,
                                              std::mt19937& rng) {
    PartitionResult current_metrics = evaluate_partition(processed, 2, current);
    log_partition_metrics("Initial two-way partition", current_metrics);

    const BalanceLimits limits = compute_balance_limits(processed, 2, options.imb);
    std::vector<std::vector<int>> global_partitions = {current};

    for (int iter = 0; iter < std::max(1, options.refine_iters); ++iter) {
        WeightedGraph graph = hypergraph_to_graph(processed, options.ncycles, rng);
        const int dims = std::max(1, std::min(processed.num_vertices - 1, std::max(1, options.eigvecs)));
        Eigen::MatrixXd embedding = leading_eigenvectors(graph, dims, options.solver_iters, rng);
        auto candidates = generate_two_way_candidates(processed,
                                                      embedding,
                                                      current,
                                                      limits,
                                                      options.best_solns,
                                                      rng);
        auto scored = score_candidates(processed, candidates, 2, limits);

        std::vector<std::vector<int>> overlay_inputs;
        for (int i = 0; i < std::min(options.best_solns, static_cast<int>(scored.size())); ++i) {
            overlay_inputs.push_back(scored[i].partition);
        }
        overlay_inputs.push_back(current);

        if (!overlay_inputs.empty()) {
            OverlayResult overlaid = overlay_partitions(overlay_inputs, processed);
            std::vector<int> contracted_partition = partition_two_way_hypergraph(overlaid.hypergraph,
                                                                                 options.imb,
                                                                                 options.eigvecs,
                                                                                 options.solver_iters,
                                                                                 options.ncycles,
                                                                                 options.best_solns,
                                                                                 {},
                                                                                 rng);
            std::vector<int> projected = project_partition(overlaid.clusters, contracted_partition, processed.num_vertices);
            projected = local_refine_partition(processed, projected, 2, limits, rng);
            candidates.push_back(projected);
            scored = score_candidates(processed, candidates, 2, limits);
        }

        if (!scored.empty()) {
            current = scored.front().partition;
            current_metrics = scored.front().metrics;
            global_partitions.push_back(current);
        }
        log_partition_metrics("Two-way refine iteration " + std::to_string(iter + 1), current_metrics);
    }

    if (global_partitions.size() > 1) {
        OverlayResult overlaid = overlay_partitions(global_partitions, processed);
        std::vector<int> contracted_partition = partition_two_way_hypergraph(overlaid.hypergraph,
                                                                             options.imb,
                                                                             options.eigvecs,
                                                                             options.solver_iters,
                                                                             options.ncycles,
                                                                             options.best_solns,
                                                                             {},
                                                                             rng);
        std::vector<int> projected = project_partition(overlaid.clusters, contracted_partition, processed.num_vertices);
        projected = local_refine_partition(processed, projected, 2, limits, rng);
        PartitionResult projected_metrics = evaluate_partition(processed, 2, projected);
        if (balance_penalty(projected_metrics.balance, limits) < balance_penalty(current_metrics.balance, limits) ||
            (balance_penalty(projected_metrics.balance, limits) == balance_penalty(current_metrics.balance, limits) &&
             projected_metrics.cutsize < current_metrics.cutsize)) {
            current = projected;
            current_metrics = projected_metrics;
        }
        log_partition_metrics("Final two-way overlay", current_metrics);
    }

    return {current, current_metrics};
}

ProcessedRefineResult k_way_spectral_refine(const Hypergraph& processed,
                                            std::vector<int> current,
                                            const SpecPartOptions& options,
                                            std::mt19937& rng) {
    PartitionResult current_metrics = evaluate_partition(processed, options.num_parts, current);
    log_partition_metrics("Initial k-way partition", current_metrics);

    const BalanceLimits limits = compute_balance_limits(processed, options.num_parts, options.imb);
    std::vector<std::vector<int>> global_partitions = {current};

    for (int iter = 0; iter < std::max(1, options.refine_iters); ++iter) {
        WeightedGraph graph = hypergraph_to_graph(processed, options.ncycles, rng);
        const int dims = std::max(1, std::min(processed.num_vertices - 1, std::max(options.eigvecs, options.num_parts - 1)));
        Eigen::MatrixXd embedding = leading_eigenvectors(graph, dims, options.solver_iters, rng);
        auto candidates = generate_k_way_candidates(processed,
                                                    embedding,
                                                    current,
                                                    options.num_parts,
                                                    limits,
                                                    options.best_solns,
                                                    rng);
        auto scored = score_candidates(processed, candidates, options.num_parts, limits);

        std::vector<std::vector<int>> overlay_inputs;
        for (int i = 0; i < std::min(options.best_solns, static_cast<int>(scored.size())); ++i) {
            overlay_inputs.push_back(scored[i].partition);
        }
        overlay_inputs.push_back(current);

        if (!overlay_inputs.empty()) {
            OverlayResult overlaid = overlay_partitions(overlay_inputs, processed);
            std::vector<int> contracted_partition = partition_k_way_hypergraph(overlaid.hypergraph,
                                                                               options.num_parts,
                                                                               options.imb,
                                                                               options.eigvecs,
                                                                               options.solver_iters,
                                                                               options.ncycles,
                                                                               options.best_solns,
                                                                               {},
                                                                               rng);
            std::vector<int> projected = project_partition(overlaid.clusters, contracted_partition, processed.num_vertices);
            projected = local_refine_partition(processed, projected, options.num_parts, limits, rng);
            candidates.push_back(projected);
            scored = score_candidates(processed, candidates, options.num_parts, limits);
        }

        if (!scored.empty()) {
            current = scored.front().partition;
            current_metrics = scored.front().metrics;
            global_partitions.push_back(current);
        }
        log_partition_metrics("K-way refine iteration " + std::to_string(iter + 1), current_metrics);
    }

    if (global_partitions.size() > 1) {
        OverlayResult overlaid = overlay_partitions(global_partitions, processed);
        std::vector<int> contracted_partition = partition_k_way_hypergraph(overlaid.hypergraph,
                                                                           options.num_parts,
                                                                           options.imb,
                                                                           options.eigvecs,
                                                                           options.solver_iters,
                                                                           options.ncycles,
                                                                           options.best_solns,
                                                                           {},
                                                                           rng);
        std::vector<int> projected = project_partition(overlaid.clusters, contracted_partition, processed.num_vertices);
        projected = local_refine_partition(processed, projected, options.num_parts, limits, rng);
        PartitionResult projected_metrics = evaluate_partition(processed, options.num_parts, projected);
        if (balance_penalty(projected_metrics.balance, limits) < balance_penalty(current_metrics.balance, limits) ||
            (balance_penalty(projected_metrics.balance, limits) == balance_penalty(current_metrics.balance, limits) &&
             projected_metrics.cutsize < current_metrics.cutsize)) {
            current = projected;
            current_metrics = projected_metrics;
        }
        log_partition_metrics("Final k-way overlay", current_metrics);
    }

    return {current, current_metrics};
}

}  // namespace

bool parse_arguments(int argc, char** argv, SpecPartOptions& options, std::string& error) {
    auto read_value = [&](int& index) -> std::string {
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + std::string(argv[index]));
        }
        ++index;
        return argv[index];
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                error.clear();
                return false;
            }
            if (arg == "--hypergraph" || arg == "--hypergraph-file") {
                options.hypergraph_file = read_value(i);
            } else if (arg == "--fixed-file") {
                options.fixed_file = read_value(i);
            } else if (arg == "--hint-file") {
                options.hint_file = read_value(i);
            } else if (arg == "--output") {
                options.output_file = read_value(i);
            } else if (arg == "--imb") {
                options.imb = std::stoi(read_value(i));
            } else if (arg == "--num-parts") {
                options.num_parts = std::stoi(read_value(i));
            } else if (arg == "--eigvecs") {
                options.eigvecs = std::stoi(read_value(i));
            } else if (arg == "--refine-iters") {
                options.refine_iters = std::stoi(read_value(i));
            } else if (arg == "--solver-iters") {
                options.solver_iters = std::stoi(read_value(i));
            } else if (arg == "--best-solns") {
                options.best_solns = std::stoi(read_value(i));
            } else if (arg == "--ncycles") {
                options.ncycles = std::stoi(read_value(i));
            } else if (arg == "--seed") {
                options.seed = std::stoi(read_value(i));
            } else if (!arg.empty() && arg[0] != '-' && options.hypergraph_file.empty()) {
                options.hypergraph_file = arg;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }
    } catch (const std::exception& ex) {
        error = ex.what();
        return false;
    }

    if (options.hypergraph_file.empty()) {
        error = "missing required --hypergraph <file>";
        return false;
    }
    if (options.num_parts < 2) {
        error = "--num-parts must be at least 2";
        return false;
    }
    return true;
}

PartitionResult specpart_run(const SpecPartOptions& options) {
    std::mt19937 rng(options.seed);
    Hypergraph original = read_hypergraph_file(options.hypergraph_file, options.fixed_file);
    std::cout << "============================================================\n";
    std::cout << "K-SpecPart C++\n";
    std::cout << "============================================================\n";
    std::cout << "Input hypergraph: vertices=" << original.num_vertices
              << " hyperedges=" << original.num_hyperedges << '\n';

    IsolateResult isolated = isolate_islands(original);
    Hypergraph processed = remove_single_hyperedges(isolated.hypergraph);
    std::cout << "Processed hypergraph: vertices=" << processed.num_vertices
              << " hyperedges=" << processed.num_hyperedges << '\n';

    std::vector<int> hint;
    if (!options.hint_file.empty()) {
        hint = read_partition_file(options.hint_file);
    }
    std::vector<int> processed_hint = remap_hint_to_processed(hint, isolated);
    std::vector<int> current = initial_partition_for_processed(processed, options, processed_hint, rng);

    ProcessedRefineResult refined = options.num_parts == 2
        ? two_way_spectral_refine(processed, current, options, rng)
        : k_way_spectral_refine(processed, current, options, rng);

    std::vector<int> full_partition = lift_partition_to_original(original, isolated, refined.partition, options.num_parts);
    const BalanceLimits original_limits = compute_balance_limits(original, options.num_parts, options.imb);
    full_partition = local_refine_partition(original, full_partition, options.num_parts, original_limits, rng);
    PartitionResult final_metrics = evaluate_partition(original, options.num_parts, full_partition);
    log_partition_metrics("Final original partition", final_metrics);

    write_partition_file(options.output_file, full_partition);
    return final_metrics;
}

}  // namespace kspecpart
