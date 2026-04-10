#include "kspecpart/exact_partitioner.hpp"

#include "kspecpart/tree_partition.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

namespace kspecpart {

namespace {

struct SearchState {
    const Hypergraph& hypergraph;
    const ExactPartitionerOptions& options;
    BalanceLimits limits;
    std::vector<int> vertex_order;
    std::vector<int> assignment;
    std::vector<int> block_weights;
    std::vector<int> edge_part_counts;
    std::vector<int> edge_active_parts;
    std::vector<int> best_partition;
    int total_weight = 0;
    int assigned_weight = 0;
    int current_cut = 0;
    int best_cut = std::numeric_limits<int>::max();
    std::size_t nodes = 0;
    bool aborted = false;
    bool use_symmetry_break = true;

    SearchState(const Hypergraph& input_hypergraph,
                const ExactPartitionerOptions& input_options,
                BalanceLimits input_limits,
                std::vector<int> input_order,
                bool input_use_symmetry_break)
        : hypergraph(input_hypergraph),
          options(input_options),
          limits(input_limits),
          vertex_order(std::move(input_order)),
          assignment(hypergraph.num_vertices, -1),
          block_weights(options.num_parts, 0),
          edge_part_counts(static_cast<std::size_t>(hypergraph.num_hyperedges) * options.num_parts, 0),
          edge_active_parts(hypergraph.num_hyperedges, 0),
          total_weight(std::accumulate(hypergraph.vwts.begin(), hypergraph.vwts.end(), 0)),
          use_symmetry_break(input_use_symmetry_break) {}

    int& edge_count(int edge, int part) {
        return edge_part_counts[static_cast<std::size_t>(edge) * options.num_parts + part];
    }

    const int& edge_count(int edge, int part) const {
        return edge_part_counts[static_cast<std::size_t>(edge) * options.num_parts + part];
    }
};

bool lexicographically_better(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    if (rhs.empty()) {
        return true;
    }
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

bool has_fixed_vertices(const Hypergraph& hypergraph) {
    return std::any_of(hypergraph.fixed.begin(), hypergraph.fixed.end(), [](int part) { return part >= 0; });
}

int exact_vertex_limit(const Hypergraph& hypergraph, const ExactPartitionerOptions& options) {
    if (options.num_parts <= 2) {
        return 26;
    }
    if (options.num_parts <= 4) {
        if (hypergraph.num_hyperedges <= 16) {
            return 23;
        }
        return hypergraph.num_hyperedges <= 32 ? 20 : 17;
    }
    if (options.num_parts <= 8) {
        return 14;
    }
    return has_fixed_vertices(hypergraph) ? 8 : 10;
}

std::vector<int> build_vertex_order(const Hypergraph& hypergraph) {
    std::vector<int> weighted_degree(hypergraph.num_vertices, 0);
    for (int vertex = 0; vertex < hypergraph.num_vertices; ++vertex) {
        int degree = 0;
        for (int idx = hypergraph.vptr[vertex]; idx < hypergraph.vptr[vertex + 1]; ++idx) {
            degree += hypergraph.hwts[hypergraph.vind[idx]];
        }
        weighted_degree[vertex] = degree;
    }

    std::vector<int> order(hypergraph.num_vertices);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        const bool lhs_fixed = hypergraph.fixed[lhs] >= 0;
        const bool rhs_fixed = hypergraph.fixed[rhs] >= 0;
        if (lhs_fixed != rhs_fixed) {
            return lhs_fixed;
        }
        if (weighted_degree[lhs] != weighted_degree[rhs]) {
            return weighted_degree[lhs] > weighted_degree[rhs];
        }
        if (hypergraph.vwts[lhs] != hypergraph.vwts[rhs]) {
            return hypergraph.vwts[lhs] > hypergraph.vwts[rhs];
        }
        return lhs < rhs;
    });
    return order;
}

bool remaining_balance_feasible(const SearchState& state) {
    long long deficit = 0;
    const int remaining_weight = state.total_weight - state.assigned_weight;
    for (int part = 0; part < state.options.num_parts; ++part) {
        if (state.block_weights[part] > state.limits.max_capacity) {
            return false;
        }
        if (state.block_weights[part] < state.limits.min_capacity) {
            deficit += static_cast<long long>(state.limits.min_capacity - state.block_weights[part]);
        }
    }
    return deficit <= remaining_weight;
}

int cut_delta_for_choice(const SearchState& state, int vertex, int part) {
    int delta = 0;
    for (int idx = state.hypergraph.vptr[vertex]; idx < state.hypergraph.vptr[vertex + 1]; ++idx) {
        const int edge = state.hypergraph.vind[idx];
        if (state.edge_active_parts[edge] == 1 && state.edge_count(edge, part) == 0) {
            delta += state.hypergraph.hwts[edge];
        }
    }
    return delta;
}

std::vector<int> candidate_parts(const SearchState& state, int vertex) {
    if (state.hypergraph.fixed[vertex] >= 0) {
        return {state.hypergraph.fixed[vertex]};
    }

    std::vector<int> candidates;
    if (state.use_symmetry_break) {
        int first_empty = -1;
        for (int part = 0; part < state.options.num_parts; ++part) {
            if (state.block_weights[part] > 0) {
                candidates.push_back(part);
            } else if (first_empty < 0) {
                first_empty = part;
            }
        }
        if (first_empty >= 0) {
            candidates.push_back(first_empty);
        }
    } else {
        candidates.resize(state.options.num_parts);
        std::iota(candidates.begin(), candidates.end(), 0);
    }

    std::stable_sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
        const int lhs_delta = cut_delta_for_choice(state, vertex, lhs);
        const int rhs_delta = cut_delta_for_choice(state, vertex, rhs);
        if (lhs_delta != rhs_delta) {
            return lhs_delta < rhs_delta;
        }
        if (state.block_weights[lhs] != state.block_weights[rhs]) {
            return state.block_weights[lhs] < state.block_weights[rhs];
        }
        return lhs < rhs;
    });
    return candidates;
}

void assign_vertex(SearchState& state, int vertex, int part) {
    state.assignment[vertex] = part;
    state.block_weights[part] += state.hypergraph.vwts[vertex];
    state.assigned_weight += state.hypergraph.vwts[vertex];

    for (int idx = state.hypergraph.vptr[vertex]; idx < state.hypergraph.vptr[vertex + 1]; ++idx) {
        const int edge = state.hypergraph.vind[idx];
        int& count = state.edge_count(edge, part);
        if (count == 0) {
            if (state.edge_active_parts[edge] == 1) {
                state.current_cut += state.hypergraph.hwts[edge];
            }
            state.edge_active_parts[edge] += 1;
        }
        count += 1;
    }
}

void unassign_vertex(SearchState& state, int vertex, int part) {
    for (int idx = state.hypergraph.vptr[vertex]; idx < state.hypergraph.vptr[vertex + 1]; ++idx) {
        const int edge = state.hypergraph.vind[idx];
        int& count = state.edge_count(edge, part);
        count -= 1;
        if (count == 0) {
            if (state.edge_active_parts[edge] == 2) {
                state.current_cut -= state.hypergraph.hwts[edge];
            }
            state.edge_active_parts[edge] -= 1;
        }
    }

    state.assigned_weight -= state.hypergraph.vwts[vertex];
    state.block_weights[part] -= state.hypergraph.vwts[vertex];
    state.assignment[vertex] = -1;
}

void search_exact_partitions(SearchState& state, int depth) {
    if (state.aborted) {
        return;
    }
    state.nodes += 1;
    if (state.nodes > state.options.max_search_nodes) {
        state.aborted = true;
        return;
    }
    if (state.current_cut > state.best_cut) {
        return;
    }
    if (!remaining_balance_feasible(state)) {
        return;
    }

    if (depth == state.hypergraph.num_vertices) {
        for (int part = 0; part < state.options.num_parts; ++part) {
            if (state.block_weights[part] < state.limits.min_capacity ||
                state.block_weights[part] > state.limits.max_capacity) {
                return;
            }
        }
        if (state.current_cut < state.best_cut ||
            (state.current_cut == state.best_cut &&
             lexicographically_better(state.assignment, state.best_partition))) {
            state.best_cut = state.current_cut;
            state.best_partition = state.assignment;
        }
        return;
    }

    const int vertex = state.vertex_order[depth];
    for (int part : candidate_parts(state, vertex)) {
        if (part < 0 || part >= state.options.num_parts) {
            continue;
        }
        if (state.block_weights[part] + state.hypergraph.vwts[vertex] > state.limits.max_capacity) {
            continue;
        }
        assign_vertex(state, vertex, part);
        search_exact_partitions(state, depth + 1);
        unassign_vertex(state, vertex, part);
        if (state.aborted) {
            return;
        }
    }
}

}  // namespace

bool should_try_exact_partitioner(const Hypergraph& hypergraph,
                                  const ExactPartitionerOptions& options) {
    if (hypergraph.num_vertices <= 0 || options.num_parts <= 0) {
        return false;
    }
    if (hypergraph.num_vertices > exact_vertex_limit(hypergraph, options)) {
        return false;
    }
    if (hypergraph.num_hyperedges > 512) {
        return false;
    }
    if (has_fixed_vertices(hypergraph) && options.num_parts > 12) {
        return false;
    }
    return true;
}

std::optional<std::vector<int>> run_exact_partitioner(const Hypergraph& hypergraph,
                                                      const ExactPartitionerOptions& options) {
    if (!should_try_exact_partitioner(hypergraph, options)) {
        return std::nullopt;
    }

    const BalanceLimits limits = compute_balance_limits(hypergraph, options.num_parts, options.imb);
    SearchState state(hypergraph,
                      options,
                      limits,
                      build_vertex_order(hypergraph),
                      !has_fixed_vertices(hypergraph));
    search_exact_partitions(state, 0);
    if (state.aborted || state.best_partition.empty()) {
        return std::nullopt;
    }
    return state.best_partition;
}

}  // namespace kspecpart
