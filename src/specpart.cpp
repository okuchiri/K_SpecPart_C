#include "kspecpart/specpart.hpp"

#include "kspecpart/cut_distillation.hpp"
#include "kspecpart/embedding.hpp"
#include "kspecpart/golden_evaluator.hpp"
#include "kspecpart/graphification.hpp"
#include "kspecpart/hypergraph.hpp"
#include "kspecpart/io.hpp"
#include "kspecpart/isolate_islands.hpp"
#include "kspecpart/optimal_partitioner.hpp"
#include "kspecpart/overlay.hpp"
#include "kspecpart/projection.hpp"
#include "kspecpart/triton_refiner.hpp"
#include "kspecpart/tree_partition.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace kspecpart {

namespace {

struct ProcessedRefineResult {
    std::vector<int> partition;
    PartitionResult metrics;
};

struct RefinedCandidate {
    std::vector<int> partition;
    PartitionResult metrics;
};

struct ExternalRuntimeContext {
    std::string work_prefix;
    std::optional<std::string> processed_hypergraph_file;
    std::optional<std::string> resolved_hmetis;
    std::optional<std::string> resolved_ilp_partitioner;
    std::optional<std::string> resolved_triton_refiner;

    ~ExternalRuntimeContext() {
        if (!processed_hypergraph_file.has_value()) {
            return;
        }
        std::error_code ec;
        std::filesystem::remove(*processed_hypergraph_file, ec);
    }
};

TreePartitionOptions make_tree_partition_options(const SpecPartOptions& options, int num_parts) {
    TreePartitionOptions tree_options;
    tree_options.num_parts = num_parts;
    tree_options.imb = options.imb;
    tree_options.eigvecs = options.eigvecs;
    tree_options.solver_iters = options.solver_iters;
    tree_options.cycles = options.ncycles;
    tree_options.best_solns = options.best_solns;
    tree_options.seed = options.seed;
    tree_options.projection_strategy = options.projection_strategy;
    tree_options.gpmetis_executable = options.gpmetis_executable;
    tree_options.enable_metis = options.enable_metis;
    tree_options.gpmetis_explicit = options.gpmetis_explicit;
    return tree_options;
}

OptimalPartitionerOptions make_optimal_partitioner_options(const SpecPartOptions& options, int num_parts) {
    OptimalPartitionerOptions external_options;
    external_options.num_parts = num_parts;
    external_options.imb = options.imb;
    external_options.seed = options.seed;
    external_options.hmetis_executable = options.hmetis_executable;
    external_options.ilp_partitioner_executable = options.ilp_partitioner_executable;
    external_options.enable_hmetis =
        options.enable_optimal_partitioner && options.enable_hmetis_partitioner;
    external_options.enable_ilp =
        options.enable_optimal_partitioner && options.enable_ilp_partitioner;
    external_options.hmetis_explicit = options.hmetis_explicit;
    external_options.ilp_partitioner_explicit = options.ilp_partitioner_explicit;
    return external_options;
}

TritonRefinerOptions make_triton_refiner_options(const SpecPartOptions& options, int num_parts) {
    TritonRefinerOptions external_options;
    external_options.num_parts = num_parts;
    external_options.imb = options.imb;
    external_options.seed = options.seed;
    external_options.refiner_executable = options.triton_refiner_executable;
    external_options.explicit_path = options.triton_refiner_explicit;
    return external_options;
}

std::string make_temp_prefix(const std::string& label, int seed) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path prefix =
        std::filesystem::temp_directory_path() /
        ("kspecpart-" + label + "-" + std::to_string(seed) + "-" + std::to_string(now));
    return prefix.string();
}

PartitionIndex empty_partition_index() {
    return {};
}

PartitionIndex binary_partition_index(const std::vector<int>& partition) {
    PartitionIndex pindex;
    for (int vertex = 0; vertex < static_cast<int>(partition.size()); ++vertex) {
        if (partition[vertex] == 0) {
            pindex.p1.push_back(vertex);
        } else if (partition[vertex] == 1) {
            pindex.p2.push_back(vertex);
        }
    }
    return pindex;
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

Eigen::MatrixXd two_way_iteration_embedding(const Hypergraph& hypergraph,
                                            const WeightedGraph& graph,
                                            const std::vector<int>& partition,
                                            const SpecPartOptions& options,
                                            const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd()) {
    const PartitionIndex pindex = binary_partition_index(partition);
    const int dims = std::max(1, options.eigvecs);
    return solve_eigs(hypergraph, graph, pindex, false, dims, options.solver_iters, 1,
                      options.seed, constraint_basis);
}

Eigen::MatrixXd k_way_iteration_embedding(const Hypergraph& hypergraph,
                                          const WeightedGraph& graph,
                                          const std::vector<int>& partition,
                                          const SpecPartOptions& options,
                                          const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd()) {
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

std::vector<int> initial_partition_for_processed(const Hypergraph& hypergraph,
                                                 const SpecPartOptions& options,
                                                 const std::vector<int>& hint,
                                                 std::mt19937& rng) {
    if (static_cast<int>(hint.size()) == hypergraph.num_vertices &&
        std::all_of(hint.begin(), hint.end(), [](int part) { return part >= 0; })) {
        return hint;
    }
    return tree_partition_best(hypergraph, make_tree_partition_options(options, options.num_parts), {}, rng);
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

std::vector<int> contract_partition_to_overlay(const std::vector<int>& partition,
                                               const std::vector<int>& clusters,
                                               int num_clusters) {
    std::vector<int> contracted(num_clusters, -1);
    for (int vertex = 0; vertex < static_cast<int>(clusters.size()) &&
                         vertex < static_cast<int>(partition.size());
         ++vertex) {
        const int cluster = clusters[vertex];
        if (cluster < 0 || cluster >= num_clusters) {
            continue;
        }
        if (contracted[cluster] < 0) {
            contracted[cluster] = partition[vertex];
        }
    }
    for (int& part : contracted) {
        if (part < 0) {
            part = 0;
        }
    }
    return contracted;
}

bool better_partition(const PartitionResult& candidate,
                      const PartitionResult& current,
                      const BalanceLimits& limits) {
    const long long candidate_penalty = balance_penalty(candidate.balance, limits);
    const long long current_penalty = balance_penalty(current.balance, limits);
    return candidate_penalty < current_penalty ||
           (candidate_penalty == current_penalty && candidate.cutsize < current.cutsize);
}

std::vector<int> refine_partition_with_external_tools(const Hypergraph& hypergraph,
                                                      const std::vector<int>& partition,
                                                      int num_parts,
                                                      const BalanceLimits& limits,
                                                      const SpecPartOptions& options,
                                                      const ExternalRuntimeContext& runtime,
                                                      int run_id,
                                                      std::mt19937& rng);

std::optional<OptimalPartitionerResult> partition_overlay_with_optimal_partitioner(
    const Hypergraph& hypergraph,
    const TreePartitionOptions& tree_options,
    const SpecPartOptions& options,
    const ExternalRuntimeContext& runtime,
    int run_id);

std::optional<ProcessedRefineResult> realize_overlay_partition(const Hypergraph& hypergraph,
                                                               const OverlayResult& overlaid,
                                                               const std::vector<int>& contracted_partition,
                                                               int num_parts,
                                                               const BalanceLimits& limits,
                                                               const SpecPartOptions& options,
                                                               const ExternalRuntimeContext& runtime,
                                                               int run_id,
                                                               std::mt19937& rng);

void log_partition_metrics(const std::string& label, const PartitionResult& metrics);

bool valid_partition_labels(const std::vector<int>& partition, int num_parts, int expected_vertices) {
    return static_cast<int>(partition.size()) == expected_vertices &&
           std::all_of(partition.begin(),
                       partition.end(),
                       [&](int part) { return part >= 0 && part < num_parts; });
}

std::vector<RefinedCandidate> refine_tree_candidates(const Hypergraph& hypergraph,
                                                     const std::vector<TreePartitionCandidate>& candidates,
                                                     int num_parts,
                                                     const BalanceLimits& limits,
                                                     const SpecPartOptions& options,
                                                     const ExternalRuntimeContext& runtime,
                                                     int run_id_base,
                                                     std::mt19937& rng) {
    std::vector<RefinedCandidate> unique_candidates;
    std::unordered_map<int, int> seen_cutsize;

    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        std::vector<int> refined = refine_partition_with_external_tools(
            hypergraph, candidates[i].partition, num_parts, limits, options, runtime, run_id_base + i, rng);
        if (!valid_partition_labels(refined, num_parts, hypergraph.num_vertices)) {
            continue;
        }

        PartitionResult metrics = evaluate_partition(hypergraph, num_parts, refined);
        if (!seen_cutsize.emplace(metrics.cutsize, i).second) {
            continue;
        }
        unique_candidates.push_back({std::move(refined), std::move(metrics)});
    }

    std::sort(unique_candidates.begin(),
              unique_candidates.end(),
              [](const RefinedCandidate& lhs, const RefinedCandidate& rhs) {
                  return lhs.metrics.cutsize < rhs.metrics.cutsize;
              });

    return unique_candidates;
}

std::vector<std::vector<int>> select_overlay_inputs(const std::vector<RefinedCandidate>& candidates,
                                                    int best_solns,
                                                    const std::vector<int>& anchor_partition) {
    std::vector<std::vector<int>> overlay_inputs;
    const int limit =
        best_solns > 0 ? std::min(best_solns, static_cast<int>(candidates.size()))
                       : static_cast<int>(candidates.size());
    overlay_inputs.reserve(limit + (anchor_partition.empty() ? 0 : 1));

    for (int i = 0; i < limit; ++i) {
        overlay_inputs.push_back(candidates[i].partition);
    }
    if (!anchor_partition.empty()) {
        overlay_inputs.push_back(anchor_partition);
    }
    return overlay_inputs;
}

std::optional<ProcessedRefineResult> run_overlay_round(const Hypergraph& hypergraph,
                                                       const std::vector<std::vector<int>>& partitions,
                                                       const TreePartitionOptions& tree_options,
                                                       const std::vector<int>& overlay_base_partition,
                                                       int num_parts,
                                                       const BalanceLimits& limits,
                                                       const SpecPartOptions& options,
                                                       const ExternalRuntimeContext& runtime,
                                                       int run_id,
                                                       std::mt19937& rng) {
    if (partitions.empty()) {
        return std::nullopt;
    }

    OverlayResult overlaid = overlay_partitions(partitions, hypergraph);
    std::vector<int> contracted_base;
    if (!overlay_base_partition.empty()) {
        contracted_base = contract_partition_to_overlay(
            overlay_base_partition, overlaid.clusters, overlaid.hypergraph.num_vertices);
    }

    std::optional<ProcessedRefineResult> best_result;

    std::mt19937 internal_rng = rng;
    std::vector<int> internal_partition =
        tree_partition_best(overlaid.hypergraph, tree_options, contracted_base, internal_rng);
    if (valid_partition_labels(internal_partition,
                               num_parts,
                               overlaid.hypergraph.num_vertices)) {
        best_result = realize_overlay_partition(hypergraph,
                                                overlaid,
                                                internal_partition,
                                                num_parts,
                                                limits,
                                                options,
                                                runtime,
                                                run_id,
                                                internal_rng);
    }

    std::optional<OptimalPartitionerResult> external =
        partition_overlay_with_optimal_partitioner(overlaid.hypergraph,
                                                   tree_options,
                                                   options,
                                                   runtime,
                                                   run_id);
    if (external.has_value() &&
        valid_partition_labels(external->partition,
                               num_parts,
                               overlaid.hypergraph.num_vertices)) {
        std::mt19937 external_rng = rng;
        std::optional<ProcessedRefineResult> external_result =
            realize_overlay_partition(hypergraph,
                                      overlaid,
                                      external->partition,
                                      num_parts,
                                      limits,
                                      options,
                                      runtime,
                                      run_id,
                                      external_rng);
        if (external_result.has_value()) {
            if (!best_result.has_value() ||
                better_partition(external_result->metrics, best_result->metrics, limits)) {
                std::cout << "Overlay external partitioner: " << external->method << '\n';
                best_result = std::move(external_result);
            } else {
                std::cout << "Overlay external partitioner: " << external->method
                          << " (rejected, internal overlay better)\n";
            }
        }
    }

    return best_result;
}

std::vector<int> refine_partition_with_external_tools(const Hypergraph& hypergraph,
                                                      const std::vector<int>& partition,
                                                      int num_parts,
                                                      const BalanceLimits& limits,
                                                      const SpecPartOptions& options,
                                                      const ExternalRuntimeContext& runtime,
                                                      int run_id,
                                                      std::mt19937& rng) {
    if (options.enable_triton_refiner && runtime.processed_hypergraph_file.has_value()) {
        const std::optional<std::vector<int>> refined = run_triton_refiner(
            *runtime.processed_hypergraph_file,
            partition,
            make_triton_refiner_options(options, num_parts),
            runtime.work_prefix,
            run_id);
        if (refined.has_value() && static_cast<int>(refined->size()) == hypergraph.num_vertices) {
            return *refined;
        }
    }
    return local_refine_partition(hypergraph, partition, num_parts, limits, rng);
}

std::optional<OptimalPartitionerResult> partition_overlay_with_optimal_partitioner(
    const Hypergraph& hypergraph,
    const TreePartitionOptions& tree_options,
    const SpecPartOptions& options,
    const ExternalRuntimeContext& runtime,
    int run_id) {
    if (!options.enable_optimal_partitioner) {
        return std::nullopt;
    }
    return run_optimal_partitioner(hypergraph,
                                   make_optimal_partitioner_options(options, tree_options.num_parts),
                                   runtime.work_prefix + ".overlay." + std::to_string(run_id));
}

std::optional<ProcessedRefineResult> realize_overlay_partition(const Hypergraph& hypergraph,
                                                               const OverlayResult& overlaid,
                                                               const std::vector<int>& contracted_partition,
                                                               int num_parts,
                                                               const BalanceLimits& limits,
                                                               const SpecPartOptions& options,
                                                               const ExternalRuntimeContext& runtime,
                                                               int run_id,
                                                               std::mt19937& rng) {
    std::vector<int> projected =
        project_partition(overlaid.clusters, contracted_partition, hypergraph.num_vertices);
    if (!valid_partition_labels(projected, num_parts, hypergraph.num_vertices)) {
        return std::nullopt;
    }

    projected = refine_partition_with_external_tools(
        hypergraph, projected, num_parts, limits, options, runtime, run_id + 10000, rng);
    if (!valid_partition_labels(projected, num_parts, hypergraph.num_vertices)) {
        return std::nullopt;
    }

    PartitionResult metrics = evaluate_partition(hypergraph, num_parts, projected);
    return ProcessedRefineResult{std::move(projected), std::move(metrics)};
}

ExternalRuntimeContext make_external_runtime_context(const Hypergraph& processed,
                                                    const SpecPartOptions& options) {
    ExternalRuntimeContext runtime;
    runtime.work_prefix = make_temp_prefix("specpart", options.seed);
    if (options.enable_optimal_partitioner) {
        if (options.enable_hmetis_partitioner) {
            runtime.resolved_hmetis = resolve_hmetis_executable(options.hmetis_executable);
        }
        if (options.enable_ilp_partitioner) {
            runtime.resolved_ilp_partitioner =
                resolve_ilp_partitioner_executable(options.ilp_partitioner_executable);
        }
        if (options.enable_hmetis_partitioner &&
            options.hmetis_explicit &&
            !runtime.resolved_hmetis.has_value()) {
            throw std::runtime_error("failed to resolve hmetis executable: " +
                                     options.hmetis_executable);
        }
        if (options.enable_ilp_partitioner &&
            options.ilp_partitioner_explicit &&
            !runtime.resolved_ilp_partitioner.has_value()) {
            throw std::runtime_error("failed to resolve ilp partitioner executable: " +
                                     options.ilp_partitioner_executable);
        }
    }
    if (options.enable_triton_refiner) {
        runtime.resolved_triton_refiner =
            resolve_triton_refiner_executable(options.triton_refiner_executable);
        if (options.triton_refiner_explicit && !runtime.resolved_triton_refiner.has_value()) {
            throw std::runtime_error("failed to resolve triton refiner executable: " +
                                     options.triton_refiner_executable);
        }
        if (runtime.resolved_triton_refiner.has_value()) {
            const std::string processed_file = runtime.work_prefix + ".processed.hgr";
            write_hypergraph_file(processed_file, processed);
            runtime.processed_hypergraph_file = processed_file;
        }
    }
    return runtime;
}

void log_external_runtime_context(const SpecPartOptions& options, const ExternalRuntimeContext& runtime) {
    std::cout << "External tools:\n";
    if (!options.enable_optimal_partitioner) {
        std::cout << "  Overlay partitioner: disabled\n";
    } else {
        if (!options.enable_ilp_partitioner) {
            std::cout << "  ILP partitioner: disabled\n";
        } else if (runtime.resolved_ilp_partitioner.has_value()) {
            std::cout << "  ILP partitioner: " << *runtime.resolved_ilp_partitioner << '\n';
        } else {
            std::cout << "  ILP partitioner: not found, overlay falls back to internal tree partition\n";
        }

        if (!options.enable_hmetis_partitioner) {
            std::cout << "  hMETIS: disabled\n";
        } else if (runtime.resolved_hmetis.has_value()) {
            std::cout << "  hMETIS: " << *runtime.resolved_hmetis << '\n';
        } else {
            std::cout << "  hMETIS: not found, overlay falls back to internal tree partition\n";
        }
    }

    if (!options.enable_triton_refiner) {
        std::cout << "  Triton/OpenROAD refine: disabled\n";
    } else if (runtime.resolved_triton_refiner.has_value()) {
        std::cout << "  Triton/OpenROAD refine: " << *runtime.resolved_triton_refiner << '\n';
    } else {
        std::cout << "  Triton/OpenROAD refine: not found, using internal local_refine fallback\n";
    }
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
                                              const ExternalRuntimeContext& runtime,
                                              std::mt19937& rng) {
    PartitionResult current_metrics = evaluate_partition(processed, 2, current);
    log_partition_metrics("Initial two-way partition", current_metrics);

    const BalanceLimits limits = compute_balance_limits(processed, 2, options.imb);
    const TreePartitionOptions tree_options = make_tree_partition_options(options, 2);
    TreePartitionOptions candidate_options = tree_options;
    candidate_options.best_solns = 0;
    const WeightedGraph graph = hypergraph_to_graph(processed, options.ncycles, rng);
    const PartitionIndex tree_fixed_vertices = empty_partition_index();
    const std::vector<int> anchor_partition = current;
    ProcessedRefineResult best_result{current, current_metrics};
    std::vector<std::vector<int>> global_partitions;

    for (int iter = 0; iter < std::max(1, options.refine_iters); ++iter) {
        const Eigen::MatrixXd embedding = two_way_iteration_embedding(processed, graph, current, options);
        std::vector<TreePartitionCandidate> tree_candidates = tree_partition_with_embedding(processed,
                                                                                            graph,
                                                                                            embedding,
                                                                                            tree_fixed_vertices,
                                                                                            candidate_options,
                                                                                            {},
                                                                                            rng);
        std::vector<RefinedCandidate> refined_candidates = refine_tree_candidates(
            processed, tree_candidates, 2, limits, options, runtime, (iter + 1) * 100, rng);
        for (int i = 0; i < static_cast<int>(refined_candidates.size()); ++i) {
            log_partition_metrics("Two-way refined tree candidate " + std::to_string(i + 1),
                                  refined_candidates[i].metrics);
        }

        if (!refined_candidates.empty()) {
            std::vector<std::vector<int>> overlay_inputs =
                select_overlay_inputs(refined_candidates, options.best_solns, anchor_partition);
            std::optional<ProcessedRefineResult> overlaid =
                run_overlay_round(processed,
                                  overlay_inputs,
                                  tree_options,
                                  {},
                                  2,
                                  limits,
                                  options,
                                  runtime,
                                  1000 + iter,
                                  rng);
            if (overlaid.has_value()) {
                current = overlaid->partition;
                current_metrics = overlaid->metrics;
            } else {
                current = refined_candidates.front().partition;
                current_metrics = refined_candidates.front().metrics;
            }

            global_partitions.push_back(current);
            if (better_partition(current_metrics, best_result.metrics, limits)) {
                best_result = {current, current_metrics};
            }
        }

        log_partition_metrics("Two-way refine iteration " + std::to_string(iter + 1), current_metrics);
    }

    if (!global_partitions.empty()) {
        std::vector<std::vector<int>> overlay_inputs = global_partitions;
        overlay_inputs.push_back(anchor_partition);
        std::optional<ProcessedRefineResult> final_overlay = run_overlay_round(processed,
                                                                               overlay_inputs,
                                                                               tree_options,
                                                                               {},
                                                                               2,
                                                                               limits,
                                                                               options,
                                                                               runtime,
                                                                               5000,
                                                                               rng);
        if (final_overlay.has_value()) {
            log_partition_metrics("Final two-way overlay", final_overlay->metrics);
            if (better_partition(final_overlay->metrics, best_result.metrics, limits)) {
                best_result = *final_overlay;
            }
        }
    }

    return best_result;
}

ProcessedRefineResult k_way_spectral_refine(const Hypergraph& processed,
                                            std::vector<int> current,
                                            const SpecPartOptions& options,
                                            const ExternalRuntimeContext& runtime,
                                            std::mt19937& rng) {
    PartitionResult current_metrics = evaluate_partition(processed, options.num_parts, current);
    log_partition_metrics("Initial k-way partition", current_metrics);

    const BalanceLimits limits = compute_balance_limits(processed, options.num_parts, options.imb);
    const TreePartitionOptions tree_options = make_tree_partition_options(options, options.num_parts);
    TreePartitionOptions candidate_options = tree_options;
    candidate_options.best_solns = 0;
    const WeightedGraph graph = hypergraph_to_graph(processed, options.ncycles, rng);
    const PartitionIndex tree_fixed_vertices = empty_partition_index();
    const std::vector<int> anchor_partition = current;
    ProcessedRefineResult best_result{current, current_metrics};
    std::vector<std::vector<int>> global_partitions;

    for (int iter = 0; iter < std::max(1, options.refine_iters); ++iter) {
        const Eigen::MatrixXd embedding = k_way_iteration_embedding(processed, graph, current, options);
        std::vector<TreePartitionCandidate> tree_candidates = tree_partition_with_embedding(processed,
                                                                                            graph,
                                                                                            embedding,
                                                                                            tree_fixed_vertices,
                                                                                            candidate_options,
                                                                                            {},
                                                                                            rng);
        std::vector<RefinedCandidate> refined_candidates =
            refine_tree_candidates(processed,
                                   tree_candidates,
                                   options.num_parts,
                                   limits,
                                   options,
                                   runtime,
                                   (iter + 1) * 100,
                                   rng);
        for (int i = 0; i < static_cast<int>(refined_candidates.size()); ++i) {
            log_partition_metrics("K-way refined tree candidate " + std::to_string(i + 1),
                                  refined_candidates[i].metrics);
        }

        if (!refined_candidates.empty()) {
            std::vector<std::vector<int>> overlay_inputs =
                select_overlay_inputs(refined_candidates, options.best_solns, anchor_partition);
            std::optional<ProcessedRefineResult> overlaid =
                run_overlay_round(processed,
                                  overlay_inputs,
                                  tree_options,
                                  current,
                                  options.num_parts,
                                  limits,
                                  options,
                                  runtime,
                                  1000 + iter,
                                  rng);
            if (overlaid.has_value()) {
                current = overlaid->partition;
                current_metrics = overlaid->metrics;
            } else {
                current = refined_candidates.front().partition;
                current_metrics = refined_candidates.front().metrics;
            }

            global_partitions.push_back(current);
            if (better_partition(current_metrics, best_result.metrics, limits)) {
                best_result = {current, current_metrics};
            }
        }

        log_partition_metrics("K-way refine iteration " + std::to_string(iter + 1), current_metrics);
    }

    if (!global_partitions.empty()) {
        std::vector<std::vector<int>> overlay_inputs = global_partitions;
        overlay_inputs.push_back(anchor_partition);
        std::optional<ProcessedRefineResult> final_overlay =
            run_overlay_round(processed,
                              overlay_inputs,
                              tree_options,
                              current,
                              options.num_parts,
                              limits,
                              options,
                              runtime,
                              5000,
                              rng);
        if (final_overlay.has_value()) {
            log_partition_metrics("Final k-way overlay", final_overlay->metrics);
            if (better_partition(final_overlay->metrics, best_result.metrics, limits)) {
                best_result = *final_overlay;
            }
        }
    }

    return best_result;
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
            } else if (arg == "--projection-strategy" || arg == "--kway-projection") {
                ProjectionStrategy strategy = options.projection_strategy;
                const std::string value = read_value(i);
                if (!parse_projection_strategy(value, strategy)) {
                    throw std::runtime_error(
                        "unknown projection strategy: " + value +
                        " (expected one of: lda, random, projection, leading)");
                }
                options.projection_strategy = strategy;
            } else if (arg == "--hmetis") {
                options.hmetis_executable = read_value(i);
                options.hmetis_explicit = true;
            } else if (arg == "--ilp-partitioner") {
                options.ilp_partitioner_executable = read_value(i);
                options.ilp_partitioner_explicit = true;
            } else if (arg == "--triton-refiner" || arg == "--openroad") {
                options.triton_refiner_executable = read_value(i);
                options.triton_refiner_explicit = true;
            } else if (arg == "--disable-optimal-partitioner") {
                options.enable_optimal_partitioner = false;
            } else if (arg == "--disable-hmetis" || arg == "--disable-hmetis-partitioner") {
                options.enable_hmetis_partitioner = false;
            } else if (arg == "--disable-ilp" || arg == "--disable-ilp-partitioner") {
                options.enable_ilp_partitioner = false;
            } else if (arg == "--disable-triton-refiner") {
                options.enable_triton_refiner = false;
            } else if (arg == "--gpmetis" || arg == "--metis-path") {
                options.gpmetis_executable = read_value(i);
                options.gpmetis_explicit = true;
            } else if (arg == "--disable-metis") {
                options.enable_metis = false;
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
    if (options.num_parts > 2) {
        std::cout << "K-way projection strategy: "
                  << projection_strategy_name(options.projection_strategy) << '\n';
    } else if (options.projection_strategy != ProjectionStrategy::kLda) {
        std::cout << "Projection strategy option is only used on the current k-way spectral path; "
                  << "two-way mode keeps the direct Julia-style two-way embedding.\n";
    }

    std::vector<int> hint;
    if (!options.hint_file.empty()) {
        hint = read_partition_file(options.hint_file);
    }
    std::vector<int> processed_hint = remap_hint_to_processed(hint, isolated);
    std::vector<int> current = initial_partition_for_processed(processed, options, processed_hint, rng);
    ExternalRuntimeContext runtime = make_external_runtime_context(processed, options);
    log_external_runtime_context(options, runtime);

    ProcessedRefineResult refined = options.num_parts == 2
        ? two_way_spectral_refine(processed, current, options, runtime, rng)
        : k_way_spectral_refine(processed, current, options, runtime, rng);

    std::vector<int> full_partition = lift_partition_to_original(original, isolated, refined.partition, options.num_parts);
    const BalanceLimits original_limits = compute_balance_limits(original, options.num_parts, options.imb);
    full_partition = local_refine_partition(original, full_partition, options.num_parts, original_limits, rng);
    PartitionResult final_metrics = evaluate_partition(original, options.num_parts, full_partition);
    log_partition_metrics("Final original partition", final_metrics);

    write_partition_file(options.output_file, full_partition);
    return final_metrics;
}

}  // namespace kspecpart
