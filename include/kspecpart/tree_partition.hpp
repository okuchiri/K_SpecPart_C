#pragma once

#include "kspecpart/definitions.hpp"

#include <Eigen/Dense>

#include <random>
#include <string>
#include <vector>

namespace kspecpart {

struct PartitionIndex;

struct BalanceLimits {
    int min_capacity = 0;
    int max_capacity = 0;
};

struct TreePartitionOptions {
    int num_parts = 2;
    int imb = 2;
    int eigvecs = 2;
    int solver_iters = 40;
    int cycles = 1;
    int best_solns = 3;
    int seed = 0;
    ProjectionStrategy projection_strategy = ProjectionStrategy::kLda;
    std::string gpmetis_executable;
    bool enable_metis = true;
    bool gpmetis_explicit = false;
};

struct TreePartitionCandidate {
    std::vector<int> partition;
    int cutsize = 0;
    std::vector<int> balance;
};

BalanceLimits compute_balance_limits(const Hypergraph& hypergraph, int num_parts, int imb);
long long balance_penalty(const std::vector<int>& balance, const BalanceLimits& limits);

std::vector<int> local_refine_partition(const Hypergraph& hypergraph,
                                        std::vector<int> partition,
                                        int num_parts,
                                        const BalanceLimits& limits,
                                        std::mt19937& rng);

std::vector<TreePartitionCandidate> tree_partition(const Hypergraph& hypergraph,
                                                   const TreePartitionOptions& options,
                                                   const std::vector<int>& base_partition,
                                                   std::mt19937& rng,
                                                   const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd());

std::vector<TreePartitionCandidate> tree_partition_with_embedding(const Hypergraph& hypergraph,
                                                                  const WeightedGraph& graph,
                                                                  const Eigen::MatrixXd& embedding,
                                                                  const PartitionIndex& fixed_vertices,
                                                                  const TreePartitionOptions& options,
                                                                  const std::vector<int>& base_partition,
                                                                  std::mt19937& rng);

std::vector<int> tree_partition_best(const Hypergraph& hypergraph,
                                     const TreePartitionOptions& options,
                                     const std::vector<int>& base_partition,
                                     std::mt19937& rng,
                                     const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd());

std::vector<int> tree_partition_best_with_embedding(const Hypergraph& hypergraph,
                                                    const WeightedGraph& graph,
                                                    const Eigen::MatrixXd& embedding,
                                                    const PartitionIndex& fixed_vertices,
                                                    const TreePartitionOptions& options,
                                                    const std::vector<int>& base_partition,
                                                    std::mt19937& rng);

std::vector<int> partition_two_way_hypergraph(const Hypergraph& hypergraph,
                                              int imb,
                                              int eigvecs,
                                              int solver_iters,
                                              int cycles,
                                              int best_solns,
                                              const std::vector<int>& base_partition,
                                              std::mt19937& rng,
                                              const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd());

std::vector<int> partition_k_way_hypergraph(const Hypergraph& hypergraph,
                                            int num_parts,
                                            int imb,
                                            int eigvecs,
                                            int solver_iters,
                                            int cycles,
                                            int best_solns,
                                            const std::vector<int>& base_partition,
                                            std::mt19937& rng,
                                            ProjectionStrategy projection_strategy = ProjectionStrategy::kLda,
                                            const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd());

}  // namespace kspecpart
