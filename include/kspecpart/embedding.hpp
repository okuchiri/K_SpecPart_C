#pragma once

#include "kspecpart/definitions.hpp"
#include "kspecpart/julia_random.hpp"

#include <Eigen/Dense>
#include <string>

namespace kspecpart {

struct PartitionIndex;

Eigen::MatrixXd solve_eigs(const Hypergraph& hypergraph,
                           const WeightedGraph& graph,
                           const PartitionIndex& pindex,
                           bool largest,
                           int requested_dims,
                           int iterations,
                           int epsilon = 1,
                           int seed = 0,
                           const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd(),
                           bool log_lobpcg = false,
                           AlgorithmRng* shared_rng = nullptr,
                           const std::string& debug_label = "");

}  // namespace kspecpart
