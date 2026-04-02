#pragma once

#include "kspecpart/definitions.hpp"

#include <Eigen/Dense>

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
                           const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd());

}  // namespace kspecpart
