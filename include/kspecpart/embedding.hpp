#pragma once

#include "kspecpart/definitions.hpp"

#include <Eigen/Dense>

#include <random>

namespace kspecpart {

Eigen::MatrixXd leading_eigenvectors(const WeightedGraph& graph, int requested_dims, int iterations, std::mt19937& rng);

}  // namespace kspecpart
