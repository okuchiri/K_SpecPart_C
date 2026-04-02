#pragma once

#include "kspecpart/definitions.hpp"

#include <Eigen/Dense>

namespace kspecpart {

// Julia uses Laplacians.jl's `akpw` helper to build a low-stretch spanning tree
// on the spectrally reordered graph. This C++ implementation keeps that module
// boundary and uses an internal AKPW-inspired multiscale heuristic.
WeightedGraph construct_akpw_lsst_tree(const WeightedGraph& graph, const Eigen::MatrixXd& embedding);

}  // namespace kspecpart
