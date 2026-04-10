#pragma once

#include <Eigen/Dense>

#include <optional>

namespace kspecpart {

struct LapackEigenResult {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
};

std::optional<LapackEigenResult> lapack_symmetric_eigen(Eigen::MatrixXd matrix);

std::optional<LapackEigenResult> lapack_generalized_symmetric_eigen(Eigen::MatrixXd matrix_a,
                                                                    Eigen::MatrixXd matrix_b);

}  // namespace kspecpart
