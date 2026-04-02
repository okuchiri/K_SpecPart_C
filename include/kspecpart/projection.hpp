#pragma once

#include "kspecpart/definitions.hpp"

#include <Eigen/Dense>

#include <vector>

namespace kspecpart {

const char* projection_strategy_name(ProjectionStrategy strategy);
bool parse_projection_strategy(const std::string& token, ProjectionStrategy& strategy);

Eigen::MatrixXd projection(const Eigen::MatrixXd& evec);
Eigen::MatrixXd dimensionality_reduction(const Eigen::MatrixXd& evec, int target_dims, int seed);

Eigen::MatrixXd concatenate_embeddings(const std::vector<Eigen::MatrixXd>& embeddings, int num_vertices);
Eigen::MatrixXd lda_reduce_embedding(const Eigen::MatrixXd& embedding,
                                     const std::vector<int>& labels,
                                     int target_dims);
Eigen::MatrixXd reduce_embedding_for_tree_partition(const Eigen::MatrixXd& embedding,
                                                    const std::vector<int>& labels,
                                                    int target_dims,
                                                    int seed,
                                                    ProjectionStrategy strategy = ProjectionStrategy::kLda);

std::vector<int> project_partition(const std::vector<int>& clusters,
                                   const std::vector<int>& contracted_partition,
                                   int num_vertices);

}  // namespace kspecpart
