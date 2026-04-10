#include "kspecpart/lapack_eigen.hpp"
#include "kspecpart/projection.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace kspecpart {

namespace {

constexpr double kJuliaLdaRegcoef = 1.0e-6;

int clamped_output_dims(const Eigen::MatrixXd& embedding, int target_dims) {
    return std::max(0, std::min(target_dims, static_cast<int>(embedding.cols())));
}

Eigen::MatrixXd leading_columns_projection(const Eigen::MatrixXd& embedding, int target_dims) {
    const int dims = clamped_output_dims(embedding, target_dims);
    if (dims == 0) {
        return Eigen::MatrixXd::Zero(embedding.rows(), 0);
    }
    return embedding.leftCols(dims);
}

std::vector<int> stable_descending_permutation(const Eigen::VectorXd& values) {
    std::vector<int> permutation(values.size());
    for (int index = 0; index < values.size(); ++index) {
        permutation[index] = index;
    }
    std::stable_sort(permutation.begin(), permutation.end(), [&values](int lhs, int rhs) {
        return values[lhs] > values[rhs];
    });
    return permutation;
}

void regularize_symmetric_in_place(Eigen::MatrixXd& matrix, double lambda) {
    if (matrix.rows() == 0 || lambda <= 0.0) {
        return;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(matrix);
    if (solver.info() != Eigen::Success || solver.eigenvalues().size() == 0) {
        return;
    }
    const double emax = solver.eigenvalues().maxCoeff();
    if (emax == 0.0) {
        return;
    }
    matrix.diagonal().array() += emax * lambda;
}

Eigen::MatrixXd julia_like_lda_fallback(const Eigen::MatrixXd& embedding, int target_dims) {
    if (embedding.cols() <= 1) {
        return leading_columns_projection(embedding, std::max(1, target_dims));
    }
    // Julia's k-way code keeps nearby alternatives (`projection(...)`,
    // random projection, leading columns) around the LDA call site. When the
    // labels do not define a meaningful discriminant problem, using the
    // odd/even projection is a deterministic fallback that still compresses
    // the concatenated embedding to the 2-D tree-partition input shape.
    return projection(embedding);
}

bool solve_mclda_gevd(const Eigen::MatrixXd& sb,
                      const Eigen::MatrixXd& sw,
                      int output_dims,
                      Eigen::MatrixXd& lda_projection) {
    if (output_dims <= 0 || sb.rows() == 0 || sw.rows() == 0) {
        return false;
    }

    std::optional<LapackEigenResult> solver =
        lapack_generalized_symmetric_eigen(sb, sw);
    if (!solver.has_value() || solver->eigenvalues.size() < output_dims) {
        return false;
    }

    const std::vector<int> order = stable_descending_permutation(solver->eigenvalues);
    lda_projection.resize(sb.rows(), output_dims);
    for (int col = 0; col < output_dims; ++col) {
        lda_projection.col(col) = solver->eigenvectors.col(order[col]);
    }
    return true;
}

bool solve_degenerate_mclda_gevd(const Eigen::MatrixXd& sw,
                                 double regcoef,
                                 int output_dims,
                                 Eigen::MatrixXd& lda_projection) {
    if (output_dims <= 0 || sw.rows() == 0 || sw.rows() != sw.cols()) {
        return false;
    }

    Eigen::MatrixXd regularized_sw = sw;
    regularize_symmetric_in_place(regularized_sw, regcoef);
    Eigen::LLT<Eigen::MatrixXd> llt(regularized_sw);
    if (llt.info() != Eigen::Success) {
        return false;
    }

    const int dims = regularized_sw.rows();
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(dims, dims);
    const Eigen::MatrixXd upper_inverse = llt.matrixU().solve(identity);
    lda_projection = upper_inverse.leftCols(std::min(output_dims, dims));
    return lda_projection.cols() > 0;
}

bool solve_mclda_whiten(const Eigen::MatrixXd& sb,
                        const Eigen::MatrixXd& sw,
                        double regcoef,
                        int output_dims,
                        Eigen::MatrixXd& lda_projection) {
    if (output_dims <= 0 || sb.rows() == 0 || sw.rows() == 0) {
        return false;
    }

    std::optional<LapackEigenResult> sw_solver = lapack_symmetric_eigen(sw);
    if (!sw_solver.has_value() || sw_solver->eigenvalues.size() == 0) {
        return false;
    }

    Eigen::VectorXd eigenvalues = sw_solver->eigenvalues;
    const double shift = regcoef > 0.0 ? regcoef * eigenvalues.maxCoeff() : 0.0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        const double regularized = eigenvalues[i] + shift;
        if (!(regularized > 0.0) || !std::isfinite(regularized)) {
            return false;
        }
        eigenvalues[i] = 1.0 / std::sqrt(regularized);
    }

    const Eigen::MatrixXd whitening =
        sw_solver->eigenvectors * eigenvalues.asDiagonal();
    const Eigen::MatrixXd whitened_sb = whitening.transpose() * (sb * whitening);

    std::optional<LapackEigenResult> sb_solver = lapack_symmetric_eigen(whitened_sb);
    if (!sb_solver.has_value() || sb_solver->eigenvalues.size() < output_dims) {
        return false;
    }

    const std::vector<int> order = stable_descending_permutation(sb_solver->eigenvalues);
    lda_projection.resize(sb.rows(), output_dims);
    for (int col = 0; col < output_dims; ++col) {
        lda_projection.col(col) = whitening * sb_solver->eigenvectors.col(order[col]);
    }
    return true;
}

std::string normalize_strategy_token(std::string token) {
    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char ch) {
        if (ch == '_' || ch == ' ') {
            return static_cast<char>('-');
        }
        return static_cast<char>(std::tolower(ch));
    });
    return token;
}

}  // namespace

const char* projection_strategy_name(ProjectionStrategy strategy) {
    switch (strategy) {
        case ProjectionStrategy::kLda:
            return "lda";
        case ProjectionStrategy::kRandomSigned:
            return "random";
        case ProjectionStrategy::kAlternatingColumns:
            return "projection";
        case ProjectionStrategy::kLeadingColumns:
            return "leading";
    }
    return "lda";
}

bool parse_projection_strategy(const std::string& token, ProjectionStrategy& strategy) {
    const std::string normalized = normalize_strategy_token(token);
    if (normalized == "lda") {
        strategy = ProjectionStrategy::kLda;
        return true;
    }
    if (normalized == "random" || normalized == "random-signed" ||
        normalized == "dimensionality-reduction") {
        strategy = ProjectionStrategy::kRandomSigned;
        return true;
    }
    if (normalized == "projection" || normalized == "alternating" ||
        normalized == "alternating-columns" || normalized == "odd-even") {
        strategy = ProjectionStrategy::kAlternatingColumns;
        return true;
    }
    if (normalized == "leading" || normalized == "leading-columns" ||
        normalized == "prefix") {
        strategy = ProjectionStrategy::kLeadingColumns;
        return true;
    }
    return false;
}

Eigen::MatrixXd projection(const Eigen::MatrixXd& evec) {
    Eigen::MatrixXd projected = Eigen::MatrixXd::Zero(evec.rows(), 2);
    for (int col = 0; col < evec.cols(); ++col) {
        // Julia uses 1-based indexing: columns 1,3,5,... go to side_0.
        projected.col(col % 2 == 0 ? 0 : 1) += evec.col(col);
    }
    return projected;
}

Eigen::MatrixXd dimensionality_reduction(const Eigen::MatrixXd& evec, int target_dims, int seed) {
    if (target_dims <= 0) {
        return Eigen::MatrixXd::Zero(evec.rows(), 0);
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> sign_dist(0, 1);

    Eigen::MatrixXd rand_projection_matrix(evec.cols(), target_dims);
    for (int row = 0; row < rand_projection_matrix.rows(); ++row) {
        for (int col = 0; col < rand_projection_matrix.cols(); ++col) {
            rand_projection_matrix(row, col) = sign_dist(rng) == 0 ? -1.0 : 1.0;
        }
    }

    return evec * rand_projection_matrix;
}

Eigen::MatrixXd concatenate_embeddings(const std::vector<Eigen::MatrixXd>& embeddings, int num_vertices) {
    int total_cols = 0;
    for (const auto& embedding : embeddings) {
        if (embedding.rows() == num_vertices) {
            total_cols += embedding.cols();
        }
    }
    if (total_cols == 0) {
        return Eigen::MatrixXd::Zero(num_vertices, 0);
    }

    Eigen::MatrixXd concatenated(num_vertices, total_cols);
    int col_offset = 0;
    for (const auto& embedding : embeddings) {
        if (embedding.rows() != num_vertices || embedding.cols() == 0) {
            continue;
        }
        concatenated.block(0, col_offset, num_vertices, embedding.cols()) = embedding;
        col_offset += embedding.cols();
    }
    return concatenated;
}

Eigen::MatrixXd lda_reduce_embedding(const Eigen::MatrixXd& embedding,
                                     const std::vector<int>& labels,
                                     int target_dims) {
    if (embedding.rows() == 0 || embedding.cols() == 0 || target_dims <= 0 ||
        embedding.rows() != static_cast<int>(labels.size())) {
        return julia_like_lda_fallback(embedding, target_dims);
    }

    std::map<int, int> class_index;
    for (int label : labels) {
        class_index.emplace(label, static_cast<int>(class_index.size()));
    }

    const int dims = embedding.cols();
    const int requested_dims = std::min(target_dims, dims);
    if (requested_dims <= 0) {
        return julia_like_lda_fallback(embedding, target_dims);
    }

    const int num_classes = static_cast<int>(class_index.size());
    Eigen::MatrixXd class_means = Eigen::MatrixXd::Zero(dims, num_classes);
    std::vector<int> class_weights(num_classes, 0);
    Eigen::MatrixXd centered = Eigen::MatrixXd::Zero(dims, embedding.rows());

    std::vector<int> label_indices(labels.size(), 0);
    for (int vertex = 0; vertex < static_cast<int>(labels.size()); ++vertex) {
        const int class_id = class_index.find(labels[vertex])->second;
        label_indices[vertex] = class_id;
        class_means.col(class_id) += embedding.row(vertex).transpose();
        class_weights[class_id] += 1;
    }
    for (int class_id = 0; class_id < num_classes; ++class_id) {
        if (class_weights[class_id] > 0) {
            class_means.col(class_id) /= static_cast<double>(class_weights[class_id]);
        }
    }
    for (int vertex = 0; vertex < static_cast<int>(labels.size()); ++vertex) {
        centered.col(vertex) =
            embedding.row(vertex).transpose() - class_means.col(label_indices[vertex]);
    }

    Eigen::MatrixXd sb = Eigen::MatrixXd::Zero(dims, dims);
    Eigen::MatrixXd sw = Eigen::MatrixXd::Zero(dims, dims);
    if (centered.cols() > 0) {
        sw.noalias() = centered * centered.transpose();
    }

    Eigen::VectorXd class_weight_vector(class_means.cols());
    for (int class_id = 0; class_id < class_means.cols(); ++class_id) {
        class_weight_vector(class_id) = static_cast<double>(class_weights[class_id]);
    }
    const double total_weight = class_weight_vector.sum();
    Eigen::VectorXd overall_mean = Eigen::VectorXd::Zero(dims);
    if (total_weight > 0.0 && class_means.cols() > 0) {
        overall_mean.noalias() = class_means * (class_weight_vector / total_weight);
    }
    Eigen::MatrixXd centered_class_means = class_means.colwise() - overall_mean;
    for (int class_id = 0; class_id < centered_class_means.cols(); ++class_id) {
        centered_class_means.col(class_id) *= std::sqrt(std::max(0.0, class_weight_vector(class_id)));
    }
    if (centered_class_means.cols() > 0) {
        sb.noalias() = centered_class_means * centered_class_means.transpose();
    }

    const int output_dims = requested_dims;
    if (output_dims <= 0) {
        return julia_like_lda_fallback(embedding, requested_dims);
    }

    Eigen::MatrixXd lda_projection;
    Eigen::MatrixXd regularized_sw = sw;
    regularize_symmetric_in_place(regularized_sw, kJuliaLdaRegcoef);
    if (!solve_mclda_gevd(sb, regularized_sw, output_dims, lda_projection) &&
        !solve_mclda_whiten(sb, sw, kJuliaLdaRegcoef, output_dims, lda_projection) &&
        !solve_degenerate_mclda_gevd(sw, kJuliaLdaRegcoef, output_dims, lda_projection)) {
        return julia_like_lda_fallback(embedding, requested_dims);
    }
    return embedding * lda_projection;
}

Eigen::MatrixXd reduce_embedding_for_tree_partition(const Eigen::MatrixXd& embedding,
                                                    const std::vector<int>& labels,
                                                    int target_dims,
                                                    int seed,
                                                    ProjectionStrategy strategy) {
    if (embedding.cols() == 0) {
        return embedding;
    }

    // Julia's k-way flow is:
    //   concatenated embedding -> lda/random projection/odd-even filter -> tree_partition
    // The Julia `lda(...)` path needs an explicit transpose before `tree_partition` because
    // MultivariateStats returns a feature-major layout. Our Eigen helpers already return
    // vertex-major matrices, so the tree partitioner can consume the result directly.
    switch (strategy) {
        case ProjectionStrategy::kLda:
            return lda_reduce_embedding(embedding, labels, 2);
        case ProjectionStrategy::kRandomSigned:
            return dimensionality_reduction(embedding, target_dims, seed);
        case ProjectionStrategy::kAlternatingColumns:
            return projection(embedding);
        case ProjectionStrategy::kLeadingColumns:
            return leading_columns_projection(embedding, target_dims);
    }
    return lda_reduce_embedding(embedding, labels, 2);
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

}  // namespace kspecpart
