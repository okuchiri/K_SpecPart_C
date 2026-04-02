#include "kspecpart/projection.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cctype>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace kspecpart {

namespace {

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
        return embedding;
    }

    std::unordered_map<int, std::vector<int>> class_members;
    for (int vertex = 0; vertex < static_cast<int>(labels.size()); ++vertex) {
        if (labels[vertex] >= 0) {
            class_members[labels[vertex]].push_back(vertex);
        }
    }
    if (class_members.size() <= 1) {
        return embedding.leftCols(std::min(target_dims, static_cast<int>(embedding.cols())));
    }

    const int dims = embedding.cols();
    Eigen::RowVectorXd overall_mean = embedding.colwise().mean();
    Eigen::MatrixXd sb = Eigen::MatrixXd::Zero(dims, dims);
    Eigen::MatrixXd sw = Eigen::MatrixXd::Zero(dims, dims);

    for (const auto& [label, members] : class_members) {
        (void)label;
        if (members.empty()) {
            continue;
        }
        Eigen::MatrixXd class_rows(members.size(), dims);
        for (int row = 0; row < static_cast<int>(members.size()); ++row) {
            class_rows.row(row) = embedding.row(members[row]);
        }
        const Eigen::RowVectorXd class_mean = class_rows.colwise().mean();
        const Eigen::RowVectorXd diff = class_mean - overall_mean;
        sb.noalias() += static_cast<double>(members.size()) * (diff.transpose() * diff);
        Eigen::MatrixXd centered = class_rows.rowwise() - class_mean;
        sw.noalias() += centered.transpose() * centered;
    }

    sw.diagonal().array() += 1e-9;
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(sb, sw);
    const int output_dims = std::min({target_dims, dims, std::max(1, static_cast<int>(class_members.size()) - 1)});
    if (output_dims <= 0 || solver.info() != Eigen::Success) {
        return embedding.leftCols(std::min(target_dims, dims));
    }

    Eigen::MatrixXd lda_projection(dims, output_dims);
    for (int col = 0; col < output_dims; ++col) {
        lda_projection.col(col) = solver.eigenvectors().col(dims - 1 - col);
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
