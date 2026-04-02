#include "kspecpart/embedding.hpp"

#include "kspecpart/cut_distillation.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace kspecpart {

namespace {

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using DenseOperator = std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>;
using BlockPreconditioner = std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>;

constexpr int kSmallGraphThreshold = 100;
constexpr int kDirectSolveThreshold = 500;
constexpr int kMaxHierarchyLevels = 8;
constexpr double kBGramTolerance = 1e-10;
constexpr double kJuliaLargeLobpcgTolerance = 1e-40;

struct CmgLevel {
    SparseMatrix matrix;
    Eigen::VectorXd inv_diag;
    SparseMatrix prolongation;
    std::vector<int> coarse_index;
    int num_coarse = 0;
    int num_vertices = 0;
    int num_nonzeros = 0;
    bool sd = false;
    bool is_last = false;
    bool iterative = true;
    Eigen::MatrixXd direct_matrix;
    Eigen::LDLT<Eigen::MatrixXd> direct_solver;
    bool has_direct_solver = false;
};

struct CmgHierarchy {
    std::vector<CmgLevel> levels;
    int input_size = 0;
    bool strict_dominant = false;
};

struct CmgLevelAux {
    bool forward = true;
    int recursive_call = 1;
    int repeat = 1;
    bool sd = false;
    bool is_last = false;
    bool iterative = true;
    int num_vertices = 0;
    int num_coarse = 0;
};

struct CmgWorkspace {
    Eigen::VectorXd x;
    Eigen::VectorXd b;
    Eigen::VectorXd tmp;
};

struct CmgPreparedState {
    std::vector<CmgLevelAux> aux;
    std::vector<CmgWorkspace> workspaces;
};

struct MinSparseResult {
    std::vector<int> min_cols;
    Eigen::VectorXd min_vals;
};

struct ForestComponentsResult {
    std::vector<int> component_index;
    int num_components = 0;
    std::vector<int> component_sizes;
};

struct RitzPair {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
};

struct CmgRowSumResult {
    Eigen::VectorXd row_sum;
    int flag = 0;
    std::vector<Eigen::Triplet<double>> triplets;
};

struct CmgValidationResult {
    int flag = 0;
    SparseMatrix matrix;
    bool augmented = false;
    int original_size = 0;
};

struct LobpcgBlocks {
    Eigen::MatrixXd block;
    Eigen::MatrixXd a_block;
    Eigen::MatrixXd b_block;
};

struct LobpcgBlockGram {
    Eigen::MatrixXd xax;
    Eigen::MatrixXd xar;
    Eigen::MatrixXd xap;
    Eigen::MatrixXd rar;
    Eigen::MatrixXd rap;
    Eigen::MatrixXd pap;
};

struct LobpcgConstraint {
    Eigen::MatrixXd y;
    Eigen::MatrixXd by;
    Eigen::MatrixXd gram_ybv;
    Eigen::MatrixXd tmp;
    Eigen::LLT<Eigen::MatrixXd> gram_chol;
    bool enabled = false;
};

struct LobpcgCholQr {
    Eigen::MatrixXd gram_vbv;
};

struct LobpcgIteratorState {
    LobpcgBlocks x_blocks;
    LobpcgBlocks temp_x_blocks;
    LobpcgBlocks p_blocks;
    LobpcgBlocks active_p_blocks;
    LobpcgBlocks r_blocks;
    LobpcgBlocks active_r_blocks;
    Eigen::VectorXd ritz_values;
    std::vector<int> lambda_perm;
    Eigen::VectorXd lambda;
    Eigen::MatrixXd v_basis;
    Eigen::VectorXd residual_norm_values;
    LobpcgBlockGram gram_a_block;
    LobpcgBlockGram gram_b_block;
    Eigen::MatrixXd gram_a;
    Eigen::MatrixXd gram_b;
    LobpcgConstraint constraint;
    LobpcgCholQr cholqr;
    std::vector<char> active_mask;
    int current_block_size = 0;
    int iteration = 1;
    bool largest = false;
};

Eigen::MatrixXd dense_graph_laplacian(const WeightedGraph& graph) {
    Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(graph.num_vertices, graph.num_vertices);
    for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
        laplacian(vertex, vertex) += graph.degrees[vertex] + 1e-6;
        for (const auto& [neighbor, weight] : graph.adjacency[vertex]) {
            laplacian(vertex, neighbor) -= weight;
        }
    }
    return laplacian;
}

SparseMatrix sparse_graph_laplacian(const WeightedGraph& graph) {
    std::vector<Eigen::Triplet<double>> triplets;
    std::size_t adjacency_entries = 0;
    for (const auto& neighbors : graph.adjacency) {
        adjacency_entries += neighbors.size();
    }
    triplets.reserve(adjacency_entries + graph.num_vertices);

    for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
        triplets.emplace_back(vertex, vertex, graph.degrees[vertex] + 1e-6);
        for (const auto& [neighbor, weight] : graph.adjacency[vertex]) {
            triplets.emplace_back(vertex, neighbor, -weight);
        }
    }

    SparseMatrix laplacian(graph.num_vertices, graph.num_vertices);
    laplacian.setFromTriplets(
        triplets.begin(), triplets.end(), [](double lhs, double rhs) { return lhs + rhs; });
    laplacian.makeCompressed();
    return laplacian;
}

double hyperedge_scale(int edge_size) {
    if (edge_size <= 1) {
        return 1.0;
    }
    return (std::floor(edge_size / 2.0) * std::ceil(edge_size / 2.0)) /
           static_cast<double>(edge_size - 1);
}

Eigen::MatrixXd apply_hypergraph_operator(const Hypergraph& hypergraph,
                                          const Eigen::MatrixXd& vectors,
                                          int epsilon) {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(hypergraph.num_vertices, vectors.cols());
    const int safe_epsilon = std::max(1, epsilon);

    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        const int start = hypergraph.eptr[edge];
        const int end = hypergraph.eptr[edge + 1];
        const int edge_size = end - start;
        if (edge_size <= 1) {
            continue;
        }

        const double coeff = static_cast<double>(hypergraph.hwts[edge]) /
                             (hyperedge_scale(edge_size) * static_cast<double>(safe_epsilon));
        Eigen::RowVectorXd mean = Eigen::RowVectorXd::Zero(vectors.cols());
        for (int idx = start; idx < end; ++idx) {
            mean += vectors.row(hypergraph.eind[idx]);
        }
        mean /= static_cast<double>(edge_size);

        for (int idx = start; idx < end; ++idx) {
            const int vertex = hypergraph.eind[idx];
            result.row(vertex) += coeff * (vectors.row(vertex) - mean);
        }
    }

    return result;
}

Eigen::MatrixXd dense_hypergraph_operator(const Hypergraph& hypergraph, int epsilon) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(hypergraph.num_vertices, hypergraph.num_vertices);
    const int safe_epsilon = std::max(1, epsilon);

    for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
        const int start = hypergraph.eptr[edge];
        const int end = hypergraph.eptr[edge + 1];
        const int edge_size = end - start;
        if (edge_size <= 1) {
            continue;
        }

        const double coeff = static_cast<double>(hypergraph.hwts[edge]) /
                             (hyperedge_scale(edge_size) * static_cast<double>(safe_epsilon));
        const double diag = coeff * (1.0 - 1.0 / static_cast<double>(edge_size));
        const double off_diag = -coeff / static_cast<double>(edge_size);

        for (int lhs = start; lhs < end; ++lhs) {
            const int u = hypergraph.eind[lhs];
            matrix(u, u) += diag;
            for (int rhs = start; rhs < end; ++rhs) {
                if (lhs == rhs) {
                    continue;
                }
                const int v = hypergraph.eind[rhs];
                matrix(u, v) += off_diag;
            }
        }
    }

    return matrix;
}

Eigen::MatrixXd apply_clique_operator(const Eigen::VectorXd& weights,
                                      double total_weight,
                                      const Eigen::MatrixXd& vectors) {
    if (vectors.rows() == 0) {
        return Eigen::MatrixXd();
    }
    if (total_weight <= 0.0) {
        return vectors;
    }

    const Eigen::RowVectorXd weighted_sum = weights.transpose() * vectors;
    Eigen::MatrixXd result = vectors;
    for (int row = 0; row < vectors.rows(); ++row) {
        result.row(row) *= weights(row);
        result.row(row).noalias() -= (weights(row) / total_weight) * weighted_sum;
    }
    return result;
}

Eigen::MatrixXd dense_clique_operator(const Hypergraph& hypergraph) {
    const int n = hypergraph.num_vertices;
    const double total_weight = std::accumulate(hypergraph.vwts.begin(), hypergraph.vwts.end(), 0.0);
    if (n == 0) {
        return Eigen::MatrixXd();
    }
    if (total_weight <= 0.0) {
        return Eigen::MatrixXd::Identity(n, n);
    }

    Eigen::VectorXd weights(n);
    for (int vertex = 0; vertex < n; ++vertex) {
        weights(vertex) = static_cast<double>(hypergraph.vwts[vertex]);
    }

    Eigen::MatrixXd matrix = weights.asDiagonal();
    matrix.noalias() -= (weights * weights.transpose()) / total_weight;
    return matrix;
}

Eigen::MatrixXd apply_biclique_operator(int num_vertices,
                                        const PartitionIndex& pindex,
                                        const Eigen::MatrixXd& vectors) {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(num_vertices, vectors.cols());
    const int left_size = static_cast<int>(pindex.p1.size());
    const int right_size = static_cast<int>(pindex.p2.size());
    if (left_size == 0 || right_size == 0 || vectors.cols() == 0) {
        return result;
    }

    Eigen::RowVectorXd left_sum = Eigen::RowVectorXd::Zero(vectors.cols());
    for (int vertex : pindex.p1) {
        if (vertex >= 0 && vertex < num_vertices) {
            left_sum += vectors.row(vertex);
        }
    }

    Eigen::RowVectorXd right_sum = Eigen::RowVectorXd::Zero(vectors.cols());
    for (int vertex : pindex.p2) {
        if (vertex >= 0 && vertex < num_vertices) {
            right_sum += vectors.row(vertex);
        }
    }

    for (int vertex : pindex.p1) {
        if (vertex >= 0 && vertex < num_vertices) {
            result.row(vertex) = static_cast<double>(right_size) * vectors.row(vertex) - right_sum;
        }
    }
    for (int vertex : pindex.p2) {
        if (vertex >= 0 && vertex < num_vertices) {
            result.row(vertex) = static_cast<double>(left_size) * vectors.row(vertex) - left_sum;
        }
    }
    return result;
}

Eigen::MatrixXd dense_biclique_operator(int num_vertices, const PartitionIndex& pindex) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(num_vertices, num_vertices);
    const int left_size = static_cast<int>(pindex.p1.size());
    const int right_size = static_cast<int>(pindex.p2.size());
    if (left_size == 0 || right_size == 0) {
        return matrix;
    }

    for (int vertex : pindex.p1) {
        if (vertex >= 0 && vertex < num_vertices) {
            matrix(vertex, vertex) += right_size;
        }
    }
    for (int vertex : pindex.p2) {
        if (vertex >= 0 && vertex < num_vertices) {
            matrix(vertex, vertex) += left_size;
        }
    }
    for (int lhs : pindex.p1) {
        if (lhs < 0 || lhs >= num_vertices) {
            continue;
        }
        for (int rhs : pindex.p2) {
            if (rhs < 0 || rhs >= num_vertices) {
                continue;
            }
            matrix(lhs, rhs) -= 1.0;
            matrix(rhs, lhs) -= 1.0;
        }
    }
    return matrix;
}

Eigen::MatrixXd helmert_complement_basis(int n) {
    Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(n, std::max(0, n - 1));
    for (int col = 0; col + 1 < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            basis(row, col) = 1.0;
        }
        basis(col + 1, col) = -static_cast<double>(col + 1);
        basis.col(col) /= std::sqrt(static_cast<double>((col + 1) * (col + 2)));
    }
    return basis;
}

void symmetrize_in_place(Eigen::MatrixXd& matrix) {
    matrix = 0.5 * (matrix + matrix.transpose());
}

bool sparse_is_symmetric(const SparseMatrix& matrix, double tolerance = 1e-12) {
    if (matrix.rows() != matrix.cols()) {
        return false;
    }
    for (int row = 0; row < matrix.rows(); ++row) {
        for (SparseMatrix::InnerIterator it(matrix, row); it; ++it) {
            if (std::abs(it.value() - matrix.coeff(it.col(), it.row())) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

CmgRowSumResult find_row_sum_and_dominance(const SparseMatrix& matrix) {
    CmgRowSumResult result;
    result.row_sum = Eigen::VectorXd::Zero(matrix.rows());
    result.triplets.reserve(matrix.nonZeros());

    for (int row = 0; row < matrix.rows(); ++row) {
        for (SparseMatrix::InnerIterator it(matrix, row); it; ++it) {
            result.row_sum(row) += it.value();
            result.triplets.emplace_back(it.row(), it.col(), it.value());
            if (it.row() != it.col() && it.value() > 0.0) {
                result.flag = 2;
            }
        }
    }
    return result;
}

CmgValidationResult validate_cmg_input(const SparseMatrix& matrix) {
    CmgValidationResult validation;
    validation.flag = 0;
    validation.matrix = matrix;
    validation.augmented = false;
    validation.original_size = matrix.rows();

    if (!sparse_is_symmetric(matrix)) {
        validation.flag = 1;
        return validation;
    }

    const CmgRowSumResult row_sum = find_row_sum_and_dominance(matrix);
    if (row_sum.flag == 2) {
        validation.flag = 2;
        return validation;
    }

    const Eigen::VectorXd diagonal = matrix.diagonal();
    Eigen::VectorXd positive_row_sum = Eigen::VectorXd::Zero(matrix.rows());
    std::vector<int> strict_indices;
    strict_indices.reserve(matrix.rows());
    for (int idx = 0; idx < matrix.rows(); ++idx) {
        positive_row_sum(idx) = 0.5 * (row_sum.row_sum(idx) + std::abs(row_sum.row_sum(idx)));
        const double diag = std::max(diagonal(idx), 1e-18);
        if (positive_row_sum(idx) / diag > 1e-13) {
            strict_indices.push_back(idx);
        }
    }

    validation.flag = 3;
    if (strict_indices.empty()) {
        return validation;
    }

    std::vector<Eigen::Triplet<double>> triplets = row_sum.triplets;
    triplets.reserve(triplets.size() + 2 * strict_indices.size() + 1);
    const int augmented_index = matrix.rows();
    double extra_diagonal = 0.0;
    for (int idx : strict_indices) {
        const double edge_weight = -positive_row_sum(idx);
        triplets.emplace_back(augmented_index, idx, edge_weight);
        triplets.emplace_back(idx, augmented_index, edge_weight);
        extra_diagonal -= edge_weight;
    }
    triplets.emplace_back(augmented_index, augmented_index, extra_diagonal);

    SparseMatrix augmented(matrix.rows() + 1, matrix.cols() + 1);
    augmented.setFromTriplets(
        triplets.begin(), triplets.end(), [](double lhs, double rhs) { return lhs + rhs; });
    augmented.makeCompressed();
    validation.matrix = augmented;
    validation.augmented = true;
    return validation;
}

Eigen::MatrixXd take_eigenvectors(const Eigen::MatrixXd& eigenvectors,
                                  bool largest,
                                  int requested_dims,
                                  int skip_smallest) {
    const int available = std::max(0, static_cast<int>(eigenvectors.cols()) - skip_smallest);
    const int dims = std::min(requested_dims, available);
    Eigen::MatrixXd result(eigenvectors.rows(), dims);
    if (dims == 0) {
        return result;
    }

    if (largest) {
        for (int col = 0; col < dims; ++col) {
            result.col(col) = eigenvectors.col(eigenvectors.cols() - 1 - col);
        }
        return result;
    }

    for (int col = 0; col < dims; ++col) {
        result.col(col) = eigenvectors.col(skip_smallest + col);
    }
    return result;
}

MinSparseResult find_min_sparse(const SparseMatrix& matrix) {
    const int n = matrix.rows();
    MinSparseResult result;
    result.min_cols.resize(n, 0);
    result.min_vals = Eigen::VectorXd::Zero(n);

    for (int row = 0; row < n; ++row) {
        bool found = false;
        double min_value = std::numeric_limits<double>::infinity();
        int min_col = row;

        for (SparseMatrix::InnerIterator it(matrix, row); it; ++it) {
            const double value = it.value();
            const int col = it.col();
            if (!found || value < min_value ||
                (value == min_value && col < min_col)) {
                found = true;
                min_value = value;
                min_col = col;
            }
        }

        result.min_cols[row] = min_col;
        result.min_vals(row) = found ? min_value : 0.0;
    }

    return result;
}

void ensure_buffer_size(std::vector<int>& buffer, int min_size) {
    if (static_cast<int>(buffer.size()) >= min_size) {
        return;
    }
    const int next_size =
        std::min(std::max(min_size, 2 * static_cast<int>(buffer.size())), std::max(min_size, 1));
    buffer.resize(next_size, 0);
}

void split_forest(std::vector<int>& parents) {
    const int n = static_cast<int>(parents.size());
    std::vector<int> ancestors(n, 0);
    std::vector<int> indegree(n + 2, 0);
    std::vector<char> visited(n, 0);
    std::vector<int> walkbuffer(20, 0);
    std::vector<int> newancestorbuff(20, 0);

    for (int vertex = 0; vertex < n; ++vertex) {
        const int parent = parents[vertex];
        if (parent >= 0 && parent < n) {
            indegree[parent] += 1;
        }
    }

    for (int start = 0; start < n; ++start) {
        int walk = start;
        bool startwalk = true;

        while (startwalk && indegree[walk] == 0 && !visited[walk]) {
            startwalk = false;
            int ancestors_in_path = 0;
            int path_size = 1;
            walkbuffer[0] = walk;
            newancestorbuff[0] = 0;

            while (path_size <= 6 || visited[walk]) {
                walk = parents[walk];
                const bool terminated =
                    walk == walkbuffer[path_size - 1] ||
                    (path_size > 1 && walk == walkbuffer[path_size - 2]);
                if (terminated) {
                    break;
                }

                ensure_buffer_size(walkbuffer, path_size + 1);
                ensure_buffer_size(newancestorbuff, path_size + 1);
                walkbuffer[path_size] = walk;
                if (visited[walk]) {
                    newancestorbuff[path_size] = ancestors_in_path;
                } else {
                    ancestors_in_path += 1;
                    newancestorbuff[path_size] = ancestors_in_path;
                }
                path_size += 1;
            }

            if (path_size > 6) {
                const int middle = (path_size - 1) / 2;
                parents[walkbuffer[middle]] = walkbuffer[middle];
                indegree[walkbuffer[middle + 1]] -= 1;

                for (int idx = middle + 1; idx < path_size; ++idx) {
                    ancestors[walkbuffer[idx]] -= ancestors[walkbuffer[middle]];
                }

                for (int idx = 0; idx <= middle; ++idx) {
                    visited[walkbuffer[idx]] = 1;
                    ancestors[walkbuffer[idx]] += newancestorbuff[idx];
                }

                walk = walkbuffer[middle + 1];
                startwalk = true;
            }

            if (!startwalk) {
                for (int idx = 0; idx < path_size; ++idx) {
                    ancestors[walkbuffer[idx]] += newancestorbuff[idx];
                    visited[walkbuffer[idx]] = 1;
                }
            }
        }
    }

    for (int start = 0; start < n; ++start) {
        int walk = start;
        bool startwalk = true;

        while (startwalk && indegree[walk] == 0) {
            startwalk = false;
            int previous = walk;
            bool cut_mode = false;
            int new_front = 0;
            int removed_ancestors = 0;

            while (true) {
                const int parent = parents[walk];
                const bool terminated = parent == walk || parent == previous;
                if (terminated) {
                    break;
                }

                if (!cut_mode &&
                    ancestors[walk] > 2 &&
                    ancestors[parent] - ancestors[walk] > 2) {
                    parents[walk] = walk;
                    indegree[parent] -= 1;
                    removed_ancestors = ancestors[walk];
                    new_front = parent;
                    cut_mode = true;
                }

                previous = walk;
                walk = parent;
                if (cut_mode) {
                    ancestors[walk] -= removed_ancestors;
                }
            }

            if (cut_mode) {
                startwalk = true;
                walk = new_front;
            }
        }
    }
}

std::vector<int> update_groups(const SparseMatrix& matrix,
                               std::vector<int> parents,
                               const Eigen::VectorXd& diagonal) {
    const int n = static_cast<int>(parents.size());
    Eigen::VectorXd incident_tree_weight = Eigen::VectorXd::Zero(n);

    for (int vertex = 0; vertex < n; ++vertex) {
        if (parents[vertex] == vertex) {
            continue;
        }
        const double weight = matrix.coeff(vertex, parents[vertex]);
        incident_tree_weight(vertex) += weight;
        incident_tree_weight(parents[vertex]) += weight;
    }

    for (int vertex = 0; vertex < n; ++vertex) {
        const double degree = diagonal(vertex);
        if (degree <= 0.0) {
            parents[vertex] = vertex;
            continue;
        }
        if (incident_tree_weight(vertex) / degree > -0.125) {
            parents[vertex] = vertex;
        }
    }
    return parents;
}

ForestComponentsResult forest_components(const std::vector<int>& parents) {
    const int n = static_cast<int>(parents.size());
    ForestComponentsResult result;
    result.component_index.assign(n, -1);
    result.component_sizes.assign(n, 0);
    std::vector<int> buffer(100, 0);
    int next_component = 0;

    for (int vertex = 0; vertex < n; ++vertex) {
        int buffer_size = 0;
        int walk = vertex;

        while (result.component_index[walk] < 0) {
            result.component_index[walk] = next_component;
            ensure_buffer_size(buffer, buffer_size + 1);
            buffer[buffer_size] = walk;
            buffer_size += 1;
            walk = parents[walk];
        }

        const int end = parents[walk];
        if (result.component_index[end] != next_component) {
            const int existing_component = result.component_index[end];
            for (int idx = 0; idx < buffer_size; ++idx) {
                result.component_index[buffer[idx]] = existing_component;
            }
        } else {
            next_component += 1;
        }
        result.component_sizes[end] += buffer_size;
    }

    result.num_components = next_component;
    while (result.num_components > 0 &&
           result.component_sizes[result.num_components - 1] == 0) {
        result.num_components -= 1;
    }
    result.component_sizes.resize(result.num_components);
    return result;
}

ForestComponentsResult steiner_group(const SparseMatrix& matrix, const Eigen::VectorXd& diagonal) {
    MinSparseResult min_sparse = find_min_sparse(matrix);
    std::vector<int> parents = std::move(min_sparse.min_cols);
    split_forest(parents);

    double min_effective_degree = std::numeric_limits<double>::infinity();
    for (int vertex = 0; vertex < diagonal.size(); ++vertex) {
        const double degree = diagonal(vertex);
        if (degree <= 0.0) {
            min_effective_degree = 0.0;
            continue;
        }
        min_effective_degree =
            std::min(min_effective_degree, std::abs(min_sparse.min_vals(vertex) / degree));
    }
    if (min_effective_degree < 0.125) {
        parents = update_groups(matrix, std::move(parents), diagonal);
    }

    return forest_components(parents);
}

SparseMatrix build_prolongation(const std::vector<int>& aggregate, int coarse_vertices) {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(aggregate.size());
    for (int vertex = 0; vertex < static_cast<int>(aggregate.size()); ++vertex) {
        triplets.emplace_back(vertex, aggregate[vertex], 1.0);
    }

    SparseMatrix prolongation(static_cast<int>(aggregate.size()), coarse_vertices);
    prolongation.setFromTriplets(
        triplets.begin(), triplets.end(), [](double lhs, double rhs) { return lhs + rhs; });
    prolongation.makeCompressed();
    return prolongation;
}

CmgLevel build_direct_last_level(const SparseMatrix& matrix, bool sd) {
    CmgLevel level;
    level.sd = sd;
    level.is_last = true;
    level.iterative = false;
    level.num_vertices = matrix.rows();

    if (matrix.rows() <= 1) {
        level.matrix.resize(0, 0);
        level.direct_matrix.resize(0, 0);
        level.num_nonzeros = 0;
        return level;
    }

    SparseMatrix reduced = matrix.topLeftCorner(matrix.rows() - 1, matrix.cols() - 1);
    reduced.makeCompressed();
    level.matrix = reduced;
    level.num_nonzeros = reduced.nonZeros();
    level.direct_matrix = Eigen::MatrixXd(reduced);
    level.direct_matrix = 0.5 * (level.direct_matrix + level.direct_matrix.transpose());
    level.direct_solver.compute(level.direct_matrix);
    level.has_direct_solver = level.direct_solver.info() == Eigen::Success;
    return level;
}

CmgHierarchy build_cmg_like_hierarchy(const SparseMatrix& laplacian) {
    CmgHierarchy hierarchy;
    const CmgValidationResult validation = validate_cmg_input(laplacian);
    hierarchy.input_size = validation.original_size;
    hierarchy.strict_dominant = validation.augmented;

    SparseMatrix current = validation.matrix;
    const int original_nnz = current.nonZeros();
    int hierarchy_nnz = 0;
    bool first_level = true;

    for (int level = 0; level < kMaxHierarchyLevels; ++level) {
        const Eigen::VectorXd diagonal = current.diagonal();

        if (current.rows() < kDirectSolveThreshold) {
            hierarchy.levels.push_back(build_direct_last_level(current, true));
            break;
        }

        const ForestComponentsResult groups = steiner_group(current, diagonal);
        const int coarse_vertices = groups.num_components;
        hierarchy_nnz += current.nonZeros();

        CmgLevel cmg_level;
        cmg_level.matrix = current;
        cmg_level.inv_diag =
            (2.0 * diagonal.array().max(1e-9)).inverse().matrix();
        cmg_level.coarse_index = groups.component_index;
        cmg_level.num_coarse = coarse_vertices;
        cmg_level.num_vertices = current.rows();
        cmg_level.num_nonzeros = current.nonZeros();
        cmg_level.sd = first_level ? validation.augmented : true;

        if (coarse_vertices <= 1 || coarse_vertices >= current.rows() - 1 ||
            hierarchy_nnz > 5 * std::max(original_nnz, 1)) {
            cmg_level.is_last = true;
            cmg_level.iterative = true;
            hierarchy.levels.push_back(std::move(cmg_level));
            break;
        }

        cmg_level.prolongation =
            build_prolongation(groups.component_index, coarse_vertices);
        cmg_level.is_last = false;
        cmg_level.iterative = true;
        hierarchy.levels.push_back(cmg_level);
        first_level = false;

        SparseMatrix coarse = cmg_level.prolongation.transpose() * current * cmg_level.prolongation;
        coarse.makeCompressed();
        current = 0.5 * (coarse + SparseMatrix(coarse.transpose()));
        current.makeCompressed();
    }

    if (hierarchy.levels.empty()) {
        CmgLevel level;
        level.matrix = current;
        level.inv_diag = (2.0 * current.diagonal().array().max(1e-9)).inverse().matrix();
        level.num_vertices = current.rows();
        level.num_nonzeros = current.nonZeros();
        level.sd = validation.augmented;
        level.is_last = true;
        level.iterative = true;
        hierarchy.levels.push_back(std::move(level));
    }

    return hierarchy;
}

std::vector<CmgLevelAux> init_cmg_level_aux(const CmgHierarchy& hierarchy) {
    std::vector<CmgLevelAux> aux(hierarchy.levels.size());
    for (int level = 0; level < static_cast<int>(hierarchy.levels.size()); ++level) {
        int repeat = 1;
        if (level == static_cast<int>(hierarchy.levels.size()) - 1) {
            repeat = 0;
        } else if (level > 0) {
            const int prev_nnz = std::max(1, hierarchy.levels[level - 1].num_nonzeros);
            const int curr_nnz = std::max(1, hierarchy.levels[level].num_nonzeros);
            repeat = std::max(static_cast<int>(std::floor(static_cast<double>(prev_nnz) /
                                                          static_cast<double>(curr_nnz) - 1.0)),
                              1);
        }
        aux[level].forward = true;
        aux[level].recursive_call = 1;
        aux[level].repeat = repeat;
        aux[level].sd = hierarchy.levels[level].sd;
        aux[level].is_last = hierarchy.levels[level].is_last;
        aux[level].iterative = hierarchy.levels[level].iterative;
        aux[level].num_vertices = hierarchy.levels[level].num_vertices;
        aux[level].num_coarse = hierarchy.levels[level].num_coarse;
    }
    return aux;
}

std::vector<CmgWorkspace> init_cmg_workspaces(const CmgHierarchy& hierarchy) {
    std::vector<CmgWorkspace> workspaces;
    workspaces.reserve(hierarchy.levels.size());
    for (const CmgLevel& level : hierarchy.levels) {
        int size = level.num_vertices;
        if (level.is_last && !level.iterative) {
            size = std::max(size, static_cast<int>(level.matrix.rows()) + 1);
        }
        CmgWorkspace workspace;
        workspace.x = Eigen::VectorXd::Zero(size);
        workspace.b = Eigen::VectorXd::Zero(size);
        workspace.tmp = Eigen::VectorXd::Zero(size);
        workspaces.push_back(std::move(workspace));
    }
    return workspaces;
}

CmgPreparedState init_cmg_preconditioner_state(const CmgHierarchy& hierarchy) {
    CmgPreparedState prepared;
    prepared.aux = init_cmg_level_aux(hierarchy);
    prepared.workspaces = init_cmg_workspaces(hierarchy);
    return prepared;
}

void reset_cmg_preconditioner_state(CmgPreparedState& prepared) {
    for (CmgLevelAux& aux : prepared.aux) {
        aux.forward = true;
        aux.recursive_call = 1;
    }
    for (CmgWorkspace& workspace : prepared.workspaces) {
        workspace.x.setZero();
        workspace.b.setZero();
        workspace.tmp.setZero();
    }
}

void interpolate_rhs(Eigen::VectorXd& coarse_rhs,
                     const std::vector<int>& coarse_index,
                     const Eigen::VectorXd& fine_vector) {
    coarse_rhs.setZero();
    for (int vertex = 0; vertex < fine_vector.size(); ++vertex) {
        const int coarse = coarse_index[vertex];
        if (coarse >= 0 && coarse < coarse_rhs.size()) {
            coarse_rhs(coarse) += fine_vector(vertex);
        }
    }
}

Eigen::VectorXd prolongate_solution(const std::vector<int>& coarse_index,
                                    const Eigen::VectorXd& coarse_solution) {
    Eigen::VectorXd fine = Eigen::VectorXd::Zero(static_cast<int>(coarse_index.size()));
    for (int vertex = 0; vertex < static_cast<int>(coarse_index.size()); ++vertex) {
        const int coarse = coarse_index[vertex];
        if (coarse >= 0 && coarse < coarse_solution.size()) {
            fine(vertex) = coarse_solution(coarse);
        }
    }
    return fine;
}

Eigen::VectorXd apply_cmg_preconditioner_vector(const CmgHierarchy& hierarchy,
                                                CmgPreparedState& prepared,
                                                const Eigen::VectorXd& rhs) {
    if (hierarchy.levels.empty()) {
        return rhs;
    }

    Eigen::VectorXd effective_rhs = rhs;
    if (hierarchy.strict_dominant &&
        hierarchy.levels.front().num_vertices == rhs.size() + 1) {
        effective_rhs.resize(rhs.size() + 1);
        effective_rhs.head(rhs.size()) = rhs;
        effective_rhs(rhs.size()) = -rhs.sum();
    }

    reset_cmg_preconditioner_state(prepared);
    prepared.workspaces.front().b.head(effective_rhs.size()) = effective_rhs;

    int level_index = 0;
    while (level_index >= 0) {
        const CmgLevel& level = hierarchy.levels[level_index];
        CmgWorkspace& workspace = prepared.workspaces[level_index];
        Eigen::VectorXd& x = workspace.x;
        Eigen::VectorXd& b = workspace.b;
        Eigen::VectorXd& tmp = workspace.tmp;
        CmgLevelAux& state = prepared.aux[level_index];

        if (state.is_last && !state.iterative) {
            x.setZero();
            if (level.has_direct_solver && level.direct_matrix.rows() > 0) {
                const int direct_size = level.direct_matrix.rows();
                x.head(direct_size) =
                    level.direct_solver.solve(b.head(direct_size));
            }
            level_index -= 1;
        } else if (state.is_last && state.iterative) {
            x = b.cwiseProduct(level.inv_diag);
            level_index -= 1;
        } else if (state.forward) {
            if (state.recursive_call > state.repeat) {
                state.recursive_call = 1;
                level_index -= 1;
            } else {
                if (state.recursive_call == 1) {
                    x = b.cwiseProduct(level.inv_diag);
                } else {
                    tmp = level.matrix * x;
                    tmp -= b;
                    tmp = (-level.inv_diag.array() * tmp.array()).matrix();
                    x += tmp;
                }

                tmp = level.matrix * x;
                tmp -= b;
                state.forward = false;
                interpolate_rhs(prepared.workspaces[level_index + 1].b, level.coarse_index, -tmp);
                level_index += 1;
            }
        } else {
            x += prolongate_solution(level.coarse_index, prepared.workspaces[level_index + 1].x);
            tmp = level.matrix * x;
            tmp -= b;
            tmp = (-level.inv_diag.array() * tmp.array()).matrix();
            x += tmp;
            state.recursive_call += 1;
            state.forward = true;
        }
    }

    Eigen::VectorXd result = prepared.workspaces.front().x;
    if (hierarchy.strict_dominant && result.size() == rhs.size() + 1) {
        return (result.head(rhs.size()).array() + result(rhs.size())).matrix();
    }
    if (result.size() == rhs.size()) {
        return result;
    }
    return result.head(std::min<int>(rhs.size(), result.size()));
}

Eigen::MatrixXd apply_cmg_preconditioner(const CmgHierarchy& hierarchy, const Eigen::MatrixXd& residuals) {
    Eigen::MatrixXd result(residuals.rows(), residuals.cols());
    CmgPreparedState prepared = init_cmg_preconditioner_state(hierarchy);
    for (int col = 0; col < residuals.cols(); ++col) {
        result.col(col) = apply_cmg_preconditioner_vector(hierarchy, prepared, residuals.col(col));
    }
    return result;
}

void project_zero_sum(Eigen::MatrixXd& vectors) {
    if (vectors.cols() == 0) {
        return;
    }
    const Eigen::RowVectorXd means = vectors.colwise().mean();
    vectors.rowwise() -= means;
}

void right_divide_upper_triangular_in_place(Eigen::MatrixXd& matrix,
                                            const Eigen::MatrixXd& upper) {
    if (matrix.cols() == 0) {
        return;
    }
    Eigen::MatrixXd transposed = matrix.transpose();
    upper.transpose().template triangularView<Eigen::Lower>().solveInPlace(transposed);
    matrix = transposed.transpose();
}

Eigen::MatrixXd eigen_b_orthonormalize(const DenseOperator& apply_b,
                                       Eigen::MatrixXd vectors,
                                       bool project_constant) {
    if (vectors.cols() == 0) {
        return Eigen::MatrixXd(vectors.rows(), 0);
    }

    for (int pass = 0; pass < 2; ++pass) {
        if (project_constant) {
            project_zero_sum(vectors);
        }
        const Eigen::MatrixXd b_vectors = apply_b(vectors);
        Eigen::MatrixXd gram = vectors.transpose() * b_vectors;
        symmetrize_in_place(gram);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(gram);
        if (solver.info() != Eigen::Success) {
            return Eigen::MatrixXd(vectors.rows(), 0);
        }

        std::vector<int> keep;
        for (int idx = 0; idx < gram.rows(); ++idx) {
            if (solver.eigenvalues()(idx) > kBGramTolerance) {
                keep.push_back(idx);
            }
        }
        if (keep.empty()) {
            return Eigen::MatrixXd(vectors.rows(), 0);
        }

        Eigen::MatrixXd transform = Eigen::MatrixXd::Zero(gram.rows(), static_cast<int>(keep.size()));
        for (int idx = 0; idx < static_cast<int>(keep.size()); ++idx) {
            transform.col(idx) = solver.eigenvectors().col(keep[idx]) /
                                 std::sqrt(std::max(solver.eigenvalues()(keep[idx]), kBGramTolerance));
        }
        vectors *= transform;
    }

    return vectors;
}

bool cholqr_orthonormalize_in_place(const DenseOperator& apply_b,
                                    Eigen::MatrixXd& vectors,
                                    Eigen::MatrixXd* a_vectors,
                                    Eigen::MatrixXd* b_vectors,
                                    Eigen::MatrixXd& gram_workspace,
                                    bool project_constant) {
    if (vectors.cols() == 0) {
        return true;
    }
    if (project_constant) {
        project_zero_sum(vectors);
        if (a_vectors != nullptr && a_vectors->cols() == vectors.cols()) {
            *a_vectors = Eigen::MatrixXd();
        }
        if (b_vectors != nullptr && b_vectors->cols() == vectors.cols()) {
            *b_vectors = Eigen::MatrixXd();
        }
    }

    Eigen::MatrixXd local_b =
        (b_vectors != nullptr && b_vectors->cols() == vectors.cols()) ? *b_vectors : apply_b(vectors);
    gram_workspace.resize(vectors.cols(), vectors.cols());
    gram_workspace.noalias() = vectors.transpose() * local_b;
    symmetrize_in_place(gram_workspace);
    gram_workspace.diagonal() = gram_workspace.diagonal().real();

    Eigen::LLT<Eigen::MatrixXd> chol(gram_workspace);
    if (chol.info() != Eigen::Success) {
        return false;
    }
    const Eigen::MatrixXd upper = chol.matrixU();
    if ((upper.diagonal().array().abs() <= std::sqrt(kBGramTolerance)).any()) {
        return false;
    }

    right_divide_upper_triangular_in_place(vectors, upper);
    if (a_vectors != nullptr && a_vectors->cols() == vectors.cols()) {
        right_divide_upper_triangular_in_place(*a_vectors, upper);
    }
    if (b_vectors != nullptr && b_vectors->cols() == vectors.cols()) {
        right_divide_upper_triangular_in_place(local_b, upper);
        *b_vectors = local_b;
    }
    return true;
}

[[maybe_unused]] Eigen::MatrixXd b_orthonormalize(const DenseOperator& apply_b,
                                                  Eigen::MatrixXd vectors,
                                                  bool project_constant = true) {
    if (vectors.cols() == 0) {
        return Eigen::MatrixXd(vectors.rows(), 0);
    }
    Eigen::MatrixXd gram;
    if (cholqr_orthonormalize_in_place(apply_b, vectors, nullptr, nullptr, gram, project_constant)) {
        return vectors;
    }
    return eigen_b_orthonormalize(apply_b, std::move(vectors), project_constant);
}

std::optional<RitzPair> solve_projected_eigenproblem(const Eigen::MatrixXd& basis,
                                                     const Eigen::MatrixXd& a_basis,
                                                     const Eigen::MatrixXd& b_basis,
                                                     bool largest,
                                                     int requested_dims) {
    if (basis.cols() == 0 || requested_dims <= 0) {
        return std::nullopt;
    }

    Eigen::MatrixXd small_a = basis.transpose() * a_basis;
    Eigen::MatrixXd small_b = basis.transpose() * b_basis;
    symmetrize_in_place(small_a);
    symmetrize_in_place(small_b);
    small_b.diagonal().array() += 1e-9;

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(small_a, small_b);
    if (solver.info() != Eigen::Success) {
        return std::nullopt;
    }

    const int take = std::min(requested_dims, static_cast<int>(basis.cols()));
    RitzPair pair;
    pair.eigenvalues.resize(take);
    pair.eigenvectors.resize(basis.cols(), take);

    if (largest) {
        for (int col = 0; col < take; ++col) {
            const int idx = static_cast<int>(basis.cols()) - 1 - col;
            pair.eigenvalues(col) = solver.eigenvalues()(idx);
            pair.eigenvectors.col(col) = solver.eigenvectors().col(idx);
        }
    } else {
        for (int col = 0; col < take; ++col) {
            pair.eigenvalues(col) = solver.eigenvalues()(col);
            pair.eigenvectors.col(col) = solver.eigenvectors().col(col);
        }
    }
    return pair;
}

Eigen::VectorXd residual_norms(const Eigen::MatrixXd& residuals) {
    Eigen::VectorXd norms(residuals.cols());
    for (int col = 0; col < residuals.cols(); ++col) {
        norms(col) = residuals.col(col).norm();
    }
    return norms;
}

Eigen::MatrixXd gather_columns_by_mask(const Eigen::MatrixXd& matrix,
                                       const std::vector<char>& mask,
                                       int block_size) {
    if (block_size <= 0) {
        return Eigen::MatrixXd(matrix.rows(), 0);
    }
    Eigen::MatrixXd gathered(matrix.rows(), block_size);
    int offset = 0;
    for (int col = 0; col < matrix.cols() && offset < block_size; ++col) {
        if (!mask[col]) {
            continue;
        }
        gathered.col(offset) = matrix.col(col);
        offset += 1;
    }
    if (offset < block_size) {
        gathered.conservativeResize(matrix.rows(), offset);
    }
    return gathered;
}

LobpcgBlocks make_empty_blocks(int rows, int cols) {
    return {Eigen::MatrixXd::Zero(rows, cols), Eigen::MatrixXd::Zero(rows, cols),
            Eigen::MatrixXd::Zero(rows, cols)};
}

LobpcgBlockGram make_block_gram(int size_x);

int size_x(const LobpcgIteratorState& state) {
    return static_cast<int>(state.x_blocks.block.cols());
}

LobpcgConstraint make_empty_constraint(int rows, int cols) {
    LobpcgConstraint constraint;
    constraint.y = Eigen::MatrixXd(rows, 0);
    constraint.by = Eigen::MatrixXd(rows, 0);
    constraint.gram_ybv = Eigen::MatrixXd(0, cols);
    constraint.tmp = Eigen::MatrixXd(0, cols);
    constraint.enabled = false;
    return constraint;
}

LobpcgConstraint initialize_constraint(const DenseOperator& apply_b,
                                       const Eigen::MatrixXd& constraint_basis,
                                       int block_cols) {
    if (constraint_basis.cols() == 0 || constraint_basis.rows() == 0) {
        return make_empty_constraint(constraint_basis.rows(), block_cols);
    }

    LobpcgConstraint constraint;
    constraint.y = constraint_basis;
    constraint.by = apply_b(constraint_basis);
    Eigen::MatrixXd gram = constraint.y.transpose() * constraint.by;
    symmetrize_in_place(gram);
    constraint.gram_chol.compute(gram);
    if (constraint.gram_chol.info() != Eigen::Success) {
        return make_empty_constraint(constraint_basis.rows(), block_cols);
    }
    constraint.gram_ybv = Eigen::MatrixXd::Zero(constraint_basis.cols(), block_cols);
    constraint.tmp = Eigen::MatrixXd::Zero(constraint_basis.cols(), block_cols);
    constraint.enabled = true;
    return constraint;
}

void apply_constraint(LobpcgConstraint& constraint, Eigen::MatrixXd& block) {
    if (!constraint.enabled || constraint.y.cols() == 0 || block.cols() == 0) {
        return;
    }

    const int cols = block.cols();
    constraint.gram_ybv.leftCols(cols).noalias() = constraint.by.transpose() * block;
    constraint.tmp.leftCols(cols) = constraint.gram_chol.solve(constraint.gram_ybv.leftCols(cols));
    block -= constraint.y * constraint.tmp.leftCols(cols);
}

void update_active_block(const std::vector<char>& mask,
                         int block_size,
                         Eigen::MatrixXd& active_block,
                         const Eigen::MatrixXd& block) {
    active_block = gather_columns_by_mask(block, mask, block_size);
}

void update_active_r_blocks(LobpcgIteratorState& state) {
    const int block_size = state.current_block_size;
    update_active_block(state.active_mask, block_size, state.active_r_blocks.block, state.r_blocks.block);
    state.active_r_blocks.a_block.resize(state.active_r_blocks.block.rows(), 0);
    state.active_r_blocks.b_block.resize(state.active_r_blocks.block.rows(), 0);
}

void update_active_rp_blocks(LobpcgIteratorState& state) {
    const int block_size = state.current_block_size;
    update_active_r_blocks(state);
    update_active_block(state.active_mask, block_size, state.active_p_blocks.block, state.p_blocks.block);
    update_active_block(state.active_mask, block_size, state.active_p_blocks.a_block, state.p_blocks.a_block);
    update_active_block(state.active_mask, block_size, state.active_p_blocks.b_block, state.p_blocks.b_block);
}

void update_block_products(LobpcgBlocks& blocks,
                           const DenseOperator& apply_a,
                           const DenseOperator& apply_b) {
    blocks.a_block = apply_a(blocks.block);
    blocks.b_block = apply_b(blocks.block);
}

LobpcgBlocks b_orthonormalize_blocks(const DenseOperator& apply_a,
                                     const DenseOperator& apply_b,
                                     LobpcgCholQr& cholqr,
                                     LobpcgBlocks blocks,
                                     bool project_constant) {
    if (blocks.block.cols() == 0) {
        return make_empty_blocks(blocks.block.rows(), 0);
    }

    if (!cholqr_orthonormalize_in_place(apply_b, blocks.block, nullptr, nullptr,
                                        cholqr.gram_vbv, project_constant)) {
        blocks.block = eigen_b_orthonormalize(apply_b, std::move(blocks.block), project_constant);
    }
    if (blocks.block.cols() == 0) {
        return make_empty_blocks(blocks.block.rows(), 0);
    }
    update_block_products(blocks, apply_a, apply_b);
    return blocks;
}

bool ortho_ab_mul_x(LobpcgBlocks& blocks,
                    LobpcgCholQr& cholqr,
                    const DenseOperator& apply_a,
                    const DenseOperator& apply_b,
                    bool project_constant) {
    blocks =
        b_orthonormalize_blocks(apply_a, apply_b, cholqr, std::move(blocks), project_constant);
    return blocks.block.cols() > 0;
}

Eigen::MatrixXd initialize_lobpcg_block(int n, int dims, int seed);
void prepare_initial_lobpcg_block(LobpcgIteratorState& state, int seed);

LobpcgIteratorState initialize_lobpcg_iterator(const DenseOperator& /*apply_a*/,
                                               const DenseOperator& apply_b,
                                               int n,
                                               int dims,
                                               int seed,
                                               bool largest,
                                               bool /*project_constant*/,
                                               const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd()) {
    LobpcgIteratorState state;
    const Eigen::MatrixXd valid_constraint_basis =
        (constraint_basis.rows() == n) ? constraint_basis : Eigen::MatrixXd();
    state.constraint = initialize_constraint(apply_b, valid_constraint_basis, dims);
    state.x_blocks.block = initialize_lobpcg_block(n, dims, seed);
    if (state.x_blocks.block.cols() == 0) {
        return state;
    }

    const int actual_dims = static_cast<int>(state.x_blocks.block.cols());
    state.temp_x_blocks = make_empty_blocks(n, actual_dims);
    state.p_blocks = make_empty_blocks(n, actual_dims);
    state.active_p_blocks = make_empty_blocks(n, actual_dims);
    state.r_blocks = make_empty_blocks(n, actual_dims);
    state.active_r_blocks = make_empty_blocks(n, actual_dims);
    state.ritz_values = Eigen::VectorXd::Zero(3 * actual_dims);
    state.lambda_perm.assign(3 * actual_dims, 0);
    state.lambda = Eigen::VectorXd::Zero(actual_dims);
    state.v_basis = Eigen::MatrixXd::Zero(3 * actual_dims, 3 * actual_dims);
    state.residual_norm_values = Eigen::VectorXd::Zero(actual_dims);
    state.gram_a_block = make_block_gram(actual_dims);
    state.gram_b_block = make_block_gram(actual_dims);
    state.gram_a = Eigen::MatrixXd::Zero(3 * actual_dims, 3 * actual_dims);
    state.gram_b = Eigen::MatrixXd::Zero(3 * actual_dims, 3 * actual_dims);
    state.cholqr.gram_vbv = Eigen::MatrixXd::Zero(actual_dims, actual_dims);
    state.active_mask.assign(actual_dims, 1);
    state.current_block_size = actual_dims;
    state.iteration = 1;
    state.largest = largest;
    return state;
}

void residuals_step(LobpcgIteratorState& state) {
    const int dims = size_x(state);
    state.r_blocks.block =
        state.x_blocks.a_block - state.x_blocks.b_block * state.ritz_values.head(dims).asDiagonal();
    state.residual_norm_values = residual_norms(state.r_blocks.block);
}

void update_mask_step(LobpcgIteratorState& state, double tolerance) {
    const int dims = size_x(state);
    state.active_mask.assign(dims, 0);
    state.current_block_size = 0;
    for (int idx = 0; idx < dims; ++idx) {
        if (state.residual_norm_values(idx) > tolerance) {
            state.active_mask[idx] = 1;
            state.current_block_size += 1;
        }
    }
}

void precond_constr_step(LobpcgIteratorState& state,
                         const BlockPreconditioner& apply_preconditioner) {
    if (state.current_block_size <= 0) {
        return;
    }
    state.active_r_blocks.block = apply_preconditioner(state.active_r_blocks.block);
    apply_constraint(state.constraint, state.active_r_blocks.block);
}

LobpcgBlockGram make_block_gram(int size_x) {
    return {Eigen::MatrixXd::Zero(size_x, size_x), Eigen::MatrixXd::Zero(size_x, size_x),
            Eigen::MatrixXd::Zero(size_x, size_x), Eigen::MatrixXd::Zero(size_x, size_x),
            Eigen::MatrixXd::Zero(size_x, size_x), Eigen::MatrixXd::Zero(size_x, size_x)};
}

void compute_xax(LobpcgBlockGram& gram, const LobpcgBlocks& x_blocks) {
    gram.xax = x_blocks.block.transpose() * x_blocks.a_block;
    symmetrize_in_place(gram.xax);
}

void compute_xar(LobpcgBlockGram& gram,
                 const LobpcgBlocks& x_blocks,
                 const LobpcgBlocks& r_blocks,
                 int bs) {
    gram.xar.leftCols(bs) = x_blocks.block.transpose() * r_blocks.a_block.leftCols(bs);
}

void compute_xap(LobpcgBlockGram& gram,
                 const LobpcgBlocks& x_blocks,
                 const LobpcgBlocks& p_blocks,
                 int bs) {
    gram.xap.leftCols(bs) = x_blocks.block.transpose() * p_blocks.a_block.leftCols(bs);
}

void compute_rar(LobpcgBlockGram& gram, const LobpcgBlocks& r_blocks, int bs) {
    Eigen::MatrixXd local =
        r_blocks.block.leftCols(bs).transpose() * r_blocks.a_block.leftCols(bs);
    symmetrize_in_place(local);
    gram.rar.topLeftCorner(bs, bs) = std::move(local);
}

void compute_rap(LobpcgBlockGram& gram,
                 const LobpcgBlocks& r_blocks,
                 const LobpcgBlocks& p_blocks,
                 int bs_r,
                 int bs_p) {
    gram.rap.topLeftCorner(bs_r, bs_p) =
        r_blocks.a_block.leftCols(bs_r).transpose() * p_blocks.block.leftCols(bs_p);
}

void compute_pap(LobpcgBlockGram& gram, const LobpcgBlocks& p_blocks, int bs) {
    Eigen::MatrixXd local =
        p_blocks.block.leftCols(bs).transpose() * p_blocks.a_block.leftCols(bs);
    symmetrize_in_place(local);
    gram.pap.topLeftCorner(bs, bs) = std::move(local);
}

void compute_xbr(LobpcgBlockGram& gram,
                 const LobpcgBlocks& x_blocks,
                 const LobpcgBlocks& r_blocks,
                 int bs) {
    gram.xar.leftCols(bs) = x_blocks.block.transpose() * r_blocks.b_block.leftCols(bs);
}

void compute_xbp(LobpcgBlockGram& gram,
                 const LobpcgBlocks& x_blocks,
                 const LobpcgBlocks& p_blocks,
                 int bs) {
    gram.xap.leftCols(bs) = x_blocks.block.transpose() * p_blocks.b_block.leftCols(bs);
}

void compute_rbp(LobpcgBlockGram& gram,
                 const LobpcgBlocks& r_blocks,
                 const LobpcgBlocks& p_blocks,
                 int bs_r,
                 int bs_p) {
    gram.rap.topLeftCorner(bs_r, bs_p) =
        r_blocks.b_block.leftCols(bs_r).transpose() * p_blocks.block.leftCols(bs_p);
}

void sort_selected_permutation(std::vector<int>& permutation,
                               const Eigen::VectorXd& values,
                               int subdim,
                               bool largest) {
    permutation.resize(std::max(subdim, 0));
    std::iota(permutation.begin(), permutation.end(), 0);
    std::stable_sort(permutation.begin(), permutation.end(), [&](int lhs, int rhs) {
        if (values(lhs) != values(rhs)) {
            return largest ? values(lhs) > values(rhs) : values(lhs) < values(rhs);
        }
        return lhs < rhs;
    });
}

void update_selected_ritz_pairs(LobpcgIteratorState& state,
                                const Eigen::VectorXd& values,
                                const Eigen::MatrixXd& vectors,
                                int subdim) {
    const int size_x_value = size_x(state);
    sort_selected_permutation(state.lambda_perm, values, subdim, state.largest);
    for (int col = 0; col < size_x_value; ++col) {
        const int index = state.lambda_perm[col];
        state.ritz_values(col) = values(index);
        state.v_basis.block(0, col, subdim, 1) = vectors.col(index);
    }
    state.lambda.head(size_x_value) = state.ritz_values.head(size_x_value);
}

void block_grams_1x1_step(LobpcgIteratorState& state) {
    compute_xax(state.gram_a_block, state.x_blocks);
}

void block_grams_2x2_step(LobpcgIteratorState& state, int block_size) {
    const int size_x_value = size_x(state);
    compute_xar(state.gram_a_block, state.x_blocks, state.active_r_blocks, block_size);
    compute_rar(state.gram_a_block, state.active_r_blocks, block_size);
    compute_xbr(state.gram_b_block, state.x_blocks, state.active_r_blocks, block_size);

    state.gram_a.topLeftCorner(size_x_value + block_size, size_x_value + block_size).setZero();
    state.gram_b.topLeftCorner(size_x_value + block_size, size_x_value + block_size).setZero();
    state.gram_a.topLeftCorner(size_x_value, size_x_value) =
        state.ritz_values.head(size_x_value).asDiagonal();
    state.gram_a.topRightCorner(size_x_value, block_size) =
        state.gram_a_block.xar.leftCols(block_size);
    state.gram_a.bottomLeftCorner(block_size, size_x_value) =
        state.gram_a_block.xar.leftCols(block_size).transpose();
    state.gram_a.bottomRightCorner(block_size, block_size) =
        state.gram_a_block.rar.topLeftCorner(block_size, block_size);
    state.gram_b.topLeftCorner(size_x_value, size_x_value).setIdentity();
    state.gram_b.topRightCorner(size_x_value, block_size) =
        state.gram_b_block.xar.leftCols(block_size);
    state.gram_b.bottomLeftCorner(block_size, size_x_value) =
        state.gram_b_block.xar.leftCols(block_size).transpose();
    state.gram_b.bottomRightCorner(block_size, block_size).setIdentity();
}

void block_grams_3x3_step(LobpcgIteratorState& state,
                          int block_size_r,
                          int block_size_p) {
    const int size_x_value = size_x(state);
    compute_xar(state.gram_a_block, state.x_blocks, state.active_r_blocks, block_size_r);
    compute_xap(state.gram_a_block, state.x_blocks, state.active_p_blocks, block_size_p);
    compute_rar(state.gram_a_block, state.active_r_blocks, block_size_r);
    compute_rap(state.gram_a_block, state.active_r_blocks, state.active_p_blocks,
                block_size_r, block_size_p);
    compute_pap(state.gram_a_block, state.active_p_blocks, block_size_p);
    compute_xbr(state.gram_b_block, state.x_blocks, state.active_r_blocks, block_size_r);
    compute_xbp(state.gram_b_block, state.x_blocks, state.active_p_blocks, block_size_p);
    compute_rbp(state.gram_b_block, state.active_r_blocks, state.active_p_blocks,
                block_size_r, block_size_p);

    const int subdim = size_x_value + block_size_r + block_size_p;
    state.gram_a.topLeftCorner(subdim, subdim).setZero();
    state.gram_b.topLeftCorner(subdim, subdim).setZero();
    state.gram_a.topLeftCorner(size_x_value, size_x_value) =
        state.ritz_values.head(size_x_value).asDiagonal();
    state.gram_a.block(0, size_x_value, size_x_value, block_size_r) =
        state.gram_a_block.xar.leftCols(block_size_r);
    state.gram_a.block(size_x_value, 0, block_size_r, size_x_value) =
        state.gram_a_block.xar.leftCols(block_size_r).transpose();
    state.gram_a.block(size_x_value, size_x_value, block_size_r, block_size_r) =
        state.gram_a_block.rar.topLeftCorner(block_size_r, block_size_r);
    state.gram_a.block(0, size_x_value + block_size_r, size_x_value, block_size_p) =
        state.gram_a_block.xap.leftCols(block_size_p);
    state.gram_a.block(size_x_value + block_size_r, 0, block_size_p, size_x_value) =
        state.gram_a_block.xap.leftCols(block_size_p).transpose();
    state.gram_a.block(size_x_value, size_x_value + block_size_r, block_size_r, block_size_p) =
        state.gram_a_block.rap.topLeftCorner(block_size_r, block_size_p);
    state.gram_a.block(size_x_value + block_size_r, size_x_value, block_size_p, block_size_r) =
        state.gram_a_block.rap.topLeftCorner(block_size_r, block_size_p).transpose();
    state.gram_a.block(size_x_value + block_size_r, size_x_value + block_size_r,
                       block_size_p, block_size_p) =
        state.gram_a_block.pap.topLeftCorner(block_size_p, block_size_p);

    state.gram_b.topLeftCorner(size_x_value, size_x_value).setIdentity();
    state.gram_b.block(0, size_x_value, size_x_value, block_size_r) =
        state.gram_b_block.xar.leftCols(block_size_r);
    state.gram_b.block(size_x_value, 0, block_size_r, size_x_value) =
        state.gram_b_block.xar.leftCols(block_size_r).transpose();
    state.gram_b.block(size_x_value, size_x_value, block_size_r, block_size_r).setIdentity();
    state.gram_b.block(0, size_x_value + block_size_r, size_x_value, block_size_p) =
        state.gram_b_block.xap.leftCols(block_size_p);
    state.gram_b.block(size_x_value + block_size_r, 0, block_size_p, size_x_value) =
        state.gram_b_block.xap.leftCols(block_size_p).transpose();
    state.gram_b.block(size_x_value, size_x_value + block_size_r, block_size_r, block_size_p) =
        state.gram_b_block.rap.topLeftCorner(block_size_r, block_size_p);
    state.gram_b.block(size_x_value + block_size_r, size_x_value, block_size_p, block_size_r) =
        state.gram_b_block.rap.topLeftCorner(block_size_r, block_size_p).transpose();
    state.gram_b.block(size_x_value + block_size_r, size_x_value + block_size_r,
                       block_size_p, block_size_p).setIdentity();
}

bool sub_problem_step(LobpcgIteratorState& state, int block_size_r, int block_size_p) {
    const int size_x_value = size_x(state);
    const int subdim = size_x_value + block_size_r + block_size_p;
    if (subdim <= 0 || size_x_value <= 0) {
        return false;
    }

    if (block_size_r == 0 && block_size_p == 0) {
        Eigen::MatrixXd gram_a_view =
            state.gram_a_block.xax.topLeftCorner(subdim, subdim);
        symmetrize_in_place(gram_a_view);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(gram_a_view);
        if (solver.info() != Eigen::Success) {
            return false;
        }
        update_selected_ritz_pairs(state, solver.eigenvalues(), solver.eigenvectors(), subdim);
        return true;
    }

    Eigen::MatrixXd gram_a_view = state.gram_a.topLeftCorner(subdim, subdim);
    Eigen::MatrixXd gram_b_view = state.gram_b.topLeftCorner(subdim, subdim);
    symmetrize_in_place(gram_a_view);
    symmetrize_in_place(gram_b_view);
    gram_b_view.diagonal().array() += 1e-9;

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(gram_a_view, gram_b_view);
    if (solver.info() != Eigen::Success) {
        return false;
    }
    update_selected_ritz_pairs(state, solver.eigenvalues(), solver.eigenvectors(), subdim);
    return true;
}

void update_x_p_step(LobpcgIteratorState& state, int block_size_r, int block_size_p) {
    const int size_x_value = size_x(state);
    const Eigen::MatrixXd x_eig = state.v_basis.topLeftCorner(size_x_value, size_x_value);
    const Eigen::MatrixXd r_eig =
        block_size_r > 0
            ? state.v_basis.block(size_x_value, 0, block_size_r, size_x_value)
            : Eigen::MatrixXd(0, size_x_value);
    const Eigen::MatrixXd p_eig =
        block_size_p > 0
            ? state.v_basis.block(size_x_value + block_size_r, 0, block_size_p, size_x_value)
            : Eigen::MatrixXd(0, size_x_value);

    state.p_blocks = make_empty_blocks(state.x_blocks.block.rows(), size_x_value);
    if (block_size_r > 0) {
        state.p_blocks.block.noalias() += state.active_r_blocks.block.leftCols(block_size_r) * r_eig;
        state.p_blocks.a_block.noalias() += state.active_r_blocks.a_block.leftCols(block_size_r) * r_eig;
        state.p_blocks.b_block.noalias() += state.active_r_blocks.b_block.leftCols(block_size_r) * r_eig;
    }
    if (block_size_p > 0) {
        state.temp_x_blocks.block = state.active_p_blocks.block.leftCols(block_size_p) * p_eig;
        state.temp_x_blocks.a_block = state.active_p_blocks.a_block.leftCols(block_size_p) * p_eig;
        state.temp_x_blocks.b_block = state.active_p_blocks.b_block.leftCols(block_size_p) * p_eig;
        state.p_blocks.block += state.temp_x_blocks.block;
        state.p_blocks.a_block += state.temp_x_blocks.a_block;
        state.p_blocks.b_block += state.temp_x_blocks.b_block;
    }

    state.temp_x_blocks.block = state.x_blocks.block * x_eig;
    state.temp_x_blocks.a_block = state.x_blocks.a_block * x_eig;
    state.temp_x_blocks.b_block = state.x_blocks.b_block * x_eig;
    if (block_size_r > 0 || block_size_p > 0) {
        state.x_blocks.block = state.temp_x_blocks.block + state.p_blocks.block;
        state.x_blocks.a_block = state.temp_x_blocks.a_block + state.p_blocks.a_block;
        state.x_blocks.b_block = state.temp_x_blocks.b_block + state.p_blocks.b_block;
    } else {
        state.x_blocks.block = state.temp_x_blocks.block;
        state.x_blocks.a_block = state.temp_x_blocks.a_block;
        state.x_blocks.b_block = state.temp_x_blocks.b_block;
    }
}

void fill_random_lobpcg_block(Eigen::MatrixXd& block, std::mt19937& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int row = 0; row < block.rows(); ++row) {
        for (int col = 0; col < block.cols(); ++col) {
            block(row, col) = uniform(rng);
        }
    }
}

bool is_effectively_zero_column(const Eigen::VectorXd& column) {
    return column.norm() <= std::sqrt(kBGramTolerance);
}

void refill_zero_columns(Eigen::MatrixXd& block, std::mt19937& rng) {
    if (block.cols() == 0) {
        return;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int col = 0; col < block.cols(); ++col) {
        if (!is_effectively_zero_column(block.col(col))) {
            continue;
        }
        for (int row = 0; row < block.rows(); ++row) {
            block(row, col) = uniform(rng);
        }
    }
}

Eigen::MatrixXd initialize_lobpcg_block(int n, int dims, int seed) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    Eigen::MatrixXd block(n, dims);
    fill_random_lobpcg_block(block, rng);
    return block;
}

void prepare_initial_lobpcg_block(LobpcgIteratorState& state, int seed) {
    apply_constraint(state.constraint, state.x_blocks.block);
    if (!state.constraint.enabled) {
        return;
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    refill_zero_columns(state.x_blocks.block, rng);
    apply_constraint(state.constraint, state.x_blocks.block);
}

Eigen::MatrixXd finalize_lobpcg_basis(const DenseOperator& apply_a,
                                      const DenseOperator& apply_b,
                                      Eigen::MatrixXd basis,
                                      bool largest) {
    if (basis.cols() == 0) {
        return Eigen::MatrixXd();
    }

    const Eigen::MatrixXd a_basis = apply_a(basis);
    const Eigen::MatrixXd b_basis = apply_b(basis);
    std::optional<RitzPair> ritz =
        solve_projected_eigenproblem(basis, a_basis, b_basis, largest, basis.cols());
    if (!ritz.has_value()) {
        return Eigen::MatrixXd();
    }

    basis *= ritz->eigenvectors;
    return basis;
}

bool ortho_active_r_step(LobpcgIteratorState& state,
                         const DenseOperator& apply_a,
                         const DenseOperator& apply_b,
                         bool project_constant) {
    return ortho_ab_mul_x(state.active_r_blocks, state.cholqr, apply_a, apply_b,
                          project_constant);
}

bool ortho_active_p_step(LobpcgIteratorState& state,
                         const DenseOperator& apply_a,
                         const DenseOperator& apply_b,
                         bool project_constant) {
    if (state.active_p_blocks.block.cols() == 0) {
        return false;
    }
    if (!cholqr_orthonormalize_in_place(apply_b,
                                        state.active_p_blocks.block,
                                        &state.active_p_blocks.a_block,
                                        &state.active_p_blocks.b_block,
                                        state.cholqr.gram_vbv,
                                        project_constant)) {
        state.active_p_blocks = b_orthonormalize_blocks(apply_a, apply_b,
                                                        state.cholqr,
                                                        std::move(state.active_p_blocks),
                                                        project_constant);
    }
    return state.active_p_blocks.block.cols() > 0;
}

bool initial_iteration_step(LobpcgIteratorState& state,
                            const DenseOperator& apply_a,
                            const DenseOperator& apply_b,
                            bool project_constant) {
    if (!ortho_ab_mul_x(state.x_blocks, state.cholqr, apply_a, apply_b, project_constant)) {
        return false;
    }
    block_grams_1x1_step(state);
    if (!sub_problem_step(state, 0, 0)) {
        return false;
    }
    update_x_p_step(state, 0, 0);
    return true;
}

bool second_iteration_step(LobpcgIteratorState& state,
                           const DenseOperator& apply_a,
                           const DenseOperator& apply_b,
                           const BlockPreconditioner& apply_preconditioner,
                           bool project_constant) {
    update_active_r_blocks(state);
    precond_constr_step(state, apply_preconditioner);
    if (!ortho_active_r_step(state, apply_a, apply_b, project_constant)) {
        state.current_block_size = 0;
        return true;
    }

    const int block_size = static_cast<int>(state.active_r_blocks.block.cols());
    block_grams_2x2_step(state, block_size);
    if (!sub_problem_step(state, block_size, 0)) {
        return false;
    }
    update_x_p_step(state, block_size, 0);
    return true;
}

bool general_iteration_step(LobpcgIteratorState& state,
                            const DenseOperator& apply_a,
                            const DenseOperator& apply_b,
                            const BlockPreconditioner& apply_preconditioner,
                            bool project_constant) {
    update_active_rp_blocks(state);
    precond_constr_step(state, apply_preconditioner);
    if (!ortho_active_r_step(state, apply_a, apply_b, project_constant)) {
        state.current_block_size = 0;
        return true;
    }

    if (!ortho_active_p_step(state, apply_a, apply_b, project_constant)) {
        const int block_size = static_cast<int>(state.active_r_blocks.block.cols());
        block_grams_2x2_step(state, block_size);
        if (!sub_problem_step(state, block_size, 0)) {
            return false;
        }
        update_x_p_step(state, block_size, 0);
        return true;
    }

    const int block_size_r = static_cast<int>(state.active_r_blocks.block.cols());
    const int block_size_p = static_cast<int>(state.active_p_blocks.block.cols());
    block_grams_3x3_step(state, block_size_r, block_size_p);
    if (!sub_problem_step(state, block_size_r, block_size_p)) {
        return false;
    }
    update_x_p_step(state, block_size_r, block_size_p);
    return true;
}

bool lobpcg_iteration_step(LobpcgIteratorState& state,
                           const DenseOperator& apply_a,
                           const DenseOperator& apply_b,
                           const BlockPreconditioner& apply_preconditioner,
                           bool project_constant,
                           double tolerance) {
    bool success = false;
    if (state.iteration == 1) {
        success = initial_iteration_step(state, apply_a, apply_b, project_constant);
    } else if (state.iteration == 2) {
        success = second_iteration_step(state, apply_a, apply_b, apply_preconditioner,
                                        project_constant);
    } else {
        success = general_iteration_step(state, apply_a, apply_b, apply_preconditioner,
                                         project_constant);
    }
    if (!success) {
        return false;
    }
    if (state.iteration > 1 && state.current_block_size == 0 &&
        state.active_r_blocks.block.cols() == 0) {
        return true;
    }

    residuals_step(state);
    update_mask_step(state, tolerance);
    return true;
}

Eigen::MatrixXd solve_lobpcg_problem(const DenseOperator& apply_a,
                                     const DenseOperator& apply_b,
                                     const BlockPreconditioner& apply_preconditioner,
                                     int n,
                                     bool largest,
                                     int requested_dims,
                                     int iterations,
                                     int seed,
                                     bool project_constant,
                                     int skip_smallest,
                                     double tolerance,
                                     const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd()) {
    const int target_dims = requested_dims + std::max(0, skip_smallest);
    if (target_dims <= 0 || 3 * target_dims > n) {
        return Eigen::MatrixXd();
    }
    LobpcgIteratorState state = initialize_lobpcg_iterator(apply_a, apply_b, n, target_dims,
                                                           seed, largest, project_constant,
                                                           constraint_basis);
    if (size_x(state) < target_dims) {
        return Eigen::MatrixXd();
    }
    prepare_initial_lobpcg_block(state, seed);
    const int max_iterations = std::max(1, iterations);
    while (state.iteration <= max_iterations) {
        if (!lobpcg_iteration_step(state, apply_a, apply_b, apply_preconditioner,
                                   project_constant, tolerance)) {
            return Eigen::MatrixXd();
        }
        if (state.current_block_size == 0) {
            break;
        }
        state.iteration += 1;
    }

    Eigen::MatrixXd finalized =
        finalize_lobpcg_basis(apply_a, apply_b,
                              state.x_blocks.block.leftCols(
                                  std::min(target_dims, static_cast<int>(state.x_blocks.block.cols()))),
                              largest);
    if (finalized.cols() < target_dims) {
        return Eigen::MatrixXd();
    }
    return take_eigenvectors(finalized, largest, requested_dims, skip_smallest);
}

Eigen::MatrixXd solve_small_graph_problem(const WeightedGraph& graph,
                                          bool largest,
                                          int requested_dims,
                                          int iterations,
                                          int seed,
                                          const Eigen::MatrixXd& constraint_basis) {
    if (graph.num_vertices == 0 || requested_dims <= 0) {
        return Eigen::MatrixXd();
    }

    const SparseMatrix laplacian = sparse_graph_laplacian(graph);
    const DenseOperator apply_a = [&](const Eigen::MatrixXd& vectors) {
        return laplacian * vectors;
    };
    const DenseOperator apply_b = [&](const Eigen::MatrixXd& vectors) {
        return vectors;
    };
    const BlockPreconditioner apply_preconditioner = [&](const Eigen::MatrixXd& residuals) {
        return residuals;
    };

    Eigen::MatrixXd iterative =
        solve_lobpcg_problem(apply_a, apply_b, apply_preconditioner, graph.num_vertices,
                             largest, requested_dims, iterations, seed, false, largest ? 0 : 1,
                             std::pow(std::numeric_limits<double>::epsilon(), 3.0 / 10.0),
                             constraint_basis);
    if (iterative.cols() == requested_dims) {
        return iterative;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(dense_graph_laplacian(graph));
    if (solver.info() != Eigen::Success) {
        return Eigen::MatrixXd::Zero(
            graph.num_vertices, std::min(requested_dims, std::max(0, graph.num_vertices - 1)));
    }

    const int skip_smallest = largest ? 0 : 1;
    return take_eigenvectors(solver.eigenvectors(), largest, requested_dims, skip_smallest);
}

Eigen::MatrixXd solve_large_graph_problem_lobpcg(const Hypergraph& hypergraph,
                                                 const WeightedGraph& graph,
                                                 const PartitionIndex& pindex,
                                                 bool largest,
                                                 int requested_dims,
                                                 int iterations,
                                                 int epsilon,
                                                 int seed,
                                                 const Eigen::MatrixXd& constraint_basis) {
    const int n = hypergraph.num_vertices;
    const SparseMatrix laplacian = sparse_graph_laplacian(graph);
    const CmgHierarchy hierarchy = build_cmg_like_hierarchy(laplacian);

    Eigen::VectorXd weights(n);
    for (int vertex = 0; vertex < n; ++vertex) {
        weights(vertex) = static_cast<double>(hypergraph.vwts[vertex]);
    }
    const double total_weight = weights.sum();

    const DenseOperator apply_a = [&](const Eigen::MatrixXd& vectors) {
        return apply_hypergraph_operator(hypergraph, vectors, epsilon);
    };
    const DenseOperator apply_b = [&](const Eigen::MatrixXd& vectors) {
        Eigen::MatrixXd result = apply_clique_operator(weights, total_weight, vectors);
        if (!pindex.p1.empty() || !pindex.p2.empty()) {
            result += 500.0 * apply_biclique_operator(n, pindex, vectors);
        }
        return result;
    };

    const BlockPreconditioner apply_preconditioner = [&](const Eigen::MatrixXd& residuals) {
        return apply_cmg_preconditioner(hierarchy, residuals);
    };
    return solve_lobpcg_problem(apply_a, apply_b, apply_preconditioner, n,
                                largest, requested_dims, iterations, seed, true, 0,
                                kJuliaLargeLobpcgTolerance, constraint_basis);
}

Eigen::MatrixXd solve_dense_generalized_problem(const Hypergraph& hypergraph,
                                                const PartitionIndex& pindex,
                                                bool largest,
                                                int requested_dims,
                                                int epsilon) {
    const int n = hypergraph.num_vertices;
    Eigen::MatrixXd a_matrix = dense_hypergraph_operator(hypergraph, epsilon);
    Eigen::MatrixXd b_matrix = dense_clique_operator(hypergraph);
    if (!pindex.p1.empty() || !pindex.p2.empty()) {
        b_matrix += 500.0 * dense_biclique_operator(n, pindex);
    }

    const Eigen::MatrixXd q_matrix = helmert_complement_basis(n);
    Eigen::MatrixXd reduced_a = q_matrix.transpose() * a_matrix * q_matrix;
    Eigen::MatrixXd reduced_b = q_matrix.transpose() * b_matrix * q_matrix;
    symmetrize_in_place(reduced_a);
    symmetrize_in_place(reduced_b);
    reduced_b.diagonal().array() += 1e-9;

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(reduced_a, reduced_b);
    if (solver.info() != Eigen::Success) {
        return Eigen::MatrixXd();
    }

    const Eigen::MatrixXd reduced_vectors =
        take_eigenvectors(solver.eigenvectors(), largest, requested_dims, 0);
    return q_matrix * reduced_vectors;
}

}  // namespace

Eigen::MatrixXd solve_eigs(const Hypergraph& hypergraph,
                           const WeightedGraph& graph,
                           const PartitionIndex& pindex,
                           bool largest,
                           int requested_dims,
                           int iterations,
                           int epsilon,
                           int seed,
                           const Eigen::MatrixXd& constraint_basis) {
    const int n = hypergraph.num_vertices;
    if (n == 0 || requested_dims <= 0) {
        return Eigen::MatrixXd();
    }
    if (n == 1) {
        return Eigen::MatrixXd::Zero(1, 1);
    }

    requested_dims = std::min(requested_dims, n - 1);
    if (requested_dims <= 0) {
        return Eigen::MatrixXd::Zero(n, 0);
    }

    if (n < kSmallGraphThreshold) {
        return solve_small_graph_problem(graph, largest, requested_dims, iterations, seed,
                                         constraint_basis);
    }

    // Julia's large-graph path uses operator-based LOBPCG with a CMG preconditioner.
    // This C++ path now mirrors that shape more closely than the old dense generalized
    // eigensolve, while keeping the dense route as a safety fallback.
    Eigen::MatrixXd iterative =
        solve_large_graph_problem_lobpcg(
            hypergraph, graph, pindex, largest, requested_dims, iterations, epsilon, seed,
            constraint_basis);
    if (iterative.cols() == requested_dims) {
        return iterative;
    }

    Eigen::MatrixXd dense = solve_dense_generalized_problem(hypergraph, pindex, largest, requested_dims, epsilon);
    if (dense.cols() == requested_dims) {
        return dense;
    }

    return solve_small_graph_problem(graph, largest, requested_dims, iterations, seed,
                                     constraint_basis);
}

}  // namespace kspecpart
