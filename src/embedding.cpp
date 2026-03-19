#include "kspecpart/embedding.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace kspecpart {

namespace {

Eigen::MatrixXd multiply_normalized_adjacency(const WeightedGraph& graph, const Eigen::MatrixXd& x) {
    Eigen::MatrixXd y = Eigen::MatrixXd::Zero(graph.num_vertices, x.cols());
    for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
        const double dv = graph.degrees[vertex] > 0.0 ? std::sqrt(graph.degrees[vertex]) : 1.0;
        for (const auto& [neighbor, weight] : graph.adjacency[vertex]) {
            const double dn = graph.degrees[neighbor] > 0.0 ? std::sqrt(graph.degrees[neighbor]) : 1.0;
            y.row(vertex).noalias() += (weight / (dv * dn)) * x.row(neighbor);
        }
    }
    return y;
}

Eigen::MatrixXd normalized_adjacency_dense(const WeightedGraph& graph) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(graph.num_vertices, graph.num_vertices);
    for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
        const double dv = graph.degrees[vertex] > 0.0 ? std::sqrt(graph.degrees[vertex]) : 1.0;
        for (const auto& [neighbor, weight] : graph.adjacency[vertex]) {
            const double dn = graph.degrees[neighbor] > 0.0 ? std::sqrt(graph.degrees[neighbor]) : 1.0;
            matrix(vertex, neighbor) = weight / (dv * dn);
        }
    }
    return matrix;
}

}  // namespace

Eigen::MatrixXd leading_eigenvectors(const WeightedGraph& graph, int requested_dims, int iterations, std::mt19937& rng) {
    const int n = graph.num_vertices;
    if (n == 0 || requested_dims <= 0) {
        return Eigen::MatrixXd();
    }
    if (n == 1) {
        return Eigen::MatrixXd::Zero(1, 1);
    }

    const int basis_cols = std::min(n, std::max(2, requested_dims + 1));
    Eigen::MatrixXd basis;

    if (n <= 256) {
        const Eigen::MatrixXd dense = normalized_adjacency_dense(graph);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(dense);
        const Eigen::VectorXd eigenvalues = solver.eigenvalues();
        const Eigen::MatrixXd eigenvectors = solver.eigenvectors();
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(),
                  order.end(),
                  [&](int lhs, int rhs) { return eigenvalues[lhs] > eigenvalues[rhs]; });
        basis = Eigen::MatrixXd::Zero(n, basis_cols);
        for (int col = 0; col < basis_cols; ++col) {
            basis.col(col) = eigenvectors.col(order[col]);
        }
    } else {
        std::normal_distribution<double> normal(0.0, 1.0);
        Eigen::MatrixXd q = Eigen::MatrixXd::Zero(n, basis_cols);
        for (int col = 0; col < basis_cols; ++col) {
            for (int row = 0; row < n; ++row) {
                q(row, col) = normal(rng);
            }
        }
        for (int row = 0; row < n; ++row) {
            q(row, 0) = std::sqrt(std::max(1e-9, graph.degrees[row]));
        }
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(q);
        q = qr.householderQ() * Eigen::MatrixXd::Identity(n, basis_cols);

        for (int iter = 0; iter < std::max(8, iterations); ++iter) {
            Eigen::MatrixXd y = multiply_normalized_adjacency(graph, q);
            Eigen::HouseholderQR<Eigen::MatrixXd> iter_qr(y);
            q = iter_qr.householderQ() * Eigen::MatrixXd::Identity(n, basis_cols);
        }

        const Eigen::MatrixXd projected = q.transpose() * multiply_normalized_adjacency(graph, q);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(projected);
        const Eigen::VectorXd eigenvalues = solver.eigenvalues();
        const Eigen::MatrixXd eigenvectors = solver.eigenvectors();
        std::vector<int> order(basis_cols);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(),
                  order.end(),
                  [&](int lhs, int rhs) { return eigenvalues[lhs] > eigenvalues[rhs]; });
        basis = Eigen::MatrixXd::Zero(n, basis_cols);
        for (int col = 0; col < basis_cols; ++col) {
            basis.col(col) = q * eigenvectors.col(order[col]);
        }
    }

    const int dims = std::min(requested_dims, std::max(1, basis_cols - 1));
    Eigen::MatrixXd embedding = Eigen::MatrixXd::Zero(n, dims);
    for (int col = 0; col < dims; ++col) {
        embedding.col(col) = basis.col(col + 1);
    }
    return embedding;
}

}  // namespace kspecpart
