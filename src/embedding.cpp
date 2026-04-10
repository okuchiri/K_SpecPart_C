#include "kspecpart/embedding.hpp"
#include "kspecpart/lapack_eigen.hpp"

#include "kspecpart/cut_distillation.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace kspecpart {

namespace {

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using DenseOperator = std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>;
using BlockPreconditioner = std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>;

constexpr int kSmallGraphThreshold = 100;
constexpr int kDirectSolveThreshold = 500;
constexpr double kBGramTolerance = 1e-10;
constexpr double kJuliaLargeLobpcgTolerance = 1e-40;
constexpr int kJuliaXoshiroWidth = 8;
constexpr int kJuliaXoshiroBulkThresholdBytes = 64;
constexpr std::uint64_t kJuliaXoshiroForkMul0 = 0x02011ce34bce797fULL;
constexpr std::uint64_t kJuliaXoshiroForkMul1 = 0x5a94851fb48a6e05ULL;
constexpr std::uint64_t kJuliaXoshiroForkMul2 = 0x3688cf5d48899fa7ULL;
constexpr std::uint64_t kJuliaXoshiroForkMul3 = 0x867b4bb4c42e5661ULL;
// Julia's runtime CMG path calls cmg_preconditioner_lap(A) -> cmg_!(A, A)
// without validateInput!, so strict-dominance augmentation is kept opt-in here.
constexpr bool kEnableStrictDominanceAugmentOnMainPath = false;

bool cmg_debug_enabled() {
    const char* raw = std::getenv("K_SPECPART_DEBUG_CMG");
    return raw != nullptr && raw[0] != '\0' && std::string(raw) != "0";
}

std::optional<std::filesystem::path> resolve_embedding_probe_dir() {
    const char* raw = std::getenv("K_SPECPART_DEBUG_EMBED_OP_DIR");
    if (raw == nullptr || raw[0] == '\0') {
        return std::nullopt;
    }
    std::filesystem::path dir(raw);
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    if (error) {
        return std::nullopt;
    }
    return dir;
}

std::optional<std::filesystem::path> resolve_lobpcg_debug_dir() {
    const char* raw = std::getenv("K_SPECPART_DEBUG_LOBPCG_DIR");
    if (raw == nullptr || raw[0] == '\0') {
        return std::nullopt;
    }
    std::filesystem::path dir(raw);
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    if (error) {
        return std::nullopt;
    }
    return dir;
}

std::optional<std::filesystem::path> resolve_lobpcg_step_debug_dir() {
    const char* raw = std::getenv("K_SPECPART_DEBUG_LOBPCG_STEP_DIR");
    if (raw == nullptr || raw[0] == '\0') {
        return std::nullopt;
    }
    std::filesystem::path dir(raw);
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    if (error) {
        return std::nullopt;
    }
    return dir;
}

void write_dense_debug_matrix(const std::filesystem::path& path, const Eigen::MatrixXd& matrix) {
    std::ofstream output(path);
    output << std::setprecision(17);
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            if (col > 0) {
                output << ' ';
            }
            output << matrix(row, col);
        }
        output << '\n';
    }
}

void write_dense_debug_matrix_with_header(const std::filesystem::path& path,
                                         const Eigen::MatrixXd& matrix) {
    std::ofstream output(path);
    output.setf(std::ios::scientific);
    output.precision(17);
    output << matrix.rows() << ' ' << matrix.cols() << '\n';
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            if (col > 0) {
                output << ' ';
            }
            output << matrix(row, col);
        }
        output << '\n';
    }
}

void write_dense_debug_vector_with_header(const std::filesystem::path& path,
                                          const Eigen::VectorXd& vector) {
    std::ofstream output(path);
    output.setf(std::ios::scientific);
    output.precision(17);
    output << vector.size() << ' ' << 1 << '\n';
    for (int row = 0; row < vector.size(); ++row) {
        output << vector(row) << '\n';
    }
}

void write_index_debug_vector_with_header(const std::filesystem::path& path,
                                          const std::vector<int>& values) {
    std::ofstream output(path);
    output << values.size() << ' ' << 1 << '\n';
    for (int value : values) {
        output << value << '\n';
    }
}

void write_index_debug_vector(const std::filesystem::path& path, const std::vector<int>& values) {
    std::ofstream output(path);
    for (int value : values) {
        output << value << '\n';
    }
}

Eigen::MatrixXd make_embedding_probe_block(int n) {
    const int cols = (n > 1) ? 2 : 1;
    Eigen::MatrixXd probe(n, cols);
    for (int row = 0; row < n; ++row) {
        probe(row, 0) = static_cast<double>((row % 17) + 1) / 17.0;
        if (cols > 1) {
            probe(row, 1) =
                static_cast<double>(((row * 37) % 29) - 14) / 29.0;
        }
    }
    return probe;
}

void maybe_dump_embedding_operator_probe(const Hypergraph& hypergraph,
                                         const PartitionIndex& pindex,
                                         int epsilon,
                                         const DenseOperator& apply_a,
                                         const DenseOperator& apply_b,
                                         const BlockPreconditioner& apply_preconditioner) {
    const std::optional<std::filesystem::path> dir = resolve_embedding_probe_dir();
    if (!dir.has_value()) {
        return;
    }

    static int probe_id = 0;
    ++probe_id;
    const std::string prefix = "probe-" + std::to_string(probe_id);
    const Eigen::MatrixXd probe = make_embedding_probe_block(hypergraph.num_vertices);
    write_dense_debug_matrix(*dir / (prefix + ".X.txt"), probe);
    write_dense_debug_matrix(*dir / (prefix + ".A.txt"), apply_a(probe));
    write_dense_debug_matrix(*dir / (prefix + ".B.txt"), apply_b(probe));
    write_dense_debug_matrix(*dir / (prefix + ".P.txt"), apply_preconditioner(probe));
    write_index_debug_vector(*dir / (prefix + ".p1.txt"), pindex.p1);
    write_index_debug_vector(*dir / (prefix + ".p2.txt"), pindex.p2);

    std::ofstream meta(*dir / (prefix + ".meta.txt"));
    meta << "vertices " << hypergraph.num_vertices << '\n';
    meta << "hyperedges " << hypergraph.num_hyperedges << '\n';
    meta << "epsilon " << epsilon << '\n';
}

void maybe_dump_lobpcg_debug_matrix(const std::string& label, const Eigen::MatrixXd& matrix) {
    const std::optional<std::filesystem::path> dir = resolve_lobpcg_debug_dir();
    if (!dir.has_value()) {
        return;
    }

    static int dump_id = 0;
    ++dump_id;
    write_dense_debug_matrix(*dir / ("lobpcg-" + std::to_string(dump_id) + "." + label + ".txt"),
                             matrix);
}

bool lobpcg_step_debug_enabled(const std::string& debug_label) {
    return !debug_label.empty() && resolve_lobpcg_step_debug_dir().has_value();
}

void maybe_dump_lobpcg_step_matrix(const std::string& debug_label,
                                   const std::string& label,
                                   const Eigen::MatrixXd& matrix) {
    const std::optional<std::filesystem::path> dir = resolve_lobpcg_step_debug_dir();
    if (!dir.has_value() || debug_label.empty()) {
        return;
    }
    write_dense_debug_matrix_with_header(*dir / (debug_label + "." + label + ".txt"), matrix);
}

void maybe_dump_lobpcg_step_vector(const std::string& debug_label,
                                   const std::string& label,
                                   const Eigen::VectorXd& vector) {
    const std::optional<std::filesystem::path> dir = resolve_lobpcg_step_debug_dir();
    if (!dir.has_value() || debug_label.empty()) {
        return;
    }
    write_dense_debug_vector_with_header(*dir / (debug_label + "." + label + ".txt"), vector);
}

void maybe_dump_lobpcg_step_indices(const std::string& debug_label,
                                    const std::string& label,
                                    const std::vector<int>& values) {
    const std::optional<std::filesystem::path> dir = resolve_lobpcg_step_debug_dir();
    if (!dir.has_value() || debug_label.empty()) {
        return;
    }
    write_index_debug_vector_with_header(*dir / (debug_label + "." + label + ".txt"), values);
}

std::uint64_t rotl64(std::uint64_t value, int shift) {
    return (value << shift) | (value >> (64 - shift));
}

double julia_bits_to_float64(std::uint64_t value) {
    return static_cast<double>(value >> 11U) * 0x1.0p-53;
}

std::uint64_t xoshiro_next_from_state(std::uint64_t& s0,
                                      std::uint64_t& s1,
                                      std::uint64_t& s2,
                                      std::uint64_t& s3) {
    const std::uint64_t result = rotl64(s0 + s3, 23) + s0;
    const std::uint64_t t = s1 << 17U;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotl64(s3, 45);
    return result;
}

void fill_random_lobpcg_contiguous(double* data, int count, AlgorithmRng& rng) {
    if (data == nullptr || count <= 0) {
        return;
    }

    int offset = 0;
    if (count * static_cast<int>(sizeof(double)) >= kJuliaXoshiroBulkThresholdBytes) {
        std::array<std::uint64_t, kJuliaXoshiroWidth> s0{};
        std::array<std::uint64_t, kJuliaXoshiroWidth> s1{};
        std::array<std::uint64_t, kJuliaXoshiroWidth> s2{};
        std::array<std::uint64_t, kJuliaXoshiroWidth> s3{};

        for (int lane = 0; lane < kJuliaXoshiroWidth; ++lane) {
            s0[static_cast<std::size_t>(lane)] = kJuliaXoshiroForkMul0 * rng.next_u64();
        }
        for (int lane = 0; lane < kJuliaXoshiroWidth; ++lane) {
            s1[static_cast<std::size_t>(lane)] = kJuliaXoshiroForkMul1 * rng.next_u64();
        }
        for (int lane = 0; lane < kJuliaXoshiroWidth; ++lane) {
            s2[static_cast<std::size_t>(lane)] = kJuliaXoshiroForkMul2 * rng.next_u64();
        }
        for (int lane = 0; lane < kJuliaXoshiroWidth; ++lane) {
            s3[static_cast<std::size_t>(lane)] = kJuliaXoshiroForkMul3 * rng.next_u64();
        }

        while (offset + kJuliaXoshiroWidth <= count) {
            for (int lane = 0; lane < kJuliaXoshiroWidth; ++lane) {
                const std::size_t idx = static_cast<std::size_t>(lane);
                const std::uint64_t result =
                    xoshiro_next_from_state(s0[idx], s1[idx], s2[idx], s3[idx]);
                data[offset + lane] = julia_bits_to_float64(result);
            }
            offset += kJuliaXoshiroWidth;
        }
    }

    while (offset < count) {
        data[offset] = rng.next_float64();
        ++offset;
    }
}

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

void emit_cmg_hierarchy_debug(const CmgHierarchy& hierarchy) {
    if (!cmg_debug_enabled()) {
        return;
    }
    std::cout << "[cmg-debug] levels=" << hierarchy.levels.size()
              << " strict_dominant=" << (hierarchy.strict_dominant ? 1 : 0)
              << " input_size=" << hierarchy.input_size << '\n';
    for (int level = 0; level < static_cast<int>(hierarchy.levels.size()); ++level) {
        const CmgLevel& info = hierarchy.levels[static_cast<std::size_t>(level)];
        std::cout << "[cmg-debug] level=" << level
                  << " n=" << info.num_vertices
                  << " nc=" << info.num_coarse
                  << " nnz=" << info.num_nonzeros
                  << " is_last=" << (info.is_last ? 1 : 0)
                  << " iterative=" << (info.iterative ? 1 : 0)
                  << " sd=" << (info.sd ? 1 : 0)
                  << '\n';
    }
}

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

struct LobpcgState {
    int iteration = 0;
    Eigen::VectorXd residual_norms;
    Eigen::VectorXd ritz_values;
    int current_block_size = 0;
    int active_r_cols = 0;
    int active_p_cols = 0;
    bool used_three_term = false;
    double r_eig_norm = 0.0;
    double p_eig_norm = 0.0;
    double p_block_norm = 0.0;
    double xar_norm = 0.0;
    double xbr_norm = 0.0;
    double rar_norm = 0.0;
};

using LobpcgTrace = std::vector<LobpcgState>;

struct LobpcgResults {
    Eigen::VectorXd lambda;
    Eigen::MatrixXd x;
    double tolerance = 0.0;
    Eigen::VectorXd residual_norms;
    int iterations = 0;
    int max_iterations = 0;
    bool converged = false;
    bool success = false;
    int block_size = 0;
    LobpcgTrace trace;
    std::vector<int> batch_iterations;
    std::vector<char> converged_mask;
    std::vector<LobpcgTrace> batch_traces;
    std::string failure_reason;
    int failure_iteration = 0;
};

struct LobpcgIterator {
    DenseOperator apply_a;
    DenseOperator apply_b;
    BlockPreconditioner apply_preconditioner;
    bool generalized = true;
    bool largest = false;
    bool project_constant = false;
    double tolerance = 0.0;
    int max_iterations = 0;
    int seed = 0;
    AlgorithmRng* shared_rng = nullptr;
    bool log_trace = false;
    std::string debug_label;
    std::string failure_stage;
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
    int last_active_r_cols = 0;
    int last_active_p_cols = 0;
    bool last_used_three_term = false;
    double last_r_eig_norm = 0.0;
    double last_p_eig_norm = 0.0;
    double last_p_block_norm = 0.0;
    double last_xar_norm = 0.0;
    double last_xbr_norm = 0.0;
    double last_rar_norm = 0.0;
    LobpcgTrace trace;
};

Eigen::MatrixXd dense_graph_laplacian(const WeightedGraph& graph) {
    Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(graph.num_vertices, graph.num_vertices);
    for (int vertex = 0; vertex < graph.num_vertices; ++vertex) {
        laplacian(vertex, vertex) +=
            static_cast<double>(graph.adjacency[vertex].size()) + 1e-6;
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
        triplets.emplace_back(vertex, vertex,
                              static_cast<double>(graph.adjacency[vertex].size()) + 1e-6);
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

Eigen::VectorXd julia_like_inv_diag(const Eigen::VectorXd& diagonal) {
    Eigen::VectorXd inv_diag(diagonal.size());
    for (int idx = 0; idx < diagonal.size(); ++idx) {
        const double value = diagonal(idx);
        inv_diag(idx) = value != 0.0 ? 1.0 / (2.0 * value) : 0.0;
    }
    return inv_diag;
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
        const double diag = diagonal(idx);
        const double ratio =
            diag != 0.0 ? positive_row_sum(idx) / diag
                        : (positive_row_sum(idx) > 0.0
                               ? std::numeric_limits<double>::infinity()
                               : 0.0);
        if (ratio > 1e-13) {
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

std::optional<RitzPair> solve_generalized_hermitian_whitened(Eigen::MatrixXd small_a,
                                                             Eigen::MatrixXd small_b) {
    if (small_a.rows() == 0 || small_a.rows() != small_a.cols() ||
        small_b.rows() != small_b.cols() || small_a.rows() != small_b.rows() ||
        small_a.rows() == 0) {
        return std::nullopt;
    }

    symmetrize_in_place(small_a);
    symmetrize_in_place(small_b);

    std::optional<LapackEigenResult> b_solver = lapack_symmetric_eigen(small_b);
    if (!b_solver.has_value()) {
        return std::nullopt;
    }

    const double max_b =
        (b_solver->eigenvalues.size() > 0) ? b_solver->eigenvalues.cwiseAbs().maxCoeff() : 0.0;
    const double keep_tolerance =
        std::max(kBGramTolerance, max_b * 1e-12);

    std::vector<int> keep;
    for (int index = 0; index < b_solver->eigenvalues.size(); ++index) {
        if (b_solver->eigenvalues(index) > keep_tolerance) {
            keep.push_back(index);
        }
    }
    if (keep.empty()) {
        return std::nullopt;
    }

    Eigen::MatrixXd transform(small_b.rows(), static_cast<int>(keep.size()));
    for (int col = 0; col < static_cast<int>(keep.size()); ++col) {
        const int index = keep[static_cast<std::size_t>(col)];
        transform.col(col) = b_solver->eigenvectors.col(index) /
                             std::sqrt(b_solver->eigenvalues(index));
    }

    Eigen::MatrixXd reduced_a = transform.transpose() * small_a * transform;
    symmetrize_in_place(reduced_a);

    std::optional<LapackEigenResult> a_solver = lapack_symmetric_eigen(reduced_a);
    if (!a_solver.has_value()) {
        return std::nullopt;
    }

    RitzPair pair;
    pair.eigenvalues = a_solver->eigenvalues;
    pair.eigenvectors = transform * a_solver->eigenvectors;
    return pair;
}

double eigenvalue_selection_tolerance(const Eigen::VectorXd& values, int count) {
    if (count <= 0 || values.size() == 0) {
        return 0.0;
    }
    const int bounded = std::min<int>(count, values.size());
    double max_abs = 0.0;
    for (int index = 0; index < bounded; ++index) {
        max_abs = std::max(max_abs, std::abs(values(index)));
    }
    return std::max(1.0, max_abs) * 1e-12;
}

void sort_eigenpair_permutation(std::vector<int>& permutation,
                                const Eigen::VectorXd& values,
                                int count,
                                bool largest) {
    const int bounded = std::max(0, std::min<int>(count, values.size()));
    permutation.resize(bounded);
    std::iota(permutation.begin(), permutation.end(), 0);
    const double tolerance = eigenvalue_selection_tolerance(values, bounded);
    std::stable_sort(permutation.begin(), permutation.end(), [&](int lhs, int rhs) {
        const double delta = values(lhs) - values(rhs);
        if (std::abs(delta) > tolerance) {
            return largest ? delta > 0.0 : delta < 0.0;
        }
        return lhs < rhs;
    });
}

std::optional<RitzPair> solve_generalized_hermitian_cholesky(Eigen::MatrixXd small_a,
                                                             Eigen::MatrixXd small_b) {
    if (small_a.rows() == 0 || small_a.rows() != small_a.cols() ||
        small_b.rows() != small_b.cols() || small_a.rows() != small_b.rows() ||
        small_a.rows() == 0) {
        return std::nullopt;
    }

    symmetrize_in_place(small_a);
    symmetrize_in_place(small_b);
    small_b.diagonal() = small_b.diagonal().real();

    Eigen::LLT<Eigen::MatrixXd> chol(small_b);
    if (chol.info() != Eigen::Success) {
        return std::nullopt;
    }
    const Eigen::MatrixXd upper = chol.matrixU();
    const Eigen::ArrayXd abs_diag = upper.diagonal().array().abs();
    const double max_diag = (abs_diag.size() > 0) ? abs_diag.maxCoeff() : 0.0;
    const double min_diag = (abs_diag.size() > 0) ? abs_diag.minCoeff() : 0.0;
    if ((abs_diag <= std::sqrt(kBGramTolerance)).any() ||
        min_diag <= max_diag * 1e-10) {
        return std::nullopt;
    }

    Eigen::MatrixXd transform = Eigen::MatrixXd::Identity(small_b.rows(), small_b.cols());
    upper.template triangularView<Eigen::Upper>().solveInPlace(transform);
    Eigen::MatrixXd reduced_a = transform.transpose() * small_a * transform;
    symmetrize_in_place(reduced_a);

    std::optional<LapackEigenResult> solver = lapack_symmetric_eigen(reduced_a);
    if (!solver.has_value()) {
        return std::nullopt;
    }

    RitzPair pair;
    pair.eigenvalues = solver->eigenvalues;
    pair.eigenvectors = transform * solver->eigenvectors;
    return pair;
}

std::optional<RitzPair> solve_generalized_hermitian_direct(Eigen::MatrixXd small_a,
                                                           Eigen::MatrixXd small_b) {
    if (small_a.rows() == 0 || small_a.rows() != small_a.cols() ||
        small_b.rows() != small_b.cols() || small_a.rows() != small_b.rows()) {
        return std::nullopt;
    }

    symmetrize_in_place(small_a);
    symmetrize_in_place(small_b);
    small_b.diagonal() = small_b.diagonal().real();

    std::optional<LapackEigenResult> solver =
        lapack_generalized_symmetric_eigen(std::move(small_a), std::move(small_b));
    if (!solver.has_value()) {
        return std::nullopt;
    }

    RitzPair pair;
    pair.eigenvalues = solver->eigenvalues;
    pair.eigenvectors = solver->eigenvectors;
    return pair;
}

std::optional<RitzPair> select_extremal_ritz_pairs(const RitzPair& full_pair,
                                                   bool largest,
                                                   int requested_dims) {
    const int available =
        std::min<int>(full_pair.eigenvalues.size(), full_pair.eigenvectors.cols());
    if (requested_dims <= 0 || available < requested_dims ||
        full_pair.eigenvectors.rows() == 0) {
        return std::nullopt;
    }

    RitzPair selected;
    selected.eigenvalues.resize(requested_dims);
    selected.eigenvectors.resize(full_pair.eigenvectors.rows(), requested_dims);
    std::vector<int> permutation;
    sort_eigenpair_permutation(permutation, full_pair.eigenvalues, available, largest);
    for (int col = 0; col < requested_dims; ++col) {
        const int index = permutation[static_cast<std::size_t>(col)];
        selected.eigenvalues(col) = full_pair.eigenvalues(index);
        selected.eigenvectors.col(col) = full_pair.eigenvectors.col(index);
    }
    return selected;
}

std::optional<RitzPair> solve_generalized_hermitian_full(Eigen::MatrixXd small_a,
                                                         Eigen::MatrixXd small_b) {
    std::optional<RitzPair> pair =
        solve_generalized_hermitian_direct(small_a, small_b);
    if (pair.has_value()) {
        return pair;
    }

    pair = solve_generalized_hermitian_cholesky(small_a, small_b);
    if (pair.has_value()) {
        return pair;
    }
    return solve_generalized_hermitian_whitened(std::move(small_a), std::move(small_b));
}

MinSparseResult find_min_sparse(const SparseMatrix& matrix) {
    const int n = matrix.cols();
    MinSparseResult result;
    result.min_cols.resize(n, 0);
    result.min_vals = Eigen::VectorXd::Zero(n);

    // Julia's `findMinSparse` walks CSC columns. This C++ port keeps the main
    // sparse type RowMajor for other helpers, so we must accumulate minima by
    // column explicitly instead of treating the outer index as a column id.
    std::vector<char> found(static_cast<std::size_t>(n), 0);
    std::vector<double> min_values(static_cast<std::size_t>(n),
                                   std::numeric_limits<double>::infinity());
    std::vector<int> min_rows(static_cast<std::size_t>(n), 0);

    for (int row = 0; row < matrix.rows(); ++row) {
        for (SparseMatrix::InnerIterator it(matrix, row); it; ++it) {
            const int col = it.col();
            const double value = it.value();
            if (!found[static_cast<std::size_t>(col)] ||
                value < min_values[static_cast<std::size_t>(col)] ||
                (value == min_values[static_cast<std::size_t>(col)] &&
                 row < min_rows[static_cast<std::size_t>(col)])) {
                found[static_cast<std::size_t>(col)] = 1;
                min_values[static_cast<std::size_t>(col)] = value;
                min_rows[static_cast<std::size_t>(col)] = row;
            }
        }
    }

    for (int col = 0; col < n; ++col) {
        result.min_cols[col] =
            found[static_cast<std::size_t>(col)] ? min_rows[static_cast<std::size_t>(col)] : col;
        result.min_vals(col) = found[static_cast<std::size_t>(col)]
                                   ? min_values[static_cast<std::size_t>(col)]
                                   : 0.0;
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

ForestComponentsResult normalize_forest_components(ForestComponentsResult result) {
    const int n = static_cast<int>(result.component_index.size());
    int max_component = -1;
    for (int vertex = 0; vertex < n; ++vertex) {
        max_component = std::max(max_component, result.component_index[vertex]);
    }

    std::vector<int> remap(std::max(0, max_component + 1), -1);
    int next_component = 0;
    for (int vertex = 0; vertex < n; ++vertex) {
        const int component = result.component_index[vertex];
        if (component < 0) {
            continue;
        }
        if (remap[component] < 0) {
            remap[component] = next_component;
            next_component += 1;
        }
        result.component_index[vertex] = remap[component];
    }

    for (int vertex = 0; vertex < n; ++vertex) {
        if (result.component_index[vertex] >= 0) {
            continue;
        }
        result.component_index[vertex] = next_component;
        next_component += 1;
    }

    result.num_components = next_component;
    result.component_sizes.assign(result.num_components, 0);
    for (int vertex = 0; vertex < n; ++vertex) {
        const int component = result.component_index[vertex];
        if (component >= 0 && component < result.num_components) {
            result.component_sizes[component] += 1;
        }
    }
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

    return normalize_forest_components(forest_components(parents));
}

SparseMatrix build_prolongation(const std::vector<int>& aggregate, int coarse_vertices) {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(aggregate.size());
    for (int vertex = 0; vertex < static_cast<int>(aggregate.size()); ++vertex) {
        if (aggregate[vertex] < 0 || aggregate[vertex] >= coarse_vertices) {
            continue;
        }
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

CmgHierarchy build_cmg_like_hierarchy(const SparseMatrix& laplacian,
                                      bool enable_strict_dominance_augment) {
    CmgHierarchy hierarchy;
    hierarchy.input_size = laplacian.rows();
    hierarchy.strict_dominant = false;

    SparseMatrix current = laplacian;
    bool sd = false;
    int sflag = 1;
    if (enable_strict_dominance_augment) {
        const CmgValidationResult validation = validate_cmg_input(laplacian);
        hierarchy.input_size = validation.original_size;
        hierarchy.strict_dominant = validation.augmented;
        current = validation.matrix;
        sd = validation.augmented;
    }

    const int original_nnz = std::max(1, static_cast<int>(laplacian.nonZeros()));
    int hierarchy_nnz = 0;
    int flag = 0;

    while (true) {
        const int n = current.rows();
        if (current.rows() < kDirectSolveThreshold) {
            break;
        }

        const Eigen::VectorXd diagonal = current.diagonal();
        const ForestComponentsResult groups = steiner_group(current, diagonal);
        const int coarse_vertices = groups.num_components;
        bool iterative = true;
        bool is_last = false;

        if (coarse_vertices == 1) {
            is_last = true;
            iterative = true;
            flag = 1;
        }

        hierarchy_nnz += current.nonZeros();
        const bool stagnated = coarse_vertices >= n - 1 || hierarchy_nnz > 5 * original_nnz;
        if (stagnated) {
            flag = 3;
            break;
        }

        CmgLevel cmg_level;
        cmg_level.matrix = current;
        cmg_level.inv_diag = julia_like_inv_diag(diagonal);
        cmg_level.coarse_index = groups.component_index;
        cmg_level.num_coarse = coarse_vertices;
        cmg_level.num_vertices = n;
        cmg_level.num_nonzeros = current.nonZeros();
        cmg_level.sd = sd;
        cmg_level.prolongation =
            build_prolongation(groups.component_index, coarse_vertices);
        cmg_level.is_last = is_last;
        cmg_level.iterative = iterative;
        hierarchy.levels.push_back(cmg_level);

        SparseMatrix coarse = cmg_level.prolongation.transpose() * current * cmg_level.prolongation;
        coarse.makeCompressed();
        current = 0.5 * (coarse + SparseMatrix(coarse.transpose()));
        current.makeCompressed();

        if (sflag == 1) {
            sd = true;
            sflag = 0;
        }
        if (coarse_vertices == 1) {
            break;
        }
    }

    if (flag == 0) {
        hierarchy.levels.push_back(build_direct_last_level(current, true));
    }

    if (hierarchy.levels.empty()) {
        // Julia's `cmg_!` can break on a first-level stagnation before pushing
        // any hierarchy level. We keep a single Jacobi-like rescue level here
        // so the C++ runtime still has a usable preconditioner state.
        CmgLevel level;
        level.matrix = current;
        level.inv_diag = julia_like_inv_diag(current.diagonal());
        level.num_vertices = current.rows();
        level.num_nonzeros = current.nonZeros();
        level.sd = sd;
        level.is_last = true;
        level.iterative = true;
        hierarchy.levels.push_back(std::move(level));
    }

    emit_cmg_hierarchy_debug(hierarchy);
    return hierarchy;
}

std::vector<CmgLevelAux> init_cmg_level_aux(const CmgHierarchy& hierarchy) {
    std::vector<CmgLevelAux> aux(hierarchy.levels.size());
    for (int level = 0; level < static_cast<int>(hierarchy.levels.size()); ++level) {
        int repeat = 1;
        if (level == 0) {
            repeat = 1;
        } else if (level == static_cast<int>(hierarchy.levels.size()) - 1) {
            repeat = 0;
        } else {
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

Eigen::VectorXd apply_cmg_preconditioner_core_vector(const CmgHierarchy& hierarchy,
                                                     CmgPreparedState& prepared,
                                                     const Eigen::VectorXd& rhs) {
    if (hierarchy.levels.empty()) {
        return rhs;
    }
    if (prepared.workspaces.empty()) {
        return rhs;
    }

    reset_cmg_preconditioner_state(prepared);
    prepared.workspaces.front().b.head(rhs.size()) = rhs;

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
    if (result.size() == rhs.size()) {
        return result;
    }
    return result.head(std::min<int>(rhs.size(), result.size()));
}

Eigen::VectorXd apply_cmg_preconditioner_sd_vector(const CmgHierarchy& hierarchy,
                                                   CmgPreparedState& prepared,
                                                   const Eigen::VectorXd& rhs) {
    if (hierarchy.levels.empty()) {
        return rhs;
    }

    const int augmented_size = hierarchy.levels.front().num_vertices;
    if (augmented_size != rhs.size() + 1) {
        return apply_cmg_preconditioner_core_vector(hierarchy, prepared, rhs);
    }

    Eigen::VectorXd augmented_rhs(rhs.size() + 1);
    augmented_rhs.head(rhs.size()) = rhs;
    augmented_rhs(rhs.size()) = -rhs.sum();

    Eigen::VectorXd augmented_solution =
        apply_cmg_preconditioner_core_vector(hierarchy, prepared, augmented_rhs);
    if (augmented_solution.size() != augmented_rhs.size()) {
        return augmented_solution.head(std::min<int>(rhs.size(), augmented_solution.size()));
    }

    return (augmented_solution.head(rhs.size()).array() +
            augmented_solution(rhs.size())).matrix();
}

Eigen::VectorXd apply_cmg_preconditioner_vector(const CmgHierarchy& hierarchy,
                                                CmgPreparedState& prepared,
                                                const Eigen::VectorXd& rhs) {
    if (hierarchy.strict_dominant &&
        !hierarchy.levels.empty() &&
        hierarchy.levels.front().num_vertices == rhs.size() + 1) {
        return apply_cmg_preconditioner_sd_vector(hierarchy, prepared, rhs);
    }
    return apply_cmg_preconditioner_core_vector(hierarchy, prepared, rhs);
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

bool cholqr_orthonormalize_standard_in_place(Eigen::MatrixXd& vectors,
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

    gram_workspace.resize(vectors.cols(), vectors.cols());
    gram_workspace.noalias() = vectors.transpose() * vectors;
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
        *b_vectors = vectors;
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

[[maybe_unused]] std::optional<RitzPair> solve_projected_eigenproblem(
    const Eigen::MatrixXd& basis,
    const Eigen::MatrixXd& a_basis,
    const Eigen::MatrixXd& b_basis,
    bool largest,
    int requested_dims) {
    if (basis.cols() == 0 || requested_dims <= 0) {
        return std::nullopt;
    }

    Eigen::MatrixXd small_a = basis.transpose() * a_basis;
    Eigen::MatrixXd small_b = basis.transpose() * b_basis;
    std::optional<RitzPair> full_pair =
        solve_generalized_hermitian_whitened(std::move(small_a), std::move(small_b));
    if (!full_pair.has_value()) {
        return std::nullopt;
    }
    return select_extremal_ritz_pairs(*full_pair, largest,
                                      std::min(requested_dims, static_cast<int>(basis.cols())));
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

int size_x(const LobpcgIterator& iterator) {
    return static_cast<int>(iterator.x_blocks.block.cols());
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

bool rebuild_constraint(LobpcgConstraint& constraint, int block_cols) {
    if (constraint.y.cols() == 0) {
        constraint.enabled = false;
        constraint.by.resize(constraint.y.rows(), 0);
        constraint.gram_ybv = Eigen::MatrixXd::Zero(0, block_cols);
        constraint.tmp = Eigen::MatrixXd::Zero(0, block_cols);
        return true;
    }

    Eigen::MatrixXd gram = constraint.y.transpose() * constraint.by;
    symmetrize_in_place(gram);
    constraint.gram_chol.compute(gram);
    if (constraint.gram_chol.info() != Eigen::Success) {
        constraint.enabled = false;
        return false;
    }
    constraint.enabled = true;
    constraint.gram_ybv = Eigen::MatrixXd::Zero(constraint.y.cols(), block_cols);
    constraint.tmp = Eigen::MatrixXd::Zero(constraint.y.cols(), block_cols);
    return true;
}

bool append_constraint_columns(LobpcgConstraint& constraint,
                               const Eigen::MatrixXd& vectors,
                               const Eigen::MatrixXd& b_vectors,
                               int block_cols) {
    if (vectors.cols() == 0) {
        return true;
    }
    if (vectors.rows() != b_vectors.rows() || vectors.cols() != b_vectors.cols()) {
        return false;
    }

    if (constraint.y.cols() == 0) {
        constraint.y = vectors;
        constraint.by = b_vectors;
        return rebuild_constraint(constraint, block_cols);
    }

    if (constraint.y.rows() != vectors.rows() || constraint.by.rows() != b_vectors.rows()) {
        return false;
    }

    Eigen::MatrixXd combined_y(constraint.y.rows(), constraint.y.cols() + vectors.cols());
    combined_y.block(0, 0, constraint.y.rows(), constraint.y.cols()) = constraint.y;
    combined_y.block(0, constraint.y.cols(), vectors.rows(), vectors.cols()) = vectors;

    Eigen::MatrixXd combined_by(constraint.by.rows(), constraint.by.cols() + b_vectors.cols());
    combined_by.block(0, 0, constraint.by.rows(), constraint.by.cols()) = constraint.by;
    combined_by.block(0, constraint.by.cols(), b_vectors.rows(), b_vectors.cols()) = b_vectors;

    constraint.y = std::move(combined_y);
    constraint.by = std::move(combined_by);
    return rebuild_constraint(constraint, block_cols);
}

void update_active_block(const std::vector<char>& mask,
                         int block_size,
                         Eigen::MatrixXd& active_block,
                         const Eigen::MatrixXd& block) {
    active_block = gather_columns_by_mask(block, mask, block_size);
}

void update_active_r_blocks(LobpcgIterator& iterator) {
    const int block_size = iterator.current_block_size;
    update_active_block(iterator.active_mask, block_size, iterator.active_r_blocks.block, iterator.r_blocks.block);
    iterator.active_r_blocks.a_block.resize(iterator.active_r_blocks.block.rows(), 0);
    iterator.active_r_blocks.b_block.resize(iterator.active_r_blocks.block.rows(), 0);
}

void update_active_rp_blocks(LobpcgIterator& iterator) {
    const int block_size = iterator.current_block_size;
    update_active_r_blocks(iterator);
    update_active_block(iterator.active_mask, block_size, iterator.active_p_blocks.block, iterator.p_blocks.block);
    update_active_block(iterator.active_mask, block_size, iterator.active_p_blocks.a_block, iterator.p_blocks.a_block);
    update_active_block(iterator.active_mask, block_size, iterator.active_p_blocks.b_block, iterator.p_blocks.b_block);
}

void update_block_products(LobpcgBlocks& blocks,
                           const DenseOperator& apply_a,
                           const DenseOperator& apply_b,
                           bool generalized) {
    blocks.a_block = apply_a(blocks.block);
    blocks.b_block = generalized ? apply_b(blocks.block) : blocks.block;
}

LobpcgBlocks b_orthonormalize_blocks(const DenseOperator& apply_a,
                                     const DenseOperator& apply_b,
                                     LobpcgCholQr& cholqr,
                                     LobpcgBlocks blocks,
                                     bool project_constant,
                                     bool generalized) {
    if (blocks.block.cols() == 0) {
        return make_empty_blocks(blocks.block.rows(), 0);
    }

    if (generalized) {
        if (!cholqr_orthonormalize_in_place(apply_b, blocks.block, nullptr, nullptr,
                                            cholqr.gram_vbv, project_constant)) {
            blocks.block =
                eigen_b_orthonormalize(apply_b, std::move(blocks.block), project_constant);
        }
    } else if (!cholqr_orthonormalize_standard_in_place(blocks.block, nullptr, nullptr,
                                                        cholqr.gram_vbv, project_constant)) {
        return make_empty_blocks(blocks.block.rows(), 0);
    }
    if (blocks.block.cols() == 0) {
        return make_empty_blocks(blocks.block.rows(), 0);
    }
    update_block_products(blocks, apply_a, apply_b, generalized);
    return blocks;
}

bool ortho_ab_mul_x(LobpcgBlocks& blocks,
                    LobpcgCholQr& cholqr,
                    const DenseOperator& apply_a,
                    const DenseOperator& apply_b,
                    bool project_constant,
                    bool generalized) {
    blocks =
        b_orthonormalize_blocks(apply_a, apply_b, cholqr, std::move(blocks), project_constant,
                                generalized);
    return blocks.block.cols() > 0;
}

void residuals_step(LobpcgIterator& iterator);
void update_mask_step(LobpcgIterator& iterator);
void precond_constr_step(LobpcgIterator& iterator);
void block_grams_1x1_step(LobpcgIterator& iterator);
void block_grams_2x2_step(LobpcgIterator& iterator, int block_size);
void sort_selected_permutation(std::vector<int>& permutation,
                               const Eigen::VectorXd& values,
                               int subdim,
                               bool largest);
bool update_selected_ritz_pairs(LobpcgIterator& iterator,
                                const Eigen::VectorXd& values,
                                const Eigen::MatrixXd& vectors,
                                int basis_rows);
void update_x_p_step(LobpcgIterator& iterator, int block_size_r, int block_size_p);

std::vector<int> active_mask_as_indices(const std::vector<char>& active_mask) {
    std::vector<int> values;
    values.reserve(active_mask.size());
    for (char value : active_mask) {
        values.push_back(value == 0 ? 0 : 1);
    }
    return values;
}

bool debug_run_cholqr_probe(LobpcgIterator& iterator,
                            LobpcgBlocks& blocks,
                            const std::string& phase_label) {
    if (blocks.block.cols() == 0) {
        return true;
    }

    maybe_dump_lobpcg_step_matrix(iterator.debug_label, phase_label + ".X_pre", blocks.block);
    if (iterator.project_constant) {
        project_zero_sum(blocks.block);
    }

    blocks.b_block = iterator.generalized ? iterator.apply_b(blocks.block) : blocks.block;
    maybe_dump_lobpcg_step_matrix(iterator.debug_label, phase_label + ".B_pre_ortho", blocks.b_block);

    iterator.cholqr.gram_vbv.resize(blocks.block.cols(), blocks.block.cols());
    iterator.cholqr.gram_vbv.noalias() = blocks.block.transpose() * blocks.b_block;
    symmetrize_in_place(iterator.cholqr.gram_vbv);
    iterator.cholqr.gram_vbv.diagonal() = iterator.cholqr.gram_vbv.diagonal().real();
    maybe_dump_lobpcg_step_matrix(iterator.debug_label,
                                  phase_label + ".gramVBV_pre_chol",
                                  iterator.cholqr.gram_vbv);

    Eigen::LLT<Eigen::MatrixXd> chol(iterator.cholqr.gram_vbv);
    if (chol.info() != Eigen::Success) {
        return false;
    }
    const Eigen::MatrixXd upper = chol.matrixU();
    maybe_dump_lobpcg_step_matrix(iterator.debug_label, phase_label + ".cholR", upper);
    if ((upper.diagonal().array().abs() <= std::sqrt(kBGramTolerance)).any()) {
        return false;
    }

    right_divide_upper_triangular_in_place(blocks.block, upper);
    if (iterator.generalized) {
        right_divide_upper_triangular_in_place(blocks.b_block, upper);
    } else {
        blocks.b_block = blocks.block;
    }
    maybe_dump_lobpcg_step_matrix(iterator.debug_label, phase_label + ".X_post_ortho", blocks.block);
    maybe_dump_lobpcg_step_matrix(iterator.debug_label, phase_label + ".B_post_ortho", blocks.b_block);

    blocks.a_block = iterator.apply_a(blocks.block);
    maybe_dump_lobpcg_step_matrix(iterator.debug_label, phase_label + ".A_post_ortho", blocks.a_block);
    return true;
}

bool debug_dump_initial_subproblem_probe(LobpcgIterator& iterator) {
    const int size_x_value = size_x(iterator);
    block_grams_1x1_step(iterator);
    maybe_dump_lobpcg_step_matrix(iterator.debug_label,
                                  "iter1.rr_1x1.gramA",
                                  iterator.gram_a_block.xax.topLeftCorner(size_x_value, size_x_value));

    Eigen::MatrixXd gram_a_view = iterator.gram_a_block.xax.topLeftCorner(size_x_value, size_x_value);
    symmetrize_in_place(gram_a_view);
    std::optional<LapackEigenResult> solver = lapack_symmetric_eigen(std::move(gram_a_view));
    if (!solver.has_value()) {
        return false;
    }

    maybe_dump_lobpcg_step_vector(iterator.debug_label, "iter1.rr_1x1.eigvals", solver->eigenvalues);
    maybe_dump_lobpcg_step_matrix(iterator.debug_label, "iter1.rr_1x1.eigvecs", solver->eigenvectors);
    sort_selected_permutation(iterator.lambda_perm, solver->eigenvalues, size_x_value, iterator.largest);
    maybe_dump_lobpcg_step_indices(iterator.debug_label,
                                   "iter1.rr_1x1.perm",
                                   std::vector<int>(iterator.lambda_perm.begin(),
                                                    iterator.lambda_perm.begin() + size_x_value));
    if (!update_selected_ritz_pairs(iterator, solver->eigenvalues, solver->eigenvectors, size_x_value)) {
        return false;
    }
    maybe_dump_lobpcg_step_vector(iterator.debug_label, "iter1.rr_1x1.ritz_values", iterator.ritz_values.head(size_x_value));
    maybe_dump_lobpcg_step_matrix(iterator.debug_label,
                                  "iter1.rr_1x1.selected_vectors",
                                  iterator.v_basis.topLeftCorner(size_x_value, size_x_value));
    return true;
}

bool debug_dump_second_subproblem_probe(LobpcgIterator& iterator, int block_size) {
    const int size_x_value = size_x(iterator);
    const int subdim = size_x_value + block_size;
    block_grams_2x2_step(iterator, block_size);
    maybe_dump_lobpcg_step_matrix(iterator.debug_label,
                                  "iter2.rr_2x2.gramA",
                                  iterator.gram_a.topLeftCorner(subdim, subdim));
    maybe_dump_lobpcg_step_matrix(iterator.debug_label,
                                  "iter2.rr_2x2.gramB",
                                  iterator.gram_b.topLeftCorner(subdim, subdim));

    Eigen::MatrixXd gram_a_view = iterator.gram_a.topLeftCorner(subdim, subdim);
    Eigen::MatrixXd gram_b_view = iterator.gram_b.topLeftCorner(subdim, subdim);
    symmetrize_in_place(gram_a_view);
    symmetrize_in_place(gram_b_view);
    std::optional<RitzPair> pair =
        solve_generalized_hermitian_full(std::move(gram_a_view), std::move(gram_b_view));
    if (!pair.has_value()) {
        return false;
    }

    maybe_dump_lobpcg_step_vector(iterator.debug_label, "iter2.rr_2x2.eigvals", pair->eigenvalues);
    maybe_dump_lobpcg_step_matrix(iterator.debug_label, "iter2.rr_2x2.eigvecs", pair->eigenvectors);
    sort_selected_permutation(iterator.lambda_perm, pair->eigenvalues, subdim, iterator.largest);
    maybe_dump_lobpcg_step_indices(iterator.debug_label,
                                   "iter2.rr_2x2.perm",
                                   std::vector<int>(iterator.lambda_perm.begin(),
                                                    iterator.lambda_perm.begin() + subdim));
    if (!update_selected_ritz_pairs(iterator, pair->eigenvalues, pair->eigenvectors, subdim)) {
        return false;
    }
    maybe_dump_lobpcg_step_vector(iterator.debug_label, "iter2.rr_2x2.ritz_values", iterator.ritz_values.head(size_x_value));
    maybe_dump_lobpcg_step_matrix(iterator.debug_label,
                                  "iter2.rr_2x2.selected_vectors",
                                  iterator.v_basis.topLeftCorner(subdim, size_x_value));
    return true;
}

void maybe_run_lobpcg_step_probe(const LobpcgIterator& source_iterator) {
    if (!lobpcg_step_debug_enabled(source_iterator.debug_label)) {
        return;
    }

    LobpcgIterator iterator = source_iterator;
    iterator.iteration = 1;
    iterator.current_block_size = size_x(iterator);
    iterator.active_mask.assign(size_x(iterator), 1);

    if (!debug_run_cholqr_probe(iterator, iterator.x_blocks, "iter1.x")) {
        return;
    }
    if (!debug_dump_initial_subproblem_probe(iterator)) {
        return;
    }
    update_x_p_step(iterator, 0, 0);
    residuals_step(iterator);
    update_mask_step(iterator);
    maybe_dump_lobpcg_step_vector(iterator.debug_label, "iter1.residuals", iterator.residual_norm_values.head(size_x(iterator)));
    maybe_dump_lobpcg_step_indices(iterator.debug_label,
                                   "iter1.active_mask",
                                   active_mask_as_indices(iterator.active_mask));

    if (iterator.current_block_size <= 0) {
        return;
    }

    iterator.iteration = 2;
    update_active_r_blocks(iterator);
    maybe_dump_lobpcg_step_matrix(iterator.debug_label, "iter2.r.precond_input", iterator.active_r_blocks.block);
    precond_constr_step(iterator);
    maybe_dump_lobpcg_step_matrix(iterator.debug_label,
                                  "iter2.r.post_precond_constr",
                                  iterator.active_r_blocks.block);
    if (!debug_run_cholqr_probe(iterator, iterator.active_r_blocks, "iter2.r")) {
        return;
    }

    const int block_size = static_cast<int>(iterator.active_r_blocks.block.cols());
    if (!debug_dump_second_subproblem_probe(iterator, block_size)) {
        return;
    }
    update_x_p_step(iterator, block_size, 0);
    residuals_step(iterator);
    update_mask_step(iterator);
    maybe_dump_lobpcg_step_vector(iterator.debug_label, "iter2.residuals", iterator.residual_norm_values.head(size_x(iterator)));
    maybe_dump_lobpcg_step_indices(iterator.debug_label,
                                   "iter2.active_mask",
                                   active_mask_as_indices(iterator.active_mask));
}

Eigen::MatrixXd initialize_lobpcg_block(int n, int dims, int seed, AlgorithmRng* shared_rng = nullptr);
void prepare_initial_lobpcg_block(LobpcgIterator& iterator, bool not_zeros);
LobpcgState snapshot_lobpcg_state(const LobpcgIterator& iterator);
bool lobpcg_converged(const LobpcgIterator& iterator);
void append_lobpcg_trace(LobpcgIterator& iterator);
LobpcgResults finalize_lobpcg_results(const LobpcgIterator& iterator,
                                      Eigen::MatrixXd finalized_basis);
std::string format_lobpcg_results(const LobpcgResults& results);
void emit_lobpcg_results_log(const LobpcgResults& results);
LobpcgResults make_empty_lobpcg_results(int rows,
                                        int total_dims,
                                        double tolerance,
                                        int max_iterations,
                                        int block_size);
void append_lobpcg_results(LobpcgResults& accumulated,
                           const LobpcgResults& batch,
                           int offset,
                           int cols_to_copy,
                           int source_offset = 0);
int choose_lobpcg_block_size(int n, int total_dims);

LobpcgIterator initialize_lobpcg_iterator(const DenseOperator& apply_a,
                                          const DenseOperator& apply_b,
                                          const BlockPreconditioner& apply_preconditioner,
                                          int n,
                                          int dims,
                                          int seed,
                                          bool generalized,
                                          bool largest,
                                          bool project_constant,
                                          double tolerance,
                                          int max_iterations,
                                          bool log_trace,
                                          const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd(),
                                          AlgorithmRng* shared_rng = nullptr,
                                          const std::string& debug_label = "") {
    LobpcgIterator iterator;
    iterator.apply_a = apply_a;
    iterator.apply_b = apply_b;
    iterator.apply_preconditioner = apply_preconditioner;
    iterator.generalized = generalized;
    iterator.largest = largest;
    iterator.project_constant = project_constant;
    iterator.tolerance = tolerance;
    iterator.max_iterations = max_iterations;
    iterator.seed = seed;
    iterator.shared_rng = shared_rng;
    iterator.log_trace = log_trace;
    iterator.debug_label = debug_label;
    iterator.failure_stage.clear();
    const Eigen::MatrixXd valid_constraint_basis =
        (constraint_basis.rows() == n) ? constraint_basis : Eigen::MatrixXd();
    iterator.constraint = initialize_constraint(apply_b, valid_constraint_basis, dims);
    iterator.x_blocks.block = initialize_lobpcg_block(n, dims, seed, shared_rng);
    if (iterator.x_blocks.block.cols() == 0) {
        return iterator;
    }

    const int actual_dims = static_cast<int>(iterator.x_blocks.block.cols());
    iterator.temp_x_blocks = make_empty_blocks(n, actual_dims);
    iterator.p_blocks = make_empty_blocks(n, actual_dims);
    iterator.active_p_blocks = make_empty_blocks(n, actual_dims);
    iterator.r_blocks = make_empty_blocks(n, actual_dims);
    iterator.active_r_blocks = make_empty_blocks(n, actual_dims);
    iterator.ritz_values = Eigen::VectorXd::Zero(3 * actual_dims);
    iterator.lambda_perm.assign(3 * actual_dims, 0);
    iterator.lambda = Eigen::VectorXd::Zero(actual_dims);
    iterator.v_basis = Eigen::MatrixXd::Zero(3 * actual_dims, 3 * actual_dims);
    iterator.residual_norm_values = Eigen::VectorXd::Zero(actual_dims);
    iterator.gram_a_block = make_block_gram(actual_dims);
    iterator.gram_b_block = make_block_gram(actual_dims);
    iterator.gram_a = Eigen::MatrixXd::Zero(3 * actual_dims, 3 * actual_dims);
    iterator.gram_b = Eigen::MatrixXd::Zero(3 * actual_dims, 3 * actual_dims);
    iterator.cholqr.gram_vbv = Eigen::MatrixXd::Zero(actual_dims, actual_dims);
    iterator.active_mask.assign(actual_dims, 1);
    iterator.current_block_size = actual_dims;
    iterator.iteration = 1;
    iterator.last_active_r_cols = 0;
    iterator.last_active_p_cols = 0;
    iterator.last_used_three_term = false;
    iterator.last_r_eig_norm = 0.0;
    iterator.last_p_eig_norm = 0.0;
    iterator.last_p_block_norm = 0.0;
    iterator.last_xar_norm = 0.0;
    iterator.last_xbr_norm = 0.0;
    iterator.last_rar_norm = 0.0;
    iterator.trace.clear();
    return iterator;
}

void reset_lobpcg_iterator_for_batch(LobpcgIterator& iterator,
                                     Eigen::MatrixXd initial_block,
                                     int seed) {
    const int dims = static_cast<int>(initial_block.cols());
    const int rows = static_cast<int>(initial_block.rows());
    iterator.seed = seed;
    iterator.failure_stage.clear();
    iterator.x_blocks.block = std::move(initial_block);
    iterator.x_blocks.a_block = Eigen::MatrixXd::Zero(rows, dims);
    iterator.x_blocks.b_block = Eigen::MatrixXd::Zero(rows, dims);
    iterator.temp_x_blocks = make_empty_blocks(rows, dims);
    iterator.p_blocks = make_empty_blocks(rows, dims);
    iterator.active_p_blocks = make_empty_blocks(rows, dims);
    iterator.r_blocks = make_empty_blocks(rows, dims);
    iterator.active_r_blocks = make_empty_blocks(rows, dims);
    iterator.ritz_values.setZero(3 * dims);
    iterator.lambda_perm.assign(3 * dims, 0);
    iterator.lambda.setZero(dims);
    iterator.v_basis.setZero(3 * dims, 3 * dims);
    iterator.residual_norm_values.setZero(dims);
    iterator.cholqr.gram_vbv.setZero(dims, dims);
    iterator.active_mask.assign(dims, 1);
    iterator.current_block_size = dims;
    iterator.iteration = 1;
    iterator.last_active_r_cols = 0;
    iterator.last_active_p_cols = 0;
    iterator.last_used_three_term = false;
    iterator.last_r_eig_norm = 0.0;
    iterator.last_p_eig_norm = 0.0;
    iterator.last_p_block_norm = 0.0;
    iterator.last_xar_norm = 0.0;
    iterator.last_xbr_norm = 0.0;
    iterator.last_rar_norm = 0.0;
    iterator.trace.clear();
}

void residuals_step(LobpcgIterator& iterator) {
    const int dims = size_x(iterator);
    const Eigen::MatrixXd& b_source =
        iterator.generalized ? iterator.x_blocks.b_block : iterator.x_blocks.block;
    iterator.r_blocks.block =
        iterator.x_blocks.a_block - b_source * iterator.ritz_values.head(dims).asDiagonal();
    iterator.residual_norm_values = residual_norms(iterator.r_blocks.block);
}

void update_mask_step(LobpcgIterator& iterator) {
    const int dims = size_x(iterator);
    iterator.active_mask.assign(dims, 0);
    iterator.current_block_size = 0;
    for (int idx = 0; idx < dims; ++idx) {
        if (iterator.residual_norm_values(idx) > iterator.tolerance) {
            iterator.active_mask[idx] = 1;
            iterator.current_block_size += 1;
        }
    }
}

void precond_constr_step(LobpcgIterator& iterator) {
    if (iterator.current_block_size <= 0) {
        return;
    }
    iterator.active_r_blocks.block = iterator.apply_preconditioner(iterator.active_r_blocks.block);
    apply_constraint(iterator.constraint, iterator.active_r_blocks.block);
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
    sort_eigenpair_permutation(permutation, values, subdim, largest);
}

bool update_selected_ritz_pairs(LobpcgIterator& iterator,
                                const Eigen::VectorXd& values,
                                const Eigen::MatrixXd& vectors,
                                int basis_rows) {
    const int size_x_value = size_x(iterator);
    const int available_values = std::min<int>(values.size(), vectors.cols());
    if (size_x_value <= 0 || basis_rows <= 0 || vectors.rows() < basis_rows ||
        available_values < size_x_value) {
        return false;
    }

    sort_selected_permutation(iterator.lambda_perm, values, available_values, iterator.largest);
    iterator.v_basis.block(0, 0, basis_rows, size_x_value).setZero();
    for (int col = 0; col < size_x_value; ++col) {
        const int index = iterator.lambda_perm[col];
        iterator.ritz_values(col) = values(index);
        iterator.v_basis.block(0, col, basis_rows, 1) = vectors.col(index);
    }
    iterator.lambda.head(size_x_value) = iterator.ritz_values.head(size_x_value);
    return true;
}

void block_grams_1x1_step(LobpcgIterator& iterator) {
    compute_xax(iterator.gram_a_block, iterator.x_blocks);
}

void block_grams_2x2_step(LobpcgIterator& iterator, int block_size) {
    const int size_x_value = size_x(iterator);
    compute_xar(iterator.gram_a_block, iterator.x_blocks, iterator.active_r_blocks, block_size);
    compute_rar(iterator.gram_a_block, iterator.active_r_blocks, block_size);
    compute_xbr(iterator.gram_b_block, iterator.x_blocks, iterator.active_r_blocks, block_size);
    iterator.last_xar_norm = iterator.gram_a_block.xar.leftCols(block_size).norm();
    iterator.last_xbr_norm = iterator.gram_b_block.xar.leftCols(block_size).norm();
    iterator.last_rar_norm = iterator.gram_a_block.rar.topLeftCorner(block_size, block_size).norm();

    iterator.gram_a.topLeftCorner(size_x_value + block_size, size_x_value + block_size).setZero();
    iterator.gram_b.topLeftCorner(size_x_value + block_size, size_x_value + block_size).setZero();
    iterator.gram_a.topLeftCorner(size_x_value, size_x_value) =
        iterator.ritz_values.head(size_x_value).asDiagonal();
    iterator.gram_a.block(0, size_x_value, size_x_value, block_size) =
        iterator.gram_a_block.xar.leftCols(block_size);
    iterator.gram_a.block(size_x_value, 0, block_size, size_x_value) =
        iterator.gram_a_block.xar.leftCols(block_size).transpose();
    iterator.gram_a.block(size_x_value, size_x_value, block_size, block_size) =
        iterator.gram_a_block.rar.topLeftCorner(block_size, block_size);
    iterator.gram_b.topLeftCorner(size_x_value, size_x_value).setIdentity();
    iterator.gram_b.block(0, size_x_value, size_x_value, block_size) =
        iterator.gram_b_block.xar.leftCols(block_size);
    iterator.gram_b.block(size_x_value, 0, block_size, size_x_value) =
        iterator.gram_b_block.xar.leftCols(block_size).transpose();
    iterator.gram_b.block(size_x_value, size_x_value, block_size, block_size).setIdentity();
}

void block_grams_3x3_step(LobpcgIterator& iterator,
                          int block_size_r,
                          int block_size_p) {
    const int size_x_value = size_x(iterator);
    compute_xar(iterator.gram_a_block, iterator.x_blocks, iterator.active_r_blocks, block_size_r);
    compute_xap(iterator.gram_a_block, iterator.x_blocks, iterator.active_p_blocks, block_size_p);
    compute_rar(iterator.gram_a_block, iterator.active_r_blocks, block_size_r);
    compute_rap(iterator.gram_a_block, iterator.active_r_blocks, iterator.active_p_blocks,
                block_size_r, block_size_p);
    compute_pap(iterator.gram_a_block, iterator.active_p_blocks, block_size_p);
    compute_xbr(iterator.gram_b_block, iterator.x_blocks, iterator.active_r_blocks, block_size_r);
    compute_xbp(iterator.gram_b_block, iterator.x_blocks, iterator.active_p_blocks, block_size_p);
    compute_rbp(iterator.gram_b_block, iterator.active_r_blocks, iterator.active_p_blocks,
                block_size_r, block_size_p);
    iterator.last_xar_norm = iterator.gram_a_block.xar.leftCols(block_size_r).norm();
    iterator.last_xbr_norm = iterator.gram_b_block.xar.leftCols(block_size_r).norm();
    iterator.last_rar_norm =
        iterator.gram_a_block.rar.topLeftCorner(block_size_r, block_size_r).norm();

    const int subdim = size_x_value + block_size_r + block_size_p;
    iterator.gram_a.topLeftCorner(subdim, subdim).setZero();
    iterator.gram_b.topLeftCorner(subdim, subdim).setZero();
    iterator.gram_a.topLeftCorner(size_x_value, size_x_value) =
        iterator.ritz_values.head(size_x_value).asDiagonal();
    iterator.gram_a.block(0, size_x_value, size_x_value, block_size_r) =
        iterator.gram_a_block.xar.leftCols(block_size_r);
    iterator.gram_a.block(size_x_value, 0, block_size_r, size_x_value) =
        iterator.gram_a_block.xar.leftCols(block_size_r).transpose();
    iterator.gram_a.block(size_x_value, size_x_value, block_size_r, block_size_r) =
        iterator.gram_a_block.rar.topLeftCorner(block_size_r, block_size_r);
    iterator.gram_a.block(0, size_x_value + block_size_r, size_x_value, block_size_p) =
        iterator.gram_a_block.xap.leftCols(block_size_p);
    iterator.gram_a.block(size_x_value + block_size_r, 0, block_size_p, size_x_value) =
        iterator.gram_a_block.xap.leftCols(block_size_p).transpose();
    iterator.gram_a.block(size_x_value, size_x_value + block_size_r, block_size_r, block_size_p) =
        iterator.gram_a_block.rap.topLeftCorner(block_size_r, block_size_p);
    iterator.gram_a.block(size_x_value + block_size_r, size_x_value, block_size_p, block_size_r) =
        iterator.gram_a_block.rap.topLeftCorner(block_size_r, block_size_p).transpose();
    iterator.gram_a.block(size_x_value + block_size_r, size_x_value + block_size_r,
                       block_size_p, block_size_p) =
        iterator.gram_a_block.pap.topLeftCorner(block_size_p, block_size_p);

    iterator.gram_b.topLeftCorner(size_x_value, size_x_value).setIdentity();
    iterator.gram_b.block(0, size_x_value, size_x_value, block_size_r) =
        iterator.gram_b_block.xar.leftCols(block_size_r);
    iterator.gram_b.block(size_x_value, 0, block_size_r, size_x_value) =
        iterator.gram_b_block.xar.leftCols(block_size_r).transpose();
    iterator.gram_b.block(size_x_value, size_x_value, block_size_r, block_size_r).setIdentity();
    iterator.gram_b.block(0, size_x_value + block_size_r, size_x_value, block_size_p) =
        iterator.gram_b_block.xap.leftCols(block_size_p);
    iterator.gram_b.block(size_x_value + block_size_r, 0, block_size_p, size_x_value) =
        iterator.gram_b_block.xap.leftCols(block_size_p).transpose();
    iterator.gram_b.block(size_x_value, size_x_value + block_size_r, block_size_r, block_size_p) =
        iterator.gram_b_block.rap.topLeftCorner(block_size_r, block_size_p);
    iterator.gram_b.block(size_x_value + block_size_r, size_x_value, block_size_p, block_size_r) =
        iterator.gram_b_block.rap.topLeftCorner(block_size_r, block_size_p).transpose();
    iterator.gram_b.block(size_x_value + block_size_r, size_x_value + block_size_r,
                       block_size_p, block_size_p).setIdentity();
}

bool sub_problem_step(LobpcgIterator& iterator, int block_size_r, int block_size_p) {
    const int size_x_value = size_x(iterator);
    const int subdim = size_x_value + block_size_r + block_size_p;
    if (subdim <= 0 || size_x_value <= 0) {
        return false;
    }

    if (block_size_r == 0 && block_size_p == 0) {
        Eigen::MatrixXd gram_a_view =
            iterator.gram_a_block.xax.topLeftCorner(subdim, subdim);
        symmetrize_in_place(gram_a_view);
        std::optional<LapackEigenResult> solver = lapack_symmetric_eigen(std::move(gram_a_view));
        if (!solver.has_value()) {
            return false;
        }
        return update_selected_ritz_pairs(iterator, solver->eigenvalues,
                                          solver->eigenvectors, subdim);
    }

    Eigen::MatrixXd gram_a_view = iterator.gram_a.topLeftCorner(subdim, subdim);
    Eigen::MatrixXd gram_b_view = iterator.gram_b.topLeftCorner(subdim, subdim);
    symmetrize_in_place(gram_a_view);
    symmetrize_in_place(gram_b_view);
    std::optional<RitzPair> pair =
        solve_generalized_hermitian_full(std::move(gram_a_view), std::move(gram_b_view));
    if (!pair.has_value()) {
        return false;
    }
    return update_selected_ritz_pairs(iterator, pair->eigenvalues, pair->eigenvectors, subdim);
}

void update_x_p_step(LobpcgIterator& iterator, int block_size_r, int block_size_p) {
    const int size_x_value = size_x(iterator);
    const Eigen::MatrixXd x_eig = iterator.v_basis.topLeftCorner(size_x_value, size_x_value);
    const Eigen::MatrixXd r_eig =
        block_size_r > 0
            ? iterator.v_basis.block(size_x_value, 0, block_size_r, size_x_value)
            : Eigen::MatrixXd(0, size_x_value);
    const Eigen::MatrixXd p_eig =
        block_size_p > 0
            ? iterator.v_basis.block(size_x_value + block_size_r, 0, block_size_p, size_x_value)
            : Eigen::MatrixXd(0, size_x_value);

    iterator.p_blocks = make_empty_blocks(iterator.x_blocks.block.rows(), size_x_value);
    iterator.last_r_eig_norm = r_eig.norm();
    iterator.last_p_eig_norm = p_eig.norm();
    if (block_size_r > 0) {
        iterator.p_blocks.block.noalias() += iterator.active_r_blocks.block.leftCols(block_size_r) * r_eig;
        iterator.p_blocks.a_block.noalias() += iterator.active_r_blocks.a_block.leftCols(block_size_r) * r_eig;
        iterator.p_blocks.b_block.noalias() += iterator.active_r_blocks.b_block.leftCols(block_size_r) * r_eig;
    }
    if (block_size_p > 0) {
        iterator.temp_x_blocks.block = iterator.active_p_blocks.block.leftCols(block_size_p) * p_eig;
        iterator.temp_x_blocks.a_block = iterator.active_p_blocks.a_block.leftCols(block_size_p) * p_eig;
        iterator.temp_x_blocks.b_block = iterator.active_p_blocks.b_block.leftCols(block_size_p) * p_eig;
        iterator.p_blocks.block += iterator.temp_x_blocks.block;
        iterator.p_blocks.a_block += iterator.temp_x_blocks.a_block;
        iterator.p_blocks.b_block += iterator.temp_x_blocks.b_block;
    }

    iterator.temp_x_blocks.block = iterator.x_blocks.block * x_eig;
    iterator.temp_x_blocks.a_block = iterator.x_blocks.a_block * x_eig;
    iterator.temp_x_blocks.b_block = iterator.x_blocks.b_block * x_eig;
    if (block_size_r > 0 || block_size_p > 0) {
        iterator.x_blocks.block = iterator.temp_x_blocks.block + iterator.p_blocks.block;
        iterator.x_blocks.a_block = iterator.temp_x_blocks.a_block + iterator.p_blocks.a_block;
        iterator.x_blocks.b_block = iterator.temp_x_blocks.b_block + iterator.p_blocks.b_block;
    } else {
        iterator.x_blocks.block = iterator.temp_x_blocks.block;
        iterator.x_blocks.a_block = iterator.temp_x_blocks.a_block;
        iterator.x_blocks.b_block = iterator.temp_x_blocks.b_block;
    }
    iterator.last_p_block_norm = iterator.p_blocks.block.norm();
}

void fill_random_lobpcg_block(Eigen::MatrixXd& block, AlgorithmRng& rng) {
    fill_random_lobpcg_contiguous(block.data(), static_cast<int>(block.size()), rng);
}

AlgorithmRng make_lobpcg_rng(int seed) {
    return AlgorithmRng(seed);
}

AlgorithmRng& select_lobpcg_rng(AlgorithmRng& local_rng, AlgorithmRng* shared_rng) {
    return shared_rng == nullptr ? local_rng : *shared_rng;
}

void fill_random_lobpcg_columns(Eigen::MatrixXd& block,
                                int start_col,
                                int num_cols,
                                AlgorithmRng& rng) {
    if (num_cols <= 0 || start_col < 0 || start_col + num_cols > block.cols()) {
        return;
    }
    fill_random_lobpcg_contiguous(block.data() + static_cast<Eigen::Index>(start_col) * block.rows(),
                                  block.rows() * num_cols,
                                  rng);
}

bool is_exact_zero_column(const Eigen::VectorXd& column) {
    for (int row = 0; row < column.size(); ++row) {
        if (column(row) != 0.0) {
            return false;
        }
    }
    return true;
}

bool has_exact_zero_columns(const Eigen::MatrixXd& block) {
    for (int col = 0; col < block.cols(); ++col) {
        if (is_exact_zero_column(block.col(col))) {
            return true;
        }
    }
    return false;
}

void refill_zero_columns(Eigen::MatrixXd& block, AlgorithmRng& rng) {
    if (block.cols() == 0) {
        return;
    }

    for (int col = 0; col < block.cols(); ++col) {
        if (!is_exact_zero_column(block.col(col))) {
            continue;
        }
        for (int row = 0; row < block.rows(); ++row) {
            block(row, col) = rng.next_float64();
        }
    }
}

Eigen::MatrixXd initialize_lobpcg_block(int n, int dims, int seed, AlgorithmRng* shared_rng) {
    AlgorithmRng local_rng = make_lobpcg_rng(seed);
    AlgorithmRng& rng = select_lobpcg_rng(local_rng, shared_rng);
    Eigen::MatrixXd block(n, dims);
    fill_random_lobpcg_block(block, rng);
    return block;
}

void prepare_initial_lobpcg_block(LobpcgIterator& iterator, bool not_zeros) {
    apply_constraint(iterator.constraint, iterator.x_blocks.block);
    if (not_zeros && !has_exact_zero_columns(iterator.x_blocks.block)) {
        return;
    }

    AlgorithmRng local_rng = make_lobpcg_rng(iterator.seed);
    AlgorithmRng& rng = select_lobpcg_rng(local_rng, iterator.shared_rng);
    refill_zero_columns(iterator.x_blocks.block, rng);
    apply_constraint(iterator.constraint, iterator.x_blocks.block);
}

Eigen::MatrixXd finalize_lobpcg_basis(const DenseOperator& apply_a,
                                      const DenseOperator& apply_b,
                                      Eigen::MatrixXd basis,
                                      bool largest) {
    // IterativeSolvers.lobpcg returns the in-place Ritz block `X` directly.
    // Re-solving a projected dense problem here introduces an extra rotation
    // and scaling step that Julia does not apply on the main path.
    (void)apply_a;
    (void)apply_b;
    (void)largest;
    return basis;
}

LobpcgState snapshot_lobpcg_state(const LobpcgIterator& iterator) {
    LobpcgState state;
    const int dims = size_x(iterator);
    state.iteration = iterator.iteration;
    state.residual_norms = iterator.residual_norm_values.head(dims);
    state.ritz_values = iterator.ritz_values.head(dims);
    state.current_block_size = iterator.current_block_size;
    state.active_r_cols = iterator.last_active_r_cols;
    state.active_p_cols = iterator.last_active_p_cols;
    state.used_three_term = iterator.last_used_three_term;
    state.r_eig_norm = iterator.last_r_eig_norm;
    state.p_eig_norm = iterator.last_p_eig_norm;
    state.p_block_norm = iterator.last_p_block_norm;
    state.xar_norm = iterator.last_xar_norm;
    state.xbr_norm = iterator.last_xbr_norm;
    state.rar_norm = iterator.last_rar_norm;
    return state;
}

Eigen::MatrixXd build_next_batch_initial_block(const Eigen::MatrixXd& previous_basis,
                                               int next_remaining_dims,
                                               int block_size,
                                               int seed,
                                               AlgorithmRng* shared_rng = nullptr) {
    if (previous_basis.cols() != block_size) {
        return initialize_lobpcg_block(previous_basis.rows(), block_size, seed, shared_rng);
    }

    if (next_remaining_dims <= 0 || next_remaining_dims >= block_size) {
        return initialize_lobpcg_block(previous_basis.rows(), block_size, seed, shared_rng);
    }

    const int cutoff = block_size - next_remaining_dims;
    Eigen::MatrixXd next_initial = previous_basis;
    next_initial.leftCols(next_remaining_dims) = previous_basis.rightCols(next_remaining_dims);

    AlgorithmRng local_rng = make_lobpcg_rng(seed);
    AlgorithmRng& rng = select_lobpcg_rng(local_rng, shared_rng);
    fill_random_lobpcg_columns(next_initial, cutoff, next_remaining_dims, rng);
    return next_initial;
}

bool lobpcg_converged(const LobpcgIterator& iterator) {
    const int dims = size_x(iterator);
    if (dims <= 0) {
        return false;
    }
    return iterator.residual_norm_values.head(dims).maxCoeff() <= iterator.tolerance;
}

void append_lobpcg_trace(LobpcgIterator& iterator) {
    if (!iterator.log_trace) {
        return;
    }
    iterator.trace.push_back(snapshot_lobpcg_state(iterator));
}

LobpcgResults finalize_lobpcg_results(const LobpcgIterator& iterator,
                                      Eigen::MatrixXd finalized_basis) {
    LobpcgResults results;
    const int dims = std::min<int>(size_x(iterator), iterator.lambda.size());
    results.lambda = iterator.lambda.head(dims);
    results.x = std::move(finalized_basis);
    results.tolerance = iterator.tolerance;
    results.residual_norms = iterator.residual_norm_values.head(dims);
    results.iterations = iterator.iteration;
    results.max_iterations = iterator.max_iterations;
    results.converged = lobpcg_converged(iterator);
    results.success = true;
    results.block_size = dims;
    if (iterator.log_trace) {
        results.trace = iterator.trace;
        results.batch_traces.push_back(iterator.trace);
    }
    results.batch_iterations.push_back(iterator.iteration);
    results.converged_mask.assign(dims, results.converged ? 1 : 0);
    return results;
}

std::string format_lobpcg_scalar(double value) {
    std::ostringstream stream;
    stream << std::setprecision(16) << value;
    return stream.str();
}

std::string format_lobpcg_vector(const Eigen::VectorXd& values) {
    if (values.size() == 0) {
        return "[]";
    }

    std::vector<std::string> formatted;
    formatted.reserve(static_cast<std::size_t>(values.size()));
    std::size_t joined_length = 0;
    for (int index = 0; index < values.size(); ++index) {
        formatted.push_back(format_lobpcg_scalar(values(index)));
        joined_length += formatted.back().size();
        if (index + 1 < values.size()) {
            joined_length += 1;
        }
    }

    std::ostringstream stream;
    stream << '[';
    if (joined_length < 40 || values.size() <= 2) {
        for (int index = 0; index < values.size(); ++index) {
            if (index > 0) {
                stream << ',';
            }
            stream << formatted[static_cast<std::size_t>(index)];
        }
    } else {
        stream << formatted[0] << ',' << formatted[1] << ", ...";
    }
    stream << ']';
    return stream.str();
}

std::string format_lobpcg_iterations(const LobpcgResults& results) {
    if (results.batch_iterations.empty()) {
        return std::to_string(results.iterations);
    }
    if (results.batch_iterations.size() == 1) {
        return std::to_string(results.batch_iterations.front());
    }

    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < results.batch_iterations.size(); ++index) {
        if (index > 0) {
            stream << ',';
        }
        stream << results.batch_iterations[index];
    }
    stream << ']';
    return stream.str();
}

std::string format_lobpcg_trace(const LobpcgTrace& trace) {
    std::ostringstream stream;
    stream << "Iteration    Maximum residual norm     bs    rcols    pcols    step    |rV|      |pV|      |P|       |XAR|     |XBR|     |RAR|\n";
    stream << "---------    ---------------------    --    -----    -----    ----    -----      -----      -----     -----     -----     -----\n";
    for (const LobpcgState& state : trace) {
        const double max_residual =
            (state.residual_norms.size() > 0) ? state.residual_norms.maxCoeff() : 0.0;
        stream << std::setw(8) << state.iteration << "    "
               << std::scientific << std::setprecision(6) << max_residual << "    "
               << std::setw(2) << std::defaultfloat << state.current_block_size << "    "
               << std::setw(5) << state.active_r_cols << "    "
               << std::setw(5) << state.active_p_cols << "    "
               << (state.used_three_term ? "3x3" : "2x2") << "    "
               << std::scientific << std::setprecision(3) << std::setw(8) << state.r_eig_norm
               << "    " << std::setw(8) << state.p_eig_norm
               << "    " << std::setw(8) << state.p_block_norm
               << "    " << std::setw(8) << state.xar_norm
               << "    " << std::setw(8) << state.xbr_norm
               << "    " << std::setw(8) << state.rar_norm << '\n';
    }
    return stream.str();
}

std::string format_lobpcg_trace_exact(const LobpcgTrace& trace) {
    std::ostringstream stream;
    stream << "LOBPCG exact trace\n";
    for (const LobpcgState& state : trace) {
        const double max_residual =
            (state.residual_norms.size() > 0) ? state.residual_norms.maxCoeff() : 0.0;
        stream << "iter=" << state.iteration
               << " max_residual=" << format_lobpcg_scalar(max_residual)
               << " residual_norms=" << format_lobpcg_vector(state.residual_norms)
               << " ritz_values=" << format_lobpcg_vector(state.ritz_values) << '\n';
    }
    return stream.str();
}

std::string format_lobpcg_results(const LobpcgResults& results) {
    std::ostringstream stream;
    stream << "Results of LOBPCG Algorithm\n";
    stream << " * Algorithm: LOBPCG - CholQR\n";
    stream << " * lambda: " << format_lobpcg_vector(results.lambda) << '\n';
    stream << " * Residual norm(s): " << format_lobpcg_vector(results.residual_norms) << '\n';
    stream << " * Convergence\n";
    stream << "   * Iterations: " << format_lobpcg_iterations(results) << '\n';
    stream << "   * Converged: " << (results.converged ? "true" : "false") << '\n';
    stream << "   * Iterations limit: " << results.max_iterations << '\n';

    if (results.batch_traces.empty()) {
        return stream.str();
    }
    if (results.batch_traces.size() == 1) {
        stream << format_lobpcg_trace(results.batch_traces.front());
        stream << format_lobpcg_trace_exact(results.batch_traces.front());
        return stream.str();
    }

    for (std::size_t batch = 0; batch < results.batch_traces.size(); ++batch) {
        stream << "Batch " << (batch + 1) << " trace\n";
        stream << format_lobpcg_trace(results.batch_traces[batch]);
        stream << format_lobpcg_trace_exact(results.batch_traces[batch]);
    }
    return stream.str();
}

std::string format_lobpcg_fallback(const LobpcgResults& results,
                                   int required_dims,
                                   const std::string& label) {
    std::ostringstream stream;
    stream << "LOBPCG fallback: " << label << '\n';
    stream << " * Reason: no usable basis was returned\n";
    if (!results.failure_reason.empty()) {
        stream << " * Failure stage: " << results.failure_reason << '\n';
    }
    if (results.failure_iteration > 0) {
        stream << " * Failure iteration: " << results.failure_iteration << '\n';
    }
    stream << " * Success flag: " << (results.success ? "true" : "false") << '\n';
    stream << " * Required columns: " << required_dims << '\n';
    stream << " * Returned columns: " << results.x.cols() << '\n';
    stream << " * Iterations: " << format_lobpcg_iterations(results) << '\n';
    return stream.str();
}

void emit_lobpcg_results_log(const LobpcgResults& results) {
    const std::string divider(60, '=');
    std::cout << divider << '\n';
    std::cout << format_lobpcg_results(results);
    if (results.batch_traces.empty()) {
        std::cout << '\n';
    }
    std::cout << divider << '\n';
}

void emit_lobpcg_fallback_log(const LobpcgResults& results,
                              int required_dims,
                              const std::string& label) {
    const std::string divider(60, '=');
    std::cout << divider << '\n';
    std::cout << format_lobpcg_fallback(results, required_dims, label);
    std::cout << divider << '\n';
}

LobpcgResults make_empty_lobpcg_results(int rows,
                                        int total_dims,
                                        double tolerance,
                                        int max_iterations,
                                        int block_size) {
    LobpcgResults results;
    results.lambda = Eigen::VectorXd::Zero(total_dims);
    results.x = Eigen::MatrixXd::Zero(rows, total_dims);
    results.tolerance = tolerance;
    results.residual_norms = Eigen::VectorXd::Zero(total_dims);
    results.iterations = 0;
    results.max_iterations = max_iterations;
    results.converged = false;
    results.success = false;
    results.block_size = block_size;
    const int num_batches =
        (block_size > 0) ? (total_dims + block_size - 1) / block_size : 0;
    results.batch_iterations.reserve(num_batches);
    results.converged_mask.assign(total_dims, 0);
    results.batch_traces.reserve(num_batches);
    return results;
}

void append_lobpcg_results(LobpcgResults& accumulated,
                           const LobpcgResults& batch,
                           int offset,
                           int cols_to_copy,
                           int source_offset) {
    if (cols_to_copy <= 0 || source_offset < 0) {
        return;
    }
    if (source_offset + cols_to_copy > batch.lambda.size() ||
        source_offset + cols_to_copy > batch.residual_norms.size() ||
        source_offset + cols_to_copy > batch.x.cols()) {
        return;
    }
    accumulated.lambda.segment(offset, cols_to_copy) =
        batch.lambda.segment(source_offset, cols_to_copy);
    accumulated.residual_norms.segment(offset, cols_to_copy) =
        batch.residual_norms.segment(source_offset, cols_to_copy);
    accumulated.x.block(0, offset, accumulated.x.rows(), cols_to_copy) =
        batch.x.middleCols(source_offset, cols_to_copy);
    for (int idx = 0; idx < cols_to_copy; ++idx) {
        accumulated.converged_mask[offset + idx] =
            (source_offset + idx < static_cast<int>(batch.converged_mask.size())) ?
                batch.converged_mask[source_offset + idx] :
                (batch.converged ? 1 : 0);
    }
    accumulated.batch_iterations.insert(accumulated.batch_iterations.end(),
                                        batch.batch_iterations.begin(),
                                        batch.batch_iterations.end());
    accumulated.batch_traces.insert(accumulated.batch_traces.end(),
                                    batch.batch_traces.begin(),
                                    batch.batch_traces.end());
    accumulated.trace.insert(accumulated.trace.end(),
                             batch.trace.begin(),
                             batch.trace.end());
}

int choose_lobpcg_block_size(int n, int total_dims) {
    int block_size = std::min(total_dims, std::max(1, n / 3));
    while (block_size > 1 && 3 * block_size > n) {
        --block_size;
    }
    return std::max(1, std::min(total_dims, block_size));
}

bool ortho_active_r_step(LobpcgIterator& iterator) {
    if (!ortho_ab_mul_x(iterator.active_r_blocks, iterator.cholqr, iterator.apply_a,
                        iterator.apply_b, iterator.project_constant, iterator.generalized)) {
        iterator.failure_stage = "orthonormalize active R";
        return false;
    }
    return true;
}

bool ortho_active_p_step(LobpcgIterator& iterator) {
    if (iterator.active_p_blocks.block.cols() == 0) {
        return false;
    }
    const bool ortho_ok = iterator.generalized
        ? cholqr_orthonormalize_in_place(iterator.apply_b,
                                         iterator.active_p_blocks.block,
                                         nullptr,
                                         nullptr,
                                         iterator.cholqr.gram_vbv,
                                         iterator.project_constant)
        : cholqr_orthonormalize_standard_in_place(iterator.active_p_blocks.block,
                                                  nullptr,
                                                  nullptr,
                                                  iterator.cholqr.gram_vbv,
                                                  iterator.project_constant);
    if (!ortho_ok) {
        iterator.active_p_blocks = b_orthonormalize_blocks(iterator.apply_a, iterator.apply_b,
                                                           iterator.cholqr,
                                                           std::move(iterator.active_p_blocks),
                                                           iterator.project_constant,
                                                           iterator.generalized);
    } else {
        update_block_products(iterator.active_p_blocks, iterator.apply_a,
                              iterator.apply_b, iterator.generalized);
    }
    if (iterator.active_p_blocks.block.cols() == 0) {
        iterator.failure_stage = "orthonormalize active P";
    }
    return iterator.active_p_blocks.block.cols() > 0;
}

bool initial_iteration_step(LobpcgIterator& iterator) {
    iterator.last_active_r_cols = 0;
    iterator.last_active_p_cols = 0;
    iterator.last_used_three_term = false;
    iterator.last_r_eig_norm = 0.0;
    iterator.last_p_eig_norm = 0.0;
    iterator.last_p_block_norm = 0.0;
    iterator.last_xar_norm = 0.0;
    iterator.last_xbr_norm = 0.0;
    iterator.last_rar_norm = 0.0;
    if (!ortho_ab_mul_x(iterator.x_blocks, iterator.cholqr, iterator.apply_a, iterator.apply_b,
                        iterator.project_constant, iterator.generalized)) {
        iterator.failure_stage = "initial orthonormalize X";
        return false;
    }
    block_grams_1x1_step(iterator);
    if (!sub_problem_step(iterator, 0, 0)) {
        iterator.failure_stage = "initial subproblem";
        return false;
    }
    update_x_p_step(iterator, 0, 0);
    return true;
}

bool second_iteration_step(LobpcgIterator& iterator) {
    update_active_r_blocks(iterator);
    iterator.last_active_r_cols = static_cast<int>(iterator.active_r_blocks.block.cols());
    iterator.last_active_p_cols = 0;
    iterator.last_used_three_term = false;
    precond_constr_step(iterator);
    if (!ortho_active_r_step(iterator)) {
        iterator.current_block_size = 0;
        return true;
    }

    const int block_size = static_cast<int>(iterator.active_r_blocks.block.cols());
    block_grams_2x2_step(iterator, block_size);
    if (!sub_problem_step(iterator, block_size, 0)) {
        iterator.failure_stage = "second-iteration subproblem";
        return false;
    }
    update_x_p_step(iterator, block_size, 0);
    return true;
}

bool general_iteration_step(LobpcgIterator& iterator) {
    if (iterator.log_trace) {
        std::cout << "LOBPCG iter " << iterator.iteration << " update_active_rp\n";
    }
    update_active_rp_blocks(iterator);
    precond_constr_step(iterator);
    if (iterator.log_trace) {
        std::cout << "LOBPCG iter " << iterator.iteration << " ortho_r\n";
    }
    if (!ortho_active_r_step(iterator)) {
        iterator.current_block_size = 0;
        return true;
    }

    const int block_size_r = static_cast<int>(iterator.active_r_blocks.block.cols());
    if (iterator.log_trace) {
        std::cout << "LOBPCG iter " << iterator.iteration
                  << " ortho_p block_r=" << block_size_r << '\n';
    }
    const bool usable_p = ortho_active_p_step(iterator);
    const int block_size_p = static_cast<int>(iterator.active_p_blocks.block.cols());
    iterator.last_active_r_cols = block_size_r;
    iterator.last_active_p_cols = block_size_p;
    iterator.last_used_three_term = usable_p && block_size_p == block_size_r;
    if (!usable_p || block_size_p != block_size_r) {
        const int block_size = block_size_r;
        block_grams_2x2_step(iterator, block_size);
        if (!sub_problem_step(iterator, block_size, 0)) {
            iterator.failure_stage = "general-iteration 2x2 subproblem";
            return false;
        }
        update_x_p_step(iterator, block_size, 0);
        return true;
    }

    if (iterator.log_trace) {
        std::cout << "LOBPCG iter " << iterator.iteration
                  << " block_grams_3x3 r=" << block_size_r
                  << " p=" << block_size_p << '\n';
    }
    block_grams_3x3_step(iterator, block_size_r, block_size_p);
    if (iterator.log_trace) {
        std::cout << "LOBPCG iter " << iterator.iteration << " sub_problem_3x3\n";
    }
    if (!sub_problem_step(iterator, block_size_r, block_size_p)) {
        iterator.failure_stage = "general-iteration 3x3 subproblem";
        return false;
    }
    if (iterator.log_trace) {
        std::cout << "LOBPCG iter " << iterator.iteration << " update_x_p_3x3\n";
    }
    update_x_p_step(iterator, block_size_r, block_size_p);
    return true;
}

bool lobpcg_iteration_step(LobpcgIterator& iterator) {
    bool success = false;
    if (iterator.iteration == 1) {
        success = initial_iteration_step(iterator);
    } else if (iterator.iteration == 2) {
        success = second_iteration_step(iterator);
    } else {
        success = general_iteration_step(iterator);
    }
    if (!success) {
        return false;
    }
    if (iterator.iteration > 1 && iterator.current_block_size == 0 &&
        iterator.active_r_blocks.block.cols() == 0) {
        return true;
    }

    residuals_step(iterator);
    update_mask_step(iterator);
    return true;
}

LobpcgResults execute_lobpcg_iterator_batch(LobpcgIterator& iterator,
                                           int target_dims,
                                           bool not_zeros) {
    LobpcgResults results;
    if (target_dims <= 0 || size_x(iterator) < target_dims) {
        results.failure_reason = "invalid batch target dims";
        return results;
    }

    if (iterator.log_trace) {
        std::cout << "LOBPCG execute batch target_dims=" << target_dims
                  << " size_x=" << size_x(iterator)
                  << " generalized=" << (iterator.generalized ? "true" : "false") << '\n';
    }
    prepare_initial_lobpcg_block(iterator, not_zeros);
    maybe_dump_lobpcg_debug_matrix("x0", iterator.x_blocks.block);
    maybe_run_lobpcg_step_probe(iterator);
    while (iterator.iteration <= iterator.max_iterations) {
        if (iterator.log_trace) {
            std::cout << "LOBPCG iter begin " << iterator.iteration
                      << " current_block_size=" << iterator.current_block_size << '\n';
        }
        if (!lobpcg_iteration_step(iterator)) {
            results.failure_reason = iterator.failure_stage.empty()
                ? "iteration step failed"
                : iterator.failure_stage;
            results.failure_iteration = iterator.iteration;
            return results;
        }
        if (iterator.log_trace) {
            std::cout << "LOBPCG iter end " << iterator.iteration
                      << " next_block_size=" << iterator.current_block_size << '\n';
        }
        append_lobpcg_trace(iterator);
        if (iterator.current_block_size == 0) {
            break;
        }
        iterator.iteration += 1;
    }

    Eigen::MatrixXd finalized;
    if (iterator.log_trace) {
        std::cout << "LOBPCG finalize basis cols="
                  << std::min(target_dims, static_cast<int>(iterator.x_blocks.block.cols()))
                  << '\n';
    }
    if (iterator.generalized) {
        finalized = finalize_lobpcg_basis(
            iterator.apply_a, iterator.apply_b,
            iterator.x_blocks.block.leftCols(
                std::min(target_dims, static_cast<int>(iterator.x_blocks.block.cols()))),
            iterator.largest);
    } else {
        finalized =
            iterator.x_blocks.block.leftCols(
                std::min(target_dims, static_cast<int>(iterator.x_blocks.block.cols())));
    }
    if (finalized.cols() < target_dims) {
        results.failure_reason = "finalize basis returned too few columns";
        results.failure_iteration = iterator.iteration;
        return results;
    }
    maybe_dump_lobpcg_debug_matrix("xfinal", finalized);
    return finalize_lobpcg_results(iterator, std::move(finalized));
}

LobpcgResults solve_lobpcg_single_batch_results(
    const DenseOperator& apply_a,
    const DenseOperator& apply_b,
    const BlockPreconditioner& apply_preconditioner,
    int n,
    bool generalized,
    bool largest,
    int requested_dims,
    int iterations,
    int seed,
    bool project_constant,
    int skip_smallest,
    double tolerance,
    bool log_trace,
    const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd(),
    AlgorithmRng* shared_rng = nullptr,
    const std::string& debug_label = "") {
    LobpcgResults results;
    const int target_dims = requested_dims + std::max(0, skip_smallest);
    if (target_dims <= 0 || 3 * target_dims > n) {
        results.failure_reason = "invalid target dims";
        return results;
    }
    const int max_iterations = std::max(1, iterations);
    LobpcgIterator iterator =
        initialize_lobpcg_iterator(apply_a, apply_b, apply_preconditioner, n, target_dims,
                                   seed, generalized, largest, project_constant, tolerance,
                                   max_iterations, log_trace, constraint_basis, shared_rng,
                                   debug_label);
    if (size_x(iterator) < target_dims) {
        results.failure_reason = "iterator init returned too few columns";
        return results;
    }
    return execute_lobpcg_iterator_batch(iterator, target_dims, true);
}

LobpcgResults solve_lobpcg_results(const DenseOperator& apply_a,
                                   const DenseOperator& apply_b,
                                   const BlockPreconditioner& apply_preconditioner,
                                   int n,
                                   bool generalized,
                                   bool largest,
                                   int requested_dims,
                                   int iterations,
                                   int seed,
                                   bool project_constant,
                                   int skip_smallest,
                                   double tolerance,
                                   bool log_trace,
                                   const Eigen::MatrixXd& constraint_basis = Eigen::MatrixXd(),
                                   AlgorithmRng* shared_rng = nullptr,
                                   const std::string& debug_label = "") {
    const int target_dims = requested_dims + std::max(0, skip_smallest);
    if (target_dims <= 0) {
        LobpcgResults results;
        results.failure_reason = "non-positive target dims";
        return results;
    }

    const int max_iterations = std::max(1, iterations);
    const int block_size = choose_lobpcg_block_size(n, target_dims);
    LobpcgResults accumulated =
        make_empty_lobpcg_results(n, target_dims, tolerance, max_iterations, block_size);
    const Eigen::MatrixXd valid_constraint_basis =
        (constraint_basis.rows() == n) ? constraint_basis : Eigen::MatrixXd();
    LobpcgIterator iterator =
        initialize_lobpcg_iterator(apply_a, apply_b, apply_preconditioner, n, block_size,
                                   seed, generalized, largest, project_constant, tolerance,
                                   max_iterations, log_trace, valid_constraint_basis, shared_rng,
                                   debug_label);
    if (size_x(iterator) < block_size) {
        LobpcgResults results;
        results.failure_reason = "iterator init returned too few columns";
        return results;
    }

    int solved_dims = 0;
    int batch_seed = seed;
    while (solved_dims < target_dims) {
        const int remaining_dims = target_dims - solved_dims;
        const bool tail_batch = remaining_dims < block_size;
        if (log_trace) {
            std::cout << "LOBPCG batch begin solved=" << solved_dims
                      << " remaining=" << remaining_dims
                      << " block=" << block_size
                      << " tail=" << (tail_batch ? "true" : "false") << '\n';
        }
        LobpcgResults batch =
            execute_lobpcg_iterator_batch(iterator, block_size, solved_dims > 0);
        if (!batch.success || batch.x.cols() < block_size || batch.lambda.size() < block_size) {
            LobpcgResults results;
            results.failure_reason =
                batch.failure_reason.empty() ? "batch solve failed" : batch.failure_reason;
            results.failure_iteration = batch.failure_iteration;
            return results;
        }

        const int cols_to_copy = std::min(block_size, remaining_dims);
        const int source_offset = tail_batch ? (block_size - cols_to_copy) : 0;
        if (log_trace) {
            std::cout << "LOBPCG batch append copy=" << cols_to_copy
                      << " source_offset=" << source_offset
                      << " batch_cols=" << batch.x.cols() << '\n';
        }
        append_lobpcg_results(accumulated, batch, solved_dims, cols_to_copy, source_offset);
        solved_dims += cols_to_copy;
        if (solved_dims >= target_dims) {
            break;
        }

        const int next_remaining_dims = target_dims - solved_dims;
        const bool next_batch_is_tail = next_remaining_dims < block_size;
        const int deflate_cols =
            next_batch_is_tail ? (block_size - next_remaining_dims) : block_size;
        if (log_trace) {
            std::cout << "LOBPCG batch next remaining=" << next_remaining_dims
                      << " next_tail=" << (next_batch_is_tail ? "true" : "false")
                      << " deflate_cols=" << deflate_cols << '\n';
        }
        const Eigen::MatrixXd batch_basis = iterator.x_blocks.block;
        const Eigen::MatrixXd batch_b_basis =
            iterator.generalized ? iterator.x_blocks.b_block : iterator.x_blocks.block;
        const Eigen::MatrixXd deflate_basis = batch_basis.leftCols(deflate_cols);
        const Eigen::MatrixXd deflate_b =
            batch_b_basis.leftCols(deflate_cols);
        if (!append_constraint_columns(iterator.constraint, deflate_basis, deflate_b, block_size)) {
            LobpcgResults results;
            results.failure_reason = "failed to update deflation constraint";
            results.failure_iteration = batch.failure_iteration;
            return results;
        }

        batch_seed += block_size;
        Eigen::MatrixXd next_initial =
            next_batch_is_tail
                ? build_next_batch_initial_block(batch_basis, next_remaining_dims, block_size,
                                                 batch_seed, iterator.shared_rng)
                : initialize_lobpcg_block(n, block_size, batch_seed, iterator.shared_rng);
        if (next_batch_is_tail) {
            const int reuse_cols = next_remaining_dims;
            if (log_trace) {
                std::cout << "LOBPCG batch reuse cols=" << reuse_cols
                          << " from batch right side\n";
            }
        }
        reset_lobpcg_iterator_for_batch(iterator, std::move(next_initial), batch_seed);
    }

    accumulated.iterations = std::accumulate(accumulated.batch_iterations.begin(),
                                             accumulated.batch_iterations.end(), 0);
    accumulated.converged =
        std::all_of(accumulated.converged_mask.begin(), accumulated.converged_mask.end(),
                    [](char value) { return value != 0; });
    accumulated.success = true;
    return accumulated;
}

Eigen::MatrixXd solve_small_graph_problem(const WeightedGraph& graph,
                                          bool largest,
                                          int requested_dims,
                                          int iterations,
                                          int seed,
                                          bool log_trace,
                                          const Eigen::MatrixXd& constraint_basis,
                                          AlgorithmRng* shared_rng,
                                          const std::string& debug_label) {
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

    // Julia's small-graph branch uses a direct single-batch LOBPCG call on the
    // graph Laplacian and does not thread the large-graph-style constraint basis
    // through this path.
    (void)constraint_basis;
    LobpcgResults results =
        solve_lobpcg_single_batch_results(
            apply_a, apply_b, apply_preconditioner, graph.num_vertices, false, largest,
            requested_dims, iterations, seed, false, largest ? 0 : 1,
            std::pow(std::numeric_limits<double>::epsilon(), 3.0 / 10.0),
            log_trace, Eigen::MatrixXd(), shared_rng, debug_label);
    if (log_trace && results.success) {
        emit_lobpcg_results_log(results);
    } else if (log_trace) {
        emit_lobpcg_fallback_log(results, requested_dims + std::max(0, largest ? 0 : 1),
                                 "small-graph spectral solve");
    }
    if (results.success &&
        results.x.cols() >= requested_dims + std::max(0, largest ? 0 : 1)) {
        return take_eigenvectors(results.x, largest, requested_dims, largest ? 0 : 1);
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
                                                 bool log_lobpcg,
                                                 const Eigen::MatrixXd& constraint_basis,
                                                 AlgorithmRng* shared_rng,
                                                 const std::string& debug_label) {
    const int n = hypergraph.num_vertices;
    const SparseMatrix laplacian = sparse_graph_laplacian(graph);
    const CmgHierarchy hierarchy =
        build_cmg_like_hierarchy(laplacian, kEnableStrictDominanceAugmentOnMainPath);

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
    maybe_dump_embedding_operator_probe(hypergraph, pindex, epsilon, apply_a, apply_b,
                                        apply_preconditioner);
    LobpcgResults results =
        solve_lobpcg_results(apply_a, apply_b, apply_preconditioner, n, true, largest,
                             requested_dims, iterations, seed, false, 0,
                             kJuliaLargeLobpcgTolerance, log_lobpcg, constraint_basis, shared_rng,
                             debug_label);
    if (log_lobpcg && results.success) {
        emit_lobpcg_results_log(results);
    } else if (log_lobpcg) {
        emit_lobpcg_fallback_log(results, requested_dims, "large-graph operator solve");
    }
    if (!results.success || results.x.cols() < requested_dims) {
        return Eigen::MatrixXd();
    }
    return take_eigenvectors(results.x, largest, requested_dims, 0);
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
                           const Eigen::MatrixXd& constraint_basis,
                           bool log_lobpcg,
                           AlgorithmRng* shared_rng,
                           const std::string& debug_label) {
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
                                         log_lobpcg, constraint_basis, shared_rng, debug_label);
    }

    // Julia's large-graph path uses operator-based LOBPCG with a CMG preconditioner.
    // This C++ path now mirrors that shape more closely than the old dense generalized
    // eigensolve, while keeping the dense route as a safety fallback.
    Eigen::MatrixXd iterative =
        solve_large_graph_problem_lobpcg(
            hypergraph, graph, pindex, largest, requested_dims, iterations, epsilon, seed,
            log_lobpcg, constraint_basis, shared_rng, debug_label);
    if (iterative.cols() == requested_dims) {
        return iterative;
    }

    Eigen::MatrixXd dense = solve_dense_generalized_problem(hypergraph, pindex, largest, requested_dims, epsilon);
    if (dense.cols() == requested_dims) {
        return dense;
    }

    return solve_small_graph_problem(graph, largest, requested_dims, iterations, seed,
                                     log_lobpcg, constraint_basis, shared_rng, debug_label);
}

}  // namespace kspecpart
