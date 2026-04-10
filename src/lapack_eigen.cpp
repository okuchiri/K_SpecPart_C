#include "kspecpart/lapack_eigen.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace kspecpart {

namespace {

extern "C" {

void dsyevr_(const char* jobz,
             const char* range,
             const char* uplo,
             const int* n,
             double* a,
             const int* lda,
             const double* vl,
             const double* vu,
             const int* il,
             const int* iu,
             const double* abstol,
             int* m,
             double* w,
             double* z,
             const int* ldz,
             int* isuppz,
             double* work,
             const int* lwork,
             int* iwork,
             const int* liwork,
             int* info);

void dsygvd_(const int* itype,
             const char* jobz,
             const char* uplo,
             const int* n,
             double* a,
             const int* lda,
             double* b,
             const int* ldb,
             double* w,
             double* work,
             const int* lwork,
             int* iwork,
             const int* liwork,
             int* info);

}  // extern "C"

Eigen::MatrixXd symmetrized(Eigen::MatrixXd matrix) {
    if (matrix.rows() != matrix.cols()) {
        return Eigen::MatrixXd();
    }
    matrix = 0.5 * (matrix + matrix.transpose()).eval();
    return matrix;
}

}  // namespace

std::optional<LapackEigenResult> lapack_symmetric_eigen(Eigen::MatrixXd matrix) {
    matrix = symmetrized(std::move(matrix));
    const int n = static_cast<int>(matrix.rows());
    if (n < 0 || matrix.cols() != n) {
        return std::nullopt;
    }
    if (n == 0) {
        return LapackEigenResult{};
    }

    const char jobz = 'V';
    const char range = 'A';
    const char uplo = 'U';
    const int lda = std::max(1, n);
    const int ldz = std::max(1, n);
    const double vl = 0.0;
    const double vu = 0.0;
    const int il = 0;
    const int iu = 0;
    const double abstol = -1.0;

    int info = 0;
    int m = 0;
    double work_query = 0.0;
    int iwork_query = 0;
    int lwork = -1;
    int liwork = -1;
    std::vector<double> eigenvalues(static_cast<std::size_t>(n), 0.0);
    Eigen::MatrixXd eigenvectors = Eigen::MatrixXd::Zero(n, n);
    std::vector<int> isuppz(static_cast<std::size_t>(2 * std::max(1, n)), 0);

    dsyevr_(&jobz, &range, &uplo, &n, matrix.data(), &lda, &vl, &vu, &il, &iu, &abstol, &m,
            eigenvalues.data(), eigenvectors.data(), &ldz, isuppz.data(), &work_query, &lwork,
            &iwork_query, &liwork, &info);
    if (info != 0) {
        return std::nullopt;
    }

    lwork = std::max(1, static_cast<int>(std::ceil(work_query)));
    liwork = std::max(1, iwork_query);
    std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
    std::vector<int> iwork(static_cast<std::size_t>(liwork), 0);

    dsyevr_(&jobz, &range, &uplo, &n, matrix.data(), &lda, &vl, &vu, &il, &iu, &abstol, &m,
            eigenvalues.data(), eigenvectors.data(), &ldz, isuppz.data(), work.data(), &lwork,
            iwork.data(), &liwork, &info);
    if (info != 0 || m <= 0) {
        return std::nullopt;
    }

    LapackEigenResult result;
    result.eigenvalues = Eigen::Map<Eigen::VectorXd>(eigenvalues.data(), m);
    result.eigenvectors = eigenvectors.leftCols(m);
    return result;
}

std::optional<LapackEigenResult> lapack_generalized_symmetric_eigen(Eigen::MatrixXd matrix_a,
                                                                    Eigen::MatrixXd matrix_b) {
    matrix_a = symmetrized(std::move(matrix_a));
    matrix_b = symmetrized(std::move(matrix_b));
    const int n = static_cast<int>(matrix_a.rows());
    if (n < 0 || matrix_a.cols() != n || matrix_b.rows() != n || matrix_b.cols() != n) {
        return std::nullopt;
    }
    if (n == 0) {
        return LapackEigenResult{};
    }

    const int itype = 1;
    const char jobz = 'V';
    const char uplo = 'U';
    const int lda = std::max(1, n);
    const int ldb = std::max(1, n);
    int info = 0;
    int lwork = -1;
    int liwork = -1;
    double work_query = 0.0;
    int iwork_query = 0;
    std::vector<double> eigenvalues(static_cast<std::size_t>(n), 0.0);

    dsygvd_(&itype, &jobz, &uplo, &n, matrix_a.data(), &lda, matrix_b.data(), &ldb,
            eigenvalues.data(), &work_query, &lwork, &iwork_query, &liwork, &info);
    if (info != 0) {
        return std::nullopt;
    }

    lwork = std::max(1, static_cast<int>(std::ceil(work_query)));
    liwork = std::max(1, iwork_query);
    std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
    std::vector<int> iwork(static_cast<std::size_t>(liwork), 0);

    dsygvd_(&itype, &jobz, &uplo, &n, matrix_a.data(), &lda, matrix_b.data(), &ldb,
            eigenvalues.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    if (info != 0) {
        return std::nullopt;
    }

    LapackEigenResult result;
    result.eigenvalues = Eigen::Map<Eigen::VectorXd>(eigenvalues.data(), n);
    result.eigenvectors = std::move(matrix_a);
    return result;
}

}  // namespace kspecpart
