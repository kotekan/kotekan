/**
 * @brief Tools for Linear Algebra.
 **/

#ifndef LINEARALGEBRA_HPP
#define LINEARALGEBRA_HPP

#include "visUtil.hpp"

#include <algorithm> // for min
#include <blaze/Blaze.h>
#include <cmath>       // for sqrt
#include <complex>     // for complex, conj, abs
#include <cstdint>     // for uint32_t
#include <random>      // for mt19937, uniform_real_distribution
#include <stdexcept>   // for invalid_argument
#include <string>      // for to_string
#include <type_traits> // for is_same_v, decay_t
#include <utility>     // for pair, declval
#include <vector>      // for vector

// Type defs for simplicity
// Map complex types to their real equivalent
template<typename T>
struct eigenval_type {
    typedef T type;
};
template<typename T>
struct eigenval_type<std::complex<T>> {
    typedef T type;
};

// A type alias for a set of eigenpairs
template<typename MT>
using real_t = typename eigenval_type<MT>::type;

template<typename MT>
using eig_t =
    std::pair<blaze::DynamicVector<real_t<MT>>, blaze::DynamicMatrix<MT, blaze::columnMajor>>;


template<typename MT>
using DynamicHermitian = blaze::HermitianMatrix<blaze::DynamicMatrix<MT, blaze::columnMajor>>;

/**
 * @brief Calculate the root-mean-square quickly.
 *
 * @param  A  Matrix to calculate RMS of.
 *
 * @returns   The RMS.
 **/
template<typename MT, bool SO>
double rms(const blaze::DenseMatrix<MT, SO>& A) {
    double t = 0.0;

    auto At = blaze::trans(A);
    auto* rd = At.data();
    size_t n = At.rows() * At.columns();

    for (unsigned int i = 0; i < n; i++) {
        t += fast_norm(rd[i]);
    }
    return std::sqrt(t / n);
}
template<typename MT, bool TF>
double rms(const blaze::DenseVector<MT, TF>& A) {
    double t = 0.0;

    auto At = blaze::trans(A);
    auto* rd = At.data();
    size_t n = At.size();

    for (unsigned int i = 0; i < n; i++) {
        t += fast_norm(rd[i]);
    }
    return std::sqrt(t / n);
}


/**
 * @brief Describe the convergence of an eigendecomposition.
 **/
struct EigConvergenceStats {

    /// Did the estimation converge?
    bool converged = false;

    /// Number iterations were performed.
    unsigned int iterations = 0;

    /// Convergence of eigenvalues
    double eps_eval = 0.0;

    /// Convergence of eigenvectors
    double eps_evec = 0.0;

    /// RMS of residuals
    double rms = 0.0;
};

/// Seed given to each thread's copy of `eigen_subspace_rng`.
constexpr uint32_t eigen_subspace_seed = 0x9e3779b9;

/**
 * @brief The random number generator used to initialise the subspace iteration.
 *
 * Every thread gets its own generator, seeded identically. Blaze's `rand` cannot be
 * used for this: it draws from a single static generator (`Random<RNG>::rng_` in
 * blaze/util/Random.h) that is neither `thread_local` nor mutex protected, so two
 * Eigen stages drawing their starting subspaces at the same time would be a data race
 * on its state.
 *
 * The fixed seed also makes the stages reproducible: a given stage draws the same
 * sequence of starting subspaces on every run.
 *
 * @return  The calling thread's generator.
 **/
inline std::mt19937& eigen_subspace_rng() {
    static thread_local std::mt19937 rng(eigen_subspace_seed);
    return rng;
}

/**
 * @brief Draw one random matrix element uniformly from [0, 1).
 *
 * That is the range `blaze::rand` uses, for both parts of a complex value.
 *
 * @param  rng  The generator to draw from.
 *
 * @return      The random element.
 **/
template<typename MT>
MT rand_subspace_element(std::mt19937& rng) {
    std::uniform_real_distribution<real_t<MT>> dist(0.0, 1.0);
    if constexpr (std::is_same_v<MT, real_t<MT>>) {
        return dist(rng);
    } else {
        // Draw in a fixed order. `blaze::rand<complex<T>>` passes two draws as
        // function arguments, whose evaluation order the compiler chooses.
        const real_t<MT> re = dist(rng);
        const real_t<MT> im = dist(rng);
        return MT(re, im);
    }
}

/**
 * @brief Low rank decomposition of a masked matrix, keeping its workspace between calls.
 *
 * Method based on one described in Wen and Zhang 2017
 * (https://doi.org/10.1137/16M1058534). Also inspired by the suggestion in
 * Saad 2017 (http://dx.doi.org/10.1137/141002037). A random starting subspace is
 * refined by subspace iteration on the matrix, with the eigenpairs read off by a
 * Rayleigh-Ritz step on a block Krylov extension of the subspace, and the masked
 * entries of the matrix progressively filled in from the current low rank estimate.
 *
 * Every matrix the iteration works on lives in this object, several of them of the
 * full n x n size: the masked and filled matrix, the rank-k reconstruction, and the
 * subspace, Krylov and Ritz intermediates that blaze would otherwise allocate for
 * itself as temporaries. They are allocated the first time solve() runs for a given
 * problem size and are reused by every later call of that size, so a caller decomposing
 * a stream of same-sized matrices, as the Eigen stages do, allocates once rather than
 * once per matrix (and pays once, rather than per matrix, for faulting in the pages
 * behind them). What solve() still allocates per call is what blaze allocates inside
 * its own routines: the LAPACK scratch arrays of qr() and heevd(), and, for the n x k
 * subspace products when a thread's share of one is above blaze's large-kernel
 * threshold, the packing buffers of blaze's own product kernel. Both are of the order
 * of the subspace size, not the matrix size.
 *
 * Reusing the workspace does not change the results: every call draws its starting
 * subspace afresh from the generator it is given, and each workspace matrix is written
 * in full before it is read. The arithmetic follows the allocating implementation this
 * replaces step for step, with one exception. The n x n rank-k reconstruction is formed
 * with BLAS gemm rather than blaze's own product kernel, which kotekan's
 * BLAZE_BLAS_IS_PARALLEL=0 build routes every large product through and which allocates
 * packing buffers of a few MB on every call. The two round differently, so the
 * eigenpairs agree with the old implementation's to a few parts per million rather
 * than to the last bit.
 *
 * The workspace is not synchronised: use one solver per thread.
 **/
template<typename MT>
class EigenMaskedSubspaceSolver {
    static_assert(std::is_same_v<MT, std::complex<real_t<MT>>>,
                  "EigenMaskedSubspaceSolver decomposes complex Hermitian matrices; the phase "
                  "it fixes in each eigenvector has no real counterpart.");

public:
    using real_type = real_t<MT>;
    using vector_type = blaze::DynamicVector<real_type>;
    using matrix_type = blaze::DynamicMatrix<MT, blaze::columnMajor>;

    /**
     * @brief Find the k largest eigenpairs of a masked matrix.
     *
     * @param  A         The matrix to decompose.
     * @param  W         A mask matrix. One includes an elements, zero excludes it.
     * @param  k         The number of eigenpairs to return.
     * @param  tol_eval  The fractional tolerance for the convergence check.
     * @param  tol_evec  The fractional tolerance for the convergence check.
     * @param  maxiter   Maximum number of iterations. Must be at least one.
     * @param  k_conv    The number of eigenpairs to use for the convergence check. If
     *                   zero, use all eigenpairs.
     * @param  p         Size of the Krylov subspace in the augmented Ritz. The Krylov
     *                   subspace holds k * p vectors, which cannot exceed the size of
     *                   the matrix.
     * @param  q         Number of subspace updates per iteration.
     * @param  rng       Generator for the random starting subspace. Defaults to the
     *                   calling thread's generator.
     *
     * @return           How the iteration converged. The eigenpairs themselves are
     *                   available from evals() and evecs() until the next call.
     *
     * @throws std::invalid_argument  If the sizes or parameters are inconsistent.
     * @throws std::runtime_error     If a LAPACK routine fails.
     **/
    EigConvergenceStats solve(const DynamicHermitian<MT>& A, const DynamicHermitian<float>& W,
                              size_t k, float tol_eval, float tol_evec, size_t maxiter,
                              size_t k_conv = 0, size_t p = 2, size_t q = 3,
                              std::mt19937& rng = eigen_subspace_rng());

    /// The eigenvalues found by the last solve(), in ascending order.
    const vector_type& evals() const {
        return evals_;
    }

    /// The eigenvectors found by the last solve(), one per column, ordered as evals().
    const matrix_type& evecs() const {
        return V_;
    }

private:
    // The types blaze evaluates these expressions into when left to make its own
    // temporaries; the workspace uses the same ones so the same kernels run on it. The
    // conjugate transpose of a column-major matrix is row-major, for one.
    template<typename E>
    using result_of_t = blaze::ResultType_t<std::decay_t<E>>;
    using diagonal_type = blaze::DiagonalMatrix<matrix_type>;
    using ctrans_type = result_of_t<decltype(blaze::ctrans(std::declval<const matrix_type&>()))>;
    using iterate_type = result_of_t<decltype(std::declval<const DynamicHermitian<MT>&>()
                                              * std::declval<const matrix_type&>())>;
    using projection_type = result_of_t<decltype(std::declval<const ctrans_type&>()
                                                 * std::declval<const matrix_type&>())>;
    using scaled_type = result_of_t<decltype(std::declval<const matrix_type&>()
                                             * std::declval<const diagonal_type&>())>;
    using overlap_type = result_of_t<decltype(std::declval<const ctrans_type&>()
                                              * std::declval<const matrix_type&>())>;

    /// Size the workspace for an n x n matrix, k eigenpairs, a Krylov factor of p and
    /// k_conv eigenpairs tested for convergence. This is the only place solve()
    /// allocates, and it only does so when a size changes.
    void resize(size_t n, size_t k, size_t p, size_t k_conv);

    /// Replace V_ by an orthonormal basis of the columns of X, which may be V_ itself.
    template<typename XT>
    void orthonormalise(const XT& X);

    /// The augmented Ritz step: extend V_ to the block Krylov subspace of Am_, find the
    /// eigenpairs of Am_ within it, and keep the k of largest eigenvalue as evals_ and V_.
    void augmented_ritz();

    /// Refill the masked entries of Am_ from the rank-k reconstruction of the current
    /// eigenpairs, which is left in Ar_.
    void backfill(const DynamicHermitian<MT>& A, const DynamicHermitian<float>& W);

    /// The problem size the workspace is sized for
    size_t n_ = 0, k_ = 0, p_ = 0;

    /// The masked matrix, with its masked entries filled from the current estimate (n x n)
    matrix_type Am_;
    /// The rank-k reconstruction from the current eigenpairs (n x n)
    matrix_type Ar_;

    /// The current subspace (n x k); on return, the eigenvectors
    matrix_type V_;
    /// A * V (n x k)
    iterate_type AV_;
    /// The Q factor used to orthonormalise an n x k subspace, and the R factor of every
    /// QR decomposition, which is never read (kp x kp)
    matrix_type Q_, R_;

    /// The block Krylov subspace (n x kp) ...
    matrix_type K_;
    /// ... its Q factor (n x kp) ...
    matrix_type QK_;
    /// ... the conjugate transpose of its orthonormal basis (kp x n) ...
    ctrans_type QKh_;
    /// ... that basis projected onto Am (kp x n) ...
    projection_type KhA_;
    /// ... the projected matrix (kp x kp), overwritten by its eigenvectors, and its
    /// eigenvalues ...
    matrix_type At_;
    vector_type evals_kp_;
    /// ... and the Ritz vectors (n x kp)
    matrix_type Vfull_;

    /// The eigenvalues on a diagonal (k x k), V * L (n x k) and the conjugate transpose
    /// of V (k x n), from which Ar_ is formed
    diagonal_type L_;
    scaled_type VL_;
    ctrans_type Vh_;

    /// The conjugate transpose of the previous iteration's subspace (k x n)
    ctrans_type Vph_;
    /// Overlap of the previous and current subspaces (k x k)
    overlap_type evec_conv_;

    /// The current and previous eigenvalues, the eigenvalue tolerances, and the
    /// fractional eigenvalue changes tested for convergence
    vector_type evals_, evalsp_, etols_, evconv_;
};


template<typename MT>
void EigenMaskedSubspaceSolver<MT>::resize(size_t n, size_t k, size_t p, size_t k_conv) {
    n_ = n;
    k_ = k;
    p_ = p;
    const size_t kp = k * p;

    // Blaze skips a resize to the current size. Otherwise the plain matrices and vectors
    // only reallocate when their capacity is too small, and with `preserve` false do not
    // copy; the one adaptor, L_, copies on every size change, but is k x k.
    Am_.resize(n, n, false);
    Ar_.resize(n, n, false);

    V_.resize(n, k, false);
    AV_.resize(n, k, false);
    Q_.resize(n, k, false);
    R_.resize(kp, kp, false);

    K_.resize(n, kp, false);
    QK_.resize(n, kp, false);
    QKh_.resize(kp, n, false);
    KhA_.resize(kp, n, false);
    At_.resize(kp, kp, false);
    evals_kp_.resize(kp, false);
    Vfull_.resize(n, kp, false);

    L_.resize(k, false);
    VL_.resize(n, k, false);
    Vh_.resize(k, n, false);

    Vph_.resize(k, n, false);
    evec_conv_.resize(k, k, false);

    evals_.resize(k, false);
    evalsp_.resize(k, false);
    etols_.resize(k, false);
    evconv_.resize(k_conv, false);
}


template<typename MT>
template<typename XT>
void EigenMaskedSubspaceSolver<MT>::orthonormalise(const XT& X) {
    // Handing qr() the spare n x k buffer and swapping it in is blaze's own move of the
    // Q factor.
    blaze::qr(X, Q_, R_);
    blaze::swap(V_, Q_);
}


template<typename MT>
void EigenMaskedSubspaceSolver<MT>::augmented_ritz() {
    // The columns of the Krylov subspace holding the k eigenpairs of the largest
    // eigenvalues, which LAPACK returns last
    const size_t top = (p_ - 1) * k_;

    // Construct the p-dimensional block Krylov subspace, i.e. {V, A V, A^2 V, ..., A^{p-1} V}
    blaze::submatrix<blaze::aligned>(K_, 0, 0, n_, k_) = V_;
    for (unsigned int i = 1; i < p_; i++) {
        auto X = blaze::submatrix<blaze::aligned>(K_, 0, k_ * (i - 1), n_, k_);
        blaze::submatrix<blaze::aligned>(K_, 0, i * k_, n_, k_) = Am_ * X;
    }

    // Find the eigenpairs of the Krylov subspace with the Ritz method. The projected
    // matrix is formed as a plain product and heevd reads its lower triangle, which is
    // what blaze::eigen does with a Hermitian matrix, minus the temporary copy it makes
    // of it; the eigenvectors overwrite it.
    blaze::qr(K_, QK_, R_);
    QKh_ = blaze::ctrans(QK_);
    KhA_ = QKh_ * Am_;
    At_ = KhA_ * QK_;
    blaze::heevd(At_, evals_kp_, 'V', 'L');
    Vfull_ = QK_ * At_;

    // Keep the highest eigenpairs
    evals_ = blaze::subvector(evals_kp_, top, k_);
    V_ = blaze::submatrix(Vfull_, 0, top, n_, k_);

    // Set the phase degeneracy if it exists, making the first element of each
    // eigenvector real. An input whose visibilities are all zero leaves that element
    // zero, with no phase to fix.
    for (size_t j = 0; j < k_; j++) {
        const MT z = V_(0, j);
        const real_type magnitude = std::abs(z);
        if (magnitude > 0)
            blaze::column(V_, j) *= std::conj(z) / magnitude;
    }
}


template<typename MT>
void EigenMaskedSubspaceSolver<MT>::backfill(const DynamicHermitian<MT>& A,
                                             const DynamicHermitian<float>& W) {
    blaze::diagonal(L_) = evals_;
    VL_ = V_ * L_;
    Vh_ = blaze::ctrans(V_);

    // The reconstruction is formed with BLAS gemm: blaze's own product kernel, which this
    // build uses for every large product, allocates packing buffers of a few MB on every
    // call, and this is the one product here big enough for that to matter. One gemm per
    // column block, on the threads blaze itself would use, keeps it parallel.
    const size_t nblocks = std::min<size_t>(blaze::getNumThreads(), n_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t b = 0; b < nblocks; b++) {
        const size_t first = b * n_ / nblocks;
        const size_t width = (b + 1) * n_ / nblocks - first;
        auto C = blaze::submatrix(Ar_, 0, first, n_, width);
        const auto B = blaze::submatrix(Vh_, 0, first, k_, width);
        blaze::gemm(C, VL_, B, MT(1), MT(0));
    }

    // Back fill the missing entries of the array: the masked entries take the
    // reconstruction, the rest the data. The reconstruction is Hermitian only to
    // rounding, so it and the filled matrix are kept as plain matrices.
    Am_ = Ar_ + (A - Ar_) % W;
}


template<typename MT>
EigConvergenceStats EigenMaskedSubspaceSolver<MT>::solve(const DynamicHermitian<MT>& A,
                                                         const DynamicHermitian<float>& W, size_t k,
                                                         float tol_eval, float tol_evec,
                                                         size_t maxiter, size_t k_conv, size_t p,
                                                         size_t q, std::mt19937& rng) {
    const size_t n = A.columns();

    if (W.columns() != n)
        throw std::invalid_argument("The mask is " + std::to_string(W.columns())
                                    + " square, but the matrix is " + std::to_string(n)
                                    + " square.");
    if (k == 0 || k > n)
        throw std::invalid_argument("Cannot find " + std::to_string(k) + " eigenpairs of a "
                                    + std::to_string(n) + " square matrix.");
    if (p == 0)
        throw std::invalid_argument("The Krylov subspace size must be at least one.");
    if (k * p > n)
        throw std::invalid_argument("The Krylov subspace of " + std::to_string(k * p)
                                    + " vectors cannot exceed the matrix size of "
                                    + std::to_string(n) + ".");
    if (maxiter == 0)
        throw std::invalid_argument("At least one iteration is needed to find eigenpairs.");

    // Set k_conv appropriately
    k_conv = k_conv == 0 ? k : k_conv;
    if (k_conv > k)
        throw std::invalid_argument("Cannot test " + std::to_string(k_conv)
                                    + " eigenpairs for convergence when only " + std::to_string(k)
                                    + " are found.");

    resize(n, k, p, k_conv);

    // Mask out
    Am_ = A % W;

    // Initialise (randomly the vector array) and orthonormalise it
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < k; j++) {
            V_(i, j) = rand_subspace_element<MT>(rng);
        }
    }
    orthonormalise(V_);

    // Initialise loop variables for holding the previous state. The convergence check
    // only needs the previous subspace conjugate transposed, so that is what is kept.
    Vph_ = blaze::ctrans(V_);
    evalsp_ = 0.0;
    etols_ = tol_evec;

    EigConvergenceStats stats;
    for (stats.iterations = 0; !stats.converged && stats.iterations < maxiter; stats.iterations++) {

        // Perform the subspace iteration steps
        for (unsigned int ss_ind = 0; ss_ind < q; ss_ind++) {
            AV_ = A * V_;
            orthonormalise(AV_);
        }

        // Calculate the eigenpairs, and back fill the missing entries of the array
        augmented_ritz();
        backfill(A, W);

        // Calculate the eigenvector convergence (L1 norm of the tested subset)
        // NOTE: there seems to be a bug in Blaze's L1 norm function so we
        // calculate it directly
        evec_conv_ = Vph_ * V_;
        for (auto& d : blaze::diagonal(evec_conv_))
            d -= 1.0;
        stats.eps_evec = blaze::sum(blaze::abs(
                             blaze::submatrix(evec_conv_, k - k_conv, k - k_conv, k_conv, k_conv)))
                         / (k_conv * k_conv);

        // Calculate the eigenvalue convergence (Summed fractional change in eigenvalues)
        evconv_ = blaze::subvector((evalsp_ - evals_) / (blaze::abs(evals_) + etols_), k - k_conv,
                                   k_conv);
        stats.eps_eval = rms(evconv_);

        evalsp_ = evals_;
        // This iteration's conjugate transposed subspace is the next one's previous
        blaze::swap(Vph_, Vh_);

        // Check convergence
        if (stats.eps_eval < tol_eval && stats.eps_evec < tol_evec) {
            stats.converged = true;
        }
    }

    // Calculate the RMS of the masked residual. Ar_ still holds the reconstruction from
    // the final eigenpairs, and Am_ is free to hold the residual.
    // TODO: the blaze norm implementation is slow and naive. This is better.
    Am_ = W % (A - Ar_);
    stats.rms = rms(Am_);
    stats.rms *= A.rows() / std::sqrt(blaze::sum(W)); // Re-norm to account for masking

    return stats;
}


/**
 * @brief Find a low rank decomposition of a masked matrix.
 *
 * A convenience wrapper around EigenMaskedSubspaceSolver for a one-off decomposition.
 * It allocates a fresh workspace on every call, so a caller decomposing many matrices
 * of one size should keep a solver instead.
 *
 * @param  A         The matrix to decompose.
 * @param  W         A mask matrix. One includes an elements, zero excludes it.
 * @param  k         The number of eigenpairs to return.
 * @param  tol_eval  The fractional tolerance for the convergence check.
 * @param  tol_evec  The fractional tolerance for the convergence check.
 * @param  maxiter   Maximum number of iterations.
 * @param  k_conv    The number of eigenpairs to use for the convergence check. If
 *                   zero, use all eigenpairs.
 * @param  p         Size of the Krylov subspace in the augmented Ritz.
 * @param  q         Number of subspace updates per iteration.
 * @param  rng       Generator for the random starting subspace. Defaults to the
 *                   calling thread's generator.
 *
 * @return           The estimated eigenpairs.
 **/
template<typename MT>
std::pair<eig_t<MT>, EigConvergenceStats>
eigen_masked_subspace(const DynamicHermitian<MT>& A,
                      const DynamicHermitian<float>& W, // Should this be symmetric
                      size_t k, float tol_eval, float tol_evec, size_t maxiter, size_t k_conv = 0,
                      size_t p = 2, size_t q = 3, std::mt19937& rng = eigen_subspace_rng()) {
    EigenMaskedSubspaceSolver<MT> solver;
    const EigConvergenceStats stats =
        solver.solve(A, W, k, tol_eval, tol_evec, maxiter, k_conv, p, q, rng);
    return {{solver.evals(), solver.evecs()}, stats};
}

/**
 * @brief Build the mask of a matrix to decompose, in place.
 *
 * One includes an element in the decomposition, zero leaves it to be filled in from
 * the low rank estimate. The mask is written through the Hermitian adaptor's element
 * access, which sets each entry and its transpose together, so an existing mask of the
 * right size is rebuilt without allocating.
 *
 * @param  mask            The mask to build, resized if it is not num_elements square.
 * @param  num_elements    Number of elements in the matrix.
 * @param  exclude_inputs  Inputs whose rows and columns are masked out. Must be in range.
 * @param  flags           Per-element flags: an input with a zero flag is masked out
 *                         like an excluded one. Empty for no flags.
 * @param  diagonal_bands  Ranges [first, second) of diagonal bands to mask out, with 0
 *                         the main diagonal; sub-diagonals are masked with their
 *                         super-diagonals. Each range must satisfy first <= second <=
 *                         num_elements.
 * @param  block_size      If not zero, mask out blocks of this size on the diagonal.
 **/
inline void fill_mask(DynamicHermitian<float>& mask, size_t num_elements,
                      const std::vector<size_t>& exclude_inputs, const std::vector<float>& flags,
                      const std::vector<std::pair<size_t, size_t>>& diagonal_bands,
                      size_t block_size) {
    if (mask.rows() != num_elements)
        mask.resize(num_elements, false);

    // Include everything ...
    for (size_t i = 0; i < num_elements; i++)
        for (size_t j = i; j < num_elements; j++)
            mask(i, j) = 1.0f;

    // ... then zero out the rows and columns of excluded and flagged inputs ...
    for (size_t i = 0; i < num_elements; i++) {
        const bool excluded =
            std::find(exclude_inputs.begin(), exclude_inputs.end(), i) != exclude_inputs.end();
        const bool flagged = i < flags.size() && flags[i] == 0.0f;
        if (!excluded && !flagged)
            continue;
        for (size_t j = 0; j < num_elements; j++)
            mask(i, j) = 0.0f;
    }

    // ... the diagonal bands, super- and sub-diagonal together ...
    for (const auto& band : diagonal_bands)
        for (size_t b = band.first; b < band.second; b++)
            for (size_t i = 0; i + b < num_elements; i++)
                mask(i, i + b) = 0.0f;

    // ... and the blocks on the diagonal.
    if (block_size > 0) {
        for (size_t start = 0; start < num_elements; start += block_size) {
            const size_t end = std::min(num_elements, start + block_size);
            for (size_t i = start; i < end; i++)
                for (size_t j = i; j < end; j++)
                    mask(i, j) = 0.0f;
        }
    }
}

/**
 * @brief Copy a packed Hermitian matrix into an existing blaze container.
 *
 * The container is resized only if it does not already have the matrix's size, so a
 * caller unpacking a stream of same-sized matrices into the same container allocates
 * once.
 *
 * @param  data  Hermitian matrix packed as upper triangle.
 * @param  A     The blaze matrix to fill.
 **/
template<typename MT>
void to_blaze_herm(const gsl_lite::span<MT>& data, DynamicHermitian<MT>& A) {
    size_t N = (size_t)std::sqrt(2 * data.size());

    if (A.rows() != N)
        A.resize(N, false);

    // TODO: check how much overhead is in this step. The cache overhead
    // must be terrible. Might want to do a raw, blocked method
    int ind = 0;
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = i; j < N; j++) {
            A(i, j) = data[ind];
            ind++;
        }
    }
}

/**
 * @brief Copy a packed Hermitian matrix into a blaze container.
 *
 * @param  data  Hermitian matrix packed as upper triangle.
 *
 * @return       The blaze matrix.
 **/
template<typename MT>
DynamicHermitian<MT> to_blaze_herm(const gsl_lite::span<MT>& data) {
    DynamicHermitian<MT> A;
    to_blaze_herm(data, A);
    return A;
}


#endif // LINEARALGEBRA_HPP
