/**
 * @brief Tools for Linear Algebra.
 **/

#ifndef LINEARALGEBRA_HPP
#define LINEARALGEBRA_HPP

#include "visUtil.hpp"

#include <blaze/Blaze.h>

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

    auto At = ~A;
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

    auto At = ~A;
    auto* rd = At.data();
    size_t n = At.size();

    for (unsigned int i = 0; i < n; i++) {
        t += fast_norm(rd[i]);
    }
    return std::sqrt(t / n);
}


/**
 * @brief Orthonormalise a set of vectors using QR.
 *
 * @param  x  Vectors to orthonormalise.
 *
 * @return    Orthonormal vectors.
 **/
template<typename MT>
blaze::DynamicMatrix<MT, blaze::columnMajor>
orth(const blaze::DynamicMatrix<MT, blaze::columnMajor>& x) {
    blaze::DynamicMatrix<MT, blaze::columnMajor> Q, R;
    qr(x, Q, R);
    return Q;
}


/**
 * @brief Find eigenvalues of a subspace with the Ritz method.
 *
 * @param  A  A Hermitian matrix to find eigenpairs of.
 * @param  V  The subspace to decompose. A set of *row* vectors.
 *
 * @return    Pair of evals and evecs (row vectors).
 **/
template<typename MT>
eig_t<MT> ritz(const DynamicHermitian<MT>& A,
               const blaze::DynamicMatrix<MT, blaze::columnMajor>& V) {
    blaze::DynamicVector<real_t<MT>> evals;
    blaze::DynamicMatrix<MT, blaze::columnMajor> evecst;

    auto Vt = orth(V);
    // auto At = blaze::evaluate(blaze::declherm(blaze::ctrans(V) * A * V));
    DynamicHermitian<MT> At = blaze::declherm(blaze::ctrans(Vt) * A * Vt);
    blaze::eigen(At, evals, evecst);

    return {evals, Vt * evecst};
}


/**
 * @brief Construct the p-dimensional block Krylov subspace,
 * i.e. {V, A V, A^2 V, ..., A^{p-1} V}
 *
 * @param  A  Operator to use.
 * @param  V  Set of vectors (column wise)
 * @param  p  The number of iterations performed.
 *
 * @return    The block Krylov subspace.
 **/
template<typename MT>
blaze::DynamicMatrix<MT, blaze::columnMajor>
krylov(const DynamicHermitian<MT>& A, const blaze::DynamicMatrix<MT, blaze::columnMajor>& V,
       unsigned int p) {
    size_t nr = V.rows();
    size_t nc = V.columns();

    // Create matrix for holding Krylov subspace and set the first block
    // TODO: this will probably break if the number of elements is not a
    // multiple of eight
    blaze::DynamicMatrix<MT, blaze::columnMajor> K{nr, nc * p};
    blaze::submatrix<blaze::aligned>(K, 0, 0, nr, nc) = V;

    // Iteratively apply A and set the result
    for (unsigned int i = 1; i < p; i++) {
        auto X = blaze::submatrix<blaze::aligned>(K, 0, nc * (i - 1), nr, nc);
        blaze::submatrix<blaze::aligned>(K, 0, i * nc, nr, nc) = A * X;
    }
    return K;
}


/**
 * @brief Find eigenvalues of a subspace with an augmented Ritz method.
 *
 * This expands the subspace using a block Krylov method, performs a Ritz
 * method, before returning only the highest eigenpairs from.
 *
 * @param  A  A Hermitian matrix to find eigenpairs of.
 * @param  V  The subspace to decompose. A set of *row* vectors.
 * @param  p  The number of iterations performed.
 *
 * @return    Pair of evals and evecs (row vectors).
 **/
template<typename MT>
eig_t<MT> augmented_ritz(const DynamicHermitian<MT>& A,
                         const blaze::DynamicMatrix<MT, blaze::columnMajor>& V, unsigned int p) {
    size_t nr = V.rows();
    size_t nc = V.columns();

    // Form the Krylov subspace and perform a Ritz step
    auto K = krylov(A, V, p);
    auto epair = ritz(A, K);

    // Set the phase degeneracy if it exists
    auto& evec = epair.second;
    std::vector<MT> norm;
    for (unsigned int j = 0; j < evec.columns(); j++) {
        MT z = evec(0, j);
        norm.push_back(std::conj(z) / std::abs(z));
    }
    for (unsigned int i = 0; i < evec.rows(); i++) {
        for (unsigned int j = 0; j < evec.columns(); j++) {
            evec(i, j) *= norm[j];
        }
    }

    // Create views of the relevant eigenvalues and vectors, these should get
    // turned into Dynamic arrays by the implicit copy constructor
    return {blaze::subvector(epair.first, (p - 1) * nc, nc),
            blaze::submatrix(epair.second, 0, (p - 1) * nc, nr, nc)};
}


/**
 * @brief Construct a rank-N approx from eigenpairs.
 *
 * @param  eigpair  Eigenvalues and vectors.
 *
 * @return          The approximate matrix.
 **/
template<typename MT>
DynamicHermitian<MT> expand_rankN(const eig_t<MT>& eigpair) {
    // Create an array with the eigenvalues on the diagonal
    blaze::DiagonalMatrix<blaze::DynamicMatrix<MT, blaze::columnMajor>> L;
    L.resize(eigpair.second.columns());
    blaze::diagonal(L) = eigpair.first;

    // Return thr rank-N approximation
    return blaze::declherm(eigpair.second * L * blaze::ctrans(eigpair.second));
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

/**
 * @brief Find a low rank decomposition of a masked matrix.
 *
 * Method based on one described in Wen and Zhang 2017
 * (https://doi.org/10.1137/16M1058534). Also inspired by the suggestion in
 * Saad 2017 (http://dx.doi.org/10.1137/141002037).
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
 *
 * @return           The estimated eigenpairs.
 **/
template<typename MT>
std::pair<eig_t<MT>, EigConvergenceStats>
eigen_masked_subspace(const DynamicHermitian<MT>& A,
                      const DynamicHermitian<float>& W, // Should this be symmetric
                      size_t k, float tol_eval, float tol_evec, size_t maxiter, size_t k_conv = 0,
                      size_t p = 2, size_t q = 3) {
    blaze::DynamicVector<real_t<MT>> evals, evalsp, etols;
    blaze::DynamicMatrix<MT, blaze::columnMajor> V, Vp;

    // Mask out
    auto Am = blaze::evaluate(blaze::declherm(A % W));
    size_t n = Am.columns();

    // Set k_conv appropriately
    k_conv = k_conv == 0 ? k : k_conv;

    // Initialise (randomly the vector array)
    V.resize(n, k);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < k; j++) {
            V(i, j) = blaze::rand<MT>();
        }
    }
    V = orth(V);

    // Initialise loop variables for holding the previous state
    Vp = V;
    evalsp.resize(k);
    evalsp = 0.0;
    etols.resize(k);
    etols = tol_evec;

    EigConvergenceStats stats;
    for (stats.iterations = 0; !stats.converged && stats.iterations < maxiter; stats.iterations++) {

        // Perform the subspace iteration steps
        for (unsigned int ss_ind = 0; ss_ind < q; ss_ind++) {
            V = orth(blaze::evaluate(A * V));
        }

        // Calculate the eigenpairs
        auto eigpair = augmented_ritz(Am, V, p);
        std::tie(evals, V) = eigpair;

        // Back fill the missing entries of the array
        auto Ar = expand_rankN(eigpair);
        Am = A % W + Ar - Ar % W;

        // Calculate the eigenvector convergence (L1 norm of the tested subset)
        // NOTE: there seems to be a bug in Blaze's L1 norm function so we
        // calculate it directly
        auto evec_conv = blaze::evaluate(blaze::ctrans(Vp) * V);
        for (auto& d : blaze::diagonal(evec_conv))
            d -= 1.0;
        // stats.eps_evec = rms(
        //     blaze::submatrix(evec_conv, k - k_conv, k - k_conv, k_conv, k_conv)
        // );
        stats.eps_evec = blaze::sum(blaze::abs(
                             blaze::submatrix(evec_conv, k - k_conv, k - k_conv, k_conv, k_conv)))
                         / (k_conv * k_conv);

        // Calculate the eigenvalue convergence (Summed fractional change in eigenvalues)
        stats.eps_eval = rms(blaze::evaluate(blaze::subvector((evalsp - evals)
                                                              / (blaze::abs(evals) + etols), k - k_conv, k_conv)));

        evalsp = evals;
        Vp = V;

        // Check convergence
        if (stats.eps_eval < tol_eval && stats.eps_evec < tol_evec) {
            stats.converged = true;
        }
        // For debugging:
        // for (uint ii = 0; ii < k; ii++) {
        //     std::cout << evals[ii] << " ";
        // }
        // std::cout << std::endl;
    }

    // Calculate the RMS
    // TODO: the blaze norm implementation is slow and naive. This is better.
    eig_t<MT> eigpair = std::make_pair(evals, V);
    stats.rms = rms(blaze::evaluate(W % (A - expand_rankN(eigpair))));
    stats.rms *= A.rows() / std::sqrt(blaze::sum(W)); // Re-norm to account for masking


    return {eigpair, stats};
}

/**
 * @brief Copy a packed Hermitian matrix into a blaze container.
 *
 * @param  data  Hermitian matrix packed as upper triangle.
 *
 * @return       The blaze matrix.
 **/
template<typename MT>
DynamicHermitian<MT> to_blaze_herm(const gsl::span<MT>& data) {
    size_t N = (size_t)std::sqrt(2 * data.size());

    DynamicHermitian<MT> A;
    A.resize(N);

    // TODO: check how much overhead is in this step. The cache overhead
    // must be terrible. Might want to do a raw, blocked method
    int ind = 0;
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = i; j < N; j++) {
            A(i, j) = data[ind];
            ind++;
        }
    }

    return A;
}


#endif // LINEARALGEBRA_HPP
