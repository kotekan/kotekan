#define BOOST_TEST_MODULE "test_linearAlgebra"

#include "LinearAlgebra.hpp" // for DynamicHermitian, EigConvergenceStats, eigen_masked_subspace

#include "gsl-lite.hpp" // for span

#include <boost/test/included/unit_test.hpp>
#include <cmath>   // for M_PI
#include <complex> // for complex, polar
#include <cstddef> // for size_t
#include <cstdint> // for uint32_t
#include <random>  // for mt19937
#include <thread>  // for thread
#include <vector>  // for vector

#if defined(__linux__) && defined(__GLIBC__)
#include <dlfcn.h> // for dlsym, RTLD_NEXT
#endif

using cfloat = std::complex<float>;

#if defined(__linux__) && defined(__GLIBC__)
// Count the aligned allocations blaze's containers make (blaze allocates them with
// posix_memalign on this platform) that are at least `allocation_threshold` bytes.
// Interposing the function in the executable catches every call in the process.
namespace {
size_t allocation_threshold = 0;
size_t large_allocations = 0;
} // namespace

extern "C" int posix_memalign(void** ptr, size_t alignment, size_t size) {
    using posix_memalign_t = int (*)(void**, size_t, size_t);
    static const posix_memalign_t real =
        reinterpret_cast<posix_memalign_t>(dlsym(RTLD_NEXT, "posix_memalign"));
    if (size >= allocation_threshold)
        large_allocations++;
    return real(ptr, alignment, size);
}
#endif

namespace {

constexpr size_t num_elements = 16;
constexpr size_t num_ev = 2;
constexpr size_t max_iterations = 20;
constexpr float tol = 1e-6f;

// Number of threads used by the concurrency checks. Two is enough to hit a shared
// generator; more makes it hit harder.
constexpr size_t num_threads = 8;

// A rank-2 Hermitian matrix: two point sources with different fringe rates, the
// first four times as bright as the second. This is the shape of input the Eigen
// stages see. The fringe rates are whole numbers of turns across the array, so the
// two sources are exactly orthogonal and the eigenvalues are known: `4 * n`
// and `n` for an array of `n` elements.
constexpr float source_amplitude[2] = {2.0f, 1.0f};
constexpr float source_turns[2] = {1.0f, 3.0f};

DynamicHermitian<cfloat> test_matrix(size_t n = num_elements,
                                     const float (&amplitude)[2] = source_amplitude,
                                     const float (&turns)[2] = source_turns) {
    blaze::DynamicMatrix<cfloat, blaze::columnMajor> M(n, n, cfloat(0.0f));
    for (size_t s = 0; s < 2; s++) {
        const float rate = 2.0f * M_PI * turns[s] / n;
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                M(i, j) +=
                    amplitude[s] * amplitude[s] * std::polar(1.0f, rate * (float(i) - float(j)));
    }
    return blaze::declherm(M);
}

// Every input included, as in the default stage configuration.
DynamicHermitian<float> test_mask(size_t n = num_elements) {
    blaze::DynamicMatrix<float, blaze::columnMajor> M(n, n, 1.0f);
    return blaze::declherm(M);
}

// The upper triangle of a matrix, packed the way the N2 frames hold it.
std::vector<cfloat> packed_upper_triangle(const DynamicHermitian<cfloat>& A) {
    std::vector<cfloat> packed;
    for (size_t i = 0; i < A.rows(); i++)
        for (size_t j = i; j < A.columns(); j++)
            packed.push_back(A(i, j));
    return packed;
}

struct Result {
    blaze::DynamicVector<float> evals;
    blaze::DynamicMatrix<cfloat, blaze::columnMajor> evecs;
    EigConvergenceStats stats;
};

// Decompose with an explicitly supplied generator.
Result decompose(const DynamicHermitian<cfloat>& A, const DynamicHermitian<float>& W,
                 std::mt19937& rng) {
    const auto out = eigen_masked_subspace(A, W, num_ev, tol, tol, max_iterations, 0, 2, 3, rng);
    return {out.first.first, out.first.second, out.second};
}

// Decompose with a solver that keeps its workspace, as the stages do, with an
// explicitly supplied generator.
Result decompose(EigenMaskedSubspaceSolver<cfloat>& solver, const DynamicHermitian<cfloat>& A,
                 const DynamicHermitian<float>& W, std::mt19937& rng) {
    const auto stats = solver.solve(A, W, num_ev, tol, tol, max_iterations, 0, 2, 3, rng);
    return {solver.evals(), solver.evecs(), stats};
}

// Decompose using the calling thread's own generator, as the stages do, after
// resetting it so every thread starts from the same point in the sequence.
Result decompose_thread_rng(const DynamicHermitian<cfloat>& A, const DynamicHermitian<float>& W) {
    eigen_subspace_rng().seed(eigen_subspace_seed);
    const auto out = eigen_masked_subspace(A, W, num_ev, tol, tol, max_iterations);
    return {out.first.first, out.first.second, out.second};
}

// The same starting subspace must give the same answer, to the last bit.
void check_identical(const Result& a, const Result& b) {
    BOOST_REQUIRE_EQUAL(a.evals.size(), b.evals.size());
    BOOST_CHECK_EQUAL(a.stats.iterations, b.stats.iterations);
    BOOST_CHECK_EQUAL(a.stats.converged, b.stats.converged);
    for (size_t i = 0; i < a.evals.size(); i++)
        BOOST_CHECK_EQUAL(a.evals[i], b.evals[i]);
    BOOST_REQUIRE_EQUAL(a.evecs.rows(), b.evecs.rows());
    BOOST_REQUIRE_EQUAL(a.evecs.columns(), b.evecs.columns());
    for (size_t i = 0; i < a.evecs.rows(); i++)
        for (size_t j = 0; j < a.evecs.columns(); j++)
            BOOST_CHECK_EQUAL(a.evecs(i, j), b.evecs(i, j));
}

} // namespace

// The eigenvalues of the test matrix are known, so check the decomposition is
// actually solving the problem before checking that it does so reproducibly.
BOOST_AUTO_TEST_CASE(eigen_masked_subspace_recovers_sources) {
    std::mt19937 rng(eigen_subspace_seed);
    const auto r = decompose(test_matrix(), test_mask(), rng);

    BOOST_REQUIRE_EQUAL(r.evals.size(), num_ev);
    BOOST_CHECK(r.stats.converged);
    // Eigenvalues come back in ascending order.
    BOOST_CHECK_CLOSE(r.evals[num_ev - 1], 4.0f * num_elements, 1e-2);
    BOOST_CHECK_CLOSE(r.evals[num_ev - 2], 1.0f * num_elements, 1e-2);
}

// `eigen_subspace_rng` must hand each thread its own generator: this is what keeps
// two Eigen stages from racing on a shared one. Every thread starting from the same
// seed must therefore see the same sequence of draws.
BOOST_AUTO_TEST_CASE(eigen_subspace_rng_is_per_thread) {
    std::mt19937 reference(eigen_subspace_seed);
    std::vector<uint32_t> expected(1000);
    for (auto& v : expected)
        v = reference();

    std::vector<std::vector<uint32_t>> drawn(num_threads);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; t++) {
        threads.emplace_back([&drawn, t]() {
            auto& rng = eigen_subspace_rng();
            rng.seed(eigen_subspace_seed);
            drawn[t].resize(1000);
            for (auto& v : drawn[t])
                v = rng();
        });
    }
    for (auto& thread : threads)
        thread.join();

    for (size_t t = 0; t < num_threads; t++)
        BOOST_CHECK(drawn[t] == expected);
}

// A thread that has not touched its generator must still start from the fixed seed,
// so a stage's results do not depend on which thread it happens to run on.
BOOST_AUTO_TEST_CASE(eigen_subspace_rng_default_seed) {
    std::mt19937 reference(eigen_subspace_seed);
    const uint32_t expected = reference();

    uint32_t drawn = 0;
    std::thread thread([&drawn]() { drawn = eigen_subspace_rng()(); });
    thread.join();

    BOOST_CHECK_EQUAL(drawn, expected);
}

// Concurrent decompositions must not disturb each other's random draws. With a
// generator shared between threads, as `blaze::rand` uses, the threads take each
// other's numbers and the starting subspaces -- and so the results -- differ.
BOOST_AUTO_TEST_CASE(eigen_masked_subspace_concurrent_matches_serial) {
    const auto A = test_matrix();
    const auto W = test_mask();

    const auto serial = decompose_thread_rng(A, W);

    std::vector<Result> concurrent(num_threads);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; t++)
        threads.emplace_back(
            [&concurrent, &A, &W, t]() { concurrent[t] = decompose_thread_rng(A, W); });
    for (auto& thread : threads)
        thread.join();

    for (const auto& result : concurrent)
        check_identical(serial, result);
}

// The same check for a caller that supplies its own generator rather than using the
// per-thread one.
BOOST_AUTO_TEST_CASE(eigen_masked_subspace_explicit_rng_is_reproducible) {
    const auto A = test_matrix();
    const auto W = test_mask();

    std::mt19937 rng(eigen_subspace_seed);
    const auto serial = decompose(A, W, rng);

    std::vector<Result> concurrent(num_threads);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; t++)
        threads.emplace_back([&concurrent, &A, &W, t]() {
            std::mt19937 thread_rng(eigen_subspace_seed);
            concurrent[t] = decompose(A, W, thread_rng);
        });
    for (auto& thread : threads)
        thread.join();

    for (const auto& result : concurrent)
        check_identical(serial, result);
}

// A different starting subspace is allowed to take a different path, but it has to
// arrive at the same eigenvalues.
BOOST_AUTO_TEST_CASE(eigen_masked_subspace_seed_independent_result) {
    const auto A = test_matrix();
    const auto W = test_mask();

    std::mt19937 rng_a(eigen_subspace_seed);
    std::mt19937 rng_b(eigen_subspace_seed + 1);
    const auto a = decompose(A, W, rng_a);
    const auto b = decompose(A, W, rng_b);

    BOOST_REQUIRE_EQUAL(a.evals.size(), b.evals.size());
    for (size_t i = 0; i < a.evals.size(); i++)
        BOOST_CHECK_CLOSE(a.evals[i], b.evals[i], 1e-2);
}

// A solver keeps its workspace from one call to the next. That reuse must not leak
// into the results: solving again after solving something else in between has to
// give exactly what a fresh solve gives.
BOOST_AUTO_TEST_CASE(solver_reuse_matches_fresh_solve) {
    const auto A = test_matrix();
    const auto W = test_mask();
    const float other_amplitude[2] = {1.0f, 3.0f};
    const float other_turns[2] = {2.0f, 5.0f};
    const auto B = test_matrix(num_elements, other_amplitude, other_turns);

    std::mt19937 rng(eigen_subspace_seed);
    const auto fresh = decompose(A, W, rng);

    EigenMaskedSubspaceSolver<cfloat> solver;
    rng.seed(eigen_subspace_seed);
    check_identical(fresh, decompose(solver, A, W, rng));

    rng.seed(eigen_subspace_seed);
    const auto other = decompose(solver, B, W, rng);
    BOOST_CHECK(other.stats.converged);

    rng.seed(eigen_subspace_seed);
    check_identical(fresh, decompose(solver, A, W, rng));
}

// The same when the problem changes size between calls, which makes the solver
// resize its workspace.
BOOST_AUTO_TEST_CASE(solver_reuse_across_sizes) {
    const size_t larger = 2 * num_elements;
    const auto A = test_matrix();
    const auto W = test_mask();
    const auto B = test_matrix(larger);
    const auto W_B = test_mask(larger);

    std::mt19937 rng(eigen_subspace_seed);
    const auto fresh_A = decompose(A, W, rng);
    rng.seed(eigen_subspace_seed);
    const auto fresh_B = decompose(B, W_B, rng);
    BOOST_CHECK(fresh_B.stats.converged);
    BOOST_CHECK_CLOSE(fresh_B.evals[num_ev - 1], 4.0f * larger, 1e-2);

    EigenMaskedSubspaceSolver<cfloat> solver;
    for (int round = 0; round < 2; round++) {
        rng.seed(eigen_subspace_seed);
        check_identical(fresh_A, decompose(solver, A, W, rng));
        rng.seed(eigen_subspace_seed);
        check_identical(fresh_B, decompose(solver, B, W_B, rng));
    }
}

// Once a solver has solved a problem of a given size, solving another of that size
// must not allocate any blaze container again: that is what the solver is for. The
// LAPACK scratch arrays blaze's qr() and heevd() make on every call are not blaze
// containers and are allowed. The size here keeps every product below the threshold
// at which blaze's own product kernel would make packing buffers of its own.
#if defined(__linux__) && defined(__GLIBC__)
BOOST_AUTO_TEST_CASE(solver_reuse_does_not_allocate) {
    const size_t n = 512;
    const auto A = test_matrix(n);
    const auto W = test_mask(n);
    // The smallest matrix in the workspace is the n x num_ev subspace
    allocation_threshold = n * num_ev * sizeof(cfloat);

    EigenMaskedSubspaceSolver<cfloat> solver;
    std::mt19937 rng(eigen_subspace_seed);

    // Only solve() is counted: copying the results out is the test's own allocation.
    large_allocations = 0;
    const auto first_stats = solver.solve(A, W, num_ev, tol, tol, max_iterations, 0, 2, 3, rng);
    // The first call builds the workspace
    BOOST_CHECK_GT(large_allocations, 0);
    const Result first{solver.evals(), solver.evecs(), first_stats};
    BOOST_CHECK(first.stats.converged);

    for (int round = 0; round < 3; round++) {
        rng.seed(eigen_subspace_seed);
        large_allocations = 0;
        const auto stats = solver.solve(A, W, num_ev, tol, tol, max_iterations, 0, 2, 3, rng);
        BOOST_CHECK_EQUAL(large_allocations, 0);
        check_identical(first, Result{solver.evals(), solver.evecs(), stats});
    }
}
#endif

// Unpacking a frame's triangle into an existing container must fill it in place and
// give the same matrix as unpacking into a new one.
BOOST_AUTO_TEST_CASE(to_blaze_herm_reuses_container) {
    const auto A = test_matrix();
    const float other_amplitude[2] = {1.0f, 3.0f};
    const float other_turns[2] = {2.0f, 5.0f};
    const auto B = test_matrix(num_elements, other_amplitude, other_turns);
    auto packed_A = packed_upper_triangle(A);
    auto packed_B = packed_upper_triangle(B);
    const gsl_lite::span<cfloat> span_A(packed_A.data(), packed_A.size());
    const gsl_lite::span<cfloat> span_B(packed_B.data(), packed_B.size());

    DynamicHermitian<cfloat> unpacked;
    to_blaze_herm(span_B, unpacked);
    const cfloat* const storage = unpacked.data();
    to_blaze_herm(span_A, unpacked);
    BOOST_CHECK_EQUAL(unpacked.data(), storage);

    const auto fresh = to_blaze_herm(span_A);
    BOOST_REQUIRE_EQUAL(unpacked.rows(), fresh.rows());
    BOOST_REQUIRE_EQUAL(unpacked.columns(), fresh.columns());
    for (size_t i = 0; i < num_elements; i++)
        for (size_t j = 0; j < num_elements; j++) {
            BOOST_CHECK_EQUAL(unpacked(i, j), fresh(i, j));
            BOOST_CHECK_EQUAL(unpacked(i, j), A(i, j));
        }
}
