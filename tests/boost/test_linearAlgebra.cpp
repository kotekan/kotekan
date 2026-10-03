#define BOOST_TEST_MODULE "test_linearAlgebra"

#include "LinearAlgebra.hpp" // for DynamicHermitian, EigConvergenceStats, eigen_masked_subspace

#include "gsl-lite.hpp" // for span

#include <atomic> // for atomic
#include <boost/test/included/unit_test.hpp>
#include <cblas.h>   // for openblas_set_num_threads
#include <cmath>     // for M_PI
#include <complex>   // for complex, polar
#include <cstddef>   // for size_t
#include <cstdint>   // for uint32_t
#include <random>    // for mt19937
#include <stdexcept> // for invalid_argument
#include <string>    // for string, to_string
#include <thread>    // for thread
#include <vector>    // for vector

#if defined(__linux__) && defined(__GLIBC__)
#include <dlfcn.h> // for dlsym, RTLD_NEXT
#endif

using cfloat = std::complex<float>;

#if defined(__linux__) && defined(__GLIBC__)
// Count the aligned allocations blaze's containers make (blaze allocates them with
// posix_memalign on this platform) that are at least `allocation_threshold` bytes.
// Interposing the function in the executable catches every call in the process, so
// counting is switched on per thread and only while a test wants it, to keep other
// threads' allocations out of the count.
namespace {
std::atomic<size_t> allocation_threshold{0};
std::atomic<size_t> large_allocations{0};
thread_local bool count_allocations = false;
} // namespace

extern "C" int posix_memalign(void** ptr, size_t alignment, size_t size) noexcept {
    using posix_memalign_t = int (*)(void**, size_t, size_t);
    static const posix_memalign_t real =
        reinterpret_cast<posix_memalign_t>(dlsym(RTLD_NEXT, "posix_memalign"));
    if (count_allocations && size >= allocation_threshold)
        large_allocations++;
    return real(ptr, alignment, size);
}
#endif

namespace {

// The stages run BLAS on one thread and leave the parallelism to blaze; so do the tests.
// A multithreaded OpenBLAS makes every LAPACK call on these small matrices a thread
// synchronisation, which turns a tenth of a second of tests into seconds.
struct SerialBlas {
    SerialBlas() {
        openblas_set_num_threads(1);
    }
};
BOOST_GLOBAL_FIXTURE(SerialBlas);

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

// A mask of the kind the stages build: the main diagonal (the autocorrelations) masked
// out when `mask_diagonal` is set, and the rows and columns of `excluded` inputs zeroed.
// Both kinds of entry are then filled in from the low rank estimate by the solver.
DynamicHermitian<float> test_mask(size_t n, bool mask_diagonal,
                                  const std::vector<size_t>& excluded) {
    blaze::DynamicMatrix<float, blaze::columnMajor> M(n, n, 1.0f);
    if (mask_diagonal)
        blaze::band(M, 0) = 0.0f;
    for (size_t e : excluded)
        for (size_t j = 0; j < n; j++)
            M(e, j) = M(j, e) = 0.0f;
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

// With the autocorrelations masked out, the solver has to fill them in from its low
// rank estimate. The test matrix is exactly rank 2, so the fill converges to the true
// values and the eigenvalues are unchanged.
BOOST_AUTO_TEST_CASE(eigen_masked_subspace_fills_masked_diagonal) {
    std::mt19937 rng(eigen_subspace_seed);
    const auto r = decompose(test_matrix(), test_mask(num_elements, true, {}), rng);

    BOOST_CHECK(r.stats.converged);
    BOOST_CHECK_CLOSE(r.evals[num_ev - 1], 4.0f * num_elements, 1e-2);
    BOOST_CHECK_CLOSE(r.evals[num_ev - 2], 1.0f * num_elements, 1e-2);
    BOOST_CHECK_LT(r.stats.rms, 1e-3);
}

// Inputs whose rows and columns are masked out drop out of the decomposition: the
// eigenvectors are zero there and the eigenvalues are those of the remaining inputs.
// Inputs 3 and 7 are four apart, so the two sources stay exactly orthogonal over the
// remaining fourteen and the eigenvalues are known exactly.
BOOST_AUTO_TEST_CASE(eigen_masked_subspace_excludes_masked_inputs) {
    const std::vector<size_t> excluded = {3, 7};
    std::mt19937 rng(eigen_subspace_seed);
    const auto r = decompose(test_matrix(), test_mask(num_elements, true, excluded), rng);

    const size_t remaining = num_elements - excluded.size();
    BOOST_CHECK(r.stats.converged);
    BOOST_CHECK_CLOSE(r.evals[num_ev - 1], 4.0f * remaining, 1e-2);
    BOOST_CHECK_CLOSE(r.evals[num_ev - 2], 1.0f * remaining, 1e-2);
    for (size_t j = 0; j < num_ev; j++)
        for (size_t e : excluded)
            BOOST_CHECK_SMALL(std::abs(r.evecs(e, j)), 1e-4f);
}

// A solver keeps its workspace from one call to the next. That reuse must not leak
// into the results: solving a series of different problems, as a stage does for the
// frames of its frequencies, has to give for each one exactly what a fresh solve gives.
// The masks differ too, so the filled-in entries the workspace carries are exercised.
BOOST_AUTO_TEST_CASE(solver_reuse_matches_fresh_solve) {
    struct Frequency {
        DynamicHermitian<cfloat> A;
        DynamicHermitian<float> W;
        Result fresh;
    };
    const float other_amplitude[2] = {1.0f, 3.0f};
    const float other_turns[2] = {2.0f, 5.0f};
    std::vector<Frequency> frequencies = {
        {test_matrix(), test_mask(), {}},
        {test_matrix(num_elements, other_amplitude, other_turns), test_mask(), {}},
        {test_matrix(), test_mask(num_elements, true, {3, 7}), {}},
        {test_matrix(num_elements, other_amplitude, other_turns),
         test_mask(num_elements, true, {1, 5}),
         {}},
    };

    std::mt19937 rng(eigen_subspace_seed);
    for (auto& f : frequencies) {
        rng.seed(eigen_subspace_seed);
        f.fresh = decompose(f.A, f.W, rng);
        BOOST_CHECK(f.fresh.stats.converged);
    }

    EigenMaskedSubspaceSolver<cfloat> solver;
    for (int round = 0; round < 3; round++) {
        for (const auto& f : frequencies) {
            rng.seed(eigen_subspace_seed);
            check_identical(f.fresh, decompose(solver, f.A, f.W, rng));
        }
    }
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
// containers and are allowed. The solves run on this thread alone so that every
// allocation they make is counted.
#if defined(__linux__) && defined(__GLIBC__)
BOOST_AUTO_TEST_CASE(solver_reuse_does_not_allocate) {
    const size_t n = 512;
    const auto A = test_matrix(n);
    const auto W = test_mask(n);
    // The smallest matrix in the workspace is the n x num_ev subspace
    allocation_threshold = n * num_ev * sizeof(cfloat);
    const size_t threads = blaze::getNumThreads();
    blaze::setNumThreads(1);

    EigenMaskedSubspaceSolver<cfloat> solver;
    std::mt19937 rng(eigen_subspace_seed);

    // Only solve() is counted: copying the results out is the test's own allocation.
    large_allocations = 0;
    count_allocations = true;
    const auto first_stats = solver.solve(A, W, num_ev, tol, tol, max_iterations, 0, 2, 3, rng);
    count_allocations = false;
    // The first call builds the workspace
    BOOST_CHECK_GT(large_allocations.load(), 0u);
    const Result first{solver.evals(), solver.evecs(), first_stats};
    BOOST_CHECK(first.stats.converged);

    for (int round = 0; round < 3; round++) {
        rng.seed(eigen_subspace_seed);
        large_allocations = 0;
        count_allocations = true;
        const auto stats = solver.solve(A, W, num_ev, tol, tol, max_iterations, 0, 2, 3, rng);
        count_allocations = false;
        BOOST_CHECK_EQUAL(large_allocations.load(), 0u);
        check_identical(first, Result{solver.evals(), solver.evecs(), stats});
    }

    blaze::setNumThreads(threads);
}
#endif

// The mask built in place through the Hermitian adaptor must equal the one the stage
// used to build in a plain matrix and copy, for every kind of entry it masks, at a size
// that is not a multiple of the SIMD width, and it must reuse its storage.
BOOST_AUTO_TEST_CASE(fill_mask_matches_reference) {
    const size_t n = 37;
    const std::vector<size_t> excluded = {0, 5, 36};
    std::vector<float> flags(n, 1.0f);
    flags[7] = 0.0f;
    flags[20] = 0.0f;
    const std::vector<std::pair<size_t, size_t>> bands = {{0, 2}, {9, 12}};
    const size_t block = 8;

    // The reference, built as EigenN2Iter::calculate_mask did
    blaze::DynamicMatrix<float, blaze::columnMajor> M(n, n, 1.0f);
    for (size_t e : excluded)
        for (size_t j = 0; j < n; j++)
            M(e, j) = M(j, e) = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (flags[i] != 0.0f)
            continue;
        for (size_t j = 0; j < n; j++)
            M(i, j) = M(j, i) = 0.0f;
    }
    for (const auto& br : bands)
        for (int64_t b = br.first; b < (int64_t)br.second; b++) {
            blaze::band(M, b) = 0.0f;
            blaze::band(M, -b) = 0.0f;
        }
    for (size_t start = 0; start < n; start += block) {
        const size_t width = std::min(n - start, block);
        blaze::submatrix(M, start, start, width, width) = 0.0f;
    }
    const DynamicHermitian<float> reference = blaze::declherm(M);

    DynamicHermitian<float> mask;
    fill_mask(mask, n, {}, {}, {}, 0);
    const float* const storage = mask.data();
    // A rebuild of the same size, with different entries masked, must reuse the storage
    fill_mask(mask, n, excluded, flags, bands, block);
    BOOST_CHECK_EQUAL(mask.data(), storage);

    BOOST_REQUIRE_EQUAL(mask.rows(), n);
    size_t masked = 0;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            BOOST_CHECK_EQUAL(mask(i, j), reference(i, j));
            masked += mask(i, j) == 0.0f;
        }
    // ... and it must actually mask something, and not everything
    BOOST_CHECK_GT(masked, 0u);
    BOOST_CHECK_LT(masked, n * n);
}

// Unpacking a frame's triangle into an existing container must fill it in place and
// give the same matrix as unpacking into a new one, for a matrix smaller than one of the
// tiles the unpack works in and for one spanning several tiles and a partial one.
BOOST_AUTO_TEST_CASE(to_blaze_herm_reuses_container) {
    for (const size_t n : {num_elements, size_t(150)}) {
        const auto A = test_matrix(n);
        const float other_amplitude[2] = {1.0f, 3.0f};
        const float other_turns[2] = {2.0f, 5.0f};
        const auto B = test_matrix(n, other_amplitude, other_turns);
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
        size_t mismatches = 0;
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                mismatches += unpacked(i, j) != fresh(i, j) || unpacked(i, j) != A(i, j);
        BOOST_CHECK_EQUAL(mismatches, 0u);
    }
}

// An autocorrelation with an imaginary part is not a Hermitian matrix. The adaptor
// rejects it with an invalid_argument, which is what EigenN2Iter catches to report the
// frame as failed, and it has to be thrown however far into the triangle it sits.
BOOST_AUTO_TEST_CASE(to_blaze_herm_rejects_complex_autocorrelation) {
    const size_t n = 150;
    auto packed = packed_upper_triangle(test_matrix(n));
    DynamicHermitian<cfloat> unpacked;
    const gsl_lite::span<cfloat> span(packed.data(), packed.size());
    BOOST_CHECK_NO_THROW(to_blaze_herm(span, unpacked));

    // The last autocorrelation is the last packed element
    packed.back() += cfloat(0.0f, 1.0f);
    BOOST_CHECK_THROW(to_blaze_herm(span, unpacked), std::invalid_argument);
    BOOST_CHECK_THROW(to_blaze_herm(span), std::invalid_argument);
}


// ---- CHIME-like matrices across convergence regimes --------------------------------
//
// The matrices the stages see: four cylinders of feeds in two polarisations with complex
// gains, point sources at random directions, receiver noise with an autocorrelation
// excess, and a mask of the shortest baselines and of excluded inputs. At 256 elements
// these solve in milliseconds and LAPACK's full decomposition is a cheap reference. How
// fast the iteration converges is set by the ratio of the fifth eigenvalue to the fourth:
// bright, well separated sources converge in a few iterations, sources of nearly equal
// flux take many.
namespace {

constexpr size_t sky_elements = 256;
// The CHIME stage parameters
constexpr size_t sky_num_ev = 4;
constexpr size_t sky_num_ev_conv = 2;
constexpr size_t sky_krylov = 2;
constexpr size_t sky_subspace = 1;
constexpr size_t sky_max_iterations = 19;
constexpr float sky_tol_eval = 1e-5f;
constexpr float sky_tol_evec = 1e-4f;
// The mask the stage builds: the shortest baselines and the excluded inputs
const std::vector<size_t> sky_excluded = {5, 40, 77, 130, 201, 250};
const std::vector<std::pair<size_t, size_t>> sky_bands = {{0, 6}};

struct Sky {
    /// Point source fluxes, in units of the noise rms; a source of flux F has an
    /// eigenvalue of about F times the number of elements
    std::vector<float> flux;
    /// RMS of the cross-correlation noise, zero for an exactly low rank matrix
    float noise = 0.0f;
    /// Inputs with no signal
    std::vector<size_t> zero_gain = {};
};

// The gains and source directions are drawn before the noise, so two skies that differ
// only in their noise have the same signal when built from generators in the same state.
DynamicHermitian<cfloat> sky_matrix(const Sky& sky, std::mt19937& rng) {
    const size_t n = sky_elements;
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    // Four cylinders 22 m apart, 32 feeds 0.3048 m apart along each, two polarisations
    // per feed, at 600 MHz
    std::vector<float> x(n), y(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = 22.0f * (i / (n / 4));
        y[i] = 0.3048f * ((i % (n / 4)) / 2);
    }
    std::vector<cfloat> gain(n);
    for (auto& g : gain)
        g = std::polar(1.0f + 0.3f * uni(rng), float(M_PI) * uni(rng));
    for (size_t i : sky.zero_gain)
        gain[i] = 0.0f;

    blaze::DynamicMatrix<cfloat, blaze::columnMajor> G(n, sky.flux.size());
    for (size_t s = 0; s < sky.flux.size(); s++) {
        const float l = 0.3f * uni(rng), m = 0.3f * uni(rng);
        for (size_t i = 0; i < n; i++) {
            const float phase = 2.0f * float(M_PI) * (x[i] * l + y[i] * m) / 0.5f;
            G(i, s) = gain[i] * std::sqrt(sky.flux[s]) * std::polar(1.0f, phase);
        }
    }
    blaze::DynamicMatrix<cfloat, blaze::columnMajor> M = G * blaze::ctrans(G);

    if (sky.noise > 0.0f) {
        for (size_t j = 0; j < n; j++) {
            for (size_t i = 0; i < j; i++) {
                const cfloat z = sky.noise * cfloat(gauss(rng), gauss(rng)) / std::sqrt(2.0f);
                M(i, j) += z;
                M(j, i) += std::conj(z);
            }
            // The autocorrelation excess of a receiver temperature thirty times the noise
            M(j, j) = cfloat(M(j, j).real() + 30.0f * sky.noise * (1.0f + 0.2f * uni(rng)), 0.0f);
        }
    }
    return blaze::declherm(M);
}

DynamicHermitian<float> sky_mask() {
    DynamicHermitian<float> mask;
    fill_mask(mask, sky_elements, sky_excluded, {}, sky_bands, 0);
    return mask;
}

// LAPACK's top eigenpairs, in the solver's ascending order
Result lapack_reference(const DynamicHermitian<cfloat>& A) {
    blaze::DynamicVector<float> evals;
    blaze::DynamicMatrix<cfloat, blaze::columnMajor> evecs;
    blaze::eigen(A, evals, evecs);
    const size_t n = A.rows();
    return {blaze::subvector(evals, n - sky_num_ev, sky_num_ev),
            blaze::submatrix(evecs, 0, n - sky_num_ev, n, sky_num_ev),
            {}};
}

Result sky_solve(const DynamicHermitian<cfloat>& A, const DynamicHermitian<float>& W,
                 float tol_eval = sky_tol_eval, float tol_evec = sky_tol_evec,
                 size_t maxiter = sky_max_iterations, size_t k_conv = sky_num_ev_conv) {
    std::mt19937 rng(eigen_subspace_seed);
    EigenMaskedSubspaceSolver<cfloat> solver;
    const auto stats = solver.solve(A, W, sky_num_ev, tol_eval, tol_evec, maxiter, k_conv,
                                    sky_krylov, sky_subspace, rng);
    return {solver.evals(), solver.evecs(), stats};
}

using DoubleMatrix = blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor>;

DoubleMatrix to_double(const blaze::DynamicMatrix<cfloat, blaze::columnMajor>& m, size_t first,
                       size_t count) {
    DoubleMatrix d(m.rows(), count);
    for (size_t j = 0; j < count; j++)
        for (size_t i = 0; i < m.rows(); i++)
            d(i, j) = std::complex<double>(m(i, first + j));
    return d;
}

// The sine of the largest principal angle between the subspaces spanned by `count`
// columns of each matrix from `first`, in double: the largest singular value of the part
// of one basis orthogonal to the other. For one column each, the angle between the two
// vectors, whatever their phases.
double sin_angle(const blaze::DynamicMatrix<cfloat, blaze::columnMajor>& U,
                 const blaze::DynamicMatrix<cfloat, blaze::columnMajor>& V, size_t first,
                 size_t count = 1) {
    const DoubleMatrix A = to_double(U, first, count), B = to_double(V, first, count);
    const DoubleMatrix R = B - A * (blaze::ctrans(A) * B);
    blaze::DynamicVector<double> s;
    blaze::svd(R, s);
    return blaze::max(s);
}

// Log how far a result is from a reference, eigenpair by eigenpair, for reading off the
// margins when a check fails
void report(const char* label, const Result& r, const Result& ref) {
    std::string line = std::string(label) + ": " + std::to_string(r.stats.iterations)
                       + " iterations, converged " + std::to_string(r.stats.converged);
    for (size_t l = 0; l < sky_num_ev; l++) {
        char buf[96];
        std::snprintf(buf, sizeof buf, " | %.6g: d(eval) %.1e sin %.1e", ref.evals[l],
                      std::abs(r.evals[l] - ref.evals[l]) / std::abs(ref.evals[l]),
                      sin_angle(ref.evecs, r.evecs, l));
        line += buf;
    }
    BOOST_TEST_MESSAGE(line);
}

} // namespace

// Bright, well separated sources with no noise: the matrix is exactly rank four, so the
// masked entries are filled in exactly and the eigenpairs are those LAPACK finds for the
// unmasked matrix. The excluded inputs have no signal, as the mask takes them out of the
// decomposition. The iteration converges in a few steps. Every eigenpair is tested for
// convergence here and below, where the eigenpairs are compared with a reference: with
// the stage's two of four, the other two are wherever the iteration left them.
BOOST_AUTO_TEST_CASE(sky_fast_convergence_matches_lapack) {
    std::mt19937 rng(1);
    const auto A = sky_matrix({{100, 30, 10, 3}, 0.0f, sky_excluded}, rng);
    const auto ref = lapack_reference(A);
    const auto r =
        sky_solve(A, sky_mask(), sky_tol_eval, sky_tol_evec, sky_max_iterations, sky_num_ev);
    report("fast", r, ref);

    BOOST_CHECK(r.stats.converged);
    BOOST_CHECK_LE(r.stats.iterations, sky_max_iterations);
    for (size_t l = 0; l < sky_num_ev; l++) {
        BOOST_CHECK_CLOSE(r.evals[l], ref.evals[l], 1e-2);
        BOOST_CHECK_LT(sin_angle(ref.evecs, r.evecs, l), 1e-3);
        for (size_t e : sky_excluded)
            BOOST_CHECK_LT(std::abs(r.evecs(e, l)), 1e-3f);
    }
}

// One source a thousand times brighter than the rest, as the Sun is: float rounding on
// the dominant eigenpair sets the floor for the accuracy of the weak ones. That floor is
// above the stage's eigenvalue tolerance for the weakest pair, so with every eigenpair
// tested the iteration never reports convergence (the stage tests the top two); the
// eigenpairs it leaves after the maximum number of iterations are nevertheless right.
BOOST_AUTO_TEST_CASE(sky_dominant_source) {
    std::mt19937 rng(2);
    const auto A = sky_matrix({{1000, 10, 5, 2}, 0.0f, sky_excluded}, rng);
    const auto ref = lapack_reference(A);
    const auto r =
        sky_solve(A, sky_mask(), sky_tol_eval, sky_tol_evec, sky_max_iterations, sky_num_ev);
    report("dominant", r, ref);

    for (size_t l = 0; l < sky_num_ev; l++) {
        BOOST_CHECK_CLOSE(r.evals[l], ref.evals[l], 1e-1);
        BOOST_CHECK_LT(sin_angle(ref.evecs, r.evecs, l), 1e-2);
    }
}

// Two sources of equal flux: their eigenvectors are only defined up to a rotation within
// the pair, so it is the pair's subspace that has to match, while the other two
// eigenvectors are as well defined as ever.
BOOST_AUTO_TEST_CASE(sky_degenerate_pair) {
    std::mt19937 rng(3);
    const auto A = sky_matrix({{20, 5, 5, 1}, 0.0f, sky_excluded}, rng);
    const auto ref = lapack_reference(A);
    const auto r =
        sky_solve(A, sky_mask(), sky_tol_eval, sky_tol_evec, sky_max_iterations, sky_num_ev);
    report("degenerate", r, ref);

    BOOST_CHECK(r.stats.converged);
    for (size_t l = 0; l < sky_num_ev; l++)
        BOOST_CHECK_CLOSE(r.evals[l], ref.evals[l], 1e-2);
    // Ascending: the pair is the middle two
    BOOST_CHECK_LT(sin_angle(ref.evecs, r.evecs, 0), 1e-3);
    BOOST_CHECK_LT(sin_angle(ref.evecs, r.evecs, 1, 2), 1e-3);
    BOOST_CHECK_LT(sin_angle(ref.evecs, r.evecs, 3), 1e-3);
}

// Five sources of nearly equal flux in noise, with nothing masked so that LAPACK's
// decomposition is the exact answer, is the slow regime: the fifth eigenvalue is within
// a few percent of the fourth. The Ritz values are bounded above by the eigenvalues
// (Cauchy interlacing) and improve with the iterations, and the subspace closes in on
// the eigenspace.
BOOST_AUTO_TEST_CASE(sky_slow_convergence_improves_with_iterations) {
    std::mt19937 rng(4);
    const auto A = sky_matrix({{10, 9.5, 9, 8.5, 8}, 1.0f}, rng);
    const auto W = test_mask(sky_elements);
    const auto ref = lapack_reference(A);
    {
        blaze::DynamicVector<float> all;
        blaze::eigen(A, all);
        BOOST_TEST_MESSAGE("slow: gap ratio lambda_5 / lambda_4 = "
                           << all[sky_elements - sky_num_ev - 1] / all[sky_elements - sky_num_ev]);
    }

    const size_t iterations[] = {1, 3, 40};
    std::vector<Result> results;
    for (size_t iters : iterations)
        results.push_back(sky_solve(A, W, 0.0f, 0.0f, iters, sky_num_ev));

    for (const auto& r : results)
        for (size_t l = 0; l < sky_num_ev; l++)
            BOOST_CHECK_LE(r.evals[l], ref.evals[l] * (1.0f + 1e-5f));
    for (size_t l = 0; l < sky_num_ev; l++) {
        BOOST_CHECK_GE(results[2].evals[l], results[0].evals[l] * (1.0f - 1e-5f));
        BOOST_CHECK_CLOSE(results[2].evals[l], ref.evals[l], 1e-1);
    }
    for (size_t i = 0; i < 3; i++)
        report("slow", results[i], ref);
    const double angle_1 = sin_angle(ref.evecs, results[0].evecs, 0, sky_num_ev);
    const double angle_3 = sin_angle(ref.evecs, results[1].evecs, 0, sky_num_ev);
    const double angle_40 = sin_angle(ref.evecs, results[2].evecs, 0, sky_num_ev);
    BOOST_TEST_MESSAGE("slow: sin(subspace angle) after 1, 3, 40 iterations: "
                       << angle_1 << " " << angle_3 << " " << angle_40);
    BOOST_CHECK_LT(angle_3, angle_1);
    BOOST_CHECK_LT(angle_40, angle_3);
    BOOST_CHECK_LT(angle_40, 1e-3);

    // At the stage's tolerances the convergence flag must agree with the statistics
    const auto r = sky_solve(A, sky_mask());
    BOOST_CHECK_EQUAL(r.stats.converged,
                      r.stats.eps_eval < sky_tol_eval && r.stats.eps_evec < sky_tol_evec);
}

// Noise, and inputs with no signal, some masked as excluded and some not: the eigenpairs
// are those of the signal alone, to the noise, and the eigenvectors are small where
// there is no signal. Noise of rms sigma moves an eigenvector of eigenvalue lambda by
// about sigma sqrt(n) / lambda, which here is 2e-3 for the weakest source.
BOOST_AUTO_TEST_CASE(sky_dead_and_excluded_inputs) {
    const std::vector<size_t> dead = {13, 110};
    std::vector<size_t> zero_gain = sky_excluded;
    zero_gain.insert(zero_gain.end(), dead.begin(), dead.end());

    // The same sky with and without noise, from generators in the same state
    std::mt19937 rng_signal(5), rng_noisy(5);
    const auto signal = sky_matrix({{100, 30, 10, 3}, 0.0f, zero_gain}, rng_signal);
    const auto A = sky_matrix({{100, 30, 10, 3}, 0.1f, zero_gain}, rng_noisy);
    const auto ref = lapack_reference(signal);
    const auto r =
        sky_solve(A, sky_mask(), sky_tol_eval, sky_tol_evec, sky_max_iterations, sky_num_ev);
    report("dead and excluded", r, ref);

    BOOST_CHECK(r.stats.converged);
    for (size_t l = 0; l < sky_num_ev; l++) {
        BOOST_CHECK_CLOSE(r.evals[l], ref.evals[l], 1.0);
        BOOST_CHECK_LT(sin_angle(ref.evecs, r.evecs, l), 5e-2);
        for (size_t i : zero_gain)
            BOOST_CHECK_LT(std::abs(r.evecs(i, l)), 1e-2f);
    }
}

// The eigenpairs of a converged masked decomposition are eigenpairs of the matrix they
// imply: the data where the mask includes it and the rank-k reconstruction where it does
// not. This is what the refill of the masked entries has to achieve.
BOOST_AUTO_TEST_CASE(sky_masked_fill_is_self_consistent) {
    std::mt19937 rng(6);
    const auto A = sky_matrix({{100, 30, 10, 3}, 1.0f, sky_excluded}, rng);
    const auto W = sky_mask();
    const auto r = sky_solve(A, W, 1e-5f, 1e-5f, 60, sky_num_ev);
    BOOST_REQUIRE(r.stats.converged);

    const size_t n = sky_elements;
    const DoubleMatrix V = to_double(r.evecs, 0, sky_num_ev);
    DoubleMatrix filled(n, n);
    for (size_t j = 0; j < n; j++)
        for (size_t i = 0; i < n; i++) {
            if (W(i, j) != 0.0f) {
                filled(i, j) = std::complex<double>(A(i, j));
                continue;
            }
            std::complex<double> fill(0.0);
            for (size_t l = 0; l < sky_num_ev; l++)
                fill += double(r.evals[l]) * V(i, l) * std::conj(V(j, l));
            filled(i, j) = fill;
        }
    DoubleMatrix VL = V;
    for (size_t l = 0; l < sky_num_ev; l++)
        blaze::column(VL, l) *= double(r.evals[l]);
    const DoubleMatrix residual = filled * V - VL;
    double residual_norm = 0.0, scale = 0.0;
    for (size_t l = 0; l < sky_num_ev; l++)
        for (size_t i = 0; i < n; i++) {
            residual_norm += std::norm(residual(i, l));
            scale += std::norm(VL(i, l));
        }
    BOOST_CHECK_LT(std::sqrt(residual_norm / scale), 1e-3);
}

// The data in the masked entries never reaches the decomposition: replacing it, here
// with loud noise on the excluded inputs and the diagonal bands, leaves the result the
// same to the last bit. This is what masks out a loud excluded input or the
// autocorrelation excess.
BOOST_AUTO_TEST_CASE(sky_masked_data_is_ignored) {
    std::mt19937 rng(7);
    const auto A = sky_matrix({{100, 30, 10, 3}, 0.1f, sky_excluded}, rng);
    const auto W = sky_mask();
    DynamicHermitian<cfloat> B = A;
    std::normal_distribution<float> gauss(0.0f, 10.0f);
    for (size_t j = 0; j < sky_elements; j++)
        for (size_t i = 0; i <= j; i++)
            if (W(i, j) == 0.0f)
                B(i, j) = cfloat(gauss(rng), i == j ? 0.0f : gauss(rng));
    check_identical(sky_solve(A, W), sky_solve(B, W));
}

// The products, the refill and the residual are split over blaze's threads, and the
// split must not change the result beyond rounding. Without OpenMP the thread count
// cannot be set and the two solves are the same.
BOOST_AUTO_TEST_CASE(sky_thread_count_agreement) {
    std::mt19937 rng(7);
    const auto A = sky_matrix({{10, 9.5, 9, 8.5, 8}, 1.0f, sky_excluded}, rng);
    const auto W = sky_mask();
    const size_t threads = blaze::getNumThreads();

    blaze::setNumThreads(1);
    const auto serial = sky_solve(A, W, 0.0f, 0.0f, 10, sky_num_ev);
    blaze::setNumThreads(3);
    const auto parallel = sky_solve(A, W, 0.0f, 0.0f, 10, sky_num_ev);
    blaze::setNumThreads(threads);

    for (size_t l = 0; l < sky_num_ev; l++) {
        BOOST_CHECK_CLOSE(serial.evals[l], parallel.evals[l], 1e-2);
        BOOST_CHECK_LT(sin_angle(serial.evecs, parallel.evecs, l), 1e-3);
    }
}
