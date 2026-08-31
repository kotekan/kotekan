#define BOOST_TEST_MODULE "test_linearAlgebra"

#include "LinearAlgebra.hpp" // for DynamicHermitian, EigConvergenceStats, eigen_masked_subspace

#include <boost/test/included/unit_test.hpp>
#include <cmath>   // for M_PI
#include <complex> // for complex, polar
#include <cstdint> // for uint32_t
#include <random>  // for mt19937
#include <thread>  // for thread
#include <vector>  // for vector

using cfloat = std::complex<float>;

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
// two sources are exactly orthogonal and the eigenvalues are known: `4 * num_elements`
// and `num_elements`.
constexpr float source_amplitude[2] = {2.0f, 1.0f};
constexpr float source_turns[2] = {1.0f, 3.0f};

DynamicHermitian<cfloat> test_matrix() {
    blaze::DynamicMatrix<cfloat, blaze::columnMajor> M(num_elements, num_elements, cfloat(0.0f));
    for (size_t s = 0; s < 2; s++) {
        const float rate = 2.0f * M_PI * source_turns[s] / num_elements;
        for (size_t i = 0; i < num_elements; i++)
            for (size_t j = 0; j < num_elements; j++)
                M(i, j) += source_amplitude[s] * source_amplitude[s]
                           * std::polar(1.0f, rate * (float(i) - float(j)));
    }
    return blaze::declherm(M);
}

// Every input included, as in the default stage configuration.
DynamicHermitian<float> test_mask() {
    blaze::DynamicMatrix<float, blaze::columnMajor> M(num_elements, num_elements, 1.0f);
    return blaze::declherm(M);
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
