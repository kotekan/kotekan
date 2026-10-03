#define BOOST_TEST_MODULE "test_lifetimeWindows"

#include "lifetimeWindows.hpp"

#include <algorithm>
#include <boost/test/included/unit_test.hpp>
#include <cstdint>
#include <vector>

using kotekan::LifetimeWindows;

// Brute-force checks of the upchannelizer's window scheduler. A window of `n` granules
// produces `n * g - o` output samples; the scheduler has to tile every gain lifetime exactly,
// whatever the availability of input data.

namespace {

// The generated upchannelizers: 256 input samples per granule, M = 4 taps
constexpr std::int64_t granularity = 256;
constexpr std::int64_t num_taps = 4;
const std::vector<std::int64_t> all_U = {2, 4, 8, 16, 32, 64, 128};
// 1 is the CPU reference in verify_cuda_upchan.j2; 32 and 64 are the CHORD and CHIME kernels
const std::vector<std::int64_t> all_max_granules = {1, 2, 3, 4, 7, 32, 64};

// Largest remainder checked exhaustively. It covers several periods of the congruence for
// every U, and a few times the largest window.
constexpr std::int64_t max_remaining = 20000;

struct Setup {
    std::int64_t U, max_granules;
    std::int64_t overlap() const {
        return (num_taps - 1) * U;
    }
};

std::vector<Setup> all_setups() {
    std::vector<Setup> setups;
    for (const std::int64_t U : all_U)
        for (const std::int64_t max_granules : all_max_granules)
            if (LifetimeWindows::valid(granularity, (num_taps - 1) * U, U, max_granules))
                setups.push_back({U, max_granules});
    return setups;
}

// Window sizes, by brute force
std::vector<std::int64_t> window_sizes(const Setup& s) {
    const std::int64_t g = granularity / s.U;
    const std::int64_t o = s.overlap() / s.U;
    std::vector<std::int64_t> sizes;
    for (std::int64_t n = 1; n <= s.max_granules; ++n)
        if (n * g - o > 0)
            sizes.push_back(n * g - o);
    return sizes;
}

// For every x in [0, max_remaining]: the smallest number of windows that sum to exactly x, or
// -1 if there is none. This is the ground truth for `reachable` and `min_num_windows`.
std::vector<std::int64_t> brute_force_min_windows(const Setup& s) {
    const std::vector<std::int64_t> sizes = window_sizes(s);
    std::vector<std::int64_t> min_windows(max_remaining + 1, -1);
    min_windows.at(0) = 0;
    for (std::int64_t x = 1; x <= max_remaining; ++x)
        for (const std::int64_t p : sizes)
            if (p <= x && min_windows.at(x - p) >= 0
                && (min_windows.at(x) < 0 || min_windows.at(x - p) + 1 < min_windows.at(x)))
                min_windows.at(x) = min_windows.at(x - p) + 1;
    return min_windows;
}

// A small deterministic generator, so that failures reproduce
struct LCG {
    std::uint64_t state;
    std::int64_t operator()(const std::int64_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return std::int64_t((state >> 33) % std::uint64_t(bound));
    }
};

} // namespace

BOOST_AUTO_TEST_CASE(parameters) {
    // U = 128 needs two granules to produce anything (512 - 384 = 128 input samples)
    BOOST_CHECK(!LifetimeWindows::valid(granularity, 3 * 128, 128, 1));
    BOOST_CHECK(LifetimeWindows::valid(granularity, 3 * 128, 128, 2));
    BOOST_CHECK_EQUAL(LifetimeWindows(granularity, 3 * 128, 128, 2).min_granules(), 2);
    BOOST_CHECK_EQUAL(LifetimeWindows(granularity, 3 * 128, 128, 2).min_produced(), 1);
    BOOST_CHECK_EQUAL(LifetimeWindows(granularity, 3 * 64, 64, 1).min_granules(), 1);
    BOOST_CHECK_EQUAL(LifetimeWindows(granularity, 3 * 64, 64, 1).min_produced(), 1);
    BOOST_CHECK_EQUAL(LifetimeWindows(granularity, 3 * 16, 16, 32).min_produced(), 13);
    BOOST_CHECK_EQUAL(LifetimeWindows(granularity, 3 * 16, 16, 32).max_produced(), 509);
    // The overlap and the granule must be whole output samples
    BOOST_CHECK(!LifetimeWindows::valid(granularity, 3 * 16 + 1, 16, 32));
    BOOST_CHECK(!LifetimeWindows::valid(granularity + 1, 3 * 16, 16, 32));
    BOOST_CHECK(!LifetimeWindows::valid(granularity, 3 * 16, 16, 0));
    // No overlap (M = 1): every multiple of g is reachable, nothing else is
    const LifetimeWindows no_overlap(granularity, 0, 16, 4);
    for (std::int64_t x = 0; x < 200; ++x)
        BOOST_CHECK_EQUAL(no_overlap.reachable(x), x % 16 == 0);
}

// `reachable` and `min_num_windows` are closed forms; compare them with a brute-force search
// over all remainders.
BOOST_AUTO_TEST_CASE(reachable_matches_brute_force) {
    for (const Setup& s : all_setups()) {
        BOOST_TEST_CONTEXT("U=" << s.U << " max_granules=" << s.max_granules) {
            const LifetimeWindows windows(granularity, s.overlap(), s.U, s.max_granules);
            const std::vector<std::int64_t> min_windows = brute_force_min_windows(s);
            std::int64_t num_mismatches = 0;
            for (std::int64_t x = 0; x <= max_remaining; ++x) {
                if (windows.reachable(x) != (min_windows.at(x) >= 0))
                    ++num_mismatches;
                if (x > 0
                    && windows.min_num_windows(x) != std::max<std::int64_t>(0, min_windows.at(x)))
                    ++num_mismatches;
            }
            BOOST_CHECK_EQUAL(num_mismatches, 0);
            BOOST_CHECK(!windows.reachable(-1));
        }
    }
}

// For every reachable remainder and every availability: `num_granules` returns the largest
// window that fits the data and the boundary and leaves a reachable remainder, 0 exactly when
// there is none, and never 0 when a full window is available. With a full window available it
// agrees with the closed-form rule (fewest windows to the boundary, largest first).
BOOST_AUTO_TEST_CASE(num_granules_is_the_largest_valid_window) {
    for (const Setup& s : all_setups()) {
        BOOST_TEST_CONTEXT("U=" << s.U << " max_granules=" << s.max_granules) {
            const LifetimeWindows windows(granularity, s.overlap(), s.U, s.max_granules);
            const std::vector<std::int64_t> min_windows = brute_force_min_windows(s);
            const std::int64_t p_min = windows.min_produced();
            const std::int64_t p_max = windows.max_produced();
            std::int64_t num_mismatches = 0, num_stalls = 0, num_rule_mismatches = 0;
            for (std::int64_t r = 1; r <= max_remaining; r += r < 2000 ? 1 : 7) {
                if (min_windows.at(r) < 0)
                    continue;
                // best[cap]: the largest valid window with at most `cap` granules
                std::vector<std::int64_t> best(s.max_granules + 1, 0);
                for (std::int64_t n = 1; n <= s.max_granules; ++n) {
                    const std::int64_t p = windows.produced(n);
                    const bool valid = p > 0 && p <= r && min_windows.at(r - p) >= 0;
                    best.at(n) = valid ? n : best.at(n - 1);
                }
                for (std::int64_t available = 0; available <= s.max_granules + 2; ++available) {
                    const std::int64_t n = windows.num_granules(r, available);
                    const std::int64_t expected = best.at(std::min(available, s.max_granules));
                    if (n != expected)
                        ++num_mismatches;
                    if (available >= s.max_granules && n == 0)
                        ++num_stalls;
                }
                const std::int64_t m = min_windows.at(r);
                const std::int64_t p_rule = std::min(p_max, r - (m - 1) * p_min);
                if (windows.produced(windows.num_granules(r, s.max_granules)) != p_rule)
                    ++num_rule_mismatches;
            }
            BOOST_CHECK_EQUAL(num_mismatches, 0);
            BOOST_CHECK_EQUAL(num_stalls, 0);
            BOOST_CHECK_EQUAL(num_rule_mismatches, 0);
        }
    }
}

// Run a stream across several lifetimes with random, often short, data availability, the way
// a starved voltage ring buffer would feed it. Every window must fit the data, no window may
// straddle a boundary, and every boundary must be hit exactly.
BOOST_AUTO_TEST_CASE(stream_hits_every_boundary) {
    LCG rng{42};
    for (const Setup& s : all_setups()) {
        const LifetimeWindows windows(granularity, s.overlap(), s.U, s.max_granules);
        // All reachable short lifetimes, plus the production and CI lifetimes (in FPGA
        // samples) where they are whole numbers of output samples
        std::vector<std::int64_t> lifetimes;
        for (std::int64_t L = 1; L <= 600; ++L)
            if (windows.reachable(L))
                lifetimes.push_back(L);
        for (const std::int64_t L_fpga : {393216, 16 * 42, 16 * 1001, 24 * 16384})
            if (L_fpga % s.U == 0 && windows.reachable(L_fpga / s.U))
                lifetimes.push_back(L_fpga / s.U);

        BOOST_TEST_CONTEXT("U=" << s.U << " max_granules=" << s.max_granules) {
            std::int64_t num_errors = 0;
            for (const std::int64_t L : lifetimes) {
                const int num_lifetimes = 3;
                std::int64_t Tbar = 0;      // output samples produced so far
                std::int64_t available = 0; // input samples available
                std::int64_t num_waits = 0; // consecutive calls that returned 0
                while (Tbar < num_lifetimes * L) {
                    // More data arrives, a random amount, often less than a full window
                    available += rng(s.max_granules * granularity + 1);
                    const std::int64_t available_granules =
                        std::min(available / granularity, s.max_granules);
                    const std::int64_t remaining = L - Tbar % L;
                    const std::int64_t n = windows.num_granules(remaining, available_granules);
                    if (n == 0) {
                        // Waiting is fine as long as it ends: a full window always works
                        if (++num_waits > 1000 || available_granules >= s.max_granules) {
                            ++num_errors;
                            break;
                        }
                        continue;
                    }
                    num_waits = 0;
                    const std::int64_t p = windows.produced(n);
                    if (!(n >= windows.min_granules() && n <= available_granules && p > 0
                          && p <= remaining)) {
                        ++num_errors;
                        break;
                    }
                    Tbar += p;
                    // A window consumes all it read but the overlap, which the next one re-reads
                    available -= n * granularity - s.overlap();
                    // The input and output streams advance together
                    if ((n * granularity - s.overlap()) != p * s.U) {
                        ++num_errors;
                        break;
                    }
                }
                if (Tbar != num_lifetimes * L)
                    ++num_errors;
            }
            BOOST_CHECK_EQUAL(num_errors, 0);
        }
    }
}

// The production lifetime, 393216 FPGA samples (about one second), must be reachable for
// every generated kernel, both the CHORD (32 granules per window) and CHIME (64) variants.
BOOST_AUTO_TEST_CASE(production_lifetimes_are_reachable) {
    for (const std::int64_t U : all_U)
        for (const std::int64_t max_granules : {32, 64}) {
            BOOST_TEST_CONTEXT("U=" << U << " max_granules=" << max_granules) {
                const LifetimeWindows windows(granularity, (num_taps - 1) * U, U, max_granules);
                BOOST_CHECK(windows.reachable(393216 / U));
            }
        }
    // The lifetime in verify_cuda_upchan.j2: U = 16, the kernel reads up to 32 granules and
    // the CPU reference up to 2
    BOOST_CHECK(LifetimeWindows(granularity, 3 * 16, 16, 32).reachable(42));
    BOOST_CHECK(LifetimeWindows(granularity, 3 * 16, 16, 2).reachable(42));
}
