#define BOOST_TEST_MODULE "test_gnss_shared_pin"

#include "gnssSharedPin.hpp" // for ref_offset, pin_rotation, RefOffset

#include <boost/test/included/unit_test.hpp>
#include <cmath>   // for abs, sqrt, cos, sin
#include <complex> // for complex, polar, arg
#include <random>  // for mt19937, normal_distribution
#include <vector>

using cd = std::complex<double>;
using gnss::pin_rotation;
using gnss::ref_offset;

namespace {

constexpr double DEG = M_PI / 180.0;

// One pol half of an instrument: 16 elements with distinct amplitudes and phases.
std::vector<cd> instrument(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ph(-M_PI, M_PI), amp(0.5, 1.5);
    std::vector<cd> v((size_t)n);
    for (auto& x : v)
        x = std::polar(amp(rng), ph(rng));
    return v;
}

std::vector<cd> rotated(const std::vector<cd>& v, double a) {
    std::vector<cd> out(v);
    for (auto& x : out)
        x *= std::polar(1.0, a);
    return out;
}

double wrap(double a) {
    return std::arg(std::polar(1.0, a));
}

} // namespace

BOOST_AUTO_TEST_CASE(offset_of_a_rotated_copy_is_the_rotation) {
    const auto F = instrument(16, 1);
    const auto G = rotated(F, 139.0 * DEG);
    const auto o = ref_offset(F.data(), G.data(), 0, 16);
    BOOST_REQUIRE(o.ok);
    BOOST_CHECK_SMALL(wrap(o.err_rad - 139.0 * DEG), 1e-12);
    BOOST_CHECK_CLOSE(o.sim, 1.0, 1e-9);
}

BOOST_AUTO_TEST_CASE(a_first_model_is_pinned_in_full) {
    const auto F = instrument(16, 2);
    auto G = rotated(F, -100.0 * DEG);
    const cd rot = pin_rotation(ref_offset(F.data(), G.data(), 0, 16), false, 1.0 * DEG);
    for (auto& x : G)
        x *= rot;
    const auto o = ref_offset(F.data(), G.data(), 0, 16);
    BOOST_CHECK_SMALL(o.err_rad, 1e-12);
}

BOOST_AUTO_TEST_CASE(a_warm_model_slews_and_never_steps) {
    // cx27 GPU0 L5 on 2026-10-06: 139 deg off the fleet. At 1 deg per 1-s consensus update it
    // must arrive in 139 updates, each step at most 1 deg, the offset falling monotonically.
    const auto F = instrument(16, 3);
    auto G = rotated(F, 139.0 * DEG);
    double prev = 139.0 * DEG;
    int n = 0;
    for (; n < 400; ++n) {
        const auto o = ref_offset(F.data(), G.data(), 0, 16);
        if (std::abs(o.err_rad) < 1e-9)
            break;
        const cd rot = pin_rotation(o, true, 1.0 * DEG);
        BOOST_REQUIRE_LE(std::abs(std::arg(rot)), 1.0 * DEG + 1e-12);
        for (auto& x : G)
            x *= rot;
        const double now = std::abs(ref_offset(F.data(), G.data(), 0, 16).err_rad);
        BOOST_REQUIRE_LT(now, prev + 1e-12);
        prev = now;
    }
    BOOST_CHECK_EQUAL(n, 139);
}

BOOST_AUTO_TEST_CASE(the_slew_takes_the_short_way_round) {
    const auto F = instrument(16, 4);
    auto G = rotated(F, -170.0 * DEG);
    const cd rot = pin_rotation(ref_offset(F.data(), G.data(), 0, 16), true, 1.0 * DEG);
    BOOST_CHECK_CLOSE(std::arg(rot), 1.0 * DEG, 1e-9); // -170 is corrected upward, not by -190
}

BOOST_AUTO_TEST_CASE(no_projection_means_no_pin) {
    const std::vector<cd> zero(16, cd(0.0, 0.0));
    const auto F = instrument(16, 5);
    BOOST_CHECK(!ref_offset(zero.data(), F.data(), 0, 16).ok);
    BOOST_CHECK(!ref_offset(F.data(), zero.data(), 0, 16).ok);
    // A zero step limit leaves a warm model where it is.
    const auto o = ref_offset(F.data(), rotated(F, 0.5).data(), 0, 16);
    BOOST_CHECK_EQUAL(pin_rotation(o, true, 0.0), cd(1.0, 0.0));
}

BOOST_AUTO_TEST_CASE(only_the_named_half_is_read) {
    // Pol 1's half is the other polarisation: its phases must not move pol 0's pin.
    auto F = instrument(32, 6);
    auto G = rotated(F, 40.0 * DEG);
    for (int e = 16; e < 32; ++e)
        G[(size_t)e] = std::polar(1.0, -2.0); // pol 1 unrelated
    const auto o0 = ref_offset(F.data(), G.data(), 0, 16);
    BOOST_CHECK_SMALL(wrap(o0.err_rad - 40.0 * DEG), 1e-12);
}

BOOST_AUTO_TEST_CASE(instances_that_started_apart_end_together) {
    // Two instances of one band: different noise, first models pinned 139 deg apart (one good
    // shadow, one poor). Pinned to the same F they converge to within the noise of their
    // shapes, not to within their starting offset.
    const auto F = instrument(16, 7);
    std::mt19937 rng(8);
    std::normal_distribution<double> nz(0.0, 0.15);
    auto noisy = [&](double a) {
        auto v = rotated(F, a);
        for (auto& x : v)
            x += cd(nz(rng), nz(rng));
        return v;
    };
    auto A = noisy(0.0), B = noisy(139.0 * DEG);
    BOOST_REQUIRE_GT(ref_offset(F.data(), B.data(), 0, 16).sim, 0.9);
    for (int n = 0; n < 300; ++n)
        for (auto* G : {&A, &B}) {
            const cd rot = pin_rotation(ref_offset(F.data(), G->data(), 0, 16), true, 1.0 * DEG);
            for (auto& x : *G)
                x *= rot;
        }
    const auto ab = ref_offset(A.data(), B.data(), 0, 16);
    BOOST_CHECK_LT(std::abs(ab.err_rad), 3.0 * DEG);
}
