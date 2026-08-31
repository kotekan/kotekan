#define BOOST_TEST_MODULE "test_div"

#include "div.hpp"

#include <boost/test/included/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

using kotekan::div_ceil;
using kotekan::num_triangle_blocks;

BOOST_AUTO_TEST_CASE(div_ceil_exact_multiples) {
    // Exact multiples reduce to plain division
    BOOST_CHECK_EQUAL(div_ceil(2048, 16), 128);
    BOOST_CHECK_EQUAL(div_ceil(2048u, 32u), 64u);
    BOOST_CHECK_EQUAL(div_ceil((size_t)1024, 16), (size_t)64);
    BOOST_CHECK_EQUAL(div_ceil(16, 16), 1);
}

BOOST_AUTO_TEST_CASE(div_ceil_partial_blocks) {
    // A partial block is still a block
    BOOST_CHECK_EQUAL(div_ceil(1, 16), 1);
    BOOST_CHECK_EQUAL(div_ceil(15, 16), 1);
    BOOST_CHECK_EQUAL(div_ceil(17, 16), 2);
    BOOST_CHECK_EQUAL(div_ceil(31u, 16u), 2u);
}

BOOST_AUTO_TEST_CASE(div_ceil_zero_numerator) {
    // Zero elements need zero blocks. (The old `((N - 1) / b) + 1` idiom
    // wrapped around for unsigned N == 0.)
    BOOST_CHECK_EQUAL(div_ceil(0, 16), 0);
    BOOST_CHECK_EQUAL(div_ceil(0u, 16u), 0u);
}

BOOST_AUTO_TEST_CASE(div_ceil_zero_divisor_throws) {
    BOOST_CHECK_THROW(div_ceil(16, 0), std::invalid_argument);
    BOOST_CHECK_THROW(div_ceil(16u, 0u), std::invalid_argument);
    BOOST_CHECK_THROW(div_ceil(0, 0), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(num_triangle_blocks_deployed_sizes) {
    // CHIME N2: 2048 elements in 32-element blocks, 64 per side
    BOOST_CHECK_EQUAL(num_triangle_blocks(2048, 32), 64 * 65 / 2);
    // n2k correlation: 16-element blocks
    BOOST_CHECK_EQUAL(num_triangle_blocks(2048, 16), 128 * 129 / 2);
    BOOST_CHECK_EQUAL(num_triangle_blocks(1024, 16), 64 * 65 / 2);
    BOOST_CHECK_EQUAL(num_triangle_blocks(128, 16), 8 * 9 / 2);
    // n2k counts: 8x8 tiles over num_elements / 8
    BOOST_CHECK_EQUAL(num_triangle_blocks(2048 / 8, 8), 32 * 33 / 2);
    BOOST_CHECK_EQUAL(num_triangle_blocks(128 / 8, 8), 2 * 3 / 2);
}

BOOST_AUTO_TEST_CASE(num_triangle_blocks_partial_blocks) {
    // A partial block row/column still counts
    BOOST_CHECK_EQUAL(num_triangle_blocks(17, 16), 3); // 2 per side
    BOOST_CHECK_EQUAL(num_triangle_blocks(15, 16), 1); // 1 per side
    BOOST_CHECK_EQUAL(num_triangle_blocks(0, 16), 0);
    BOOST_CHECK_THROW(num_triangle_blocks(16, 0), std::invalid_argument);
}

// Both helpers must remain usable in constant expressions (e.g. gpu_N2_size)
static_assert(kotekan::div_ceil(2048, 16) == 128, "div_ceil is constexpr");
static_assert(kotekan::num_triangle_blocks(2048, 16) == 128 * 129 / 2,
              "num_triangle_blocks is constexpr");

// ---------------------------------------------------------------------------
// Integer-type safety: negative operands, values near the type limits, and the
// mixed signed/unsigned combinations the call sites actually produce.
// ---------------------------------------------------------------------------

using kotekan::div_noremainder;
using kotekan::mod;
using kotekan::round_down;
using kotekan::round_up;

BOOST_AUTO_TEST_CASE(div_floors_towards_negative_infinity) {
    BOOST_CHECK_EQUAL(kotekan::div(7, 3), 2);
    BOOST_CHECK_EQUAL(kotekan::div(6, 3), 2);
    BOOST_CHECK_EQUAL(kotekan::div(0, 3), 0);
    // Truncated division would give 0, 0, -1, -1 here
    BOOST_CHECK_EQUAL(kotekan::div(-1, 3), -1);
    BOOST_CHECK_EQUAL(kotekan::div(-2, 3), -1);
    BOOST_CHECK_EQUAL(kotekan::div(-3, 3), -1);
    BOOST_CHECK_EQUAL(kotekan::div(-4, 3), -2);
    BOOST_CHECK_EQUAL(kotekan::div(-6, 3), -2);
}

BOOST_AUTO_TEST_CASE(mod_is_nonnegative_for_negative_x) {
    BOOST_CHECK_EQUAL(mod(7, 3), 1);
    BOOST_CHECK_EQUAL(mod(0, 3), 0);
    // The old `(x - y + 1) % y` returned 0, -1, -2, 0 for these
    BOOST_CHECK_EQUAL(mod(-1, 3), 2);
    BOOST_CHECK_EQUAL(mod(-2, 3), 1);
    BOOST_CHECK_EQUAL(mod(-3, 3), 0);
    BOOST_CHECK_EQUAL(mod(-4, 3), 2);
}

BOOST_AUTO_TEST_CASE(div_and_mod_are_consistent) {
    // `x == div(x, y) * y + mod(x, y)` with `0 <= mod(x, y) < y`, which is the
    // property the ringbuffer index arithmetic relies on
    for (int x = -40; x <= 40; ++x) {
        for (int y = 1; y <= 7; ++y) {
            const int q = kotekan::div(x, y);
            const int r = mod(x, y);
            BOOST_CHECK_EQUAL(q * y + r, x);
            BOOST_CHECK(r >= 0);
            BOOST_CHECK(r < y);
        }
    }
}

BOOST_AUTO_TEST_CASE(div_near_type_minimum) {
    // `(x - y + 1) / y` underflowed here; the truncate-then-adjust form does not
    constexpr int min = std::numeric_limits<int>::min();
    BOOST_CHECK_EQUAL(kotekan::div(min, 1), min);
    BOOST_CHECK_EQUAL(kotekan::div(min, 2), min / 2);
    BOOST_CHECK_EQUAL(kotekan::div(min + 1, 2), min / 2);
    BOOST_CHECK_EQUAL(mod(min, 2), 0);
    BOOST_CHECK_EQUAL(mod(min + 1, 2), 1);
}

BOOST_AUTO_TEST_CASE(round_up_zero_unsigned) {
    // The old postcondition `r - y < x` underflowed whenever `r < y`, so this
    // aborted in every debug build
    BOOST_CHECK_EQUAL(round_up(0u, 8u), 0u);
    BOOST_CHECK_EQUAL(round_up((size_t)0, 16), (size_t)0);
    BOOST_CHECK_EQUAL(round_up(0, 8), 0);
    BOOST_CHECK_EQUAL(round_down(0u, 8u), 0u);
}

BOOST_AUTO_TEST_CASE(round_up_near_unsigned_maximum) {
    // Results that are still representable must come back exactly. (Above
    // 4294967288 the next multiple of 8 does not fit in `unsigned` at all, so
    // there is no right answer to return.)
    BOOST_CHECK_EQUAL(round_up(4294967288u, 8u), 4294967288u);
    BOOST_CHECK_EQUAL(round_up(4294967287u, 8u), 4294967288u);
    BOOST_CHECK_EQUAL(round_up((size_t)0xfffffffffffffff8ull, (size_t)8),
                      (size_t)0xfffffffffffffff8ull);
}

BOOST_AUTO_TEST_CASE(round_trip_near_signed_maximum) {
    constexpr int max = std::numeric_limits<int>::max();
    // `div_ceil`'s old `r * y >= x` postcondition was undefined behaviour here
    BOOST_CHECK_EQUAL(div_ceil(max, 3), max / 3 + 1);
    BOOST_CHECK_EQUAL(div_ceil(max - 1, 3), (max - 1) / 3); // max - 1 is a multiple of 3
    BOOST_CHECK_EQUAL(round_down(max, 8), max - 7);
    BOOST_CHECK_EQUAL(round_up(max - 7, 8), max - 7);
}

BOOST_AUTO_TEST_CASE(num_triangle_blocks_wide_matrices) {
    // `nb * (nb + 1) / 2` overflowed int32 from nb == 46341 upwards; halving
    // before multiplying is exact up to nb == 65535
    // nb == 65535 is the largest block count whose triangle still fits in int32;
    // `nb * (nb + 1)` overflowed well before that
    BOOST_CHECK_EQUAL(num_triangle_blocks(46341 * 8, 8), 1073767311);
    BOOST_CHECK_EQUAL(num_triangle_blocks(65535 * 8, 8), 2147450880);
    BOOST_CHECK_EQUAL(num_triangle_blocks((int64_t)65536 * 8, 8), (int64_t)65536 * 65537 / 2);
    // nb == 2^30, whose triangle needs 60 bits
    BOOST_CHECK_EQUAL(num_triangle_blocks((int64_t)1 << 33, 8),
                      ((int64_t)1 << 29) * (((int64_t)1 << 30) + 1));
}

BOOST_AUTO_TEST_CASE(small_integer_types_promote_rather_than_wrap) {
    // Integer promotion is what keeps these safe: the result is the promoted
    // type, so it holds values that would not fit back into the argument type
    const uint8_t x8 = 250, y8 = 8;
    BOOST_CHECK_EQUAL(round_up(x8, y8), 256);
    BOOST_CHECK_EQUAL(div_ceil(x8, y8), 32);
    static_assert(std::is_same_v<decltype(round_up(x8, y8)), int>,
                  "small unsigned types promote to int");

    const int8_t xs = -100, ys = 7;
    BOOST_CHECK_EQUAL(kotekan::div(xs, ys), -15);
    BOOST_CHECK_EQUAL(mod(xs, ys), 5);
    BOOST_CHECK_EQUAL(kotekan::div(xs, ys) * ys + mod(xs, ys), xs);
}

BOOST_AUTO_TEST_CASE(mixed_signedness_with_nonnegative_values) {
    // `size_t` against an `int` literal is the shape most call sites have
    BOOST_CHECK_EQUAL(div_ceil((size_t)1000, 16), (size_t)63);
    BOOST_CHECK_EQUAL(round_down((size_t)1000, 16), (size_t)992);
    BOOST_CHECK_EQUAL(round_up((size_t)1000, 16), (size_t)1008);
    BOOST_CHECK_EQUAL(div_noremainder((size_t)1024, 16), (size_t)64);
    BOOST_CHECK_EQUAL(num_triangle_blocks((size_t)128, 16), (size_t)36);
    // ...and the reverse, plus the signed/signed pairing `div`/`mod` are used with
    BOOST_CHECK_EQUAL(div_ceil(1000, (size_t)16), (size_t)63);
    BOOST_CHECK_EQUAL(kotekan::div((std::ptrdiff_t)-7, (std::ptrdiff_t)4), (std::ptrdiff_t)-2);
    BOOST_CHECK_EQUAL(mod((std::ptrdiff_t)-7, (std::ptrdiff_t)4), (std::ptrdiff_t)1);
}

BOOST_AUTO_TEST_CASE(ringbuffer_wrap_pattern) {
    // How `mod` is used in NDArrayRingBuffer and the generated CUDA wrappers:
    // map a monotonically increasing position onto a ring of `extent` slots
    constexpr std::ptrdiff_t extent = 6;
    for (std::ptrdiff_t pos = 0; pos < 4 * extent; ++pos) {
        const std::ptrdiff_t slot = mod(pos, extent);
        BOOST_CHECK(slot >= 0);
        BOOST_CHECK(slot < extent);
        BOOST_CHECK_EQUAL(slot, pos % extent);
    }
}
