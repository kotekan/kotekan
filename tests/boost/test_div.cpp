#define BOOST_TEST_MODULE "test_div"

#include "div.hpp"

#include <boost/test/included/unit_test.hpp>
#include <cstddef>
#include <stdexcept>

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
