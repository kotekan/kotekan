#define BOOST_TEST_MODULE "test_modulo"

#include "N2Util.hpp"  // for N2::modulo
#include "visUtil.hpp" // for modulo, frameID

#include <boost/test/included/unit_test.hpp>
#include <cstdint>

// frameID is modulo<int> over a buffer's num_frames. These tests pin the
// reduce-on-write semantics for both copies of the class: the value stays in
// [0, n) through 2^32 increments (the point at which an unreduced int count
// wraps and `% n` becomes discontinuous unless n divides 2^32), a decrement
// below zero lands on n-1, and deltas larger than the type are reduced first.

template<typename M>
static void check_sequence_survives_2_32(unsigned n) {
    // Nine chunks of 1e9 pass through both 2^31 and 2^32.
    M x(n);
    int64_t total = 0;
    // BOOST_REQUIRE, not BOOST_CHECK: an implementation that counts unreduced
    // makes this loop signed-overflow UB, and at -O2 a soft check failure
    // becomes a non-terminating loop emitting unbounded output.
    for (int chunk = 0; chunk < 9; ++chunk) {
        x += 1000000000;
        total += 1000000000;
        BOOST_REQUIRE_EQUAL((int)x, (int)(total % n));
        BOOST_REQUIRE((int)x >= 0 && (int)x < (int)n);
    }
    // ...and keeps counting in step after the wrap.
    for (int k = 0; k < 100; ++k) {
        ++x;
        ++total;
        BOOST_REQUIRE_EQUAL((int)x, (int)(total % n));
    }
}

template<typename M>
static void check_basic_semantics(unsigned n) {
    M x(n);
    BOOST_CHECK_EQUAL((int)x, 0);

    // Post-increment returns the old value and advances the counter.
    M old = x++;
    BOOST_CHECK_EQUAL((int)old, 0);
    BOOST_CHECK_EQUAL((int)x, 1);

    // Stepping forward through a full lap lands back on 0.
    for (unsigned k = 1; k < n; ++k)
        ++x;
    BOOST_CHECK_EQUAL((int)x, 0);

    // Decrementing below zero wraps to n-1, not to the unsigned wrap of -1 mod n.
    --x;
    BOOST_CHECK_EQUAL((int)x, (int)n - 1);
    x -= 1;
    BOOST_CHECK_EQUAL((int)x, (int)n - 2);
    BOOST_CHECK_EQUAL((int)(x - 1), (int)n - 3);
    BOOST_CHECK_EQUAL((int)(x + 3), 1);

    // Assignment reduces too, negatives included.
    x = 2 * (int)n + 5;
    BOOST_CHECK_EQUAL((int)x, (int)(5 % n));
    x = -1;
    BOOST_CHECK_EQUAL((int)x, (int)n - 1);

    // Comparisons work on the reduced value.
    M y(n);
    y = 5;
    x = 5;
    BOOST_CHECK(x == y);
    ++y;
    BOOST_CHECK(x != y);
    BOOST_CHECK(x < y);
}

// An unsigned T has no negative intermediate to reduce, so the decrement path
// has to be right in the arithmetic rather than in a sign fixup: at 0 a
// decrement must land on n-1, not on ((max value) % n).
template<typename M>
static void check_unsigned_semantics(unsigned n) {
    M x(n);
    BOOST_CHECK_EQUAL((uint64_t)x, 0u);
    --x;
    BOOST_CHECK_EQUAL((uint64_t)x, (uint64_t)n - 1);
    x -= 1;
    BOOST_CHECK_EQUAL((uint64_t)x, (uint64_t)n - 2);
    ++x;
    ++x;
    BOOST_CHECK_EQUAL((uint64_t)x, 0u);
    // A delta far larger than the base, and larger than a 32-bit type.
    x += 5000000000LL;
    BOOST_CHECK_EQUAL((uint64_t)x, (uint64_t)(5000000000LL % n));
}

BOOST_AUTO_TEST_CASE(_modulo_basic) {
    check_basic_semantics<modulo<int>>(24);
    check_basic_semantics<N2::modulo<int>>(24);
    check_basic_semantics<modulo<int>>(4);
    check_basic_semantics<modulo<int>>(7);
}

BOOST_AUTO_TEST_CASE(_modulo_unsigned_base) {
    check_unsigned_semantics<modulo<size_t>>(24);
    check_unsigned_semantics<N2::modulo<size_t>>(24);
    check_unsigned_semantics<modulo<unsigned>>(7);
}

// A delta whose magnitude exceeds T: the increment is reduced before it is
// combined, so it cannot overflow the stored value on the way in.
BOOST_AUTO_TEST_CASE(_modulo_delta_larger_than_type) {
    modulo<int> x(24);
    x += int64_t(5000000000LL);
    BOOST_CHECK_EQUAL((int)x, (int)(5000000000LL % 24));
    modulo<int> y(24);
    y -= int64_t(5000000000LL);
    BOOST_CHECK_EQUAL((int)y, (int)((24 - 5000000000LL % 24) % 24));
}

BOOST_AUTO_TEST_CASE(_modulo_survives_int_wrap) {
    // 24 does not divide 2^32 (2^32 mod 24 == 16), so an unreduced count skips here.
    check_sequence_survives_2_32<modulo<int>>(24);
    check_sequence_survives_2_32<N2::modulo<int>>(24);
    // 4 does divide 2^32: continuous either way, and must stay so.
    check_sequence_survives_2_32<modulo<int>>(4);
    // An odd base, for good measure.
    check_sequence_survives_2_32<modulo<int>>(7);
}

BOOST_AUTO_TEST_CASE(_modulo_incremental_lap_count) {
    // The incremental path the pipelines actually take: one ++ per frame, checked
    // against the true count over many laps.
    modulo<int> x(24);
    for (int64_t k = 1; k <= 5000000; ++k) {
        ++x;
        if ((int)x != (int)(k % 24)) {
            BOOST_FAIL("modulo<int>(24) diverged from k % 24 at k=" << k);
            break;
        }
    }
    BOOST_CHECK_EQUAL((int)x, (int)(5000000 % 24));
}
