#define BOOST_TEST_MODULE "test_modulo"

#include "N2Util.hpp"  // for N2::modulo
#include "visUtil.hpp" // for modulo, frameID

#include <boost/test/included/unit_test.hpp>
#include <cstdint>

// frameID is modulo<int> over a buffer's num_frames. Until 2026-09-04 the count
// was kept unreduced and taken mod n (an UNSIGNED n) only when read; after 2^32
// increments the int wrapped and the unsigned conversion in that modulo jumped
// by (2^32 mod n) -- on a 24-frame buffer the sequence went 15 -> 0 and skipped
// eight slots. Two such skips left the shared bf-mask buffer with a producer
// waiting on a full slot and its consumers waiting on an empty one, wedging the
// node ~15 h after start. These tests pin the reduce-on-write semantics for both
// copies of the class.

template<typename M>
static void check_sequence_survives_2_32(unsigned n) {
    // Nine chunks of 1e9 pass through both 2^31 and 2^32 -- the old
    // implementation overflows its int at the third chunk and never recovers.
    M x(n);
    int64_t total = 0;
    for (int chunk = 0; chunk < 9; ++chunk) {
        x += 1000000000;
        total += 1000000000;
        BOOST_CHECK_EQUAL((int)x, (int)(total % n));
    }
    // ...and keeps counting in step after the wrap.
    for (int k = 0; k < 100; ++k) {
        ++x;
        ++total;
        BOOST_CHECK_EQUAL((int)x, (int)(total % n));
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

    // Decrementing below zero wraps to n-1 (the old code produced the unsigned
    // wrap of -1 mod n instead: 15, not 23, on a 24-frame buffer).
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

BOOST_AUTO_TEST_CASE(_modulo_basic) {
    check_basic_semantics<modulo<int>>(24);
    check_basic_semantics<N2::modulo<int>>(24);
    check_basic_semantics<modulo<int>>(4);
    check_basic_semantics<modulo<int>>(7);
}

BOOST_AUTO_TEST_CASE(_modulo_survives_int_wrap) {
    // 24 does not divide 2^32 (2^32 mod 24 == 16): the shape that skipped.
    check_sequence_survives_2_32<modulo<int>>(24);
    check_sequence_survives_2_32<N2::modulo<int>>(24);
    // 4 does divide 2^32, so the old code was accidentally continuous here;
    // the fix must not change that.
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
