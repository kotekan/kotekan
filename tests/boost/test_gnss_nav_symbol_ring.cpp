#define BOOST_TEST_MODULE "test_gnss_nav_symbol_ring"

#include "gnssNavSymbolRing.hpp" // for NavSymbolRing

#include <boost/test/included/unit_test.hpp>
#include <cstdint> // for int64_t, int8_t
#include <vector>  // for vector

using gnss::NavSymbolRing;

BOOST_AUTO_TEST_CASE(span_held_only_when_complete) {
    NavSymbolRing r;
    const int64_t base = 46'000'000'000'000LL; // ~46e9 s of 1 ms symbols since the GPS epoch
    std::vector<int8_t> s = {1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1};
    r.put(base, (int)s.size(), s.data(), 7);
    int8_t out[12];
    BOOST_TEST(r.get(base, 12, out));
    for (int k = 0; k < 12; ++k)
        BOOST_TEST(out[k] == s[k]);
    BOOST_TEST(r.get(base + 3, 4, out));
    BOOST_TEST(out[0] == s[3]);
    // One period past the held span, or before it: no wipe.
    BOOST_TEST(!r.get(base + 1, 12, out));
    BOOST_TEST(!r.get(base - 1, 2, out));
}

BOOST_AUTO_TEST_CASE(out_of_order_and_overwrite) {
    NavSymbolRing r;
    const int8_t a[2] = {1, 1}, b[2] = {-1, -1};
    r.put(102, 2, b, 1);
    r.put(100, 2, a, 1);
    int8_t out[4];
    BOOST_TEST(r.get(100, 4, out));
    BOOST_TEST(out[0] == 1);
    BOOST_TEST(out[3] == -1);
    // N symbols later the slot holds the new index, not the old one.
    r.put(100 + NavSymbolRing::N, 2, b, 1);
    BOOST_TEST(!r.get(100, 1, out));
    BOOST_TEST(r.get(100 + NavSymbolRing::N, 1, out));
    BOOST_TEST(out[0] == -1);
}

BOOST_AUTO_TEST_CASE(new_stream_drops_held) {
    NavSymbolRing r;
    const int8_t a[3] = {1, -1, 1};
    r.put(5, 3, a, 1);
    r.put(8, 3, a, 2);
    int8_t out[6];
    BOOST_TEST(!r.get(5, 6, out));
    BOOST_TEST(r.get(8, 3, out));
}

BOOST_AUTO_TEST_CASE(negative_index) {
    NavSymbolRing r;
    const int8_t a[2] = {-1, 1};
    r.put(-1, 2, a, 0);
    int8_t out[2];
    BOOST_TEST(r.get(-1, 2, out));
    BOOST_TEST(out[0] == -1);
    BOOST_TEST(out[1] == 1);
}
