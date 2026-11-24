#define BOOST_TEST_MODULE "test_gnss_nav_sym"

#include "gnssNavSym.hpp" // for nsym_record, NSYM_HEADER_BYTES, NSYM_MAGIC

#include <boost/test/included/unit_test.hpp>
#include <cstdint> // for int64_t, uint8_t, uint16_t, uint32_t
#include <cstring> // for memcpy
#include <vector>  // for vector

namespace {

template<typename T>
T at(const std::vector<uint8_t>& b, size_t off) {
    T v;
    std::memcpy(&v, b.data() + off, sizeof(T));
    return v;
}

} // namespace

// The layout CHORD's relay parses (NSYM v1): a u32 length, then the header fields in order,
// then the signs, bit 1 for -1, least significant bit first.
BOOST_AUTO_TEST_CASE(record_layout) {
    const std::vector<int8_t> signs = {1, -1, -1, 1, 1, 1, 1, 1, -1, 1};
    const auto r =
        gnss::nsym_record(7, 'C', 2, 45, 0xDEADBEEF, true, 1234567890123LL, signs, 1.5, 2.5, 3.5);
    BOOST_CHECK_EQUAL(gnss::NSYM_HEADER_BYTES, 54u);
    BOOST_REQUIRE_EQUAL(r.size(), 4 + 54 + 2);
    BOOST_CHECK_EQUAL(at<uint32_t>(r, 0), 54u + 2u);
    BOOST_CHECK_EQUAL(at<uint32_t>(r, 4), gnss::NSYM_MAGIC);
    BOOST_CHECK_EQUAL(at<uint16_t>(r, 8), 1);
    BOOST_CHECK_EQUAL(at<uint32_t>(r, 10), 7u);
    BOOST_CHECK_EQUAL(at<uint8_t>(r, 14), 'C');
    BOOST_CHECK_EQUAL(at<uint8_t>(r, 15), 2);
    BOOST_CHECK_EQUAL(at<uint16_t>(r, 16), 45);
    BOOST_CHECK_EQUAL(at<uint32_t>(r, 18), 0xDEADBEEFu);
    BOOST_CHECK_EQUAL(at<uint16_t>(r, 22), 1); // resolved
    BOOST_CHECK_EQUAL(at<uint16_t>(r, 24), 10);
    BOOST_CHECK_EQUAL(at<int64_t>(r, 26), 1234567890123LL);
    BOOST_CHECK_EQUAL(at<double>(r, 34), 1.5);
    BOOST_CHECK_EQUAL(at<double>(r, 42), 2.5);
    BOOST_CHECK_EQUAL(at<double>(r, 50), 3.5);
    BOOST_CHECK_EQUAL(r[58], 0x06); // symbols 1 and 2 are -1
    BOOST_CHECK_EQUAL(r[59], 0x01); // symbol 8 is -1
}

BOOST_AUTO_TEST_CASE(heartbeat) {
    const auto r = gnss::nsym_record(3, 0, 0, 0, 0, false, 0, {}, 0.0, 9.0, 0.0);
    BOOST_REQUIRE_EQUAL(r.size(), 4 + 54);
    BOOST_CHECK_EQUAL(at<uint32_t>(r, 0), 54u);
    BOOST_CHECK_EQUAL(at<uint8_t>(r, 14), 0);
    BOOST_CHECK_EQUAL(at<uint16_t>(r, 22), 0);
    BOOST_CHECK_EQUAL(at<uint16_t>(r, 24), 0);
}
