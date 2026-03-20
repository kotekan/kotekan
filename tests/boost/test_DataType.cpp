#define BOOST_TEST_MODULE "test_DataType"

#include <DataType.hpp>
#include <boost/test/included/unit_test.hpp>

using namespace kotekan;

BOOST_AUTO_TEST_CASE(test1) {
    std::cout << "Testing DataType class...\n";

    std::cout << "Test 1:\n";
    std::cout << "    uint1x8:     " << uint1x8 << "\n";
    std::cout << "    uint4x2:     " << uint4x2 << "\n";
    std::cout << "    uint8:       " << uint8 << "\n";
    std::cout << "    uint16:      " << uint16 << "\n";
    std::cout << "    uint32:      " << uint32 << "\n";
    std::cout << "    uint64:      " << uint64 << "\n";
    std::cout << "    int4x2:      " << int4x2 << "\n";
    std::cout << "    int4x2_swapped_withoffset: " << int4x2_swapped_withoffset << "\n";
    std::cout << "    int8:        " << int8 << "\n";
    std::cout << "    int16:       " << int16 << "\n";
    std::cout << "    int32:       " << int32 << "\n";
    std::cout << "    int64:       " << int64 << "\n";
    std::cout << "    float16:     " << float16 << "\n";
    std::cout << "    float32:     " << float32 << "\n";
    std::cout << "    float64:     " << float64 << "\n";

    std::cout << "Test 2:\n";
    std::cout << "    unsigned char:      " << GetDataType_v<unsigned char> << "\n";
    std::cout << "    unsigned short:     " << GetDataType_v<unsigned short> << "\n";
    std::cout << "    unsigned int:       " << GetDataType_v<unsigned int> << "\n";
    std::cout << "    unsigned long:      " << GetDataType_v<unsigned long> << "\n";
    std::cout << "    unsigned long long: " << GetDataType_v<unsigned long long> << "\n";
    std::cout << "    signed char:        " << GetDataType_v<signed char> << "\n";
    std::cout << "    short:              " << GetDataType_v<short> << "\n";
    std::cout << "    int:                " << GetDataType_v<int> << "\n";
    std::cout << "    long:               " << GetDataType_v<long> << "\n";
    std::cout << "    long long:          " << GetDataType_v<long long> << "\n";
#if KOTEKAN_FLOAT16
    std::cout << "    float16_t:          " << GetDataType_v<float16_t> << "\n";
#endif
    std::cout << "    float:              " << GetDataType_v<float> << "\n";
    std::cout << "    double:             " << GetDataType_v<double> << "\n";

    std::cout << "Success.\n";
}

// These must always hold

static_assert(GetDataType_v<GetType_t<uint1x8>> == uint1x8);
static_assert(GetDataType_v<GetType_t<uint4x2>> == uint4x2);
static_assert(GetDataType_v<GetType_t<uint8>> == uint8);
static_assert(GetDataType_v<GetType_t<uint16>> == uint16);
static_assert(GetDataType_v<GetType_t<uint32>> == uint32);
static_assert(GetDataType_v<GetType_t<uint64>> == uint64);
static_assert(GetDataType_v<GetType_t<int4x2>> == int4x2);
static_assert(GetDataType_v<GetType_t<int4x2_swapped_withoffset>> == int4x2_swapped_withoffset);
static_assert(GetDataType_v<GetType_t<int8>> == int8);
static_assert(GetDataType_v<GetType_t<int16>> == int16);
static_assert(GetDataType_v<GetType_t<int32>> == int32);
static_assert(GetDataType_v<GetType_t<int64>> == int64);
static_assert(GetDataType_v<GetType_t<float16>> == float16);
static_assert(GetDataType_v<GetType_t<float32>> == float32);
static_assert(GetDataType_v<GetType_t<float64>> == float64);

// Test the low-bit data types exhaustively at compile time

BOOST_AUTO_TEST_CASE(test_uint1x8_t) {
    std::cout << "Testing uint1x8_t...\n";
    for (int val = 0x00; val <= 0xff; ++val) {
        const bool v0 = val & 0x01;
        const bool v1 = val & 0x02;
        const bool v2 = val & 0x04;
        const bool v3 = val & 0x08;
        const bool v4 = val & 0x10;
        const bool v5 = val & 0x20;
        const bool v6 = val & 0x40;
        const bool v7 = val & 0x80;
        const uint1x8_t x(v0, v1, v2, v3, v4, v5, v6, v7);
        BOOST_CHECK_EQUAL(x.val, val);
        const uint1x8_t y{std::uint8_t(val)};
        BOOST_CHECK_EQUAL(y[0], v0);
        BOOST_CHECK_EQUAL(y[1], v1);
        BOOST_CHECK_EQUAL(y[2], v2);
        BOOST_CHECK_EQUAL(y[3], v3);
        BOOST_CHECK_EQUAL(y[4], v4);
        BOOST_CHECK_EQUAL(y[5], v5);
        BOOST_CHECK_EQUAL(y[6], v6);
        BOOST_CHECK_EQUAL(y[7], v7);
        for (int val2 = 0x00; val2 <= 0xff; ++val2) {
            const bool w0 = val2 & 0x01;
            const bool w1 = val2 & 0x02;
            const bool w2 = val2 & 0x04;
            const bool w3 = val2 & 0x08;
            const bool w4 = val2 & 0x10;
            const bool w5 = val2 & 0x20;
            const bool w6 = val2 & 0x40;
            const bool w7 = val2 & 0x80;
            const uint1x8_t x2(w0, w1, w2, w3, w4, w5, w6, w7);
            BOOST_CHECK_EQUAL(x2 == x, val == val2);
            BOOST_CHECK_EQUAL(x2 != x, val != val2);
        }
    }
    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_uint4x2_t) {
    std::cout << "Testing uint4x2_t...\n";
    for (int val = 0x00; val <= 0xff; ++val) {
        const unsigned int v0 = (val & 0x0f) >> 0;
        const unsigned int v1 = (val & 0xf0) >> 4;
        BOOST_REQUIRE(v0 <= 15);
        BOOST_REQUIRE(v1 <= 15);
        const uint4x2_t x(v0, v1);
        BOOST_CHECK_EQUAL(x.val, val);
        const uint4x2_t y{std::uint8_t(val)};
        BOOST_CHECK_EQUAL(y[0], v0);
        BOOST_CHECK_EQUAL(y[1], v1);
        for (int val2 = 0x00; val2 <= 0xff; ++val2) {
            const unsigned int w0 = (val2 & 0x0f) >> 0;
            const unsigned int w1 = (val2 & 0xf0) >> 4;
            const uint4x2_t x2(w0, w1);
            BOOST_CHECK_EQUAL(x2 == x, val == val2);
            BOOST_CHECK_EQUAL(x2 != x, val != val2);
        }
    }
    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_int4x2_t) {
    std::cout << "Testing int4x2_t...\n";
    for (int val = 0x00; val <= 0xff; ++val) {
        const int v0 = std::int8_t(((val & 0x0f) >> 0) << 4) >> 4;
        const int v1 = std::int8_t(((val & 0xf0) >> 4) << 4) >> 4;
        BOOST_REQUIRE(-8 <= v0 && v0 <= +7);
        BOOST_REQUIRE(-8 <= v1 && v1 <= +7);
        const int4x2_t x(v0, v1);
        BOOST_CHECK_EQUAL(x.val, val);
        const int4x2_t y{std::uint8_t(val)};
        BOOST_CHECK_EQUAL(y[0], v0);
        BOOST_CHECK_EQUAL(y[1], v1);
        for (int val2 = 0x00; val2 <= 0xff; ++val2) {
            const int w0 = std::int8_t(((val2 & 0x0f) >> 0) << 4) >> 4;
            const int w1 = std::int8_t(((val2 & 0xf0) >> 4) << 4) >> 4;
            const int4x2_t x2(w0, w1);
            BOOST_CHECK_EQUAL(x2 == x, val == val2);
            BOOST_CHECK_EQUAL(x2 != x, val != val2);
        }
    }
    std::cout << "Success.\n";
}

BOOST_AUTO_TEST_CASE(test_int4x2_swapped_withoffset_t) {
    std::cout << "Testing int4x2_swapped_withoffset_t...\n";
    for (int val = 0x00; val <= 0xff; ++val) {
        const int v0 = ((val & 0x0f) >> 0) - 8;
        const int v1 = ((val & 0xf0) >> 4) - 8;
        BOOST_REQUIRE(-8 <= v0 && v0 <= +7);
        BOOST_REQUIRE(-8 <= v1 && v1 <= +7);
        const int4x2_swapped_withoffset_t x(v0, v1);
        BOOST_CHECK_EQUAL(x.val, val);
        const int4x2_swapped_withoffset_t y{std::uint8_t(val)};
        BOOST_CHECK_EQUAL(y[0], v0);
        BOOST_CHECK_EQUAL(y[1], v1);
        for (int val2 = 0x00; val2 <= 0xff; ++val2) {
            const int w0 = ((val2 & 0x0f) >> 0) - 8;
            const int w1 = ((val2 & 0xf0) >> 4) - 8;
            const int4x2_swapped_withoffset_t x2(w0, w1);
            BOOST_CHECK_EQUAL(x2 == x, val == val2);
            BOOST_CHECK_EQUAL(x2 != x, val != val2);
        }
    }
    std::cout << "Success.\n";
}
