#define BOOST_TEST_MODULE "test_guowang_1202_nav"

#include "guowang1202Nav.hpp" // for gw1202_crc24q, gw1202_nav_period, gw1202_nav_encode, ...

#include <array>
#include <boost/test/included/unit_test.hpp>
#include <cstdint>
#include <string>
#include <vector>

using namespace guowang;

namespace {

// DIGUI-H24-4's pages as parse_slow.py printed them from the 2026-11-04 08:51Z window (w078);
// every repeat of a page was identical.
const std::array<std::string, 4> H24_4 = {
    "28b21fabc3bfffea88000002a0d2ad2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d5",
    "53535353535353535353535320d2c53535353535353535353535353535353535353535353535356",
    "53d21fd3d3d3d3d3d3d3d3d3a0d2dd3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d7",
    "54545454545454545454545420d2f54545454545454545454545454545454545454545454545450"};

uint32_t bits_value(const uint8_t* b, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; i++)
        v = (v << 1) | (b[i] & 1);
    return v;
}

std::string sign_string(const int8_t* s, int n) {
    std::string out;
    for (int i = 0; i < n; i++)
        out += s[i] < 0 ? '1' : '0';
    return out;
}

} // namespace

BOOST_AUTO_TEST_CASE(pages_parse_and_the_crc_matches_the_sky) {
    Gw1202NavPages pages;
    BOOST_REQUIRE(parse_gw1202_nav_pages(H24_4, pages));
    BOOST_CHECK_EQUAL(bits_value(pages.page[0].data(), 8), 0x28u);
    BOOST_CHECK_EQUAL(bits_value(&pages.page[3][312], 4), 0x0u);
    std::array<std::string, 4> bad = H24_4;
    bad[1] += "0";
    BOOST_CHECK(!parse_gw1202_nav_pages(bad, pages));
    bad = H24_4;
    bad[2][5] = 'g';
    BOOST_CHECK(!parse_gw1202_nav_pages(bad, pages));
    BOOST_REQUIRE(parse_gw1202_nav_pages(H24_4, pages));

    // The CRCs the satellite sent with two clean subframes (parse_slow.py, w078): page 4 at
    // SOW 291069 -> f22358, page 1 at SOW 291108 -> 131dd3.
    uint8_t period[GW1202_NAV_PERIOD_BITS];
    gw1202_nav_period(pages, 291069, period);
    for (int i = 0; i < GW1202_NAV_TRAILER_BITS; i++)
        BOOST_CHECK_EQUAL(period[i], 0);
    BOOST_CHECK_EQUAL(bits_value(period + 12, 3), 4u);
    BOOST_CHECK_EQUAL(bits_value(period + 15, 20), 291069u);
    BOOST_CHECK_EQUAL(bits_value(period + 351, 24), 0xf22358u);
    gw1202_nav_period(pages, 291108, period);
    BOOST_CHECK_EQUAL(bits_value(period + 12, 3), 1u);
    BOOST_CHECK_EQUAL(bits_value(period + 351, 24), 0x131dd3u);
}

BOOST_AUTO_TEST_CASE(encoder_matches_the_reference) {
    // The period at SOW 291069 encoded from state 0 (python reference, logic 1 -> '1' = -1).
    Gw1202NavPages pages;
    BOOST_REQUIRE(parse_gw1202_nav_pages(H24_4, pages));
    uint8_t period[GW1202_NAV_PERIOD_BITS];
    gw1202_nav_period(pages, 291069, period);
    int8_t sym[GW1202_NAV_PERIOD_SYMBOLS];
    uint8_t state = 0;
    gw1202_nav_encode(period, GW1202_NAV_PERIOD_BITS, state, sym);
    const std::string s = sign_string(sym, GW1202_NAV_PERIOD_SYMBOLS);
    BOOST_CHECK_EQUAL(s.substr(0, 64),
                      "0000000000000000000000001110111111110011110001100010100110100101");
    BOOST_CHECK_EQUAL(s.substr(750 - 32), "01001111001111001011010100100100");
    BOOST_CHECK_EQUAL(state, 0x06);
    int neg = 0;
    for (char c : s)
        neg += c == '1';
    BOOST_CHECK_EQUAL(neg, 302);
}

BOOST_AUTO_TEST_CASE(frame_signs_follow_symbols_and_carry_the_state) {
    Gw1202NavPages pages;
    BOOST_REQUIRE(parse_gw1202_nav_pages(H24_4, pages));
    // Period 97023 = SOW 291069. Ask for frames across its start and the next period's start.
    const int64_t f0 = 97023LL * GW1202_NAV_PERIOD_FRAMES - 10;
    const int n = 2 * GW1202_NAV_PERIOD_FRAMES + 20;
    std::vector<int8_t> signs((size_t)n);
    gw1202_nav_signs(pages, f0, n, signs.data());
    // Frames pair up: an even frame and the next share a symbol.
    for (int k = 10; k + 1 < n; k += 2)
        BOOST_CHECK_EQUAL(signs[(size_t)k], signs[(size_t)k + 1]);
    // The first 24 symbols of a period are the zero trailer: with the encoder state from the
    // previous CRC they are not all +1 (the state carries), unlike an encoder started from 0.
    uint8_t period[GW1202_NAV_PERIOD_BITS];
    int8_t sym[GW1202_NAV_PERIOD_SYMBOLS];
    uint8_t state = 0;
    gw1202_nav_period(pages, 291066, period);
    gw1202_nav_encode(period, GW1202_NAV_PERIOD_BITS, state, sym);
    gw1202_nav_period(pages, 291069, period);
    gw1202_nav_encode(period, GW1202_NAV_PERIOD_BITS, state, sym);
    for (int f = 0; f < GW1202_NAV_PERIOD_FRAMES; f++)
        BOOST_CHECK_EQUAL(signs[(size_t)(10 + f)], sym[f / 2]);
    // The symbols after the trailer and the state do not depend on where the stream started.
    uint8_t state2 = 0;
    int8_t sym2[GW1202_NAV_PERIOD_SYMBOLS];
    gw1202_nav_encode(period, GW1202_NAV_PERIOD_BITS, state2, sym2);
    for (int k = 2 * (GW1202_NAV_TRAILER_BITS + 6); k < GW1202_NAV_PERIOD_SYMBOLS; k++)
        BOOST_CHECK_EQUAL(sym[k], sym2[k]);
    BOOST_CHECK_EQUAL(state, state2);
}
