#define BOOST_TEST_MODULE "test_guowang_1202_code"

#include "guowang1202Code.hpp" // for generate_gw1202_code, generate_gw1202_frame, GW1202_*

#include <array>
#include <boost/test/included/unit_test.hpp>
#include <cstdint>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace guowang;

namespace {

// The code as numpy.packbits(chips < 0).hex(): MSB first, 1 = chip -1, the last byte padded.
std::string packed_hex(const std::array<int8_t, GW1202_CODE_LENGTH>& c) {
    std::string out;
    for (int b = 0; b < (GW1202_CODE_LENGTH + 7) / 8; ++b) {
        unsigned v = 0;
        for (int i = 0; i < 8; ++i) {
            const int k = 8 * b + i;
            v = (v << 1) | ((k < GW1202_CODE_LENGTH && c[(size_t)k] < 0) ? 1u : 0u);
        }
        char h[3];
        std::snprintf(h, sizeof h, "%02x", v);
        out += h;
    }
    return out;
}

} // namespace

// The codes read off the sky chip for chip: DIGUI-26 (the 2026-10-30 baseband dump) and DIGUI-79
// (the 2026-11-02 beam dump), as python/scripts/gnss/guowang/test_guowang.py test_guowang_1202
// holds them.
BOOST_AUTO_TEST_CASE(matches_the_codes_read_off_the_sky) {
    BOOST_CHECK_EQUAL(
        packed_hex(generate_gw1202_code(1213)),
        "7c69ad3164f0c5f7e073030bd30ac1c3cac03395bccd8632a0787a16cca62e8a1ea8f4016d6a429f1eb1b2a16a"
        "eb7ca576b03305ef3fc9d0869bb8fe4f46eded1ba6b148894cce558333b0c73e868aa0cc2b004a36f31cbc251a"
        "2223192b369ead19916c55daa4a10a50d681bf27437beb4b5e11c2ba40bd2119f8fb7679256d3c5775342f7ed9"
        "1af2516b712c5c3d559363b1a86974a0000bbcef916af0a4e95cf40308bb9d08a3a2bf6de97a96b008fc47ddaa"
        "3c82716108ef22ac70804fa1b8f95936595303201660f2a72d5b27cf4bf339dac26d5568062290f39ba72cb5ce"
        "4a834269849ad33173940ea2fad9e181da0961b1968134acfd1e4d5db08788");
    BOOST_CHECK_EQUAL(
        packed_hex(generate_gw1202_code(2027)),
        "30746637c4100cc9ef44a13aa55326e5ad8ffe9420209355d5f157aae61c4dc0045fff7dd4d9c21fa44f60535f"
        "71e20444586c0ce5e5d95a41a924291e0db8a3991ea46130b182a1588593394ba87bd804c72e1c92ff12c43ea2"
        "90d664342c276a11499724565eebb77b191f50d9df45aab08841fc12abdbf239ec00eee56fa49fc0aa548fea98"
        "7cbce9974825226f1d582bdd3cc1025e3be939f61177d2b2f93f23811772705308a35708a45c6c672d3f4ad2a7"
        "da9f27ff40cd7a706736854f7d54b6f28141621c6f34f39c1c8aa2ec8fc23b0b7d0abd79c35c2babe06c5e9d76"
        "c2321356598c100638b5c87803439630586220048ca56b4ce774dec9cba5e8");
}

BOOST_AUTO_TEST_CASE(every_phase_is_a_distinct_balanced_code) {
    std::set<std::string> seen;
    for (int ph = 0; ph < GW1202_G2P_PERIOD; ++ph) {
        const auto c = generate_gw1202_code(ph);
        int sum = 0;
        for (int8_t x : c) {
            BOOST_REQUIRE(x == 1 || x == -1);
            sum += x;
        }
        // A Gold code's imbalance is one of the three-valued correlation levels, |sum| <= 65+.
        BOOST_CHECK_LE(std::abs(sum), 130);
        seen.insert(packed_hex(c));
    }
    BOOST_CHECK_EQUAL((int)seen.size(), GW1202_G2P_PERIOD);
}

BOOST_AUTO_TEST_CASE(frame_is_the_code_then_zeros_and_prn_is_phase_plus_one) {
    const std::vector<int8_t> f = generate_gw1202_frame(2028);
    BOOST_REQUIRE_EQUAL((int)f.size(), GW1202_FRAME_LENGTH);
    const auto c = generate_gw1202_code(2027);
    for (int k = 0; k < GW1202_CODE_LENGTH; ++k)
        BOOST_CHECK_EQUAL(f[(size_t)k], c[(size_t)k]);
    for (int k = GW1202_CODE_LENGTH; k < GW1202_FRAME_LENGTH; ++k)
        BOOST_CHECK_EQUAL(f[(size_t)k], 0);
    BOOST_CHECK_THROW(generate_gw1202_frame(0), std::out_of_range);
    BOOST_CHECK_THROW(generate_gw1202_frame(GW1202_MAX_PRN + 1), std::out_of_range);
    BOOST_CHECK_NO_THROW(generate_gw1202_frame(GW1202_MAX_PRN));
    BOOST_CHECK_THROW(generate_gw1202_code(-1), std::out_of_range);
    BOOST_CHECK_THROW(generate_gw1202_code(GW1202_G2P_PERIOD), std::out_of_range);
}
