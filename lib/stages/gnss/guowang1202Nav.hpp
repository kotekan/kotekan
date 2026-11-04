#ifndef GUOWANG_1202_NAV_HPP
#define GUOWANG_1202_NAV_HPP

#include <array>
#include <stdint.h>
#include <string>

/**
 * The data sign of the GuoWang 1202.025 MHz ranging millisecond, predicted from time and the
 * satellite's message pages (docs/chord/guowang/PAYLOAD.md, PER_ELEMENT.md section 5).
 *
 * Every 2 ms frame's ranging code carries one sign, and one symbol of the low-rate channel spans
 * two frames (4 ms, 250 symbols/s). The symbols are a rate-1/2 K=7 convolutional code
 * (generators 171 and 133 octal, run continuously) over a 125 bit/s stream of 3 s periods of
 * 375 bits on the BDT 3 s grid: 12 trailer bits of unknown content, a 3-bit page (1..4, cycling
 * every 12 s), the period's BDT second of week (20 bits), a 316-bit page payload that is
 * constant per satellite and page, and a CRC-24Q over the page, SOW and payload. With the
 * trailer taken as zero the predicted signs agree with the sky at 98 % (11-04, two satellites).
 *
 * Frames are counted from the BDT week start: frame f starts at transmit time f x 2 ms, so
 * period f / 1500 and symbol (f mod 1500) / 2 of it. BDT = UTC + 4 s.
 */
namespace guowang {

constexpr int GW1202_NAV_PERIOD_BITS = 375;   ///< 3 s at 125 bit/s
constexpr int GW1202_NAV_TRAILER_BITS = 12;   ///< bits 0..11 of a period: content unknown, taken as 0
constexpr int GW1202_NAV_PAGE_BITS = 316;     ///< payload bits per page
constexpr int GW1202_NAV_PAGES = 4;           ///< pages 1..4, one per period, 12 s cycle
constexpr int GW1202_NAV_PERIOD_SYMBOLS = 750; ///< 250 symbols/s x 3 s
constexpr int GW1202_NAV_PERIOD_FRAMES = 1500; ///< 2 ms frames per period
constexpr int GW1202_NAV_PERIODS_PER_WEEK = 201600;
constexpr int GW1202_NAV_BDT_MINUS_UTC_S = 4;

/// One satellite's page payloads, bits 0/1 in transmission order (as parse_slow.py prints them,
/// 79 hex digits per page).
struct Gw1202NavPages {
    std::array<std::array<uint8_t, GW1202_NAV_PAGE_BITS>, GW1202_NAV_PAGES> page;
};

/// Parse the four pages from 79-digit hex strings (page 1 first). False on a bad length or digit.
bool parse_gw1202_nav_pages(const std::array<std::string, GW1202_NAV_PAGES>& hex,
                            Gw1202NavPages& out);

/// CRC-24Q (polynomial 0x1864CFB, zero initial value, no final XOR) over @p n bits, first bit
/// first, as GPS CNAV and BeiDou use it.
uint32_t gw1202_crc24q(const uint8_t* bits, int n);

/// The 375 bits of the period that starts at BDT second of week @p sow (a multiple of 3):
/// 12 zero trailer bits, the page ((sow / 3) mod 4 + 1), the SOW, the payload, the CRC.
void gw1202_nav_period(const Gw1202NavPages& pages, uint32_t sow,
                       uint8_t out[GW1202_NAV_PERIOD_BITS]);

/// Encode @p n bits into 2n symbols (+1 for logic 0, -1 for 1). @p state is the encoder's six
/// previous bits and carries between calls; start a stream from 0.
void gw1202_nav_encode(const uint8_t* bits, int n, uint8_t& state, int8_t* symbols);

/// The sign of every frame in [frame_first, frame_first + n), frames counted from the BDT week
/// start (f = floor(t_tx / 2 ms)). The encoder is run from the period before the first one
/// asked for, so its state at a period start is right up to the unknown trailer bits.
void gw1202_nav_signs(const Gw1202NavPages& pages, int64_t frame_first, int n, int8_t* signs);

} // namespace guowang

#endif // GUOWANG_1202_NAV_HPP
