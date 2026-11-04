#include "guowang1202Nav.hpp"

#include <vector>

namespace guowang {

namespace {

constexpr uint32_t CRC24Q_POLY = 0x1864CFB;
constexpr unsigned G1 = 0171, G2 = 0133; // octal, GPS convention: G1 gives the first symbol

int parity(unsigned x) {
    return __builtin_popcount(x) & 1;
}

int hex_digit(char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

void put_bits(uint8_t* out, uint32_t value, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (uint8_t)((value >> (n - 1 - i)) & 1u);
}

} // namespace

bool parse_gw1202_nav_pages(const std::array<std::string, GW1202_NAV_PAGES>& hex,
                            Gw1202NavPages& out) {
    for (int p = 0; p < GW1202_NAV_PAGES; p++) {
        const std::string& h = hex[(size_t)p];
        if ((int)h.size() * 4 != GW1202_NAV_PAGE_BITS)
            return false;
        for (size_t i = 0; i < h.size(); i++) {
            const int d = hex_digit(h[i]);
            if (d < 0)
                return false;
            put_bits(&out.page[(size_t)p][4 * i], (uint32_t)d, 4);
        }
    }
    return true;
}

uint32_t gw1202_crc24q(const uint8_t* bits, int n) {
    uint32_t c = 0;
    for (int i = 0; i < n; i++) {
        c ^= (uint32_t)(bits[i] & 1) << 23;
        c = (c & 0x800000) ? ((c << 1) ^ CRC24Q_POLY) & 0xFFFFFF : (c << 1) & 0xFFFFFF;
    }
    return c;
}

void gw1202_nav_period(const Gw1202NavPages& pages, uint32_t sow,
                       uint8_t out[GW1202_NAV_PERIOD_BITS]) {
    const int page = (int)((sow / 3) % GW1202_NAV_PAGES) + 1;
    uint8_t* b = out;
    for (int i = 0; i < GW1202_NAV_TRAILER_BITS; i++)
        *b++ = 0;
    uint8_t* block = b;
    put_bits(b, (uint32_t)page, 3);
    b += 3;
    put_bits(b, sow, 20);
    b += 20;
    const auto& pl = pages.page[(size_t)(page - 1)];
    for (int i = 0; i < GW1202_NAV_PAGE_BITS; i++)
        *b++ = pl[(size_t)i];
    put_bits(b, gw1202_crc24q(block, (int)(b - block)), 24);
}

void gw1202_nav_encode(const uint8_t* bits, int n, uint8_t& state, int8_t* symbols) {
    unsigned s = state & 0x3F;
    for (int i = 0; i < n; i++) {
        const unsigned reg = ((unsigned)(bits[i] & 1) << 6) | s;
        symbols[2 * i] = (int8_t)(1 - 2 * parity(reg & G1));
        symbols[2 * i + 1] = (int8_t)(1 - 2 * parity(reg & G2));
        s = reg >> 1;
    }
    state = (uint8_t)s;
}

void gw1202_nav_signs(const Gw1202NavPages& pages, int64_t frame_first, int n, int8_t* signs) {
    if (n <= 0)
        return;
    const int64_t p0 = frame_first / GW1202_NAV_PERIOD_FRAMES
                       - (frame_first < 0 && frame_first % GW1202_NAV_PERIOD_FRAMES ? 1 : 0);
    const int64_t p1 = (frame_first + n - 1) / GW1202_NAV_PERIOD_FRAMES;
    uint8_t bits[GW1202_NAV_PERIOD_BITS];
    int8_t sym[GW1202_NAV_PERIOD_SYMBOLS];
    uint8_t state = 0;
    for (int64_t p = p0 - 1; p <= p1; p++) {
        int64_t pw = p % GW1202_NAV_PERIODS_PER_WEEK;
        if (pw < 0)
            pw += GW1202_NAV_PERIODS_PER_WEEK;
        gw1202_nav_period(pages, (uint32_t)(3 * pw), bits);
        gw1202_nav_encode(bits, GW1202_NAV_PERIOD_BITS, state, sym);
        if (p < p0)
            continue;
        for (int f = 0; f < GW1202_NAV_PERIOD_FRAMES; f++) {
            const int64_t k = p * GW1202_NAV_PERIOD_FRAMES + f - frame_first;
            if (k >= 0 && k < n)
                signs[k] = sym[f / 2];
        }
    }
}

} // namespace guowang
