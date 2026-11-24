#include "gnssNavSym.hpp"

#include <cstring> // for memcpy

namespace gnss {

namespace {

template<typename T>
void put(std::vector<uint8_t>& out, T v) {
    uint8_t b[sizeof(T)];
    std::memcpy(b, &v, sizeof(T)); // the hosts are little-endian (x86-64, aarch64)
    out.insert(out.end(), b, b + sizeof(T));
}

} // namespace

std::vector<uint8_t> nsym_record(uint32_t seq, char sys, uint8_t sig, uint16_t prn,
                                 uint32_t stream_id, bool resolved, int64_t sym0,
                                 const std::vector<int8_t>& signs, double t_dec, double t_out,
                                 double rx_utc_sym0) {
    const size_t n = signs.size();
    const uint32_t len = (uint32_t)(NSYM_HEADER_BYTES + (n + 7) / 8);
    std::vector<uint8_t> out;
    out.reserve(4 + len);
    put<uint32_t>(out, len);
    put<uint32_t>(out, NSYM_MAGIC);
    put<uint16_t>(out, 1);
    put<uint32_t>(out, seq);
    put<uint8_t>(out, (uint8_t)sys);
    put<uint8_t>(out, sig);
    put<uint16_t>(out, prn);
    put<uint32_t>(out, stream_id);
    put<uint16_t>(out, resolved ? 1 : 0);
    put<uint16_t>(out, (uint16_t)n);
    put<int64_t>(out, sym0);
    put<double>(out, t_dec);
    put<double>(out, t_out);
    put<double>(out, rx_utc_sym0);
    out.resize(4 + len, 0);
    for (size_t i = 0; i < n; ++i)
        if (signs[i] < 0)
            out[4 + NSYM_HEADER_BYTES + i / 8] |= (uint8_t)(1u << (i % 8));
    return out;
}

} // namespace gnss
