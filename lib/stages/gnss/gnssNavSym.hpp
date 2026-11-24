/**
 * @file
 * @brief The nav-symbol feed's rows and records: GnssNavSymbols' out_buf frames and the NSYM
 *        stream GnssNavSymbolSend writes.
 */

#ifndef GNSS_NAV_SYM_HPP
#define GNSS_NAV_SYM_HPP

#include <cstddef> // for size_t
#include <cstdint> // for int8_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t
#include <vector>  // for vector

namespace gnss {

/// Decisions a slot can add in one frame: a 1 ms record ends at most one symbol.
constexpr int NAV_SYM_MAX = 8;

/// One slot's newest decisions in a GnssNavSymbols out_buf frame. A frame is an int32 row
/// count, four bytes of padding, and that many rows.
struct NavSymRow {
    int32_t prn;
    int32_t n;         ///< decisions in sign, at most NAV_SYM_MAX
    int64_t first;     ///< the stream index of sign[0]
    int64_t stream_id; ///< the stream's start (t0) in ms
    double t0;     ///< the stream's start, host-labelled UTC: symbol k starts at t0 + k * sym_dt
    double sym_dt; ///< the symbol period on the receive-time grid, s
    double t_dec;  ///< host UTC when the decisions were made
    int8_t sign[NAV_SYM_MAX]; ///< +-1, the stream's open sign
};
static_assert(sizeof(NavSymRow) == 56, "NavSymRow should be 56 bytes");

/// Bytes of an out_buf frame for @p n_prn slots.
inline size_t nav_sym_frame_bytes(int n_prn) {
    return 8 + (size_t)n_prn * sizeof(NavSymRow);
}

/// A stream's anchor, from a decoded message: its symbol @c k has satellite-time index @c sv,
/// and the stream's symbols times @c sign are the transmitted ones (+1 for a 0 bit).
struct NavSymAnchor {
    int64_t k;
    int64_t sv;
    int sign;
};

/// One NSYM v1 record with its u32 length prefix, little-endian: u32 magic "NSYM", u16
/// version, u32 seq, u8 sys, u8 sig, u16 prn, u32 stream_id, u16 flags (bit0 resolved), u16 n,
/// i64 sym0, f64 t_dec, f64 t_out, f64 rx_utc_sym0, then ceil(n/8) bytes of signs (bit 1 = -1,
/// LSB first). No signs and sys, sig and prn 0 make a heartbeat.
std::vector<uint8_t> nsym_record(uint32_t seq, char sys, uint8_t sig, uint16_t prn,
                                 uint32_t stream_id, bool resolved, int64_t sym0,
                                 const std::vector<int8_t>& signs, double t_dec, double t_out,
                                 double rx_utc_sym0);

/// The NSYM header's length, without the length prefix.
constexpr size_t NSYM_HEADER_BYTES = 4 + 2 + 4 + 1 + 1 + 2 + 4 + 2 + 2 + 8 + 8 + 8 + 8;

constexpr uint32_t NSYM_MAGIC = 0x4D59534E;

} // namespace gnss

#endif // GNSS_NAV_SYM_HPP
