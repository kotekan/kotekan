/**
 * @file
 * @brief Wire format for the distributed channelized search: per-channel
 *        coarse-correlation surfaces P_c shipped from @ref GnssChannelCorrelator
 *        to @ref GnssSearchAggregator.
 *
 * One frame == one channel's contribution for one acquisition snapshot. The
 * payload is the per-channel correlation P_c[prn][win][d][q] (complex float),
 * the distributable half of @ref gnss::channelized_acquire (see
 * @ref gnss::channel_correlate). Only channels whose passband covers the carrier
 * (@c covers == 1) carry a payload; the rest emit a header-only frame so the
 * aggregator's lock-step gather stays aligned across every channel buffer.
 */

#ifndef GNSS_CORR_FRAME_HPP
#define GNSS_CORR_FRAME_HPP

#include <cstddef> // for size_t
#include <cstdint> // for int32_t, int64_t

namespace gnss {

/// Header at the start of every @ref GnssChannelCorrelator output frame.
/// Followed (covering channels only) by the complex P_c payload
/// (@c n_prn * nwin * nd * Mp, PRN-major then window, Doppler, coarse-lag), then
/// the raw window-0 block (@c raw_hops complex) this channel contributed -- the
/// voltage the central refine (@ref GnssSearchAggregator) despreads on a fine cp
/// grid to climb from the coarse acquire peak to the true (lockable) code phase.
struct CorrFrameHeader {
    int32_t magic;          ///< CORR_FRAME_MAGIC
    int32_t chan_freq;      ///< global frequency index of this correlator's channel
    int64_t snap_start_hop; ///< absolute hop index of snapshot hop 0
    int32_t covers;         ///< 1 if this channel covers the carrier (payload present)
    int32_t n_prn;          ///< PRNs in the payload
    int32_t nwin;           ///< windows per snapshot
    int32_t nd;             ///< Doppler trials
    int32_t Mp;             ///< coarse-lag range (replica hop-period)
    int32_t sph;            ///< full-rate samples per hop (fine-lag range / r2c 2N)
    int32_t raw_hops;       ///< hops of window-0 raw appended for the refine (= hops/record)
};

constexpr int32_t CORR_FRAME_MAGIC = 0x47435231; // 'GCR1'

/// Byte offset of the complex payload (8-byte aligned past the header).
inline size_t corr_payload_offset() {
    const size_t h = sizeof(CorrFrameHeader);
    return (h + 7) & ~size_t(7);
}

/// Complex floats in a full P_c payload for the given dimensions.
inline size_t corr_payload_complex(int n_prn, int nwin, int nd, int Mp) {
    return (size_t)n_prn * nwin * nd * Mp;
}

/// Byte offset of the raw window-0 block (immediately after the P_c payload).
inline size_t corr_raw_offset(int n_prn, int nwin, int nd, int Mp) {
    return corr_payload_offset() + corr_payload_complex(n_prn, nwin, nd, Mp) * 2 * sizeof(float);
}

/// Total frame bytes for the given dimensions (covering worst case: P_c + raw).
inline size_t corr_frame_size(int n_prn, int nwin, int nd, int Mp, int raw_hops) {
    return corr_raw_offset(n_prn, nwin, nd, Mp) + (size_t)raw_hops * 2 * sizeof(float);
}

/// Flat index of P_c[prn][win][d][q] within the complex payload.
inline size_t corr_index(int prn, int win, int d, int q, int nwin, int nd, int Mp) {
    return (((size_t)prn * nwin + win) * nd + d) * Mp + q;
}

} // namespace gnss

#endif // GNSS_CORR_FRAME_HPP
