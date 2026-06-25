/**
 * @file
 * @brief Per-channel coarse correlation -- the distributable half of the
 *        channelized GNSS search.
 *  - GnssChannelCorrelator : public kotekan::Stage
 */

#ifndef GNSS_CHANNEL_CORRELATOR_HPP
#define GNSS_CHANNEL_CORRELATOR_HPP

#include "Config.hpp"                 // for Config
#include "Stage.hpp"                  // for Stage
#include "buffer.hpp"                 // for Buffer
#include "bufferContainer.hpp"        // for bufferContainer
#include "gnssChannelizedAcquire.hpp" // for AcquireWorkspace
#include "gnssChannelizedReplica.hpp" // for ChannelizedReplicaBank

#include <complex> // for complex
#include <memory>  // for unique_ptr
#include <string>  // for string
#include <vector>  // for vector

/**
 * @class GnssChannelCorrelator
 * @brief Run the per-channel coarse correlation P_c(d,q) for one sub-band's
 *        channel(s) and ship it to the central @ref GnssSearchAggregator.
 *
 * This is the X-engine-local half of the distributed search: it owns a single
 * 195 kHz channel (the CHORD comb unit), accumulates a snapshot of @c nwin
 * integration windows, and for each PRN/window computes the Fourier-domain
 * coarse correlation @ref gnss::channel_correlate -- P_c[prn][win][d][q]. It does
 * NOT combine across channels; the cross-channel fine-lag DFT and |D|^2 peak are
 * done centrally by @ref GnssSearchAggregator, which gathers the P_c from every
 * channel. Because the cross-channel combine is non-linear (|sum_c|^2), each
 * window's P_c must be shipped whole (see @ref GnssCorrFrame.hpp).
 *
 * Unlike @ref GnssChannelizedSearch (async, drop-on-busy), this stage is
 * loss-less and lock-step: it back-pressures its input and emits exactly one
 * frame per snapshot, so the aggregator's gather stays channel-aligned. Channels
 * that do not cover the carrier emit a cheap header-only frame (covers=0).
 *
 * @conf signal/sample_rate/f_offset/spectrum_length/num_taps/pfb_window  As the F-engine.
 * @conf channel_offset   Int (default 0). Global frequency index of this channel.
 * @conf prns             Array<Int>. PRN universe to correlate.
 * @conf doppler_min/max/step  Double. Acquisition Doppler grid, Hz.
 * @conf acquire_windows  Int (default 4). Integration windows per snapshot.
 * @conf hops_per_record  Int (default = replica period). Hops per window.
 * @conf doppler_margin_hz Double (default 5000). Covering-channel selection band.
 *
 * @buffer in_buf   This channel's hops, [hop][1] cfloat32.
 * @buffer out_buf  P_c surfaces, @ref gnss::CorrFrameHeader + payload.
 *
 * @author Keith Vanderlinde
 */
class GnssChannelCorrelator : public kotekan::Stage {
public:
    GnssChannelCorrelator(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);
    ~GnssChannelCorrelator() override = default;
    void main_thread() override;

private:
    /// Fill `out` (one frame) with this channel's P_c for the snapshot in `_snapshot`.
    void correlate_snapshot(char* out, long long snap_start_hop);

    Buffer* in_buf;
    Buffer* out_buf;

    int _N;
    int _fft_len;
    int _chan_freq;       ///< this channel's global frequency index
    int _hops_per_record; ///< hops per integration window
    int _acquire_windows; ///< windows per snapshot
    double _sample_rate;
    double _doppler_margin_hz;
    bool _covers;         ///< this channel covers the carrier band
    std::vector<int> _prns;
    std::vector<double> _doppler_grid;

    std::unique_ptr<gnss::ChannelizedReplicaBank> _replica;
    gnss::AcquireWorkspace _acq_ws;

    std::vector<std::complex<float>> _snapshot; ///< nwin*hpr hops (one channel)
    size_t _snap_hops;
};

#endif // GNSS_CHANNEL_CORRELATOR_HPP
