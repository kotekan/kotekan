/**
 * @file
 * @brief Central cross-channel aggregation of the distributed GNSS search:
 *        gather per-channel P_c, combine, peak, publish detections.
 *  - GnssSearchAggregator : public kotekan::Stage
 */

#ifndef GNSS_SEARCH_AGGREGATOR_HPP
#define GNSS_SEARCH_AGGREGATOR_HPP

#include "Config.hpp"                 // for Config
#include "Stage.hpp"                  // for Stage
#include "buffer.hpp"                 // for Buffer
#include "bufferContainer.hpp"        // for bufferContainer
#include "gnssChannelizedReplica.hpp" // for ChannelizedReplicaBank
#include "restServer.hpp"             // for connectionInstance

#include <memory> // for unique_ptr
#include <mutex>  // for mutex
#include <string> // for string
#include <vector> // for vector

/**
 * @class GnssSearchAggregator
 * @brief Combine per-channel coarse correlations from many
 *        @ref GnssChannelCorrelator into a full-band acquisition surface.
 *
 * The central half of the distributed search. It gathers, in lock-step, one
 * P_c frame per snapshot from every channel correlator (the CHORD comb, fanned
 * out into independent sub-band streams). For each PRN it forms the cross-channel
 * fine-lag DFT and incoherent |D|^2 surface (@ref gnss::aggregate_accumulate)
 * over the covering channels and all windows, then reduces to the peak
 * (@ref gnss::channelized_peak). This is the only place the channels are
 * recombined -- a tiny transform of correlation values, never of the signal --
 * so the full-band code-phase resolution is recovered from sub-band-local work.
 *
 * Because it runs the same cores as @ref gnss::channelized_acquire over the same
 * channels, the surface is bit-identical to a monolithic full-band search with
 * matching parameters. Detections (PRN -> Doppler, code phase, SNR) are published
 * over REST (@c GET unique_name/get_detections), matching
 * @ref GnssChannelizedSearch's seeding convention so the broker can poll and seed
 * the per-channel trackers directly.
 *
 * @conf signal/sample_rate/f_offset/spectrum_length/num_taps/pfb_window  As the F-engine.
 * @conf prns             Array<Int>. PRN universe (must match the correlators).
 * @conf doppler_min/max/step  Double. Acquisition Doppler grid (must match).
 * @conf acquire_snr      Double (default 12). Surface SNR to declare a detection.
 * @conf hold_snapshots   Int (default 5). Keep a detection valid this many misses.
 * @conf doppler_margin_hz Double (default 5000). Covering-channel selection band.
 *
 * @buffer in_bufs  Per-channel P_c streams (@ref gnss::CorrFrameHeader + payload).
 *
 * @author Keith Vanderlinde
 */
class GnssSearchAggregator : public kotekan::Stage {
public:
    GnssSearchAggregator(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);
    ~GnssSearchAggregator() override = default;
    void main_thread() override;

private:
    void get_detections_callback(kotekan::connectionInstance& conn);

    struct Detection {
        double doppler_hz = 0.0;
        double code_phase_chips = 0.0;
        float snr = 0.0f;
        bool valid = false;
        int misses = 0;
    };

    std::vector<Buffer*> in_bufs;

    int _fft_len;
    int _hold_snapshots;
    double _sample_rate;
    double _acquire_snr;
    std::vector<int> _prns;
    std::vector<double> _doppler_grid;

    std::unique_ptr<gnss::ChannelizedReplicaBank> _replica;

    std::vector<Detection> _detections;
    std::mutex _det_mtx;
};

#endif // GNSS_SEARCH_AGGREGATOR_HPP
