/**
 * @file
 * @brief Coherently recombine per-subband GNSS despread products into the
 *        full-band complex amplitude.
 *  - GnssCoherentCombiner : public kotekan::Stage
 */

#ifndef GNSS_COHERENT_COMBINER_HPP
#define GNSS_COHERENT_COMBINER_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "restServer.hpp"      // for connectionInstance

#include <mutex>  // for mutex
#include <string> // for string
#include <vector> // for vector

/**
 * @class GnssCoherentCombiner
 * @brief The reassembly seam of the distributed-band pipeline.
 *
 * Each @ref GnssChannelizedTracker emits, per PRN per window, the un-normalized
 * coherent correlation @f$ G_m @f$ and replica energy @f$ E_m @f$ over its channel
 * slice. Because the despread is a sum over channels, both are additive across the
 * channel partition, so this stage forms the full-band matched-filter amplitude
 *   @f$ \hat A = (\sum_m G_m) / (\sum_m E_m) @f$
 * -- identical to despreading all covering channels in one place. That recovers
 * the full despread sensitivity from the per-subband (per-node, on CHORD) split.
 *
 * The trackers run lockstep on the same channelized windows, so the i-th frame of
 * every input is the same window and its records are in the same PRN order. The
 * output uses the standard GNSS record layout (matches @ref GnssChannelizedCorrelator)
 * so existing readers consume it unchanged.
 *
 * @conf n_prn  Int. Records (PRNs) per frame; default from in-frame size.
 *
 * @buffer in_bufs Per-subband tracker record streams (RECORD_FLOATS floats/PRN:
 *                 0=PRN 1=dop 2=cp 3=corr.re 4=corr.im 5=energy 6=n_chan 9,10=UTC).
 * @buffer out_buf Combined records (0=PRN 1=dop 2=cp 3=|A| 4=A.re 5=A.im 6=|A| 9,10=UTC).
 *
 * @author Keith Vanderlinde
 */
class GnssCoherentCombiner : public kotekan::Stage {
public:
    GnssCoherentCombiner(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);
    void main_thread() override;

    static constexpr int RECORD_FLOATS = 11;
    static constexpr int RECORD_UTC_SLOT = 9;

private:
    /// broker poll: latest full-band |A| (and seed) per PRN, for drop decisions.
    void get_status_callback(kotekan::connectionInstance& conn);

    std::vector<Buffer*> in_bufs;
    Buffer* out_buf;
    int _n_prn;

    // Latest combined record snapshot for REST status (full-band |A| per PRN).
    std::vector<int> _st_prn;
    std::vector<float> _st_amp, _st_dop, _st_cp;
    std::mutex _st_mtx;
};

#endif // GNSS_COHERENT_COMBINER_HPP
