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

#include <complex> // for complex
#include <mutex>   // for mutex
#include <string>  // for string
#include <vector>  // for vector

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
 * Optional temporal integration (@c integration_length records): the per-record
 * full-band amplitude is accumulated K ways before emission, giving both the
 * robust **incoherent** amplitude @f$ \sqrt{\langle|A|^2\rangle} @f$ (slot 3,
 * ~sqrt(K) SNR, no phase needed) and the **coherent** mean @f$ \langle A\rangle @f$
 * (slots 4/5/6, up to K SNR and unbiased, but only valid once the Doppler seed is
 * fine enough that the carrier phase is stable across the K records -- with a
 * coarse Doppler grid the coherent mean decorrelates and only slot 3 is usable).
 *
 * Two integration cadences (@c integration_mode):
 *  - @c block (default): accumulate K records, emit once, reset. One output per K
 *    records; the historical behaviour, byte-identical when the key is absent.
 *  - @c rolling: an exponential moving average with time constant K records
 *    (alpha = 1/K), updated every record and emitted every @c output_every records
 *    WITHOUT reset. Bias-corrected (divide by 1-(1-alpha)^n) so it reads a true
 *    running mean from the first record. Incoherent integration has no nav-bit cap
 *    and only needs the tracker to hold the code/Doppler bin, so a long rolling K
 *    (e.g. minutes of records) lets a weak sat climb out continuously -- you watch
 *    slot 3 (and the nav-wiped slot 8) grow instead of waiting K records per sample.
 *    The coherent slots 4-6 carry the same nav-bit limitation as in block mode.
 *
 * Optional **nav-bit wipe** (@c navwipe_bit_records > 0): coherent integration past
 * the 20 ms GPS data bit. Each record (one code period) lies wholly within one data
 * bit, so @f$ A_{rec} = d \cdot (\text{clean despread}) @f$ -- a constant +-1 sign. Over
 * the @c integration_length window the per-record A is buffered, grouped into
 * @c navwipe_bit_records-record bit epochs (alignment found by maximising per-bit
 * coherent power), the +-1 bit estimated per epoch by squaring (the global sign cancels
 * in |.|), wiped, and coherently summed -- giving a deep |A| that keeps growing past
 * 20 ms (slot 8), where the plain coherent mean (slot 6) decorrelates at the nav bit.
 *
 * @conf n_prn  Int. Records (PRNs) per frame; default from in-frame size.
 * @conf integration_length Int (default 1). block: tracker records accumulated per output.
 *       rolling: EMA time constant in records (the effective integration depth).
 * @conf integration_mode String (default "block"). "block" or "rolling" (see above).
 * @conf output_every Int (rolling only; default max(1, integration_length/10)). Records
 *       between rolling emits -- decouples the EMA update rate (every record) from the
 *       output/record cadence.
 * @conf navwipe_bit_records Int (default 0=off). Records per nav bit (~20 ms / record);
 *       e.g. 20 at 5 MSPS / 1 ms records. Needs integration_length >> this. In rolling mode
 *       the wipe runs over a sliding window of the last integration_length records.
 *
 * @buffer in_bufs Per-subband tracker record streams (RECORD_FLOATS floats/PRN:
 *                 0=PRN 1=dop 2=cp 3=corr.re 4=corr.im 5=energy 6=n_chan 9,10=UTC).
 * @buffer out_buf Combined records (0=PRN 1=dop 2=cp 3=|A|_incoh 4=<A>.re 5=<A>.im
 *                 6=|<A>|_coh 7=n_chan 8=|A|_navwipe 9,10=UTC).
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

    /// Nav-bit-wiped deep coherent amplitude from a window of per-record (A, capture-UTC):
    /// bin records into nav-bit epochs by their ABSOLUTE code-period index (from UTC, so
    /// valve drops just leave gaps, not misalignment), bit-sync, per-epoch +-1 by squaring,
    /// wipe, coherent-sum / N. 0 if too short. Needs capture-time UTC (capture_utc0 > 0).
    double navwipe_amplitude(const std::vector<std::complex<double>>& a,
                             const std::vector<double>& utc) const;

    std::vector<Buffer*> in_bufs;
    Buffer* out_buf;
    int _n_prn;
    int _integration_length; ///< block: records/output; rolling: EMA time constant (records)
    bool _rolling;           ///< rolling EMA integration vs block-and-reset
    int _emit_every;         ///< rolling: records between emits (output cadence)
    int _navwipe_bit_records; ///< records per nav bit (0 = no wipe)
    std::vector<std::vector<std::complex<double>>> _navbuf; ///< per-PRN per-record A over the window
    std::vector<std::vector<double>> _navutc;              ///< per-PRN per-record capture UTC

    // Latest combined record snapshot for REST status (full-band |A| per PRN).
    std::vector<int> _st_prn;
    std::vector<float> _st_amp, _st_coh, _st_deep, _st_dop, _st_cp;
    std::mutex _st_mtx;
};

#endif // GNSS_COHERENT_COMBINER_HPP
