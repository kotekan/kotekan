/**
 * @file
 * @brief Single-dish voltage peel: reconstruct each seeded PRN's channelized replica and SUBTRACT
 *        it from the channelized voltage stream, emitting the residual (GNSS-cleaned) voltages.
 *  - GnssVoltagePeel : public kotekan::Stage
 */

#ifndef GNSS_VOLTAGE_PEEL_HPP
#define GNSS_VOLTAGE_PEEL_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssChannelizedReplica.hpp"
#include "restServer.hpp"

#include "json.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

/**
 * @class GnssVoltagePeel
 * @brief The voltage-peel half of the GNSS-mitigation pipeline (single dish, one signal).
 *
 * Mirrors @ref GnssChannelizedTracker's front end -- one dish's channelized voltages, the broker's
 * per-PRN seeds {code phase, Doppler, code-rate}, and the same @ref gnss::ChannelizedReplicaBank --
 * but instead of recording the despread it reconstructs the transmitted waveform and removes it:
 * for each active PRN, over a code-period window, it despreads the covering channels to get the
 * complex gain @f$a=\langle X,R\rangle/\langle R,R\rangle@f$ (the dish's own gain toward the sat,
 * per the single-dish/no-beamform architecture) and subtracts @f$a\,R@f$ from the voltages. PRNs are
 * peeled SUCCESSIVELY (each despread off the running residual), so Gold-code cross-talk between sats
 * does not floor the depth. The residual channelized stream is emitted for the correlator/imager (or,
 * on the prototype, a search/monitor that should no longer see the peeled sats).
 *
 * One signal per stage (the replica bank is per-signal); chain stages for multiple constellations
 * (GPS L1CA -> Galileo E1 -> ... each peeling its sats from the prior residual). The nav bit / overlay
 * is folded into @f$R@f$ by the replica bank (per-chip, exact across bit edges); a per-record complex
 * gain already absorbs a whole-record +-1 bit, so the bit is only needed where a window straddles an
 * edge (broker-broadcast or self-decoded -- future).
 *
 * @conf signal/sample_rate/f_offset/spectrum_length/channel_offset/n_channels/num_taps/pfb_window
 *                        -- as @ref GnssChannelizedTracker (define the front end + replica).
 * @conf prns             Array<Int>. PRNs to peel.
 * @conf doppler_hz/code_phase_chips  Array<Double>. Seeds (broker-updatable via set_seeds).
 * @conf pullin_chips     Double (default 0.5). Code pull-in half-window (chips) locked to peak |A|,
 *                        so a small residual code error still peels on-peak. SET 0 when smoothing the
 *                        gain (fll_gain>0/gain_alpha>0): a per-record re-pick jumps the carrier phase
 *                        by ~2*pi*f_IF*dtau and wrecks the phase track -- rely on the l-a cp_rate.
 * @conf pullin_step      Double (default 0.25). Pull-in grid step (chips).
 * @conf fll_gain         Double (default 0). Carrier FLL gain: track the residual carrier phase (so the
 *                        gain can be derotated before smoothing). 0 = feed-forward (use the broker dop).
 * @conf gain_alpha       Double (default 0). Complex-gain EMA rate on the DEROTATED, de-bitted gain --
 *                        the v2 depth knob. 0 = v1 (per-record gain, ~3 dB self-noise); ~0.05-0.1
 *                        averages the gain over ~10-40 records -> the gain-estimate noise (hence the
 *                        subtracted self-noise) drops ~sqrt(window), so the peel deepens toward the
 *                        prototype's ~35 dB. A decision-directed sign tracks the C/A nav bit / L5-NH
 *                        overlay / pilot (=+1) uniformly, so only the SLOW gain is averaged.
 * @conf fll_lock_amp     Double (default 0). Freeze the FLL + gain EMA when |A| < this (dropout).
 * @conf fll_max_gap_s    Double (default 0.005). Skip the FLL discriminator across a larger record gap.
 * @conf active_prns      Array<Int> (optional). Initial go/no-go mask (default all on).
 *
 * @buffer in_buf  This dish's channels, [hop][n_chan] cfloat32.
 * @buffer out_buf Residual channels, same layout, one code-period window per frame.
 *
 * Depends on libfftw3. @author Keith Vanderlinde
 */
class GnssVoltagePeel : public kotekan::Stage {
public:
    GnssVoltagePeel(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    void main_thread() override;

private:
    /// broker -> peel: set the consensus seeds + active set (same body as the tracker's).
    void set_seeds_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    /// REST: per-PRN pre-/post-peel |A| (the achieved peel depth), for the viewer/CLI.
    void get_status_callback(kotekan::connectionInstance& conn);

    Buffer* in_buf;
    Buffer* out_buf;

    double _sample_rate;
    double _f_offset;
    double _doppler_margin_hz;

    int _N;
    int _fft_len;
    int _chan_offset;
    int _n_chan;
    int _hops_per_record;

    double _pullin_chips;
    double _pullin_step;
    double _fll_gain;      ///< carrier FLL gain (0 = feed-forward)
    double _gain_alpha;    ///< derotated-gain EMA rate (0 = per-record gain, v1)
    double _fll_lock_amp;  ///< freeze FLL + gain EMA below this |A|
    double _fll_max_gap;   ///< skip the FLL discriminator across a record gap larger than this (s)

    std::vector<int> _prns;
    std::vector<double> _doppler;
    std::vector<double> _code_phase;
    std::vector<double> _code_phase_rate; ///< chips/hop, from the broker (folds in the l-a clock bias)
    std::vector<double> _doppler_rate;    ///< Hz/s, from the broker: 2nd-order carrier feed-forward
                                          ///< for the v2 smooth-gain (removes the zenith Doppler ramp
                                          ///< that otherwise decoheres the EMA). See main_thread.
    std::vector<long long> _ref_hop;
    std::vector<uint8_t> _active;
    std::mutex _seed_mtx;

    // Diagnostics (peel depth), per PRN: EMA of the pre- and post-subtraction despread |A|.
    std::vector<double> _pre_amp;
    std::vector<double> _post_amp;
    std::mutex _diag_mtx;

    std::unique_ptr<gnss::ChannelizedReplicaBank> _replica;
};

#endif // GNSS_VOLTAGE_PEEL_HPP
