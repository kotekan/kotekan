/**
 * @file
 * @brief Per-subband channelized GNSS tracker: despread a seeded PRN set over one
 *        subband's channel slice and emit raw correlation + replica energy.
 *  - GnssChannelizedTracker : public kotekan::Stage
 */

#ifndef GNSS_CHANNELIZED_TRACKER_HPP
#define GNSS_CHANNELIZED_TRACKER_HPP

#include "Config.hpp"                 // for Config
#include "Stage.hpp"                  // for Stage
#include "buffer.hpp"                 // for Buffer
#include "bufferContainer.hpp"        // for bufferContainer
#include "gnssChannelizedReplica.hpp" // for ChannelizedReplicaBank
#include "restServer.hpp"             // for connectionInstance

#include "json.hpp" // for json
#include <memory>   // for unique_ptr
#include <mutex>    // for mutex
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class GnssChannelizedTracker
 * @brief The track half of the distributed-band split, per subband.
 *
 * Consumes one subband's channel slice from @ref GnssSubbandSplit (channels
 * @c channel_offset .. @c channel_offset + n_chan, where n_chan follows from the
 * input frame size) and, for each active PRN at its *seeded* code phase + Doppler,
 * despreads the carrier's covering channels that fall within this subband. It
 * emits the **un-normalized** coherent correlation @f$ G_m = \sum_{c\in S_m}\sum_n
 * X_c \overline{R_c} @f$ and the replica energy @f$ E_m = \sum_{c\in S_m}\sum_n
 * |R_c|^2 @f$ -- not the normalized amplitude. Both are additive across subbands,
 * so @ref GnssCoherentCombiner forms the full-band estimate @f$ (\sum_m G_m) /
 * (\sum_m E_m) @f$, recovering the full despread sensitivity from the channel
 * partition (see @ref gnssChannelizedDespread.hpp).
 *
 * Acquisition is decoupled: seeds arrive from config (validation) or, later, from
 * the broker / @ref GnssTrackState. The replica is generated for all @c
 * spectrum_length global channels at the subband's true carrier (so the band
 * geometry is correct), then only the owned covering channels are despread.
 *
 * @conf signal           String (default "GPS_L1CA"). SignalDescriptor name.
 * @conf sample_rate      Double. Full (pre-channelization) sample rate, Hz.
 * @conf f_offset         Double (default 0). Carrier offset from band centre, Hz.
 * @conf spectrum_length  Int. Global channels per hop (the F-engine N).
 * @conf channel_offset   Int (default 0). Global index of this subband's channel 0.
 * @conf num_taps         Int. PFB taps; must match the F-engine.
 * @conf pfb_window       String (default "hamming"). Must match the F-engine.
 * @conf prns             Array<Int>. PRNs to track.
 * @conf doppler_hz       Array<Double>. Seeded Doppler per PRN (scalar broadcasts).
 * @conf code_phase_chips Array<Double>. Seeded code phase per PRN (scalar broadcasts).
 * @conf doppler_margin_hz Double (default 5000). Extra band for covering-channel select.
 * @conf hops_per_record  Int (default: one code period). Coherent window, hops.
 * @conf pullin_chips     Double (default 0). Code pull-in half-window (chips): despread
 *                        over seed +/- this and lock to the peak |A|, so a stale/drifting
 *                        seed still locks (DLL-style). 0 = single despread at the seed.
 * @conf pullin_step      Double (default 0.5). Pull-in grid step (chips).
 * @conf active_prns      Array<Int> (optional). Initial go/no-go mask (default all on).
 * @conf capture_utc0     Double (default 0). UTC of sample 0; 0 = wall-clock at emit.
 *
 * @buffer in_buf  This subband's channels, [hop][n_chan] cfloat32.
 * @buffer out_buf One record per PRN per window (RECORD_FLOATS floats each).
 *
 * Depends on libfftw3. @author Keith Vanderlinde
 */
class GnssChannelizedTracker : public kotekan::Stage {
public:
    GnssChannelizedTracker(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container);
    void main_thread() override;

    // Record layout (floats): 0=PRN 1=Doppler 2=code_phase 3=corr.re 4=corr.im
    // 5=replica_energy 6=n_chan_used 7,8=spare 9,10=UTC (double in slots 9..10).
    static constexpr int RECORD_FLOATS = 11;
    static constexpr int RECORD_UTC_SLOT = 9;

private:
    /// broker -> tracker: set the consensus seeds + active set. Body is a JSON array
    /// of {prn, doppler_hz, code_phase_chips}; PRNs absent from the list go inactive.
    void set_seeds_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    Buffer* in_buf;
    Buffer* out_buf;

    double _sample_rate;
    double _capture_utc0;
    double _doppler_margin_hz;

    int _N;           ///< global channels per hop (spectrum_length)
    int _fft_len;     ///< real-FFT length = 2*_N
    int _chan_offset; ///< global index of this subband's local channel 0
    int _n_chan;      ///< channels owned by this subband
    int _hops_per_record;

    double _pullin_chips; ///< code pull-in half-window (chips); 0 = despread at the seed only
    double _pullin_step;  ///< code pull-in grid step (chips)

    std::vector<int> _prns;
    std::vector<double> _doppler;
    std::vector<double> _code_phase;
    std::vector<uint8_t> _active;
    std::mutex _seed_mtx; ///< guards _doppler/_code_phase/_active (REST set_seeds vs main loop)

    std::unique_ptr<gnss::ChannelizedReplicaBank> _replica;
};

#endif // GNSS_CHANNELIZED_TRACKER_HPP
