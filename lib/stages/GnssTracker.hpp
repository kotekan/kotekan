/**
 * @file
 * @brief Lockstep GNSS tracking stage (the "locked" half of the search/track split).
 *  - GnssTracker : public kotekan::Stage
 */

#ifndef GNSS_TRACKER_HPP
#define GNSS_TRACKER_HPP

#include "Config.hpp"
#include "GnssCorrelatorCore.hpp"
#include "GnssTrackState.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssSignal.hpp"
#include "restServer.hpp"
#include "json.hpp"

#include <complex>
#include <memory>
#include <string>
#include <vector>

/**
 * @class GnssTracker
 * @brief Tracks an evolving set of locked PRNs, one record per PRN per window.
 *
 * The bounded-cost, lockstep half of the GNSS calibrator. It consumes every
 * input block (so carrier phase / code phase / FLL stay continuous) but only
 * correlates PRNs it currently owns, each over a narrow Doppler window around
 * its tracked value -- so cost is @c O(N_locked * few_bins), independent of the
 * stream and safe to run inline in the real-time path.
 *
 * New PRNs arrive as @ref gnss::Seed hand-offs from a @ref GnssAcquire stage via
 * a shared @ref gnss::TrackState (same @c state_group); a PRN whose SNR stays
 * below @c lock_snr for @c drop_after consecutive blocks is released back to the
 * search. The tracker publishes the set it owns so the search skips them.
 *
 * The per-PRN DSP (narrow Doppler correlate, parabolic peak refine, nav-bit
 * strip, FLL, continuous carrier phase) is exactly GpsReplicaCorrelator's
 * track path, here driven by the dynamic owned-set instead of a fixed mask.
 *
 * @par Buffers
 * @buffer in_buf  int16 real samples (Airspy / rawFileRead).
 * @buffer out_buf float32 [n_prn, RECORD_FLOATS] records (see gnssRecord.hpp).
 *
 * @conf signal, sample_rate, f_offset   As GpsReplicaCorrelator.
 * @conf prns               Array<int>. The PRN universe this tracker can hold.
 * @conf state_group        String. Shared name pairing this tracker with its
 *                          GnssAcquire (default "gnss").
 * @conf track_doppler_step Double (default 50). Fine Doppler bin, Hz.
 * @conf track_doppler_guard Int (default 3). Bins each side of the tracked
 *                          Doppler searched per block.
 * @conf lock_snr           Double (default 12). peak/mean to count as locked.
 * @conf drop_after         Int (default 200). Consecutive below-threshold
 *                          blocks before a PRN is released to the search.
 * @conf record_decimate    Int (default 1). Emit one record frame per this many
 *                          tracked blocks (tracking still runs every block).
 * @conf capture_utc0       Double (default 0). UTC of sample 0 (see GpsReplica).
 *
 * @author Keith Vanderlinde
 */
class GnssTracker : public kotekan::Stage {
public:
    GnssTracker(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~GnssTracker() override = default;

    void main_thread() override;

    /// REST: report the currently tracked PRNs and their state.
    void get_locked_callback(kotekan::connectionInstance& conn);

private:
    /// Assemble exactly @c Ns real samples into @c block across input frames.
    bool fill_block();
    /// Per-PRN nav-strip + FLL + continuous-phase update (GpsReplicaCorrelator's).
    void track_phase_update(int p, std::complex<float> raw, float snr, double doppler_bin,
                            double doppler_seed, double code_phase_chips, float& nav_sign,
                            float& phase_cont, float& doppler_out);

    Buffer* in_buf;
    Buffer* out_buf;
    int frame_in;
    int frame_out;

    std::unique_ptr<gnss::CorrelatorCore> core;
    gnss::SignalDescriptor _sig;
    double _sample_rate;
    double _f_offset;
    double _capture_utc0;
    std::vector<int> _prns;
    int _Ns;

    double _track_step;
    int _track_guard;
    double _lock_snr;
    int _drop_after;
    int _record_decimate;

    gnss::TrackState* _state;

    // --- input block assembly ---
    std::vector<float> block;
    int16_t* in_local;
    int in_pos;
    int in_samples_per_frame;

    // --- per-PRN-index tracking state ---
    std::vector<uint8_t> trk_active;         ///< owned (being tracked) this PRN?
    std::vector<uint8_t> trk_locked;         ///< above lock_snr last block?
    std::vector<int> trk_below;              ///< consecutive below-threshold blocks
    std::vector<float> trk_sign;             ///< cumulative nav-bit sign (+-1)
    std::vector<std::complex<float>> trk_prev;
    std::vector<double> trk_cont_phase;      ///< accumulated unwrapped phase, rad
    std::vector<double> trk_prev_wrapped;
    std::vector<double> trk_prev_doppler;
    std::vector<double> trk_prev_code_phase;
    std::vector<double> trk_doppler;         ///< FLL absolute Doppler estimate, Hz

    // --- latest per-PRN record fields (emitted on decimate boundaries) ---
    std::vector<float> last_amp;
    std::vector<float> last_snr;
    std::vector<float> last_code_phase;
    std::vector<std::complex<float>> last_cpx;
};

#endif // GNSS_TRACKER_HPP
