/**
 * @file
 * @brief Async per-subband channelized GNSS search (the search half of the
 *        distributed split), reporting detections over REST.
 *  - GnssChannelizedSearch : public kotekan::Stage
 */

#ifndef GNSS_CHANNELIZED_SEARCH_HPP
#define GNSS_CHANNELIZED_SEARCH_HPP

#include "Config.hpp"                 // for Config
#include "Stage.hpp"                  // for Stage
#include "buffer.hpp"                 // for Buffer
#include "bufferContainer.hpp"        // for bufferContainer
#include "gnssChannelizedAcquire.hpp" // for AcquireWorkspace
#include "gnssChannelizedReplica.hpp" // for ChannelizedReplicaBank
#include "restServer.hpp"             // for connectionInstance

#include <complex>            // for complex
#include <condition_variable> // for condition_variable
#include <memory>             // for unique_ptr
#include <mutex>              // for mutex
#include <string>             // for string
#include <thread>             // for thread
#include <vector>             // for vector

/**
 * @class GnssChannelizedSearch
 * @brief Non-backpressuring channelized acquisition over one subband's channels.
 *
 * Mirrors @ref GnssAcquire's design in the channelized domain: the consumer loop
 * copies a snapshot of channelized hops and immediately releases every frame, so
 * the producer never waits; a worker thread runs the heavy parallel-code-phase
 * search (the incoherent-accumulation @ref gnssChannelizedAcquire, FFT coarse
 * correlation) over the snapshot on this subband's owned channels. Because each
 * subband holds only a slice of the carrier's code power it is less sensitive than
 * the full band -- but it only has to find a sat in *some* subband; the broker then
 * seeds every subband's @ref GnssChannelizedTracker at one consensus (code phase,
 * Doppler) so the per-subband despreads recombine coherently to full sensitivity.
 *
 * Detections (PRN -> Doppler, code phase, SNR) are published over REST
 * (@c GET unique_name/get_detections) for the broker to poll. Code phase + Doppler
 * follow the tracker's seeding convention (r2c Doppler-sign flip + sub-hop refine),
 * and code phase is grid-consistent with the tracker (both on the hop*2N window
 * grid), so a reported seed is directly usable -- modulo slow code-Doppler drift,
 * which periodic re-seeding by the broker absorbs.
 *
 * @conf signal/sample_rate/f_offset/spectrum_length/num_taps/pfb_window  As the F-engine.
 * @conf channel_offset   Int (default 0). Global index of this subband's channel 0. Describes
 *                        a CONTIGUOUS subband only.
 * @conf channel_ids      Array<Int> (optional). Explicit global index per local channel, for a
 *                        SPARSE subband -- CHORD's corner-turn hands a node every 16th bin, so
 *                        its covering set for one carrier is a stride-16 comb, not a run.
 *                        Overrides channel_offset. Empty = contiguous, unchanged.
 * @conf n_channels       Int. Channels owned by this subband.
 * @conf prns             Array<Int>. PRN universe to search.
 * @conf doppler_min/max/step  Double. Acquisition Doppler grid, Hz.
 * @conf acquire_snr      Double (default 12). Surface SNR to declare a detection.
 * @conf acquire_windows  Int (default 64). |D|^2 windows to integrate per snapshot.
 * @conf doppler_margin_hz Double (default 5000). Covering-channel selection band.
 *
 * @par Almanac-narrowed Doppler (POST @c unique_name/set_doppler_hints)
 * The orbit fixes each sat's Doppler to ~tens of Hz (geometry + a common clock-frequency
 * bias); only that prediction is unknown a priori. The broker POSTs per-PRN hints
 * @c [{prn, doppler_hz, margin_hz}] and the search then scans only @c doppler_hz +- margin_hz
 * (at @c doppler_step) for that PRN instead of the blind @c doppler_min..max grid -- far cheaper
 * and lower false-alarm, and it integrates at the right cell so weak sats clear. A PRN with no
 * hint (broker absent / unpredicted) falls back to the full blind grid, so behaviour is
 * unchanged without the broker. (On a disciplined-clock receiver the prediction is exact and
 * the search collapses to a code-phase pin.)
 *
 * @buffer in_buf This subband's channels, [hop][n_channels] cfloat32.
 *
 * @author Keith Vanderlinde
 */
class GnssChannelizedSearch : public kotekan::Stage {
public:
    GnssChannelizedSearch(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);
    ~GnssChannelizedSearch() override;
    void main_thread() override;

private:
    void search_worker();
    void search_snapshot();
    void get_detections_callback(kotekan::connectionInstance& conn);
    void set_doppler_hints_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);
    void set_nh_hint_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);

    struct Detection {
        double doppler_hz = 0.0;
        double code_phase_chips = 0.0;
        long long ref_hop = 0; ///< snapshot reference hop (sample_seq/fft_len) cp0 is anchored to
        float snr = 0.0f;
        bool valid = false;
        int misses = 0; ///< consecutive non-detecting snapshots since last valid hit
        /// Secondary-code (NH) alignment this detection was found at -- the @c nh_phase the
        /// replica generator was given, i.e. overlay chip for absolute primary-period k is
        /// secondary[(k + nh) mod len]. -1 when the signal has no secondary code.
        ///
        /// The search SCANS all alignments and therefore MEASURES this, per satellite, every
        /// pass. Reporting it lets a consumer skip reconstructing it from absolute time (the
        /// broker's --cl-assist), which needs the wall clock and the modelled range good to well
        /// under half a primary period -- 0.5 ms for L5 NH20 -- and silently picks the wrong
        /// period when they are not. Measured beats computed, and per-satellite beats a fitted
        /// common constant.
        int nh = -1;
        /// Code phase in the OVERLAID code's own length (n_nh * code_length), or -1 when the
        /// signal has no secondary code. This is what a tracker despreading the long code needs;
        /// @c code_phase_chips is reduced mod the PRIMARY length and cannot carry the period.
        double code_phase_long_chips = -1.0;
        /// PHYSICAL code phase at @c ref_hop, in the overlaid code's length; -1 if none.
        ///
        /// This, not @c code_phase_chips, is what a consumer should transport. cp0 back-
        /// references the phase to sample 0 through a Doppler-scaled rate, ~2e15 samples away,
        /// which multiplies the reported Doppler's error by 5903 chips/Hz = 0.58 PERIODS per Hz
        /// -- and the acquire's Doppler is only good to ~1.5 Hz, so the overlay period that
        /// comes out is a coin flip. Referenced to the search's own fixed replica anchor
        /// instead, the same lever is 1e-4 chips/Hz: 4e7 times smaller, because the anchor is a
        /// small fixed index rather than the absolute stream position.
        double code_phase_at_ref_chips = -1.0;
    };

    /// Global spectrum index of local channel @c lc, and its inverse (-1 = not ours).
    /// Identity+offset for a contiguous subband; a lookup when @c channel_ids is given.
    int global_of_local(int lc) const;
    int local_of_global(int c) const;

    Buffer* in_buf;

    int _N;
    int _fft_len;
    int _chan_offset;
    int _n_chan;
    std::vector<int> _chan_ids; ///< global index per local channel; empty = contiguous
    int _hops_per_record; ///< hops per integration window
    int _acquire_windows; ///< windows accumulated per snapshot
    int _hold_snapshots;  ///< keep a detection valid this many non-detecting snapshots
    double _sample_rate;
    double _doppler_margin_hz;
    double _acquire_snr;
    double _doppler_step; ///< grid step, Hz (used to build a PRN's narrow almanac grid)
    std::vector<int> _prns;
    std::vector<double> _doppler_grid; ///< blind fallback grid (doppler_min..max)

    /// Per-PRN almanac Doppler hint from the broker (POST set_doppler_hints): when valid, the
    /// search scans only doppler +- margin for that PRN instead of the blind grid. `t_recv` is a
    /// steady-clock stamp (s) at receipt -> a hint the broker stops refreshing (the sat set) EXPIRES
    /// after _hint_ttl_s, so with _require_hint the PRN drops out of the active scan set.
    struct DopHint {
        bool valid = false;
        double doppler = 0.0;
        double margin = 0.0;
        double t_recv = 0.0;
    };
    std::vector<DopHint> _dop_hints; ///< parallel to _prns (live, updated by REST)
    /// The hint table AS OF the snapshot, copied under _hint_mtx when the snapshot finalizes.
    /// The worker searches THESE, not the live table: a pass can run for minutes, during which
    /// the broker keeps refreshing hints to predict the CURRENT sky -- but the data is frozen
    /// at snapshot time. High-elevation sats (the best targets) have the highest Doppler rate
    /// (~0.65 Hz/s), so a hint read 4-5 minutes into a pass is off by ~150-200 Hz for data
    /// captured at pass start: outside the narrowed window entirely, a silent per-PRN miss
    /// that gets WORSE the later in the pass the PRN is processed. Epoch-matching the hints to
    /// the data removes it; residual = broker interval (10 s) x rate ~ 7 Hz.
    std::vector<DopHint> _snap_hints;
    double _snap_taken_s = 0.0; ///< steady_s() when the snapshot finalized (for hint TTL)
    /// Per-PRN SECONDARY-CODE ALIGNMENT hint (POST set_nh_hint), the Doppler hint's twin and by
    /// far the bigger saving: the acquire builds a FULL surface per alignment, so a 20-chip
    /// overlay costs 20 surfaces per PRN and is ~92% of a pass (measured 2026-08-04: acquire
    /// 40.00 s vs refine 3.62 s at live parameters).
    ///
    /// It is redundant work. The alignment advances exactly one index per primary code period,
    /// and a period is an exact hop count, so
    ///     nh(H) = (nh0 + (H - H0) / period_hops) mod n_nh
    /// is pure F-engine counter arithmetic. NO absolute clock, no range model, no utc0_sample0
    /// -- deliberately unlike the combiner's --cl-hint, whose GPS-time route carries a
    /// bootstrap dependency and an anchor-latency failure mode. The hint only ever refines
    /// something THIS stage already measured and reported, so there is nothing to bootstrap:
    /// no hint scans all n_nh (the old behaviour), the first detection establishes it, and a
    /// stale hint expires back to the full scan.
    ///
    /// Accuracy comes from the revisit being short. An entirely unmodelled 4 kHz Doppler slips
    /// the period count by 0.041 periods over 12 s -- but 4.3 periods over the 1276 s revisit
    /// this fleet had before 2026-08-04, where the hint would have been worse than useless.
    struct NhHint {
        bool valid = false;
        int nh = 0;             ///< alignment index measured at ref_hop
        long long ref_hop = 0;  ///< absolute hop it was measured at
        double t_recv = 0.0;
    };
    std::vector<NhHint> _nh_hints;      ///< live, updated by REST
    std::vector<NhHint> _nh_snap_hints; ///< frozen with the snapshot, like _snap_hints
    /// Alignments scanned either side of the predicted one. 1 (three of twenty) keeps ~6.7x of
    /// the saving while tolerating an off-by-one hint; 0 takes the full 20x and trusts it.
    int _nh_hint_span = 1;
    std::mutex _hint_mtx;
    /// require_hint: scan ONLY PRNs with a fresh broker hint (visible sats), SKIP the rest (no blind
    /// grid) -> cost tracks the visible count, and the set follows the sky (mid-run PRN swap) when
    /// `prns` lists the whole constellation. hint_ttl_s: a hint older than this counts as absent.
    bool _require_hint = false;
    int _acquire_threads = 1; ///< parallelism of the aggregate (see channelized_accumulate)
    int _acquire_fine_step = 1; ///< fine-lag decimation of the surface (see AcquisitionSurface)
    int _prns_per_pass = 0;     ///< eligible PRNs searched per snapshot (0 = all, as before)
    int _prn_cursor = 0;        ///< round-robin start into the ELIGIBLE list, advanced per pass
    double _hint_dop_sign = -1.0; ///< physical->internal Doppler map (see the hint window note)
    long _last_surface_cells = 0; ///< cells in the last surface (for the noise-ceiling log)
    long _empty_hint_passes = 0;  ///< require_hint passes that scanned nothing (rate-limited WARN)
    int _last_cov_n = -1;         ///< last logged covering set (log once per set, not snapshot)
    int _last_cov_first = -1, _last_cov_last = -1;
    double _hint_ttl_s = 8.0;

    std::unique_ptr<gnss::ChannelizedReplicaBank> _replica;
    gnss::AcquireWorkspace _acq_ws;

    /// Fine code-phase refine (post-detection), in SAMPLES: scan +-_refine_span at _refine_step
    /// with an exact despread. See the note at the refine loop for why the step is DERIVED
    /// rather than 1 -- on a wide bank a 1-sample step is thousands of times finer than the
    /// covering set can resolve, and costs one replica build each.
    int _refine_span = 0;
    /// Refine integration length in hops; 0 = the full record. The refine only LOCALISES a peak
    /// that already cleared detection, and costs one full n_chan x n_hops replica build per
    /// trial phase -- 82% of all search compute at CHORD dimensions. The airspy prototype does
    /// 1/33 of that work because its record IS one code period, where CHORD's spans 16.
    int _refine_hops = 0;
    int _refine_step = 0;

    /// Precomputed repl0 (code phase 0, Doppler 0), BANDED to the covering channels: [prn]
    /// [local channel][hop], with only the covering rows populated (the rest stay empty --
    /// channelized_accumulate reads none of them). repl0 depends only on (PRN, covering set,
    /// Mp), none of which change between snapshots, so it is built once and reused. Worker
    /// thread only, like _acq_ws.
    /// Indexed [nh alignment][prn][local channel][hop]; one alignment (index 0) unless
    /// @c nh_search, in which case one per secondary-code alignment. See the note at the
    /// build site for why a dataless pilot needs this.
    std::vector<std::vector<std::vector<std::vector<std::complex<float>>>>> _repl0;
    /// EXACT nh FACTORISATION. The twenty alignments of a PRN are the same primary code with a
    /// different +-1 overlay chip on each code period, so building twenty is ~20x redundant
    /// (1223 s and 1696 MB at CHORD dimensions -- twenty minutes before the first pass, on
    /// every restart).
    ///
    /// The generator applies the overlay PER CHIP (gnssChannelizedReplica.cpp:
    /// cv *= overlay_sign(floor(chip/L), nh_phase)), so a hop straddling a period boundary
    /// already gets the correct mix of two signs -- 195.3125 hops per period means 16 of every
    /// 3125 hops straddle, and a per-HOP sign would quietly corrupt them. That per-chip
    /// resolution is what makes an exact decomposition possible. Writing each hop as the part
    /// belonging to the earlier period plus the part belonging to the later one,
    ///     R_nh[m] = s(k0[m] + nh) * head[m]  +  s(k0[m] + 1 + nh) * tail[m]
    /// and head/tail follow from TWO generator calls via the nav_bit hook (which takes the
    /// absolute CHIP index, so it can impose any per-period pattern):
    ///     A: nav_bit = +1            -> A[m] = head[m] + tail[m]
    ///     B: nav_bit = (-1)^period   -> B[m] = (-1)^k0[m] * (head[m] - tail[m])
    /// Two calls per PRN instead of twenty, exact to the same precision as before. Measured
    /// 2026-08-04: 1223 s -> 124 s, 1696 MB -> 170 MB, pass 4.4 s -> 1.67 s (the smaller bank
    /// also stops thrashing L3), and PRN 4 still detected at snr 11652 -- a sign error anywhere
    /// in the decomposition would have collapsed that to noise.
    std::vector<std::vector<std::vector<std::complex<float>>>> _repl_head; ///< [prn][chan][hop]
    std::vector<std::vector<std::vector<std::complex<float>>>> _repl_tail;
    std::vector<long long> _repl_k0; ///< [hop] period index of that hop's first chip
    std::vector<std::vector<std::complex<float>>> _repl_scratch; ///< materialised R_nh, reused
    std::vector<float> _sgn0, _sgn1; ///< per-hop overlay sign pair for the current alignment
    std::vector<int> _repl0_cover; ///< the covering set _repl0 was built for (rebuild key)
    /// Search every secondary-code (Neuman-Hofman) alignment and keep the best peak. Costs
    /// secondary_length() acquires per PRN and buys back the coherent loss that an
    /// overlay-blind replica suffers over a multi-period window -- 12.7 dB rms on L5 Q5,
    /// with exact nulls. Off by default: a signal without a secondary code, or a window
    /// inside one code period, gains nothing.
    bool _nh_search = false;
    int _n_nh = 1; ///< alignments actually built (1 when not searching)

    std::vector<Detection> _detections; ///< per-PRN latest detection
    std::mutex _det_mtx;

    // --- snapshot hand-off (consumer loop -> worker) ---
    std::vector<std::complex<float>> _snapshot; ///< acquire_windows*hpr hops, n_chan each
    size_t _snap_hops;                          ///< hops the snapshot holds
    long long _snap_start_hop;                  ///< absolute hop index of snapshot hop 0
                                                ///< (so reported code phase is absolute-referenced)
    std::thread _worker;
    std::mutex _m;
    std::condition_variable _cv;
    bool _snap_ready;
    bool _worker_busy;
    bool _worker_stop;
};

#endif // GNSS_CHANNELIZED_SEARCH_HPP
