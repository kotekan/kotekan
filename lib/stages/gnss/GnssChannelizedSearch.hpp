/**
 * @file
 * @brief Asynchronous channelized GNSS acquisition over one subband, reporting detections
 *        over REST.
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
#include "gnssSeedTransport.hpp"      // for gnss::CudaRefineGroup
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
 * The consumer loop copies a snapshot of channelized hops and releases every frame at once, so
 * the producer never waits. A worker thread searches the snapshot's covering channels
 * (@ref gnss::channelized_accumulate, or GnssCudaAcquire on the GPU), refines each detection's
 * code phase with exact despreads, and converts it with @ref gnss::detection_phase.
 *
 * Detections (PRN -> Doppler, code phases, SNR) are served at @c GET unique_name/get_detections
 * for the broker, with the Doppler sign-corrected for the r2c fold and the code phases as
 * defined by @ref gnss::DetectionPhase.
 *
 * REST: @c set_doppler_hints and @c set_nh_hint narrow the search per PRN; @c set_prns
 * changes the slot -> PRN map.
 *
 * @conf signal/sample_rate/spectrum_length/num_taps/pfb_window  As the F-engine.
 * @conf f_offset         Double (default 0). Carrier frequency in the channelized spectrum, Hz
 *                        (bin f_offset / (sample_rate / fft_len)).
 * @conf channel_offset   Int (default 0). Global index of this subband's first channel, for
 *                        the no-coverage log.
 * @conf channel_ids      Array<Int>. Global index of each local channel (a strided comb or a
 *                        contiguous run).
 * @conf n_channels       Int. Channels owned by this subband.
 * @conf prns             Array<Int>. PRN in each slot.
 * @conf doppler_min/max/step  Double (default -6000/6000/500). Blind Doppler grid, Hz. A
 *                        top-level /clock_profile block overrides min/max.
 * @conf acquire_snr      Double (default 12). Surface SNR to declare a detection.
 * @conf acquire_windows  Int (default 64). |D|^2 windows to integrate per snapshot.
 * @conf hops_per_record  Int (default: the replica period). Coherent window, hops.
 * @conf hold_snapshots   Int (default 5). Missed snapshots a detection survives.
 * @conf doppler_margin_hz Double (default 5000). Covering-channel selection band.
 * @conf hint_ttl_s       Double (default 8). Age at which a hint counts as absent.
 * @conf nh_search        Bool (default false). Search every secondary-code alignment.
 * @conf nh_hint_span     Int (default 1). Alignments scanned either side of a hinted one.
 * @conf acquire_threads  Int (default 1). Threads for the CPU acquire and the CPU refine.
 * @conf acquire_fine_step Int (default 1). Fine-lag decimation of the surface, samples.
 * @conf refine_span      Int (default fft_len). Refine half-width, samples.
 * @conf refine_step      Int (default 1). Refine step, samples.
 * @conf refine_hops      Int (default 0 = the full record). Refine integration, hops.
 * @conf prn_freq_offset_hz Array<Double> (optional). Per-slot FDMA carrier offsets, Hz.
 * @conf use_cuda_acquire Bool (default false). Acquire and refine on the GPU (CUDA builds only).
 * @conf prns_per_pass, blind_prns_per_pass, require_hint, acq_pairsum_select,
 *       nh_label_consensus, code_doppler_sign  Retired: accepted only at 0, 0, true, true, true
 *                        and 1, the behaviour the stage has.
 *
 * @par Doppler hints (POST @c unique_name/set_doppler_hints)
 * The broker posts per-PRN predictions @c [{prn, doppler_hz, margin_hz}]. A hinted PRN is
 * scanned only over @c doppler_hz +- margin_hz, on the @c doppler_step grid; an unhinted PRN
 * is skipped for the pass, so the cost follows the visible set even when @c prns lists a whole
 * constellation.
 *
 * @buffer in_buf This subband's channels, [hop][n_channels] cfloat32, with GnssChanMetadata
 *                (each frame is placed by its sample_seq).
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
    // ---- Slot membership (docs/CHORD_LIVE_PRN_RECONFIG.md) ----

    /// POST: the whole slot -> PRN map in slot order, as a bare array or @c {"prns": [...]},
    /// the payload cudaGnssChordTrackState::set_prns_callback takes. The slot count is fixed
    /// and PRNs must be distinct. Any @c at_hop is ignored: the map is staged and applied before
    /// the next pass.
    void set_prns_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    /// Apply any staged map. Worker thread only, between passes: the bank, the refine engines
    /// and the hint tables belong to the pass, and a swap inside one would change the code
    /// under a half-built surface.
    void apply_prn_swaps();

    std::mutex _prn_mtx; ///< guards @c _prns and the staging below
    std::vector<int> _pending_prns;
    bool _prn_pending = false;

    void set_doppler_hints_callback(kotekan::connectionInstance& conn,
                                    nlohmann::json& json_request);
    void set_nh_hint_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);

    struct Detection {
        double doppler_hz = 0.0;
        double code_phase_chips = 0.0;
        long long ref_hop = 0; ///< snapshot start hop (sample_seq / fft_len): the detection epoch
        float snr = 0.0f;
        bool valid = false;
        int misses = 0; ///< consecutive non-detecting snapshots since last valid hit
        /// Secondary-code alignment: the @c nh_phase of the winning replica (or the label vote's
        /// correction of it), so the overlay chip for primary period k
        /// is secondary[(k + nh) mod len]. Measured per satellite every pass; -1 unless
        /// alignments were searched.
        int nh = -1;
        /// Code phase mod the overlaid length (n_nh * code_length), or -1 unless alignments
        /// were searched. @c code_phase_chips is mod the primary length and cannot carry the
        /// overlay period.
        double code_phase_long_chips = -1.0;
        /// Physical code phase at @c ref_hop, mod the overlaid length; -1 unless alignments were
        /// searched. This is the one to transport: @c code_phase_chips is referred to sample 0
        /// and carries a Doppler lever of thousands of chips per Hz at CHORD uptime, while this
        /// one is referred to the search's fixed replica anchor (~1e-4 chips/Hz).
        double code_phase_at_ref_chips = -1.0;
    };

    /// Local channel of global spectrum index @c c (-1 = not ours), from @c channel_ids.
    int local_of_global(int c) const;

    Buffer* in_buf;

    int _N;
    int _fft_len;
    int _chan_offset;
    int _n_chan;
    std::vector<int> _chan_ids; ///< global index per local channel
    int _hops_per_record;       ///< hops per integration window
    int _acquire_windows;       ///< windows accumulated per snapshot
    int _hold_snapshots;        ///< keep a detection valid this many non-detecting snapshots
    double _sample_rate;
    double _doppler_margin_hz;
    double _acquire_snr;
    double _doppler_step; ///< grid step, Hz, of both the blind and the hinted grids
    /// Slot -> PRN, from config and then set_prns_callback. Written only by the worker
    /// (apply_prn_swaps) under @c _prn_mtx; other threads read it under the same lock.
    std::vector<int> _prns;
    /// The full Doppler extent (doppler_min..max, or the /clock_profile range) on the step grid:
    /// it sets the covering set and the GPU engine's size, and is scanned only for a hint whose
    /// window holds no grid point.
    std::vector<double> _doppler_grid;

    /// Per-PRN Doppler hint (POST set_doppler_hints): physical Doppler and margin, Hz.
    /// @c t_recv is steady_s() at receipt; a hint older than @c _hint_ttl_s counts as absent.
    struct DopHint {
        bool valid = false;
        double doppler = 0.0;
        double margin = 0.0;
        double t_recv = 0.0;
    };
    std::vector<DopHint> _dop_hints; ///< parallel to _prns (live, updated by REST)
    /// The hint table as of the snapshot, copied under _hint_mtx when the snapshot finalizes.
    /// The worker searches these, not the live table: the broker keeps predicting the current
    /// sky while a pass works on frozen data, and a satellite's Doppler drifts up to ~1 Hz/s.
    std::vector<DopHint> _snap_hints;
    double _snap_taken_s = 0.0; ///< steady_s() when the snapshot finalized (for hint TTL)
    /// Per-PRN secondary-code alignment hint (POST set_nh_hint): alignment @c nh at absolute
    /// hop @c ref_hop. Each alignment costs a full acquisition surface, so this is the larger
    /// saving. A fresh hint limits the scan to pred +- _nh_hint_span, with
    ///     pred = (nh + llround((H - ref_hop) / period_hops)) mod n_nh
    /// at snapshot start hop H. period_hops is not an integer (195.3125 at CHORD), so the
    /// rounded period count, and pred, can be off by one; the span absorbs that. The count also
    /// ignores code Doppler (0.04 periods per 12 s at 4 kHz on L5), so a hint degrades with age.
    struct NhHint {
        bool valid = false;
        int nh = 0;            ///< alignment index at ref_hop
        long long ref_hop = 0; ///< absolute hop the alignment refers to
        double t_recv = 0.0;
    };
    std::vector<NhHint> _nh_hints;      ///< live, updated by REST
    std::vector<NhHint> _nh_snap_hints; ///< frozen with the snapshot, like _snap_hints
    /// Per-alignment results for the PRN being searched, for the NHPROF log line and the
    /// label vote. Worker thread only.
    struct NhProfEntry {
        int nh;
        double snr;
        long tau;
        double dop; ///< peak Doppler, internal (r2c) sign
    };
    std::vector<NhProfEntry> _nh_prof;
    /// Alignments scanned either side of the predicted one; 0 trusts the prediction exactly.
    int _nh_hint_span = 1;
    std::mutex _hint_mtx;
    int _acquire_threads = 1;     ///< threads for channelized_accumulate and refine_peak
    int _acquire_fine_step = 1;   ///< fine-lag decimation of the surface (see AcquisitionSurface)
    long _last_surface_cells = 0; ///< cells in the last surface (for the noise-ceiling log)
    long _empty_hint_passes = 0;  ///< passes with no fresh hint at all (rate-limited WARN)
    int _last_cov_n = -1;         ///< last logged covering set (log once per set, not snapshot)
    int _last_cov_first = -1, _last_cov_last = -1;
    double _hint_ttl_s = 8.0; ///< hint age, s, beyond which it counts as absent (<= 0: never)

    std::unique_ptr<gnss::ChannelizedReplicaBank> _replica;
    gnss::AcquireWorkspace _acq_ws;

    long _grid_bin_warns = 0; ///< times a non-bin-aligned Doppler grid was seen

#ifdef GNSS_CUDA
    /// Device-resident acquire (docs/gnss_gpu_search.md). Null unless @c use_cuda_acquire and
    /// the engine can represent the geometry exactly; the CPU acquire is the fallback.
    std::unique_ptr<class GnssCudaAcquire> _cuda_acq;
    bool _cuda_acq_wanted = false;
    long _cuda_acq_fallbacks = 0; ///< times a pass fell back to the CPU (rate-limited WARN)

    /// GPU refine: one GnssCudaDespread per group of <= 64 covering channels (DespreadJob's
    /// chan_mask is a uint64_t), and the group descriptors refine_peak_cuda takes. Empty means
    /// the CPU refine runs. Independent of _cuda_acq: either can be unavailable alone.
    std::vector<std::unique_ptr<class GnssCudaDespread>> _cuda_ref;
    std::vector<gnss::CudaRefineGroup> _cuda_ref_groups;
#endif

    /// Post-detection code-phase refine: one exact despread per trial offset, over
    /// +-_refine_span samples in steps of _refine_step samples. The step must stay below the
    /// fine-lag lobe width, ~fft_len / (comb span in bins) samples, or the scan can skip the
    /// true lobe (see refine_peak); a much finer step only adds despreads.
    int _refine_span = 0;
    /// Refine integration length in hops (0 = the full record). The refine only localises a
    /// peak that already cleared detection, and its cost is linear in this.
    int _refine_hops = 0;
    int _refine_step = 0;

    /// Replica at code phase 0 and Doppler 0 on the covering channels, split by code period so
    /// that every overlay alignment is a per-hop sign pair. Hop m is the part from period k0[m]
    /// (head) plus the part from period k0[m] + 1 (tail):
    ///     R_nh[m] = s(k0[m] + nh) * head[m]  +  s(k0[m] + 1 + nh) * tail[m]
    /// The generator applies the overlay per chip, and the nav_bit hook takes the absolute chip
    /// index, so two generator calls give head and tail:
    ///     A: nav_bit = +1            -> A[m] = head[m] + tail[m]
    ///     B: nav_bit = (-1)^period   -> B[m] = (-1)^k0[m] * (head[m] - tail[m])
    /// The split assumes every chip of hop m lies in period k0[m] or k0[m] + 1. Only covering
    /// rows are filled. Built once per covering set (and cached on disk); worker thread only.
    std::vector<std::vector<std::vector<std::complex<float>>>> _repl_head; ///< [prn][chan][hop]
    std::vector<std::vector<std::vector<std::complex<float>>>> _repl_tail;
    std::vector<long long> _repl_k0; ///< [hop] period index at sample anchor + m*fft_len
    std::vector<std::vector<std::complex<float>>> _repl_scratch; ///< materialised R_nh, reused
    std::vector<float> _sgn0, _sgn1; ///< per-hop overlay sign pair for the current alignment
    std::vector<int> _repl0_cover;   ///< covering set of the head/tail bank (rebuild key)
    /// Search every secondary-code alignment and keep the best peak, at one surface per
    /// alignment scanned. Without the right alignment a multi-period window loses coherent
    /// gain (12.7 dB rms on L5 Q5 for an overlay-blind replica, with exact nulls). When off,
    /// the acquire materialises alignment 0 and the refine runs with the overlay off.
    bool _nh_search = false;
    int _n_nh = 1; ///< alignments in a full scan (1 when not searching)

    std::vector<Detection> _detections; ///< per-PRN latest detection
    std::mutex _det_mtx;

    // --- snapshot hand-off (consumer loop -> worker) ---
    std::vector<std::complex<float>> _snapshot; ///< acquire_windows*hpr hops, n_chan each
    size_t _snap_hops;                          ///< hops the snapshot holds
    long long _snap_start_hop;                  ///< absolute hop index of snapshot hop 0
    std::thread _worker;
    std::mutex _m;
    std::condition_variable _cv;
    bool _snap_ready;
    bool _worker_busy;
    bool _worker_stop;
};

#endif // GNSS_CHANNELIZED_SEARCH_HPP
