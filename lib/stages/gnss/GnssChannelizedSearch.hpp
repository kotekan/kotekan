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
 * REST: @c set_doppler_hints and @c set_nh_hint narrow the search per PRN; @c set_prns and
 * @c get_prns change and read the slot -> PRN map.
 *
 * @conf signal/sample_rate/spectrum_length/num_taps/pfb_window  As the F-engine.
 * @conf f_offset         Double (default 0). Carrier frequency in the channelized spectrum, Hz
 *                        (bin f_offset / (sample_rate / fft_len)).
 * @conf channel_offset   Int (default 0). Global index of this subband's channel 0, for a
 *                        contiguous subband.
 * @conf channel_ids      Array<Int> (optional). Global index of each local channel, for a
 *                        sparse subband (a strided comb). Overrides channel_offset.
 * @conf n_channels       Int. Channels owned by this subband.
 * @conf prns             Array<Int>. PRN in each slot.
 * @conf doppler_min/max/step  Double (default -6000/6000/500). Blind Doppler grid, Hz. A
 *                        top-level /clock_profile block overrides min/max.
 * @conf acquire_snr      Double (default 12). Surface SNR to declare a detection.
 * @conf acquire_windows  Int (default 64). |D|^2 windows to integrate per snapshot.
 * @conf hops_per_record  Int (default: the replica period). Coherent window, hops.
 * @conf hold_snapshots   Int (default 5). Missed snapshots a detection survives.
 * @conf doppler_margin_hz Double (default 5000). Covering-channel selection band.
 * @conf hint_doppler_sign Double (default -1). Physical (hinted) to internal Doppler sign.
 * @conf hint_ttl_s       Double (default 8). Age at which a hint counts as absent.
 * @conf require_hint     Bool (default false). Skip PRNs without a fresh Doppler hint.
 * @conf prns_per_pass    Int (default 0 = all). Eligible PRNs searched per pass, round-robin.
 * @conf blind_prns_per_pass Int (default 0 = no limit). Unhinted PRNs blind-scanned per pass.
 * @conf nh_search        Bool (default false). Search every secondary-code alignment.
 * @conf nh_hint_span     Int (default 1). Alignments scanned either side of a hinted one.
 * @conf nh_label_consensus Bool (default false). See @c _nh_label_consensus.
 * @conf acq_pairsum_select Bool (default false). See @c _acq_pairsum_select.
 * @conf acquire_threads  Int (default 1). Threads for the CPU acquire and the CPU refine.
 * @conf acquire_fine_step Int (default 1). Fine-lag decimation of the surface, samples.
 * @conf refine_span      Int (default fft_len). Refine half-width, samples.
 * @conf refine_step      Int (default 1). Refine step, samples.
 * @conf refine_hops      Int (default 0 = the full record). Refine integration, hops.
 * @conf code_doppler_sign Double (default 1). See ChannelizedReplicaBank::code_doppler_sign.
 * @conf prn_freq_offset_hz Array<Double> (optional). Per-slot FDMA carrier offsets, Hz.
 * @conf use_cuda_acquire Bool (default false). Acquire on the GPU (CUDA builds only).
 * @conf use_cuda_refine  Bool (default use_cuda_acquire). Refine on the GPU.
 *
 * @par Doppler hints (POST @c unique_name/set_doppler_hints)
 * The broker posts per-PRN predictions @c [{prn, doppler_hz, margin_hz}]. A hinted PRN is
 * scanned only over @c doppler_hz +- margin_hz, on the @c doppler_step grid; an unhinted PRN
 * gets the blind grid, or is skipped under @c require_hint.
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
    // ---- Slot membership (docs/CHORD_LIVE_PRN_RECONFIG.md) ----

    /// POST: the whole slot -> PRN map in slot order, as a bare array or @c {"prns": [...]},
    /// the payload cudaGnssChordTrackState::set_prns_callback takes. The slot count is fixed
    /// and PRNs must be distinct. Any @c at_hop is ignored: the map is staged and applied before
    /// the next pass.
    void set_prns_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    /// GET: the live map, plus the swap count.
    void get_prns_callback(kotekan::connectionInstance& conn);
    /// Apply any staged map. Worker thread only, between passes: the bank, the refine engines
    /// and the hint tables belong to the pass, and a swap inside one would change the code
    /// under a half-built surface.
    void apply_prn_swaps();

    std::mutex _prn_mtx; ///< guards @c _prns and the staging below
    std::vector<int> _pending_prns;
    bool _prn_pending = false;
    uint64_t _prn_swaps = 0;
    std::string _prn_last_err;

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
        /// correction of it, under nh_label_consensus), so the overlay chip for primary period k
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
    std::vector<double> _doppler_grid; ///< blind fallback grid (doppler_min..max)

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
    /// Serve the scanned alignments' majority period label instead of the winner's own when
    /// they disagree. A per-alignment argmax can land on an adjacent alignment's peak cell
    /// (same tau, wrong nh), and (nh + lag) then mislabels the period by +-1 while the fine
    /// phase stays right. The other alignments of the pass vote at no extra cost; no history.
    bool _nh_label_consensus = false;
    /// Choose each surface's peak cell on the mainlobe pair-sum instead of the raw sample, so
    /// an on-grid overlay Doppler sideband cannot beat a scalloped true peak. The served snr
    /// stays the chosen cell's raw peak/mean.
    bool _acq_pairsum_select = false;
    std::mutex _hint_mtx;
    /// Scan only PRNs with a fresh Doppler hint and skip the rest, so the cost follows the
    /// visible set even when @c prns lists a whole constellation.
    bool _require_hint = false;
    int _acquire_threads = 1;     ///< threads for channelized_accumulate and refine_peak
    int _acquire_fine_step = 1;   ///< fine-lag decimation of the surface (see AcquisitionSurface)
    int _prns_per_pass = 0;       ///< eligible PRNs searched per snapshot (0 = all)
    int _prn_cursor = 0;          ///< round-robin start into the eligible list, advanced per pass
    double _hint_dop_sign = -1.0; ///< physical -> internal Doppler sign (see the hinted grid)
    long _last_surface_cells = 0; ///< cells in the last surface (for the noise-ceiling log)
    long _empty_hint_passes = 0;  ///< require_hint passes that scanned nothing (rate-limited WARN)
    int _last_cov_n = -1;         ///< last logged covering set (log once per set, not snapshot)
    int _last_cov_first = -1, _last_cov_last = -1;
    double _hint_ttl_s = 8.0; ///< hint age, s, beyond which it counts as absent (<= 0: never)

    std::unique_ptr<gnss::ChannelizedReplicaBank> _replica;
    gnss::AcquireWorkspace _acq_ws;

    /// Unhinted PRNs blind-scanned per pass, in rotation (0 = no limit). An unhinted PRN costs
    /// a full blind grid, so this bounds the cost of reacquisition when most listed satellites
    /// are below the horizon. No effect with require_hint, which skips every unhinted PRN.
    int _blind_prns_per_pass = 0;
    int _blind_cursor = 0;    ///< rotates through the unhinted set, advanced once per pass
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
    bool _cuda_ref_wanted = false;
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

    /// Sized [alignment][prn] and cleared on a PRN swap, but never filled or read: the search
    /// materialises its replica from _repl_head and _repl_tail.
    std::vector<std::vector<std::vector<std::vector<std::complex<float>>>>> _repl0;
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
