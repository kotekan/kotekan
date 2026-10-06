#ifndef GNSS_GPU_RECORD_ASSEMBLE_HPP
#define GNSS_GPU_RECORD_ASSEMBLE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssElemCal.hpp"
#include "gnssElemSteer.hpp"
#include "gnssProjSubspace.hpp"
#include "restServer.hpp"
#include "json.hpp"    // nlohmann::json for the set_elem_gain POST

#include <atomic>
#include <complex>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

/**
 * @class GnssGpuRecordAssemble
 * @brief Host tail of the phase-F GPU tracking chain: gnssGpuChain frames -> tracker records.
 *
 * Consumes the cudaProcess output (control block + raw per-channel E/P/L correlations, layout
 * gnssGpuChain.hpp) and performs GnssChannelizedTracker's pass-2: cross-channel summation over
 * each PRN's covering mask, the carrier-NCO phase integration + derotation (phase continuity
 * state lives here; the slope f_nco = ctrim + ff rides in the control block), and the
 * gnssRecord.hpp record floats. Emits one rec_buf frame per record window with the window's
 * absolute sample in GnssChanMetadata -- byte-compatible with the CPU tracker's output, so the
 * combiner/broker/viewer are untouched.
 *
 * @conf in_buf   gnssGpuChain frames from the cudaProcess chain
 * @conf out_buf  tracker record frames (n_prn * record_floats * float)
 * @conf prns     PRN list (must match the cudaGnssTrack command's)
 * @conf sample_rate  (for the NCO dt; default 5e6)
 */
class GnssGpuRecordAssemble : public kotekan::Stage {
public:
    GnssGpuRecordAssemble(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);
    ~GnssGpuRecordAssemble() override;
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
    /// Slot -> PRN. SEEDED from config, then FOLLOWED FROM THE FRAME (@ref follow_frame_prns):
    /// after a live swap the config value is stale and the frame's is authoritative.
    std::vector<int> _prns;
    double _sample_rate;

    /// WALL-CLOCK FALLBACK, LATCHED ONCE (see main_thread). When the producer supplies no
    /// frame0_utc the records have no absolute anchor, and the old fallback stamped
    /// `system_clock::now()` per record -- which on CHORD puts the four sub-records of a frame
    /// microseconds apart instead of 10.49 ms, because they are assembled back to back. That is
    /// not a small error: every CROSS-RECORD estimator in GnssCoherentCombiner works in UTC and
    /// derives its grid from the MINIMUM consecutive spacing, so a burst of near-equal stamps
    /// scrambles the record order inside the transform. Anchoring once and extrapolating by
    /// wstart keeps the same (host-clock) origin while making the grid exactly uniform.
    double _wall_anchor = 0.0; ///< now() - wstart/rate at the first unanchored frame; 0 = unset
    uint64_t _no_utc0_frames = 0; ///< frames stamped from the fallback (for the rate-limited warn)

    /// Element axis (CHORD). 0 = single-antenna airspy layout, byte-for-byte.
    int _n_elements = 0;
    /// Which antenna the record HEADER's correlation slots carry -- the broker's loop reference.
    int _reference_element = 0;
    bool _elem_hold_on_reanchor = true;  ///< keep element cal across a carrier re-anchor
    std::vector<uint8_t> _elem_prev_ok;  ///< element-cal continuity, decoupled from carrier
    std::vector<double> _fnco_prev;      ///< previous record's f_nco: the slope in force over
                                         ///< the gap [t_prev, t_now] (the [4e] pairing fix)
    /// SELF-CALIBRATED ELEMENT SUM (gnssElemCal.hpp; CHORD_GNSS_STATE 8.21.5). When enabled the
    /// header correlation slots carry the calibrated weighted MEAN over all elements instead of
    /// the bare reference element: same phase convention (reference-anchored), same "one
    /// element" scale, per-record SNR up ~sqrt(N_healthy) -- which is what makes the per-record
    /// carrier phase estimable from one instance (the phase-floor fix) and hands the broker's
    /// DLL/carrier loops the array gain for free. Until each PRN's cal is warm (~3 tau of
    /// updates) the header is the reference element, byte-identical to the historical output.
    bool _elem_sum = false;
    double _elem_sum_tau_s = 0.5;  ///< cal EMA time constant -- fast enough to follow the
                                   ///< inter-element fringe rotation as a satellite transits
    // ── #102 ELEMENT STEERING (see gnssElemSteer.hpp) ─────────────────────────────
    gnss::ElemSteer _steer;      ///< per-(sat, channel, element) geometric phasors
    std::mutex _steer_mtx;       ///< REST update vs combine-loop read
    double _steer_t0 = 0.0;      ///< steady-clock epoch for freshness
    double _elem_sum_min_w = 0.02; ///< weight gate vs the strongest element: absent/unpowered
                                   ///< elements (EMA of pure noise) fall below and are excluded
    std::vector<gnss::ElemCal> _cal; ///< per PRN slot
    /// HOLD vs ADAPT (config elem_sum_adapt, live /set_elem_sum_adapt): false freezes every
    /// PRN's live weights and lets _cal_shadow learn instead; _cal_sim[p] is the shadow's
    /// agreement with the held weights (the capture detector), -1 until both are warm.
    std::atomic<bool> _elem_adapt{true};
    std::vector<gnss::ElemCal> _cal_shadow;
    std::vector<double> _cal_sim;
    /// SHARED INSTRUMENT MODEL (config elem_sum_shared, live /set_elem_sum_shared). With
    /// geometry steered, every satellite's per-element vector is the same instrument to within
    /// the per-dish pattern residual -- WITHIN A POLARISATION. Between the two feeds of one
    /// dish the phase is direction-dependent by up to +-150 deg (the E/H-plane sidelobe
    /// patterns differ), so the two halves cannot share one phase. The model is therefore one
    /// per-element gain per pol (_g_shared, each half normalised to sum|g| = 1, learned as a
    /// slow consensus of the per-PRN SHADOW cals) plus ONE complex coefficient per PRN
    /// (_pol_c: the pol-1 sub-beam against the pol-0 sub-beam, a fast EMA). A bright satellite
    /// in a transit can capture a per-PRN learner in seconds; it moves a 300-s consensus by a
    /// few percent and a sub-beam ratio not at all, so the weak satellites keep the array gain.
    /// shared implies held (_elem_adapt = false): the live weights are rebuilt from the model
    /// every record and the learners only ever feed the consensus.
    std::atomic<bool> _elem_shared{false};
    double _elem_shared_tau_s = 300.0;   ///< consensus EMA (slower than any transit)
    double _elem_pol_tau_s = 3.0;        ///< per-PRN inter-pol coefficient EMA
    std::vector<std::complex<double>> _g_shared; ///< [n_elem]: pol-0 half then pol-1 half
    bool _g_shared_warm = false;
    uint8_t _g_shared_collapsed = 0;     ///< last consensus refused (one element > 50%)
    int _g_shared_n = 0;                 ///< PRNs that fed the last consensus
    double _g_shared_t = 0.0;            ///< steady time of the last consensus refresh
    /// THE TRANSIT FREEZE. A satellite near boresight captures every weak satellite's per-PRN
    /// learner at once (its leakage dominates their per-element despread), and a consensus of
    /// captured learners is a captured model. The broker posts the pooled (all-constellation)
    /// nearest-to-boresight separation with each geometry post; while it is inside
    /// _elem_shared_freeze_deg, and for _elem_shared_freeze_hold_s after, neither the model
    /// nor any inter-pol coefficient learns. Written by the REST thread, read by main.
    double _elem_shared_freeze_deg = 6.0;
    double _elem_shared_freeze_hold_s = 60.0;
    std::atomic<double> _bore_sep_deg{1.0e9};   ///< last posted pooled boresight separation
    std::atomic<double> _bore_post_t{-1.0e18};  ///< steady time of that post
    double _freeze_until = -1.0e18;             ///< main thread: frozen while now < this
    /// THE PHASE PIN. The model's global phase per pol is pinned to the FIRST healthy model
    /// (all elements, <ref, G> real positive), not to one element: an element's weight can
    /// collapse, and a pin on it then goes to noise independently on every instance, which
    /// cancels the instances in the fleet sum. The first model is pinned on the reference
    /// element, which is what makes the instances agree to begin with.
    std::vector<std::complex<double>> _g_pin_ref;
    bool _g_pin_ref_ok = false;
    /// THE FLEET REFERENCE (#154). The own pin above comes from one shadow, so a poor first one
    /// (noise before the broker re-anchors after an F-engine re-base, a weak shadow at a cold
    /// start) leaves this instance's global phase arbitrary against the others and the fleet sum
    /// loses it until a restart. F (config elem_sum_shared_ref, live /set_elem_sum_shared_ref) is
    /// one vector per band shared by every instance, a snapshot of the fleet consensus
    /// (python/scripts/gnss/elem_shared_ref.py). Mode "live" pins each pol <F, G> real positive
    /// instead: in full for a first model, slewed at _fleet_slew_rad_s for a warm one
    /// (gnssSharedPin.hpp). "log" only reports the offset; "off" ignores F. A half F does not
    /// describe (sim < _fleet_min_sim) keeps the own pin, which tracks the model's phase in live
    /// mode so that falling back never steps it.
    std::vector<std::complex<double>> _g_fleet_ref; ///< [n_elem]; empty = no reference
    std::atomic<int> _fleet_mode{0};                ///< 0 off, 1 log, 2 live
    std::atomic<bool> _fleet_ref_present{false};    ///< _g_fleet_ref non-empty (for REST)
    double _fleet_slew_rad_s = M_PI / 180.0;
    double _fleet_min_sim = 0.5; ///< a 16-element noise model scores ~0.2 against any F
    /// Per pol, after the last consensus update (diagnostics, read unlocked by REST):
    double _fleet_err_deg[2] = {0.0, 0.0}; ///< arg<F, G>: the offset still to remove
    double _fleet_sim[2] = {0.0, 0.0};     ///< |<F, G>| / (|F| |G|)
    bool _fleet_applied[2] = {false, false};
    bool _fleet_slewing[2] = {false, false}; ///< one WARN per slew episode, one when done
    /// REST staging (under _gain_mtx), consumed by shared_consensus on the main thread.
    std::vector<std::complex<double>> _pending_fleet_ref;
    bool _pending_fleet_ref_set = false;
    int _pending_fleet_mode = -1;
    double _pending_fleet_slew_deg_s = -1.0;
    void fleet_ref_apply_pending();
    bool shared_frozen(double now_s);
    std::vector<std::complex<double>> _pol_num; ///< per PRN: EMA of B1 conj(B0)
    std::vector<double> _pol_den;               ///< per PRN: EMA of |B0|^2
    std::vector<double> _pol_warmth;            ///< per PRN: -> 1 with tau
    std::vector<std::complex<double>> _pol_c;   ///< per PRN: the coefficient in force (diagnostic)
    std::vector<std::complex<double>> _w_scratch; ///< [n_elem] held weights being built
    void shared_reset_prn(size_t p);
    void shared_consensus(double now_s);
    void shared_hold(size_t p);
    void shared_pol_update(size_t p, const std::complex<double>* g_prompt, double dt_s);
    std::vector<gnss::ElemSteer::cf> _steer_buf; ///< this PRN's [n_chan][n_elem] phasors, copied under _steer_mtx
    uint8_t _steer_nchan_warned = 0;
    std::vector<uint8_t> _anchor_warned; ///< one WARN per PRN when the phase anchor moves off
                                         ///< the reference element (a one-time phase step
                                         ///< downstream); cleared on cal reset
    /// Scratch, [n_rows_spec][n_elem]: the per-antenna covering-mask sum, reused per PRN so the
    /// per-record path does not allocate.
    std::vector<std::complex<double>> _g_elem;

    // NCO state per PRN slot (pass-2's half of the carrier machinery).
    std::vector<double> _phi;
    std::vector<double> _phi_cyc;   ///< NCO phase, UNWRAPPED, in cycles (the export's time base;
                                    ///< _phi is the same phase wrapped for the rotation)
    std::vector<double> _phi_cmd_prev; ///< previous record's commanded phase (cycles)
    std::vector<uint8_t> _phi_cmd_ok;
    std::vector<double> _fcar_prev; ///< previous record's replica f_ref (to size the re-pin step)
    std::vector<uint8_t> _fcar_prev_ok;
    std::vector<std::complex<double>> _a_prev;
    std::vector<uint8_t> _a_prev_ok;
    std::vector<int64_t> _wstart_prev;

    /// Per-channel PROMPT-phase dump (chan_dump_prn / chan_dump_decim / chan_dump_path):
    /// DIAGNOSTIC (2026-07-21, L5 ADR-wander): the channel-width A/B showed the wander
    /// amplitude depends on the despread channel set (narrow 5-ch = 5-6x WORSE than the
    /// full 10) -> the mechanism lives in the per-channel phases the cross-channel sum
    /// normally hides. For the one listed PRN, every decim-th record writes one line per
    /// covering channel: "utc ch corr_re corr_im energy" (raw, pre-NCO-rotation -- the
    /// cross-channel RELATIVE phases are the observable). ~60 KB/s at 100 Hz x 10 ch.
    int _chan_dump_prn = -1;   ///< PRN number to dump (-1 = disabled)
    int _chan_dump_decim = 10; ///< dump every Nth record of that PRN
    long long _chan_dump_ctr = 0;
    FILE* _chan_dump = nullptr;
    int _phi_dump_prn = -1;   ///< --phase-dump-prn (see the .cpp): the fold's inputs and effect
    int _phi_dump_left = 0;
    FILE* _phi_dump = nullptr;

    /// PER-CHANNEL PROMPT SPECTRUM (task #32, docs/CHORD_JOINT_TRACKING.md P1). The general
    /// form of the chan_dump above: for EVERY PRN, accumulate the NCO-derotated, element-
    /// combined prompt per covering channel over a window, and serve it on
    /// `<unique_name>/get_spectrum?window=N`. A delay is a phase ramp across frequency, and this
    /// is the
    /// sufficient statistic for the fleet-level phase-slope delay fit in the broker --
    /// per-(PRN, channel) complex sums, ~4 kB per poll, never per-element data (the 30 Gbps
    /// full-CHORD trap). The derotation is the SAME `rot` the record's prompt gets, one
    /// common phase per record: it stops the residual-carrier winding across the window
    /// without touching the cross-channel RELATIVE phases, which are the observable.
    /// Enabled by the presence of `channel_ids` in the config (the generator wires the same
    /// per-GPU list the despread runs); absent -> fully inert, airspy/legacy byte-identical.
    ///
    /// ⚠️ WINDOWS ARE ADDRESSABLE AND HOP-QUANTISED (task #53, 2026-08-12). They used to be
    /// "whatever accumulated since your last GET", with a reset-on-read -- so the window was
    /// defined by WHEN THE BROKER'S REQUEST ARRIVED, and the broker polls 12 instances
    /// SEQUENTIALLY. The instances were therefore never summing the same records, and the
    /// broker could not repair it: re-polling a laggard returns a NEW SHORTER window, never
    /// the records it missed. There is no way to ask for the past.
    ///
    /// That misalignment is not cosmetic. Each instance's channels are a COMB spanning
    /// ~18.75 MHz (7 channels, stride 16), so the cross-instance phase relationship is a delay
    /// ramp -2*pi*f*tau, and the broker was absorbing the window offset into a FREE PHASE PER
    /// INSTANCE fitted from the data it then summed -- a self-reference that aligns noise and,
    /// when it fails, drops the whole chain to the quadrature fallback (gps_l5 measured
    /// align 0.143 with 9/12 satellites on `quad`). See task #52.
    ///
    /// Now: window index = floor(wstart / _spec_win_samples), derived from the F-engine sample
    /// clock, so every instance assigns a record to the SAME window with no negotiation. A ring
    /// of completed windows lets a laggard still be asked for the window its peers already
    /// returned. Reads are IDEMPOTENT -- no reset -- so a second poller is harmless.
    /// PER-CHANNEL COMB EXPORT (gnssRecord.hpp's chan block). Appends the UNSUMMED per-channel
    /// prompt after the PRN records -- the same NCO-derotated, element-combined value the
    /// spectrum ring accumulates, but PER RECORD, because a cross-record rate fit cannot be
    /// done on a window sum. Off by default; requires channel_ids.
    bool _chan_export = false;
    std::vector<int> _spec_freq_ids;             ///< [n_chan] F-engine freq_id per channel
    int64_t _spec_win_samples = 0;               ///< window length, SAMPLES (0 = legacy mode)
    /// One accumulated window. Slot for index i is _spec_ring[i % depth], so a window is
    /// evicted only when the ring wraps past it -- no bookkeeping list, and the slot's own
    /// `idx` is what says whether it still holds what you asked for.
    struct SpecWindow {
        int64_t idx = -1;                        ///< window index, or -1 for an unused slot
        int64_t w0 = -1, w1 = -1;                ///< wstart of the first/last record in it
        std::vector<double> re, im, energy;      ///< [n_prn * n_chan]
        std::vector<int> nrec;                   ///< [n_prn]
        /// [n_prn] the NCO phase _phi[p] at this window's FIRST record, and how many times
        /// the PRN re-anchored inside it. PUBLISHED, NOT SUBTRACTED (task #52) -- the export's
        /// phase currency, without which windows cannot be related to each other at all.
        std::vector<double> phi0;
        std::vector<int> nreanchor;
    };
    std::vector<SpecWindow> _spec_ring;          ///< depth from config; index -> idx % depth
    int64_t _spec_max_idx = -1;                  ///< newest index SEEN; complete windows are < this
    std::mutex _spec_mtx;                        ///< guards _spec_* between main_thread and REST
    // PATH B: an injected per-element complex gain prior (e.g. N^2 eigenvector, sky removed).
    // The REST callback stages it here; main_thread swaps it out and seeds every PRN's ElemCal.
    std::mutex _gain_mtx;                         ///< guards _pending_gain between REST and main_thread
    std::vector<std::complex<double>> _pending_gain;
    bool _pending_gain_set = false;
    /// LIVE REFERENCE SWAP (KV, 2026-08-20): /set_reference_element stages the new element
    /// here; main_thread applies it at the next frame boundary -- atomically with respect to
    /// the per-record loop, under the same producer/consumer pattern as the gain prior above.
    /// -1 = nothing pending. Applying rebuilds every PRN's ElemCal COLD (the stored prior and
    /// all learned gains are phase-anchored to the OLD reference and do not transfer), so the
    /// header rides the new bare reference for ~3 tau while the cal re-warms.
    int _pending_ref = -1;
    std::vector<std::complex<double>> _spec_scratch; ///< [n_elem] per-channel cal-combine input

    // ── THE BEAM CUBE: the (channel x element) axis, un-collapsed (2026-09-03) ────────────
    /// ⚠️ BOTH AXES ALREADY SURVIVE THIS STAGE -- SEPARATELY, AND THAT IS THE WHOLE PROBLEM.
    /// The element blocks are summed over the covering channels (the per-antenna covering-mask
    /// sum in main_thread), and the comb block is "NCO-derotated and ELEMENT-COMBINED, i.e. one
    /// element-equivalent per channel" (gnssRecord.hpp). So a beam map can be resolved in
    /// frequency OR in element, never in both -- while `corr` on the host is literally
    /// [rows][n_chan][n_elem] and has carried the joint quantity all along. Two different sums
    /// over one array, taken a few lines apart, and neither keeps what a per-element
    /// per-subband beam map needs.
    ///
    /// Why that matters and gets worse: the beam evolves across a wide signal (L5 spans ~20 MHz
    /// over 52 channels here), and a BOC signal puts its power in TWO lobes tens of MHz apart,
    /// so a frequency-collapsed per-element map averages a split spectrum and describes neither
    /// lobe. Per element, because separating a feed problem from an array problem is exactly
    /// what the element axis is for.
    ///
    /// ⚠️ DELIBERATELY **NOT** IN THE RECORD. config/chord_gnss_node.yaml called this "a real
    /// change to the assembler and the schema"; the schema half is avoidable. A beam map wants
    /// an INTEGRATED power, not a per-record stream -- so this is an accumulator served over
    /// REST next to /get_spectrum, and the frame layout, record_stride() and every downstream
    /// consumer are untouched. No flag day.
    ///
    /// The value is |A_e,c|^2 with A_e,c = G_e,c / E_c -- the SAME per-channel replica energy
    /// normalises every element, because one replica is correlated against all of them, so the
    /// ratios stay comparable across antennas (the property the beam map is built on). It is
    /// INCOHERENT, so no NCO rotation is applied or needed: `rot` cancels in the magnitude.
    /// Still BIASED by the noise pedestal -- debiasing is the broker's job, from the probe
    /// PRNs, in the power domain, exactly as for the element archive's p2.
    bool _cube_on = false;
    /// Channels per output subband bin. 0 (default) = one bin per channel: the finest cube
    /// the instrument can produce. Binning trades frequency resolution for archive volume,
    /// which is the binding constraint here -- NOT memory, and not compute.
    int _cube_bin_width = 0;
    int _cube_bins = 0;                    ///< derived: number of subband bins

    /// ── ARC PRESERVATION: WHY THIS IS A WINDOW RING AND NOT A RESET-ON-READ SUM ─────────
    /// The first version of this accumulator kept only SUM |A|^2 and was reset on read. That
    /// is correct for a beam map -- an incoherent power carries no phase, so there is nothing
    /// for a misaligned window to decohere -- and it is USELESS for anything that needs the
    /// arc: element calibration, a phased-array map, delay/TEC. Keeping the coherent sum is
    /// one extra float per cell and it cannot be recovered later, so it is kept.
    ///
    /// ⚠️ THE COHERENT SUM IS MEANINGLESS WITHOUT ITS PHASE REFERENCE (task #52). Each record
    /// is rotated by exp(-i*_phi[p]); `_phi[p]` is ZEROED on a fresh acquisition and STEPPED
    /// on a continuous re-pin, and the broker re-pins seeds every ~2 minutes. Integrate ~96
    /// records without publishing the origin and the consumer holds a number whose phase
    /// reference it cannot know. So phi0 -- _phi[p] at the window's FIRST record -- is
    /// published, exactly as the spectrum ring does.
    ///
    /// ⚠️ DO **NOT** "TIDY" THIS BY REFERENCING TO THE WINDOW'S OWN FIRST RECORD. That was
    /// tried (45fe3a438) to remove the arbitrary per-instance origin of _phi, and it gives
    /// every window its own arbitrary constant: measured coherence across 7 consecutive
    /// windows fell to 0.38 against a 0.378 random baseline, i.e. destroyed. Publish the raw
    /// phi, do not subtract an origin.
    ///
    /// ⚠️ AND LOST ARC LOOKS EXACTLY LIKE SIGNAL. Coherence inside one window is capped by the
    /// residual carrier (1 Hz of residual is ~6.3 rad over 1 s and the sum collapses) and by
    /// the data/overlay symbol rate; the incoherent sum has neither cap. A satellite whose
    /// carrier loop is limping therefore yields a SMALL coherent sum -- indistinguishable
    /// from "the beam is weak here", which is the very quantity a beam map measures. Both
    /// sums are stored so |SUM coh|^2 / SUM |.|^2 is available as the coherence, and that
    /// ratio is the self-check: at ~1/N the arc was lost in that window and the archive says
    /// so itself, instead of the loss arriving later as a plausible map.
    ///
    /// WINDOWS ARE ADDRESSABLE, not reset-on-read, for the reason #53 made them so for the
    /// spectrum: the index is floor(wstart / win_samples) on the F-engine sample clock, so
    /// every instance assigns a record to the same window WITHOUT talking to any other
    /// instance, and a second consumer no longer steals anyone's data.
    struct CubeWindow {
        int64_t idx = -1;                  ///< window index, or -1 for an unused slot
        int64_t w0 = -1, w1 = -1;          ///< wstart of the first/last record in it
        double utc0 = 0.0;                 ///< UTC of sample 0 as stamped on its records (v3)
        std::vector<double> coh_re, coh_im; ///< [n_prn * n_bin * n_elem] SUM A_e,c * rot
        std::vector<double> incoh;          ///< [n_prn * n_bin * n_elem] SUM |A_e,c|^2
        std::vector<double> w;              ///< [n_prn * n_bin] (record, channel) term count
        std::vector<double> energy;         ///< [n_prn * n_bin] SUM replica energy
        std::vector<int> nrec;              ///< [n_prn] records contributing
        std::vector<double> phi0;           ///< [n_prn] _phi[p] at this window's FIRST record
        std::vector<int> nreanchor;         ///< [n_prn] UNFOLDED resets inside it (see below)
        /// Records assigned to this window so far, ALL PRNs -- not per-slot like `nrec`. Its
        /// only job is the unit check in cube_window_for(): a window that never sees a second
        /// record is what a window length given in hops rather than samples looks like.
        int nrec_seen = 0;
    };
    std::vector<CubeWindow> _cube_ring;    ///< depth from config; index -> idx % depth
    int64_t _cube_win_samples = 0;         ///< window length in F-engine samples
    int64_t _cube_max_idx = -1;            ///< newest index SEEN; complete windows are < this
    int _cube_singleton_windows = 0;    ///< consecutive windows that held exactly one record
    bool _cube_win_warned = false;      ///< the unit-error warning fires once, not per record
    std::mutex _cube_mtx;                  ///< guards _cube_* between main_thread and REST
    /// ── THE PUSH LEG: completed windows go OUT, they are not fetched ───────────────────
    /// ⚠️ A POLLED ENDPOINT CANNOT PRODUCE A COMPLETE DATASET, and this stage's own
    /// neighbourhood already learned that: task #59's leg exists because "the broker used to
    /// make ~60 REST round trips per cycle and then infer which instance and which window each
    /// reply described; #52, #53, #46 and the 6x error in the #33 carrier-rate feed were all
    /// that inference going wrong." /get_beam_cube's ring is 8 windows ~ 8 seconds of
    /// tolerance; any consumer stall longer than that loses those windows permanently, and
    /// addressability makes the loss VISIBLE, never recoverable. So the archive path is a
    /// push: a completed window is packed and bufferSent, and the address (chain, instance,
    /// absolute window index, wstart) travels WITH the data on the F-engine's own clock.
    /// The REST endpoint stays, demoted to interactive inspection.
    ///
    /// ⚠️ BACKPRESSURE MUST DROP, NEVER BLOCK (KV, 2026-09-04). This stage sits in the
    /// real-time path; waiting for an empty frame would stall the tracker to protect an
    /// archive, which is the wrong way round. So the acquire is non-blocking (is_frame_empty
    /// first -- safe because this stage is the buffer's only producer) and a full buffer
    /// increments `_cube_dropped`.
    ///
    /// ⚡ AND THE DROP COUNT IS CARRIED IN THE NEXT FRAME. An archive that silently loses
    /// windows is worse than one that loses them loudly: "we have every record" then becomes a
    /// claim nobody can check. The counter is cumulative since start, so a reader differences
    /// consecutive frames to learn exactly how many windows fell in that gap, and a gap in the
    /// window index that is NOT matched by a rise in the counter means the loss happened
    /// somewhere else (bufferSend's drop_frames, the far side) -- a distinction worth having.
    Buffer* _cube_out_buf = nullptr;
    int _cube_out_id = 0;
    int _cube_max_bins = 0;               ///< frame is sized for this many bins, zero-padded
    int _cube_max_prn = 0;                ///< and this many PRN slots -- UNIFORM across senders
    int64_t _cube_dropped = 0;            ///< windows lost to a full buffer, cumulative
    std::string _cube_chain;              ///< "<host>/<stage>" stamped in every frame
    int _cube_gpu = -1;                   ///< which GPU, for the reader's convenience
    /// Pack one completed window into `_cube_out_buf` and mark it full. Caller holds _cube_mtx.
    void emit_cube_window(const CubeWindow& C);

    /// Accumulate into the window owning `wstart`, opening/clearing the ring slot on a
    /// boundary crossing. Caller holds _cube_mtx.
    CubeWindow& cube_window_for(int64_t wstart);
    void beam_cube_callback(kotekan::connectionInstance& conn);
    /// Accumulate one record's channels into the window that owns `wstart`, opening/clearing
    /// the ring slot on a boundary crossing. Caller holds _spec_mtx.
    SpecWindow& spec_window_for(int64_t wstart);
    void spectrum_callback(kotekan::connectionInstance& conn);
    void set_elem_gain_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    /// #102: per-satellite geometry for the element steering (POST {"<prn>": [az_deg, el_deg]}).
    void set_sat_geometry_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    void set_elem_sum_adapt_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    void set_elem_sum_shared_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    /// #154: {"ref": [[re, im], ...] ([] clears), "mode": "off|log|live", "slew_deg_s": x}, every
    /// field optional; staged, applied at the next consensus update.
    void set_elem_sum_shared_ref_callback(kotekan::connectionInstance& conn,
                                          nlohmann::json& request);
    void get_elem_cal_callback(kotekan::connectionInstance& conn);
    void set_reference_element_callback(kotekan::connectionInstance& conn,
                                        nlohmann::json& request);

    /// LIVE SLOT MEMBERSHIP (docs/CHORD_LIVE_PRN_RECONFIG.md). Reconcile @c _prns against the
    /// PRN the PRODUCER stamped into this frame's @ref gnss_gpu::PrnCtl, and cold-reset every
    /// per-slot accumulator belonging to a slot whose satellite changed.
    ///
    /// ⚠️ THIS STAGE FOLLOWS THE FRAME; IT IS NOT RECONFIGURED. It could have grown its own
    /// /set_prns endpoint to be pushed in step with the producer's, and that would have been
    /// two copies of slot->PRN with no interlock -- the same shape as the config-vs-sky
    /// divergence this whole mechanism exists to end. The producer owns membership, the
    /// identity rides the data, and a frame that straddles a swap labels itself correctly with
    /// no coordination at all. Returns the number of slots that changed (0 in steady state).
    int follow_frame_prns(const void* pctl, int n_prn);

    // ── BRIGHT-SATELLITE PROJECTION (gnssProjSubspace.hpp; PROJECTION_PLAN.md phase 1) ────
    /// v' = v - Q (Q^H v) per channel, on every row of every slot that is not itself a source.
    /// mode 0 = OFF (default). 1 = SHADOW: the projected prompt feeds a second per-PRN learner
    /// (_cal_proj) and the capture diagnostics only; nothing live changes. 2 = LIVE: the rows
    /// are projected IN PLACE in the input frame before anything reads them (this stage is the
    /// frame's only consumer), so the combine, the taps, the cube, the element blocks and
    /// everything downstream see projected data. Config elem_proj_mode, live POST
    /// /set_elem_proj. Sources, in order: this chain's own slots inside elem_proj_deg of
    /// boresight (their prompt row, zero latency); the sibling chains' rows through the
    /// process-wide board (same GPU, same channels, matched by freq_id); the top eigenvector(s)
    /// of the probe rows' covariance (any emitter, named or not). A slot with no geometry for
    /// elem_proj_probe_since_s after it started running is a probe (geometry is posted only
    /// above the horizon), unless the broker names them in "_probes".
    std::atomic<int> _proj_mode{0};
    std::atomic<double> _proj_deg{4.0};
    std::atomic<int> _proj_rank_max{2};
    std::atomic<double> _proj_max_age_s{2.0};
    std::atomic<double> _proj_probe_frac_min{0.5};
    int _proj_kmax_alloc = 2;                 ///< basis columns allocated (config elem_proj_rank_max)
    /// Covariance horizons. The own-row tracker accumulates in the SOURCE'S STEERED frame (its
    /// geometric phase ramp removed), so it can average for seconds without smearing the
    /// direction as the satellite moves -- the offline result: a re-steered 10-window mean
    /// nulls -35..-48 dB where a 1-window raw vector gives -30..-34. The probe stack has no
    /// geometry to remove (its emitter is unnamed), so it stays short.
    double _proj_tau_s = 4.0;
    double _proj_probe_tau_s = 1.0;
    std::vector<gnss::ElemSteer::cf> _proj_steer; ///< [kmax][n_chan][n_elem] the sources' steering
    double _proj_probe_since_s = 90.0;
    double _bore_az_deg = 180.0, _bore_el_deg = 81.41;
    std::string _proj_group;                  ///< board key: the GPU instance ("gnss0")
    char _proj_sys = '?';                     ///< constellation letter of this chain's PRNs
    std::vector<int> _proj_fids;              ///< [n_chan] freq_id per channel (channel_ids)
    bool _proj_ready = false;
    uint8_t _proj_nchan_warned = 0;
    std::vector<std::unique_ptr<gnss::ProjSubspace>> _proj_own; ///< per slot, while it is a source
    gnss::ProjSubspace _proj_probe;           ///< the probe-row stack
    std::vector<gnss::ProjBasis> _proj_Q;     ///< [n_chan] the basis in force this record
    std::vector<uint8_t> _proj_isB;           ///< [n_prn] slot is a source this record
    std::vector<uint8_t> _proj_isProbe;       ///< [n_prn] slot fed the probe stack this record
    std::vector<uint8_t> _slot_was_run;       ///< [n_prn] run flag of the previous record
    std::vector<double> _slot_run_since;      ///< [n_prn] steady time the slot started running
    std::vector<uint8_t> _slot_probe_broker;  ///< [n_prn] named a probe by the broker (under _steer_mtx)
    std::atomic<bool> _probes_from_broker{false};
    int64_t _proj_wstart_prev = 0;
    int _proj_k_rec = 0;                      ///< max k over channels this record (0 = inert)
    uint64_t _proj_solve_ctr = 0;
    bool _proj_pub_own = false, _proj_pub_probe = false;
    std::vector<gnss::ElemCal> _cal_proj;     ///< per slot: the learner fed the PROJECTED prompt
    std::vector<double> _cap_plain, _cap_proj, _b_cos2, _sim_pp; ///< per slot (-1 = not measured)
    std::vector<std::complex<double>> _g_proj;    ///< [n_elem] projected, steered, channel-summed prompt
    std::vector<std::complex<double>> _v_scratch; ///< [n_elem]
    std::vector<gnss::ElemSteer::cf> _proj_steer_all; ///< [n_prn][n_chan][n_elem] every steered slot's table this record
    std::vector<uint8_t> _proj_steered;           ///< [n_prn] slot had a fresh table this record
    std::string _proj_owner_probe;                ///< unique_name + "/probe" (the board owner of the probe stack)
    std::vector<int> _proj_sig;                   ///< signature of the source set, for change detection
    std::vector<uint8_t> _proj_probe_used;        ///< [n_chan] a probe direction was used this record
    const double* _proj_corr_rec = nullptr;       ///< this record's raw corr rows (proj_identify)
    // Served state: written by main_thread, read by REST without a lock (torn reads of a
    // diagnostic double are acceptable; a lock on the per-record path is not).
    std::vector<double> _proj_probe_frac;     ///< [n_chan] probe-stack component-0 energy fraction
    std::vector<uint8_t> _proj_probe_on;      ///< [n_chan] probe trigger latched (hysteresis)
    double _proj_log_t = -1.0e18;             ///< steady time of the last source-change log line
    std::vector<int> _proj_k_ch;              ///< [n_chan] k in force
    std::vector<int> _proj_src_ch;            ///< [n_chan] bitmask: 1 own row, 2 sibling, 4 probe stack
    double _proj_us = 0.0;                    ///< EMA of the per-record projection CPU time
    uint64_t _proj_active_records = 0;
    std::string _proj_desc;                   ///< the current sources (log/REST); under _proj_mtx
    std::mutex _proj_mtx;
    void proj_prepare_record(const double* corr, const void* pctl_rec, int n_chan, int n_e,
                             int64_t wstart, double utc, double now_s);
    void proj_reset_slot(size_t p, double now_s);
    void proj_slot_inplace(double* corr_rw, const void* pctl_slot, int n_chan, int n_e,
                           int n_rows);
    void proj_slot_shadow(const double* corr, const void* pctl_slot, int n_chan, int n_e,
                          bool steered);
    void proj_slot_diag(size_t p, const void* pctl_slot, int n_chan, int n_e, bool steered);
    /// Which own slot a raw-frame direction IS: among this record's raw prompt rows on the
    /// channel, the one lying along q (cos^2 > 0.5) when it is the only one, or the one 10x
    /// brighter (cos^2 x |row|^2) than the next when several do. -1 = none: nothing of ours
    /// along it, or several rows along it with no dominant one -- the signature of a common
    /// interferer captured in every victim's row, i.e. an emitter we do not track. @p n_along
    /// receives the count of rows along q.
    int proj_identify(const std::complex<double>* q, int ch, int n_e, const void* pctl_rec,
                      int* n_along);
    std::vector<uint8_t> _proj_dropped; ///< [n_prn] a probe direction was this steered slot's own signature and was dropped this record
    void set_elem_proj_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
};

#endif
