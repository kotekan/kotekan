#ifndef GNSS_GPU_RECORD_ASSEMBLE_HPP
#define GNSS_GPU_RECORD_ASSEMBLE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssElemCal.hpp"
#include "gnssElemSteer.hpp"
#include "restServer.hpp"
#include "json.hpp"    // nlohmann::json for the set_elem_gain POST

#include <complex>
#include <mutex>
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
    /// Channels per output subband bin. 0 (default) = no binning, one bin per channel: the
    /// finest cube the instrument can produce. Binning trades frequency resolution for archive
    /// volume, which is the binding constraint here -- NOT memory, and not compute.
    int _cube_bin_width = 0;
    int _cube_bins = 0;                    ///< derived: number of subband bins
    std::mutex _cube_mtx;                  ///< guards _cube_* between main_thread and REST
    /// [n_prn * _cube_bins * n_elem] running SUM of |A_e,c|^2, and [n_prn * _cube_bins] the
    /// number of (record, channel) terms behind each bin. RESET ON READ: a poll returns exactly
    /// the interval since the previous poll, with the weight needed to combine intervals
    /// offline by addition. Unlike the spectrum ring this needs no cross-instance window
    /// alignment -- it carries no phase, so there is nothing for a misaligned window to
    /// decohere; only the weights have to be honest, and they are reported.
    std::vector<double> _cube_p2;
    std::vector<double> _cube_w;
    std::vector<double> _cube_t0;          ///< [1] wall clock at the start of the open interval
    void beam_cube_callback(kotekan::connectionInstance& conn);
    /// Accumulate one record's channels into the window that owns `wstart`, opening/clearing
    /// the ring slot on a boundary crossing. Caller holds _spec_mtx.
    SpecWindow& spec_window_for(int64_t wstart);
    void spectrum_callback(kotekan::connectionInstance& conn);
    void set_elem_gain_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    /// #102: per-satellite geometry for the element steering (POST {"<prn>": [az_deg, el_deg]}).
    void set_sat_geometry_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
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
};

#endif
