#ifndef GNSS_GPU_RECORD_ASSEMBLE_HPP
#define GNSS_GPU_RECORD_ASSEMBLE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssElemCal.hpp"
#include "gnssElemSteer.hpp"
#include "gnssLaneDirection.hpp" // for LaneDirection, InstDelay (#190)
#include "gnssProjSubspace.hpp"
#include "gnssSharedPin.hpp"     // for ReseedState
#include "prometheusMetrics.hpp" // for Gauge
#include "restServer.hpp"

#include "json.hpp" // nlohmann::json for the REST POST callbacks

#include <atomic>
#include <complex>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

/**
 * @class GnssGpuRecordAssemble
 * @brief Host tail of the GPU tracking chain: gnssGpuChain frames -> tracker records.
 *
 * Consumes the cudaProcess output (a control block and raw per-channel E/P/L correlations,
 * laid out in gnssGpuChain.hpp). For each PRN it sums the correlations over the PRN's covering
 * channel mask, integrates and applies the carrier-NCO phase (the phase state lives here; the
 * slope f_nco comes in the control block), and writes the gnssRecord.hpp record that the
 * combiner, broker and viewer read. Emits one out_buf frame per record, with the record's
 * start sample (wstart) in GnssChanMetadata::sample_seq.
 *
 * @conf in_buf   gnssGpuChain frames from the cudaProcess chain
 * @conf out_buf  tracker records: n_prn * record_stride floats, plus the per-channel block
 *                with chan_export
 * @conf prns     initial slot -> PRN map; its length must equal the frame's n_prn
 * @conf sample_rate  rate of the wstart sample counter, Hz
 * @conf record_cp_mod_chips  Float (default 0 = off). Reduce record slot 2 (the code-phase
 *                argument, float32) mod this many chips: the primary code length on a chain
 *                whose baked code is long enough to cost float32 precision (GLO_L2OC_P_OC).
 */
class GnssGpuRecordAssemble : public kotekan::Stage {
public:
    GnssGpuRecordAssemble(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
    /// Slot -> PRN, seeded from config and then taken from each frame (@ref follow_frame_prns),
    /// which is authoritative after a live swap.
    std::vector<int> _prns;
    double _sample_rate;
    /// Slot 2's modulus (chips), 0 = unreduced; see record_cp_mod_chips.
    double _rec_cp_mod_chips = 0.0;

    /// Element axis (>= 1).
    int _n_elements = 0;
    /// The element the record header's correlation slots carry: the broker's loop reference.
    int _reference_element = 0;
    std::vector<uint8_t> _elem_prev_ok; ///< element-cal continuity, independent of the carrier's
    std::vector<double> _fnco_prev;     ///< previous record's f_nco; the NCO advance over
                                        ///< [t_prev, t_now] uses the mean of it and this f_nco
    /// Self-calibrated element sum (gnssElemCal.hpp). Once a PRN's cal is warm (~3 tau) the
    /// header correlation slots carry the calibrated weighted mean over the elements: the
    /// reference element's phase and single-element scale, with per-record SNR up
    /// ~sqrt(N_healthy). Until then the header is the bare reference element.
    bool _elem_sum = false;
    double _elem_sum_tau_s = 0.5; ///< cal EMA time constant (s), short enough to follow the
                                  ///< inter-element fringe rotation as a satellite transits
    // Element steering (gnssElemSteer.hpp).
    gnss::ElemSteer _steer;          ///< per-(slot, channel, element) geometric phasors
    std::mutex _steer_mtx;           ///< guards _steer (REST vs main)
    double _elem_sum_min_w = 0.02;   ///< the reference element stays the phase anchor while its
                                     ///< weight is at least this fraction of the strongest's
    std::vector<gnss::ElemCal> _cal; ///< per PRN slot
    /// Hold vs adapt (config elem_sum_adapt, POST /set_elem_sum_adapt). With adapt false every
    /// PRN's live weights are held and _cal_shadow learns instead; _cal_sim[p] is the shadow's
    /// normalised |<held, shadow>|^2 (the capture detector), -1 until both are warm.
    std::atomic<bool> _elem_adapt{true};
    std::vector<gnss::ElemCal> _cal_shadow;
    std::vector<double> _cal_sim;
    /// Shared instrument model (config elem_sum_shared, POST /set_elem_sum_shared). With
    /// geometry steered, every satellite sees the same per-element instrument, but only within
    /// a polarisation: between a dish's two feeds the phase is direction dependent. So the
    /// model is one per-element gain per pol (_g_shared, each half normalised to sum|g| = 1, a
    /// slow consensus of the per-PRN shadow cals) plus one complex inter-pol coefficient per
    /// PRN (_pol_c, a fast EMA). A bright satellite captures a per-PRN learner in seconds but
    /// barely moves the consensus. Shared implies held (_elem_adapt = false): the live weights
    /// are rebuilt from the model every record.
    std::atomic<bool> _elem_shared{false};
    double _elem_shared_tau_s = 300.0;           ///< consensus EMA, s (slower than any transit)
    double _elem_pol_tau_s = 3.0;                ///< per-PRN inter-pol coefficient EMA, s
    std::vector<std::complex<double>> _g_shared; ///< [n_elem]: pol-0 half then pol-1 half
    bool _g_shared_warm = false;
    uint8_t _g_shared_collapsed = 0; ///< last consensus refused (one element > 50%)
    int _g_shared_n = 0;             ///< PRNs that fed the last consensus
    double _g_shared_t = 0.0;        ///< steady time of the last consensus pass
    /// Transit freeze. Near boresight a bright satellite's leakage captures every weak
    /// satellite's per-PRN learner, and a consensus of captured learners is a captured model.
    /// While the broker's pooled (all-constellation) nearest-to-boresight separation is inside
    /// _elem_shared_freeze_deg, and for _elem_shared_freeze_hold_s after, neither the model nor
    /// any inter-pol coefficient learns. A post older than 120 s arms no new freeze.
    double _elem_shared_freeze_deg = 6.0;
    double _elem_shared_freeze_hold_s = 60.0;
    /// No first model while the nearest satellite is inside this wider cone: a model formed at
    /// 5-8 deg learns the leakage as its shape.
    double _elem_shared_form_deg = 8.0;
    // Per-channel noise weights for the cross-channel sums (gnssChanPower.hpp).
    bool _chan_norm = false;           ///< config chan_noise_norm
    std::vector<int> _chan_norm_elems; ///< config chan_noise_norm_elems: the frame inputs the
                                       ///< power is averaged over
    std::vector<double> _chan_w;       ///< [n_chan] this frame's weights (1 when off or unknown)
    std::vector<double> _chan_pw;      ///< [n_chan] scratch: the looked-up power
    std::vector<double> _chan_w_pub;   ///< the last applied weights, for /get_elem_cal
    std::mutex _chan_w_mtx;
    std::atomic<bool> _chan_norm_applied{false};
    std::atomic<long> _chan_norm_miss{0};      ///< frames without a complete power table
    std::atomic<double> _bore_sep_deg{1.0e9};  ///< last posted pooled boresight separation, deg
    std::atomic<double> _bore_post_t{-1.0e18}; ///< steady time of that post
    double _freeze_until = -1.0e18;            ///< main thread: frozen while now < this
    /// Own phase pin. The model's global phase per pol is pinned to the first healthy model
    /// (all elements, <ref, G> real positive), not to one element: an element's weight can
    /// collapse, and a pin on it then goes to noise independently on every instance, which
    /// cancels the instances in the fleet sum. The first model is pinned on the reference
    /// element, which is what makes the instances agree to begin with.
    std::vector<std::complex<double>> _g_pin_ref;
    bool _g_pin_ref_ok = false;
    /// Fleet reference F (config elem_sum_shared_ref, POST /set_elem_sum_shared_ref): one
    /// vector per band shared by every instance (python/scripts/gnss/elem_shared_ref.py), so a
    /// poor own pin cannot leave this instance's global phase arbitrary against the others.
    /// "live" pins each pol's <F, G> real positive, in full for a first model and slewed at
    /// _fleet_slew_rad_s for a warm one (gnssSharedPin.hpp); "off" ignores F. A half F does not
    /// describe (sim < _fleet_min_sim) keeps the own pin, which follows the model's phase in
    /// live mode so that falling back never steps it.
    std::vector<std::complex<double>> _g_fleet_ref; ///< [n_elem]; empty = no reference
    std::atomic<int> _fleet_mode{0};                ///< 0 off, 1 live
    std::atomic<bool> _fleet_ref_present{false};    ///< _g_fleet_ref non-empty (for REST)
    double _fleet_slew_rad_s = M_PI / 180.0;
    double _fleet_min_sim = 0.5; ///< a 16-element noise model scores ~0.2 against any F
    /// Per pol, after the last consensus update (diagnostics, read unlocked by REST):
    double _fleet_err_deg[2] = {0.0, 0.0}; ///< arg<F, G>: the offset still to remove
    double _fleet_sim[2] = {0.0, 0.0};     ///< |<F, G>| / (|F| |G|)
    bool _fleet_applied[2] = {false, false};
    bool _fleet_slewing[2] = {false, false}; ///< one WARN per slew episode, one when done
    /// A warm half below _fleet_min_sim for this long takes the reference's shape (live mode
    /// only). Once per episode, at most once per interval.
    double _fleet_reseed_s = 600.0;
    /// ... and only while the sibling pol matches F at this similarity or better: both pols off
    /// F at once means F is the stale one.
    double _fleet_reseed_sib_sim = 0.8;
    gnss::ReseedState _fleet_reseed[2];
    /// REST staging (under _gain_mtx), applied by fleet_ref_apply_pending in the next
    /// shared_consensus on the main thread, which runs only while elem_sum_shared is on.
    std::vector<std::complex<double>> _pending_fleet_ref;
    bool _pending_fleet_ref_set = false;
    int _pending_fleet_mode = -1;
    double _pending_fleet_slew_deg_s = -1.0;
    void fleet_ref_apply_pending();
    bool shared_frozen(double now_s);
    std::vector<std::complex<double>> _pol_num; ///< per PRN: EMA of B1 conj(B0)
    std::vector<double> _pol_den;               ///< per PRN: EMA of |B0|^2
    std::vector<double> _pol_warmth;            ///< per PRN: -> 1 with _elem_pol_tau_s
    std::vector<std::complex<double>> _pol_c;   ///< per PRN: the coefficient in force (diagnostic)
    std::vector<std::complex<double>> _w_scratch; ///< [n_elem] held weights being built
    void shared_reset_prn(size_t p);
    void shared_consensus(double now_s);
    void shared_hold(size_t p);
    void shared_pol_update(size_t p, const std::complex<double>* g_prompt, double dt_s);
    std::vector<gnss::ElemSteer::cf>
        _steer_buf; ///< this PRN's [n_chan][n_elem] phasors, copied under _steer_mtx
    uint8_t _steer_nchan_warned = 0;
    std::vector<uint8_t> _anchor_warned; ///< one WARN per PRN when the phase anchor leaves the
                                         ///< reference element; cleared when the cal is rebuilt
    /// Scratch, [ROWS_PLAIN][n_elem]: the per-antenna covering-mask sum, reused per PRN so the
    /// per-record path does not allocate.
    std::vector<std::complex<double>> _g_elem;

    // Carrier NCO state per PRN slot.
    std::vector<double> _phi;
    std::vector<double> _phi_cyc;      ///< NCO phase in cycles, unwrapped, for the commanded-phase
                                       ///< export; _phi takes the same steps, wrapped, in rad
    std::vector<double> _phi_cmd_prev; ///< previous record's commanded phase (cycles)
    std::vector<uint8_t> _phi_cmd_ok;
    std::vector<uint8_t> _fcar_prev_ok;
    std::vector<uint8_t> _a_prev_ok;
    std::vector<int64_t> _wstart_prev;

    /// Per-channel prompt spectrum, served on `<unique_name>/get_spectrum?window=N`: per PRN
    /// and covering channel, the window sum of the prompt with the header's element combine
    /// and the record's NCO rotation `rot`, for the broker's phase-slope delay fit. `rot` is
    /// one phase per record, so it stops the residual carrier winding the sum without touching
    /// the cross-channel phases. Needs `channel_ids`. The window index is floor(wstart /
    /// _spec_win_samples) on the F-engine sample clock, so every instance puts a record in the
    /// same window; a ring keeps recent windows and reads reset nothing.
    ///
    /// Comb export (gnssRecord.hpp's chan block): the same per-channel prompt, with early and
    /// late, written per record for fits across records. Off by default; needs channel_ids.
    bool _chan_export = false;
    std::vector<int> _spec_freq_ids; ///< [n_chan] F-engine freq_id per channel
    int64_t _spec_win_samples = 0;   ///< window length, samples
    /// One accumulated window. Window i lives in _spec_ring[i mod depth] until the ring wraps
    /// past it; the slot's own `idx` says which window it holds.
    struct SpecWindow {
        int64_t idx = -1;                   ///< window index, or -1 for an unused slot
        int64_t w0 = -1, w1 = -1;           ///< wstart of the first/last record in it
        std::vector<double> re, im, energy; ///< [n_prn * n_chan]
        std::vector<int> nrec;              ///< [n_prn]
        /// [n_prn] _phi[p] at the window's first record, published so a consumer can put
        /// windows on one phase reference; and the count of fresh acquisitions (reanchored == 1)
        /// in the window, across which no constant relates its records.
        std::vector<double> phi0;
        std::vector<int> nreanchor;
    };
    std::vector<SpecWindow> _spec_ring; ///< depth from config; index -> idx % depth
    int64_t _spec_max_idx = -1;         ///< newest index seen; complete windows are < this
    std::mutex _spec_mtx;               ///< guards _spec_* between main_thread and REST
    // Per-element complex gain prior from POST /set_elem_gain (e.g. an N2 eigenvector with the
    // sky removed): staged here by REST, seeded into every PRN's ElemCal by main_thread.
    std::mutex _gain_mtx; ///< guards every _pending_* field between REST and main_thread
    std::vector<std::complex<double>> _pending_gain;
    bool _pending_gain_set = false;
    /// Reference element staged by POST /set_reference_element, applied at the next frame
    /// boundary; -1 = nothing pending. Applying rebuilds every PRN's ElemCal cold (prior and
    /// learned gains are anchored to the old reference) and resets the shared model, so the
    /// header rides the bare new reference for ~3 tau.
    int _pending_ref = -1;
    std::vector<std::complex<double>> _spec_scratch; ///< [n_elem] per-channel cal-combine input

    // Beam cube: per-(PRN, subband bin, element) sums over a window.
    /// The element blocks sum over channels and the comb block over elements; the cube keeps
    /// both axes, from `corr` (unsteered, after any live projection), because the beam changes
    /// across a wide or BOC signal. Pushed to cube_buf, not written into the record. Each term
    /// is A_e,c = G_e,c / E_c: one replica is
    /// correlated against every element, so a common channel energy keeps the antenna ratios
    /// comparable. The incoherent sum still carries the noise pedestal; the reader debiases it
    /// from the probe PRNs, in the power domain.
    bool _cube_on = false;
    int _cube_bins = 0; ///< subband bins: one per channel

    /// The incoherent sum is the beam; the coherent sum (terms rotated by `rot`) keeps the arc
    /// for calibration, phased-array maps and delay. Its phase reference _phi[p] resets on a
    /// fresh acquisition and steps on a re-pin, so phi0 is published with it. Do not reference
    /// each window to its own first record instead: that gives every window its own arbitrary
    /// phase. Residual carrier, a limping carrier loop and symbol flips the replica does not
    /// remove all shrink the coherent sum as a weak beam would, so the window's coherence is
    /// |SUM coh|^2 / (w * SUM |A|^2): 1 when fully coherent, ~1/w when the arc was lost.
    /// Windows are indexed as for the spectrum ring.
    struct CubeWindow {
        int64_t idx = -1;                   ///< window index, or -1 for an unused slot
        int64_t w0 = -1, w1 = -1;           ///< wstart of the first/last record in it
        double utc0 = 0.0;                  ///< UTC of sample 0 as stamped on its records
        std::vector<double> coh_re, coh_im; ///< [n_prn * n_bin * n_elem] SUM A_e,c * rot
        std::vector<double> incoh;          ///< [n_prn * n_bin * n_elem] SUM |A_e,c|^2
        std::vector<double> w;              ///< [n_prn * n_bin] (record, channel) term count
        std::vector<double> energy;         ///< [n_prn * n_bin] SUM replica energy
        std::vector<int> nrec;              ///< [n_prn] records contributing
        std::vector<double> phi0;           ///< [n_prn] _phi[p] at this window's first record
        std::vector<int> nreanchor;         ///< [n_prn] fresh acquisitions (reanchored == 1)
        /// cube_window_for() calls into this window so far: one per running PRN per record,
        /// so not a record count. Used only by the window-length unit check there (a length
        /// given in hops rather than samples gives one record per window).
        int nrec_seen = 0;
    };
    std::vector<CubeWindow> _cube_ring; ///< depth from config; index -> idx % depth
    int64_t _cube_win_samples = 0;      ///< window length in F-engine samples
    int64_t _cube_max_idx = -1;         ///< newest index seen; complete windows are < this
    int _cube_singleton_windows = 0;    ///< consecutive windows that saw only one call
    bool _cube_win_warned = false;      ///< the unit-error warning fires once, not per record
    /// Push leg: each completed window is pushed to cube_buf with its address (chain, window
    /// index, wstart). It never blocks the tracker: if the next frame is not empty (safe to test,
    /// as this stage is the only producer) the window is dropped and counted in _cube_dropped,
    /// which every frame carries, so a gap in the window index without a rise in the count was
    /// not dropped here.
    Buffer* _cube_out_buf = nullptr;
    int _cube_out_id = 0;
    int _cube_max_bins = 0;    ///< frame is sized for this many bins, zero-padded
    int _cube_max_prn = 0;     ///< and this many PRN slots, the same on every sender
    int64_t _cube_dropped = 0; ///< windows lost to a full buffer, cumulative
    std::string _cube_chain;   ///< "<host>/<stage>" (or beam_cube_chain), stamped in every frame
    int _cube_gpu = -1;        ///< beam_cube_gpu, stamped in every frame for the reader
    /// Pack a completed window into `_cube_out_buf` and mark it full, or count it dropped if
    /// the next frame is not empty.
    void emit_cube_window(const CubeWindow& C);

    /// The ring slot for the window owning `wstart`, cleared when a new window takes it.
    /// Opening a later window completes the previous newest one, which is pushed to cube_buf.
    CubeWindow& cube_window_for(int64_t wstart);
    /// The ring slot for the window owning `wstart`, cleared when a new window takes it. Caller
    /// holds _spec_mtx.
    SpecWindow& spec_window_for(int64_t wstart);
    void spectrum_callback(kotekan::connectionInstance& conn);
    void set_elem_gain_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    /// Geometry for the element steering: POST {"<prn>": [az_deg, el_deg, az_rate_dps,
    /// el_rate_dps, t_utc], ...} (a bare [az_deg, el_deg] is held), with optional
    /// "_bore": [sep_deg, t_utc].
    void set_sat_geometry_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    void set_elem_sum_adapt_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    void set_elem_sum_shared_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    /// {"ref": [[re, im], ...] ([] clears), "mode": "off|live", "slew_deg_s": x}, every
    /// field optional; staged, applied at the next consensus update (about once a second, and
    /// only while elem_sum_shared is on).
    void set_elem_sum_shared_ref_callback(kotekan::connectionInstance& conn,
                                          nlohmann::json& request);
    void get_elem_cal_callback(kotekan::connectionInstance& conn);
    void set_reference_element_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    /// Live slot membership (docs/CHORD_LIVE_PRN_RECONFIG.md). Reconcile @c _prns with the
    /// PRNs the producer stamped into this frame's first-record @ref gnss_gpu::PrnCtl, and
    /// cold-reset every per-slot accumulator of a slot whose satellite changed. The producer
    /// owns slot membership, so this stage has no /set_prns of its own. Returns the number of
    /// slots that changed (0 in steady state).
    int follow_frame_prns(const void* pctl, int n_prn);

    // Bright-satellite projection (gnssProjSubspace.hpp).
    /// v' = v - Q (Q^H v) per channel, on every row of every slot that is not itself a source.
    /// Mode (elem_proj_mode, POST /set_elem_proj) 0 = off. 1 = shadow: the projected prompt
    /// feeds a second learner (_cal_proj) and the capture diagnostics only. 2 = live: rows are
    /// projected in place in the input frame (this stage is its only consumer) before anything
    /// reads them, so everything downstream sees projected data. Basis sources, in order: own
    /// slots inside elem_proj_deg of boresight (their prompt row); sibling chains' entries on
    /// the ProjBoard (same group, matched by freq_id); the top eigenvectors of the probe rows'
    /// covariance. A probe direction (own or a sibling's) is used only while the broker reports
    /// a satellite inside elem_proj_deg, and not when proj_identify names it as one steered
    /// slot's own signature. A slot that has run for _proj_probe_since_s and has no fresh
    /// geometry is a probe.
    std::atomic<int> _proj_mode{0};
    std::atomic<double> _proj_deg{4.0};
    std::atomic<int> _proj_rank_max{2};
    std::atomic<double> _proj_max_age_s{2.0};
    std::atomic<double> _proj_probe_frac_min{0.5};
    int _proj_kmax_alloc = 2; ///< basis columns allocated
    /// Covariance time constants. The own-row tracker accumulates in the source's steered
    /// frame (its geometric phase ramp removed), so it can average for seconds without
    /// smearing the direction as the satellite moves. The probe stack has no geometry to
    /// remove (its emitter is unnamed), so its time constant is short.
    double _proj_tau_s = 4.0;
    double _proj_probe_tau_s = 1.0;
    std::vector<gnss::ElemSteer::cf> _proj_steer; ///< [kmax][n_chan][n_elem] the sources' steering
    double _proj_probe_since_s = 90.0;
    double _bore_az_deg = 180.0, _bore_el_deg = 81.41;
    std::string _proj_group;     ///< board key: the GPU instance ("gnss0")
    char _proj_sys = '?';        ///< constellation letter of this chain's PRNs
    std::vector<int> _proj_fids; ///< [n_chan] freq_id per channel (channel_ids)
    // #190 lane directions: every running non-probe slot's averaged steered prompt row,
    // published per channel on the ProjBoard (src 2) for GnssN2Project. Config
    // elem_dir_publish off|all, elem_dir_tau_s (0.5), elem_dir_every records (4),
    // elem_dir_coh_min (0.05: ~3x the noise floor 1/n, so probe slots never publish). Owner
    // "<unique_name>/lane/<prn>"; retired when the slot
    // stops or changes PRN. Rows are taken after the in-place elem projection, so a
    // victim's row carries no bright-source leakage.
    bool _lane_on = false;
    int _lane_every = 4;
    double _lane_coh_min = 0.02;
    std::string _lane_group;
    std::vector<int> _lane_fids;
    gnss::LaneDirection _lane;
    std::vector<std::complex<double>> _lane_scratch;
    std::vector<double> _lane_t_prev; ///< [n_prn] utc of the last fed record, 0 = none
    std::vector<int> _lane_prn, _lane_pub, _lane_cnt;
    std::vector<std::string> _lane_owner;
    // lane_transfer_freq_ids: channels outside channel_ids where the same satellites carry
    // another signal (the Galileo E6 PRS lobe beside E6-B/C, GPS L2 P(Y) beside L2C). Each gets
    // the direction of the nearest channel published this round, turned through the slot's
    // geometric delay: q(f2) = q(f1) exp(-i sign 2 pi (f2 - f1) tau_e), under the same owner.
    // With lane_transfer_inst_slope (default true) the transfer also turns each element by its
    // instrumental delay (gnss::InstDelay), measured from adjacent published channels of every
    // slot and forgotten over lane_transfer_inst_tau_s (300 s); until that has averaged for
    // tau_s the transfer is geometric only.
    std::vector<int> _lane_xfer_fids;
    std::vector<char> _lane_ok;                     ///< [n_chan] published this round
    std::vector<std::complex<double>> _lane_pubdir; ///< [n_chan][n_elem]
    std::vector<gnss::ElemSteer::cf> _lane_shift;   ///< [n_elem]
    bool _inst_on = false;
    int _inst_df_fid = 0; ///< the channel spacing the estimate is fed at, freq_ids
    gnss::InstDelay _inst;
    double _inst_gauge_t = 0.0; ///< utc of the last gauge update
    kotekan::prometheus::Gauge* _m_inst_rms_ns = nullptr;
    kotekan::prometheus::Gauge* _m_inst_weight = nullptr;
    kotekan::prometheus::Gauge* _m_inst_warm = nullptr;
    bool _proj_ready = false;
    uint8_t _proj_nchan_warned = 0;
    std::vector<std::unique_ptr<gnss::ProjSubspace>> _proj_own; ///< per slot, while it is a source
    gnss::ProjSubspace _proj_probe;                             ///< the probe-row stack
    std::vector<gnss::ProjBasis> _proj_Q; ///< [n_chan] the basis in force this record
    std::vector<uint8_t> _proj_isB;       ///< [n_prn] slot is a source this record
    std::vector<uint8_t> _proj_isProbe;   ///< [n_prn] slot fed the probe stack this record
    std::vector<uint8_t> _slot_was_run;   ///< [n_prn] run flag of the previous record
    std::vector<double> _slot_run_since;  ///< [n_prn] steady time the slot started running
    int64_t _proj_wstart_prev = 0;
    int _proj_k_rec = 0; ///< max k over channels this record (0 = inert)
    uint64_t _proj_solve_ctr = 0;
    bool _proj_pub_own = false, _proj_pub_probe = false;
    std::vector<gnss::ElemCal> _cal_proj; ///< per slot: the learner fed the projected prompt
    std::vector<double> _cap_plain, _cap_proj, _b_cos2, _sim_pp; ///< per slot (-1 = not measured)
    std::vector<std::complex<double>>
        _g_proj; ///< [n_elem] projected, steered, channel-summed prompt
    std::vector<std::complex<double>> _v_scratch; ///< [n_elem]
    std::vector<uint8_t> _proj_steered;           ///< [n_prn] slot had a fresh table this record
    std::string _proj_owner_probe; ///< unique_name + "/probe" (the board owner of the probe stack)
    std::vector<int> _proj_sig;    ///< signature of the source set, for change detection
    std::vector<uint8_t> _proj_probe_used;  ///< [n_chan] a probe direction was used this record
    const double* _proj_corr_rec = nullptr; ///< this record's raw corr rows (proj_identify)
    // Served state: written by main_thread, read by REST without a lock (torn reads of a
    // diagnostic double are acceptable; a lock on the per-record path is not).
    std::vector<double> _proj_probe_frac; ///< [n_chan] probe-stack component-0 energy fraction
    std::vector<uint8_t> _proj_probe_on;  ///< [n_chan] probe trigger latched (hysteresis)
    double _proj_log_t = -1.0e18;         ///< steady time of the last source-change log line
    std::vector<int> _proj_k_ch;          ///< [n_chan] k in force
    std::vector<int> _proj_src_ch;        ///< [n_chan] bitmask: 1 own row, 2 sibling, 4 probe stack
    double _proj_us = 0.0;                ///< EMA of proj_prepare_record's elapsed time, us
    uint64_t _proj_active_records = 0;
    std::string _proj_desc; ///< the current sources (log/REST); under _proj_mtx
    std::mutex _proj_mtx;
    void proj_prepare_record(const double* corr, const void* pctl_rec, int n_chan, int n_e,
                             int64_t wstart, double utc, double now_s);
    void proj_reset_slot(size_t p, double now_s);
    void proj_slot_inplace(double* corr_rw, const void* pctl_slot, int n_chan, int n_e, int n_rows);
    void proj_slot_shadow(const double* corr, const void* pctl_slot, int n_chan, int n_e,
                          bool steered);
    void proj_slot_diag(size_t p, const void* pctl_slot, int n_chan, int n_e, bool steered);
    /// Which own slot a raw-frame direction is: among this record's raw prompt rows on the
    /// channel (running, non-probe slots), the one lying along q (cos^2 >= 0.5) when it is the
    /// only one, or the one 10x brighter (cos^2 x |row|^2) than the next when several do.
    /// -1 = none: nothing of ours along it, or several rows along it with no dominant one --
    /// the signature of a common interferer captured in every victim's row, i.e. an emitter we
    /// do not track. @p n_along receives the count of rows along q.
    int proj_identify(const std::complex<double>* q, int ch, int n_e, const void* pctl_rec,
                      int* n_along);
    std::vector<uint8_t> _proj_dropped; ///< [n_prn] a probe direction was this steered slot's own
                                        ///< signature and was dropped this record
    void set_elem_proj_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
};

#endif
