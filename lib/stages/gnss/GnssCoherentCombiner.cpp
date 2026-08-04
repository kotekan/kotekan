#include "GnssCoherentCombiner.hpp"

#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata (the record's absolute hop)
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "visUtil.hpp"          // for frameID

#include "gnssChannelizedDespread.hpp" // for overlay_wipe
#include "gnssOverlay.hpp"             // for overlay_by_name (the secondary-overlay registry)

#include "json.hpp"   // for json
#include <algorithm> // for max
#include <cmath>      // for hypot, sqrt
#include <functional> // for bind
#include <string>     // for string

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssCoherentCombiner);

GnssCoherentCombiner::GnssCoherentCombiner(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssCoherentCombiner::main_thread, this)) {
    in_bufs = get_buffer_array("in_bufs");
    for (Buffer* b : in_bufs)
        b->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // ELEMENT AXIS (CHORD). 0 = the single-antenna airspy layout, byte-for-byte; the per-element
    // path below is then skipped entirely and every accumulator/emit is unchanged.
    _n_elements = config.get_default<int>(unique_name, "n_elements", 0);
    _rec_stride = gnss::record_stride(_n_elements);

    // Records per frame: configured, else inferred from the first input frame size.
    _n_prn = config.get_default<int>(unique_name, "n_prn", 0);
    if (_n_prn <= 0 && !in_bufs.empty())
        _n_prn = in_bufs[0]->frame_size / (int)sizeof(float) / _rec_stride;
    // Record width is a C++ constant; the frames are sized in yaml (n_prn * record_floats *
    // sizeof_float) with nothing linking the two. A stale config under-sizes them -- and here it
    // is doubly nasty, because the inference above would then silently compute the WRONG n_prn.
    {
        const size_t need = (size_t)_n_prn * _rec_stride * sizeof(float);
        for (Buffer* b : in_bufs)
            if ((size_t)b->frame_size < need) {
                FATAL_ERROR("GnssCoherentCombiner: input frame {:d} B < {:d} PRN x {:d} floats "
                            "({:d} B). Bump 'record_floats' in the config to {:d}.",
                            (size_t)b->frame_size, _n_prn, _rec_stride, need, _rec_stride);
                return;
            }
        if ((size_t)out_buf->frame_size < need) {
            FATAL_ERROR("GnssCoherentCombiner: out_buf frame {:d} B < {:d} PRN x {:d} floats "
                        "({:d} B). Bump 'record_floats' in the config to {:d}.",
                        (size_t)out_buf->frame_size, _n_prn, _rec_stride, need, _rec_stride);
            return;
        }
    }

    _integration_length = std::max(1, config.get_default<int>(unique_name, "integration_length", 1));
    const std::string mode = config.get_default<std::string>(unique_name, "integration_mode", "block");
    _rolling = (mode == "rolling");
    // Rolling output cadence: every output_every records (default ~10 outputs per time
    // constant). Block mode emits once per integration_length records (ignores this).
    _emit_every = std::max(1, config.get_default<int>(unique_name, "output_every",
                                                      std::max(1, _integration_length / 10)));
    _navwipe_bit_records = config.get_default<int>(unique_name, "navwipe_bit_records", 0);
    // Known secondary overlay (the L5 Neuman-Hofman pilot overlay) -> deep coherent integration
    // by searching its alignment, instead of the squaring nav-bit estimate. Both buffer the
    // per-record A.
    // NOT mutually exclusive with navwipe_bit_records since S4 phase 2: setting BOTH selects
    // the COMPOSED wipe (overlay first, then the nav-bit wipe on the result), which is what a
    // data component carrying a secondary overlay needs -- GPS L5-I (CNAV under NH10) is the
    // only such chain here. A dataless pilot still sets the overlay alone.
    const std::string sec = config.get_default<std::string>(unique_name, "secondary_overlay", "");
    // TABLE-DRIVEN dispatch (gnssOverlay.cpp): one registry row per known name carries the
    // overlay's shape and generator, so adding a signal means adding a row, not a branch.
    // Per-signal commentary (search floors, the COHERENT length-1 trick, ...) lives with
    // the rows. An unknown (or empty) name keeps the historical fall-through: no overlay.
    if (const gnss::OverlayDescriptor* ov = gnss::overlay_by_name(sec)) {
        if (ov->per_prn) {
            // Per-PRN overlays (B1C/E5a/B2a/L1CO): cache the whole PRN range here and pick
            // by the slot's current PRN at wipe time (the search re-assigns channels).
            _l1co.resize((size_t)ov->n_prn);
            for (int prn = 1; prn <= ov->n_prn; ++prn)
                _l1co[(size_t)(prn - 1)] = ov->generate(prn);
        } else {
            _secondary = ov->generate(0);
        }
        // Documentary cross-check: the SignalDescriptor secondary_length fields must agree
        // with the registry (they change nothing at runtime -- this keeps them honest).
        for (const std::string& msg : gnss::overlay_registry_check())
            WARN("overlay registry/descriptor mismatch: {:s}", msg);
    }
    // deep_coherent: opt-in, because it is only correct when the DESPREAD REPLICA already
    // carries the secondary overlay. Config, not inference: the combiner does not see the
    // tracker's signal descriptor, and guessing wrong means summing records whose overlay signs
    // still alternate -- which cancels rather than integrates.
    _deep_plain = config.get_default<bool>(unique_name, "deep_coherent", false);
    // Hops per record frame, purely to turn sample_seq into the hop index the rest of the chain
    // speaks (see the hpp). Unset is not an error -- pow_fft_len ships alongside so the currency
    // is always self-declared.
    _fft_len = config.get_default<int>(unique_name, "fft_len", 0);
    _wipe_buffer = (_navwipe_bit_records > 0 || !_secondary.empty() || !_l1co.empty()
                    || _deep_plain);

    // Real-time cost instrumentation -- see the header note. The derived quantity is the DUTY
    // CYCLE: (d ingest_us + d emit_us) / (d records x record_period). < 1 keeps up; >= 1 is
    // saturated and no amount of valve_slack fixes it.
    {
        using namespace kotekan::prometheus;
        auto& m = Metrics::instance();
        _m_ingest_us = &m.add_counter("kotekan_gnss_combiner_ingest_time_us_total", unique_name);
        _m_emit_us = &m.add_counter("kotekan_gnss_combiner_emit_time_us_total", unique_name);
        _m_records = &m.add_counter("kotekan_gnss_combiner_records_total", unique_name);
        _m_emits = &m.add_counter("kotekan_gnss_combiner_emits_total", unique_name);
        _m_last_emit_us = &m.add_gauge("kotekan_gnss_combiner_last_emit_time_us", unique_name);
        _m_last_emit_prns = &m.add_gauge("kotekan_gnss_combiner_last_emit_prns", unique_name);
    }
    // Auto-coherence (default ON, the clock_profile philosophy: automatic unless overridden):
    // integration_length is a MAX -- at emit the deep wipe is also evaluated over trailing
    // sub-windows (full, 1/2, 1/4, ... octaves) and the best deep SNR wins, so the deep |A|
    // reads the longest span the clock (+ sky) actually COHERES over instead of a decohered
    // full-window average. The chosen span is exported as coherence_s via get_status -- a live
    // measurement of the receiver clock's usable coherence time.
    _auto_coherence = config.get_default<bool>(unique_name, "auto_coherence", true);
    // Shared-carrier-loop observable: measure the full-band cross-record phase walk. Data
    // signals need the bit-robust SQUARED product (+-1 symbol flips cancel; unambiguous to
    // +-1/(4*dt)); a dataless pilot (L2C CL) sets carrier_pilot: true for the unsquared
    // product -- twice the unaliased range, no squaring loss at low SNR.
    _carrier_pilot = config.get_default<bool>(unique_name, "carrier_pilot", false);
    if (_wipe_buffer) {
        _navbuf.assign(_n_prn, {});
        _navutc.assign(_n_prn, {});
        _navhead.assign(_n_prn, {});
        _navwipe.assign(_n_prn, {});
    }
    // PEEL DEPTH: buffer the residual prompt (record slots 20-23) and run the SAME deep estimator
    // on it at emit. Requires the wipe buffer (the deep estimator's machinery), so a peel chain
    // must configure whatever deep path it uses (navwipe_bit_records for GPS L1CA).
    _peel_depth = config.get_default<bool>(unique_name, "peel_depth", false);
    _bit_export = config.get_default<bool>(unique_name, "bit_export", false);
    _bit_pred_horizon_s = config.get_default<double>(unique_name, "bit_pred_horizon_s", 4.0);
    if (_bit_export) {
        _st_nav_obs.assign(_n_prn, {});
        _st_bit_pred.assign(_n_prn, {});
    }
    // UNCONDITIONAL: the agreement flag is WRITTEN in the wipe section, which runs on every
    // overlay chain regardless of bit_export -- allocating these under the guard segfaulted
    // the whole node at launch on the L5/L2C combiners (2026-07-26), the exact disease the
    // tracker's peel vectors had the day before (nco_ok, also launch-fatal). Rule: allocate
    // where the WRITER runs, not where the consumer is configured.
    _bp_veto.assign(_n_prn, 0);
    _bp_agree.assign(_n_prn, 1);
    if (_peel_depth) {
        _navbuf_res.assign(_n_prn, {});
        _navhead_res.assign(_n_prn, {});
        if (!_wipe_buffer)
            WARN("GnssCoherentCombiner: peel_depth set but no deep-wipe path configured "
                 "(navwipe_bit_records / secondary_overlay) -- peel depth will read 0");
    }

    // Per-record phase dump (debug instrumentation; see the header comment). Append mode so a
    // relaunch extends the record rather than destroying the episode it was launched to catch.
    const auto dump_prns =
        config.get_default<std::vector<int>>(unique_name, "phase_dump_prns", {});
    if (!dump_prns.empty()) {
        const std::string dump_path = config.get_default<std::string>(
            unique_name, "phase_dump_path", "/tmp/gnss_phase_dump.txt");
        _phase_dump = std::fopen(dump_path.c_str(), "a");
        if (_phase_dump) {
            _phase_dump_prn.assign(256, false);
            for (int prn : dump_prns)
                if (prn >= 0 && prn < 256)
                    _phase_dump_prn[prn] = true;
            INFO("phase dump: {:d} PRN(s) -> {:s}", (int)dump_prns.size(), dump_path);
        } else {
            WARN("phase dump: cannot open {:s}; disabled", dump_path);
        }
    }

    _st_prn.assign(_n_prn, 0);
    _st_amp.assign(_n_prn, 0.0f);
    _st_coh.assign(_n_prn, 0.0f);
    _st_deep.assign(_n_prn, 0.0f);
    _st_deep_snr.assign(_n_prn, 0.0f);
    _st_amp_snr.assign(_n_prn, 0.0f);
    _st_amp_dbi.assign(_n_prn, 0.0f);
    _st_nh_phase.assign(_n_prn, -1);
    _st_dop.assign(_n_prn, 0.0f);
    _st_cp.assign(_n_prn, 0.0f);
    _st_dll_disc.assign(_n_prn, 0.0f);
    _st_e_pow.assign(_n_prn, 0.0f);
    _st_p_pow.assign(_n_prn, 0.0f);
    _st_l_pow.assign(_n_prn, 0.0f);
    _st_nchan.assign(_n_prn, 0.0f);
    _st_rungs.assign(_n_prn, {});
    _st_head_frac.assign(_n_prn, 0.0f);
    _st_s4.assign(_n_prn, 0.0f);
    _st_s4_raw.assign(_n_prn, 0.0f);
    _st_sigma_phi.assign(_n_prn, 0.0f);
    _st_snr_q.assign(_n_prn, NAN);
    _st_car_resid.assign(_n_prn, 0.0f);
    _adr_cyc.assign(_n_prn, 0.0);
    _trim_cyc.assign(_n_prn, 0.0);
    _adr_cph_prev.assign(_n_prn, 0.0);
    _adr_rate.assign(_n_prn, 0.0);
    _adr_t0.assign(_n_prn, 0.0);
    _adr_arc.assign(_n_prn, 0);
    _adr_n.assign(_n_prn, 0);
    _adr_ok.assign(_n_prn, 0);
    _st_utc.assign(_n_prn, 0.0);
    _st_adr.assign(_n_prn, 0.0);
    _st_trim.assign(_n_prn, 0.0);
    _st_adr_lock.assign(_n_prn, 0.0);
    _st_adr_arc.assign(_n_prn, 0);
    _st_adr_n.assign(_n_prn, 0);
    _st_coh_s.assign(_n_prn, 0.0f);
    _st_deep_floor.assign(_n_prn, 0.0f);
    _st_deep_pow.assign(_n_prn, 0.0f);
    _st_peel_deep.assign(_n_prn, 0.0f);
    _st_peel_incoh.assign(_n_prn, 0.0f);
    _st_peel_deep_snr.assign(_n_prn, 0.0f);
    _dr_phase.assign(_n_prn, -1);
    _dr_utc.assign(_n_prn, 0.0);
    _dr_prn.assign(_n_prn, -1);
    _dr_rec_dt.assign(_n_prn, 0.0);
    _adr_vprev.assign(_n_prn, std::complex<double>(0.0, 0.0));
    _adr_vprev_ok.assign(_n_prn, 0);
    _adr_smooth = std::max(1, config.get_default<int>(unique_name, "adr_smooth_records", 1));
    _adr_blk_v.assign(_n_prn, std::complex<double>(0.0, 0.0));
    _adr_blk_dcmd.assign(_n_prn, 0.0);
    _adr_blk_n.assign(_n_prn, 0);
    _adr_blk_prev.assign(_n_prn, std::complex<double>(0.0, 0.0));
    _adr_blk_prev_ok.assign(_n_prn, 0);
    _adr_blk_prev_utc.assign(_n_prn, 0.0);
    _adr_blk_rate.assign(_n_prn, 0.0);
    _adr_blk_rate_ok.assign(_n_prn, 0);
    _st_deep_rec.assign(_n_prn, 0);

    // nh time-assist: only meaningful for a per-PRN overlay pilot (B1C/E5a/B2a: _l1co populated),
    // where the weak sats lose the long alignment search. Harmless (and off) for the shared-
    // overlay / navwipe combiners.
    _nh_assist = config.get_default<bool>(unique_name, "nh_assist", false) && !_l1co.empty();
    _nh_min_refs = config.get_default<int>(unique_name, "nh_min_refs", 3);
    _nh_search_chips = config.get_default<int>(unique_name, "nh_search_chips", 2);
    _nh_ref_margin = config.get_default<double>(unique_name, "nh_ref_margin", 2.0);
    _nh_hint.assign(_n_prn, -1);

    // PER-RECORD EXPORT (2026-08-04) -- the raw material for cross-node coherent combining.
    // One instance sees 7 of 106 channels, so its per-record SNR is ~8.8 dB below the fleet's,
    // and that per-record SNR is what caps the coherent span (8.18.2). Summing the per-record
    // COMPLEX prompts across instances -- not the window means, which the phase noise has already
    // destroyed -- recovers it. That combine cannot live inside a stage that sees one node, so
    // the stage's job is to publish the buffer it already keeps and let the broker do the sum.
    //
    // Exported as [hop, re, im, energy] per record, newest last:
    //   * hop is the EXACT alignment key -- every node's sample_seq descends from one F-engine
    //     counter, so equal hop is the same sky with no tolerance (the fleet DLL groups on it).
    //   * re/im are energy-NORMALIZED amplitudes, so they are directly comparable instance to
    //     instance; energy travels alongside because the ML combine weights by it (noise
    //     variance goes as 1/energy). Publishing the ratio alone would throw the weights away.
    // Off by default: this is ~100 records x n_prn of JSON, far too much for the status poll the
    // viewer runs, so it lives on its own endpoint and only exists when asked for.
    // PHASE-RATE SEARCH. Default OFF: it changes what deep_snr means (a max over rates rather
    // than a straight sum), so a config that has not opted in keeps the old, comparable number.
    _deep_rate = config.get_default<bool>(unique_name, "deep_rate_search", false);
    _deep_rate_min_q = config.get_default<double>(unique_name, "deep_rate_min_q", 10.0);
    _st_deep_rate.assign(_n_prn, 0.0f);
    _st_deep_rate_q.assign(_n_prn, 0.0f);

    _rec_export = config.get_default<int>(unique_name, "record_export", 0);
    if (_rec_export > 0) {
        _recex.assign(_n_prn, {});
        _st_recex.assign(_n_prn, {});
    }
}

GnssCoherentCombiner::~GnssCoherentCombiner() {
    if (_phase_dump)
        std::fclose(_phase_dump);
}

void GnssCoherentCombiner::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_status",
        std::bind(&GnssCoherentCombiner::get_status_callback, this, _1));
    if (_nh_assist)
        kotekan::restServer::instance().register_post_callback(
            unique_name + "/set_nh_hint",
            std::bind(&GnssCoherentCombiner::set_nh_hint_callback, this, _1, _2));
    if (_rec_export > 0)
        kotekan::restServer::instance().register_get_callback(
            unique_name + "/get_records",
            std::bind(&GnssCoherentCombiner::get_records_callback, this, _1));

    std::vector<frameID> in_ids;
    for (Buffer* b : in_bufs)
        in_ids.emplace_back(b);
    frameID out_id(out_buf);

    // Per-PRN integration accumulators. block: running SUMS over _integration_length records.
    // rolling: exponential moving AVERAGES (alpha = 1/_integration_length), bias-corrected at
    // emit by dividing by (1 - (1-alpha)^n_roll) so they read true running means from record 1.
    std::vector<double> acc_pow(_n_prn), acc_ar(_n_prn), acc_ai(_n_prn), acc_nchan(_n_prn);
    std::vector<double> acc_pow2(_n_prn); // <|A|^4>, for the noise-debiased incoherent significance
    std::vector<double> acc_epow(_n_prn), acc_lpow(_n_prn); // <|E|^2>, <|L|^2> for the DLL disc
    // <|A|^2> built EXACTLY like acc_epow/acc_lpow -- the raw per-record prompt power, no
    // overlay wipe and no head/tail split. acc_pow is the science amplitude and takes whichever
    // straddle-immune form is available (|v_cur|^2, or |head|^2+|tail|^2), which is right for
    // |A| and WRONG as the P in q = 2P/(E+L): the segment split alone costs a factor f^2+(1-f)^2
    // -- 0.5 at a mid-record boundary -- and would read as a collapsing peak. E and L are
    // unsegmented, so their P must be too. Publish-only; nothing else reads it.
    std::vector<double> acc_ppow(_n_prn);
    // Boundary fraction f = <head energy>/<prompt energy>: where the code-period boundary sits
    // inside the window (0/1 = edge, ~0.5 = the old bistable's null zone). Pure diagnostic.
    std::vector<double> acc_he(_n_prn), acc_pe(_n_prn);
    // Shared carrier loop: vector-averaged cross-record phase product per PRN over the emit
    // window. S += (A_k conj A_{k-1})^{2|1}; at emit resid_hz = arg(S)/{2|1}/(2 pi dt_med).
    std::vector<std::complex<double>> car_S(_n_prn);
    std::vector<double> car_dt(_n_prn);
    std::vector<int> car_n(_n_prn);
    std::vector<std::complex<double>> car_prev(_n_prn);
    std::vector<double> car_prev_utc(_n_prn, -1.0);
    std::vector<float> ref_prn(_n_prn), ref_dop(_n_prn), ref_cp(_n_prn);
    std::vector<double> ref_utc(_n_prn);
    int n_acc = 0;              // block: records in the current block
    // Per-antenna state (empty and untouched when n_elements == 0).
    _ge_r.assign((size_t)_n_elements, 0.0);
    _ge_i.assign((size_t)_n_elements, 0.0);
    _acc_epow_el.assign((size_t)_n_prn * _n_elements, 0.0);
    _acc_ear.assign((size_t)_n_prn * _n_elements, 0.0);
    _acc_eai.assign((size_t)_n_prn * _n_elements, 0.0);
    _alpha_el = 1.0 / (double)_integration_length;
    long long n_roll = 0;      // rolling: total records seen (for EMA bias correction)
    long long win_hop = -1;    // absolute hop index labelling the current E/P/L window
    int since_emit = 0;        // rolling: records since the last emit
    const double alpha = 1.0 / (double)_integration_length; // rolling EMA weight

    while (!stop_thread) {
        // Gather the i-th frame of every subband (same window, same PRN order).
        std::vector<float*> ins;
        bool stopping = false;
        for (size_t i = 0; i < in_bufs.size(); ++i) {
            float* in = (float*)in_bufs[i]->wait_for_full_frame(unique_name, in_ids[i]);
            if (in == nullptr) {
                stopping = true;
                break;
            }
            ins.push_back(in);
        }
        if (stopping)
            break;

        // ALIGN THE SUBBANDS BY THE RECORD'S OWN UTC, not by frame index.
        //
        // The gather above takes the i-th frame of every input, which is only correct while
        // every input drops identically. That held for one host's two GPUs until a voltage tap
        // dropped on one of them (kotekan_gnss_tap_dropped_frames_total is nonzero in normal
        // running), and it can never hold across hosts. A single lost frame then offsets one
        // subband by a record FOREVER, and the combine sums different epochs -- which does not
        // error, it just quietly stops being coherent and reads as "no improvement".
        //
        // Records carry their capture UTC at RECORD_UTC_SLOT, so alignment is available for
        // free: advance whichever stream lags the newest by more than half a record period.
        // Tolerance is derived from the observed spacing rather than assumed, because the
        // record period is 10.5 ms on CHORD and 1 ms on airspy.
        if (in_bufs.size() > 1) {
            const auto rec_utc = [this](const float* in) {
                return *reinterpret_cast<const double*>(in + RECORD_UTC_SLOT);
            };
            if (_rec_dt_est <= 0.0 && _prev_utc0 > 0.0) {
                const double d = rec_utc(ins[0]) - _prev_utc0;
                if (d > 0.0)
                    _rec_dt_est = d; // first positive gap: enough to bound "half a record"
            }
            if (_rec_dt_est > 0.0) {
                // Bounded: a stream more than a few frames behind is a real fault, not jitter,
                // and spinning here would stall every other subband behind it.
                for (int guard = 0; guard < 64 && !stopping; ++guard) {
                    double newest = rec_utc(ins[0]);
                    for (size_t i = 1; i < ins.size(); ++i)
                        newest = std::max(newest, rec_utc(ins[i]));
                    bool moved = false;
                    for (size_t i = 0; i < ins.size(); ++i) {
                        if (newest - rec_utc(ins[i]) > 0.5 * _rec_dt_est) {
                            in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);
                            float* nx =
                                (float*)in_bufs[i]->wait_for_full_frame(unique_name, in_ids[i]);
                            if (nx == nullptr) {
                                stopping = true;
                                break;
                            }
                            ins[i] = nx;
                            moved = true;
                            ++_realigns;
                        }
                    }
                    if (!moved)
                        break;
                }
                if (stopping)
                    break;
            }
            _prev_utc0 = rec_utc(ins[0]);
        }

        // ABSOLUTE HOP INDEX of this record's window, from the frame metadata the record
        // producer already stamps (GnssGpuRecordAssemble / GnssChannelizedTracker set
        // sample_seq = window_start). This is the fleet DLL's grouping key: every node's
        // sample_seq descends from the same F-engine fpga_seq_num, so equal hop IS the same sky
        // -- an integer match, where the record's capture UTC would need a tolerance. Read here
        // because the frames are released a few lines below. Input 0 only: the realign loop
        // above has already put every input on the same window.
        // THIS record's hop, kept alive past the block below: win_hop labels the emitted WINDOW
        // (newest record when rolling, first when blocking), while the per-record export needs
        // the hop of the record actually being buffered -- the same number only in rolling mode.
        long long cur_hop = -1;
        {
            const GnssChanMetadata* mi = get_gnss_chan_metadata(in_bufs[0], in_ids[0]);
            cur_hop = (mi && mi->sample_seq >= 0)
                          ? (_fft_len > 0 ? mi->sample_seq / _fft_len : (long long)mi->sample_seq)
                          : -1;
            if (_rolling || n_acc == 0) // rolling: the newest record, as ref_utc is; block: the
                win_hop = cur_hop;      // window's first, as ref_utc is
        }

        // Cost clock starts only AFTER wait_for_full_frame returns: blocking on input is IDLE,
        // not work, and counting it would make an under-loaded stage look saturated.
        const double _t_ingest0 = current_time();

        for (int p = 0; p < _n_prn; ++p) {
            // Sum the un-normalized correlation and replica energy across subbands
            // (the cross-channel coherent combine), then the full-band amplitude.
            double gr = 0.0, gi = 0.0, energy = 0.0, nchan = 0.0;
            double er = 0.0, ei = 0.0, e_energy = 0.0, lr = 0.0, li = 0.0, l_energy = 0.0;
            double ghr = 0.0, ghi = 0.0, h_energy = 0.0; // prompt HEAD segment (slots 16-18)
            const float* ref = ins[0] + (size_t)p * _rec_stride; // PRN/dop/cp/UTC reference
            if (_n_elements > 0) {
                std::fill(_ge_r.begin(), _ge_r.end(), 0.0);
                std::fill(_ge_i.begin(), _ge_i.end(), 0.0);
            }
            for (float* in : ins) {
                const float* rec = in + (size_t)p * _rec_stride;
                gr += rec[3];
                gi += rec[4];
                energy += rec[5];
                nchan += rec[6];
                // Early/Late correlators (gnssRecord.hpp): additive across subbands exactly
                // like the prompt -- sum raw G and energy, normalize full-band below. Only the
                // POWERS feed the DLL discriminator, so per-subband NCO phases are irrelevant.
                er += rec[gnss::REC_E_RE];
                ei += rec[gnss::REC_E_IM];
                e_energy += rec[gnss::REC_E_ENERGY];
                lr += rec[gnss::REC_L_RE];
                li += rec[gnss::REC_L_IM];
                l_energy += rec[gnss::REC_L_ENERGY];
                // Head segment: additive across subbands like the prompt it is a piece of.
                ghr += rec[gnss::REC_PH_RE];
                ghi += rec[gnss::REC_PH_IM];
                h_energy += rec[gnss::REC_PH_ENERGY];
                // PER-ANTENNA prompt, summed across subbands exactly like the reference
                // element's -- additive for the same reason (raw G sums; the shared energy
                // normalizes once, below).
                if (_n_elements > 0) {
                    const float* eb0 = in + gnss::elem_offset((int)p, 0, _n_elements);
                    for (int el = 0; el < _n_elements; ++el) {
                        const float* eb = eb0 + (size_t)el * gnss::ELEM_FLOATS;
                        _ge_r[el] += eb[gnss::ELEM_P_RE];
                        _ge_i[el] += eb[gnss::ELEM_P_IM];
                    }
                }
            }
            const double ar = energy > 0.0 ? gr / energy : 0.0;
            const double ai = energy > 0.0 ? gi / energy : 0.0;
            // PER-ANTENNA amplitude A_e = G_e / E, normalized by the SHARED replica energy --
            // the same E every antenna's correlation was formed against, so the ratios are
            // directly comparable across antennas. That comparability is the whole point: the
            // beam map is the relative response across dishes, and dividing each antenna by its
            // own energy would be dividing by the same number anyway.
            //
            // Accumulated the same way the reference element is (EMA when rolling, running sum
            // otherwise) so a per-antenna value means exactly what rec[3] means, one antenna at
            // a time. |A_e|^2 is incoherent -- it needs no per-antenna phase model and grows
            // like sqrt(K) -- while <A_e> keeps the phase for the complex gain.
            if (_n_elements > 0) {
                for (int el = 0; el < _n_elements; ++el) {
                    const double aer = energy > 0.0 ? _ge_r[el] / energy : 0.0;
                    const double aei = energy > 0.0 ? _ge_i[el] / energy : 0.0;
                    const double p2e = aer * aer + aei * aei;
                    const size_t k = (size_t)p * _n_elements + el;
                    if (_rolling) {
                        _acc_epow_el[k] += _alpha_el * (p2e - _acc_epow_el[k]);
                        _acc_ear[k] += _alpha_el * (aer - _acc_ear[k]);
                        _acc_eai[k] += _alpha_el * (aei - _acc_eai[k]);
                    } else {
                        _acc_epow_el[k] += p2e;
                        _acc_ear[k] += aer;
                        _acc_eai[k] += aei;
                    }
                }
            }
            // Head amplitude normalized by the TOTAL prompt energy, so head + tail == A
            // exactly and the segmented overlay wipe's units match the unsegmented one's.
            // A tracker that doesn't segment (CPU chain) writes PH == P -> head == A,
            // tail == 0, and every consumer reduces to the old whole-record behaviour.
            const double ahr = energy > 0.0 ? ghr / energy : 0.0;
            const double ahi = energy > 0.0 ? ghi / energy : 0.0;
            const double e2 = e_energy > 0.0 ? (er * er + ei * ei) / (e_energy * e_energy) : 0.0;
            const double l2 = l_energy > 0.0 ? (lr * lr + li * li) / (l_energy * l_energy) : 0.0;

            const double utc_p = *reinterpret_cast<const double*>(ref + RECORD_UTC_SLOT);
            const std::complex<double> A(ar, ai);
            // Overlay-wiped record v_cur: for an overlay pilot with a LOCKED dead-reckon anchor
            // (_dr_phase at _dr_utc), de-rotate the KNOWN secondary LINEARLY -- SEGMENTED at the
            // code-period boundary inside the record (head chip k, tail chip k+1, same convention
            // as overlay_wipe), so straddling records DON'T cancel. The result is overlay-free at
            // full amplitude for any boundary_f. Hoisted above p2 (2026-07-22) because it now feeds
            // THREE straddle-immune consumers: the incoherent power (p2 below), the carrier residual
            // fit (emit), and the ADR increment. Until an anchor is pinned (or after a re-seed),
            // v_ok is false and each consumer falls back to its raw/segment form.
            const int prn_p = (int)std::lround(ref[0]);
            const std::vector<int8_t>* ov_adr = overlay_for(prn_p);
            bool v_ok = false;
            std::complex<double> v_cur(0.0, 0.0);
            if (ov_adr && !ov_adr->empty() && energy > 0.0 && _dr_phase[p] >= 0
                && _dr_rec_dt[p] > 0.0 && _dr_prn[p] == prn_p) {
                const int L_ov = (int)ov_adr->size();
                const long long idx =
                    (long long)std::llround((utc_p - _dr_utc[p]) / _dr_rec_dt[p]);
                const int k = (int)(((_dr_phase[p] + idx) % L_ov + L_ov) % L_ov);
                const double c0 = (double)(*ov_adr)[(size_t)k];
                const double c1 = (double)(*ov_adr)[(size_t)((k + 1) % L_ov)];
                // head (ahr/ahi) belongs to chip k, tail (A - head) to chip k+1; a tracker
                // that doesn't segment writes head == A (tail 0) -> whole-record de-rotate.
                const std::complex<double> H =
                    h_energy > 0.0 ? std::complex<double>(ahr, ahi) : A;
                v_cur = H * c0 + (A - H) * c1;
                v_ok = true;
            }
            // Incoherent power (|A|^2 + |A|^4 moments -> noise-debiased amp_snr / s4). Straddle-
            // immune: the overlay-wiped |v_cur|^2 (full signal at any boundary_f) when anchored;
            // else the segment-power sum |head|^2 + |tail|^2, immune to the overlay SIGN so no
            // |2f-1| null (only a mild f^2+(1-f)^2 factor); else raw |A|^2. Was |A|^2, which nulls
            // at boundary_f=0.5 -> the moment debias booked the full/null bimodality as noise and
            // collapsed amp_snr/s4 (hence sig-gate flapping + drops) at every straddle.
            double p2;
            if (v_ok) {
                p2 = std::norm(v_cur);
            } else if (h_energy > 0.0) {
                const std::complex<double> H(ahr, ahi);
                p2 = std::norm(H) + std::norm(A - H);
            } else {
                p2 = ar * ar + ai * ai;
            }
            {
                // Full-band carrier phase walk vs the previous record (gap-gated: a valve drop
                // would alias the product). Squared for data signals, plain for pilots.
                const double dt_c = utc_p - car_prev_utc[p];
                if (car_prev_utc[p] > 0.0 && dt_c > 0.0 && dt_c < 0.050) {
                    std::complex<double> prod = A * std::conj(car_prev[p]);
                    if (!carrier_raw()) // squared: cancels nav bits AND unwiped overlay chips
                        prod *= prod;
                    car_S[p] += prod;
                    car_dt[p] += dt_c;
                    car_n[p] += 1;

                    // ---- ACCUMULATED CARRIER PHASE (ADR), the precise ranging observable.
                    // Phi_rx = Phi_cmd + arg(A)/2pi (gnssRecord.hpp). We accumulate it in
                    // INCREMENTS, because both halves are ambiguous on their own and both
                    // ambiguities die in the difference:
                    //   * the commanded half arrives mod 1 cycle -> continue it to the
                    //     integer nearest the reported Doppler's f*dt (a few cycles per
                    //     record; the prediction is good to ~1e-3, the margin is 0.5)
                    //   * the measured half is the SAME phase product the loop already
                    //     forms, so it inherits the bit/overlay robustness: squaring folds
                    //     the +-1 flip away and halves the unambiguous range to +-0.25
                    //     cycles -- still ~250x the true per-record increment (the carrier
                    //     loop holds the residual under ~1 Hz = 1e-3 cycles per ms record).
                    // Accumulated in DOUBLE: ADR reaches ~1e6 cycles in a pass, where a
                    // float32 mantissa quantizes to 0.06 cycles -- 1000x coarser than the
                    // measurement. This is why ADR ships via REST, not a record slot.
                    const double cph = ref[gnss::REC_CPHASE];
                    // K-record smoothed-block mode (the fold-walk fix, see the hpp note):
                    // overlay pilots with a live de-rotation anchor accumulate block vectors
                    // and take the cross-BLOCK product. Requires the anchor for EVERY record
                    // (an unanchored record would re-inject the overlay into the block sum),
                    // so anchor loss breaks the arc -- honest, like a gap.
                    const bool smooth_mode =
                        _adr_smooth > 1 && ov_adr && !ov_adr->empty();
                    if (energy <= 0.0) {
                        _adr_ok[p] = 0; // not despread this record (inactive PRN): no phase
                    } else if (smooth_mode && !v_ok) {
                        _adr_ok[p] = 0; // anchor lost/absent: no de-rotated vector to block
                        _adr_blk_prev_ok[p] = 0;
                        _adr_blk_rate_ok[p] = 0;
                        _adr_blk_n[p] = 0;
                    } else if (smooth_mode && _adr_ok[p]) {
                        _adr_blk_dcmd[p] += cph;
                        _trim_cyc[p] += ref[gnss::REC_TRIM_INC];
                        _adr_blk_v[p] += v_cur;
                        _adr_n[p] += 1;
                        if (++_adr_blk_n[p] >= _adr_smooth) {
                            if (_adr_blk_prev_ok[p]
                                && (utc_p - _adr_blk_prev_utc[p])
                                       < 2.5 * (double)_adr_smooth * dt_c) {
                                // RAW block product, deliberately MEMORYLESS (v3). The v2
                                // rate-predicted EMA unwrap was retracted after a 45-min
                                // soak read 67-726 m (r=+0.87 with loop residual): the L5
                                // loop residuals WANDER (the saga's flicker-FM), and every
                                // truth-crossing of +-0.5 cyc from the drifting prediction
                                // locked the unwrap onto the wrong branch -- the same
                                // wrong-integer absorbing state the slot-15 design note
                                // documents. Memoryless raw errors stay BOUNDED per event;
                                // K=10's +-50 Hz margin covers every observed residual
                                // median (<=27 Hz; only >50 Hz excursions can fold).
                                const double dres_blk =
                                    std::arg(_adr_blk_v[p] * std::conj(_adr_blk_prev[p]))
                                    / (2.0 * M_PI);
                                _adr_cyc[p] += _adr_blk_dcmd[p] - dres_blk;
                            }
                            else
                                _adr_blk_rate_ok[p] = 0; // no product -> rate not continuous
                            // (no-product case = the arc's FIRST block -- its dcmd only sets
                            // the arbitrary, per-arc-ambiguous origin; dropping it is a
                            // constant offset, not a discontinuity.)
                            _adr_blk_prev[p] = _adr_blk_v[p];
                            _adr_blk_prev_ok[p] = 1;
                            _adr_blk_prev_utc[p] = utc_p;
                            _adr_blk_v[p] = std::complex<double>(0.0, 0.0);
                            _adr_blk_dcmd[p] = 0.0;
                            _adr_blk_n[p] = 0;
                        }
                    } else if (smooth_mode) {
                        // Arc start, smoothed mode (v_ok is true here): seed the first block.
                        _adr_cyc[p] = 0.0;
                        _trim_cyc[p] = 0.0;
                        _adr_arc[p] += 1;
                        _adr_t0[p] = utc_p;
                        _adr_n[p] = 0;
                        _adr_ok[p] = 1;
                        _adr_blk_v[p] = v_cur;
                        _adr_blk_dcmd[p] = 0.0;
                        _adr_blk_n[p] = 1;
                        _adr_blk_prev_ok[p] = 0;
                        _adr_blk_rate_ok[p] = 0;
                    } else if (_adr_ok[p]) {
                        // The commanded-phase increment arrives ready-made from the tracker
                        // (record slot 15): bounded, float-exact, NOTHING to unwrap. The
                        // earlier mod-1-phase-plus-unwrap design is dead -- see gnssRecord.hpp
                        // for how it silently locked BeiDou C24 onto a wrong integer.
                        const double dcmd = cph;
                        // dres is SUBTRACTED: arg(A) is the despread residual, which lives in
                        // the r2c-flipped INTERNAL convention (like the NCO and the trim),
                        // while the commanded phase above is physical. Same flip, same reason.
                        // On-sky proof (2026-07-13): with dres added, every satellite's ADR
                        // rate was off by 2x its MEAN uncorrected carrier residual -- which is
                        // ~0 for a satellite whose carrier loop is working, and therefore
                        // invisible on all of them but one. PRN10's loop was railed at the
                        // 100 Hz trim clamp, carrying a persistent +12.9 Hz residual, and read
                        // +24.9 Hz off its predicted Doppler: 2 x 12.9. The same railed
                        // satellite exposed both sign conventions that six healthy ones hid.
                        double dres;
                        if (v_ok && _adr_vprev_ok[p]) {
                            // Overlay wiped LINEARLY: the plain product of two overlay-free
                            // records -> arg is the pure carrier increment, /1 (not /2). Same
                            // r2c flip / subtraction convention as the squared branch below.
                            const std::complex<double> vprod = v_cur * std::conj(_adr_vprev[p]);
                            dres = std::arg(vprod) / (2.0 * M_PI);
                        } else {
                            dres = std::arg(prod) / (2.0 * M_PI * (carrier_raw() ? 1.0 : 2.0));
                        }
                        _adr_cyc[p] += dcmd - dres;
                        // The commanded-trim increment (slot 19) rides the SAME arc
                        // lifecycle as the ADR it contaminates, so downstream can subtract
                        // trim_cycles from adr_cycles per arc with no bookkeeping.
                        _trim_cyc[p] += ref[gnss::REC_TRIM_INC];
                        _adr_n[p] += 1;
                    } else {
                        // Arc start: the phase is ambiguous by an integer number of cycles,
                        // so every arc begins at zero and downstream levels each arc on its
                        // own (the standard carrier-phase ambiguity, one per arc).
                        _adr_cyc[p] = 0.0;
                        _trim_cyc[p] = 0.0;
                        _adr_arc[p] += 1;
                        _adr_t0[p] = utc_p;
                        _adr_n[p] = 0;
                        _adr_ok[p] = 1;
                    }
                } else if (car_prev_utc[p] > 0.0 || dt_c <= 0.0) {
                    // A gap (valve drop, PRN dropped/re-seeded, records out of order) breaks
                    // phase continuity: we cannot know how many whole cycles went by
                    // unobserved. Break the arc -- a silent slip is far worse than an
                    // honest one.
                    _adr_ok[p] = 0;
                    _adr_blk_prev_ok[p] = 0; // block continuity broken with the arc
                    _adr_blk_rate_ok[p] = 0;
                    _adr_blk_n[p] = 0;
                }
                car_prev[p] = A;
                car_prev_utc[p] = utc_p;
                // Carry this record's overlay-corrected amplitude to the next record as the ADR
                // wipe's previous. The dt_c gate + _adr_ok above enforce continuity, so the
                // plain product only ever spans two contiguous records (a gap zeroes _adr_ok,
                // routing the next record through arc-start, which skips the product).
                _adr_vprev[p] = v_cur;
                _adr_vprev_ok[p] = v_ok ? 1 : 0;
            }
            // Per-record phase dump (debug): raw despread A + E/L powers, one line per record.
            // Enough to reconstruct the intra-window phase trajectory offline (arg A, plus the
            // commanded increment for ADR-style continuity) and the E-L code-offset signature.
            if (_phase_dump && energy > 0.0 && (int)ref[0] >= 0
                && (int)ref[0] < (int)_phase_dump_prn.size() && _phase_dump_prn[(int)ref[0]]) {
                std::fprintf(_phase_dump,
                             "%.6f %d %.6e %.6e %.6e %.6e %.6f %.3f %.3f %.6e %.6e\n", utc_p,
                             (int)ref[0], ar, ai, e2, l2, (double)ref[gnss::REC_CPHASE],
                             (double)ref[1], (double)ref[2], ahr, ahi);
            }
            if (_rolling) { // exponential moving average (no reset; integrates indefinitely)
                acc_pow[p] += alpha * (p2 - acc_pow[p]);
                acc_pow2[p] += alpha * (p2 * p2 - acc_pow2[p]);
                acc_epow[p] += alpha * (e2 - acc_epow[p]);
                acc_lpow[p] += alpha * (l2 - acc_lpow[p]);
                acc_ppow[p] += alpha * ((ar * ar + ai * ai) - acc_ppow[p]);
                acc_he[p] += alpha * (h_energy - acc_he[p]);
                acc_pe[p] += alpha * (energy - acc_pe[p]);
                acc_ar[p] += alpha * (ar - acc_ar[p]);
                acc_ai[p] += alpha * (ai - acc_ai[p]);
                acc_nchan[p] += alpha * (nchan - acc_nchan[p]);
                if (_wipe_buffer && energy > 0.0) {
                    // Sliding window of the last _integration_length recs (deep wipe). Records
                    // with NO despread (PRN inactive this window: broker drop/coast churn) are
                    // NOT buffered -- the wipe is UTC-indexed and gap-robust, so they become
                    // gaps exactly like valve drops. Buffering their exact zeros instead
                    // diluted the amplitude while contributing NOTHING to the orthogonal-noise
                    // estimate: a mostly-zero window reported a tiny deep |A| at an enormous
                    // fake significance (observed 500 sigma at |A|=0.04 on flapping CL sats).
                    _navbuf[p].emplace_back(ar, ai);
                    _navutc[p].push_back(utc_p);
                    // Head segment (0 when the tracker doesn't segment -> h_energy 0 -> treat
                    // the whole record as head so the segmented wipe reduces to unsegmented).
                    _navhead[p].emplace_back(h_energy > 0.0 ? ahr : ar,
                                             h_energy > 0.0 ? ahi : ai);
                    // Overlay-wiped record for the LINEAR carrier fit; 0 when unanchored (the fit
                    // skips zero-norm records) so no un-wiped overlay pollutes it.
                    _navwipe[p].push_back(v_ok ? v_cur : std::complex<double>(0.0, 0.0));
                    if (_peel_depth) { // residual prompt, same slide (see the block-mode note)
                        double rr = 0.0, ri = 0.0, rhr = 0.0, rhi = 0.0;
                        for (float* in2 : ins) {
                            const float* rc = in2 + (size_t)p * _rec_stride;
                            rr += rc[gnss::REC_RES_RE];
                            ri += rc[gnss::REC_RES_IM];
                            rhr += rc[gnss::REC_RES_PH_RE];
                            rhi += rc[gnss::REC_RES_PH_IM];
                        }
                        _navbuf_res[p].emplace_back(rr / energy, ri / energy);
                        _navhead_res[p].emplace_back(h_energy > 0.0 ? rhr / energy : rr / energy,
                                                     h_energy > 0.0 ? rhi / energy : ri / energy);
                    }
                    if ((int)_navbuf[p].size() > _integration_length) {
                        _navbuf[p].erase(_navbuf[p].begin());
                        _navutc[p].erase(_navutc[p].begin());
                        _navhead[p].erase(_navhead[p].begin());
                        _navwipe[p].erase(_navwipe[p].begin());
                        if (_peel_depth) {
                            _navbuf_res[p].erase(_navbuf_res[p].begin());
                            _navhead_res[p].erase(_navhead_res[p].begin());
                        }
                    }
                }
                if (_rec_export > 0 && energy > 0.0)
                    push_recex(p, cur_hop, ar, ai, energy);
                ref_prn[p] = ref[0]; // reference = the latest record (rolling has no block start)
                ref_dop[p] = ref[1];
                ref_cp[p] = ref[2];
                ref_utc[p] = utc_p;
            } else { // block: running sums, reset at emit
                acc_pow[p] += p2;
                acc_pow2[p] += p2 * p2;
                acc_epow[p] += e2;
                acc_lpow[p] += l2;
                acc_ppow[p] += ar * ar + ai * ai;
                acc_he[p] += h_energy;
                acc_pe[p] += energy;
                acc_ar[p] += ar;
                acc_ai[p] += ai;
                acc_nchan[p] += nchan;
                if (_wipe_buffer && energy > 0.0) { // per-record (A, UTC) for the deep wipe;
                    // zero-energy (inactive) records are gaps, not zeros -- see the rolling
                    // branch comment (fake-significance pathology otherwise).
                    _navbuf[p].emplace_back(ar, ai);
                    _navutc[p].push_back(utc_p);
                    _navhead[p].emplace_back(h_energy > 0.0 ? ahr : ar,
                                             h_energy > 0.0 ? ahi : ai);
                    _navwipe[p].push_back(v_ok ? v_cur : std::complex<double>(0.0, 0.0));
                    if (_peel_depth) {
                        // Residual prompt (slots 20-23), normalized by the SAME prompt energy so
                        // it lives in the same |A| units as _navbuf -- the depth ratio is then a
                        // pure dB with no scale to track. The residual head belongs to the same
                        // code-period segment as the prompt head, so the deep estimator's overlay
                        // wipe applies unchanged.
                        double rr = 0.0, ri = 0.0, rhr = 0.0, rhi = 0.0;
                        for (float* in2 : ins) {
                            const float* rc = in2 + (size_t)p * _rec_stride;
                            rr += rc[gnss::REC_RES_RE];
                            ri += rc[gnss::REC_RES_IM];
                            rhr += rc[gnss::REC_RES_PH_RE];
                            rhi += rc[gnss::REC_RES_PH_IM];
                        }
                        _navbuf_res[p].emplace_back(rr / energy, ri / energy);
                        _navhead_res[p].emplace_back(h_energy > 0.0 ? rhr / energy : rr / energy,
                                                     h_energy > 0.0 ? rhi / energy : ri / energy);
                    }
                }
                if (_rec_export > 0 && energy > 0.0)
                    push_recex(p, cur_hop, ar, ai, energy);
                if (n_acc == 0) { // window reference from the first record of the block
                    ref_prn[p] = ref[0];
                    ref_dop[p] = ref[1];
                    ref_cp[p] = ref[2];
                    ref_utc[p] = utc_p;
                }
            }
        }

        if (_m_ingest_us)
            _m_ingest_us->inc((uint64_t)std::max(0.0, (current_time() - _t_ingest0) * 1e6));
        if (_m_records)
            _m_records->inc(); // the DATA clock: records consumed (x record period = data time)

        for (size_t i = 0; i < in_bufs.size(); ++i)
            in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);

        if (_rolling) {
            ++n_roll;
            if (++since_emit < _emit_every)
                continue; // EMA keeps integrating; emit only every _emit_every records
            since_emit = 0;
        } else if (++n_acc < _integration_length) {
            continue; // keep accumulating
        }

        float* out = (float*)out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;
        // Again after the blocking wait: downstream backpressure is not this stage's cost.
        const double _t_emit0 = current_time();
        // block: divide sums by the block count. rolling: the EMA is already a mean, so just
        // undo the warm-up bias (1-(1-alpha)^n_roll) -> a true running mean from record 1.
        const double inv = _rolling ? 1.0 / (1.0 - std::pow(1.0 - alpha, (double)n_roll))
                                    : 1.0 / (double)n_acc;
        std::vector<float> p_pow_out(_n_prn, 0.0f); // fleet-DLL prompt power (see acc_ppow)
        std::vector<double> deep_snr(_n_prn, 0.0); // significance of the deep detection (for REST)
        std::vector<double> amp_snr(_n_prn, 0.0);  // noise-debiased incoherent significance (REST)
        std::vector<int> nh_phase(_n_prn, -1);     // secondary-overlay alignment found (L5; -1 = n/a)
        std::vector<double> amp_dbi(_n_prn, 0.0);  // noise-debiased (unbiased) signal amplitude
        std::vector<double> s4_raw(_n_prn, 0.0);   // amplitude scintillation index (raw)
        std::vector<double> sigma_phi(_n_prn, NAN); // carrier-phase jitter (rad); NaN = NOT MEASURED
                                                    // this emit (short window / no-wipe fallback /
                                                    // no carrier data). Overwritten with a real value
                                                    // only when carrier_resid_hz actually fits.
        std::vector<double> snr_q(_n_prn, NAN);     // modulation-immune per-record SNR (LINEAR pilots
                                                    // only); NaN = not available (data fit / no fit)
        std::vector<double> coh_s(_n_prn, 0.0);    // measured coherence: span of the winning window
        std::vector<int> deep_rec(_n_prn, 0);      // records in the winning deep window
        std::vector<double> deep_floor(_n_prn, 0.0); // rectification floor at the reported rung
        // EVERY rung, not only the winner (2026-08-04). deep_snr is a MAX over the ladder, and a
        // max cannot say WHY it is small: "the phase never cohered" (short rungs also at floor)
        // and "it cohered but no window cleared the bar" (short rungs high, long rungs collapsing)
        // are opposite diagnoses wanting opposite fixes. Costs nothing -- the rungs are already
        // computed and thrown away. (len, snr, span_s), longest-first as walked.
        std::vector<std::vector<std::array<float, 3>>> rungs(_n_prn);
        std::vector<double> deep_rate(_n_prn, 0.0);   // phase-rate removed before the deep sum
        std::vector<double> deep_rate_q(_n_prn, 0.0); // its peak/median -- the gate, and the SNR
        std::vector<double> peel_rsnr(_n_prn, 0.0);  // residual deep significance (peel_depth)
        std::vector<NavObs> nav_obs(_bit_export ? _n_prn : 0); // full-window bit decisions (P7a)
        // FIXED-WINDOW, noise-debiased coherent power (Hz) for the beam map: the coherent
        // twin of the incoherent x = s^2/N. Computed from the FULL window's wipe every emit
        // -- NO ladder, NO detection gate, so a pixel average over emits is UNBIASED at
        // arbitrarily weak signal (the map's radiometry rule: average the linear estimator
        // INCLUDING its noise; never threshold-then-average). E[snr^2] under noise is the
        // wipe's selection mean (overlay: max over n_align trials of ~Exp(2) -> 2(ln n +
        // gamma); navwipe: floor^2 + 1); subtracting it centers noise on ZERO, and the
        // map's below-horizon pedestal absorbs the approximation residue exactly as it
        // does x's clip bias. Coherent integration buys sqrt(T_coh/T_rec) sensitivity --
        // ~15 dB (GPS) / 12 (E1C) / 10 (B1C) deeper sidelobes per unit dwell.
        std::vector<double> deep_pow(_n_prn, 0.0);
        // A wiped rung whose SNR is indistinguishable from its NOISE floor must not be
        // reported as coherence: the wipe's free parameters rectify pure noise to a
        // window-length-dependent floor, so an unlocked sat otherwise reports coh = the
        // FULL window at deep ~= floor -- a convincing false lock (observed 2026-07-12
        // when the carrier loop briefly ran open). Rungs must clear FLOOR_MARGIN x floor
        // to be eligible; when none does, coherence_s = 0 (the honest "no coherent
        // detection") and the full-window numbers are reported for diagnostics only.
        constexpr double FLOOR_MARGIN = 2.0;
        // Auto-coherence ladder: deep-wipe window lengths to try -- the full buffer (always,
        // legacy behaviour) plus up to 4 octave-shorter trailing suffixes (>= min_len). Walked
        // longest-first; a shorter window must beat the incumbent deep SNR by >5%, so ties
        // integrate DEEP and the winner sits at the knee of the coherence curve.
        auto ladder = [&](size_t n, size_t min_len) {
            std::vector<size_t> lens;
            if (n > 0)
                lens.push_back(n);
            if (_auto_coherence)
                for (size_t l = n / 2; l >= std::max<size_t>(min_len, 2) && lens.size() < 5; l /= 2)
                    lens.push_back(l);
            return lens;
        };
        // DURATION of a trailing window of `len` records. _navutc holds record START times, so
        // first-to-last spans len-1 record periods and the window itself is ONE PERIOD LONGER.
        // coherence_s was reported as the bare span and so came out one record short -- 0.999 s
        // for a 1.000 s L1/L5 window, 3.98 s for L2C's 4.00 (20 ms records make the error 20x
        // more visible there, which is how it was noticed). The rec_dt and power-density code
        // below already used this exact len/(len-1) form; only the REPORTED number missed it.
        // Gap-robust: dropped records leave the spacing of those present unchanged.
        // Shortest ladder rung, in RECORDS, from a floor expressed in TIME (2026-07-27).
        // The hardcoded 64 below was a RECORD COUNT written for 1 ms records -- 64 ms at L1,
        // but 1.28 s at L2C's 20 ms records, where it also killed the third rung outright
        // (200/4 = 50 < 64). L2C was therefore offered only 4 s and 2 s windows, both of which
        // need better than 0.25 Hz of carrier: measured live, the two certified sats sat at
        // 0.064 and 0.181 Hz while EVERY uncertified one was >= 0.38 Hz, including the
        // strongest satellite on the band at 294 sigma. Strength cannot rescue a window that
        // nulls. Same bug class as buffer_depth and coherence_s: a COUNT where a TIME was
        // meant, on a node whose bands disagree about what a record is.
        const double rec_dt_floor_s = 0.064;
        auto min_rung = [&](const std::vector<double>& u, size_t hard_min) {
            size_t n = hard_min;
            if (u.size() > 1) {
                const double dt = (u.back() - u.front()) / (double)(u.size() - 1);
                if (dt > 0.0)
                    n = std::max<size_t>(2, (size_t)std::ceil(rec_dt_floor_s / dt));
            }
            return std::max<size_t>(hard_min, n);
        };
        // LINEAR PHASE-RATE SEARCH over one rung. Returns the rate (Hz) whose removal maximises
        // the coherent sum, and peak/median of the spectrum as the quality that gates it.
        //
        // Records are uniform in HOP, so (utc - utc0)/dt is an integer even across gaps: the
        // series lands on a uniform grid with holes, and a hole contributes exactly zero. dt is
        // the MINIMUM consecutive spacing, not the mean -- with gaps the mean is not the record
        // period, and a wrong dt rescales every frequency in the answer.
        auto rate_search = [](const std::vector<std::complex<double>>& x,
                              const std::vector<double>& u, double& best_f, double& q) {
            best_f = 0.0;
            q = 0.0;
            const size_t L = x.size();
            if (L < 8 || u.size() != L)
                return;
            double dt = 0.0;
            for (size_t k = 1; k < L; ++k) {
                const double d = u[k] - u[k - 1];
                if (d > 1e-9 && (dt == 0.0 || d < dt))
                    dt = d;
            }
            if (dt <= 0.0)
                return;
            std::vector<long long> m(L);
            long long span = 0;
            for (size_t k = 0; k < L; ++k) {
                m[k] = std::llround((u[k] - u[0]) / dt);
                span = std::max(span, m[k]);
            }
            // 4x oversampled so the true rate is never more than 1/8 bin from a sample; the
            // peak's own width is 1/(L dt), and losing half a bin costs ~4% of the amplitude.
            const size_t nb = (size_t)std::min<long long>(4 * (span + 1), 4096);
            if (nb < 8)
                return;
            std::vector<std::complex<double>> tw(nb); // integer-indexed: no trig in the inner loop
            for (size_t i = 0; i < nb; ++i)
                tw[i] = std::polar(1.0, -2.0 * M_PI * (double)i / (double)nb);
            std::vector<double> mag(nb, 0.0);
            for (size_t i = 0; i < nb; ++i) {
                std::complex<double> acc(0.0, 0.0);
                for (size_t k = 0; k < L; ++k)
                    acc += x[k] * tw[(size_t)(((long long)i * m[k]) % (long long)nb)];
                mag[i] = std::abs(acc);
            }
            const size_t ip = (size_t)(std::max_element(mag.begin(), mag.end()) - mag.begin());
            std::vector<double> srt(mag);
            std::nth_element(srt.begin(), srt.begin() + nb / 2, srt.end());
            const double med = srt[nb / 2];
            q = med > 0.0 ? mag[ip] / med : 0.0;
            // Bins run 0..nb-1 over one full record-rate period; fold the upper half negative so
            // the reported rate is the signed offset a carrier loop could actually consume.
            const double frac = (double)ip / (double)nb;
            best_f = (frac < 0.5 ? frac : frac - 1.0) / dt;
        };
        auto coh_span = [](const std::vector<double>& u, size_t len) {
            if (u.empty() || len < 2 || u.size() < len)
                return 0.0;
            return (u.back() - u[u.size() - len]) * (double)len / (double)(len - 1);
        };
        const double Keff = _rolling ? (double)_integration_length : (double)n_acc;
        for (int p = 0; p < _n_prn; ++p) {
            float* rec = out + (size_t)p * _rec_stride;
            for (int f = 0; f < _rec_stride; ++f) // whole record: element blocks too
                rec[f] = 0.0f;
            rec[0] = ref_prn[p];
            rec[1] = ref_dop[p];
            rec[2] = ref_cp[p];
            rec[3] = (float)std::sqrt(acc_pow[p] * inv);             // |A|_incoh = sqrt<|A|^2>
            rec[4] = (float)(acc_ar[p] * inv);                       // <A>.re (coherent mean)
            rec[5] = (float)(acc_ai[p] * inv);                       // <A>.im
            rec[6] = (float)std::hypot(acc_ar[p], acc_ai[p]) * inv;  // |<A>|_coh
            rec[7] = (float)(acc_nchan[p] * inv);                    // covering channels used
            // PER-ANTENNA BEAM MAP. Same `inv` normalization as the reference element above, so
            // CMB_ELEM_AMP_INCOH is directly comparable with rec[3] -- for a single-antenna
            // configuration with reference_element = e they are the same number.
            if (_n_elements > 0) {
                for (int el = 0; el < _n_elements; ++el) {
                    float* eb = rec + gnss::RECORD_FLOATS + (size_t)el * gnss::ELEM_FLOATS;
                    const size_t k = (size_t)p * _n_elements + el;
                    const double mr = _acc_ear[k] * inv, mi = _acc_eai[k] * inv;
                    eb[gnss::CMB_ELEM_AMP_INCOH] = (float)std::sqrt(_acc_epow_el[k] * inv);
                    eb[gnss::CMB_ELEM_MEAN_RE] = (float)mr;
                    eb[gnss::CMB_ELEM_MEAN_IM] = (float)mi;
                    eb[gnss::CMB_ELEM_AMP_COH] = (float)std::hypot(mr, mi);
                }
            }
            // DLL observables (gnssRecord.hpp): window-averaged full-band Early/Late powers and
            // the normalized early-minus-late discriminator. ~Linear in the code-phase error
            // over ~+-dll_spacing (tau ~ disc/4 chips at 0.5-chip spacing); the BROKER closes
            // the loop from these at ~Hz. Noise averages out over the window, so no per-record
            // decisions anywhere (R1).
            {
                const double e_pow = acc_epow[p] * inv;
                const double l_pow = acc_lpow[p] * inv;
                rec[gnss::CMB_E_POW] = (float)e_pow;
                rec[gnss::CMB_L_POW] = (float)l_pow;
                rec[gnss::CMB_DLL_DISC] =
                    (e_pow + l_pow) > 0.0 ? (float)((e_pow - l_pow) / (e_pow + l_pow)) : 0.0f;
                // FLEET DLL (docs/CHORD_GNSS_SHARED_DLL.md): the PROMPT power, held for the
                // status snapshot below. E and L are already in the record; this one is not,
                // and acc_ppow is reset here in block mode, so it has to be carried out.
                p_pow_out[p] = (float)(acc_ppow[p] * inv);
                // Boundary fraction f (diagnostic; the inv normalization cancels in the ratio).
                rec[gnss::CMB_HEAD_FRAC] =
                    acc_pe[p] > 0.0 ? (float)(acc_he[p] / acc_pe[p]) : 0.0f;
                // Shared carrier observable: full-band residual carrier (Hz). The broker
                // integrates this into carrier_trim_hz for every subband tracker -- ONE loop at
                // full-band SNR instead of N noise-driven per-channel FLLs. PREFER a bit-robust
                // phase-SLOPE fit over the whole deep-wipe window (_navbuf): its long baseline
                // gives a low-variance frequency, the clean measurement the loop needs to hold
                // <0.5 Hz over a 1 s coherent window (OFFLINE-VALIDATED: with this residual removed,
                // the L1 deep builds 77 sigma@1 s / 94@2 s vs a ~20 ms wall without). The old
                // consecutive-record product (car_S) is short-baseline and doubly-noisy (squaring)
                // -> it made the shared loop noise-inject; kept only as the fallback for non-wipe
                // configs where _navbuf isn't populated.
                double resid_hz = 0.0;
                if (_wipe_buffer && _navbuf[p].size() >= 16) {
                    // Overlay pilots: fit the LINEAR overlay-wiped stream (_navwipe) so the loop
                    // never squares the raw A -- which cancels at boundary_f=0.5 and drove the
                    // loop open there (2026-07-22 root cause). Count the anchored (non-zero) wiped
                    // records; fall back to the squared raw fit if too few are anchored -- data
                    // signals + the pre-anchor bootstrap have _navwipe all-zero (no overlay), so
                    // they take the raw path exactly as before.
                    size_t nw = 0;
                    for (const auto& z : _navwipe[p])
                        if (std::norm(z) > 0.0)
                            ++nw;
                    resid_hz = (nw >= 16)
                        ? carrier_resid_hz(_navwipe[p], _navutc[p], &sigma_phi[p], /*prewiped=*/true,
                                           &snr_q[p])
                        : carrier_resid_hz(_navbuf[p], _navutc[p], &sigma_phi[p], /*prewiped=*/false,
                                           &snr_q[p]);
                } else if (car_n[p] > 0 && std::abs(car_S[p]) > 0.0) {
                    const double dphi = std::arg(car_S[p]) / (carrier_raw() ? 1.0 : 2.0);
                    resid_hz = dphi / (2.0 * M_PI * (car_dt[p] / car_n[p]));
                }
                rec[gnss::CMB_CARRIER_RESID] = (float)resid_hz;
                // Phase arc id -- the raw capture's slip record. The accumulated phase itself
                // is a double and ships via get_status (adr_cycles); what a float slot CAN
                // carry usefully is "is this emit on the same unbroken arc as the last one".
                rec[gnss::CMB_ARC] = (float)_adr_arc[p];
                car_S[p] = {0.0, 0.0};
                car_dt[p] = 0.0;
                car_n[p] = 0;
            }
            // Noise-debiased incoherent significance: <|A|^2> = s^2 + N is biased by the noise
            // floor N (so |A| ~= floor for weak sats), but the moments separate them --
            // s^2 = sqrt(<|A|^2>^2 - Var(|A|^2)), N = <|A|^2> - s^2 -- giving a signal-vs-noise SNR
            // that is ~0 for noise (not the floor) and grows ~sqrt(K) for a real sat. The honest
            // "is it really there", independent of nav-bit wipe.
            {
                const double m2 = acc_pow[p] * inv, m4 = acc_pow2[p] * inv;
                const double var = std::max(0.0, m4 - m2 * m2);
                const double s2 = std::sqrt(std::max(0.0, m2 * m2 - var));
                const double noise = m2 - s2;
                amp_dbi[p] = std::sqrt(s2); // unbiased signal amplitude (~0 for noise, =signal for a sat)
                // Normalise by the H0 std of the s^2 estimator (~N/K^{1/4}) so the significance is
                // ~1 for noise REGARDLESS of K (a stable lock threshold) and grows for real signal.
                amp_snr[p] = noise > 1e-12 ? s2 * std::pow(Keff, 0.25) / noise : 0.0;
                // S4, the AMPLITUDE-SCINTILLATION INDEX -- and the direct, deterministic
                // detector of the multipath that the beam map has been eating blind. It falls
                // straight out of the moments we already accumulate:
                //     S4^2 = Var(|A|^2) / <|A|^2>^2
                // Thermal noise alone produces a floor which must be subtracted -- but NOT
                // using the moment debias above: it DEFINES `noise` as everything that is not
                // the mean, so its implied thermal variance is identically Var, and the
                // "debiased" S4 comes out exactly 0.0000 for every satellite. (Measured.
                // A debias that can only ever return zero is not a debias.) The floor has to
                // come from an INDEPENDENT estimator of the true SNR -- and we have one that
                // is immune to amplitude modulation by construction: the COHERENT deep. It is
                // applied further down, once deep_snr/deep_rec exist. Here: the raw index.
                s4_raw[p] = (m2 > 1e-12) ? std::sqrt(var) / m2 : 0.0;
            }
            if (!_secondary.empty() && _navwipe_bit_records > 0) {
                // ---- OVERLAY *AND* NAVWIPE (S4 phase 2) -- the GPS L5-I case ----
                // L5-I is the only chain on this node that is BOTH a data component and an
                // overlay carrier: 50 sps CNAV under the 10-chip NH10. It needs the two wipes
                // COMPOSED, in order -- NH10 off each 1 ms record first, then the nav-bit wipe
                // over 20-record symbols on the result. Until now these were mutually
                // exclusive branches, which is why phase 1 could only reach the 20 ms data-bit
                // boundary (measured: ~14 dB below the L5-Q sibling at the 1 s rung).
                //
                // ★ THE PHASE SEARCH CANNOT BE COHERENT HERE. overlay_wipe() picks its
                // alignment by maximising a coherent sum over the whole window -- valid for a
                // dataless pilot, meaningless for a data channel, where the +-1 symbols destroy
                // that sum past 20 ms. So the alignment is chosen by the NAVWIPE statistic
                // instead: wipe at each of the L candidate alignments and keep the one whose
                // nav-bit wipe reads highest. That criterion is bit-robust by construction
                // (navwipe estimates each symbol's sign), which is the whole point.
                const auto& ab = _navbuf[p];
                const auto& ub = _navutc[p];
                const auto& hb = _navhead[p];
                const std::vector<int8_t>& ov = _secondary;
                const size_t L = std::max<size_t>(ov.size(), 1);
                // Floor = the navwipe floor (nb free symbol signs) PLUS the extra selection
                // over L overlay alignments. Both selections are real and both must be paid:
                // reporting only the navwipe term would let a noise-only sat clear its bar.
                const double sel = std::sqrt(std::log((double)std::max(_navwipe_bit_records, 2)))
                                   + std::sqrt(std::log((double)std::max<size_t>(L, 2)));
                auto nav_floor = [&](size_t len) {
                    const double nb = std::max(1.0, (double)len / (double)_navwipe_bit_records);
                    return std::sqrt(2.0 * nb / M_PI) + sel;
                };
                bool cleared = false;
                double full_amp = 0.0, full_snr = 0.0;
                size_t full_len = 0;
                std::vector<std::complex<double>> aw, hw; // NH-wiped records + wiped heads
                // ★ The alignment is searched ONCE, on the longest window, and reused for every
                // rung. It is a property of the SIGNAL (the overlay advances deterministically in
                // capture-UTC), not of the trailing window we happen to be scoring, so a per-rung
                // search would (a) cost L x the navwipe on the band with the least real-time
                // slack and (b) be statistically worse -- every extra maximisation rectifies more
                // noise, and the floor would have to grow to pay for selections that buy nothing.
                // ★★ AND IT IS SCORED BY A CHEAP O(n) STATISTIC, NOT BY navwipe.
                // The first cut of this branch scored each alignment with navwipe_amplitude --
                // which runs its OWN bit-phase search over navwipe_bit_records alignments. The
                // two searches then MULTIPLY: 10 NH x 20 bit-phase = 200 inner passes per PRN
                // per emit, which took /l5_valve to 57327 in six minutes (2026-07-29 02:05Z).
                // No ladder tuning fixes a product.
                //
                // navwipe is only needed to MEASURE the winner, not to RANK the candidates.
                // Ranking just needs a statistic that peaks at the right alignment and is blind
                // to the symbol signs: sum over fixed blocks of |coherent sum of the NH-wiped
                // records in that block|. A wrong NH alignment leaves residual +-1 overlay
                // modulation INSIDE every block, which cancels the block sum; the right one
                // leaves only the (constant-within-a-symbol) data sign, which |.| discards.
                // The block grid is fixed and generally misaligned to the true symbol epoch --
                // deliberately fine, because that misalignment costs EVERY candidate the same
                // and this statistic is only ever used for argmax. navwipe then does the real
                // bit-phase search once, at the winner.
                // Cost: L x O(n) to rank + ONE navwipe per rung, i.e. 10 + 20 instead of 10 x 20.
                //
                // ⚠️ KNOWN LIMIT, measured: selection is reliable at per-record S >~ 0.25
                // (19-20/20 on synthetic data) and degrades below it (6/20 at S=0.12). A sat
                // that weak cannot clear its floor anyway, and the failure is FAIL-CLOSED: a
                // wrong alignment despreads noise, navwipe reads low, the floor gate rejects it,
                // and the next emit (0.1 s later) re-picks. It cannot manufacture a detection.
                int best_ph = -1;
                {
                    double best = -1.0;
                    const size_t B = (size_t)std::max(_navwipe_bit_records, 1);
                    for (size_t ph = 0; ph < L; ++ph) {
                        if (!gnss::overlay_apply(ab, ub, ov, (int)ph, &hb, aw, &hw))
                            continue;
                        // POWER, not magnitude: under noise E[|s|^2] is a constant independent
                        // of the alignment, so it biases every candidate equally and cancels in
                        // the argmax, while the signal term adds linearly. Measured on synthetic
                        // NH10 x random-symbol data, power roughly DOUBLES the margin over the
                        // runner-up vs sum|s| (4.95 vs 2.16 at per-record S=1; 1.36 vs 1.15 at
                        // S=0.25) for identical cost.
                        double stat = 0.0;
                        for (size_t i = 0; i < aw.size(); i += B) {
                            std::complex<double> s(0.0, 0.0);
                            const size_t e = std::min(i + B, aw.size());
                            for (size_t j = i; j < e; ++j)
                                s += aw[j];
                            stat += std::norm(s);
                        }
                        if (stat > best) {
                            best = stat;
                            best_ph = (int)ph;
                        }
                    }
                }
                if (best_ph >= 0) {
                    nh_phase[p] = best_ph;
                    // ★★ THE BIT EPOCH IS SEARCHED ONCE, ON THE LONGEST RUNG, AND PINNED FOR THE
                    // REST. navwipe's internal bit sync is navwipe_bit_records full passes over
                    // the window; re-running it per rung is what still cost ~140 drops/s after
                    // the NH-ranking fix (L5-I: 20 alignments x 1000 records = 100x L2C's 1 x 200,
                    // then x ~5 rungs). The epoch is a property of the signal, not of the trailing
                    // window, so pinning it is EXACT -- the shorter rungs are strict suffixes of
                    // the window the epoch was found on.
                    //
                    // ⚠️ The phase is WINDOW-RELATIVE (bit index counts from each window's own
                    // utc[0]), so it must be shifted by the two windows' record-index offset --
                    // derived from UTC, never from buffer positions: a valve drop skips an index,
                    // and position arithmetic would slide every bit epoch by the gap.
                    int pin = -1;
                    double rec_dt_full = 0.0;
                    if (ub.size() >= 2) {
                        std::vector<double> dt;
                        dt.reserve(ub.size() - 1);
                        for (size_t r = 1; r < ub.size(); ++r)
                            dt.push_back(ub[r] - ub[r - 1]);
                        std::nth_element(dt.begin(), dt.begin() + dt.size() / 2, dt.end());
                        rec_dt_full = dt[dt.size() / 2];
                    }
                    for (size_t len :
                         ladder(ab.size(), min_rung(ub, 4 * (size_t)_navwipe_bit_records))) {
                        const bool is_full = (len == ab.size());
                        const std::vector<std::complex<double>> as(ab.end() - (long)len, ab.end());
                        const std::vector<double> us(ub.end() - (long)len, ub.end());
                        const std::vector<std::complex<double>> hs(hb.end() - (long)len, hb.end());
                        if (!gnss::overlay_apply(as, us, ov, best_ph, &hs, aw, &hw))
                            continue;
                        double snr = 0.0;
                        int used_ph = -1;
                        // Shift the pinned epoch into THIS rung's frame (0 on the full window).
                        int pin_here = -1;
                        if (pin >= 0 && rec_dt_full > 0.0) {
                            const long long d =
                                (long long)std::llround((us.front() - ub.front()) / rec_dt_full);
                            const long long m = ((pin + d) % _navwipe_bit_records
                                                 + _navwipe_bit_records)
                                                % _navwipe_bit_records;
                            pin_here = (int)m;
                        }
                        // Export the per-symbol decisions on the full window only -- same rule as
                        // the plain navwipe path (the longest rung sees every symbol the broker's
                        // CNAV stitcher wants).
                        const double amp = navwipe_amplitude(
                            aw, us, &snr, &hw, (is_full && _bit_export) ? &nav_obs[p] : nullptr,
                            pin_here, &used_ph);
                        if (pin < 0 && used_ph >= 0)
                            pin = used_ph; // first (longest) rung searched: pin the rest to it
                        if (full_len == 0) { // ladder walks longest-first
                            full_amp = amp;
                            full_snr = snr;
                            full_len = len;
                        }
                        rungs[p].push_back({(float)len, (float)snr, (float)coh_span(ub, len)});
                        if (snr < FLOOR_MARGIN * nav_floor(len))
                            continue; // indistinguishable from noise rectification
                        if (!cleared || snr > deep_snr[p] * 1.05) {
                            rec[8] = (float)amp;
                            deep_snr[p] = snr;
                            deep_rec[p] = (int)len;
                            deep_floor[p] = nav_floor(len);
                            coh_s[p] = coh_span(ub, len);
                            cleared = true;
                        }
                    }
                }
                if (!cleared && full_len > 0) {
                    rec[8] = (float)full_amp;
                    deep_snr[p] = full_snr;
                    deep_rec[p] = (int)full_len;
                    deep_floor[p] = nav_floor(full_len);
                    coh_s[p] = 0.0; // no rung beat its floor: no coherent detection
                }
                // ★★ RETIRE THE WINDOW. Every branch owns its own clear (the overlay branch
                // and the plain-navwipe branch each have one further down), and this branch
                // was added WITHOUT one -- so _navbuf grew without bound: 1000 records at the
                // first emit, 2000 at the second, 3000 at the third. That single omission
                // produced every symptom this cost me three days to chase: an emit ~4000x its
                // op count, onset ALWAYS delayed (the buffer has to grow first), duty climbing
                // 0.671 -> 1.56 over six minutes (which I misread as "satellites acquiring"),
                // a deeper rec_buf delaying but never fixing it, and a shorter
                // integration_length appearing to make things WORSE. A missing clear does not
                // look like a leak from the outside; it looks like an expensive algorithm.
                if (!_rolling) {
                    _navbuf[p].clear();
                    _navutc[p].clear();
                    _navhead[p].clear();
                    _navwipe[p].clear();
                    if (_peel_depth) {
                        _navbuf_res[p].clear();
                        _navhead_res[p].clear();
                    }
                }
            } else if (!_secondary.empty() || !_l1co.empty()) {
                // Dataless pilot: wipe the KNOWN secondary overlay at its best-fitting alignment
                // -> deep coherent |A| past the primary period (no nav-bit estimate). L5 Q5 uses
                // one PRN-independent NH overlay; L1C-P uses the per-PRN L1CO, picked by the
                // slot's current PRN (a slot's PRN can change as the search re-assigns channels).
                const std::vector<int8_t>* ov = &_secondary;
                if (_l1co.empty()) {
                    // L5: _secondary already points at the shared overlay.
                } else {
                    const int prn = (int)std::lround(ref_prn[p]);
                    ov = (prn >= 1 && prn <= (int)_l1co.size()) ? &_l1co[(size_t)(prn - 1)] : nullptr;
                }
                if (ov && !ov->empty()) {
                    const auto& ab = _navbuf[p];
                    const auto& ub = _navutc[p];
                    const auto& hb = _navhead[p]; // head segments (segmented wipe: the overlay
                                                  // flips MID-record at the period boundary)
                    // Overlay-wipe floor: best of ~n_align alignment trials of a Rayleigh
                    // sum -> noise rectifies to ~sqrt(2 ln n) sigma, window-length-free.
                    const double flr =
                        std::sqrt(2.0 * std::log((double)std::max<size_t>(ov->size(), 2))) + 1.0;
                    deep_floor[p] = flr;
                    bool cleared = false;
                    gnss::OverlayWipeResult full{};
                    size_t full_len = 0;
                    for (size_t len : ladder(ab.size(), min_rung(ub, 2 * ov->size()))) {
                        gnss::OverlayWipeResult ow;
                        if (len == ab.size()) {
                            ow = gnss::overlay_wipe(ab, ub, *ov, &hb);
                        } else {
                            const std::vector<std::complex<double>> as(ab.end() - (long)len,
                                                                       ab.end());
                            const std::vector<double> us(ub.end() - (long)len, ub.end());
                            const std::vector<std::complex<double>> hs(hb.end() - (long)len,
                                                                       hb.end());
                            ow = gnss::overlay_wipe(as, us, *ov, &hs);
                        }
                        if (full_len == 0) { // ladder walks longest-first
                            full = ow;
                            full_len = len;
                        }
                        rungs[p].push_back({(float)len, (float)ow.snr, (float)coh_span(ub, len)});
                        if (ow.snr < FLOOR_MARGIN * flr)
                            continue; // indistinguishable from noise rectification
                        if (!cleared || ow.snr > deep_snr[p] * 1.05) {
                            rec[8] = (float)ow.amplitude;
                            deep_snr[p] = ow.snr;
                            nh_phase[p] = ow.phase;
                            deep_rec[p] = (int)len;
                            coh_s[p] = coh_span(ub, len);
                            cleared = true;
                        }
                    }
                    if (!cleared && full_len > 0) {
                        rec[8] = (float)full.amplitude;
                        deep_snr[p] = full.snr;
                        nh_phase[p] = full.phase;
                        deep_rec[p] = (int)full_len;
                        coh_s[p] = 0.0; // no rung beat its floor: no coherent detection
                    }
                    // PEEL DEPTH on the OVERLAY path (P7b). A pilot's deep uses overlay_wipe,
                    // so the residual must be measured the SAME way -- and, critically, on the
                    // SAME TRAILING WINDOW AND PHASE the prompt just won with. overlay_wipe_at
                    // takes `phase` relative to the START of the array it is given, so feeding
                    // the FULL window a phase that belongs to a shorter winning rung
                    // misaligns the residual wipe by (N - len) mod L chips -- and a misaligned
                    // wipe does not cohere, which INFLATES the reported depth. (Measured
                    // 2026-07-24: that is what produced the first >=47 dB pilot readings.)
                    // Same window, same phase, same estimator = an honest ratio.
                    if (_peel_depth && deep_rec[p] > 0 && nh_phase[p] >= 0
                        && _navbuf_res[p].size() == _navutc[p].size()
                        && _navhead_res[p].size() == _navbuf_res[p].size()
                        && (size_t)deep_rec[p] <= _navbuf_res[p].size()) {
                        const size_t len = (size_t)deep_rec[p];
                        const std::vector<std::complex<double>> rs(
                            _navbuf_res[p].end() - (long)len, _navbuf_res[p].end());
                        const std::vector<double> us(ub.end() - (long)len, ub.end());
                        const std::vector<std::complex<double>> hs(
                            _navhead_res[p].end() - (long)len, _navhead_res[p].end());
                        const auto rw =
                            gnss::overlay_wipe_at(rs, us, *ov, nh_phase[p], &hs);
                        rec[gnss::CMB_PEEL_DEEP] = (float)rw.amplitude;
                        peel_rsnr[p] = rw.snr;
                        double s2 = 0.0;
                        for (const auto& v : rs)
                            s2 += std::norm(v);
                        rec[gnss::CMB_PEEL_INCOH] = (float)std::sqrt(s2 / (double)len);
                    }
                    // DEAD-RECKON ANCHOR (selection-free coherent power): a floor-cleared
                    // win pins the overlay alignment at a capture-UTC; it advances one chip
                    // per primary period deterministically, so later emits can wipe at the
                    // PREDICTED phase -- one straight coherent sum, no max-over-alignments,
                    // E[snr^2] = 2 exactly. That removes the extreme-value variance that
                    // made the coherent single-emit floors WORSE than the incoherent ones
                    // (2026-07-12: C 26 vs 20.3 dB-Hz) and is what unlocks the coherent
                    // observable's sqrt(T_coh/T_rec) sidelobe advantage.
                    if (cleared) {
                        _dr_phase[p] = nh_phase[p];
                        _dr_utc[p] = ub[ub.size() - (size_t)deep_rec[p]];
                        _dr_prn[p] = (int)std::lround(ref_prn[p]);
                        // Record period over the winning window (UTC-exact) -> the ADR wipe
                        // projects any later record's overlay chip index from this anchor.
                        _dr_rec_dt[p] = deep_rec[p] > 1
                            ? (ub.back() - _dr_utc[p]) / (double)(deep_rec[p] - 1) : 0.0;
                    } else if (_dr_prn[p] != (int)std::lround(ref_prn[p])) {
                        _dr_phase[p] = -1; // slot reassigned to another sat: anchor invalid
                        _dr_rec_dt[p] = 0.0;
                    }
                    // PILOT AGREEMENT GATE (see the bit_pred block): does the anchor's
                    // projection to the winning rung's FIRST RECORD (real ub timestamp --
                    // gap-immune) equal the phase the search just found? Only judged when
                    // the search had real SNR; default true so "cannot evaluate" does not
                    // silently veto (the elevation/age/ambiguity gates own those cases).
                    _bp_agree[p] = true;
                    if (_dr_phase[p] >= 0 && _dr_rec_dt[p] > 0.0 && deep_rec[p] > 0
                        && nh_phase[p] >= 0 && deep_snr[p] > FLOOR_MARGIN * flr
                        && ub.size() >= (size_t)deep_rec[p]) {
                        const int L_ov2 = (int)ov->size();
                        const double t0w = ub[ub.size() - (size_t)deep_rec[p]];
                        const long long stp =
                            (long long)std::llround((t0w - _dr_utc[p]) / _dr_rec_dt[p]);
                        const int proj =
                            (int)(((_dr_phase[p] + stp) % L_ov2 + L_ov2) % L_ov2);
                        if (proj != nh_phase[p]) {
                            _bp_agree[p] = false;
                            ++_bp_veto[p];
                        }
                    }
                    if (full_len > 1 && ub.size() >= full_len) {
                        const double t0f = ub[ub.size() - full_len];
                        const double span = ub.back() - t0f;
                        const double T = span * (double)full_len / (double)(full_len - 1);
                        const double rec_dt = span / (double)(full_len - 1);
                        bool dr_ok = _dr_phase[p] >= 0 && rec_dt > 0.0
                                     && _dr_prn[p] == (int)std::lround(ref_prn[p]);
                        if (dr_ok) {
                            const long long steps =
                                (long long)std::llround((t0f - _dr_utc[p]) / rec_dt);
                            const int L_ov = (int)ov->size();
                            const int ph = (int)(((_dr_phase[p] + steps) % L_ov + L_ov) % L_ov);
                            const auto dw = gnss::overlay_wipe_at(ab, ub, *ov, ph, &hb);
                            // /2: snr^2 normalizes noise PER-AXIS (1-dof); the incoherent
                            // x uses full complex (2-dof) noise power. Dividing by 2 puts
                            // deep_pow on x's C/N0-density convention EXACTLY (derivation
                            // 2026-07-12: snr^2 = n*A^2/sigma_axis^2, x = A^2/(2 sigma^2)
                            // -> deep_pow_raw = 2*x/T_rec). Healthy-sat coh/inc then reads
                            // ~0 dB + the incoherent estimator's amplitude-flutter penalty.
                            if (T > 0.0 && dw.snr > 0.0)
                                deep_pow[p] = (dw.snr * dw.snr - 2.0) / (2.0 * T);
                            else
                                dr_ok = false;
                        }
                        if (!dr_ok) { // no anchor yet: searched full window, selection-debiased
                            const double bias = 2.0
                                * (std::log((double)std::max<size_t>(ov->size(), 2)) + 0.5772);
                            if (T > 0.0)
                                deep_pow[p] = (full.snr * full.snr - bias) / (2.0 * T);
                        }
                    }
                }
                // nh_assist defers the block-mode clear: the second pass (below) still needs
                // this window's per-record A/head to run the hinted wipe for the weak sats.
                if (!_rolling && !_nh_assist) {
                    _navbuf[p].clear();
                    _navutc[p].clear();
                    _navhead[p].clear();
                    _navwipe[p].clear();
                    if (_peel_depth) {
                        _navbuf_res[p].clear();
                        _navhead_res[p].clear();
                    }
                }
            } else if (_navwipe_bit_records > 0) {
                const auto& ab = _navbuf[p];
                const auto& ub = _navutc[p];
                const auto& hb = _navhead[p]; // head segments -> symbol-aligned bit epochs
                // Navwipe floor GROWS with window length: nb free +-1 bit signs rectify
                // noise to ~sqrt(2 nb / pi) sigma, plus the bit-phase selection over
                // _navwipe_bit_records alignments (~sqrt(ln br)). The ~7 sigma "GPS deep
                // floor" at 50 bits is exactly this -- and it is why an unlocked sat's
                // ladder drifts to the LONGEST rung (bigger floor) if not gated.
                const double sel = std::sqrt(std::log((double)std::max(_navwipe_bit_records, 2)));
                auto nav_floor = [&](size_t len) {
                    const double nb = std::max(1.0, (double)len / (double)_navwipe_bit_records);
                    return std::sqrt(2.0 * nb / M_PI) + sel;
                };
                bool cleared = false;
                double full_amp = 0.0, full_snr = 0.0;
                size_t full_len = 0;
                for (size_t len :
                     ladder(ab.size(), min_rung(ub, 4 * (size_t)_navwipe_bit_records))) {
                    double snr = 0.0, amp;
                    if (len == ab.size()) {
                        // Full window: also export the per-bit decisions (nav_obs) when asked --
                        // the longest rung sees every bit in the window, which is what the
                        // broker's LNAV stitcher wants.
                        amp = navwipe_amplitude(ab, ub, &snr, &hb,
                                                _bit_export ? &nav_obs[p] : nullptr);
                    } else {
                        const std::vector<std::complex<double>> as(ab.end() - (long)len, ab.end());
                        const std::vector<double> us(ub.end() - (long)len, ub.end());
                        const std::vector<std::complex<double>> hs(hb.end() - (long)len, hb.end());
                        amp = navwipe_amplitude(as, us, &snr, &hs);
                    }
                    if (full_len == 0) { // ladder walks longest-first
                        full_amp = amp;
                        full_snr = snr;
                        full_len = len;
                    }
                    rungs[p].push_back({(float)len, (float)snr, (float)coh_span(ub, len)});
                    if (snr < FLOOR_MARGIN * nav_floor(len))
                        continue; // indistinguishable from noise rectification
                    if (!cleared || snr > deep_snr[p] * 1.05) {
                        rec[8] = (float)amp;
                        deep_snr[p] = snr;
                        deep_rec[p] = (int)len;
                        deep_floor[p] = nav_floor(len);
                        coh_s[p] = coh_span(ub, len);
                        cleared = true;
                    }
                }
                if (!cleared && full_len > 0) {
                    rec[8] = (float)full_amp;
                    deep_snr[p] = full_snr;
                    deep_rec[p] = (int)full_len;
                    deep_floor[p] = nav_floor(full_len);
                    coh_s[p] = 0.0; // no rung beat its floor: no coherent detection
                }
                if (full_len > 1 && ub.size() >= full_len) {
                    const double span = ub.back() - ub[ub.size() - full_len];
                    const double T = span * (double)full_len / (double)(full_len - 1);
                    const double flr0 = nav_floor(full_len);
                    if (T > 0.0) // /2: same 1-dof -> 2-dof convention as the overlay path
                        deep_pow[p] = (full_snr * full_snr - (flr0 * flr0 + 1.0)) / (2.0 * T);
                }
                // PEEL DEPTH: the SAME navwipe deep, on the residual buffer, over the FULL window.
                // No ladder -- a healthy residual sits at the rectification floor, and a ladder
                // would just pick the noisiest short rung. deep(residual) vs deep(prompt) = the
                // suppression; when the residual is at floor the depth is a LOWER BOUND, which the
                // consumer reads off deep_floor exactly as diag/peel_depth.py does offline.
                // Measure the residual on the SAME TRAILING WINDOW the prompt's ladder won
                // with (deep_rec), not the full buffer: the two are normalized per-record but
                // a different integration length is a different estimator, and the ratio has
                // to compare like with like. (navwipe_amplitude re-runs its own bit sync, so
                // unlike the overlay path there is no phase to carry across -- only the
                // length.) Falls back to the full window when no rung was recorded.
                if (_peel_depth && !_navbuf_res[p].empty()
                    && _navbuf_res[p].size() == _navutc[p].size()
                    && _navhead_res[p].size() == _navbuf_res[p].size()) {
                    const size_t len =
                        (deep_rec[p] > 0 && (size_t)deep_rec[p] <= _navbuf_res[p].size())
                            ? (size_t)deep_rec[p]
                            : _navbuf_res[p].size();
                    const std::vector<std::complex<double>> rs(
                        _navbuf_res[p].end() - (long)len, _navbuf_res[p].end());
                    const std::vector<double> us(_navutc[p].end() - (long)len,
                                                 _navutc[p].end());
                    const std::vector<std::complex<double>> hs(
                        _navhead_res[p].end() - (long)len, _navhead_res[p].end());
                    double rsnr = 0.0;
                    const double ramp = navwipe_amplitude(rs, us, &rsnr, &hs);
                    rec[gnss::CMB_PEEL_DEEP] = (float)ramp;
                    peel_rsnr[p] = rsnr;
                    // Incoherent residual amplitude: RMS of the per-record |A_res|, the
                    // phase-blind twin of CMB_AMP_INCOH -- always defined, even at floor.
                    double s2 = 0.0;
                    for (const auto& v : rs)
                        s2 += std::norm(v);
                    rec[gnss::CMB_PEEL_INCOH] = (float)std::sqrt(s2 / (double)len);
                }
                if (!_rolling) { // block clears the window; rolling slides it (capped above)
                    _navbuf[p].clear();
                    _navutc[p].clear();
                    _navhead[p].clear();
                    _navwipe[p].clear();
                    if (_peel_depth) {
                        _navbuf_res[p].clear();
                        _navhead_res[p].clear();
                    }
                }
            } else if (_deep_plain) {
                // PLAIN COHERENT RUNG -- no wipe. The replica already carries the secondary
                // overlay (GPS_L5_Q_NH: NH20 baked into 204600 chips), so each record's
                // amplitude arrives with the overlay removed chip by chip, straddling or not,
                // and the deep integration is a straight sum.
                //
                // This exists because the wipe rungs CANNOT serve that case: overlay_apply
                // advances the overlay one chip per RECORD, an identity that holds on airspy
                // (record = one primary period) and fails on CHORD, where a record is 2048
                // hops = 10.4857 periods. Before this branch such a config had no route to
                // coherence_s at all -- every path ran through a wipe -- so the stage reported
                // 0 forever and the pilot integrated one 10.5 ms record at a time.
                const auto& ab = _navbuf[p];
                const auto& ub = _navutc[p];
                bool cleared = false;
                gnss::OverlayWipeResult full{};
                size_t full_len = 0;
                // Floor with NO alignment search in it. The overlay rungs pay
                // sqrt(2 ln n_align) for taking a max over alignments; a straight sum takes no
                // max, so its floor is the plain Rayleigh tail and the bar sits LOWER -- which
                // is the sensitivity this branch is for. n = 2 keeps the log defined and leaves
                // a little margin for the ladder's own handful of rungs.
                const double flr = std::sqrt(2.0 * std::log(2.0)) + 1.0;
                for (size_t len : ladder(ab.size(), min_rung(ub, 4))) {
                    std::vector<std::complex<double>> w(ab.end() - (long)len, ab.end());
                    // PHASE-RATE SEARCH (2026-08-04). Measured on sky: the per-record prompts
                    // carry a strong LINEAR phase ramp -- lag-1 coherence 0.49-0.67 against a
                    // shuffled null of 0.04-0.12, and a ramp-corrected sum recovers 0.65-0.80 of
                    // the power where the straight sum recovers 0.02-0.05. That is why every
                    // rung read the Rayleigh value (~1) while the incoherent moments said +10 to
                    // +15 dB per record: the sum was integrating a rotating phasor. Not sign
                    // flips (the squaring test says 0.03-0.13) and not phase noise (a random walk
                    // has no lag-1 structure like this).
                    //
                    // The ramp rate exceeds the record-rate Nyquist and jumps between polls, so
                    // it cannot simply be predicted and subtracted -- it has to be searched. The
                    // search is over a uniform grid because the records ARE uniform in hop; the
                    // gaps (a PRN the broker drops for a window) become zeros in the transform,
                    // which is exact rather than approximate.
                    double rate_hz = 0.0, rate_q = 0.0;
                    if (_deep_rate) {
                        const std::vector<double> us(ub.end() - (long)len, ub.end());
                        rate_search(w, us, rate_hz, rate_q);
                        if (rate_q >= _deep_rate_min_q && rate_hz != 0.0) {
                            // Derotate and let the EXISTING estimator score it, so deep_snr,
                            // deep_amplitude and coherence_s stay in the currency every consumer
                            // already reads. The search decides WHETHER; coherent_sum decides
                            // what the number is.
                            const double t0 = us.front();
                            for (size_t k = 0; k < w.size(); ++k)
                                w[k] *= std::polar(1.0, -2.0 * M_PI * rate_hz * (us[k] - t0));
                        } else {
                            rate_hz = 0.0; // nothing found: score the straight sum, as before
                        }
                    }
                    const gnss::OverlayWipeResult cs = gnss::coherent_sum(w);
                    if (full_len == 0) { // ladder walks longest-first
                        full = cs;
                        full_len = len;
                        deep_rate[p] = rate_hz;
                        deep_rate_q[p] = rate_q;
                    }
                    rungs[p].push_back({(float)len, (float)cs.snr, (float)coh_span(ub, len)});
                    // THE GATE IS THE SEARCH'S OWN SPECTRUM, not a closed-form floor. A max over
                    // bins needs an alignment penalty, and the closed form sqrt(2 ln n)+1 gets it
                    // badly wrong here (predicts 4.53, noise actually reads ~3.1) because 4x
                    // oversampled bins are nowhere near independent. peak/median is measured on
                    // the same data, so it is immune to rung length, bin count and correlation
                    // alike: on sky it read 17.9-22.0 on signal and 2.8-6.1 on noise. Same
                    // self-calibrating principle as the fleet DLL's q_floor.
                    if (_deep_rate && rate_hz != 0.0) {
                        if (rate_q < _deep_rate_min_q)
                            continue;
                    } else if (cs.snr < FLOOR_MARGIN * flr)
                        continue; // indistinguishable from noise
                    if (!cleared || cs.snr > deep_snr[p] * 1.05) {
                        deep_rate[p] = rate_hz;
                        deep_rate_q[p] = rate_q;
                        rec[8] = (float)cs.amplitude;
                        deep_snr[p] = cs.snr;
                        deep_rec[p] = (int)len;
                        deep_floor[p] = flr;
                        coh_s[p] = coh_span(ub, len);
                        cleared = true;
                    }
                }
                if (!cleared && full_len > 0) {
                    rec[8] = (float)full.amplitude;
                    deep_snr[p] = full.snr;
                    deep_rec[p] = (int)full_len;
                    deep_floor[p] = flr;
                    coh_s[p] = 0.0; // no rung beat its floor: no coherent detection
                }
                if (full_len > 1 && ub.size() >= full_len) {
                    const double span = ub.back() - ub[ub.size() - full_len];
                    const double T = span * (double)full_len / (double)(full_len - 1);
                    if (T > 0.0) // same 1-dof -> 2-dof debias as the other paths
                        deep_pow[p] = (full.snr * full.snr - (flr * flr + 1.0)) / (2.0 * T);
                }
            }
            *reinterpret_cast<double*>(rec + RECORD_UTC_SLOT) = ref_utc[p];
            // Absolute hop of this window, alongside the UTC rather than instead of it (slot 9
            // has consumers, and overlay_apply/coh_span are UTC-based). Makes a recorded .raw
            // self-describing about WHICH window each row is, exactly -- the same key the fleet
            // DLL groups on. Same value for every PRN: one window, one hop.
            *reinterpret_cast<long long*>(rec + gnss::CMB_HOP_SLOT) = win_hop;
            if (!_rolling) {
                acc_pow[p] = acc_pow2[p] = acc_ar[p] = acc_ai[p] = acc_nchan[p] = acc_epow[p] = acc_lpow[p] = acc_ppow[p] = acc_he[p] = acc_pe[p] = 0.0; // block: reset
                for (int el = 0; el < _n_elements; ++el) { // per-antenna block reset
                    const size_t k = (size_t)p * _n_elements + el;
                    _acc_epow_el[k] = _acc_ear[k] = _acc_eai[k] = 0.0;
                }
            }
        }
        if (!_rolling)
            n_acc = 0;

        // ---- nh TIME-ASSIST (second pass) ----------------------------------------------------
        // The blind alignment search (above) needs SNR to win over 1800 trials; the weak sats
        // lose it and never certify. Here we seed their deep wipe with the geometrically-correct
        // alignment instead. The broker POSTs each PRN's predicted absolute overlay index (from
        // BeiDou time + range, its own convention constant); at ONE emit the (nh_found - hint)
        // offset is the SAME for every sat (shared window start + a shared time->index map), so we
        // measure it from the sats the blind search locked CONFIDENTLY and apply it to the rest.
        // Purely additive: a hinted wipe is used only if it clears its floor AND beats the blind
        // result, so a wrong hint (or a mislocked reference outvoted by the cluster) changes
        // nothing. Requires the deferred navbuf (block mode) still holding this window.
        if (_nh_assist) {
            std::lock_guard<std::mutex> lk(_nh_mtx);
            // "Confident" reference bar. NOT 3x floor: B1C's blind wipe (1800 trials on a
            // 1 s window, floor ~4.9, cert gate 2x) pins even HEALTHY sats' deep_snr to a
            // narrow band just above 2x floor -- a 3x bar left the assist DORMANT on the
            // very signal it targets (measured 07-17: 0 qualifying emits/15 min at 3x).
            // The real mislock protection is the >=nh_min_refs cluster agreement (+-3
            // chips) and the fail-safe adoption, so the bar = the certification gate.
            const double NH_REF_MARGIN = _nh_ref_margin;
            auto hint_of = [&](int p) -> int {
                const int prn = (int)std::lround(ref_prn[p]);
                return (prn >= 1 && prn <= _n_prn) ? _nh_hint[prn - 1] : -1;
            };
            auto ovlen = [&](int p) -> int {
                const int prn = (int)std::lround(ref_prn[p]);
                return (prn >= 1 && prn <= (int)_l1co.size()) ? (int)_l1co[prn - 1].size() : 0;
            };
            // pivot on the strongest confidently-locked sat, then take the offset cluster median
            double pivot = -1.0, best = 0.0;
            for (int p = 0; p < _n_prn; ++p) {
                const int L = ovlen(p), h = hint_of(p);
                if (L <= 0 || h < 0 || coh_s[p] <= 0.0)
                    continue;
                if (deep_snr[p] > NH_REF_MARGIN * deep_floor[p] && deep_snr[p] > best) {
                    best = deep_snr[p];
                    pivot = (double)(((nh_phase[p] - h) % L + L) % L);
                }
            }
            if (pivot >= 0.0) {
                std::vector<double> cl;
                for (int p = 0; p < _n_prn; ++p) {
                    const int L = ovlen(p), h = hint_of(p);
                    if (L <= 0 || h < 0 || coh_s[p] <= 0.0
                        || deep_snr[p] <= NH_REF_MARGIN * deep_floor[p])
                        continue;
                    const double o = (double)(((nh_phase[p] - h) % L + L) % L);
                    const double d = std::remainder(o - pivot, (double)L); // agree with the pivot?
                    if (std::abs(d) <= 3.0)
                        cl.push_back(pivot + d);
                }
                if ((int)cl.size() >= _nh_min_refs) {
                    std::nth_element(cl.begin(), cl.begin() + cl.size() / 2, cl.end());
                    const long long offset = (long long)std::llround(cl[cl.size() / 2]);
                    std::string dbg; // per-emit assist trace (INFO, cheap: one line per emit)
                    for (int p = 0; p < _n_prn; ++p) {
                        const int L = ovlen(p), h = hint_of(p), prn = (int)std::lround(ref_prn[p]);
                        if (L <= 0 || h < 0)
                            continue;
                        if (deep_snr[p] > NH_REF_MARGIN * deep_floor[p] && coh_s[p] > 0.0)
                            continue; // already confidently locked -- leave it
                        const auto& ab = _navbuf[p];
                        const auto& ub = _navutc[p];
                        const auto& hb = _navhead[p];
                        if (ab.size() < 2)
                            continue;
                        const auto& ov = _l1co[(size_t)(prn - 1)];
                        const int hph0 = (int)(((h + offset) % L + L) % L);
                        const double flr =
                            std::sqrt(2.0 * std::log((double)std::max<size_t>(ov.size(), 2))) + 1.0;
                        // Try the hinted phase +-nh_search_chips (default +-2 = 5 trials, not
                        // 1800). The hint rides the broker almanac's range: with the BRDC
                        // source (m-accurate, PRN-indexed, sat clock included) the prediction
                        // is sub-chip, so +-2 is generous; the TLE era needed +-8 because
                        // celestrak's BeiDou IGSO PRN labels are WRONG for C31/C38/C39
                        // (measured 07-17: 16-20k km = 5-7 chips off). Widen via config if
                        // ever back on the TLE fallback. The consensus offset from correctly-
                        // labelled refs is unaffected either way (median).
                        //
                        // ...AND down the same rung LADDER the blind path uses. A full-window
                        // STRAIGHT sum decoheres as sinc(f_resid * T): the routine sub-Hz
                        // carrier residual (C37 at -0.7 Hz, 07-17) costs x0.37 over 1 s and
                        // pinned every hinted wipe at snr 3-6 vs floor 9.8 while the BLIND
                        // path certified the same sats on its shorter rungs. Same physics,
                        // same ladder.
                        gnss::OverlayWipeResult dw{};
                        int hph = hph0;
                        size_t dlen = ab.size();
                        for (size_t len : ladder(ab.size(), min_rung(ub, ov.size() / 8))) {
                            const std::vector<std::complex<double>> as(ab.end() - (long)len,
                                                                       ab.end());
                            const std::vector<double> us(ub.end() - (long)len, ub.end());
                            const std::vector<std::complex<double>> hsg(hb.end() - (long)len,
                                                                        hb.end());
                            for (int dph = -_nh_search_chips; dph <= _nh_search_chips; ++dph) {
                                const int ph = ((hph0 + dph) % L + L) % L;
                                const auto w = gnss::overlay_wipe_at(as, us, ov, ph, &hsg);
                                if (w.snr > dw.snr) {
                                    dw = w;
                                    hph = ph;
                                    dlen = len;
                                }
                            }
                        }
                        {
                            char b[96];
                            snprintf(b, sizeof b, " C%d:h%d+o=%d snr%.1f/blind%.1f%s", prn,
                                     h, hph, dw.snr, deep_snr[p],
                                     (dw.snr > FLOOR_MARGIN * flr && dw.snr > deep_snr[p])
                                         ? "*ADOPT" : "");
                            dbg += b;
                        }
                        if (dw.snr > FLOOR_MARGIN * flr && dw.snr > deep_snr[p]) {
                            float* rec = out + (size_t)p * _rec_stride;
                            rec[8] = (float)dw.amplitude;
                            deep_snr[p] = dw.snr;
                            nh_phase[p] = hph;
                            deep_rec[p] = (int)dlen;
                            deep_floor[p] = flr;
                            const double t0r = ub[ub.size() - dlen];
                            coh_s[p] = coh_span(ub, dlen);
                            const double span = ub.back() - t0r;
                            const double T = dlen > 1
                                ? span * (double)dlen / (double)(dlen - 1) : 0.0;
                            if (T > 0.0) // 17-trial selection, ~2-dof debias still fair
                                deep_pow[p] = (dw.snr * dw.snr - 2.0) / (2.0 * T);
                            _dr_phase[p] = hph; // seed the anchor: future emits dead-reckon it
                            _dr_utc[p] = t0r;
                            _dr_prn[p] = prn;
                            _dr_rec_dt[p] =
                                dlen > 1 ? span / (double)(dlen - 1) : 0.0; // for the ADR wipe
                        }
                    }
                    if (!dbg.empty())
                        WARN("nh-assist refs={:d} off={:d}:{:s}", (int)cl.size(), (int)offset,
                             dbg); // WARN so it clears log_level: warn (debug aid; demote later)
                }
            }
            if (!_rolling) // the deferred block-mode clear, now that the window is consumed
                for (int p = 0; p < _n_prn; ++p) {
                    _navbuf[p].clear();
                    _navutc[p].clear();
                    _navhead[p].clear();
                    _navwipe[p].clear();
                    // The residual rides the SAME window and must be cleared with it. Missing
                    // this desynced the two on the nh-assist chains (B1C/L1C, whose clear is
                    // deferred to here): _navbuf_res kept growing while _navbuf reset, the
                    // size guard in the depth block then skipped every emit, and the pilots
                    // peeled with no instrument reporting it (2026-07-24).
                    if (_peel_depth) {
                        _navbuf_res[p].clear();
                        _navhead_res[p].clear();
                    }
                }
        }

        // Publish the latest full-band amplitudes for the broker's drop decisions.
        {
            std::lock_guard<std::mutex> lk(_st_mtx);
            _st_pow_hop = win_hop; // one window, one hop -- not per PRN
            for (int p = 0; p < _n_prn; ++p) {
                const float* rec = out + (size_t)p * _rec_stride;
                _st_prn[p] = (int)std::lround(rec[0]);
                _st_amp[p] = rec[3];
                _st_coh[p] = rec[6];
                _st_deep[p] = rec[8];
                _st_deep_snr[p] = (float)deep_snr[p];
                _st_nh_phase[p] = nh_phase[p];
                _st_amp_snr[p] = (float)amp_snr[p];
                _st_amp_dbi[p] = (float)amp_dbi[p];
                _st_dop[p] = rec[1];
                _st_cp[p] = rec[2];
                _st_coh_s[p] = (float)coh_s[p];
                _st_deep_rec[p] = deep_rec[p];
                _st_deep_floor[p] = (float)deep_floor[p];
                _st_rungs[p] = rungs[p];
                _st_deep_rate[p] = (float)deep_rate[p];
                _st_deep_rate_q[p] = (float)deep_rate_q[p];
                if (_rec_export > 0)
                    _st_recex[p] = _recex[p];
                _st_deep_pow[p] = (float)deep_pow[p];
                _st_peel_deep[p] = rec[gnss::CMB_PEEL_DEEP];
                _st_peel_incoh[p] = rec[gnss::CMB_PEEL_INCOH];
                _st_peel_deep_snr[p] = (float)peel_rsnr[p];
                if (_bit_export)
                    _st_nav_obs[p] = std::move(nav_obs[p]);
                // P7b: a pilot's overlay is deterministic -- project the pinned dead-reckon
                // anchor forward and publish the chips the peel should subtract with. Same
                // projection the ADR wipe uses (idx from UTC, chip = phase+idx mod L), so the
                // two can never disagree about which chip a record carries.
                if (_bit_export) {
                    BitPred bp;
                    const int prn_e = (int)std::lround(rec[0]);
                    const std::vector<int8_t>* ov = overlay_for(prn_e);
                    const double t_e = ref_utc[p];
                    // STALE-ANCHOR GATE (2026-07-26). Without `cleared` the anchor is never
                    // re-pinned AND never expires, so a weak/setting sat keeps publishing
                    // chips from an anchor that has slipped -- measured live: BDS 23 at -180
                    // chips, GAL 21 at -5.4, all with deep_snr ~0. Wrong-but-confident bits
                    // are the one failure this path must not produce (they subtract at the
                    // WRONG SIGN half the time); no fresh pin, no publication -- the tracker
                    // then reads `nobits` and simply does not subtract, which is the safe
                    // direction. Healthy sats re-pin every cleared emit (~1 s), so 60 s is
                    // generous for real locks and far below the ~17-30 min half-chip slip.
                    const bool anchor_fresh = (t_e - _dr_utc[p]) < 60.0;
                    // AGREEMENT GATE (pilot half of navbit-health, 2026-07-26). The searched
                    // wipe re-finds the TRUE overlay phase every emit (nh_phase, full window
                    // when not cleared); the anchor's projection is what bit_pred publishes.
                    // When both exist and the search had real SNR, they must agree exactly --
                    // an integer-chip mismatch means the anchor is WRONG NOW, and the age
                    // gate cannot see it: a marginal sat can clear rarely, pin a bad phase on
                    // one weak win, then publish confidently for the full 60 s. This is the
                    // same question the broker's navbit_health asks on GPS (do the published
                    // bits match the sky?), asked where pilots can answer it: nav_obs does
                    // not exist on the overlay path, but nh_phase does, and it IS the air.
                    // Fail closed: disagree -> publish nothing this emit.
                    // anchor-vs-search agreement computed in the WIPE section, where the
                    // rung's REAL first-record timestamp is in scope. First cut reconstructed
                    // it as t_e - (len-1)*dt, which assumes a GAPLESS window; gaps are
                    // routine, so the gate fired on healthy anchors and starved the pilot
                    // chains of bits (BDS: 7 locked, 1 publishing; caught by the warmup
                    // counts within 15 min). Time-difference step counts are gap-immune;
                    // count-based reconstructions are not.
                    if (ov && !ov->empty() && _dr_phase[p] >= 0 && _dr_rec_dt[p] > 0.0
                        && _dr_prn[p] == prn_e && t_e > 0.0 && anchor_fresh
                        && _bp_agree[p]) {
                        const int L_ov = (int)ov->size();
                        const double dt_r = _dr_rec_dt[p];
                        // start at the first primary period at/after this emit
                        const long long i0 =
                            (long long)std::ceil((t_e - _dr_utc[p]) / dt_r);
                        const int n = (int)std::ceil(_bit_pred_horizon_s / dt_r) + 2;
                        // TABLE SEMANTICS, DECLARED (2026-07-27; see BitPred::record_grid).
                        // utc0 is the RAW RECORD-START time of cell 0 and the table is
                        // record-indexed, full stop. The 2026-07-26 -(1-bf)*dt shift is GONE:
                        // it existed only to disguise this table as time-domain so the
                        // consumer's half-record probe would land right, and measurement
                        // showed the disguise parks the consumer's floor() exactly on its own
                        // discontinuity (psi + bf = 1.000 +- 0.02 on every satellite ->
                        // floor(N + 0.5)). The consumer now indexes by record instead, so
                        // there is nothing to compensate for and bf does not enter at all.
                        bp.utc0 = _dr_utc[p] + (double)i0 * dt_r;
                        bp.bit_s = dt_r;
                        bp.record_grid = true;
                        bp.bits.reserve((size_t)n);
                        for (int j = 0; j < n; ++j) {
                            const long long k = _dr_phase[p] + i0 + j;
                            const int kk = (int)(((k % L_ov) + L_ov) % L_ov);
                            const int8_t c = (*ov)[(size_t)kk];
                            bp.bits.push_back(c >= 0 ? (int8_t)1 : (int8_t)-1);
                        }
                    }
                    _st_bit_pred[p] = std::move(bp);
                }
                _st_dll_disc[p] = rec[gnss::CMB_DLL_DISC];
                // FLEET DLL: the discriminator's INGREDIENTS. Ratios do not sum, so the
                // broker needs these to combine instances (docs/CHORD_GNSS_SHARED_DLL.md).
                _st_e_pow[p] = rec[gnss::CMB_E_POW];
                _st_l_pow[p] = rec[gnss::CMB_L_POW];
                _st_p_pow[p] = p_pow_out[p];
                _st_nchan[p] = rec[gnss::CMB_NCHAN];
                _st_head_frac[p] = rec[gnss::CMB_HEAD_FRAC];
                // S4, thermal floor removed using the COHERENT SNR (independent of the
                // amplitude fluctuation we are trying to measure). Per-record power SNR
                // x = deep_snr^2 / K; a non-central chi-square with 2 dof then has
                //     S4_thermal^2 = (1 + 2x) / (1 + x)^2
                // and the signal's own fading is what is left over.
                {
                    const double K = (double)std::max(1, deep_rec[p]);
                    const double x = (deep_snr[p] > 0.0) ? deep_snr[p] * deep_snr[p] / K : 0.0;
                    const double s4_th2 = (1.0 + 2.0 * x) / ((1.0 + x) * (1.0 + x));
                    const double s4r = _st_s4_raw[p];
                    _st_s4[p] = (float)std::sqrt(std::max(0.0, s4r * s4r - s4_th2));
                }
                _st_s4_raw[p] = (float)s4_raw[p];
                _st_sigma_phi[p] = (float)sigma_phi[p];
                _st_snr_q[p] = (float)snr_q[p];
                _st_car_resid[p] = rec[gnss::CMB_CARRIER_RESID];
                // ADR snapshot: the phase itself is a double (REST only); the record carries
                // the arc id, which is what makes a raw capture self-describing about slips.
                const double utc_e = *reinterpret_cast<const double*>(rec + RECORD_UTC_SLOT);
                _st_utc[p] = utc_e;
                _st_adr[p] = _adr_ok[p] ? _adr_cyc[p] : 0.0;
                _st_trim[p] = _adr_ok[p] ? _trim_cyc[p] : 0.0;
                _st_adr_arc[p] = _adr_arc[p];
                _st_adr_n[p] = _adr_ok[p] ? _adr_n[p] : 0;
                _st_adr_lock[p] =
                    (_adr_ok[p] && _adr_t0[p] > 0.0) ? (utc_e - _adr_t0[p]) : 0.0;
            }
        }
        {
            const double us = std::max(0.0, (current_time() - _t_emit0) * 1e6);
            if (_m_emit_us)
                _m_emit_us->inc((uint64_t)us);
            if (_m_emits)
                _m_emits->inc();
            if (_m_last_emit_us)
                _m_last_emit_us->set(us);
            if (_m_last_emit_prns) {
                // BOUND BY _navbuf's OWN SIZE, not _n_prn. _navbuf is allocated only under
                // _wipe_buffer (see the ctor), so in a config with no wipe -- no
                // navwipe_bit_records, no secondary_overlay, no L1CO -- it is EMPTY, and
                // looping to _n_prn indexes past the end. The compiler vectorises the .empty()
                // checks, so that reads 32 bytes from a null data pointer and SIGSEGVs:
                // `vmovdqu (%rax),%ymm1` with rax = 0, in main_thread, every time.
                //
                // A conflict-free auto-merge: this metrics block came from one branch and the
                // _wipe_buffer gating from the other, and nothing made them disagree textually.
                // It is CHORD-specific for the same reason -- airspy sets secondary_overlay
                // (L5_NH20), so _navbuf is always allocated there and the prototype never sees
                // it. That is also why it read as "the merge broke GnssCoherentCombiner": the
                // stage is only the messenger.
                int active = 0;
                for (size_t p = 0; p < _navbuf.size(); ++p)
                    if (!_navbuf[p].empty())
                        ++active;
                _m_last_emit_prns->set((double)active); // cost scales with THIS, not with n_prn
            }
        }
        out_buf->mark_frame_full(unique_name, out_id++);
    }
}

void GnssCoherentCombiner::set_nh_hint_callback(kotekan::connectionInstance& conn,
                                                nlohmann::json& request) {
    // Broker POST: [{"prn":N,"nh":k}, ...] -- k = predicted absolute overlay index (broker's
    // own convention; the per-emit consensus offset absorbs the constant). Missing/dropped PRNs
    // keep their last hint; the deep wipe only trusts a hint whose PRN is currently active.
    try {
        std::lock_guard<std::mutex> lk(_nh_mtx);
        for (const auto& s : request) {
            const int prn = s.at("prn").get<int>();
            const int nh = s.at("nh").get<int>();
            if (prn >= 1 && prn <= _n_prn && nh >= 0)
                _nh_hint[prn - 1] = nh; // keyed by PRN (these pilots' slots are PRN-1-indexed)
        }
    } catch (const std::exception& e) {
        conn.send_error(e.what(), kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void GnssCoherentCombiner::push_recex(int p, long long hop, double re, double im, double energy) {
    auto& r = _recex[p];
    r.push_back({(double)hop, re, im, energy});
    if ((int)r.size() > _rec_export)
        r.erase(r.begin(), r.begin() + ((int)r.size() - _rec_export));
}

void GnssCoherentCombiner::get_records_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::array();
    std::lock_guard<std::mutex> lk(_st_mtx);
    for (int p = 0; p < _n_prn; ++p) {
        if (_st_recex[p].empty())
            continue; // a PRN with no records is silence, not a row of zeros
        reply.push_back({{"prn", _st_prn[p]},
                         {"doppler_hz", _st_dop[p]},
                         {"code_phase_chips", _st_cp[p]},
                         // [hop, re, im, energy] per record, oldest first. See _rec_export: hop
                         // is the exact cross-node key, re/im are energy-normalized (so directly
                         // comparable across instances), energy is the ML combining weight.
                         {"records", _st_recex[p]}});
    }
    conn.send_json_reply(reply);
}

void GnssCoherentCombiner::get_status_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::array();
    std::lock_guard<std::mutex> lk(_st_mtx);
    for (int p = 0; p < _n_prn; ++p)
        reply.push_back({{"prn", _st_prn[p]},
                         {"amplitude", _st_amp[p]},
                         {"coh_amplitude", _st_coh[p]},
                         {"deep_amplitude", _st_deep[p]},
                         {"deep_snr", _st_deep_snr[p]},
                         {"amp_snr", _st_amp_snr[p]},
                         {"nh_phase", _st_nh_phase[p]},
                         {"unbiased_amplitude", _st_amp_dbi[p]},
                         {"doppler_hz", _st_dop[p]},
                         {"code_phase_chips", _st_cp[p]},
                         {"coherence_s", _st_coh_s[p]},
                         {"deep_records", _st_deep_rec[p]},
                         {"deep_floor", _st_deep_floor[p]},
                         {"deep_pow_hz", _st_deep_pow[p]},
                         // EVERY ladder rung this emit, longest-first: [[records, snr, span_s], ...].
                         // deep_snr is the MAX over these, and a max hides the shape. A signal that
                         // never coheres sits at the floor on EVERY rung; one that coheres but is
                         // too weak rises as the rung shortens (less phase loss) or peaks mid-ladder
                         // at the knee. Same floor applies to all rungs on the plain branch
                         // (deep_floor); the wipe branches' floor grows with rung length.
                         {"rungs", _st_rungs[p]},
                         // PHASE-RATE SEARCH: the linear rate removed before the deep sum (Hz,
                         // signed, modulo the record rate -- the true ramp exceeds that Nyquist),
                         // and the peak/median of its spectrum. deep_rate_q is the honest
                         // detection statistic for the coherent rung: measured on sky at 17.9-22.0
                         // on signal and 2.8-6.1 on noise. 0 = the search did not run or found
                         // nothing and the straight sum was scored instead.
                         {"deep_rate_hz", _st_deep_rate[p]},
                         {"deep_rate_q", _st_deep_rate_q[p]},
                         // PEEL DEPTH (docs/gnss_voltage_peel_live.md): deep/incoherent |A| of the
                         // voltage-peel RESIDUAL. depth_db = 20*log10(deep_amplitude/peel_deep);
                         // 0 when this chain does not peel. At the combiner floor -> a LOWER BOUND.
                         {"peel_deep", _st_peel_deep[p]},
                         {"peel_incoh", _st_peel_incoh[p]},
                         // The residual deep's SIGNIFICANCE: compare THIS against deep_floor for
                         // the at-floor / lower-bound decision (peel_deep is an amplitude and
                         // deep_floor a significance -- comparing those is a units error).
                         {"peel_deep_snr", _st_peel_deep_snr[p]},
                         {"dll_disc", _st_dll_disc[p]},
                         // FLEET DLL (docs/CHORD_GNSS_SHARED_DLL.md). dll_disc is a RATIO and
                         // ratios do not sum: one instance correlates 7 x 195.3 kHz = 6.7% of
                         // the L5 lobe, and the full-band discriminator is
                         // (SUM e_pow - SUM l_pow)/(SUM e_pow + SUM l_pow) over every instance,
                         // which is not any function of the per-instance dll_disc values. So
                         // the three POWERS ship raw. All are built the same way -- |sum of
                         // subband correlations|^2 / energy^2 -- so q = 2*p_pow/(e_pow+l_pow)
                         // is comparable instance to instance and, summed, is the fleet's.
                         {"e_pow", _st_e_pow[p]},
                         {"p_pow", _st_p_pow[p]},
                         {"l_pow", _st_l_pow[p]},
                         {"n_chan", _st_nchan[p]},
                         // The GROUPING KEY: absolute hop index of the window these powers were
                         // accumulated over, from the F-engine sample counter every node shares.
                         // Equal hop IS the same sky, exactly -- no tolerance, unlike "utc".
                         // pow_fft_len declares the currency: 0 means pow_hop is a raw SAMPLE
                         // index (this stage was configured without fft_len), so a consumer can
                         // never mix hops and samples by accident. -1 = no metadata on the
                         // record frames at all.
                         {"pow_hop", _st_pow_hop},
                         {"pow_fft_len", _fft_len},
                         // boundary fraction f: where the code-period boundary sits inside
                         // the despread window (segmented-wipe diagnostic; ~0.5 was the old
                         // bistable's null zone, now harmless).
                         {"boundary_f", _st_head_frac[p]},
                         // multipath / scintillation pair (2026-07-14): S4 = amplitude
                         // fluctuation (thermal floor removed), sigma_phi = carrier-phase
                         // jitter about the slope fit, radians. Multipath moves BOTH.
                         {"s4", _st_s4[p]},
                         {"s4_raw", _st_s4_raw[p]},
                         {"sigma_phi", _st_sigma_phi[p]},
                         {"snr_q", _st_snr_q[p]},
                         // Nonzero means a subband dropped a frame relative to the others and
                         // was realigned. Index-lockstep gathering absorbed that silently as a
                         // permanent epoch offset, so it is worth being able to see.
                         {"subband_realigns", (double)_realigns},
                         {"carrier_hz_resid", _st_car_resid[p]},
                         // Accumulated carrier phase (cycles) on the current unbroken arc,
                         // its id, its length in records, and how long it has held (s). 0
                         // cycles with a bumped arc id means the arc just restarted.
                         {"utc", _st_utc[p]},
                         {"adr_cycles", _st_adr[p]},
                         // Commanded-trim integral on the SAME arc (subtract from
                         // adr_cycles to remove the carrier loop's known transients --
                         // docs/adr_trim_subtraction.md)
                         {"trim_cycles", _st_trim[p]},
                         {"adr_arc", _st_adr_arc[p]},
                         {"adr_records", _st_adr_n[p]},
                         {"adr_lock_s", _st_adr_lock[p]}});
    // P7a nav_obs: the last emit's per-bit sign decisions, only on rows that have any --
    // the broker's LNAV stitcher consumes these (see NavObs for the timing convention).
    if (_bit_export)
        for (int p = 0; p < _n_prn; ++p) {
            // Pilot chains publish PREDICTIONS (deterministic overlay -- the broker forwards
            // them straight into nav_bits); data chains publish OBSERVATIONS for LNAV decode.
            const BitPred& bp = _st_bit_pred[(size_t)p];
            if (!bp.bits.empty()) {
                nlohmann::json pb = nlohmann::json::array();
                for (int8_t c : bp.bits)
                    pb.push_back((int)c);
                if (_bp_veto[(size_t)p] > 0)
                    reply[(size_t)p]["bp_veto"] = _bp_veto[(size_t)p];
                // "grid": the table's own declaration of what utc0 means (see
                // BitPred::record_grid). A consumer that does not know the key must keep
                // treating the table as time-domain, which is the pre-2026-07-27 behaviour.
                reply[(size_t)p]["bit_pred"] = {{"utc0", bp.utc0},
                                                {"bit_s", bp.bit_s},
                                                {"grid", bp.record_grid ? "record" : "time"},
                                                {"bits", pb}};
            }
            const NavObs& o = _st_nav_obs[(size_t)p];
            if (o.bits.empty())
                continue;
            nlohmann::json bits = nlohmann::json::array();
            for (const auto& b : o.bits)
                bits.push_back({b.first, (int)b.second});
            reply[(size_t)p]["nav_obs"] = {{"utc_ref", o.utc_ref},
                                           {"rec_dt", o.rec_dt},
                                           {"phase", o.phase},
                                           {"br", o.br},
                                           {"rot_re", o.rot_re},
                                           {"rot_im", o.rot_im},
                                           {"bits", bits}};
        }
    conn.send_json_reply(reply);
}

double
GnssCoherentCombiner::navwipe_amplitude(const std::vector<std::complex<double>>& a,
                                        const std::vector<double>& utc, double* snr_out,
                                        const std::vector<std::complex<double>>* head, NavObs* obs,
                                        int pin_phase, int* out_phase) const {
    const int br = _navwipe_bit_records;
    const int nrec = (int)a.size();
    if (br <= 0 || nrec < 2 * br)
        return 0.0;
    if (head && (int)head->size() != nrec)
        head = nullptr; // size mismatch (mixed-schema window): fall back to unsegmented
    using cd = std::complex<double>;

    // Absolute code-period index per record from capture-UTC -- a valve drop just skips an
    // index, so bit epochs stay aligned (vs binning by buffer position, which a gap shifts).
    std::vector<double> dt;
    dt.reserve(nrec - 1);
    for (int r = 1; r < nrec; ++r)
        dt.push_back(utc[r] - utc[r - 1]);
    std::nth_element(dt.begin(), dt.begin() + dt.size() / 2, dt.end());
    const double rec_dt = dt[dt.size() / 2]; // median step = the no-drop record period
    if (!(rec_dt > 0.0))
        return 0.0;
    std::vector<long long> cpi(nrec);
    for (int r = 0; r < nrec; ++r)
        cpi[r] = (long long)std::llround((utc[r] - utc[0]) / rec_dt);

    // Per-bit coherent sums for a given epoch phase: records are sorted by cpi, so bit indices
    // arrive non-decreasing and a bit is a run of equal floor((period+phase)/br). SEGMENTED
    // (head non-null): the head belongs to period cpi[r], the tail to period cpi[r]+1 -- a
    // record straddling a bit boundary splits across the two bits instead of smearing the
    // symbol transition through one of them (tail bit >= head bit, and the NEXT record's head
    // bit >= this tail bit since cpi is strictly increasing, so streaming order is preserved).
    auto bit_sums = [&](int phase, std::vector<cd>* out, double* powsum,
                        std::vector<long long>* out_bi = nullptr) {
        cd s(0.0, 0.0);
        long long cur = 0;
        bool have = false;
        double g = 0.0;
        int nb = 0;
        auto enter_bit = [&](long long bi) {
            if (have && bi != cur) {
                if (out)
                    out->push_back(s);
                if (out_bi)
                    out_bi->push_back(cur); // which bit that sum belonged to (gap-honest)
                g += std::abs(s);
                ++nb;
                s = cd(0.0, 0.0);
            }
            cur = bi;
            have = true;
        };
        for (int r = 0; r < nrec; ++r) {
            enter_bit((cpi[r] + phase) / br);
            if (head) {
                s += (*head)[r];
                enter_bit((cpi[r] + 1 + phase) / br); // no-op unless the boundary crosses a bit
                s += a[r] - (*head)[r];               // tail -> the period the record ends in
            } else {
                s += a[r];
            }
        }
        if (have) {
            if (out)
                out->push_back(s);
            if (out_bi)
                out_bi->push_back(cur);
            g += std::abs(s);
            ++nb;
        }
        if (powsum)
            *powsum = nb >= 2 ? g / nb : -1.0;
    };

    // Bit sync: the phase (0..br-1) maximising the mean per-bit coherent power -- UNLESS the
    // caller pinned it. The search is br full passes over the window, so on a chain with many
    // records per symbol it dominates everything else this function does; pinning is what makes
    // the ladder affordable there (see the header note, and the caller's UTC-derived shift).
    int best_phase = 0;
    if (pin_phase >= 0) {
        best_phase = ((pin_phase % br) + br) % br;
    } else {
        double best_g = -1.0;
        for (int phase = 0; phase < br; ++phase) {
            double g;
            bit_sums(phase, nullptr, &g);
            if (g > best_g) {
                best_g = g;
                best_phase = phase;
            }
        }
    }
    if (out_phase)
        *out_phase = best_phase;
    std::vector<cd> s;
    std::vector<long long> s_bi;
    bit_sums(best_phase, &s, nullptr, obs ? &s_bi : nullptr);
    if (s.size() < 2)
        return 0.0;

    // Estimate the +-1 per bit by squaring (theta0 = 1/2 arg sum s^2; the global sign cancels
    // in |.|), wipe, and coherently sum -> the deep coherent gain (mean of the wiped A). The
    // component ORTHOGONAL to the aligned signal (the imaginary part after rot) carries no signal,
    // so it measures the per-bit noise -> the deep's uncertainty and a significance SNR.
    cd sumsq(0.0, 0.0);
    for (const cd& v : s)
        sumsq += v * v;
    const cd rot = std::polar(1.0, -0.5 * std::arg(sumsq));
    cd deep(0.0, 0.0);
    double noise2 = 0.0; // sum of squared orthogonal (noise) components, one per bit
    if (obs) { // export the per-bit decisions the wipe is about to apply (P7a nav_obs)
        obs->utc_ref = utc[0];
        obs->rec_dt = rec_dt;
        obs->phase = best_phase;
        obs->br = br;
        obs->rot_re = std::real(rot);
        obs->rot_im = std::imag(rot);
        obs->bits.clear();
        obs->bits.reserve(s.size());
    }
    for (size_t bi = 0; bi < s.size(); ++bi) {
        const cd& v = s[bi];
        const cd vr = v * rot; // align the signal onto the real axis
        const double sgn = std::real(vr) >= 0.0 ? 1.0 : -1.0;
        deep += sgn * v;
        noise2 += std::imag(vr) * std::imag(vr);
        if (obs && bi < s_bi.size())
            obs->bits.emplace_back(s_bi[bi], (int8_t)(sgn > 0 ? 1 : -1));
    }
    const int nb = (int)s.size();
    const double sigma_bit = std::sqrt(noise2 / (double)nb);    // per-bit noise (1 axis)
    const double noise_sum = sigma_bit * std::sqrt((double)nb); // noise std of the coherent sum
    // SNR = significance of the deep detection (coherent sum / its noise); the deep's standard
    // error is noise_sum/nrec, so SNR == deep / its uncertainty. Real lock SNR >> 1; noise ~1.
    if (snr_out)
        *snr_out = noise_sum > 0.0 ? std::abs(deep) / noise_sum : 0.0;
    return std::abs(deep) / (double)nrec; // coherent mean of the wiped per-record A
}

double
GnssCoherentCombiner::carrier_resid_hz(const std::vector<std::complex<double>>& a,
                                       const std::vector<double>& utc,
                                       double* sigma_phi_out, bool prewiped,
                                       double* snr_q_out) const {
    const int n = (int)a.size();
    // snr_q_out defaults to NaN ("not available"): it is only produced on the LINEAR (pilot /
    // prewiped) path below, and every early return must leave it as the not-measured sentinel.
    if (snr_q_out != nullptr)
        *snr_q_out = NAN;
    // CONTRACT: this function ALWAYS writes *sigma_phi_out. Too few records to fit a
    // phase slope -> return the residual's "no measurement" value (0.0, which the broker's
    // carrier loop reads as skip -- see gps_distributed_broker.py `if resid == 0.0`) and mark
    // the jitter UNMEASURED with NaN, NOT 0.0. A 0.0 here is indistinguishable from a pristine
    // lock ("silence reads as success"); NaN ships as JSON null and reads as "not measured".
    if (n < 16) {
        if (sigma_phi_out != nullptr)
            *sigma_phi_out = NAN;
        return 0.0;
    }
    using cd = std::complex<double>;
    // Bit-robust phase-SLOPE fit: square A to cancel the +-1 nav-bit pi flips (a data signal), or
    // fit the raw phase for a dataless pilot. TWO STAGES, because a sequential record-to-record
    // unwrap alone is fragile: at 1 ms records (GPS) the window has ~1000 unwrap steps at 2x the
    // per-record phase noise, and ONE deep fade mid-window slips the accumulated phase by 2*pi,
    // biasing the slope by several Hz. The broker loop then injects that error into the NCO
    // (observed 2026-07-12: per-emit resid swings +-1..7 Hz on exactly the low-record-energy
    // sats, capping GPS coherence at 0.125-0.25 s while E/C's 4/10 ms records converged).
    //   Stage 1 -- slip-free coarse frequency: lag-L delay-and-multiply, ONE arg() of a
    //   weighted complex sum (no accumulation, a fade only down-weights its pairs).
    //   Stage 2 -- demodulate by the coarse frequency, then the weighted (|A|^2) LSQ of the
    //   sequentially-unwrapped phase vs CAPTURE-UTC (uneven gaps contribute their true span).
    //   Post-demod the true slope is ~0, so steps sit deep inside +-pi and cannot slip.
    // Raw phase ONLY for a truly dataless pilot -- an overlay pilot's pre-wipe records still
    // carry +-1 secondary chips, which scramble raw-phase fits like nav bits (see carrier_raw()).
    // SQUARE to cancel the +-1 flips ONLY when the stream still carries them: a data signal's nav
    // bits, or an overlay pilot's un-wiped secondary chips. A truly dataless pilot (carrier_raw)
    // or an already-overlay-wiped stream (prewiped) fits the raw phase -- mult 1, half the noise,
    // and no |2f-1| straddle null.
    const bool sq = !carrier_raw() && !prewiped;
    const double mult = sq ? 2.0 : 1.0; // squared phase advances at mult * the carrier
    // Stage 1: coarse f over pairs L records apart. L=4 trades variance against unambiguous
    // range: +-1/(2*L*dt*mult) (+-62 Hz at 1 ms data records) covers any post-seed residual.
    const int L = 4;
    cd pair_sum(0.0, 0.0);
    double dt_sum = 0.0;
    int n_pair = 0;
    for (int r = L; r < n; ++r) {
        const double w = std::norm(a[r]) * std::norm(a[r - L]);
        if (!(w > 0.0))
            continue;
        const cd vr = sq ? a[r] * a[r] : a[r];
        const cd vp = sq ? a[r - L] * a[r - L] : a[r - L];
        pair_sum += w * vr * std::conj(vp);
        dt_sum += utc[r] - utc[r - L];
        ++n_pair;
    }
    double f_coarse = 0.0;
    if (n_pair > 0 && std::abs(pair_sum) > 0.0 && dt_sum > 0.0)
        f_coarse = std::arg(pair_sum) / (2.0 * M_PI * (dt_sum / n_pair) * mult);
    // Stage 2: LSQ slope of the unwrapped DEMODULATED phase.
    double phi = 0.0, raw_prev = 0.0;
    bool have = false;
    double Sw = 0, Swt = 0, Swtt = 0, Swp = 0, Swtp = 0;
    const double t0 = utc[0];
    for (int r = 0; r < n; ++r) {
        const double w = std::norm(a[r]); // |A|^2
        if (!(w > 0.0))
            continue;
        const cd v = (sq ? a[r] * a[r] : a[r])
                     * std::polar(1.0, -2.0 * M_PI * f_coarse * mult * (utc[r] - t0));
        const double cur = std::arg(v); // (-pi, pi]
        if (have) {
            double d = cur - raw_prev;
            while (d > M_PI)
                d -= 2.0 * M_PI;
            while (d < -M_PI)
                d += 2.0 * M_PI;
            phi += d;
        } else {
            phi = cur;
            have = true;
        }
        raw_prev = cur;
        const double t = utc[r] - t0;
        Sw += w;
        Swt += w * t;
        Swtt += w * t * t;
        Swp += w * phi;
        Swtp += w * t * phi;
    }
    const double det = Sw * Swtt - Swt * Swt;
    if (!(std::fabs(det) > 0.0)) {
        if (sigma_phi_out != nullptr)
            *sigma_phi_out = NAN; // degenerate fit -> jitter unmeasured, not zero
        return 0.0;
    }
    const double slope = (Sw * Swtp - Swt * Swp) / det; // d(phi)/dt (rad/s)
    const double icept = (Swtt * Swp - Swt * Swtp) / det;
    // SIGMA_PHI (scintillation / multipath phase jitter): the weighted RMS of the SAME fit's
    // residuals. The fit already models the carrier the loop is meant to remove, so what is
    // left is exactly what we want -- phase that is NOT a frequency offset: multipath phase
    // swing, scintillation, and thermal noise. Divide by `mult` to undo the bit-robust
    // squaring and get radians on the true carrier. A second pass is needed because the
    // accumulators above cannot yield residuals; it is O(n) on a window already in cache.
    // MODULATION-IMMUNE (quadrature-referenced) noise -- snr_q_out. The 4th-moment incoherent
    // debias (<|A|^2>,<|A|^4> -> s2,N, feeding cn0_inc) is fooled by BOC pilots: a steep BOC ACF
    // turns code-phase jitter into large per-record AMPLITUDE modulation, which the moment debias
    // books as noise, so cn0_inc under-reads (E1C ~-7, B1C ~-13 dB). Here the carrier is fitted
    // out (f_coarse + the slope), so the signal sits at a FIXED phase and amplitude modulation
    // lives entirely on the in-phase axis; the QUADRATURE component |v|*sin(resid) is thermal
    // noise with the amplitude divided back out (a*(n_perp/a) = n_perp), so its variance is the
    // per-component noise power INDEPENDENT of the modulation. This needs a phase reference, i.e.
    // it is COHERENCE-REFERENCED -- valid only while the fit holds over the window (phase wander
    // << pi); a decohered window has no fixed signal axis and this is not meaningful. LINEAR path
    // only (sq==false: overlay-wiped or dataless pilots -- exactly the BOC signals); a squared
    // (data) fit has no single amplitude axis, so snr_q stays NaN there and cn0_inc is used.
    if (sigma_phi_out != nullptr || snr_q_out != nullptr) {
        double phi2 = 0.0, raw_prev2 = 0.0, Sw2 = 0.0, Sr2 = 0.0;
        // quadrature-noise accumulators (uniform per-record weight -- see the |v| cancellation):
        double Kq = 0.0, S_im = 0.0, S_im2 = 0.0, S_pow = 0.0;
        const bool want_q = (snr_q_out != nullptr) && !sq;
        bool have2 = false;
        for (int r = 0; r < n; ++r) {
            const double w = std::norm(a[r]);
            if (!(w > 0.0))
                continue;
            const cd v = (sq ? a[r] * a[r] : a[r])
                         * std::polar(1.0, -2.0 * M_PI * f_coarse * mult * (utc[r] - t0));
            const double cur = std::arg(v);
            if (have2) {
                double d = cur - raw_prev2;
                while (d > M_PI)
                    d -= 2.0 * M_PI;
                while (d < -M_PI)
                    d += 2.0 * M_PI;
                phi2 += d;
            } else {
                phi2 = cur;
                have2 = true;
            }
            raw_prev2 = cur;
            const double t = utc[r] - t0;
            const double resid = phi2 - (icept + slope * t);
            Sw2 += w;
            Sr2 += w * resid * resid;
            if (want_q) {
                // |v|*sin(resid) = the record's amplitude projected onto the quadrature (noise)
                // axis of the carrier-removed frame. For thermal noise resid ~ n_perp/|v|, so this
                // is ~ n_perp regardless of |v| -> modulation-immune per-record noise sample.
                const double imq = std::sqrt(w) * std::sin(resid);
                Kq += 1.0;
                S_im += imq;
                S_im2 += imq * imq;
                S_pow += w; // <|v|^2> = <|A|^2> over the same window (rotation preserves magnitude)
            }
        }
        if (sigma_phi_out != nullptr)
            *sigma_phi_out = (Sw2 > 0.0) ? std::sqrt(Sr2 / Sw2) / mult : NAN; // unmeasured -> NaN
        if (want_q && Kq >= 16.0) {
            const double mean_im = S_im / Kq;
            const double var_im = std::max(0.0, S_im2 / Kq - mean_im * mean_im); // per-component noise
            const double noise = 2.0 * var_im;             // complex thermal power / record
            const double m2 = S_pow / Kq;                  // total power / record
            const double sig = std::max(0.0, m2 - noise);  // modulation-immune signal power / record
            *snr_q_out = (noise > 1e-300) ? sig / noise : NAN; // per-record SNR; /t_rec -> C/N0 downstream
        }
    }
    // Total residual = the slip-free coarse stage PLUS the fine LSQ slope of what it left.
    return f_coarse + slope / mult / (2.0 * M_PI);      // -> residual carrier (Hz)
}
