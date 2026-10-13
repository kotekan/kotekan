#include "GnssCoherentCombiner.hpp"

#include "GnssChanMetadata.hpp"        // for get_gnss_chan_metadata (the record's absolute hop)
#include "StageFactory.hpp"            // for REGISTER_KOTEKAN_STAGE
#include "gnssChannelizedDespread.hpp" // for overlay_apply, coherent_sum
#include "gnssOverlay.hpp"             // for overlay_by_name (the secondary-overlay registry)
#include "visUtil.hpp"                 // for frameID

#include "json.hpp" // for json

#include <algorithm>  // for max
#include <cmath>      // for hypot, sqrt
#include <functional> // for bind
#include <string>     // for string

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssCoherentCombiner);

// Cap on the phase rate the deep fold may apply (Hz). It must admit the real carrier residual,
// up to ~25 Hz with the carrier open loop, or the fold derotates by a noise bin. It excludes a
// residual NH20 overlay's 50 Hz sidebands, which alias to +-45.4 Hz on the 95.37 Hz record grid;
// the harmonics alias inside it (100 Hz to 4.6 Hz, 200 Hz to 9.2 Hz), so if fold amplitudes
// drop, check for picks near 4.6 Hz steps.
static constexpr double DEEP_RATE_MAX_HZ = 25.0;
// Peak/median the rate search must reach before its rate is applied.
static constexpr double DEEP_RATE_MIN_Q = 10.0;

GnssCoherentCombiner::GnssCoherentCombiner(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssCoherentCombiner::main_thread, this)) {
    // One record stream: the node's assembler sums every covering channel, and the broker
    // combines across nodes.
    const std::vector<Buffer*> in_bufs = get_buffer_array("in_bufs");
    if (in_bufs.size() != 1) {
        FATAL_ERROR("GnssCoherentCombiner: in_bufs has {:d} buffers; exactly one is supported",
                    in_bufs.size());
        return;
    }
    in_buf = in_bufs[0];
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _n_elements = config.get<int>(unique_name, "n_elements");
    if (_n_elements < 1) {
        FATAL_ERROR("GnssCoherentCombiner: n_elements {:d} must be >= 1", _n_elements);
        return;
    }
    _rec_stride = gnss::record_stride(_n_elements);

    _n_prn = config.get<int>(unique_name, "n_prn");
    if (_n_prn < 1) {
        FATAL_ERROR("GnssCoherentCombiner: n_prn {:d} must be >= 1", _n_prn);
        return;
    }
    // The record width is a C++ constant but the frames are sized in yaml, so check that every
    // frame holds n_prn records.
    {
        const size_t need = (size_t)_n_prn * _rec_stride * sizeof(float);
        if ((size_t)in_buf->frame_size < need) {
            FATAL_ERROR("GnssCoherentCombiner: input frame {:d} B < {:d} PRN x {:d} floats "
                        "({:d} B). Bump 'record_floats' in the config to {:d}.",
                        (size_t)in_buf->frame_size, _n_prn, _rec_stride, need, _rec_stride);
            return;
        }
        if ((size_t)out_buf->frame_size < need) {
            FATAL_ERROR("GnssCoherentCombiner: out_buf frame {:d} B < {:d} PRN x {:d} floats "
                        "({:d} B). Bump 'record_floats' in the config to {:d}.",
                        (size_t)out_buf->frame_size, _n_prn, _rec_stride, need, _rec_stride);
            return;
        }
    }

    _integration_length =
        std::max(1, config.get_default<int>(unique_name, "integration_length", 1));
    // Integration is a rolling EMA (see the hpp); the key remains only to refuse another mode.
    const std::string mode =
        config.get_default<std::string>(unique_name, "integration_mode", "rolling");
    if (mode != "rolling") {
        FATAL_ERROR("GnssCoherentCombiner: integration_mode '{:s}' is not supported; it must be "
                    "'rolling'",
                    mode);
        return;
    }
    // Output cadence: ~10 outputs per time constant.
    _emit_every = std::max(1, _integration_length / 10);
    _navwipe_bit_records = config.get_default<int>(unique_name, "navwipe_bit_records", 0);
    // secondary_overlay: a shared overlay over a data component (GPS L5-I: CNAV under NH10). The
    // overlay is wiped first and the nav-bit wipe runs on the result, so it needs
    // navwipe_bit_records.
    const std::string sec = config.get_default<std::string>(unique_name, "secondary_overlay", "");
    // The overlay comes from the gnssOverlay.cpp registry; an empty name means no overlay.
    if (!sec.empty()) {
        const gnss::OverlayDescriptor* ov = gnss::overlay_by_name(sec);
        if (!ov || _navwipe_bit_records <= 0) {
            FATAL_ERROR("GnssCoherentCombiner: secondary_overlay {:s} must be a gnssOverlay.cpp "
                        "row and needs navwipe_bit_records > 0 (the overlay is wiped only ahead "
                        "of the nav-bit wipe)",
                        sec);
            return;
        }
        _secondary = ov->generate();
        // SignalDescriptor::secondary_length must agree with the registry: ChannelizedReplicaBank
        // applies shared overlays based on it.
        for (const std::string& msg : gnss::overlay_registry_check())
            WARN("overlay registry/descriptor mismatch: {:s}", msg);
    }
    // Without a nav-bit wipe the deep |A| is the plain coherent sum, with a phase-rate search
    // and the sky-corrected prompt as a second candidate. These keys remain only to refuse any
    // other setting.
    if (!config.get_default<bool>(unique_name, "deep_coherent", true)
        || !config.get_default<bool>(unique_name, "deep_rate_search", true)
        || !config.get_default<bool>(unique_name, "sky_deep", true)
        || config.get_default<bool>(unique_name, "phase_track", false)
        || config.get_default<double>(unique_name, "deep_rate_min_q", DEEP_RATE_MIN_Q)
               != DEEP_RATE_MIN_Q) {
        FATAL_ERROR("GnssCoherentCombiner: deep_coherent, deep_rate_search and sky_deep must be "
                    "true, phase_track false and deep_rate_min_q {:g}",
                    DEEP_RATE_MIN_Q);
        return;
    }
    // Converts sample_seq to the absolute hop index (see the hpp).
    _fft_len = config.get<int>(unique_name, "fft_len");
    if (_fft_len < 1) {
        FATAL_ERROR("GnssCoherentCombiner: fft_len {:d} must be >= 1", _fft_len);
        return;
    }

    // Real-time cost (see the hpp). Duty cycle = (d ingest_us + d emit_us) / (d records x record
    // period): below 1 the stage keeps up; at or above 1 it is saturated and more buffering only
    // delays the drops.
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
    _navbuf.assign(_n_prn, {});
    _navsky.assign(_n_prn, {});
    _navutc.assign(_n_prn, {});
    _navhead.assign(_n_prn, {});
    _bit_export = config.get_default<bool>(unique_name, "bit_export", false);
    if (_bit_export)
        _st_nav_obs.assign(_n_prn, {});

    _st_prn.assign(_n_prn, 0);
    _st_amp.assign(_n_prn, 0.0f);
    _st_coh.assign(_n_prn, 0.0f);
    _st_deep.assign(_n_prn, 0.0f);
    _st_deep_snr.assign(_n_prn, 0.0f);
    _st_amp_snr.assign(_n_prn, 0.0f);
    _st_dop.assign(_n_prn, 0.0f);
    _st_cp.assign(_n_prn, 0.0f);
    _st_dll_disc.assign(_n_prn, 0.0f);
    _st_e_pow.assign(_n_prn, 0.0f);
    _st_p_pow.assign(_n_prn, 0.0f);
    _st_l_pow.assign(_n_prn, 0.0f);
    _st_nchan.assign(_n_prn, 0.0f);
    _st_head_frac.assign(_n_prn, 0.0f);
    _st_s4.assign(_n_prn, 0.0f);
    _st_s4_raw.assign(_n_prn, 0.0f);
    _st_sigma_phi.assign(_n_prn, 0.0f);
    _st_car_resid.assign(_n_prn, 0.0f);
    _adr_cyc.assign(_n_prn, 0.0);
    _trim_cyc.assign(_n_prn, 0.0);
    _res_cyc.assign(_n_prn, 0.0);
    _st_res.assign(_n_prn, 0.0);
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
    _st_deep_rec.assign(_n_prn, 0);

    _st_deep_rate.assign(_n_prn, 0.0f);
    _st_deep_rate_q.assign(_n_prn, 0.0f);
    _st_deep_rate_full.assign(_n_prn, 0.0f);
    _st_deep_rate_full_q.assign(_n_prn, 0.0f);
    _st_coh_frac.assign(_n_prn, 0.0f);

    // record_export: keep the last N per-record prompts per PRN for /get_records, so the broker
    // can sum the complex prompts across instances (each sees only part of the band). Each entry
    // is [hop, re, im, energy]: hop is the exact cross-node alignment key (every node's
    // sample_seq comes from one F-engine counter), re/im are energy-normalized, and energy is
    // the ML combining weight. Off by default; too large for get_status.
    _rec_export = config.get_default<int>(unique_name, "record_export", 0);
    if (_rec_export > 0) {
        _recex.assign(_n_prn, {});
        _st_recex.assign(_n_prn, {});
    }
}

void GnssCoherentCombiner::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_status",
        std::bind(&GnssCoherentCombiner::get_status_callback, this, _1));
    if (_rec_export > 0)
        kotekan::restServer::instance().register_get_callback(
            unique_name + "/get_records",
            std::bind(&GnssCoherentCombiner::get_records_callback, this, _1));
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_elements",
        std::bind(&GnssCoherentCombiner::get_elements_callback, this, _1));

    frameID in_id(in_buf);
    frameID out_id(out_buf);

    // Per-PRN integration accumulators: exponential moving averages (alpha =
    // 1/_integration_length), bias-corrected at emit by dividing by (1 - (1-alpha)^n_roll) so
    // they read true running means from record 1.
    std::vector<double> acc_pow(_n_prn), acc_ar(_n_prn), acc_ai(_n_prn), acc_nchan(_n_prn);
    std::vector<double> acc_pow2(_n_prn); // <|A|^4>, for the noise-debiased incoherent significance
    std::vector<double> acc_epow(_n_prn), acc_lpow(_n_prn); // <|E|^2>, <|L|^2> for the DLL disc
    // Raw prompt power <|A|^2>, built like acc_epow/acc_lpow (no overlay wipe, no head/tail
    // split), for the P in q = 2P/(E+L). acc_pow takes a straddle-immune form instead, whose
    // segment split alone scales it by f^2+(1-f)^2 (0.5 at a mid-record boundary), so it cannot
    // be compared with the unsegmented E and L. Published only.
    std::vector<double> acc_ppow(_n_prn);
    // Boundary fraction f = <head energy>/<prompt energy>: where the code-period boundary sits
    // in the window (0 or 1 = at an edge). Diagnostic only.
    std::vector<double> acc_he(_n_prn), acc_pe(_n_prn);
    // Shared carrier loop: vector-averaged cross-record phase product per PRN over the emit
    // window. S += (A_k conj A_{k-1})^2; at emit resid_hz = arg(S)/2/(2 pi <dt>).
    std::vector<std::complex<double>> car_S(_n_prn);
    std::vector<double> car_dt(_n_prn);
    std::vector<int> car_n(_n_prn);
    std::vector<std::complex<double>> car_prev(_n_prn);
    std::vector<double> car_prev_utc(_n_prn, -1.0);
    std::vector<float> ref_prn(_n_prn), ref_dop(_n_prn), ref_cp(_n_prn);
    std::vector<double> ref_utc(_n_prn);
    // Per-antenna state.
    _acc_epow_el.assign((size_t)_n_prn * _n_elements, 0.0);
    _acc_ear.assign((size_t)_n_prn * _n_elements, 0.0);
    _acc_eai.assign((size_t)_n_prn * _n_elements, 0.0);
    _acc_eu_r.assign((size_t)_n_prn * _n_elements, 0.0);
    _acc_eu_i.assign((size_t)_n_prn * _n_elements, 0.0);
    _acc_euq.assign((size_t)_n_prn * _n_elements, 0.0);
    long long n_roll = 0;   // total records seen (for EMA bias correction)
    long long win_hop = -1; // absolute hop index labelling the current E/P/L window
    int since_emit = 0;     // records since the last emit
    const double alpha = 1.0 / (double)_integration_length; // EMA weight

    while (!stop_thread) {
        float* in = (float*)in_buf->wait_for_full_frame(unique_name, in_id);
        if (in == nullptr)
            break;

        // Absolute hop index of this record's window, from the frame metadata
        // (GnssGpuRecordAssemble sets sample_seq to the window start). Every node's sample_seq
        // derives from the same F-engine counter, so equal hop is the same window exactly, with
        // no tolerance as UTC would need; the fleet DLL groups on it. cur_hop is this record's
        // hop, for the per-record export; win_hop labels the emitted window (its newest record,
        // as ref_utc is).
        long long cur_hop = -1;
        {
            const GnssChanMetadata* mi = get_gnss_chan_metadata(in_buf, in_id);
            cur_hop = (mi && mi->sample_seq >= 0) ? mi->sample_seq / _fft_len : -1;
            win_hop = cur_hop;
        }

        // The cost clock starts after the input wait: blocking on input is idle time, and
        // counting it would make an under-loaded stage look saturated.
        const double _t_ingest0 = current_time();

        for (int p = 0; p < _n_prn; ++p) {
            // The un-normalized correlation and replica energy over the covering channels, and
            // from them the full-band amplitude.
            const float* ref = in + (size_t)p * _rec_stride;
            const double gr = ref[3], gi = ref[4], energy = ref[5], nchan = ref[6];
            // Early/Late (gnssRecord.hpp): only their powers feed the DLL discriminator.
            const double er = ref[gnss::REC_E_RE], ei = ref[gnss::REC_E_IM],
                         e_energy = ref[gnss::REC_E_ENERGY];
            const double lr = ref[gnss::REC_L_RE], li = ref[gnss::REC_L_IM],
                         l_energy = ref[gnss::REC_L_ENERGY];
            // Prompt head segment (slots 16-18).
            const double ghr = ref[gnss::REC_PH_RE], ghi = ref[gnss::REC_PH_IM],
                         h_energy = ref[gnss::REC_PH_ENERGY];
            // Sky-phase-corrected prompt (REC_SKY_*).
            const double skr = ref[gnss::REC_SKY_RE], ski = ref[gnss::REC_SKY_IM];
            const double ar = energy > 0.0 ? gr / energy : 0.0;
            const double ai = energy > 0.0 ? gi / energy : 0.0;
            // Per-antenna amplitude A_e = G_e / E, normalized by the shared replica energy E that
            // every antenna correlated against, so values are comparable across antennas (the
            // beam map is the relative response across dishes). Accumulated like the prompt.
            // |A_e|^2 is incoherent and needs no per-antenna phase model; <A_e> keeps the phase.
            {
                const float* eb0 = in + gnss::elem_offset((int)p, 0, _n_elements);
                // Unweighted array sum of the raw per-antenna prompts, for the leave-one-out
                // reference below. The raw G_e are upstream of the element cal, so no element's
                // noise can enter its own reference through a fitted weight.
                double gsum_r = 0.0, gsum_i = 0.0;
                for (int el = 0; el < _n_elements; ++el) {
                    const float* eb = eb0 + (size_t)el * gnss::ELEM_FLOATS;
                    gsum_r += eb[gnss::ELEM_P_RE];
                    gsum_i += eb[gnss::ELEM_P_IM];
                }
                for (int el = 0; el < _n_elements; ++el) {
                    const float* eb = eb0 + (size_t)el * gnss::ELEM_FLOATS;
                    const double ger = eb[gnss::ELEM_P_RE], gei = eb[gnss::ELEM_P_IM];
                    const double aer = energy > 0.0 ? ger / energy : 0.0;
                    const double aei = energy > 0.0 ? gei / energy : 0.0;
                    const double p2e = aer * aer + aei * aei;
                    // Complex gain product A_e * conj(ref_e), with ref_e the other elements' sum
                    // in this record. The sky phase is common to both and cancels per record,
                    // which the raw mean _acc_ear cannot do (it averages the phase itself and
                    // collapses). The element's noise is uncorrelated with its leave-one-out
                    // reference, so the mean is unbiased for g_e * <|s_ref|^2> with no debias.
                    const double rr = energy > 0.0 ? (gsum_r - ger) / energy : 0.0;
                    const double ri = energy > 0.0 ? (gsum_i - gei) / energy : 0.0;
                    const double eur = aer * rr + aei * ri;
                    const double eui = aei * rr - aer * ri;
                    const double euq = rr * rr + ri * ri;
                    const size_t k = (size_t)p * _n_elements + el;
                    _acc_epow_el[k] += alpha * (p2e - _acc_epow_el[k]);
                    _acc_ear[k] += alpha * (aer - _acc_ear[k]);
                    _acc_eai[k] += alpha * (aei - _acc_eai[k]);
                    _acc_eu_r[k] += alpha * (eur - _acc_eu_r[k]);
                    _acc_eu_i[k] += alpha * (eui - _acc_eu_i[k]);
                    _acc_euq[k] += alpha * (euq - _acc_euq[k]);
                }
            }
            // Head amplitude normalized by the total prompt energy, so head + tail == A and the
            // segmented wipe works in the same units as the unsegmented one. A record without a
            // head segment (h_energy == 0) is treated below as all head (tail 0).
            const double ahr = energy > 0.0 ? ghr / energy : 0.0;
            const double ahi = energy > 0.0 ? ghi / energy : 0.0;
            // Sky-corrected amplitude, normalized like A so the deep estimators and their floors
            // apply unchanged. Zero means absent (cold cal or elem_sum off).
            const std::complex<double> A_sky(energy > 0.0 ? skr / energy : 0.0,
                                             energy > 0.0 ? ski / energy : 0.0);
            const double e2 = e_energy > 0.0 ? (er * er + ei * ei) / (e_energy * e_energy) : 0.0;
            const double l2 = l_energy > 0.0 ? (lr * lr + li * li) / (l_energy * l_energy) : 0.0;

            const double utc_p = *reinterpret_cast<const double*>(ref + RECORD_UTC_SLOT);
            const std::complex<double> A(ar, ai);
            // Incoherent power for the |A|^2 and |A|^4 moments (amp_snr, s4), in the first
            // straddle-immune form available: |head|^2 + |tail|^2, immune to the overlay sign
            // (only a mild f^2+(1-f)^2 factor); else |A|^2. On an overlay signal plain |A|^2 nulls
            // at boundary_f = 0.5, and the moment debias would read that bimodality as noise.
            double p2;
            if (h_energy > 0.0) {
                const std::complex<double> H(ahr, ahi);
                p2 = std::norm(H) + std::norm(A - H);
            } else {
                p2 = ar * ar + ai * ai;
            }
            {
                // Full-band carrier phase walk against the previous record, gated on the gap (a
                // dropped record would alias the product). Squared: cancels nav bits and overlay
                // chips.
                const double dt_c = utc_p - car_prev_utc[p];
                if (car_prev_utc[p] > 0.0 && dt_c > 0.0 && dt_c < 0.050) {
                    std::complex<double> prod = A * std::conj(car_prev[p]);
                    prod *= prod;
                    car_S[p] += prod;
                    car_dt[p] += dt_c;
                    car_n[p] += 1;

                    // Accumulated carrier phase (ADR). Phi_rx = Phi_cmd - arg(A)/2pi
                    // (gnssRecord.hpp), accumulated in per-record increments so neither half
                    // needs unwrapping: the tracker sends the commanded increment (record slot
                    // 15), and the measured increment is this phase product, which inherits its
                    // bit and overlay robustness (squaring halves the unambiguous range to +-0.25
                    // cycles per record). Double precision: ADR reaches ~1e6 cycles in a pass,
                    // where float32 resolves only 0.06 cycles, so it ships via REST rather than
                    // a record slot.
                    if (energy <= 0.0) {
                        _adr_ok[p] = 0; // not despread this record (inactive PRN): no phase
                    } else if (_adr_ok[p]) {
                        // The commanded increment comes ready-made from the tracker (record slot
                        // 15): bounded and float-exact, with nothing to unwrap.
                        const double dcmd = ref[gnss::REC_CPHASE];
                        // dres is subtracted: arg(A) is the despread residual in the r2c-flipped
                        // internal convention, like the NCO and the trim, while the commanded
                        // phase is physical. With the wrong sign the ADR rate is off by twice
                        // the satellite's mean carrier residual, which is invisible while the
                        // carrier loop holds the residual near zero.
                        const double dres = std::arg(prod) / (2.0 * M_PI * 2.0);
                        _adr_cyc[p] += dcmd - dres;
                        _res_cyc[p] += dres;
                        // The commanded-trim increment (slot 19) follows the ADR's arc
                        // lifecycle, so downstream can subtract trim_cycles from adr_cycles per
                        // arc.
                        _trim_cyc[p] += ref[gnss::REC_TRIM_INC];
                        _adr_n[p] += 1;
                    } else {
                        // Arc start: the phase is ambiguous by an integer number of cycles,
                        // so every arc begins at zero and downstream levels each arc on its
                        // own (the standard carrier-phase ambiguity, one per arc).
                        _adr_cyc[p] = 0.0;
                        _trim_cyc[p] = 0.0;
                        _res_cyc[p] = 0.0;
                        _adr_arc[p] += 1;
                        _adr_t0[p] = utc_p;
                        _adr_n[p] = 0;
                        _adr_ok[p] = 1;
                    }
                } else if (car_prev_utc[p] > 0.0 || dt_c <= 0.0) {
                    // A gap (dropped record, PRN dropped or re-seeded, records out of order)
                    // loses count of whole cycles, so break the arc rather than slip silently.
                    _adr_ok[p] = 0;
                }
                car_prev[p] = A;
                car_prev_utc[p] = utc_p;
            }
            // Exponential moving average (no reset; integrates indefinitely).
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
            if (energy > 0.0) {
                // Sliding window of the last _integration_length records for the deep
                // estimators. Records with no despread (PRN inactive this window) are not
                // buffered: the wipes index by UTC, so they become gaps like dropped records.
                // Zeros would dilute the amplitude without adding to the orthogonal-noise
                // estimate, inflating the significance.
                _navbuf[p].emplace_back(ar, ai);
                _navsky[p].push_back(A_sky);
                _navutc[p].push_back(utc_p);
                // Head segment; without one the whole record is the head, which reduces the
                // segmented wipe to the unsegmented one.
                _navhead[p].emplace_back(h_energy > 0.0 ? ahr : ar, h_energy > 0.0 ? ahi : ai);
                if ((int)_navbuf[p].size() > _integration_length) {
                    _navbuf[p].erase(_navbuf[p].begin());
                    _navsky[p].erase(_navsky[p].begin());
                    _navutc[p].erase(_navutc[p].begin());
                    _navhead[p].erase(_navhead[p].begin());
                }
            }
            if (_rec_export > 0 && energy > 0.0)
                push_recex(p, cur_hop, ar, ai, energy);
            ref_prn[p] = ref[0]; // reference = the latest record
            ref_dop[p] = ref[1];
            ref_cp[p] = ref[2];
            ref_utc[p] = utc_p;
        }

        if (_m_ingest_us)
            _m_ingest_us->inc((uint64_t)std::max(0.0, (current_time() - _t_ingest0) * 1e6));
        if (_m_records)
            _m_records->inc(); // the data clock: records consumed (x record period = data time)

        in_buf->mark_frame_empty(unique_name, in_id++);

        ++n_roll;
        if (++since_emit < _emit_every)
            continue; // EMA keeps integrating; emit only every _emit_every records
        since_emit = 0;

        float* out = (float*)out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;
        // Again after the blocking wait: downstream backpressure is not this stage's cost.
        const double _t_emit0 = current_time();
        // The EMA is already a mean, so just undo the warm-up bias (1-(1-alpha)^n_roll) -> a
        // true running mean from record 1.
        const double inv = 1.0 / (1.0 - std::pow(1.0 - alpha, (double)n_roll));
        std::vector<float> p_pow_out(_n_prn, 0.0f); // fleet-DLL prompt power (see acc_ppow)
        std::vector<double> deep_snr(_n_prn, 0.0);  // significance of the deep detection (for REST)
        std::vector<double> amp_snr(_n_prn, 0.0);   // noise-debiased incoherent significance (REST)
        std::vector<double> s4_raw(_n_prn, 0.0);    // amplitude scintillation index (raw)
        // Carrier-phase jitter (rad); NaN = not measured this emit. Only carrier_resid_hz sets it.
        std::vector<double> sigma_phi(_n_prn, NAN);
        std::vector<double> coh_s(_n_prn, 0.0); // measured coherence: span of the winning window
        std::vector<int> deep_rec(_n_prn, 0);   // records in the winning deep window
        std::vector<double> deep_floor(_n_prn, 0.0);     // rectification floor at the reported rung
        std::vector<double> deep_rate(_n_prn, 0.0);      // phase-rate removed before the deep sum
        std::vector<double> deep_rate_q(_n_prn, 0.0);    // its peak/median, which gates it
        std::vector<double> deep_rate_full(_n_prn, 0.0); // uncapped argmax: the carrier loop's
        std::vector<double> deep_rate_full_q(_n_prn, 0.0); // measurement, not the fold's
        std::vector<double> coh_frac(_n_prn, 0.0); // |sum|/sum|.| of the winning deep stream
        std::vector<NavObs> nav_obs(_bit_export ? _n_prn : 0); // full-window bit decisions
        // Each deep estimator's free parameters (alignment, bit signs, candidate choice) rectify
        // pure noise to a floor, so without a gate an unlocked satellite would report coherence
        // over the full window. A rung must reach FLOOR_MARGIN x its floor to count; when none
        // does, coherence_s = 0 and the full-window numbers are reported for diagnostics only.
        constexpr double FLOOR_MARGIN = 2.0;
        // Auto-coherence ladder: integration_length is a maximum. The deep estimator runs on the
        // full buffer and up to 4 octave-shorter trailing windows of at least min_len records,
        // and the best deep SNR wins, so coherence_s reports the longest span that actually
        // coheres. Walked longest-first; a shorter window must beat the incumbent deep SNR by
        // 5%, so ties go to the longer window.
        auto ladder = [&](size_t n, size_t min_len) {
            std::vector<size_t> lens;
            if (n > 0)
                lens.push_back(n);
            for (size_t l = n / 2; l >= std::max<size_t>(min_len, 2) && lens.size() < 5; l /= 2)
                lens.push_back(l);
            return lens;
        };
        // Shortest ladder rung: 64 ms, converted to records from the observed spacing since the
        // record period differs between bands, and never below the caller's hard_min records.
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
        // Linear phase-rate search over one rung: best_f is the rate (Hz) whose removal maximises
        // the coherent sum, limited to |f| <= DEEP_RATE_MAX_HZ, and q its peak/median. The median
        // and the optional uncapped argmax full_f / full_q use every bin. Records are uniform in
        // hop, so (utc - utc0)/dt is an integer even across gaps and a hole contributes exactly
        // zero; dt is the minimum consecutive spacing, since with gaps the mean is not the record
        // period.
        auto rate_search = [](const std::vector<std::complex<double>>& x,
                              const std::vector<double>& u, double& best_f, double& q,
                              double* full_f = nullptr, double* full_q = nullptr) {
            best_f = 0.0;
            q = 0.0;
            if (full_f)
                *full_f = 0.0;
            if (full_q)
                *full_q = 0.0;
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
            // Up to 4x oversampled (nb is capped at 4096): the true rate then lies within 1/8 of
            // the natural resolution 1/((span+1) dt) of a bin, costing at most ~2.5% of the
            // amplitude.
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
            size_t ip = 0, ip_full = 0;
            double best_m = -1.0, best_m_full = -1.0;
            for (size_t i = 0; i < nb; ++i) {
                const double fr = (double)i / (double)nb;
                const double f_i = (fr < 0.5 ? fr : fr - 1.0) / dt;
                if (mag[i] > best_m_full) {
                    best_m_full = mag[i];
                    ip_full = i;
                }
                if (std::abs(f_i) > DEEP_RATE_MAX_HZ)
                    continue; // outside the physical window: may set the floor, cannot win
                if (mag[i] > best_m) {
                    best_m = mag[i];
                    ip = i;
                }
            }
            std::vector<double> srt(mag);
            std::nth_element(srt.begin(), srt.begin() + nb / 2, srt.end());
            const double med = srt[nb / 2];
            q = med > 0.0 ? mag[ip] / med : 0.0;
            // Bins run 0..nb-1 over one full record-rate period; fold the upper half negative so
            // the reported rate is the signed offset a carrier loop could actually consume.
            const double frac = (double)ip / (double)nb;
            best_f = (frac < 0.5 ? frac : frac - 1.0) / dt;
            if (full_f) {
                // Parabolic peak interpolation for the carrier loop's measurement: the grid
                // quantizes to 1/(nb dt), ~0.19 Hz at 128 records of 10.5 ms. The fold's own
                // pick stays bin-centred, so deep_snr does not depend on this.
                const double m0 = mag[(ip_full + nb - 1) % nb];
                const double m1 = mag[ip_full];
                const double m2 = mag[(ip_full + 1) % nb];
                const double den = m0 - 2.0 * m1 + m2;
                double db = 0.0;
                if (den < 0.0) // a genuine peak; flat/noise keeps the bin centre
                    db = std::max(-0.5, std::min(0.5, 0.5 * (m0 - m2) / den));
                const double fr2 = ((double)ip_full + db) / (double)nb;
                double f2v = fr2 - std::floor(fr2);
                *full_f = (f2v < 0.5 ? f2v : f2v - 1.0) / dt;
            }
            if (full_q)
                *full_q = med > 0.0 ? best_m_full / med : 0.0;
        };
        // Duration of a trailing window of len records. _navutc holds record start times, so
        // first-to-last spans len-1 periods and the window is one period longer.
        auto coh_span = [](const std::vector<double>& u, size_t len) {
            if (u.empty() || len < 2 || u.size() < len)
                return 0.0;
            return (u.back() - u[u.size() - len]) * (double)len / (double)(len - 1);
        };
        const double Keff = (double)_integration_length;
        for (int p = 0; p < _n_prn; ++p) {
            float* rec = out + (size_t)p * _rec_stride;
            for (int f = 0; f < _rec_stride; ++f) // whole record: element blocks too
                rec[f] = 0.0f;
            rec[0] = ref_prn[p];
            rec[1] = ref_dop[p];
            rec[2] = ref_cp[p];
            rec[3] = (float)std::sqrt(acc_pow[p] * inv);            // |A|_incoh = sqrt<|A|^2>
            rec[4] = (float)(acc_ar[p] * inv);                      // <A>.re (coherent mean)
            rec[5] = (float)(acc_ai[p] * inv);                      // <A>.im
            rec[6] = (float)std::hypot(acc_ar[p], acc_ai[p]) * inv; // |<A>|_coh
            rec[7] = (float)(acc_nchan[p] * inv);                   // covering channels used
            // Per-antenna beam map, with the same `inv` normalization as rec[3], so
            // CMB_ELEM_AMP_INCOH is directly comparable with it.
            {
                // The element-gain REST snapshot: same emit and normalization, under the status
                // mutex since REST reads it from another thread.
                std::lock_guard<std::mutex> elk(_st_mtx);
                if (_st_elem.size() != (size_t)_n_prn * _n_elements * 4)
                    _st_elem.assign((size_t)_n_prn * _n_elements * 4, 0.0f);
                _st_elem_keff = Keff;
                for (int el = 0; el < _n_elements; ++el) {
                    float* eb = rec + gnss::RECORD_FLOATS + (size_t)el * gnss::ELEM_FLOATS;
                    const size_t k = (size_t)p * _n_elements + el;
                    const double mr = _acc_ear[k] * inv, mi = _acc_eai[k] * inv;
                    eb[gnss::CMB_ELEM_AMP_INCOH] = (float)std::sqrt(_acc_epow_el[k] * inv);
                    eb[gnss::CMB_ELEM_MEAN_RE] = (float)mr;
                    eb[gnss::CMB_ELEM_MEAN_IM] = (float)mi;
                    eb[gnss::CMB_ELEM_AMP_COH] = (float)std::hypot(mr, mi);
                    float* se = _st_elem.data() + k * 4;
                    se[0] = (float)(_acc_eu_r[k] * inv);
                    se[1] = (float)(_acc_eu_i[k] * inv);
                    se[2] = (float)(_acc_epow_el[k] * inv);
                    se[3] = (float)(_acc_euq[k] * inv);
                }
            }
            // DLL observables (gnssRecord.hpp): window-averaged full-band Early/Late powers and
            // the normalized early-minus-late discriminator, roughly linear in the code-phase
            // error within the correlator spacing. The broker closes the loop from these.
            {
                const double e_pow = acc_epow[p] * inv;
                const double l_pow = acc_lpow[p] * inv;
                rec[gnss::CMB_E_POW] = (float)e_pow;
                rec[gnss::CMB_L_POW] = (float)l_pow;
                rec[gnss::CMB_DLL_DISC] =
                    (e_pow + l_pow) > 0.0 ? (float)((e_pow - l_pow) / (e_pow + l_pow)) : 0.0f;
                // Fleet DLL (docs/CHORD_GNSS_SHARED_DLL.md): the prompt power, for the status
                // snapshot. E and L are in the record and P is not, so it is carried out here.
                p_pow_out[p] = (float)(acc_ppow[p] * inv);
                // Boundary fraction f (diagnostic; the inv normalization cancels in the ratio).
                rec[gnss::CMB_HEAD_FRAC] = acc_pe[p] > 0.0 ? (float)(acc_he[p] / acc_pe[p]) : 0.0f;
                // Shared carrier observable: the full-band residual carrier (Hz), which the
                // broker integrates into carrier_trim_hz for every subband tracker. Preferred
                // estimate: a bit-robust phase-slope fit over the whole deep-wipe window, whose
                // long baseline gives a low-variance frequency. The short-baseline
                // consecutive-record product car_S is the fallback while the window holds fewer
                // than 16 records.
                double resid_hz = 0.0;
                if (_navbuf[p].size() >= 16) {
                    resid_hz = carrier_resid_hz(_navbuf[p], _navutc[p], &sigma_phi[p]);
                } else if (car_n[p] > 0 && std::abs(car_S[p]) > 0.0) {
                    const double dphi = std::arg(car_S[p]) / 2.0; // squared product
                    resid_hz = dphi / (2.0 * M_PI * (car_dt[p] / car_n[p]));
                }
                rec[gnss::CMB_CARRIER_RESID] = (float)resid_hz;
                // Phase arc id, the recorded slip history. The ADR itself is a double and ships
                // via get_status (adr_cycles); the float slot says whether this emit is on the
                // same unbroken arc as the last.
                rec[gnss::CMB_ARC] = (float)_adr_arc[p];
                car_S[p] = {0.0, 0.0};
                car_dt[p] = 0.0;
                car_n[p] = 0;
            }
            // Noise-debiased incoherent significance. <|A|^2> = s^2 + N includes the noise floor
            // N, but the moments separate the two: s^2 = sqrt(<|A|^2>^2 - Var(|A|^2)),
            // N = <|A|^2> - s^2. The resulting SNR stays small for noise and grows ~sqrt(K) for a
            // signal, with no nav-bit wipe.
            {
                const double m2 = acc_pow[p] * inv, m4 = acc_pow2[p] * inv;
                const double var = std::max(0.0, m4 - m2 * m2);
                const double s2 = std::sqrt(std::max(0.0, m2 * m2 - var));
                const double noise = m2 - s2;
                // Normalise by the H0 std of the s^2 estimator (~N/K^{1/4}) so noise reads ~1
                // regardless of K.
                amp_snr[p] = noise > 1e-12 ? s2 * std::pow(Keff, 0.25) / noise : 0.0;
                // Raw S4 amplitude-scintillation index, S4^2 = Var(|A|^2) / <|A|^2>^2, a direct
                // multipath detector. Its thermal floor cannot come from the moment debias
                // above, which defines the noise as everything that is not the mean and would
                // give S4 = 0 identically; it is removed at publish time using the coherent
                // deep SNR, which is independent of amplitude modulation.
                s4_raw[p] = (m2 > 1e-12) ? std::sqrt(var) / m2 : 0.0;
            }
            if (!_secondary.empty() && _navwipe_bit_records > 0) {
                // Overlay and nav-bit wipes composed, for a data component under a shared overlay
                // (GPS L5-I: CNAV under NH10): the overlay comes off each record first, then the
                // nav-bit wipe runs on the result. The alignment cannot be chosen by maximising
                // a coherent sum over the whole window, which the data symbols destroy, so it is
                // ranked by a statistic blind to the symbol signs (below).
                const auto& ab = _navbuf[p];
                const auto& ub = _navutc[p];
                const auto& hb = _navhead[p];
                const std::vector<int8_t>& ov = _secondary;
                const size_t L = std::max<size_t>(ov.size(), 1);
                // Floor = the nav-bit wipe's floor (nb free symbol signs) plus the selection over
                // L overlay alignments; without the second term noise could clear the bar.
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
                // The alignment is ranked once, on the full window (best_ph, relative to its
                // first record), and passed to overlay_apply for every rung; a per-rung search
                // would cost L passes per rung and add selection noise the floor must pay for.
                // The rank statistic is O(n), unlike navwipe_amplitude with its own bit-phase
                // search: the sum over fixed navwipe_bit_records-record blocks of |sum of the
                // wiped records|^2. A wrong alignment leaves +-1 overlay chips inside each block,
                // which cancel; the right one leaves only the data sign, which |.|^2 discards.
                // The block grid ignores the symbol epoch, which costs every candidate the same
                // and so leaves the argmax alone. Selection is reliable at per-record S >~ 0.25;
                // below it a wrong pick despreads mostly noise, which the floor gate rejects.
                int best_ph = -1;
                {
                    double best = -1.0;
                    const size_t B = (size_t)std::max(_navwipe_bit_records, 1);
                    for (size_t ph = 0; ph < L; ++ph) {
                        if (!gnss::overlay_apply(ab, ub, ov, (int)ph, &hb, aw, &hw))
                            continue;
                        // Power, not magnitude: under noise E[|s|^2] does not depend on the
                        // alignment, so it biases every candidate equally, while the signal term
                        // adds linearly.
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
                    // The bit epoch is searched once, on the longest rung, and pinned for the
                    // rest: navwipe's bit sync costs navwipe_bit_records passes per call. The
                    // shorter rungs are suffixes of that window, so the pinned epoch is exact
                    // once shifted into each rung's frame. The phase is window-relative, so the
                    // shift is the windows' record-index offset, derived from UTC and never from
                    // buffer positions (a dropped record skips an index).
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
                        // Shift the pinned epoch into this rung's frame (0 on the full window).
                        int pin_here = -1;
                        if (pin >= 0 && rec_dt_full > 0.0) {
                            const long long d =
                                (long long)std::llround((us.front() - ub.front()) / rec_dt_full);
                            const long long m =
                                ((pin + d) % _navwipe_bit_records + _navwipe_bit_records)
                                % _navwipe_bit_records;
                            pin_here = (int)m;
                        }
                        // Export the per-symbol decisions on the full window only, as in the
                        // plain nav-bit path.
                        const double amp = navwipe_amplitude(
                            aw, us, &snr, &hw, (is_full && _bit_export) ? &nav_obs[p] : nullptr,
                            pin_here, &used_ph);
                        if (pin < 0 && used_ph >= 0)
                            pin = used_ph;   // first (longest) rung searched: pin the rest to it
                        if (full_len == 0) { // ladder walks longest-first
                            full_amp = amp;
                            full_snr = snr;
                            full_len = len;
                        }
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
            } else if (_navwipe_bit_records > 0) {
                const auto& ab = _navbuf[p];
                const auto& ub = _navutc[p];
                const auto& hb = _navhead[p]; // head segments -> symbol-aligned bit epochs
                // The nav-bit wipe's floor grows with window length: nb free +-1 bit signs rectify
                // noise to ~sqrt(2 nb / pi) sigma, plus the bit-phase selection over
                // _navwipe_bit_records alignments (~sqrt(ln br)); ~7 sigma at 50 bits. Ungated,
                // an unlocked satellite's ladder would drift to the longest rung.
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
                        // Full window: also export the per-bit decisions (nav_obs) when asked;
                        // the longest rung sees every bit in the window.
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
            } else {
                // Plain coherent sum, no wipe. The replica already carries the secondary overlay
                // (GPS_L5_Q_NH: NH20 tiled into 204600 chips), so each record arrives
                // overlay-free and the deep integration is a straight sum. The overlay wipe cannot
                // serve this case: overlay_apply advances the overlay one chip per record, which
                // holds only when a record is one primary period, and a CHORD record is 2048
                // hops = 10.4857 periods.
                const auto& ab = _navbuf[p];
                const auto& ub = _navutc[p];
                bool cleared = false;
                gnss::OverlayWipeResult full{};
                size_t full_len = 0;
                double full_cf = 0.0;
                // A straight sum takes no max over alignments, so its floor is the plain Rayleigh
                // tail. The sky-corrected stream is one more candidate, a max paid through n_cand;
                // the factor 2 allows a little for the ladder's own rungs.
                // The sky-corrected stream (slots 24/25, all zero = absent) is checked over the
                // window, so a few cold records at the start of an arc do not disable it.
                const auto& skb = _navsky[p];
                bool have_sky = skb.size() == ab.size();
                if (have_sky) {
                    double e_sky = 0.0;
                    for (const auto& v : skb)
                        e_sky += std::norm(v);
                    have_sky = e_sky > 0.0;
                }
                const size_t n_cand = 1 + (have_sky ? 1 : 0);
                const double flr = std::sqrt(2.0 * std::log(2.0 * (double)n_cand)) + 1.0;
                for (size_t len : ladder(ab.size(), min_rung(ub, 4))) {
                    std::vector<std::complex<double>> w(ab.end() - (long)len, ab.end());
                    // Sky-corrected candidate: the per-record common phase is nearly identical
                    // across elements and ~white in time, so other elements measure it where
                    // temporal neighbours cannot; the producer derotated one half of the elements
                    // by the other half's phase (ElemCal::combine_split). Scored as one more
                    // candidate (the floor pays for it) rather than replacing the raw stream, so
                    // a bad cal loses on merit.
                    std::vector<std::complex<double>> wsky;
                    if (have_sky)
                        wsky.assign(skb.end() - (long)len, skb.end());
                    // Phase-rate search: the per-record prompts can carry a linear phase ramp,
                    // which makes the straight sum integrate a rotating phasor. The rate can
                    // exceed the record-rate Nyquist and changes between emits, so it is searched
                    // rather than predicted. Records are uniform in hop, so the grid is uniform
                    // and gaps (records the broker dropped) are exact zeros in the transform.
                    double rate_hz = 0.0, rate_q = 0.0;
                    double rate_fhz = 0.0, rate_fq = 0.0;
                    const std::vector<double> us(ub.end() - (long)len, ub.end());
                    rate_search(w, us, rate_hz, rate_q, &rate_fhz, &rate_fq);
                    if (have_sky) {
                        // The sky correction removes the per-record phase including any linear
                        // part, so the corrected stream gets its own rate search.
                        double f2 = 0.0, q2 = 0.0;
                        rate_search(wsky, us, f2, q2);
                        if (q2 >= DEEP_RATE_MIN_Q && f2 != 0.0) {
                            const double t0 = us.front();
                            for (size_t k = 0; k < wsky.size(); ++k)
                                wsky[k] *= std::polar(1.0, -2.0 * M_PI * f2 * (us[k] - t0));
                        }
                    }
                    if (rate_q >= DEEP_RATE_MIN_Q && rate_hz != 0.0) {
                        // Derotate and let coherent_sum score it, so deep_snr, deep_amplitude and
                        // coherence_s keep their meaning: the search decides whether to derotate,
                        // coherent_sum decides the value.
                        const double t0 = us.front();
                        for (size_t k = 0; k < w.size(); ++k)
                            w[k] *= std::polar(1.0, -2.0 * M_PI * rate_hz * (us[k] - t0));
                    } else {
                        rate_hz = 0.0; // nothing found: score the straight sum
                    }
                    // Per-record magnitudes are invariant under derotation, so one pass serves
                    // every candidate's coherence fraction |sum| / sum|.| below.
                    double sum_abs_raw = 0.0, sum_abs_sky = 0.0;
                    for (const auto& v : w)
                        sum_abs_raw += std::abs(v);
                    if (have_sky)
                        for (const auto& v : wsky)
                            sum_abs_sky += std::abs(v);
                    gnss::OverlayWipeResult cs = gnss::coherent_sum(w);
                    bool sky_won = false;
                    if (have_sky) {
                        const gnss::OverlayWipeResult csk = gnss::coherent_sum(wsky);
                        if (csk.snr > cs.snr) {
                            cs = csk;
                            sky_won = true;
                        }
                    }
                    // Coherence fraction of the stream that won: the sky-corrected records have
                    // their own magnitudes.
                    const double sum_abs = sky_won ? sum_abs_sky : sum_abs_raw;
                    const double cf = sum_abs > 0.0 ? cs.amplitude * (double)len / sum_abs : 0.0;
                    if (full_len == 0) { // ladder walks longest-first
                        full = cs;
                        full_len = len;
                        full_cf = cf;
                        deep_rate[p] = rate_hz;
                        deep_rate_q[p] = rate_q;
                        deep_rate_full[p] = rate_fhz;
                        deep_rate_full_q[p] = rate_fq;
                    }
                    // With a rate applied, the gate is the search's own peak/median q rather than
                    // a closed-form floor: oversampled bins are far from independent, so
                    // sqrt(2 ln n) + 1 overstates the noise maximum. q is measured on the same
                    // data, so it does not depend on rung length or bin count.
                    if (rate_hz != 0.0) {
                        if (rate_q < DEEP_RATE_MIN_Q)
                            continue;
                    } else if (cs.snr < FLOOR_MARGIN * flr)
                        continue; // indistinguishable from noise
                    if (!cleared || cs.snr > deep_snr[p] * 1.05) {
                        deep_rate[p] = rate_hz;
                        deep_rate_q[p] = rate_q;
                        deep_rate_full[p] = rate_fhz;
                        deep_rate_full_q[p] = rate_fq;
                        rec[8] = (float)cs.amplitude;
                        deep_snr[p] = cs.snr;
                        deep_rec[p] = (int)len;
                        deep_floor[p] = flr;
                        coh_s[p] = coh_span(ub, len);
                        coh_frac[p] = cf;
                        cleared = true;
                    }
                }
                if (!cleared && full_len > 0) {
                    rec[8] = (float)full.amplitude;
                    deep_snr[p] = full.snr;
                    deep_rec[p] = (int)full_len;
                    deep_floor[p] = flr;
                    coh_s[p] = 0.0; // no rung beat its floor: no coherent detection
                    coh_frac[p] = full_cf;
                }
            }
            rec[gnss::CMB_COH_FRAC] = (float)coh_frac[p];
            *reinterpret_cast<double*>(rec + RECORD_UTC_SLOT) = ref_utc[p];
            // Absolute hop of this window, beside the UTC: the key the fleet DLL groups on, so a
            // recorded .raw says exactly which window each row is. The same for every PRN.
            *reinterpret_cast<long long*>(rec + gnss::CMB_HOP_SLOT) = win_hop;
        }

        // REST snapshot of this emit.
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
                _st_amp_snr[p] = (float)amp_snr[p];
                _st_dop[p] = rec[1];
                _st_cp[p] = rec[2];
                _st_coh_s[p] = (float)coh_s[p];
                _st_deep_rec[p] = deep_rec[p];
                _st_deep_floor[p] = (float)deep_floor[p];
                _st_deep_rate[p] = (float)deep_rate[p];
                _st_deep_rate_q[p] = (float)deep_rate_q[p];
                _st_deep_rate_full[p] = (float)deep_rate_full[p];
                _st_deep_rate_full_q[p] = (float)deep_rate_full_q[p];
                _st_coh_frac[p] = (float)coh_frac[p];
                if (_rec_export > 0)
                    _st_recex[p] = _recex[p];
                if (_bit_export)
                    _st_nav_obs[p] = std::move(nav_obs[p]);
                _st_dll_disc[p] = rec[gnss::CMB_DLL_DISC];
                // Fleet DLL: the discriminator's ingredients. Ratios do not sum, so the broker
                // needs these to combine instances (docs/CHORD_GNSS_SHARED_DLL.md).
                _st_e_pow[p] = rec[gnss::CMB_E_POW];
                _st_l_pow[p] = rec[gnss::CMB_L_POW];
                _st_p_pow[p] = p_pow_out[p];
                _st_nchan[p] = rec[gnss::CMB_NCHAN];
                _st_head_frac[p] = rec[gnss::CMB_HEAD_FRAC];
                // S4 with the thermal floor removed using the coherent SNR, which is independent
                // of the amplitude fluctuation being measured. Per-record power SNR
                // x = deep_snr^2 / K; for a non-central chi-square with 2 dof
                //     S4_thermal^2 = (1 + 2x) / (1 + x)^2
                {
                    const double K = (double)std::max(1, deep_rec[p]);
                    const double x = (deep_snr[p] > 0.0) ? deep_snr[p] * deep_snr[p] / K : 0.0;
                    const double s4_th2 = (1.0 + 2.0 * x) / ((1.0 + x) * (1.0 + x));
                    const double s4r = _st_s4_raw[p];
                    _st_s4[p] = (float)std::sqrt(std::max(0.0, s4r * s4r - s4_th2));
                }
                _st_s4_raw[p] = (float)s4_raw[p];
                _st_sigma_phi[p] = (float)sigma_phi[p];
                _st_car_resid[p] = rec[gnss::CMB_CARRIER_RESID];
                // ADR snapshot: the phase is a double (REST only); the record carries the arc id.
                const double utc_e = *reinterpret_cast<const double*>(rec + RECORD_UTC_SLOT);
                _st_utc[p] = utc_e;
                _st_adr[p] = _adr_ok[p] ? _adr_cyc[p] : 0.0;
                _st_res[p] = _adr_ok[p] ? _res_cyc[p] : 0.0;
                _st_trim[p] = _adr_ok[p] ? _trim_cyc[p] : 0.0;
                _st_adr_arc[p] = _adr_arc[p];
                _st_adr_n[p] = _adr_ok[p] ? _adr_n[p] : 0;
                _st_adr_lock[p] = (_adr_ok[p] && _adr_t0[p] > 0.0) ? (utc_e - _adr_t0[p]) : 0.0;
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
                int active = 0;
                for (int p = 0; p < _n_prn; ++p)
                    if (!_navbuf[p].empty())
                        ++active;
                _m_last_emit_prns->set((double)active); // emit cost scales with this, not n_prn
            }
        }
        out_buf->mark_frame_full(unique_name, out_id++);
    }
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
                         // [hop, re, im, energy] per record, oldest first (see record_export
                         // in the constructor).
                         {"records", _st_recex[p]}});
    }
    conn.send_json_reply(reply);
}

void GnssCoherentCombiner::get_elements_callback(kotekan::connectionInstance& conn) {
    // Per-element complex gain, per PRN and element: [u_re, u_im, p2, q] with
    // u = <A_e * conj(LOO ref)> (the gain), p2 = <|A_e|^2> (incoherent beam-map power, not
    // debiased) and q = <|LOO ref|^2> (the normalizer: amp_e ~ |u|/q). keff is the effective
    // record count behind the averages, for significance. The parts are exported raw so any
    // combination can be rebuilt downstream.
    nlohmann::json reply = nlohmann::json::array();
    std::lock_guard<std::mutex> lk(_st_mtx);
    if (_st_elem.size() == (size_t)_n_prn * _n_elements * 4) {
        for (int p = 0; p < _n_prn; ++p) {
            if (_st_prn[p] < 0)
                continue;
            nlohmann::json els = nlohmann::json::array();
            for (int el = 0; el < _n_elements; ++el) {
                const float* se = _st_elem.data() + ((size_t)p * _n_elements + el) * 4;
                els.push_back({se[0], se[1], se[2], se[3]});
            }
            reply.push_back({{"prn", _st_prn[p]},
                             {"doppler_hz", _st_dop[p]},
                             {"pow_hop", _st_pow_hop},
                             {"keff", _st_elem_keff},
                             {"n_elem", _n_elements},
                             {"elems", els}});
        }
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

                         {"doppler_hz", _st_dop[p]},
                         {"code_phase_chips", _st_cp[p]},
                         {"coherence_s", _st_coh_s[p]},
                         {"deep_records", _st_deep_rec[p]},
                         {"deep_floor", _st_deep_floor[p]},
                         // Phase-rate search: the linear rate removed before the deep sum (Hz,
                         // signed, modulo the record rate) and its spectrum's peak/median, the
                         // detection statistic for the plain coherent rung. 0 = the search did not
                         // run or found nothing, and the straight sum was scored.
                         {"deep_rate_hz", _st_deep_rate[p]},
                         {"deep_rate_q", _st_deep_rate_q[p]},
                         // The uncapped argmax of the same spectrum and its q, which the carrier
                         // loop reads. deep_rate_hz is the fold's pick, capped at DEEP_RATE_MAX_HZ;
                         // beyond the cap it is a noise bin.
                         {"deep_rate_full_hz", _st_deep_rate_full[p]},
                         {"deep_rate_full_q", _st_deep_rate_full_q[p]},
                         // coh_frac = |sum|/sum|.| of the winning deep stream, a coherence measure
                         // that, unlike deep_snr, does not scale with record count.
                         {"coh_frac", _st_coh_frac[p]},
                         {"dll_disc", _st_dll_disc[p]},
                         // Fleet DLL (docs/CHORD_GNSS_SHARED_DLL.md): dll_disc is a ratio, and the
                         // full-band discriminator is (sum e_pow - sum l_pow)/(sum e_pow + sum
                         // l_pow) over every instance, so the three powers ship raw. All are built
                         // as |correlation|^2 / energy^2, so q = 2*p_pow/(e_pow+l_pow) is
                         // comparable across instances.
                         {"e_pow", _st_e_pow[p]},
                         {"p_pow", _st_p_pow[p]},
                         {"l_pow", _st_l_pow[p]},
                         {"n_chan", _st_nchan[p]},
                         // Grouping key: absolute hop index of the window these powers cover, from
                         // the F-engine counter every node shares; equal hop is the same window
                         // exactly. pow_fft_len is the FFT length behind it; pow_hop -1 means the
                         // record frames carried no metadata.
                         {"pow_hop", _st_pow_hop},
                         {"pow_fft_len", _fft_len},
                         // Boundary fraction f: where the code-period boundary sits in the despread
                         // window (segmented-wipe diagnostic).
                         {"boundary_f", _st_head_frac[p]},
                         // S4 = amplitude fluctuation (thermal floor removed); sigma_phi =
                         // carrier-phase jitter about the slope fit (rad). Multipath moves both.
                         {"s4", _st_s4[p]},
                         {"s4_raw", _st_s4_raw[p]},
                         {"sigma_phi", _st_sigma_phi[p]},
                         {"carrier_hz_resid", _st_car_resid[p]},
                         {"utc", _st_utc[p]},
                         // Accumulated carrier phase (cycles) on the current unbroken arc, its
                         // id, its length in records, and how long it has held (s). 0 cycles
                         // with a new arc id means the arc just restarted.
                         {"adr_cycles", _st_adr[p]},
                         // The residual half of the ADR alone (sum of measured dres, same arc).
                         // Its change over adr_records across a poll span is a mHz-class
                         // carrier-rate measurement with no commanded integral in it; the
                         // broker's rate feed reads this, not adr_cycles.
                         {"res_cycles", _st_res[p]},
                         // Commanded-trim integral on the same arc; subtract from adr_cycles to
                         // remove the carrier loop's known transients.
                         {"trim_cycles", _st_trim[p]},
                         {"adr_arc", _st_adr_arc[p]},
                         {"adr_records", _st_adr_n[p]},
                         {"adr_lock_s", _st_adr_lock[p]}});
    // bit_export: the last emit's per-bit sign decisions (nav_obs; see NavObs for the timing
    // convention), only on rows that have any.
    if (_bit_export)
        for (int p = 0; p < _n_prn; ++p) {
            const NavObs& o = _st_nav_obs[(size_t)p];
            if (o.bits.empty())
                continue;
            nlohmann::json bits = nlohmann::json::array();
            for (const auto& b : o.bits)
                bits.push_back({b.first, (int)b.second});
            reply[(size_t)p]["nav_obs"] = {
                {"utc_ref", o.utc_ref}, {"rec_dt", o.rec_dt}, {"phase", o.phase}, {"br", o.br},
                {"rot_re", o.rot_re},   {"rot_im", o.rot_im}, {"bits", bits}};
        }
    conn.send_json_reply(reply);
}

double GnssCoherentCombiner::navwipe_amplitude(const std::vector<std::complex<double>>& a,
                                               const std::vector<double>& utc, double* snr_out,
                                               const std::vector<std::complex<double>>* head,
                                               NavObs* obs, int pin_phase, int* out_phase) const {
    const int br = _navwipe_bit_records;
    const int nrec = (int)a.size();
    if (br <= 0 || nrec < 2 * br)
        return 0.0;
    if (head && (int)head->size() != nrec)
        head = nullptr; // size mismatch (mixed-schema window): fall back to unsegmented
    using cd = std::complex<double>;

    // Code-period index per record from capture UTC: a dropped record skips an index, so bit
    // epochs stay aligned (binning by buffer position would shift them).
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

    // Per-bit coherent sums for a given epoch phase. Records are sorted by cpi, so bit indices
    // arrive non-decreasing and a bit is a run of equal (cpi + phase) / br. Segmented (head
    // non-null): the head belongs to period cpi[r] and the tail to cpi[r]+1, so a record
    // straddling a bit boundary splits between the two bits instead of smearing the symbol
    // transition. Streaming order holds because cpi is strictly increasing.
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
                    out_bi->push_back(cur); // which bit that sum belongs to
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

    // Bit sync: the phase (0..br-1) maximising the mean per-bit coherent amplitude |sum|, unless
    // the caller pinned it. The search is br full passes over the window (see the hpp).
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

    // Estimate each bit's +-1 by squaring (theta0 = 1/2 arg sum s^2; the global sign cancels in
    // |.|), wipe, and sum coherently for the deep amplitude (mean of the wiped A). The component
    // orthogonal to the aligned signal (the imaginary part after rot) carries no signal, so it
    // measures the per-bit noise for the significance.
    cd sumsq(0.0, 0.0);
    for (const cd& v : s)
        sumsq += v * v;
    const cd rot = std::polar(1.0, -0.5 * std::arg(sumsq));
    cd deep(0.0, 0.0);
    double noise2 = 0.0; // sum of squared orthogonal (noise) components, one per bit
    if (obs) {           // export the per-bit decisions the wipe is about to apply
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
    // SNR = coherent sum / its noise std, i.e. the deep amplitude over its standard error
    // noise_sum/nrec. Choosing each bit's sign rectifies noise to ~sqrt(2 nb / pi), not ~1.
    if (snr_out)
        *snr_out = noise_sum > 0.0 ? std::abs(deep) / noise_sum : 0.0;
    return std::abs(deep) / (double)nrec; // coherent mean of the wiped per-record A
}

double GnssCoherentCombiner::carrier_resid_hz(const std::vector<std::complex<double>>& a,
                                              const std::vector<double>& utc,
                                              double* sigma_phi_out) const {
    const int n = (int)a.size();
    // *sigma_phi_out is always written: NaN when there is no fit, since 0.0 would read as a
    // perfect lock. The returned residual is then 0.0, which the broker's carrier loop skips
    // (gnss_broker/carrierloop.py).
    if (n < 16) {
        if (sigma_phi_out != nullptr)
            *sigma_phi_out = NAN;
        return 0.0;
    }
    using cd = std::complex<double>;
    // Bit-robust phase-slope fit in two stages, because a sequential record-to-record unwrap
    // alone is fragile: over ~1000 steps one deep fade can slip the accumulated phase by 2 pi
    // and bias the slope by several Hz.
    //   Stage 1, slip-free coarse frequency: lag-L delay-and-multiply, one arg() of a weighted
    //   complex sum (a fade only down-weights its pairs).
    //   Stage 2: demodulate by the coarse frequency, then a |A|^2-weighted least-squares fit of
    //   the sequentially unwrapped phase against capture UTC (gaps contribute their true span).
    //   After demodulation the slope is ~0, so the steps stay well inside +-pi.
    // A is squared to cancel +-1 nav-bit and overlay flips.
    const double mult = 2.0; // squared phase advances at mult * the carrier
    // Stage 1: coarse f from pairs L records apart. L = 4 trades variance against the
    // unambiguous range +-1/(2*L*dt*mult), +-62 Hz for squared 1 ms records.
    const int L = 4;
    cd pair_sum(0.0, 0.0);
    double dt_sum = 0.0;
    int n_pair = 0;
    for (int r = L; r < n; ++r) {
        const double w = std::norm(a[r]) * std::norm(a[r - L]);
        if (!(w > 0.0))
            continue;
        const cd vr = a[r] * a[r];
        const cd vp = a[r - L] * a[r - L];
        pair_sum += w * vr * std::conj(vp);
        dt_sum += utc[r] - utc[r - L];
        ++n_pair;
    }
    double f_coarse = 0.0;
    if (n_pair > 0 && std::abs(pair_sum) > 0.0 && dt_sum > 0.0)
        f_coarse = std::arg(pair_sum) / (2.0 * M_PI * (dt_sum / n_pair) * mult);
    // Stage 2: least-squares slope of the unwrapped demodulated phase.
    double phi = 0.0, raw_prev = 0.0;
    bool have = false;
    double Sw = 0, Swt = 0, Swtt = 0, Swp = 0, Swtp = 0;
    const double t0 = utc[0];
    for (int r = 0; r < n; ++r) {
        const double w = std::norm(a[r]); // |A|^2
        if (!(w > 0.0))
            continue;
        const cd v = (a[r] * a[r]) * std::polar(1.0, -2.0 * M_PI * f_coarse * mult * (utc[r] - t0));
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
    // sigma_phi: the weighted RMS of the same fit's residuals, divided by mult to give radians on
    // the true carrier. The fit removes the frequency offset, so what is left is multipath,
    // scintillation and thermal noise. A second pass is needed because the accumulators above
    // cannot yield residuals.
    if (sigma_phi_out != nullptr) {
        double phi2 = 0.0, raw_prev2 = 0.0, Sw2 = 0.0, Sr2 = 0.0;
        bool have2 = false;
        for (int r = 0; r < n; ++r) {
            const double w = std::norm(a[r]);
            if (!(w > 0.0))
                continue;
            const cd v =
                (a[r] * a[r]) * std::polar(1.0, -2.0 * M_PI * f_coarse * mult * (utc[r] - t0));
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
        }
        *sigma_phi_out = (Sw2 > 0.0) ? std::sqrt(Sr2 / Sw2) / mult : NAN; // unmeasured -> NaN
    }
    // Total residual = the slip-free coarse stage plus the fine least-squares slope of the rest.
    return f_coarse + slope / mult / (2.0 * M_PI); // -> residual carrier (Hz)
}
