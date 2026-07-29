#include "GnssGpuRecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "visUtil.hpp" // for frameID
#include "gnssGpuChain.hpp"
#include "gnssRecord.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssGpuRecordAssemble);

GnssGpuRecordAssemble::GnssGpuRecordAssemble(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssGpuRecordAssemble::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    _prns = config.get<std::vector<int>>(unique_name, "prns");
    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    // Per-channel prompt-phase dump (see the hpp note; diagnostic, default off).
    _chan_dump_prn = config.get_default<int>(unique_name, "chan_dump_prn", -1);
    _chan_dump_decim = std::max(1, config.get_default<int>(unique_name, "chan_dump_decim", 10));
    if (_chan_dump_prn >= 0) {
        const std::string path = config.get_default<std::string>(
            unique_name, "chan_dump_path", "/tmp/gnss_chan_phase_dump.txt");
        _chan_dump = std::fopen(path.c_str(), "a");
    }
    // ELEMENT AXIS (CHORD). 0 = the single-antenna airspy layout, byte-for-byte -- the default,
    // so nothing existing changes. >0 appends n_elements per-antenna blocks to every PRN record
    // (gnssRecord.hpp). The GPU correlation array grows the same axis: [rows][n_chan][n_elem].
    _n_elements = config.get_default<int>(unique_name, "n_elements", 0);
    // Which element the record HEADER's correlation slots carry. The broker closes the DLL and
    // carrier loops off those slots, so this must be a phase-coherent single-antenna view (see
    // gnssRecord.hpp) -- pick a healthy, high-gain feed. Ignored when n_elements == 0.
    _reference_element = config.get_default<int>(unique_name, "reference_element", 0);
    if (_n_elements > 0 && (_reference_element < 0 || _reference_element >= _n_elements)) {
        FATAL_ERROR("GnssGpuRecordAssemble: reference_element {:d} outside [0, n_elements={:d})",
                    _reference_element, _n_elements);
        return;
    }
    const int n = (int)_prns.size();
    // Record width is a C++ constant; the frame is sized in yaml (n_prn * record_floats *
    // sizeof_float) with nothing linking the two. A stale config under-sizes the frame and this
    // stage writes past its end -- die at construction rather than corrupt memory.
    const int rec_stride = gnss::record_stride(_n_elements);
    const size_t rec_bytes_need = (size_t)n * rec_stride * sizeof(float);
    if ((size_t)out_buf->frame_size < rec_bytes_need) {
        FATAL_ERROR("GnssGpuRecordAssemble: out_buf frame {:d} B < {:d} PRN x {:d} floats "
                    "({:d} B). Bump 'record_floats' in the config to {:d}.",
                    (size_t)out_buf->frame_size, n, rec_stride, rec_bytes_need, rec_stride);
        return;
    }
    _phi.assign(n, 0.0);
    _phi_cyc.assign(n, 0.0);
    _phi_cmd_prev.assign(n, 0.0);
    _phi_cmd_ok.assign(n, 0);
    _fcar_prev.assign(n, 0.0);
    _fcar_prev_ok.assign(n, 0);
    _a_prev.assign(n, {0.0, 0.0});
    _a_prev_ok.assign(n, 0);
    _wstart_prev.assign(n, 0);
}

GnssGpuRecordAssemble::~GnssGpuRecordAssemble() {
    if (_chan_dump)
        std::fclose(_chan_dump);
}

void GnssGpuRecordAssemble::main_thread() {
    using namespace gnss_gpu;
    frameID frame_in(in_buf), frame_out(out_buf);
    const int n_prn = (int)_prns.size();

    while (!stop_thread) {
        const uint8_t* in = in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in == nullptr)
            return;
        FrameHdr hdr;
        std::memcpy(&hdr, in, sizeof(hdr));
        if (hdr.n_prn != n_prn) {
            FATAL_ERROR("GnssGpuRecordAssemble: frame n_prn {:d} != config {:d}", hdr.n_prn,
                        n_prn);
            return;
        }
        const int n_chan = hdr.n_chan;
        // Output rows per spec: 4 normally, 6 when the writer peels (rows 4/5 = the peel
        // residual). 0 means a writer that predates the field -- read it as 4.
        const int n_rows_spec = (hdr.n_rows_spec > 0) ? hdr.n_rows_spec : ROWS_PLAIN;
        const int64_t* winstart = (const int64_t*)(in + off_winstart());
        const PrnCtl* pctl = (const PrnCtl*)(in + off_prnctl());
        const double* corr = (const double*)(in + off_corr(n_prn));      // double2 rows
        const double* energy =
            (const double*)(in + off_energy(n_prn, n_chan, n_rows_spec));

        for (int r = 0; r < hdr.n_rec && !stop_thread; ++r) {
            float* out = (float*)out_buf->wait_for_empty_frame(unique_name, frame_out);
            if (out == nullptr)
                return;
            const int64_t wstart = winstart[r];
            const double utc =
                (hdr.utc0 > 0.0) ? hdr.utc0 + (double)wstart / _sample_rate
                                 : std::chrono::duration<double>(
                                       std::chrono::system_clock::now().time_since_epoch())
                                       .count();

            for (int p = 0; p < n_prn; ++p) {
                const int rec_stride = gnss::record_stride(_n_elements);
                float* rec = out + (size_t)p * rec_stride;
                // Zero the WHOLE record, element blocks included: a PRN that does not run this
                // window must not leave a previous window's per-antenna correlations behind.
                for (int f = 0; f < rec_stride; ++f)
                    rec[f] = 0.0f;
                rec[0] = (float)_prns[p];
                *reinterpret_cast<double*>(rec + gnss::RECORD_UTC_SLOT) = utc;
                const PrnCtl& c = pctl[(size_t)r * n_prn + p];
                if (!c.run) {
                    _a_prev_ok[p] = 0;
                    _fcar_prev_ok[p] = 0;
                    _phi_cyc[p] = 0.0;
                    _phi_cmd_ok[p] = 0;
                    continue;
                }
                // Cross-channel sum over the covering mask, per correlator trial
                // (E, P, L, P_HEAD -- the prompt's head segment, gnssRecord.hpp slots 16-18),
                // plus, when the chain peels, the residual prompt and its head (rows 4/5 ->
                // gnssRecord.hpp slots 20-23). PrnCtl::job0 already carries the per-spec row
                // stride, so indexing is job0 + t either way.
                //
                // NB rows 0-3 hold the FULL, un-peeled correlation even when peeling: the
                // analytic add-back was applied on the device (docs/gnss_voltage_peel_live.md).
                // That is what leaves everything below this stage -- combiner, broker, viewer,
                // TEC -- unable to tell whether a peel happened.
                // With an element axis the correlation array is [rows][n_chan][n_elem] and the
                // covering-mask sum runs PER ANTENNA. The ENERGIES do not: one replica is
                // correlated against every antenna, so energy stays [rows][n_chan] and is summed
                // once (gnssRecord.hpp -- this is why energies live in the record header and only
                // the correlations go in the element blocks).
                //
                // n_e == 1 with n_elements == 0 makes the index expression below collapse to
                // exactly the single-antenna one, so that path is unchanged.
                const int n_e = (_n_elements > 0) ? _n_elements : 1;
                const int ref_e = (_n_elements > 0) ? _reference_element : 0;
                std::complex<double> g3[6];  // reference element, for the header + NCO/gain state
                double e3[6];                // element-independent
                _g_elem.assign((size_t)n_rows_spec * n_e, std::complex<double>(0.0, 0.0));
                for (int t = 0; t < n_rows_spec; ++t) {
                    const size_t row = (size_t)(c.job0 + t) * n_chan;
                    double e = 0.0;
                    for (int ch = 0; ch < n_chan; ++ch)
                        if ((c.chan_mask >> ch) & 1ULL) {
                            e += energy[row + ch];
                            const size_t base = (row + ch) * n_e;
                            for (int el = 0; el < n_e; ++el)
                                _g_elem[(size_t)t * n_e + el] += std::complex<double>(
                                    corr[2 * (base + el)], corr[2 * (base + el) + 1]);
                        }
                    g3[t] = _g_elem[(size_t)t * n_e + ref_e];
                    e3[t] = e;
                }
                // Per-channel PROMPT dump (diagnostic, see hpp): raw pre-rotation per-channel
                // correlations -- the cross-channel relative phases are the observable.
                if (_chan_dump && _prns[p] == _chan_dump_prn
                    && (++_chan_dump_ctr % _chan_dump_decim) == 0) {
                    const size_t prow = (size_t)(c.job0 + 1) * n_chan; // trial 1 = PROMPT
                    for (int ch = 0; ch < n_chan; ++ch)
                        if ((c.chan_mask >> ch) & 1ULL)
                            std::fprintf(_chan_dump, "%.6f %d %.6e %.6e %.6e\n", utc, ch,
                                         corr[2 * ((prow + ch) * n_e + ref_e)],
                                         corr[2 * ((prow + ch) * n_e + ref_e) + 1],
                                         energy[prow + ch]);
                }
                rec[1] = c.fcar_report;
                rec[2] = (float)c.cp_seed;
                rec[6] = c.n_owned;
                rec[gnss::REC_E_RE] = (float)g3[0].real();
                rec[gnss::REC_E_IM] = (float)g3[0].imag();
                rec[gnss::REC_E_ENERGY] = (float)e3[0];
                rec[gnss::REC_L_RE] = (float)g3[2].real();
                rec[gnss::REC_L_IM] = (float)g3[2].imag();
                rec[gnss::REC_L_ENERGY] = (float)e3[2];

                // Carrier NCO (pass-2 half): the command's fence re-anchor resets the phase
                // history exactly like the tracker's in-place reset; f_nco (ctrim + ff ramp)
                // changes the slope of phi, never jumps it.
                if (c.reanchored == 1 || (c.reanchored == 2 && !_fcar_prev_ok[p])) {
                    // FRESH acquisition: no phase history to preserve. Break the arc.
                    _phi[p] = 0.0;
                    _phi_cyc[p] = 0.0;
                    _a_prev_ok[p] = 0;
                    _phi_cmd_ok[p] = 0;
                } else if (c.reanchored == 2) {
                    // PHASE-CONTINUOUS RE-PIN. Re-pinning f_ref steps the ABSOLUTELY-ANCHORED
                    // replica phase by df*t_abs -- thousands of cycles at soak age. The old code
                    // folded that step into an EXPORT-ONLY offset and then ZEROED the NCO, so the
                    // exported ADR survived but the DATA did not: the rotation applied to A moved
                    // by the step, i.e. every re-pin punched an effectively random phase jump into
                    // the very correlation the combiner deep-integrates. That is the residual C/N0
                    // sawtooth -- and it showed exactly where this explanation says it must, in the
                    // COHERENT estimator at precisely max_anchor_age_s (30.0 s on every strong sat,
                    // 0.2 dB on GPS to 1.6 dB on B1C) and NOT in the phase-blind incoherent one
                    // (2026-07-14, on-sky).
                    //
                    // The step is FOLDABLE: an NCO absorbs an arbitrary constant phase. Put it in
                    // the NCO instead of the export and the despread output is continuous THROUGH
                    // the re-pin -- the commanded phase stays exactly as continuous as before (the
                    // export algebra is unchanged: fcar*t - phi_cyc is invariant under this fold),
                    // so the validated ADR arcs are untouched.
                    //
                    // This is what makes a re-pin CHEAP, which is the real prize: with the phase
                    // step folded and the code-currency step already translated above, the replica
                    // can be re-pinned as often as we like. max_anchor_age_s: 0 re-pins EVERY
                    // record, so f_ref never goes stale and the within-record decoherence that
                    // grows with anchor age (the OTHER half of the sawtooth, ~(dop_rate*age*t_rec)^2
                    // -- negligible on GPS's 1 ms record, ~1 dB on B1C's 10 ms) never accumulates.
                    const double t_pin = (double)wstart / _sample_rate;
                    const double dcyc = (c.fcar - _fcar_prev[p]) * t_pin;
                    _phi_cyc[p] += dcyc;
                    _phi[p] = std::remainder(_phi[p] + 2.0 * M_PI * dcyc, 2.0 * M_PI);
                }
                const double dt = (double)(wstart - _wstart_prev[p]) / _sample_rate;
                if (_a_prev_ok[p] && dt > 0.0) {
                    // Commanded-trim increment (slot 19, see gnssRecord.hpp): the broker's
                    // ctrim reconstructed from the identity f_nco = ctrim + ff and
                    // fcar_report = fcar - ff + ctrim => ctrim = (f_nco + fcar_report -
                    // fcar)/2. Integrated over this record so downstream TEC can subtract
                    // the loop's transients from the ADR exactly.
                    const double ctrim_hz =
                        0.5 * ((double)c.f_nco + (double)c.fcar_report - c.fcar);
                    rec[gnss::REC_TRIM_INC] = (float)(ctrim_hz * dt);
                    _phi[p] += 2.0 * M_PI * c.f_nco * dt;
                    // Keep the ROTATION phase bounded, but track the NCO phase UNWRAPPED (in
                    // cycles) for the carrier-phase export. remainder() slides phi by whole
                    // multiples of 2*pi, which a rotation cannot see and a mod-1 phase export
                    // cannot see either -- but a per-record INCREMENT sees every one of them as
                    // a spurious +-1 cycle. The NCO wraps ~f_nco times a second, so the ADR bled
                    // about one cycle per wrap: measured as a per-satellite ADR-rate error of a
                    // few Hz, scaling with each satellite's carrier trim (2026-07-13).
                    _phi[p] = std::remainder(_phi[p], 2.0 * M_PI);
                    _phi_cyc[p] += c.f_nco * dt;
                }
                const std::complex<double> rot = std::polar(1.0, -_phi[p]);
                const std::complex<double> g_corr = g3[1] * rot;
                rec[3] = (float)g_corr.real();
                rec[4] = (float)g_corr.imag();
                rec[5] = (float)e3[1];
                // Head segment: SAME NCO rotation as the prompt (it is the same correlation,
                // restricted to the hops before the code-period boundary), so head + tail
                // reconstructs P exactly and the combiner can wipe each side with its own
                // overlay chip.
                const std::complex<double> gh_corr = g3[3] * rot;
                rec[gnss::REC_PH_RE] = (float)gh_corr.real();
                rec[gnss::REC_PH_IM] = (float)gh_corr.imag();
                rec[gnss::REC_PH_ENERGY] = (float)e3[3];
                // PEEL RESIDUAL (slots 20-23): the same prompt correlation taken on the voltage
                // after this PRN's own waveform was subtracted, and its head segment. SAME NCO
                // rotation as the prompt -- it is the same correlation, so the combiner can
                // deep-integrate it with the same per-segment overlay wipe. Left at zero when the
                // chain does not peel, which every existing consumer already reads as "absent".
                if (n_rows_spec > ROW_RES_PH) {
                    const std::complex<double> gr = g3[ROW_RES_P] * rot;
                    const std::complex<double> grh = g3[ROW_RES_PH] * rot;
                    rec[gnss::REC_RES_RE] = (float)gr.real();
                    rec[gnss::REC_RES_IM] = (float)gr.imag();
                    rec[gnss::REC_RES_PH_RE] = (float)grh.real();
                    rec[gnss::REC_RES_PH_IM] = (float)grh.imag();
                }
                // ELEMENT BLOCKS. Every antenna gets the SAME NCO rotation: the NCO is a per-PRN
                // model of the code/carrier, not a per-antenna quantity, so rotating each element
                // by `rot` preserves exactly the inter-element phase differences -- which are the
                // measurement. Rotating per element would divide out the very thing being mapped.
                if (_n_elements > 0) {
                    for (int el = 0; el < _n_elements; ++el) {
                        float* eb = out + gnss::elem_offset(p, el, _n_elements);
                        const std::complex<double> ge = _g_elem[(size_t)1 * n_e + el] * rot;
                        const std::complex<double> gE = _g_elem[(size_t)0 * n_e + el] * rot;
                        const std::complex<double> gL = _g_elem[(size_t)2 * n_e + el] * rot;
                        const std::complex<double> gH = _g_elem[(size_t)3 * n_e + el] * rot;
                        eb[gnss::ELEM_P_RE] = (float)ge.real();
                        eb[gnss::ELEM_P_IM] = (float)ge.imag();
                        eb[gnss::ELEM_E_RE] = (float)gE.real();
                        eb[gnss::ELEM_E_IM] = (float)gE.imag();
                        eb[gnss::ELEM_L_RE] = (float)gL.real();
                        eb[gnss::ELEM_L_IM] = (float)gL.imag();
                        eb[gnss::ELEM_PH_RE] = (float)gH.real();
                        eb[gnss::ELEM_PH_IM] = (float)gH.imag();
                        if (n_rows_spec > ROW_RES_PH) {
                            const std::complex<double> gr =
                                _g_elem[(size_t)ROW_RES_P * n_e + el] * rot;
                            const std::complex<double> grh =
                                _g_elem[(size_t)ROW_RES_PH * n_e + el] * rot;
                            eb[gnss::ELEM_RES_RE] = (float)gr.real();
                            eb[gnss::ELEM_RES_IM] = (float)gr.imag();
                            eb[gnss::ELEM_RES_PH_RE] = (float)grh.real();
                            eb[gnss::ELEM_RES_PH_IM] = (float)grh.imag();
                        }
                    }
                }

                _a_prev[p] = (e3[1] > 0.0) ? g_corr / e3[1] : std::complex<double>(0.0, 0.0);
                _a_prev_ok[p] = 1;
                _wstart_prev[p] = wstart;

                // COMMANDED CARRIER PHASE (cycles mod 1), the GPU twin of the CPU tracker's
                // export -- see gnssRecord.hpp: replica f_ref*t_abs + the NCO's phi. Adding
                // the combiner's measured arg(A) reconstructs the received carrier phase,
                // with the re-pin's replica-phase step cancelling instead of slipping.
                // phi enters NEGATED: f_ref is physical-signed, the NCO is in the r2c-flipped
                // internal convention (see GnssChannelizedTracker for the on-sky measurement
                // that settled this -- a satellite with its trim pinned at the clamp).
                // (no _phi_fix term any more: the re-pin step is folded into _phi_cyc itself,
                // which leaves fcar*t_abs - phi_cyc continuous on its own.)
                const double t_abs_w = (double)wstart / _sample_rate;
                const double phi_cmd_cyc = c.fcar * t_abs_w - _phi_cyc[p];
                // the INCREMENT (gnssRecord.hpp): bounded, float-exact, unwrap-free
                rec[gnss::REC_CPHASE] =
                    _phi_cmd_ok[p] ? (float)(phi_cmd_cyc - _phi_cmd_prev[p]) : 0.0f;
                _phi_cmd_prev[p] = phi_cmd_cyc;
                _phi_cmd_ok[p] = 1;
                _fcar_prev[p] = c.fcar;
                _fcar_prev_ok[p] = 1;
            }

            if (out_buf->metadata_pool) {
                out_buf->allocate_new_metadata_object(frame_out);
                get_gnss_chan_metadata(out_buf, frame_out)->sample_seq = wstart;
            }
            out_buf->mark_frame_full(unique_name, frame_out);
            frame_out++;
        }
        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in++;
    }
}
