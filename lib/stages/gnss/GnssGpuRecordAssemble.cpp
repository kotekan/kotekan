#include "GnssGpuRecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "visUtil.hpp" // for frameID
#include "gnssGpuChain.hpp"
#include "gnssRecord.hpp"

#include <chrono>
#include <cmath>
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
    const int n = (int)_prns.size();
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
        const int64_t* winstart = (const int64_t*)(in + off_winstart());
        const PrnCtl* pctl = (const PrnCtl*)(in + off_prnctl());
        const double* corr = (const double*)(in + off_corr(n_prn));      // double2 rows
        const double* energy = (const double*)(in + off_energy(n_prn, n_chan));

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
                float* rec = out + (size_t)p * gnss::RECORD_FLOATS;
                for (int f = 0; f < gnss::RECORD_FLOATS; ++f)
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
                // (E, P, L, P_HEAD -- the prompt's head segment, gnssRecord.hpp slots 16-18).
                std::complex<double> g3[4];
                double e3[4];
                for (int t = 0; t < 4; ++t) {
                    const size_t row = (size_t)(c.job0 + t) * n_chan;
                    std::complex<double> g(0.0, 0.0);
                    double e = 0.0;
                    for (int ch = 0; ch < n_chan; ++ch)
                        if ((c.chan_mask >> ch) & 1ULL) {
                            g += std::complex<double>(corr[2 * (row + ch)],
                                                      corr[2 * (row + ch) + 1]);
                            e += energy[row + ch];
                        }
                    g3[t] = g;
                    e3[t] = e;
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
