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
                    continue;
                }
                // Cross-channel sum over the covering mask, per correlator trial (E, P, L).
                std::complex<double> g3[3];
                double e3[3];
                for (int t = 0; t < 3; ++t) {
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
                if (c.reanchored) {
                    _phi[p] = 0.0;
                    _a_prev_ok[p] = 0;
                }
                const double dt = (double)(wstart - _wstart_prev[p]) / _sample_rate;
                if (_a_prev_ok[p] && dt > 0.0) {
                    _phi[p] += 2.0 * M_PI * c.f_nco * dt;
                    _phi[p] = std::remainder(_phi[p], 2.0 * M_PI);
                }
                const std::complex<double> rot = std::polar(1.0, -_phi[p]);
                const std::complex<double> g_corr = g3[1] * rot;
                rec[3] = (float)g_corr.real();
                rec[4] = (float)g_corr.imag();
                rec[5] = (float)e3[1];
                _a_prev[p] = (e3[1] > 0.0) ? g_corr / e3[1] : std::complex<double>(0.0, 0.0);
                _a_prev_ok[p] = 1;
                _wstart_prev[p] = wstart;
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
