#include "GpsReplicaCorrelator.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp" // for make_input_desc, make_gps_record_desc
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "fftwPlannerLock.hpp" // for fftw_planner_mutex
#include "gpsCACode.hpp"       // for generate_ca_code, CA_CODE_LENGTH
#include "kotekanLogging.hpp"  // for DEBUG, INFO, FATAL_ERROR

#include <algorithm> // for max
#include <cmath>     // for round, cos, sin, sqrt, M_PI
#include <complex>   // for complex
#include <functional>// for bind
#include <memory>    // for shared_ptr
#include <mutex>     // for lock_guard

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GpsReplicaCorrelator);

// fftwf_complex is layout-compatible with std::complex<float>; this alias keeps
// the DSP arithmetic readable while the FFTW calls keep using the raw arrays.
static inline std::complex<float>* as_cpx(fftwf_complex* p) {
    return reinterpret_cast<std::complex<float>*>(p);
}

GpsReplicaCorrelator::GpsReplicaCorrelator(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GpsReplicaCorrelator::main_thread, this)),
    z_time(nullptr), a_freq(nullptr), z(nullptr), d(nullptr), D(nullptr), corr(nullptr),
    p_fwd_analytic(nullptr), p_inv_analytic(nullptr), p_fwd_data(nullptr), p_inv_corr(nullptr) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    _f_offset = config.get_default<double>(unique_name, "f_offset", 1e6);
    _doppler_min = config.get_default<double>(unique_name, "doppler_min", -6000.0);
    _doppler_max = config.get_default<double>(unique_name, "doppler_max", 6000.0);
    _doppler_step = config.get_default<double>(unique_name, "doppler_step", 500.0);
    _incoherent_ms = config.get_default<int>(unique_name, "incoherent_ms", 1);
    _track_phase = config.get_default<bool>(unique_name, "track_phase", false);
    _prns = config.get<std::vector<int>>(unique_name, "prns");

    if (_prns.empty())
        FATAL_ERROR("GpsReplicaCorrelator: 'prns' list is empty; nothing to correlate.");
    if (_incoherent_ms < 1)
        FATAL_ERROR("GpsReplicaCorrelator: incoherent_ms must be >= 1.");
    if (_doppler_step <= 0 || _doppler_max < _doppler_min)
        FATAL_ERROR("GpsReplicaCorrelator: invalid Doppler grid.");

    _Ns = (int)std::round(_sample_rate / 1000.0); // samples per 1 ms code period

    for (double f = _doppler_min; f <= _doppler_max + 1e-6; f += _doppler_step)
        _doppler_grid.push_back(f);

    const int n_prn = (int)_prns.size();
    const int n_dop = (int)_doppler_grid.size();

    INFO("GpsReplicaCorrelator: Fs={:.3f} MHz, Ns={:d}, f_offset={:.3f} kHz, {:d} PRNs, {:d} "
         "Doppler bins ({:.0f}..{:.0f} Hz step {:.0f}), incoherent_ms={:d}, track_phase={}",
         _sample_rate / 1e6, _Ns, _f_offset / 1e3, n_prn, n_dop, _doppler_min, _doppler_max,
         _doppler_step, _incoherent_ms, _track_phase);

    // The output frame must hold exactly one [n_prn, RECORD_FLOATS] record block.
    const size_t record_bytes = (size_t)n_prn * RECORD_FLOATS * sizeof(float);
    if ((size_t)out_buf->frame_size != record_bytes)
        FATAL_ERROR("GpsReplicaCorrelator: out_buf frame_size ({:d} B) must equal "
                    "n_prn*RECORD_FLOATS*4 = {:d} B.",
                    out_buf->frame_size, record_bytes);

    // Tag the int16 input for the producer/consumer layout cross-check.
    // out_buf is deliberately left untagged: rawFileWrite (the record sink, and
    // the integration test's capture) refuses NDArray-tagged buffers, so the
    // gps_records layout is documented by make_gps_record_desc() but only a
    // future NDArray-aware consumer would assert it -- same convention as
    // SimpleCrosscorr leaving its out_buf untagged.
    in_buf->set_frame_desc(kotekan_airspy::make_input_desc(in_buf->frame_size / sizeof(int16_t)));

    block.assign(_Ns, 0.0f);
    block_fill = 0;
    in_local = nullptr;
    in_pos = 0;
    in_samples_per_frame = in_buf->frame_size / sizeof(int16_t);

    accum.assign((size_t)n_prn * n_dop * _Ns, 0.0f);
    last_corr.assign((size_t)n_prn * n_dop * _Ns, std::complex<float>(0.0f, 0.0f));
    blocks_in_accum = 0;

    // --- FFTW buffers and plans (planner call needs the global lock) ---
    auto alloc = [&]() { return (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * _Ns); };
    z_time = alloc();
    a_freq = alloc();
    z = alloc();
    d = alloc();
    D = alloc();
    corr = alloc();

    {
        std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
        p_fwd_analytic = fftwf_plan_dft_1d(_Ns, z_time, a_freq, FFTW_FORWARD, FFTW_MEASURE);
        p_inv_analytic = fftwf_plan_dft_1d(_Ns, a_freq, z, FFTW_BACKWARD, FFTW_MEASURE);
        p_fwd_data = fftwf_plan_dft_1d(_Ns, d, D, FFTW_FORWARD, FFTW_MEASURE);
        p_inv_corr = fftwf_plan_dft_1d(_Ns, D, corr, FFTW_BACKWARD, FFTW_MEASURE);

        // Precompute conj(FFT(resampled code)) per PRN. Reuse z_time/a_freq as
        // scratch (plans are already built; FFTW_MEASURE may have dirtied them).
        code_fft_conj.assign((size_t)n_prn * _Ns, std::complex<float>(0.0f, 0.0f));
        std::complex<float>* zt = as_cpx(z_time);
        std::complex<float>* af = as_cpx(a_freq);
        for (int p = 0; p < n_prn; ++p) {
            const auto code = gps::generate_ca_code(_prns[p]); // ±1, length 1023
            for (int n = 0; n < _Ns; ++n) {
                const int chip = (int)((long)n * gps::CA_CODE_LENGTH / _Ns) % gps::CA_CODE_LENGTH;
                zt[n] = std::complex<float>((float)code[chip], 0.0f);
            }
            fftwf_execute(p_fwd_analytic); // a_freq = FFT(code)
            std::complex<float>* dst = &code_fft_conj[(size_t)p * _Ns];
            for (int k = 0; k < _Ns; ++k)
                dst[k] = std::conj(af[k]);
        }
    }
}

GpsReplicaCorrelator::~GpsReplicaCorrelator() {
    std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
    if (p_fwd_analytic)
        fftwf_destroy_plan(p_fwd_analytic);
    if (p_inv_analytic)
        fftwf_destroy_plan(p_inv_analytic);
    if (p_fwd_data)
        fftwf_destroy_plan(p_fwd_data);
    if (p_inv_corr)
        fftwf_destroy_plan(p_inv_corr);
    for (fftwf_complex* b : {z_time, a_freq, z, d, D, corr})
        if (b)
            fftwf_free(b);
}

bool GpsReplicaCorrelator::fill_block() {
    block_fill = 0;
    while (block_fill < _Ns) {
        if (in_local == nullptr || in_pos >= in_samples_per_frame) {
            if (in_local != nullptr) {
                in_buf->mark_frame_empty(unique_name, frame_in);
                frame_in = (frame_in + 1) % in_buf->num_frames;
            }
            in_local = (int16_t*)in_buf->wait_for_full_frame(unique_name, frame_in);
            if (in_local == nullptr)
                return false;
            in_pos = 0;
        }
        const int take = std::min(_Ns - block_fill, in_samples_per_frame - in_pos);
        for (int i = 0; i < take; ++i)
            block[block_fill + i] = (float)in_local[in_pos + i];
        block_fill += take;
        in_pos += take;
    }
    return true;
}

void GpsReplicaCorrelator::to_analytic() {
    std::complex<float>* zt = as_cpx(z_time);
    std::complex<float>* af = as_cpx(a_freq);
    std::complex<float>* zo = as_cpx(z);

    for (int n = 0; n < _Ns; ++n)
        zt[n] = std::complex<float>(block[n], 0.0f);

    fftwf_execute(p_fwd_analytic); // af = FFT(real block)

    // Analytic weighting: keep & double the positive half, zero the negative
    // half, leave DC (and Nyquist when _Ns even) unscaled.
    const int half = _Ns / 2;
    for (int k = 1; k < _Ns; ++k) {
        if (k < half || (_Ns % 2 == 0 && k == half)) {
            if (k != half)
                af[k] *= 2.0f; // strictly-positive frequencies doubled
        } else {
            af[k] = std::complex<float>(0.0f, 0.0f); // negative frequencies zeroed
        }
    }

    fftwf_execute(p_inv_analytic); // zo = IFFT(weighted); FFTW is unnormalized
    const float norm = 1.0f / (float)_Ns;
    for (int n = 0; n < _Ns; ++n)
        zo[n] *= norm;
}

void GpsReplicaCorrelator::correlate_prn(int p) {
    const int n_dop = (int)_doppler_grid.size();
    std::complex<float>* zc = as_cpx(z);
    std::complex<float>* dc = as_cpx(d);
    std::complex<float>* Dc = as_cpx(D);
    std::complex<float>* cc = as_cpx(corr);
    const std::complex<float>* code = &code_fft_conj[(size_t)p * _Ns];
    const float inv_ns = 1.0f / (float)_Ns;

    for (int di = 0; di < n_dop; ++di) {
        // Carrier wipeoff: d[n] = z[n] * exp(-j 2 pi (f_offset+fd) n / Fs),
        // generated with an incremental phasor (one complex multiply per
        // sample) rather than a per-sample trig call.
        const double w = -2.0 * M_PI * (_f_offset + _doppler_grid[di]) / _sample_rate;
        const std::complex<float> rot((float)std::cos(w), (float)std::sin(w));
        std::complex<float> ph(1.0f, 0.0f);
        for (int n = 0; n < _Ns; ++n) {
            dc[n] = zc[n] * ph;
            ph *= rot;
        }

        fftwf_execute(p_fwd_data); // Dc = FFT(d)
        for (int k = 0; k < _Ns; ++k)
            Dc[k] = Dc[k] * code[k]; // multiply by conj(FFT(code))
        fftwf_execute(p_inv_corr);  // cc = IFFT(Dc), unnormalized

        float* acc = &accum[((size_t)p * n_dop + di) * _Ns];
        std::complex<float>* lc = &last_corr[((size_t)p * n_dop + di) * _Ns];
        for (int k = 0; k < _Ns; ++k) {
            const std::complex<float> v = cc[k] * inv_ns;
            acc[k] += std::norm(v); // |v|^2 accumulates incoherently
            lc[k] = v;              // most-recent block's complex correlation
        }
    }
}

void GpsReplicaCorrelator::main_thread() {
    frame_in = 0;
    frame_out = 0;

    const int n_prn = (int)_prns.size();
    const int n_dop = (int)_doppler_grid.size();

    while (!stop_thread) {
        if (!fill_block())
            break;

        to_analytic();
        for (int p = 0; p < n_prn; ++p)
            correlate_prn(p);
        blocks_in_accum++;

        if (blocks_in_accum < _incoherent_ms)
            continue;

        // --- emit one record block ---
        float* out_local = (float*)out_buf->wait_for_empty_frame(unique_name, frame_out);
        if (out_local == nullptr)
            break;

        for (int p = 0; p < n_prn; ++p) {
            // Find the peak over the (Doppler, lag) surface for this PRN, and
            // the mean for a crude SNR.
            float peak = -1.0f;
            int best_di = 0, best_k = 0;
            double sum = 0.0;
            for (int di = 0; di < n_dop; ++di) {
                const float* acc = &accum[((size_t)p * n_dop + di) * _Ns];
                for (int k = 0; k < _Ns; ++k) {
                    sum += acc[k];
                    if (acc[k] > peak) {
                        peak = acc[k];
                        best_di = di;
                        best_k = k;
                    }
                }
            }
            const double mean = sum / ((double)n_dop * _Ns);
            const std::complex<float> cpx =
                last_corr[((size_t)p * n_dop + best_di) * _Ns + best_k];

            // peak/last_corr hold |.|^2 magnitudes; report amplitude.
            const float peak_amp = std::sqrt(peak / (float)_incoherent_ms);
            const float snr = (mean > 0.0) ? (float)(peak / mean) : 0.0f;
            const float code_phase_chips = (float)best_k * (float)gps::CA_CODE_LENGTH / (float)_Ns;

            float* rec = out_local + (size_t)p * RECORD_FLOATS;
            rec[0] = (float)_prns[p];
            rec[1] = (float)_doppler_grid[best_di];
            rec[2] = code_phase_chips;
            rec[3] = peak_amp;
            rec[4] = _track_phase ? cpx.real() : 0.0f;
            rec[5] = _track_phase ? cpx.imag() : 0.0f;
            rec[6] = snr;

            DEBUG("PRN {:2d}: doppler {:+.0f} Hz, code phase {:.1f} chips, amp {:.3g}, snr {:.1f}",
                  _prns[p], _doppler_grid[best_di], code_phase_chips, peak_amp, snr);
        }

        out_buf->mark_frame_full(unique_name, frame_out);
        frame_out = (frame_out + 1) % out_buf->num_frames;

        std::fill(accum.begin(), accum.end(), 0.0f);
        blocks_in_accum = 0;
    }
}
