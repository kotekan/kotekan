#include "GpsReplicaCorrelator.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp" // for make_input_desc, make_gps_record_desc
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "fftwPlannerLock.hpp" // for fftw_planner_mutex
#include "gnssSignal.hpp"      // for SignalDescriptor, signal_by_name
#include "gpsCACode.hpp"       // for generate_ca_code
#include "gpsL2CCode.hpp"      // for generate_l2cm_code, generate_l2cl_code
#include "kotekanLogging.hpp"  // for DEBUG, INFO, FATAL_ERROR

#include "restServer.hpp" // for restServer, connectionInstance, HTTP_RESPONSE

#include <algorithm> // for max
#include <chrono>    // for system_clock (record UTC stamp)
#include <cmath>     // for round, cos, sin, sqrt, M_PI
#include <complex>   // for complex
#include <functional>// for bind
#include <memory>    // for shared_ptr
#include <mutex>     // for lock_guard
#include <set>       // for set (mask lookup)
#include <stdexcept> // for runtime_error
#include <string>    // for string
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GpsReplicaCorrelator);

// fftwf_complex is layout-compatible with std::complex<float>; this alias keeps
// the DSP arithmetic readable while the FFTW calls keep using the raw arrays.
static inline std::complex<float>* as_cpx(fftwf_complex* p) {
    return reinterpret_cast<std::complex<float>*>(p);
}

// Sub-sample peak offset in [-0.5, 0.5] from a 3-point parabolic fit around a
// maximum (left, centre, right). Returns 0 if centre is not the peak or the
// curvature is degenerate.
static inline double parab_offset(double l, double c, double r) {
    const double denom = l - 2.0 * c + r;
    if (denom >= 0.0) // not a concave peak
        return 0.0;
    double d = 0.5 * (l - r) / denom;
    if (d < -0.5)
        d = -0.5;
    if (d > 0.5)
        d = 0.5;
    return d;
}

// Once a PRN is phase-locked its Doppler is known to ~Hz, so the per-block
// search narrows to +-this many grid bins around the tracked value instead of
// sweeping the full grid -- the dominant compute saving in tracking mode.
static constexpr int LOCK_DOPPLER_GUARD = 2;
// FLL loop-filter gain (per block) for the tracked-Doppler estimate.
static constexpr double FLL_ALPHA = 0.1;

// Per-PRN spreading code (bipolar +-1) for the selected signal. Only called
// with descriptors already validated by signal_by_name(); the throw is a
// safety net for an unwired signal.
static std::vector<int8_t> replica_code(const gnss::SignalDescriptor& sig, int prn) {
    const std::string name = sig.name;
    if (name == "GPS_L1CA") {
        const auto a = gps::generate_ca_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GPS_L2C_CM")
        return gps::generate_l2cm_code(prn);
    if (name == "GPS_L2C_CL")
        return gps::generate_l2cl_code(prn);
    throw std::runtime_error("GpsReplicaCorrelator: no replica generator for " + name);
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

    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr)
        FATAL_ERROR("GpsReplicaCorrelator: unknown signal '{:s}' (GPS_L1CA, GPS_L2C_CM, "
                    "GPS_L2C_CL).",
                    signal);
    _sig = *sig;

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    _capture_utc0 = config.get_default<double>(unique_name, "capture_utc0", 0.0);
    _f_offset = config.get_default<double>(unique_name, "f_offset", 1e6);
    _doppler_min = config.get_default<double>(unique_name, "doppler_min", -6000.0);
    _doppler_max = config.get_default<double>(unique_name, "doppler_max", 6000.0);
    _doppler_step = config.get_default<double>(unique_name, "doppler_step", 500.0);
    _incoherent_ms = config.get_default<int>(unique_name, "incoherent_ms", 1);
    _track_phase = config.get_default<bool>(unique_name, "track_phase", false);
    _lock_snr = config.get_default<double>(unique_name, "lock_snr", 15.0);
    _prns = config.get<std::vector<int>>(unique_name, "prns");

    if (_prns.empty())
        FATAL_ERROR("GpsReplicaCorrelator: 'prns' list is empty; nothing to correlate.");
    if (_incoherent_ms < 1)
        FATAL_ERROR("GpsReplicaCorrelator: incoherent_ms must be >= 1.");
    if (_track_phase && _incoherent_ms != 1)
        WARN("GpsReplicaCorrelator: track_phase with incoherent_ms={:d}: phase is taken from "
             "the last block of each window and nav-bit tracking runs at that ({:d} ms) cadence; "
             "use incoherent_ms=1 for clean 1 ms phase.",
             _incoherent_ms, _incoherent_ms);
    if (_doppler_step <= 0 || _doppler_max < _doppler_min)
        FATAL_ERROR("GpsReplicaCorrelator: invalid Doppler grid.");

    _time_assisted = _sig.time_assisted;
    const int comb_mult = _sig.time_multiplexed ? 2 : 1;
    _comb_rate = _sig.chip_rate_hz * comb_mult;

    // Block size. Normally one full code period (the FFT parallel-code-phase
    // search needs a period). A time-assisted long-period pilot (L2C CL, 1.5 s)
    // instead uses a short coherent window: the code phase is predicted from
    // time, so the per-block replica is that predicted slice of the code and the
    // FFT only refines the residual within +-half the window.
    double block_period;
    if (_time_assisted) {
        block_period = config.get_default<double>(unique_name, "coherent_ms", 10.0) * 1e-3;
        const double seed_chips = config.get_default<double>(unique_name, "code_phase_seed_chips",
                                                             0.0);
        _seed_comb = seed_chips * comb_mult + _sig.tdm_phase; // component chip -> combined index
    } else {
        block_period = _sig.code_period_s;
        _seed_comb = 0.0;
    }
    _Ns = (int)std::round(_sample_rate * block_period);
    constexpr int MAX_NS = 1 << 20; // ~1 M-point FFT cap
    if (_Ns < 8 || _Ns > MAX_NS)
        FATAL_ERROR("GpsReplicaCorrelator: signal '{:s}' block {:.4f} s at Fs {:.3f} MHz gives "
                    "Ns={:d} (cap {:d}); reduce coherent_ms or sample_rate.",
                    _sig.name, block_period, _sample_rate / 1e6, _Ns, MAX_NS);

    for (double f = _doppler_min; f <= _doppler_max + 1e-6; f += _doppler_step)
        _doppler_grid.push_back(f);

    const int n_prn = (int)_prns.size();
    const int n_dop = (int)_doppler_grid.size();

    INFO("GpsReplicaCorrelator: signal={:s} ({:.0f} chips, {:.3f} ms), Fs={:.3f} MHz, Ns={:d}, "
         "f_offset={:.3f} kHz, {:d} PRNs, {:d} Doppler bins ({:.0f}..{:.0f} Hz step {:.0f}), "
         "incoherent_ms={:d}, track_phase={}",
         _sig.name, (double)_sig.code_length, _sig.code_period_s * 1e3, _sample_rate / 1e6, _Ns,
         _f_offset / 1e3, n_prn, n_dop, _doppler_min, _doppler_max, _doppler_step, _incoherent_ms,
         _track_phase);

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

    // Stage 2 tracking state, one slot per PRN, all unlocked to start.
    trk_locked.assign(n_prn, 0);
    trk_sign.assign(n_prn, 1.0f);
    trk_prev.assign(n_prn, std::complex<float>(0.0f, 0.0f));
    trk_cont_phase.assign(n_prn, 0.0);
    trk_prev_wrapped.assign(n_prn, 0.0);
    trk_prev_doppler.assign(n_prn, 0.0);
    trk_prev_code_phase.assign(n_prn, 0.0);
    trk_doppler.assign(n_prn, 0.0);

    // Go/no-go mask: default all on, or the configured active_prns subset.
    const auto active_prns = config.get_default<std::vector<int>>(unique_name, "active_prns", {});
    _active.assign(n_prn, 1);
    if (!active_prns.empty()) {
        const std::set<int> on(active_prns.begin(), active_prns.end());
        for (int i = 0; i < n_prn; ++i)
            _active[i] = on.count(_prns[i]) ? 1 : 0;
    }

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

        code_fft_conj.assign((size_t)n_prn * _Ns, std::complex<float>(0.0f, 0.0f));

        if (_time_assisted) {
            // Block is a short window of a long code; the replica differs every
            // block (a different slice of the code), so cache the full code per
            // PRN and (re)build the windowed FFTs per block in main_thread.
            full_code.resize(n_prn);
            for (int p = 0; p < n_prn; ++p)
                full_code[p] = replica_code(_sig, _prns[p]);
        } else {
            // Periodic code: one full-period replica FFT per PRN, precomputed
            // once. Reuse z_time/a_freq as scratch. Combined-stream chip count
            // over one period = code_length * comb_mult (a time-multiplexed
            // signal interleaves two components; this code fills its parity).
            std::complex<float>* zt = as_cpx(z_time);
            std::complex<float>* af = as_cpx(a_freq);
            const long comb_chips = (long)_sig.code_length * comb_mult;
            for (int p = 0; p < n_prn; ++p) {
                const std::vector<int8_t> code = replica_code(_sig, _prns[p]);
                for (int n = 0; n < _Ns; ++n) {
                    const long c = (long)n * comb_chips / _Ns % comb_chips;
                    float chipval;
                    if (_sig.time_multiplexed)
                        chipval = ((c & 1) == _sig.tdm_phase) ? (float)code[c >> 1] : 0.0f;
                    else
                        chipval = (float)code[c];
                    zt[n] = std::complex<float>(chipval, 0.0f);
                }
                fftwf_execute(p_fwd_analytic); // a_freq = FFT(resampled code)
                std::complex<float>* dst = &code_fft_conj[(size_t)p * _Ns];
                for (int k = 0; k < _Ns; ++k)
                    dst[k] = std::conj(af[k]);
            }
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

void GpsReplicaCorrelator::rebuild_replica_ffts(long long block_start_sample) {
    // Predicted combined-chip phase at the block start: the time-assist seed
    // plus the chips elapsed. The long code repeats every comb_period combined
    // chips, so wrap there. Reuses z_time/a_freq as scratch (called before
    // to_analytic, which then overwrites them with the data block).
    std::complex<float>* zt = as_cpx(z_time);
    std::complex<float>* af = as_cpx(a_freq);
    const long comb_period = (long)_sig.code_length * 2;   // CL: CM+CL slots over 1.5 s
    const double per_sample = _comb_rate / _sample_rate;   // combined chips per sample
    const double base = _seed_comb + _comb_rate * (double)block_start_sample / _sample_rate;
    const long base0 = (long)std::floor(base);
    const int n_prn = (int)_prns.size();
    for (int p = 0; p < n_prn; ++p) {
        const std::vector<int8_t>& code = full_code[p]; // ±1, full long code
        for (int n = 0; n < _Ns; ++n) {
            long c = (base0 + (long)((double)n * per_sample)) % comb_period;
            if (c < 0)
                c += comb_period;
            // this code occupies its tdm_phase parity of the combined chips
            zt[n] = std::complex<float>(((c & 1) == _sig.tdm_phase) ? (float)code[c >> 1] : 0.0f,
                                        0.0f);
        }
        fftwf_execute(p_fwd_analytic); // a_freq = FFT(windowed replica)
        std::complex<float>* dst = &code_fft_conj[(size_t)p * _Ns];
        for (int k = 0; k < _Ns; ++k)
            dst[k] = std::conj(af[k]);
    }
}

void GpsReplicaCorrelator::correlate_prn(int p, int di_lo, int di_hi) {
    const int n_dop = (int)_doppler_grid.size();
    std::complex<float>* zc = as_cpx(z);
    std::complex<float>* dc = as_cpx(d);
    std::complex<float>* Dc = as_cpx(D);
    std::complex<float>* cc = as_cpx(corr);
    const std::complex<float>* code = &code_fft_conj[(size_t)p * _Ns];
    const float inv_ns = 1.0f / (float)_Ns;

    for (int di = di_lo; di <= di_hi; ++di) {
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

void GpsReplicaCorrelator::track_phase_update(int p, std::complex<float> raw, float snr,
                                              double doppler_bin, double doppler_seed,
                                              double code_phase_chips, float& nav_sign,
                                              float& phase_cont, float& doppler_out) {
    // Wrap a phase difference into (-pi, pi].
    auto wrap = [](double x) {
        while (x > M_PI)
            x -= 2.0 * M_PI;
        while (x <= -M_PI)
            x += 2.0 * M_PI;
        return x;
    };

    // Allowable block-to-block bin wander for the peak to count as the same
    // signal: +-1.5 Doppler bins and a few chips of code phase (circular).
    constexpr double CODE_TOL_CHIPS = 3.0;
    const double dopp_tol = 1.5 * _doppler_step;
    const double dt = _incoherent_ms * 1e-3; // integration period, s

    // Below the lock threshold the peak is just the noise maximum -- drop lock.
    if (snr < _lock_snr) {
        trk_locked[p] = 0;
        nav_sign = 0.0f;
        phase_cont = 0.0f;
        doppler_out = (float)doppler_seed;
        return;
    }

    double code_jump = std::fabs(code_phase_chips - trk_prev_code_phase[p]);
    code_jump = std::min(code_jump, (double)_sig.code_length - code_jump); // circular
    const bool consistent = trk_locked[p]
                            && std::fabs(doppler_bin - trk_prev_doppler[p]) <= dopp_tol
                            && code_jump <= CODE_TOL_CHIPS;

    if (!consistent) {
        // (Re)acquire lock: start a fresh continuous-phase record, seeding the
        // FLL with the parabola-refined acquisition Doppler.
        trk_sign[p] = 1.0f;
        const double w = std::atan2(raw.imag(), raw.real());
        trk_cont_phase[p] = w;
        trk_prev_wrapped[p] = w;
        trk_prev[p] = raw;
        trk_doppler[p] = doppler_seed;
    } else {
        // Predict the expected phase advance from the tracked carrier frequency
        // (residual relative to this block's wipe bin), so a leftover ~pi step
        // is unambiguously a nav-bit flip even when the residual Doppler is a
        // large fraction of a bin -- the FLL "de-rotates" the detector.
        const double pred = trk_prev_wrapped[p] + 2.0 * M_PI * (trk_doppler[p] - doppler_bin) * dt;
        std::complex<float> corr = trk_sign[p] * raw;
        double w = std::atan2(corr.imag(), corr.real());
        if (std::fabs(wrap(w - pred)) > M_PI / 2.0) {
            trk_sign[p] = -trk_sign[p];
            corr = -corr;
            w = std::atan2(corr.imag(), corr.real());
        }
        // Accumulate the true (small) phase increment; update the FLL estimate.
        const double dphi = wrap(w - trk_prev_wrapped[p]);
        trk_cont_phase[p] += dphi;
        const double meas_doppler = doppler_bin + dphi / (2.0 * M_PI * dt);
        trk_doppler[p] = (1.0 - FLL_ALPHA) * trk_doppler[p] + FLL_ALPHA * meas_doppler;
        trk_prev_wrapped[p] = w;
        trk_prev[p] = corr;
    }

    trk_locked[p] = 1;
    trk_prev_doppler[p] = doppler_bin;
    trk_prev_code_phase[p] = code_phase_chips;
    nav_sign = trk_sign[p];
    phase_cont = (float)trk_cont_phase[p];
    doppler_out = (float)trk_doppler[p];
}

void GpsReplicaCorrelator::get_mask_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply;
    std::lock_guard<std::mutex> lk(_mask_mutex);
    reply["prns"] = _prns;
    std::vector<bool> act(_active.begin(), _active.end());
    reply["active"] = act;
    conn.send_json_reply(reply);
}

void GpsReplicaCorrelator::set_mask_callback(kotekan::connectionInstance& conn,
                                             nlohmann::json& request) {
    try {
        std::lock_guard<std::mutex> lk(_mask_mutex);
        if (request.count("active_prns")) {
            // Whole-mask update: enable exactly the listed PRNs (an almanac job
            // posts the currently-visible set), disable the rest.
            const auto list = request["active_prns"].get<std::vector<int>>();
            const std::set<int> on(list.begin(), list.end());
            for (size_t i = 0; i < _prns.size(); ++i)
                _active[i] = on.count(_prns[i]) ? 1 : 0;
        } else if (request.count("prn")) {
            // Single toggle.
            const int prn = request["prn"];
            const bool on = request.value("active", true);
            for (size_t i = 0; i < _prns.size(); ++i)
                if (_prns[i] == prn)
                    _active[i] = on ? 1 : 0;
        } else {
            conn.send_error("expected 'active_prns' list or 'prn'+'active'",
                            kotekan::HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
    } catch (const std::exception& e) {
        conn.send_error(e.what(), kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    get_mask_callback(conn); // echo the resulting mask
}

void GpsReplicaCorrelator::main_thread() {
    frame_in = 0;
    frame_out = 0;

    const int n_prn = (int)_prns.size();
    const int n_dop = (int)_doppler_grid.size();

    using namespace std::placeholders;
    kotekan::restServer& rest_server = kotekan::restServer::instance();
    rest_server.register_get_callback(unique_name + "/get_mask",
                                      std::bind(&GpsReplicaCorrelator::get_mask_callback, this, _1));
    rest_server.register_post_callback(
        unique_name + "/set_mask",
        std::bind(&GpsReplicaCorrelator::set_mask_callback, this, _1, _2));

    // Doppler bin window for PRN p: the full grid when unlocked, a few bins
    // around the tracked Doppler once locked. Tracking state only changes at
    // emit, so this is stable across an integration window.
    auto win = [&](int p, int& lo, int& hi) {
        if (_track_phase && trk_locked[p]) {
            int c = (int)std::lround((trk_doppler[p] - _doppler_min) / _doppler_step);
            c = std::max(0, std::min(n_dop - 1, c));
            lo = std::max(0, c - LOCK_DOPPLER_GUARD);
            hi = std::min(n_dop - 1, c + LOCK_DOPPLER_GUARD);
        } else {
            lo = 0;
            hi = n_dop - 1;
        }
    };

    // The carrier wipe-off in correlate_prn restarts its phasor at 1 each block
    // (local sample index). To keep the removed carrier continuous across block
    // boundaries we record each block's global start sample and de-rotate the
    // kept peak by exp(-j 2 pi (f_offset+fd) * block_start / Fs) below; without
    // this the peak phase jumps by 2 pi (f_offset+fd) Ns/Fs every block (e.g. pi
    // per block for fd an odd multiple of the grid step), which the nav-bit
    // detector would mistake for a bit flip every block.
    const long long fs_int = std::llround(_sample_rate);
    long long block_index = 0;
    long long last_block_start = 0;
    std::vector<uint8_t> active(n_prn, 1); // per-block snapshot of the go/no-go mask

    while (!stop_thread) {
        if (!fill_block())
            break;

        last_block_start = block_index * (long long)_Ns;
        block_index++;

        {
            std::lock_guard<std::mutex> lk(_mask_mutex);
            active = _active;
        }

        // Time-assisted: window the replica to this block's predicted code phase
        // before correlating (must precede to_analytic, which reuses the scratch).
        if (_time_assisted)
            rebuild_replica_ffts(last_block_start);

        to_analytic();
        for (int p = 0; p < n_prn; ++p) {
            if (!active[p])
                continue; // masked off: skip correlation entirely
            int lo, hi;
            win(p, lo, hi);
            correlate_prn(p, lo, hi);
        }
        blocks_in_accum++;

        if (blocks_in_accum < _incoherent_ms)
            continue;

        // --- emit one record block ---
        float* out_local = (float*)out_buf->wait_for_empty_frame(unique_name, frame_out);
        if (out_local == nullptr)
            break;

        // Record timestamp (UTC seconds), shared by all PRNs in this block.
        // When capture_utc0 is set we anchor to the sample stream
        // (capture_utc0 + sample/Fs) -- correct for offline replay and the time
        // base the L2C CL seed uses; otherwise fall back to the wall-clock at
        // emit (fine for alt/az, where the sky moves ~degrees/minute).
        const double utc =
            (_capture_utc0 > 0.0)
                ? _capture_utc0 + (double)last_block_start / _sample_rate
                : std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
                      .count();

        for (int p = 0; p < n_prn; ++p) {
            float* rec = out_local + (size_t)p * RECORD_FLOATS;

            // Masked-off PRN: emit a zeroed record (keeps the frame layout) and
            // drop any stale lock so it re-acquires cleanly when re-enabled.
            if (!active[p]) {
                rec[0] = (float)_prns[p];
                for (int f = 1; f < RECORD_UTC_SLOT; ++f)
                    rec[f] = 0.0f;
                *reinterpret_cast<double*>(rec + RECORD_UTC_SLOT) = utc;
                trk_locked[p] = 0;
                continue;
            }

            // Search only the bins this PRN actually correlated (full grid when
            // unlocked, narrow window when locked); the rest of accum is zero.
            int di_lo, di_hi;
            win(p, di_lo, di_hi);

            float peak = -1.0f;
            int best_di = di_lo, best_k = 0;
            double sum = 0.0;
            for (int di = di_lo; di <= di_hi; ++di) {
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
            const double mean = sum / ((double)(di_hi - di_lo + 1) * _Ns);

            // De-rotate the kept peak to undo the per-block wipe-off phase reset,
            // so its phase is continuous across blocks (see note above the loop).
            // Reduce block_start mod Fs first for an exact fractional cycle count
            // (valid for integer-Hz f_offset / Doppler / sample_rate).
            const double fwipe = _f_offset + _doppler_grid[best_di];
            double cyc = fwipe * (double)(last_block_start % fs_int) / _sample_rate;
            cyc -= std::floor(cyc);
            const double carg = -2.0 * M_PI * cyc;
            const std::complex<float> cpx =
                last_corr[((size_t)p * n_dop + best_di) * _Ns + best_k]
                * std::complex<float>((float)std::cos(carg), (float)std::sin(carg));

            // peak/last_corr hold |.|^2 magnitudes; report amplitude.
            const float peak_amp = std::sqrt(peak / (float)_incoherent_ms);
            const float snr = (mean > 0.0) ? (float)(peak / mean) : 0.0f;

            // Sub-bin / sub-sample peak refinement by parabolic interpolation on
            // the |corr|^2 surface (skipped at a window edge / for Doppler).
            const float* accp = &accum[(size_t)p * n_dop * _Ns];
            double dop_refined = _doppler_grid[best_di];
            if (best_di > di_lo && best_di < di_hi) {
                const double dd = parab_offset(accp[((size_t)(best_di - 1)) * _Ns + best_k], peak,
                                               accp[((size_t)(best_di + 1)) * _Ns + best_k]);
                dop_refined += dd * _doppler_step;
            }
            const int km1 = (best_k - 1 + _Ns) % _Ns, kp1 = (best_k + 1) % _Ns;
            const double dk = parab_offset(accp[(size_t)best_di * _Ns + km1], peak,
                                           accp[(size_t)best_di * _Ns + kp1]);
            const float code_phase_chips =
                (float)(((best_k + dk) * (double)_sig.code_length) / _Ns);

            float nav_sign = 0.0f, phase_cont = 0.0f, doppler_out = (float)dop_refined;
            if (_track_phase)
                track_phase_update(p, cpx, snr, _doppler_grid[best_di], dop_refined,
                                   code_phase_chips, nav_sign, phase_cont, doppler_out);

            rec[0] = (float)_prns[p];
            rec[1] = doppler_out; // FLL-refined when locked, else parabola-refined
            rec[2] = code_phase_chips;
            rec[3] = peak_amp;
            rec[4] = _track_phase ? cpx.real() : 0.0f;
            rec[5] = _track_phase ? cpx.imag() : 0.0f;
            rec[6] = snr;
            rec[7] = nav_sign;
            rec[8] = phase_cont;
            *reinterpret_cast<double*>(rec + RECORD_UTC_SLOT) = utc; // trailing float64

            DEBUG("PRN {:2d}: doppler {:+.1f} Hz, code phase {:.2f} chips, amp {:.3g}, snr {:.1f}, "
                  "nav {:+.0f}, phase {:.2f} rad",
                  _prns[p], doppler_out, code_phase_chips, peak_amp, snr, nav_sign, phase_cont);
        }

        out_buf->mark_frame_full(unique_name, frame_out);
        frame_out = (frame_out + 1) % out_buf->num_frames;

        std::fill(accum.begin(), accum.end(), 0.0f);
        blocks_in_accum = 0;
    }
}
