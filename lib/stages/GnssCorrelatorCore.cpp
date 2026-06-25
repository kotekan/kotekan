#include "GnssCorrelatorCore.hpp"

#include "fftwPlannerLock.hpp" // for fftw_planner_mutex
#include "gpsCACode.hpp"       // for generate_ca_code
#include "gpsL2CCode.hpp"      // for generate_l2cm_code, generate_l2cl_code
#include "gpsL5Code.hpp"       // for generate_l5i_code, generate_l5q_code

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

namespace gnss {

// fftwf_complex is layout-compatible with std::complex<float>.
static inline std::complex<float>* as_cpx(fftwf_complex* p) {
    return reinterpret_cast<std::complex<float>*>(p);
}

// Per-PRN spreading code (bipolar +-1) for the selected signal. Same dispatch
// as GpsReplicaCorrelator::replica_code.
static std::vector<int8_t> replica_code(const SignalDescriptor& sig, int prn) {
    const std::string name = sig.name;
    if (name == "GPS_L1CA") {
        const auto a = gps::generate_ca_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GPS_L2C_CM")
        return gps::generate_l2cm_code(prn);
    if (name == "GPS_L2C_CL")
        return gps::generate_l2cl_code(prn);
    if (name == "GPS_L5_I") {
        const auto a = gps::generate_l5i_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GPS_L5_Q") {
        const auto a = gps::generate_l5q_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    throw std::runtime_error("CorrelatorCore: no replica generator for " + name);
}

CorrelatorCore::CorrelatorCore(const SignalDescriptor& sig, double sample_rate, double f_offset,
                               const std::vector<int>& prns) :
    _sig(sig), _sample_rate(sample_rate), _f_offset(f_offset), _prns(prns),
    z_time(nullptr), a_freq(nullptr), z(nullptr), d(nullptr), Dsave(nullptr), D(nullptr),
    corr(nullptr), p_fwd_analytic(nullptr), p_inv_analytic(nullptr), p_fwd_data(nullptr),
    p_inv_corr(nullptr) {

    if (_sig.time_assisted)
        throw std::runtime_error("CorrelatorCore: time-assisted signals (e.g. L2C CL) are not "
                                 "supported; use GpsReplicaCorrelator.");
    if (_prns.empty())
        throw std::runtime_error("CorrelatorCore: empty PRN list.");

    _Ns = (int)std::round(_sample_rate * _sig.code_period_s);
    constexpr int MAX_NS = 1 << 20;
    if (_Ns < 8 || _Ns > MAX_NS)
        throw std::runtime_error("CorrelatorCore: Ns out of range; check sample_rate/signal.");

    const int comb_mult = _sig.time_multiplexed ? 2 : 1;
    const int n_prn = (int)_prns.size();

    auto alloc = [&]() { return (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * _Ns); };
    z_time = alloc();
    a_freq = alloc();
    z = alloc();
    d = alloc();
    Dsave = alloc();
    D = alloc();
    corr = alloc();

    std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
    p_fwd_analytic = fftwf_plan_dft_1d(_Ns, z_time, a_freq, FFTW_FORWARD, FFTW_MEASURE);
    p_inv_analytic = fftwf_plan_dft_1d(_Ns, a_freq, z, FFTW_BACKWARD, FFTW_MEASURE);
    p_fwd_data = fftwf_plan_dft_1d(_Ns, d, Dsave, FFTW_FORWARD, FFTW_MEASURE);
    p_inv_corr = fftwf_plan_dft_1d(_Ns, D, corr, FFTW_BACKWARD, FFTW_MEASURE);

    // One full-period replica FFT per PRN, precomputed once. A time-multiplexed
    // signal (L2C CM/CL) interleaves two components at 2x the chip rate; the
    // replica fills only its tdm_phase parity of the combined chips.
    code_fft_conj.assign((size_t)n_prn * _Ns, std::complex<float>(0.0f, 0.0f));
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

CorrelatorCore::~CorrelatorCore() {
    std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
    for (fftwf_plan p : {p_fwd_analytic, p_inv_analytic, p_fwd_data, p_inv_corr})
        if (p)
            fftwf_destroy_plan(p);
    for (fftwf_complex* b : {z_time, a_freq, z, d, Dsave, D, corr})
        if (b)
            fftwf_free(b);
}

void CorrelatorCore::to_analytic(const float* block) {
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
                af[k] *= 2.0f;
        } else {
            af[k] = std::complex<float>(0.0f, 0.0f);
        }
    }

    fftwf_execute(p_inv_analytic); // zo = IFFT(weighted); FFTW unnormalized
    const float norm = 1.0f / (float)_Ns;
    for (int n = 0; n < _Ns; ++n)
        zo[n] *= norm;
}

void CorrelatorCore::prepare_doppler(double fd) {
    std::complex<float>* zc = as_cpx(z);
    std::complex<float>* dc = as_cpx(d);
    // Carrier wipeoff d[n] = z[n] * exp(-j 2 pi (f_offset+fd) n / Fs), generated
    // with an incremental phasor (one complex multiply per sample).
    const double w = -2.0 * M_PI * (_f_offset + fd) / _sample_rate;
    const std::complex<float> rot((float)std::cos(w), (float)std::sin(w));
    std::complex<float> ph(1.0f, 0.0f);
    for (int n = 0; n < _Ns; ++n) {
        dc[n] = zc[n] * ph;
        ph *= rot;
    }
    fftwf_execute(p_fwd_data); // Dsave = FFT(d), kept for reuse across PRNs
}

void CorrelatorCore::correlate_into(int p, float* mag2, std::complex<float>* cpx) {
    std::complex<float>* Ds = as_cpx(Dsave);
    std::complex<float>* Di = as_cpx(D);
    std::complex<float>* cc = as_cpx(corr);
    const std::complex<float>* code = &code_fft_conj[(size_t)p * _Ns];

    for (int k = 0; k < _Ns; ++k)
        Di[k] = Ds[k] * code[k]; // multiply by conj(FFT(code)) into the IFFT input
    fftwf_execute(p_inv_corr);   // cc = IFFT(Di), unnormalized

    const float inv_ns = 1.0f / (float)_Ns;
    for (int k = 0; k < _Ns; ++k) {
        const std::complex<float> v = cc[k] * inv_ns;
        mag2[k] += std::norm(v);
        if (cpx)
            cpx[k] = v;
    }
}

} // namespace gnss
