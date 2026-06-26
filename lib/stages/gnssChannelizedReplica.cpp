#include "gnssChannelizedReplica.hpp"

#include "fftwPlannerLock.hpp" // for fftw_planner_mutex
#include "gnssBandPlan.hpp"    // for ChannelBand, covering_channels
#include "gpsCACode.hpp"       // for generate_ca_code
#include "gpsL2CCode.hpp"      // for generate_l2cm_code, generate_l2cl_code
#include "gpsL5Code.hpp"       // for generate_l5i_code, generate_l5q_code

#include <algorithm> // for max
#include <cmath>     // for cos, floor, llround, M_PI
#include <mutex>     // for lock_guard
#include <numeric>   // for gcd
#include <stdexcept> // for runtime_error
#include <string>    // for string

namespace gnss {

using cf = std::complex<float>;

namespace {
// Dispatch the primary spreading code by signal name. NOTE: L2C is time-multiplexed
// (CM/CL interleaved to 1.023 Mcps) -- here we generate the bare component code at
// its 511.5 kcps rate, which is approximate for the real interleaved signal; and
// L2C CL is time-assisted (1.5 s period -- track-only, not blind-acquirable). L5
// (I5/Q5) is the clean, fully-supported L2/L5 target. The NH secondary overlay
// (L5 NH10/NH20, applied per 1 ms primary period) is NOT applied here -- a constant
// sign within a <=1 ms window, so it only matters for multi-period coherent
// integration (the pilot's long-integration benefit), a separate refinement.
std::vector<int8_t> signal_code(const std::string& name, int prn) {
    if (name == "GPS_L5_I") {
        auto a = gps::generate_l5i_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GPS_L5_Q") {
        auto a = gps::generate_l5q_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GPS_L2C_CM")
        return gps::generate_l2cm_code(prn);
    if (name == "GPS_L2C_CL")
        return gps::generate_l2cl_code(prn);
    auto a = gps::generate_ca_code(prn); // default: GPS_L1CA
    return std::vector<int8_t>(a.begin(), a.end());
}
} // namespace

ChannelizedReplicaBank::ChannelizedReplicaBank(const SignalDescriptor& sig, double sample_rate,
                                               double f_offset, int spectrum_length, int num_taps,
                                               dsp::Window window, const std::vector<int>& prns) :
    _sig(sig), _sample_rate(sample_rate), _f_offset(f_offset), _N(spectrum_length),
    _fft_len(2 * spectrum_length), _num_taps(num_taps), _fold(nullptr), _spec(nullptr),
    _p_fwd(nullptr) {

    _proto = dsp::pfb_prototype(_fft_len, _num_taps, window);

    const long code_samples = (long)std::llround(_sig.code_period_s * _sample_rate);
    _repl_period_hops =
        (int)(code_samples / std::max<long>(1, std::gcd((long)_fft_len, code_samples)));

    _full_code.resize(prns.size());
    for (size_t p = 0; p < prns.size(); ++p)
        _full_code[p] = signal_code(_sig.name, prns[p]);

    std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
    _fold = (float*)fftwf_malloc(sizeof(float) * _fft_len);
    _spec = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (_N + 1));
    _p_fwd = fftwf_plan_dft_r2c_1d(_fft_len, _fold, _spec, FFTW_ESTIMATE);
}

ChannelizedReplicaBank::~ChannelizedReplicaBank() {
    if (_p_fwd) {
        std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
        fftwf_destroy_plan(_p_fwd);
    }
    if (_fold)
        fftwf_free(_fold);
    if (_spec)
        fftwf_free(_spec);
}

int8_t ChannelizedReplicaBank::code_chip(int p, double chip_phase) const {
    const long len = _sig.code_length;
    long idx = (long)std::floor(chip_phase);
    idx %= len;
    if (idx < 0)
        idx += len;
    return _full_code[p][idx];
}

std::vector<int> ChannelizedReplicaBank::covering_bins(double doppler_hz,
                                                       double doppler_margin_hz) const {
    // Global r2c bin grid: bin j centred at j*Fs/(2N) over [0, Fs/2), natural order.
    std::vector<ChannelBand> chans(_N);
    const double bin_w = _sample_rate / _fft_len;
    for (int j = 0; j < _N; ++j)
        chans[j] = {(freq_id_t)j, j * bin_w, bin_w};
    SignalDescriptor local = _sig;
    local.carrier_hz = _f_offset;
    const double max_dop = std::abs(doppler_hz) + doppler_margin_hz;
    const auto ids = covering_channels(chans, local, max_dop);
    return std::vector<int>(ids.begin(), ids.end());
}

std::vector<std::vector<cf>> ChannelizedReplicaBank::channels(int p, long long window_start_sample,
                                                              double code_phase_chips,
                                                              double doppler_hz, int n_hops) {
    const int warm = _num_taps - 1; // prime the continuous polyphase delay line
    const int total = n_hops + warm;
    const long long start = window_start_sample - (long long)warm * _fft_len;

    _replica_hist.assign((size_t)_fft_len * _num_taps, 0.0f);
    std::vector<float> block(_fft_len);
    std::vector<std::vector<cf>> chan(_N, std::vector<cf>(n_hops));

    // Code-Doppler feed-forward: the spreading code clock scales with the carrier
    // Doppler, chip_rate*(1 + f_d/f_RF) (~1.8 chip/s at L1). The code phase is read
    // at the ABSOLUTE sample index n, so without this term a seed referenced to
    // sample 0 drifts by code_Doppler * (seed age) -- a few chips for a fast sat,
    // which is what keeps high-Doppler sats from locking. Same sign as the carrier
    // (both track the same line-of-sight velocity); GnssChannelizedSearch applies
    // the matching term when referencing its measured cp back to sample 0.
    const double chip_per_sample =
        _sig.chip_rate_hz / _sample_rate * (1.0 + code_doppler_sign * doppler_hz / _sig.carrier_hz);
    const double wcarrier = 2.0 * M_PI * (_f_offset + doppler_hz) / _sample_rate;

    for (int h = 0; h < total; ++h) {
        for (int i = 0; i < _fft_len; ++i) {
            const long long n = start + (long long)h * _fft_len + i;
            if (n < 0) {
                block[i] = 0.0f; // pre-stream: matches F-engine zero init
                continue;
            }
            const int8_t c = code_chip(p, code_phase_chips + (double)n * chip_per_sample);
            // Real passband replica code*cos(carrier); the r2c bank keeps the
            // positive-frequency half (the +carrier image), matching the data.
            block[i] = c * (float)std::cos(wcarrier * (double)n);
        }
        dsp::pfb_push(_replica_hist.data(), block.data(), _fft_len, _num_taps);
        dsp::pfb_fold(_replica_hist.data(), _proto.data(), _fft_len, _num_taps, _fold);
        fftwf_execute(_p_fwd); // _fold (real) -> _spec (r2c, _N+1 bins)
        if (h < warm)
            continue;
        const int m = h - warm;
        // Natural order, dropping Nyquist: channel j = positive-freq bin j.
        auto* spec = reinterpret_cast<cf*>(_spec);
        for (int j = 0; j < _N; ++j)
            chan[j][m] = spec[j];
    }
    return chan;
}

} // namespace gnss
