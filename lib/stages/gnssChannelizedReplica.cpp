#include "gnssChannelizedReplica.hpp"

#include "fftwPlannerLock.hpp" // for fftw_planner_mutex
#include "gnssBandPlan.hpp"    // for ChannelBand, covering_channels
#include "gpsCACode.hpp"       // for generate_ca_code
#include "gpsL1CCode.hpp"      // for generate_l1cp_code
#include "gpsL2CCode.hpp"      // for generate_l2cm_code, generate_l2cl_code
#include "beidouB1CCode.hpp"
#include "galileoE1Code.hpp"
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
// Dispatch the primary spreading code by signal name. Returns the BARE component code
// at its component chip rate (e.g. L2C CM = 10230 chips @ 511.5 kcps). For a
// time-multiplexed signal the constructor then zero-stuffs this onto the COMBINED stream
// (CM/CL interleaved to 1.023 Mcps) at the tdm_phase parity -- the real signal's
// structure; despreading the bare 511.5 kcps code loses ~9-12 dB (gps_l2c_subband_validate.py).
// L2C CL is time-assisted (1.5 s period -- track-only, not blind-acquirable). L5
// (I5/Q5) is the clean, fully-supported L2/L5 target. This returns the BARE primary
// code; the NH secondary overlay (L5 NH10/NH20, one +-1 chip per 1 ms primary period)
// is applied on top in channels()/hoprate_stream via overlay_sign() -- a constant sign
// within one period (invisible to acquisition), but it unlocks multi-period coherent
// integration of the dataless Q5 pilot once the period alignment (nh_phase) is set.
std::vector<int8_t> signal_code(const std::string& name, int prn) {
    if (name == "GPS_L1C_P") {
        auto a = gps::generate_l1cp_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GPS_L5_I") {
        auto a = gps::generate_l5i_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GPS_L5_Q") {
        auto a = gps::generate_l5q_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GAL_E1C") {
        auto a = galileo::generate_e1c_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GAL_E1B") {
        auto a = galileo::generate_e1b_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "BDS_B1C_P") {
        auto a = beidou::generate_b1cp_code(prn);
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

    // Time-multiplexing (L2C CM/CL): the transmitted component sits at 2x its chip rate,
    // interleaved with a sibling. Model the COMBINED stream -- the component at its
    // tdm_phase parity, zeros where the sibling is -- so the channelized replica matches
    // the real signal's spectrum (NOT the bare 511.5 kcps component, which loses ~9-12 dB;
    // see python/scripts/gps_l2c_subband_validate.py). comb_mult=1 -> ordinary signal.
    // BOC(m,m) is modeled as an EXPANDED code, exactly like TDM: each chip becomes 2m
    // half-cycle slots of alternating sign (sine-BOC: + first) at 2m x the chip rate. This
    // replaces the old analytic per-sample sub-carrier branch so the sub-carrier lives IN
    // the code table -- which is what the per-chip hoprate path (and the GPU despread kernel
    // ported from it) require: they assume constant chips. Numerically identical to the
    // analytic branch (unit-tested); the cp API stays in COMPONENT chips throughout
    // (cp0 = comb_mult * cp internally). TDM and BOC are mutually exclusive.
    if (_sig.time_multiplexed && _sig.mod == Modulation::BOC)
        throw std::invalid_argument("ChannelizedReplicaBank: TDM+BOC not supported");
    if (_sig.mod == Modulation::BOC && _sig.boc_m != _sig.boc_n)
        throw std::invalid_argument("ChannelizedReplicaBank: only BOC(m,m) supported (got "
                                    + std::string(_sig.name) + ")");
    _comb_mult = _sig.time_multiplexed ? 2 : (_sig.mod == Modulation::BOC ? 2 * _sig.boc_m : 1);
    _eff_chip_rate = _sig.chip_rate_hz * _comb_mult;
    _eff_code_length = _sig.code_length * _comb_mult;
    _full_code.resize(prns.size());
    for (size_t p = 0; p < prns.size(); ++p) {
        auto comp = signal_code(_sig.name, prns[p]); // bare component code (+-1), code_length
        if (_comb_mult == 1) {
            _full_code[p] = std::move(comp);
        } else if (_sig.time_multiplexed) {
            std::vector<int8_t> comb((size_t)_eff_code_length, 0); // zero-stuffed sibling slots
            for (long k = 0; k < _sig.code_length; ++k)
                comb[(size_t)(_comb_mult * k + _sig.tdm_phase)] = comp[(size_t)k];
            _full_code[p] = std::move(comb);
        } else { // BOC(m,m): alternating-sign half-cycles, sine convention (+ first)
            std::vector<int8_t> comb((size_t)_eff_code_length);
            for (long k = 0; k < _sig.code_length; ++k)
                for (int j = 0; j < _comb_mult; ++j)
                    comb[(size_t)(_comb_mult * k + j)] =
                        (int8_t)((j % 2) ? -comp[(size_t)k] : comp[(size_t)k]);
            _full_code[p] = std::move(comb);
        }
    }

    // Secondary (Neuman-Hofman) overlay: one +-1 chip per primary period (NH10 on L5 I5,
    // NH20 on L5 Q5). Applied per period in channels()/hoprate_stream so a multi-period
    // coherent integration of the dataless Q5 pilot doesn't decorrelate at the 1 ms primary
    // rollover. Empty (no-op) for signals without a secondary code (L1 C/A, L2C).
    if (_sig.secondary_length > 0) {
        const std::string name(_sig.name);
        if (name == "GPS_L5_Q")
            _secondary.assign(gps::L5_NH20.begin(), gps::L5_NH20.end());
        else if (name == "GPS_L5_I")
            _secondary.assign(gps::L5_NH10.begin(), gps::L5_NH10.end());
        else if (name == "GAL_E1C")
            _secondary.assign(galileo::E1C_CS25.begin(), galileo::E1C_CS25.end());
        _secondary_length = (int)_secondary.size();
    }

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
    const long len = _eff_code_length; // combined-stream length (== code_length for comb_mult 1)
    long idx = (long)std::floor(chip_phase);
    idx %= len;
    if (idx < 0)
        idx += len;
    return _full_code[p][idx];
}

int ChannelizedReplicaBank::overlay_sign(long long period, int nh_phase) const {
    if (_secondary_length <= 0)
        return 1; // no secondary code -> composes as a no-op
    long long k = (period + nh_phase) % _secondary_length;
    if (k < 0)
        k += _secondary_length;
    return _secondary[(size_t)k];
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
                                                              double doppler_hz, int n_hops,
                                                              int nh_phase) {
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
        _eff_chip_rate / _sample_rate * (1.0 + code_doppler_sign * doppler_hz / _sig.carrier_hz);
    const double cp0 = (double)_comb_mult * code_phase_chips; // CM chips -> combined chips
    const double wcarrier = 2.0 * M_PI * (_f_offset + doppler_hz) / _sample_rate;

    // Carrier as a phasor recurrence: cos(wcarrier*n) = Re(e^{i wcarrier n}), advanced
    // by one complex multiply per sample instead of a transcendental cos() each --
    // the per-sample cos dominated replica generation (esp. CHORD's large fft_len).
    // Renormalised per hop so the unit-modulus phasor doesn't drift; matches the direct
    // cos to ~1e-12, and is in fact steadier at large n (no huge-argument range loss).
    const std::complex<double> cstep(std::cos(wcarrier), std::sin(wcarrier));
    std::complex<double> cph = std::polar(1.0, wcarrier * (double)start); // phasor at n=start

    for (int h = 0; h < total; ++h) {
        for (int i = 0; i < _fft_len; ++i) {
            const long long n = start + (long long)h * _fft_len + i;
            if (n >= 0) {
                const double phase = cp0 + (double)n * chip_per_sample;
                int8_t c = code_chip(p, phase);
                // BOC sub-carrier: baked into the EXPANDED code table at construction (each
                // chip = 2m alternating-sign half-cycles at 2m x the rate) -- shared by this
                // per-sample path, the per-chip hoprate path, AND the GPU despread kernel.
                // The old analytic per-sample branch here would now double-apply it.
                // Apply the secondary (NH) overlay per primary period when an alignment is set
                // (nh_phase >= 0): the primary code repeats every _eff_code_length chips, the
                // overlay flips at those rollovers, so a multi-period coherent despread stays
                // aligned. nh_phase < 0 -> overlay off (raw despread, the default).
                if (_secondary_length > 0 && nh_phase >= 0)
                    c = (int8_t)(c * overlay_sign((long long)std::floor(phase / (double)_eff_code_length),
                                                  nh_phase));
                // Real passband replica code*cos(carrier); the r2c bank keeps the
                // positive-frequency half (the +carrier image), matching the data.
                block[i] = c * (float)cph.real();
            } else {
                block[i] = 0.0f; // pre-stream: matches F-engine zero init
            }
            cph *= cstep; // advance the carrier phasor (kept aligned with n)
        }
        cph /= std::abs(cph); // renormalise to unit modulus once per hop
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

ChannelizedReplicaBank::HopRateFilter
ChannelizedReplicaBank::hoprate_filter(const std::vector<int>& want, double doppler_hz) const {
    const double cps =
        _eff_chip_rate / _sample_rate * (1.0 + code_doppler_sign * doppler_hz / _sig.carrier_hz);
    const double wc = 2.0 * M_PI * (_f_offset + doppler_hz) / _sample_rate;
    const int Lf = _fft_len * _num_taps;
    const std::complex<double> I(0.0, 1.0);

    HopRateFilter f;
    f.doppler_hz = doppler_hz;
    f.n_chips = (int)std::ceil((double)(Lf - 1) * cps) + 2; // chips the filter spans
    f.chans = want;
    const int nw = (int)want.size();
    f.PhiA.resize(nw);
    f.PhiB.resize(nw);
    // Per channel, the cumulative filter Phi[k] = Σ_{<k} filt[k] for both carrier images
    // (A: +carrier, B: -carrier); filt = prototype modulated to the channel-carrier offset.
    // A chip's hop contribution is Phi[k_hi+1]-Phi[k_lo] over its integer tap range -- exact.
    for (int ci = 0; ci < nw; ++ci) {
        const double off = 2.0 * M_PI * (double)want[ci] / (double)_fft_len;
        f.PhiA[ci].assign(Lf + 1, 0.0);
        f.PhiB[ci].assign(Lf + 1, 0.0);
        for (int k = 0; k < Lf; ++k) {
            const double pk = _proto[k];
            f.PhiA[ci][k + 1] = f.PhiA[ci][k] + pk * std::exp(-I * (off + wc) * (double)k);
            f.PhiB[ci][k + 1] = f.PhiB[ci][k] + pk * std::exp(-I * (off - wc) * (double)k);
        }
    }
    return f;
}

std::vector<std::vector<cf>>
ChannelizedReplicaBank::hoprate_stream(const HopRateFilter& f, int p, long long window_start_sample,
                                       double code_phase_chips, double doppler_hz, int n_hops,
                                       const std::function<float(long long)>& nav_bit,
                                       int nh_phase) const {
    // Per-hop carrier + code phase use the CURRENT Doppler (exact); the filter shape f is
    // built for a nearby Doppler (it barely moves). R_j[m] = 1/2 ( e^{+i wc n_m} Σ_d code·d̂·
    // (PhiA over chip d) + e^{-i wc n_m} ... PhiB ), d the chip relative to the window edge.
    const double cps =
        _eff_chip_rate / _sample_rate * (1.0 + code_doppler_sign * doppler_hz / _sig.carrier_hz);
    const double cp0 = (double)_comb_mult * code_phase_chips; // CM chips -> combined chips
    const double wc = 2.0 * M_PI * (_f_offset + doppler_hz) / _sample_rate;
    const int Lf = _fft_len * _num_taps;
    const int nw = (int)f.PhiA.size();

    std::vector<std::vector<cf>> chan(nw, std::vector<cf>(n_hops));
    const long long n0 = window_start_sample + _fft_len - 1;
    std::complex<double> pa = std::polar(1.0, std::fmod(wc * (double)n0, 2.0 * M_PI));
    const std::complex<double> pstep = std::polar(1.0, std::fmod(wc * (double)_fft_len, 2.0 * M_PI));
    for (int m = 0; m < n_hops; ++m) {
        const long long n_m = n0 + (long long)m * _fft_len;
        const double C = cp0 + (double)n_m * cps;
        const long long chip0 = (long long)std::floor(C);
        const double phi = C - (double)chip0;
        const std::complex<double> pb = std::conj(pa);
        for (int ci = 0; ci < nw; ++ci) {
            std::complex<double> sA(0.0, 0.0), sB(0.0, 0.0);
            for (int d = 0; d < f.n_chips; ++d) {
                int klo = (int)std::floor((phi + d - 1.0) / cps) + 1; // chip d's tap range
                int khi = (int)std::floor((phi + d) / cps);
                if (klo < 0)
                    klo = 0;
                if (khi > Lf - 1)
                    khi = Lf - 1;
                if (khi < klo)
                    continue;
                const long long cidx = chip0 - d;
                double cv = (double)code_chip(p, (double)cidx);
                if (nav_bit)
                    cv *= nav_bit(cidx);
                if (_secondary_length > 0 && nh_phase >= 0)
                    cv *= overlay_sign(
                        (long long)std::floor((double)cidx / (double)_eff_code_length), nh_phase);
                sA += cv * (f.PhiA[ci][khi + 1] - f.PhiA[ci][klo]);
                sB += cv * (f.PhiB[ci][khi + 1] - f.PhiB[ci][klo]);
            }
            chan[ci][m] = (cf)(0.5 * (pa * sA + pb * sB));
        }
        pa *= pstep;
        pa /= std::abs(pa);
    }
    return chan;
}

std::vector<std::vector<cf>>
ChannelizedReplicaBank::channels_hoprate(int p, long long window_start_sample,
                                         double code_phase_chips, double doppler_hz, int n_hops,
                                         const std::vector<int>& want,
                                         const std::function<float(long long)>& nav_bit,
                                         int nh_phase) const {
    return hoprate_stream(hoprate_filter(want, doppler_hz), p, window_start_sample,
                          code_phase_chips, doppler_hz, n_hops, nav_bit, nh_phase);
}

HopRateReplicaStream::HopRateReplicaStream(const ChannelizedReplicaBank& bank, int prn_index,
                                           std::vector<int> channels, double refresh_hz) :
    _bank(bank), _prn(prn_index), _chans(std::move(channels)), _refresh_hz(refresh_hz) {}

const std::vector<std::vector<cf>>&
HopRateReplicaStream::generate(long long window_start, double cp, double doppler_hz, int n_hops,
                               const std::function<float(long long)>& nav_bit) {
    if (!_built || std::fabs(doppler_hz - _filter.doppler_hz) > _refresh_hz) {
        _filter = _bank.hoprate_filter(_chans, doppler_hz); // slow part -- only on a real drift
        _built = true;
        ++_rebuilds;
    }
    _out = _bank.hoprate_stream(_filter, _prn, window_start, cp, doppler_hz, n_hops, nav_bit);
    return _out;
}

} // namespace gnss
