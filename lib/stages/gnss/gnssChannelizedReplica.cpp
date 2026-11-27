#include "gnssChannelizedReplica.hpp"

#include "beidouB1CCode.hpp"
#include "beidouB1ICode.hpp"
#include "beidouB2aCode.hpp"
#include "beidouB2bCode.hpp"
#include "beidouB3ICode.hpp"
#include "fftwPlannerLock.hpp" // for fftw_planner_mutex
#include "galileoE1Code.hpp"
#include "galileoE5aCode.hpp"
#include "galileoE5bCode.hpp"
#include "galileoE6Code.hpp"
#include "glonassCACode.hpp"
#include "glonassL2OCCode.hpp"
#include "glonassL3OCCode.hpp"
#include "gnssBandPlan.hpp" // for ChannelBand, covering_channels
#include "gpsCACode.hpp"    // for generate_ca_code
#include "gpsL1CCode.hpp"   // for generate_l1cp_code
#include "gpsL2CCode.hpp"   // for generate_l2cm_code, generate_l2cl_code
#include "gpsL5Code.hpp"    // for generate_l5i_code, generate_l5q_code
#include "guowang1202Code.hpp"

#include <algorithm> // for max
#include <cassert>   // for assert (FDMA filter/PRN pairing)
#include <cmath>     // for cos, floor, llround, M_PI
#include <mutex>     // for lock_guard
#include <numeric>   // for gcd
#include <stdexcept> // for runtime_error
#include <string>    // for string

namespace gnss {

using cf = std::complex<float>;

namespace {
// Primary code tiled by a per-PRN secondary code (E5a-Q CS100, B2a-P Weil-1021), with
// overlay_sign()'s sign convention (chip * secondary[period]). A code phase in the tiled space
// is primary cp + primary_length * (period index mod overlay_length).
template<typename Prim, typename Sec>
std::vector<int8_t> bake_secondary(const Prim& a, const Sec& s) {
    std::vector<int8_t> tiled;
    tiled.reserve(a.size() * s.size());
    for (size_t k = 0; k < s.size(); ++k)
        for (size_t i = 0; i < a.size(); ++i)
            tiled.push_back((int8_t)(a[i] * s[k]));
    return tiled;
}

// The primary spreading code for a signal name, at its component chip rate (L2C CM is 10230
// chips at 511.5 kcps). The constructor expands it onto the combined stream for TDM and BOC
// signals. Shared secondary overlays are applied per period by overlay_sign(); the *_NH and
// *_CS signals return the code with the overlay already tiled in.
std::vector<int8_t> signal_code(const std::string& name, int prn) {
    if (name == "GPS_L1C_P") {
        auto a = gps::generate_l1cp_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GPS_L1C_D") {
        auto a = gps::generate_l1cd_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GPS_L5_I") {
        auto a = gps::generate_l5i_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GPS_L5_Q" || name == "QZSS_L5_Q") { // QZSS: PRN 193..202 of the same table
        auto a = gps::generate_l5q_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GPS_L5_Q_NH" || name == "QZSS_L5_Q_NH") {
        // Q5 primary tiled by the NH20 overlay, with overlay_sign()'s sign convention: a code
        // phase in this 204600-chip space is primary cp + 10230 * (period index mod 20).
        auto a = gps::generate_l5q_code(prn);
        std::vector<int8_t> tiled;
        tiled.reserve((size_t)gps::L5_CODE_LENGTH * gps::L5_NH20_LENGTH);
        for (int k = 0; k < gps::L5_NH20_LENGTH; ++k)
            for (int i = 0; i < gps::L5_CODE_LENGTH; ++i)
                tiled.push_back((int8_t)(a[(size_t)i] * gps::L5_NH20[(size_t)k]));
        return tiled;
    }
    if (name == "GPS_L5_I_NH") {
        // I5 primary tiled by the NH10 overlay, as GPS_L5_Q_NH tiles NH20: a code phase in this
        // 102300-chip space is primary cp + 10230 * (period index mod 10).
        auto a = gps::generate_l5i_code(prn);
        std::vector<int8_t> tiled;
        tiled.reserve((size_t)gps::L5_CODE_LENGTH * gps::L5_NH10_LENGTH);
        for (int k = 0; k < gps::L5_NH10_LENGTH; ++k)
            for (int i = 0; i < gps::L5_CODE_LENGTH; ++i)
                tiled.push_back((int8_t)(a[(size_t)i] * gps::L5_NH10[(size_t)k]));
        return tiled;
    }
    if (name == "SBAS_L5_I") {
        auto a = gps::generate_l5i_code(prn); // PRN 120..158: the SBAS rows of the L5 table
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "SBAS_L5_I_MC") {
        // I5 primary tiled by the 2-chip Manchester pair (+1, -1), as GPS_L5_Q_NH tiles NH20:
        // a code phase in this 20460-chip space is primary cp + 10230 * (period index mod 2).
        auto a = gps::generate_l5i_code(prn);
        std::vector<int8_t> tiled;
        tiled.reserve((size_t)gps::L5_CODE_LENGTH * 2);
        for (int k = 0; k < 2; ++k)
            for (int i = 0; i < gps::L5_CODE_LENGTH; ++i)
                tiled.push_back((int8_t)(a[(size_t)i] * (k == 0 ? 1 : -1)));
        return tiled;
    }
    if (name == "GAL_E1C") {
        auto a = galileo::generate_e1c_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GAL_E1C_CS") // the shared CS25 baked in, as the *_CS codes are per PRN
        return bake_secondary(galileo::generate_e1c_code(prn), galileo::E1C_CS25);
    if (name == "GAL_E1B") {
        auto a = galileo::generate_e1b_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "BDS_B1C_P") {
        auto a = beidou::generate_b1cp_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "BDS_B1C_D") {
        auto a = beidou::generate_b1cd_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "BDS_B1I") {
        auto a = beidou::generate_b1i_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "BDS_B1I_NH") // the D1 NH20 baked in, as BDS_B3I_NH below
        return bake_secondary(beidou::generate_b1i_code(prn), beidou::B1I_NH20);
    if (name == "GW_1202")
        return guowang::generate_gw1202_frame(prn);
    if (name == "BDS_B2I") {
        // B2I uses B1I's ranging code: the ICD defines one generator for C_B1I and C_B2I.
        // Only the carrier and the broadcasting satellites differ.
        auto a = beidou::generate_b1i_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "BDS_B3I") {
        auto a = beidou::generate_b3i_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "BDS_B3I_NH") {
        // B3I primary tiled by the D1 NH20, as GPS_L5_Q_NH above: a code phase in this
        // 204600-chip space is primary cp + 10230 * (period index mod 20).
        auto a = beidou::generate_b3i_code(prn);
        std::vector<int8_t> tiled;
        tiled.reserve((size_t)beidou::B3I_CODE_LENGTH * beidou::B3I_NH_LENGTH);
        for (int k = 0; k < beidou::B3I_NH_LENGTH; ++k)
            for (int i = 0; i < beidou::B3I_CODE_LENGTH; ++i)
                tiled.push_back((int8_t)(a[(size_t)i] * beidou::B3I_NH20[(size_t)k]));
        return tiled;
    }
    if (name == "GLO_L2OF") {
        // No PRN: FDMA satellites share one code and are separated by carrier, which is
        // set_prn_freq_offsets()'s job.
        (void)prn;
        const auto& a = glonass::ca_code();
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GLO_L2OF_MC") {
        // The shared C/A code tiled by the Manchester halves of a 20 ms bit (ten +1 periods,
        // ten -1): a code phase in this 10220-chip space is cp + 511 * (period index mod 20).
        (void)prn;
        const auto& a = glonass::ca_code();
        std::vector<int8_t> tiled;
        tiled.reserve(a.size() * 20);
        for (int k = 0; k < 20; ++k)
            for (int8_t c : a)
                tiled.push_back((int8_t)(c * (k < 10 ? 1 : -1)));
        return tiled;
    }
    if (name == "GLO_L2OC_P")
        return glonass::generate_l2ocp_code(prn); // already a vector (zeros = TDM'd data slots)
    // L2OC-p with the shared OC2 baked in: 50 periods of 40920 chips, 1 s (a code phase in this
    // space is primary cp + 40920 * (period index mod 50)).
    if (name == "GLO_L2OC_P_OC")
        return bake_secondary(glonass::generate_l2ocp_code(prn), glonass::L2OC_OC2);
    if (name == "GLO_L3OC_P") {
        auto a = glonass::generate_l3ocp_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GLO_L3OC_P_NH") {
        // L3OC-p tiled by the shared NH10, as GPS_L5_Q_NH and BDS_B3I_NH above: a code phase in
        // this 102300-chip space is primary cp + 10230 * (period index mod 10).
        auto a = glonass::generate_l3ocp_code(prn);
        std::vector<int8_t> tiled;
        tiled.reserve((size_t)glonass::L3OC_CODE_LENGTH * glonass::L3OC_NH_LENGTH);
        for (int k = 0; k < glonass::L3OC_NH_LENGTH; ++k)
            for (int i = 0; i < glonass::L3OC_CODE_LENGTH; ++i)
                tiled.push_back((int8_t)(a[(size_t)i] * glonass::L3OC_NH10[(size_t)k]));
        return tiled;
    }
    if (name == "GLO_L3OC_D") {
        auto a = glonass::generate_l3ocd_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GAL_E6_C") {
        auto a = galileo::generate_e6c_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GAL_E6_B") {
        auto a = galileo::generate_e6b_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GAL_E6_C_CS") // per-PRN CS100 baked in, exactly as GAL_E5A_Q_CS below
        return bake_secondary(galileo::generate_e6c_code(prn),
                              galileo::generate_e6c_secondary(prn));
    if (name == "GAL_E5A_Q") {
        auto a = galileo::generate_e5aq_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GAL_E5A_Q_CS") // per-PRN CS100 baked in (see bake_secondary above)
        return bake_secondary(galileo::generate_e5aq_code(prn), galileo::e5aq_secondary(prn));
    if (name == "GAL_E5A_I") {
        auto a = galileo::generate_e5ai_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GAL_E5A_I_CS") // the shared CS20 baked in: one F/NAV symbol
        return bake_secondary(galileo::generate_e5ai_code(prn), galileo::e5ai_secondary());
    if (name == "BDS_B2A_P") {
        auto a = beidou::generate_b2ap_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "BDS_B2A_P_CS") // per-PRN Weil-1021 secondary baked in
        return bake_secondary(beidou::generate_b2ap_code(prn), beidou::b2ap_secondary(prn));
    if (name == "BDS_B2A_D") {
        auto a = beidou::generate_b2ad_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "BDS_B2A_D_CS") // the shared 5-chip secondary baked in: one B-CNAV2 symbol
        return bake_secondary(beidou::generate_b2ad_code(prn), beidou::b2ad_secondary());
    if (name == "BDS_B2B_I") {
        auto a = beidou::generate_b2bi_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GAL_E5B_Q") {
        auto a = galileo::generate_e5bq_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GAL_E5B_Q_CS") // per-PRN CS100 baked in, exactly as GAL_E5A_Q_CS above
        return bake_secondary(galileo::generate_e5bq_code(prn), galileo::e5bq_secondary(prn));
    if (name == "GAL_E5B_I") {
        auto a = galileo::generate_e5bi_code(prn);
        return std::vector<int8_t>(a.begin(), a.end());
    }
    if (name == "GAL_E5B_I_CS") // the shared CS4 baked in: one I/NAV symbol
        return bake_secondary(galileo::generate_e5bi_code(prn), galileo::e5bi_secondary());
    if (name == "GPS_L2C_CM" || name == "QZSS_L2C_CM") // QZSS: PRN 193..202 of the same tables
        return gps::generate_l2cm_code(prn);
    if (name == "GPS_L2C_CL" || name == "QZSS_L2C_CL")
        return gps::generate_l2cl_code(prn);
    if (name == "GPS_L1CA_BIT") // twenty identical periods, one 20 ms LNAV bit
        return bake_secondary(gps::generate_ca_code(prn), std::vector<int8_t>(20, 1));
    if (name == "SBAS_L1_SYM") // two identical periods, one 2 ms SBAS symbol
        return bake_secondary(gps::generate_ca_code(prn), std::vector<int8_t>(2, 1));
    auto a = gps::generate_ca_code(prn); // default: GPS_L1CA and SBAS_L1 (PRN 120..158)
    return std::vector<int8_t>(a.begin(), a.end());
}
} // namespace

std::vector<int8_t> prn_secondary(const std::string& name, int prn) {
    if (name == "GAL_E5A_Q") {
        auto s = galileo::e5aq_secondary(prn);
        return std::vector<int8_t>(s.begin(), s.end());
    }
    if (name == "GAL_E5B_Q") {
        auto s = galileo::e5bq_secondary(prn);
        return std::vector<int8_t>(s.begin(), s.end());
    }
    if (name == "GAL_E6_C") {
        auto s = galileo::generate_e6c_secondary(prn);
        return std::vector<int8_t>(s.begin(), s.end());
    }
    if (name == "BDS_B2A_P") {
        auto s = beidou::b2ap_secondary(prn);
        return std::vector<int8_t>(s.begin(), s.end());
    }
    return {};
}

ChannelizedReplicaBank::ChannelizedReplicaBank(const SignalDescriptor& sig, double sample_rate,
                                               double f_offset, int spectrum_length, int num_taps,
                                               dsp::Window window, const std::vector<int>& prns) :
    ChannelizedReplicaBank(sig, Channelization{sample_rate, spectrum_length, num_taps, window},
                           f_offset, prns) {}

ChannelizedReplicaBank::ChannelizedReplicaBank(const SignalDescriptor& sig,
                                               const Channelization& chz, double f_offset,
                                               const std::vector<int>& prns) :
    _sig(sig), _chz(chz), _sample_rate(chz.sample_rate), _f_offset(f_offset),
    _N(chz.spectrum_length), _fft_len(chz.fft_len()), _num_taps(chz.num_taps), _fold(nullptr),
    _spec(nullptr), _p_fwd(nullptr) {

    _proto = dsp::pfb_prototype(_fft_len, _num_taps, chz.window);

    const long code_samples = (long)std::llround(_sig.code_period_s * _sample_rate);
    _repl_period_hops =
        (int)(code_samples / std::max<long>(1, std::gcd((long)_fft_len, code_samples)));

    // Time multiplexing (L2C CM/CL): the transmitted component runs at twice its chip rate,
    // interleaved with a sibling. Model the combined stream, with the component at its
    // tdm_phase parity and zeros at the sibling's, so the replica has the real signal's
    // spectrum; the bare 511.5 kcps component would lose ~9-12 dB.
    // BOC(m,m) is expanded the same way: each chip becomes 2m half-cycles of alternating sign
    // (sine phasing, + first) at 2m times the chip rate. The subcarrier then lives in the code
    // table, which the per-chip hop-rate path and the GPU kernel need, since they assume
    // constant chips. Code phases stay in component chips (cp0 = comb_mult * cp internally).
    // TDM and BOC are mutually exclusive.
    if (_sig.time_multiplexed && _sig.mod == Modulation::BOC)
        throw std::invalid_argument("ChannelizedReplicaBank: TDM+BOC not supported");
    if (_sig.mod == Modulation::BOC && _sig.boc_m != _sig.boc_n)
        throw std::invalid_argument("ChannelizedReplicaBank: only BOC(m,m) supported (got "
                                    + std::string(_sig.name) + ")");
    _comb_mult = _sig.time_multiplexed ? 2 : (_sig.mod == Modulation::BOC ? 2 * _sig.boc_m : 1);
    _eff_chip_rate = _sig.chip_rate_hz * _comb_mult;
    _eff_code_length = _sig.code_length * _comb_mult;
    _prns = prns;
    _full_code.resize(prns.size());
    for (size_t p = 0; p < prns.size(); ++p)
        _full_code[p] = build_full_code(prns[p]);

    // Secondary (Neuman-Hofman) overlay: one +-1 chip per primary period (NH10 on L5 I5, NH20
    // on L5 Q5), applied per period in channels()/hoprate_stream so a coherent integration over
    // several periods stays aligned across the primary rollover. Empty for signals without a
    // secondary code. Only overlays shared by every PRN are listed here; per-PRN overlays are
    // tiled into the code (the *_CS signals).
    if (_sig.secondary_length > 0) {
        const std::string name(_sig.name);
        if (name == "GPS_L5_Q" || name == "QZSS_L5_Q")
            _secondary.assign(gps::L5_NH20.begin(), gps::L5_NH20.end());
        else if (name == "SBAS_L5_I")
            _secondary.assign({1, -1}); // the SBAS bi-binary (Manchester) pair, 2 ms
        else if (name == "GPS_L5_I")
            _secondary.assign(gps::L5_NH10.begin(), gps::L5_NH10.end());
        else if (name == "GAL_E1C")
            _secondary.assign(galileo::E1C_CS25.begin(), galileo::E1C_CS25.end());
        else if (name == "GAL_E5A_I") {
            const auto o = galileo::e5ai_secondary(); // shared CS20 (20 ms = one F/NAV symbol)
            _secondary.assign(o.begin(), o.end());
        } else if (name == "BDS_B2A_D") {
            const auto o = beidou::b2ad_secondary(); // shared 5-chip (5 ms = one B-CNAV2 symbol)
            _secondary.assign(o.begin(), o.end());
        } else if (name == "GAL_E5B_I") {
            const auto o = galileo::e5bi_secondary(); // shared CS4 (4 ms = one I/NAV symbol)
            _secondary.assign(o.begin(), o.end());
        } else if (name == "BDS_B3I") // shared NH20 spreading D1 (20 ms = one NAV symbol)
            _secondary.assign(beidou::B3I_NH20.begin(), beidou::B3I_NH20.end());
        else if (name == "BDS_B1I") // same D1 NH20 one band up
            _secondary.assign(beidou::B1I_NH20.begin(), beidou::B1I_NH20.end());
        else if (name == "GLO_L3OC_P") // shared NH10 on the pilot (10 ms); tiled in GLO_L3OC_P_NH
            _secondary.assign(glonass::L3OC_NH10.begin(), glonass::L3OC_NH10.end());
        _secondary_length = (int)_secondary.size();
    }

    std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
    _spec = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (_N + 1));
    if (_chz.complex_input) {
        _p_fwd = fftwf_plan_dft_1d(_fft_len, _spec, _spec, FFTW_FORWARD, FFTW_ESTIMATE);
    } else {
        _fold = (float*)fftwf_malloc(sizeof(float) * _fft_len);
        _p_fwd = fftwf_plan_dft_r2c_1d(_fft_len, _fold, _spec, FFTW_ESTIMATE);
    }
}

std::vector<int8_t> ChannelizedReplicaBank::build_full_code(int prn) const {
    auto comp = signal_code(_sig.name, prn); // bare component code (+-1), code_length
    if (_comb_mult == 1)
        return comp;
    if (_sig.time_multiplexed) {
        std::vector<int8_t> comb((size_t)_eff_code_length, 0); // zero-stuffed sibling slots
        for (long k = 0; k < _sig.code_length; ++k)
            comb[(size_t)(_comb_mult * k + _sig.tdm_phase)] = comp[(size_t)k];
        return comb;
    }
    // BOC(m,m): alternating-sign half-cycles, sine convention (+ first)
    std::vector<int8_t> comb((size_t)_eff_code_length);
    for (long k = 0; k < _sig.code_length; ++k)
        for (int j = 0; j < _comb_mult; ++j)
            comb[(size_t)(_comb_mult * k + j)] =
                (int8_t)((j % 2) ? -comp[(size_t)k] : comp[(size_t)k]);
    return comb;
}

bool ChannelizedReplicaBank::set_prn(int p, int prn) {
    if (p < 0 || p >= (int)_prns.size())
        return false;
    if (_prns[(size_t)p] == prn)
        return true; // already there; a no-op swap must not cost the caller a state reset
    // FDMA is refused: the satellite identity is in the carrier offset _prn_df[p], and this
    // function does not know the new satellite's. Keeping the old offset or zeroing it would
    // both despread at the wrong frequency without any error.
    if (!_prn_df.empty()
        && std::any_of(_prn_df.begin(), _prn_df.end(), [](double d) { return d != 0.0; }))
        return false;
    std::vector<int8_t> code;
    try {
        code = build_full_code(prn);
    } catch (const std::exception&) {
        return false; // no code for this PRN on this signal -- refuse rather than despread zeros
    }
    // A table of the wrong length (a short component from signal_code) would let the gather
    // read past the end of the slot.
    if ((long)code.size() != _eff_code_length)
        return false;
    _full_code[(size_t)p] = std::move(code);
    _prns[(size_t)p] = prn;
    return true;
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

double ChannelizedReplicaBank::window_advance_chips(long long window_start_sample,
                                                    double doppler_hz) const {
    // cps must be the same double the generators compute; one ulp of difference is n0 * 1 ulp
    // ~ 1e-3 chips at n0 ~ 1.9e15. The product is in long double: n0 is exact in a 64-bit
    // mantissa and the reduced result is good to ~3e-7 chips, against ~0.5 chips in double.
    const double cps =
        _eff_chip_rate / _sample_rate * (1.0 + code_doppler_sign * doppler_hz / _sig.carrier_hz);
    const long double n0 = (long double)(window_start_sample + _fft_len - 1); // hoprate_stream's
    const long double L = (long double)_eff_code_length;                      // per-hop reference
    const long double adv = n0 * (long double)cps;
    long double r = adv - std::floor(adv / L) * L;
    if (r < 0.0L)
        r += L;
    return (double)r;
}

double ChannelizedReplicaBank::phase_from_arg(double arg, long long window_start_sample,
                                              double doppler_hz) const {
    const double L = (double)_eff_code_length;
    double ph = std::fmod(
        (double)_comb_mult * arg + window_advance_chips(window_start_sample, doppler_hz), L);
    return (ph < 0.0) ? ph + L : ph;
}

double ChannelizedReplicaBank::arg_from_phase(double phase, long long window_start_sample,
                                              double doppler_hz) const {
    const double L = (double)_eff_code_length;
    double ph = std::fmod(phase - window_advance_chips(window_start_sample, doppler_hz), L);
    if (ph < 0.0)
        ph += L;
    return ph / (double)_comb_mult;
}

void ChannelizedReplicaBank::set_prn_freq_offsets(const std::vector<double>& df_hz) {
    _prn_df = df_hz;
}

int ChannelizedReplicaBank::prn_freq_offsets_set() const {
    int n = 0;
    for (double d : _prn_df)
        if (d != 0.0)
            ++n;
    return n;
}

double ChannelizedReplicaBank::carrier_offset(int prn_index) const {
    if (prn_index < 0 || (size_t)prn_index >= _prn_df.size())
        return _f_offset;
    return _f_offset + _prn_df[(size_t)prn_index];
}

std::vector<int> ChannelizedReplicaBank::covering_bins(double doppler_hz, double doppler_margin_hz,
                                                       int prn_index) const {
    // Global bin grid, natural order: bin j centred at j*Fs/(2N) over [0, Fs/2) for real input.
    std::vector<ChannelBand> chans(_N);
    const double bin_w = _sample_rate / _fft_len;
    for (int j = 0; j < _N; ++j)
        chans[j] = {(freq_id_t)j, _chz.chan_freq(j), bin_w};
    SignalDescriptor local = _sig;
    local.carrier_hz = carrier_offset(prn_index);
    const double max_dop = std::abs(doppler_hz) + doppler_margin_hz;
    const auto ids = covering_channels(chans, local, max_dop);
    return std::vector<int>(ids.begin(), ids.end());
}

std::vector<int> ChannelizedReplicaBank::covering_bins_union(double doppler_hz,
                                                             double doppler_margin_hz) const {
    if (_prn_df.empty())
        return covering_bins(doppler_hz, doppler_margin_hz, -1);
    std::vector<int> all;
    for (size_t p = 0; p < _prn_df.size(); ++p) {
        const auto b = covering_bins(doppler_hz, doppler_margin_hz, (int)p);
        all.insert(all.end(), b.begin(), b.end());
    }
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    return all;
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
    // I/Q input uses a complex delay line instead.
    std::vector<cf> hist_iq, block_iq;
    if (_chz.complex_input) {
        hist_iq.assign((size_t)_fft_len * _num_taps, cf(0.0f, 0.0f));
        block_iq.resize((size_t)_fft_len);
    }
    std::vector<std::vector<cf>> chan(_N, std::vector<cf>(n_hops));

    // Code-Doppler feed-forward: the code clock scales with the carrier Doppler,
    // chip_rate*(1 + f_d/f_RF), with the same sign as the carrier. The code phase is read at
    // the absolute sample index, so without this term a phase referred to sample 0 drifts by
    // the code Doppler times its age. GnssChannelizedSearch applies the matching term when it
    // refers a measured cp back to sample 0.
    const double chip_per_sample =
        _eff_chip_rate / _sample_rate * (1.0 + code_doppler_sign * doppler_hz / _sig.carrier_hz);
    const double cp0 = (double)_comb_mult * code_phase_chips; // CM chips -> combined chips
    const double wcarrier = 2.0 * M_PI * (carrier_offset(p) + doppler_hz) / _sample_rate;

    // Carrier by phasor recurrence, cos(wcarrier*n) = Re(e^{i wcarrier n}): one complex
    // multiply per sample instead of a cos(), renormalised every hop so the modulus does not
    // drift. Agrees with the direct cos to ~1e-12.
    const std::complex<double> cstep(std::cos(wcarrier), std::sin(wcarrier));
    // The start phase is reduced in long double: wcarrier * start reaches ~7e15, where a
    // double's ULP is 1 rad (see hoprate_stream_into). This is the reference the hop-rate path
    // is checked against, so it must be at least as exact.
    const long double wcL = 2.0L * (long double)M_PI * (long double)(carrier_offset(p) + doppler_hz)
                            / (long double)_sample_rate;
    std::complex<double> cph = // phasor at n=start
        std::polar(1.0, (double)std::fmod(wcL * (long double)start, 2.0L * (long double)M_PI));

    for (int h = 0; h < total; ++h) {
        for (int i = 0; i < _fft_len; ++i) {
            const long long n = start + (long long)h * _fft_len + i;
            if (n >= 0) {
                // Long double, as for wcL. code_chip needs the phase mod the code length and
                // the overlay needs the period index, so both come from one evaluation.
                const long double phaseL =
                    (long double)cp0 + (long double)n * (long double)chip_per_sample;
                const long double LcL = (long double)_eff_code_length;
                const long long period = (long long)std::floor(phaseL / LcL);
                const double phase = (double)(phaseL - (long double)period * LcL);
                int8_t c = code_chip(p, phase);
                // A BOC subcarrier is already in the expanded code table. The secondary
                // overlay flips at primary-code rollovers and is applied only when an
                // alignment is set (nh_phase >= 0).
                if (_secondary_length > 0 && nh_phase >= 0)
                    c = (int8_t)(c * overlay_sign(period, nh_phase));
                // Real passband replica code*cos(carrier). It goes through the same r2c bank
                // as the data, so it lands in the same carrier image.
                block[i] = c * (float)cph.real();
                // I/Q input: the conjugate of code*e^{+i carrier}, since the bank's channels
                // are the conjugate of the channelizer's (see Channelization).
                if (_chz.complex_input)
                    block_iq[(size_t)i] = (float)c * (cf)std::conj(cph);
            } else {
                block[i] = 0.0f; // pre-stream: matches F-engine zero init
                if (_chz.complex_input)
                    block_iq[(size_t)i] = cf(0.0f, 0.0f);
            }
            cph *= cstep; // advance the carrier phasor (kept aligned with n)
        }
        cph /= std::abs(cph); // renormalise to unit modulus once per hop
        auto* spec = reinterpret_cast<cf*>(_spec);
        if (_chz.complex_input) {
            dsp::pfb_push(hist_iq.data(), block_iq.data(), _fft_len, _num_taps);
            dsp::pfb_fold(hist_iq.data(), _proto.data(), _fft_len, _num_taps, spec);
        } else {
            dsp::pfb_push(_replica_hist.data(), block.data(), _fft_len, _num_taps);
            dsp::pfb_fold(_replica_hist.data(), _proto.data(), _fft_len, _num_taps, _fold);
        }
        fftwf_execute(_p_fwd); // _fold (real) -> _spec (r2c, _N+1 bins), or _spec in place
        if (h < warm)
            continue;
        const int m = h - warm;
        // Natural order, dropping Nyquist: channel j = positive-freq bin j.
        for (int j = 0; j < _N; ++j)
            chan[j][m] = spec[j];
        if (_chz.time_order_fold)
            for (int j = 0; j < _N; ++j)
                chan[j][m] *= (cf)std::polar(1.0, -2.0 * M_PI * (double)j / (double)_fft_len);
    }
    return chan;
}

ChannelizedReplicaBank::HopRateFilter ChannelizedReplicaBank::hoprate_header(double doppler_hz,
                                                                             int prn_index) const {
    const double cps =
        _eff_chip_rate / _sample_rate * (1.0 + code_doppler_sign * doppler_hz / _sig.carrier_hz);
    const int Lf = _fft_len * _num_taps;

    HopRateFilter f;
    f.prn_index = prn_index;
    f.n_chips = (int)std::ceil((double)(Lf - 1) * cps) + 2; // chips the filter spans
    // Truncation to max_chips (0 = the full span). Synthesis cost is linear in the gather
    // depth, but the span (Lf = num_taps * fft_len = 65536 samples, 210 chips at CHORD's
    // 312.8 samples/chip) is the channel response, so truncating changes the replica too.
    if (_max_chips > 0 && f.n_chips > _max_chips) {
        if (_chips_centered) {
            // Keep the central max_chips: the prototype peak sits mid-span, and a one-sided
            // cap shorter than about half the span cuts through it.
            f.d_first = (f.n_chips - _max_chips) / 2;
            f.n_chips = f.d_first + _max_chips;
        } else {
            f.n_chips = _max_chips;
        }
    }
    f.wc_built = 2.0 * M_PI * (carrier_offset(prn_index) + doppler_hz) / _sample_rate;
    return f;
}

namespace {
// One channel's filter taps for both carrier images (A: +carrier, B: -carrier), the prototype
// modulated to the channel-carrier offset: emit(k, ea, eb) for k = 0 .. Lf-1. The carrier
// phasors come by recurrence, one complex multiply per tap instead of an exp, re-anchored
// exactly every 1024 taps so the rounding cannot accumulate. Both table builders run this one
// loop, so their prefix sums agree.
template<class Emit>
void hoprate_channel_taps(const float* proto, int Lf, double off, double wc, Emit&& emit) {
    const std::complex<double> I(0.0, 1.0);
    const std::complex<double> za = std::exp(-I * (off + wc));
    const std::complex<double> zb = std::exp(-I * (off - wc));
    std::complex<double> pa, pb;
    for (int k = 0; k < Lf; ++k) {
        if ((k & 1023) == 0) {
            pa = std::exp(-I * (off + wc) * (double)k);
            pb = std::exp(-I * (off - wc) * (double)k);
        }
        const double pk = proto[k];
        const std::complex<double> ea = pk * pa;
        const std::complex<double> eb = pk * pb;
        pa *= za;
        pb *= zb;
        emit(k, ea, eb);
    }
}
} // namespace

void ChannelizedReplicaBank::image_weights(int chan, std::complex<double>& wa,
                                           std::complex<double>& wb) const {
    const std::complex<double> ramp =
        _chz.time_order_fold ? std::polar(1.0, -2.0 * M_PI * (double)chan / (double)_fft_len) : 1.0;
    // The conjugate of I/Q input has the -carrier image only, at twice a real carrier's half.
    wa = _chz.complex_input ? 0.0 : ramp;
    wb = _chz.complex_input ? 2.0 * ramp : ramp;
}

ChannelizedReplicaBank::HopRateFilter
ChannelizedReplicaBank::hoprate_filter(const std::vector<int>& want, double doppler_hz,
                                       int prn_index) const {
    HopRateFilter f = hoprate_header(doppler_hz, prn_index);
    const double wc = f.wc_built;
    const int Lf = _fft_len * _num_taps;
    const int nw = (int)want.size();
    f.PhiA.resize(nw);
    f.PhiB.resize(nw);
    // Per channel, the cumulative filter Phi[k] = sum_{j<k} filt[j] for both carrier images;
    // a chip's contribution to a hop is Phi[k_hi+1] - Phi[k_lo] over its integer tap range.
    for (int ci = 0; ci < nw; ++ci) {
        const double off = 2.0 * M_PI * (double)want[ci] / (double)_fft_len;
        auto& A = f.PhiA[ci];
        auto& B = f.PhiB[ci];
        A.assign(Lf + 1, 0.0);
        B.assign(Lf + 1, 0.0);
        // Accumulated exactly as hoprate_filter_f32 accumulates, so that its floats are these
        // entries rounded once: whether the compiler contracts the tap product into the sum
        // depends on the code around it, so the two builders keep the same shape.
        std::complex<double> sa = 0.0, sb = 0.0;
        hoprate_channel_taps(
            _proto.data(), Lf, off, wc,
            [&](int k, const std::complex<double>& ea, const std::complex<double>& eb) {
                sa = sa + ea;
                sb = sb + eb;
                A[k + 1] = sa;
                B[k + 1] = sb;
            });
        if (_chz.complex_input || _chz.time_order_fold) {
            std::complex<double> wa, wb;
            image_weights(want[ci], wa, wb);
            for (int k = 0; k <= Lf; ++k) {
                A[k] *= wa;
                B[k] *= wb;
            }
        }
    }
    return f;
}

ChannelizedReplicaBank::HopRateFilter
ChannelizedReplicaBank::hoprate_filter_f32(const std::vector<int>& want, double doppler_hz,
                                           int prn_index, float* phiA, float* phiB) const {
    HopRateFilter f = hoprate_header(doppler_hz, prn_index);
    const double wc = f.wc_built;
    const int Lf = _fft_len * _num_taps;
    for (size_t ci = 0; ci < want.size(); ++ci) {
        const double off = 2.0 * M_PI * (double)want[ci] / (double)_fft_len;
        float* A = phiA + 2 * ci * (size_t)(Lf + 1);
        float* B = phiB + 2 * ci * (size_t)(Lf + 1);
        A[0] = A[1] = B[0] = B[1] = 0.0f;
        // The running sums are the double prefix sums hoprate_filter stores, so each float
        // here is that table's entry rounded once.
        std::complex<double> sa = 0.0, sb = 0.0;
        hoprate_channel_taps(
            _proto.data(), Lf, off, wc,
            [&](int k, const std::complex<double>& ea, const std::complex<double>& eb) {
                sa = sa + ea;
                sb = sb + eb;
                A[2 * (k + 1)] = (float)sa.real();
                A[2 * (k + 1) + 1] = (float)sa.imag();
                B[2 * (k + 1)] = (float)sb.real();
                B[2 * (k + 1) + 1] = (float)sb.imag();
            });
        if (_chz.complex_input || _chz.time_order_fold) {
            std::complex<double> wa, wb;
            image_weights((int)want[ci], wa, wb);
            for (int k = 0; k <= Lf; ++k) {
                const std::complex<double> a = wa * std::complex<double>(A[2 * k], A[2 * k + 1]);
                const std::complex<double> b = wb * std::complex<double>(B[2 * k], B[2 * k + 1]);
                A[2 * k] = (float)a.real();
                A[2 * k + 1] = (float)a.imag();
                B[2 * k] = (float)b.real();
                B[2 * k + 1] = (float)b.imag();
            }
        }
    }
    return f;
}

std::vector<std::vector<cf>>
ChannelizedReplicaBank::hoprate_stream(const HopRateFilter& f, int p, long long window_start_sample,
                                       double code_phase_chips, double doppler_hz, int n_hops,
                                       const std::function<float(long long)>& nav_bit,
                                       int nh_phase) const {
    std::vector<std::vector<cf>> chan_out;
    hoprate_stream_into(f, p, window_start_sample, code_phase_chips, doppler_hz, n_hops, nav_bit,
                        nh_phase, chan_out);
    return chan_out;
}

void ChannelizedReplicaBank::hoprate_stream_into(
    const HopRateFilter& f, int p, long long window_start_sample, double code_phase_chips,
    double doppler_hz, int n_hops, const std::function<float(long long)>& nav_bit, int nh_phase,
    std::vector<std::vector<cf>>& chan) const {
    // Per-hop carrier + code phase use the CURRENT Doppler (exact); the filter shape f is
    // built for a nearby Doppler (it barely moves). R_j[m] = 1/2 ( e^{+i wc n_m} Σ_d code·d̂·
    // (PhiA over chip d) + e^{-i wc n_m} ... PhiB ), d the chip relative to the window edge.
    const double cps =
        _eff_chip_rate / _sample_rate * (1.0 + code_doppler_sign * doppler_hz / _sig.carrier_hz);
    const double cp0 = (double)_comb_mult * code_phase_chips; // CM chips -> combined chips
    // FDMA: f carries the carrier of the PRN it was built for; streamed for another PRN it
    // would give a well-formed replica at the wrong carrier. -1 (CDMA) matches any p.
    assert(f.prn_index < 0 || f.prn_index == p);
    const double wc = 2.0 * M_PI * (carrier_offset(p) + doppler_hz) / _sample_rate;
    const int nw = (int)f.PhiA.size();

    // Reuse the caller's buffer; only grow when it does not already fit.
    if ((int)chan.size() != nw)
        chan.assign((size_t)nw, std::vector<cf>((size_t)n_hops));
    for (auto& row : chan)
        if ((int)row.size() != n_hops)
            row.assign((size_t)n_hops, cf(0.0f, 0.0f));
    const long long n0 = window_start_sample + _fft_len - 1;
    // The carrier's absolute-sample product is in long double. wc ~ 2.31 rad/sample and n0
    // reaches ~3e15 after ten days of F-engine uptime, so wc*n0 ~ 7e15 lies in [2^52, 2^53),
    // where a double's ULP is one radian; long double's is 7.4e-4 rad. The error grows with the
    // sample counter, so it does not show soon after an F-engine restart.
    const long double wcL = 2.0L * (long double)M_PI * (long double)(carrier_offset(p) + doppler_hz)
                            / (long double)_sample_rate;
    std::complex<double> pa =
        std::polar(1.0, (double)std::fmod(wcL * (long double)n0, 2.0L * (long double)M_PI));
    const std::complex<double> pstep =
        std::polar(1.0, std::fmod(wc * (double)_fft_len, 2.0 * M_PI));
    // Per-chip tap ranges and code values for one hop. The channel enters only through Phi, so
    // these are computed once per hop for all channels.
    std::vector<int> k_lo((size_t)f.n_chips), k_hi((size_t)f.n_chips);
    std::vector<double> cvs((size_t)f.n_chips);

    for (int m = 0; m < n_hops; ++m) {
        const long long n_m = n0 + (long long)m * _fft_len;
        const long long chip0 = hop_chip_taps(f, cp0, cps, n_m, k_lo, k_hi);
        const std::complex<double> pb = std::conj(pa);
        for (int d = 0; d < f.n_chips; ++d) {
            const long long want_chip = chip0 - (long long)d;
            double cv = 0.0;
            if (k_hi[(size_t)d] >= k_lo[(size_t)d] && d >= f.d_first) {
                cv = (double)code_chip(p, (double)want_chip);
                if (nav_bit)
                    cv *= nav_bit(want_chip);
                if (_secondary_length > 0 && nh_phase >= 0)
                    cv *= overlay_sign(
                        (long long)std::floor((double)want_chip / (double)_eff_code_length),
                        nh_phase);
            }
            cvs[(size_t)d] = cv;
        }

        for (int ci = 0; ci < nw; ++ci) {
            std::complex<double> sA(0.0, 0.0), sB(0.0, 0.0);
            for (int d = 0; d < f.n_chips; ++d) {
                if (k_hi[(size_t)d] < k_lo[(size_t)d])
                    continue;
                const double cv = cvs[(size_t)d];
                const size_t t1 = (size_t)k_hi[(size_t)d] + 1, t0 = (size_t)k_lo[(size_t)d];
                const std::complex<double> dA = f.PhiA[ci][t1] - f.PhiA[ci][t0];
                const std::complex<double> dB = f.PhiB[ci][t1] - f.PhiB[ci][t0];
                sA += cv * dA;
                sB += cv * dB;
            }
            chan[ci][m] = (cf)(0.5 * (pa * sA + pb * sB));
        }
        pa *= pstep;
        pa /= std::abs(pa);
    }
}

long long ChannelizedReplicaBank::hop_chip_taps(const HopRateFilter& f, double cp0, double cps,
                                                long long n_m, std::vector<int>& k_lo,
                                                std::vector<int>& k_hi) const {
    const int Lf = _fft_len * _num_taps;
    // The code's absolute-sample product is in long double: n_m*cps ~ 3.8e14 chips lies in
    // [2^48, 2^49), where a double's ULP is 0.0625 chips, ~30% of the tap spacing (cps). That
    // would hand taps to the neighbouring chip. hoprate_stream_into is the reference the GPU
    // replica is checked against.
    const long double CL = (long double)cp0 + (long double)n_m * (long double)cps;
    const long long chip0 = (long long)std::floor(CL);
    const double phi = (double)(CL - (long double)chip0);

    // The chip channels() assigns to tap k, at absolute sample n_m - k. The tap ranges below
    // must agree with it exactly, since channels() is what matches the F-engine, so it uses
    // the same precision as CL. It is evaluated in double relative to chip0 first: the two
    // differ by under 1e-4 chip (CL's long-double rounding at CHORD's ~4e14 chips), so away
    // from a chip boundary they floor alike, and only the rare value within CHIP_TIE of one
    // takes the long double (software arithmetic on aarch64, ten times the cost of the rest).
    constexpr double CHIP_TIE = 1e-3;
    auto chip_at = [&](int k) {
        const double x = phi - (double)k * cps;
        const double fx = std::floor(x);
        if (x - fx > CHIP_TIE && fx + 1.0 - x > CHIP_TIE)
            return chip0 + (long long)fx;
        return (long long)std::floor((long double)cp0
                                     + (long double)(n_m - (long long)k) * (long double)cps);
    };

    // Chip d owns the taps k with floor(C - k*cps) == chip0 - d, i.e.
    // k in ((phi+d-1)/cps, (phi+d)/cps]. In floating point the boundary tap lands on the
    // wrong side whenever (phi+d)/cps is an exact integer, and since cps is rational
    // (1023/320000 on CHORD L5) that happens systematically, ~8 times per replica period.
    // The partition stays exact (chip d+1's klo is chip d's khi+1), but the tap would take
    // its neighbour's code sign, an error of 2*proto[k]; so each bound is snapped against
    // chip_at().
    int klo = 0;
    for (int d = 0; d < f.n_chips; ++d) {
        int khi = (int)std::floor((phi + (double)d) / cps);
        const long long want_chip = chip0 - (long long)d;
        if (khi >= 0 && khi < Lf && chip_at(khi) < want_chip)
            --khi; // one tap too far: khi belongs to the next chip down
        else if (khi + 1 >= 0 && khi + 1 < Lf && chip_at(khi + 1) >= want_chip)
            ++khi; // one tap short: khi+1 is still this chip's
        if (khi > Lf - 1)
            khi = Lf - 1;
        k_lo[(size_t)d] = klo;
        k_hi[(size_t)d] = khi;
        klo = khi + 1;
    }
    return chip0;
}

void ChannelizedReplicaBank::hoprate_adjoint(const HopRateFilter& f, long long window_start_sample,
                                             double code_phase_chips, double doppler_hz, int n_hops,
                                             const std::vector<const cf*>& data, long long chip_lo,
                                             std::vector<std::complex<double>>& z) const {
    // The replica's hop m is R_m = sum_d code(chip0 - d) G_{m,d}, with G_{m,d} =
    // 1/2 (e^{+i wc n_m} (PhiA over chip d's taps) + e^{-i wc n_m} (PhiB ...)), as in
    // hoprate_stream_into; the despread sums data * conj(R). So chip c collects
    // data * conj(G_{m,d}) from every (m, d) with chip0(m) - d == c.
    assert(f.prn_index < 0);
    const double cps =
        _eff_chip_rate / _sample_rate * (1.0 + code_doppler_sign * doppler_hz / _sig.carrier_hz);
    const double cp0 = (double)_comb_mult * code_phase_chips;
    const double wc = 2.0 * M_PI * (carrier_offset(-1) + doppler_hz) / _sample_rate;
    const int nw = (int)f.PhiA.size();
    if ((int)data.size() != nw)
        throw std::invalid_argument("hoprate_adjoint: one data stream per filter channel");
    const long long n0 = window_start_sample + _fft_len - 1;
    const long double wcL = 2.0L * (long double)M_PI
                            * (long double)(carrier_offset(-1) + doppler_hz)
                            / (long double)_sample_rate;
    std::complex<double> pa =
        std::polar(1.0, (double)std::fmod(wcL * (long double)n0, 2.0L * (long double)M_PI));
    const std::complex<double> pstep =
        std::polar(1.0, std::fmod(wc * (double)_fft_len, 2.0 * M_PI));
    std::vector<int> k_lo((size_t)f.n_chips), k_hi((size_t)f.n_chips);
    const long long chip_hi = chip_lo + (long long)z.size();

    for (int m = 0; m < n_hops; ++m) {
        const long long n_m = n0 + (long long)m * _fft_len;
        const long long chip0 = hop_chip_taps(f, cp0, cps, n_m, k_lo, k_hi);
        const std::complex<double> pb = std::conj(pa);
        for (int d = f.d_first; d < f.n_chips; ++d) {
            const long long c = chip0 - (long long)d;
            if (c < chip_lo || c >= chip_hi || k_hi[(size_t)d] < k_lo[(size_t)d])
                continue;
            const size_t t1 = (size_t)k_hi[(size_t)d] + 1, t0 = (size_t)k_lo[(size_t)d];
            std::complex<double> acc(0.0, 0.0);
            for (int ci = 0; ci < nw; ++ci) {
                const std::complex<double> g = 0.5
                                               * (pa * (f.PhiA[ci][t1] - f.PhiA[ci][t0])
                                                  + pb * (f.PhiB[ci][t1] - f.PhiB[ci][t0]));
                acc += std::complex<double>(data[(size_t)ci][m]) * std::conj(g);
            }
            z[(size_t)(c - chip_lo)] += acc;
        }
        pa *= pstep;
        pa /= std::abs(pa);
    }
}

std::vector<std::vector<cf>> ChannelizedReplicaBank::channels_hoprate(
    int p, long long window_start_sample, double code_phase_chips, double doppler_hz, int n_hops,
    const std::vector<int>& want, const std::function<float(long long)>& nav_bit,
    int nh_phase) const {
    return hoprate_stream(hoprate_filter(want, doppler_hz, p), p, window_start_sample,
                          code_phase_chips, doppler_hz, n_hops, nav_bit, nh_phase);
}

} // namespace gnss
