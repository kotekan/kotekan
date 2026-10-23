/**
 * @file
 * @brief Host-side Psi table builders (build_psi_table, build_psi_split) for
 *        cudaGnssPsiSynth.cu's kernels: here, with the stages, because they read the
 *        ChannelizedReplicaBank (a stage-side class) and the CUDA library must not.
 */

#include "cudaGnssPsiSynth.hpp"
#include "gnssChannelizedReplica.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace gnss_cuda {

namespace {

using cd = std::complex<double>;

/// The per-chip kernel K_d(phi), d in [0, n_chips), at chip fraction @c phi from the double
/// PhiB prefix sum: fractional edges at x + 1/2 (frac), or the production kernel's snapped
/// cells floor(x) + 1 (see build_psi_table).
void psi_kernel(const std::vector<cd>& Phi, int Lf, double inv, int n_chips, bool frac, double phi,
                std::vector<cd>& K) {
    // Prefix sum at a fractional tap index, clamped to [0, Lf].
    auto phi_lin = [&](double x) -> cd {
        if (x <= 0.0)
            return cd(0.0, 0.0);
        if (x >= (double)Lf)
            return Phi[(size_t)Lf];
        const int i = (int)std::floor(x);
        const double fr = x - (double)i;
        return Phi[(size_t)i] + fr * (Phi[(size_t)i + 1] - Phi[(size_t)i]);
    };
    auto phi_snap = [&](double x) -> cd {
        int t = (int)std::floor(x) + 1;
        t = t < 0 ? 0 : (t > Lf ? Lf : t);
        return Phi[(size_t)t];
    };
    // The smooth edge sits at x + 0.5 (tap units): the rect model's exclusive edge
    // floor(x) + 1 averages to x + 0.5 over the sub-sample phase, so this is the
    // zero-mean-delay smoothing of it. psi_interp_proto.py's x + 1.0 is half a sample
    // (1.6e-3 chips, 4.7 cm) late against production, which the sky test resolves.
    K.resize((size_t)n_chips);
    cd prevv = frac ? phi_lin((phi - 1.0) * inv + 0.5) : phi_snap((phi - 1.0) * inv);
    for (int d = 0; d < n_chips; ++d) {
        const cd cur =
            frac ? phi_lin((phi + (double)d) * inv + 0.5) : phi_snap((phi + (double)d) * inv);
        K[(size_t)d] = cur - prevv;
        prevv = cur;
    }
}

/// Psi over one period of @c code (P chips, grid G): out[m*G + g] = sum_d code[(m-d) mod P]
/// K_d(g/G), d over [0, n_chips) for the plain table (out has G*P entries), or over
/// (m, n_chips) for the previous-period lookback table (prev: out has G*(n_chips-1) entries,
/// m <= n_chips-2). Accumulated in double, stored float2.
void psi_convolve(const std::vector<cd>& Phi, int Lf, double inv, int n_chips, bool frac,
                  const int8_t* code, int P, int G, bool prev, float2* out) {
    std::vector<cd> K;
    const int M = prev ? n_chips - 1 : P;
    std::vector<double> acc_r((size_t)M), acc_i((size_t)M);
    for (int g = 0; g < G; ++g) {
        psi_kernel(Phi, Lf, inv, n_chips, frac, (double)g / (double)G, K);
        std::fill(acc_r.begin(), acc_r.end(), 0.0);
        std::fill(acc_i.begin(), acc_i.end(), 0.0);
        for (int d = 0; d < n_chips; ++d) {
            const double kr = K[(size_t)d].real(), ki = K[(size_t)d].imag();
            int idx = (P - d % P) % P;                    // (0 - d) mod P, then walks with m
            const int m_end = prev ? (d < M ? d : M) : M; // prev: only m < d
            for (int mm = 0; mm < m_end; ++mm) {
                const double cv = (double)code[(size_t)idx];
                acc_r[(size_t)mm] += cv * kr;
                acc_i[(size_t)mm] += cv * ki;
                if (++idx == P)
                    idx = 0;
            }
        }
        for (int mm = 0; mm < M; ++mm)
            out[(size_t)mm * G + (size_t)g] =
                make_float2((float)acc_r[(size_t)mm], (float)acc_i[(size_t)mm]);
    }
}

/// The bank's filter at the full span, with cps and n_chips checked.
gnss::ChannelizedReplicaBank::HopRateFilter psi_filter(const gnss::ChannelizedReplicaBank& bank,
                                                       int prn_index, const std::vector<int>& chans,
                                                       double doppler_hz, double sample_rate,
                                                       double* inv_cps) {
    auto f = bank.hoprate_filter(chans, doppler_hz, prn_index);
    const int Lf = bank.fft_len() * bank.num_taps();
    const double cps = bank.eff_chip_rate() / sample_rate
                       * (1.0 + bank.code_doppler_sign * doppler_hz / bank.carrier_hz());
    const int n_full = (int)std::ceil((double)(Lf - 1) * cps) + 2;
    if (f.d_first != 0 || f.n_chips != n_full)
        throw std::runtime_error("build_psi_table: the bank truncates the PFB span (max_chips "
                                 "must be 0): a truncated gather is not interpolable");
    *inv_cps = 1.0 / cps;
    return f;
}

} // namespace

PsiTable build_psi_table(const gnss::ChannelizedReplicaBank& bank, int prn_index,
                         const std::vector<int>& chans, double doppler_hz, double sample_rate,
                         int G, bool frac) {
    const auto t0 = std::chrono::steady_clock::now();
    PsiTable T;
    double inv = 0.0;
    const auto f = psi_filter(bank, prn_index, chans, doppler_hz, sample_rate, &inv);
    const int Lf = bank.fft_len() * bank.num_taps();
    const int L = (int)bank.eff_code_length();
    const auto& code = bank.full_code(prn_index);
    const int nc = (int)chans.size();
    T.G = G;
    T.L = L;
    T.n_chips = f.n_chips;
    T.inv_cps = inv;
    T.wc = f.wc_built;
    T.theta.resize(nc);
    T.psi.assign((size_t)nc * G * L, make_float2(0.f, 0.f));
    for (int ci = 0; ci < nc; ++ci) {
        T.theta[(size_t)ci] = psi_theta(chans[(size_t)ci], bank.fft_len(), f.wc_built);
        psi_convolve(f.PhiB[(size_t)ci], Lf, inv, f.n_chips, frac, code.data(), L, G, false,
                     T.psi.data() + (size_t)ci * G * L);
    }
    T.build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return T;
}

PsiCodeSplit psi_split_code(const int8_t* code, long L, int P) {
    if (P <= 0 || L % P != 0)
        throw std::runtime_error("build_psi_split: the primary period does not divide the code");
    PsiCodeSplit cs;
    cs.P = P;
    cs.Ns = (int)(L / P);
    // Period signs against the first period: the first non-zero chip carries the sign (a
    // TDM stream has zeros), and every chip must agree, or the code is not a sign tiling.
    int i0 = 0;
    while (i0 < P && code[(size_t)i0] == 0)
        ++i0;
    if (i0 == P)
        throw std::runtime_error("build_psi_split: all-zero primary period");
    cs.sec.resize((size_t)cs.Ns);
    for (int q = 0; q < cs.Ns; ++q) {
        const int sg = (int)code[(size_t)q * P + i0] * (int)code[(size_t)i0];
        for (int i = 0; i < P; ++i)
            if ((int)code[(size_t)q * P + i] != sg * (int)code[(size_t)i])
                throw std::runtime_error("build_psi_split: period " + std::to_string(q)
                                         + " is not a sign of the first period");
        cs.sec[(size_t)q] = (int8_t)sg;
    }
    cs.prim.assign(code, code + P);
    return cs;
}

PsiSplit build_psi_split_from(const gnss::ChannelizedReplicaBank& bank, int prn_index,
                              const std::vector<int>& chans, double doppler_hz, double sample_rate,
                              int G, const PsiCodeSplit& cs, bool frac) {
    const auto t0 = std::chrono::steady_clock::now();
    PsiSplit S;
    double inv = 0.0;
    const auto f = psi_filter(bank, prn_index, chans, doppler_hz, sample_rate, &inv);
    const int Lf = bank.fft_len() * bank.num_taps();
    const int P = cs.P;
    if (P <= 0 || (int)cs.prim.size() != P || cs.Ns <= 0 || (int)cs.sec.size() != cs.Ns)
        throw std::runtime_error("build_psi_split: malformed code split");
    if (f.n_chips >= P)
        throw std::runtime_error("build_psi_split: the PFB span reaches back more than one "
                                 "primary period");
    const int nc = (int)chans.size();
    S.G = G;
    S.P = P;
    S.Ns = cs.Ns;
    S.sec = cs.sec;
    S.n_chips = f.n_chips;
    S.n_prev = G * (f.n_chips - 1);
    S.inv_cps = inv;
    S.wc = f.wc_built;
    S.theta.resize(nc);
    S.psi_plain.assign((size_t)nc * G * P, make_float2(0.f, 0.f));
    S.psi_prev.assign((size_t)nc * S.n_prev, make_float2(0.f, 0.f));
    for (int ci = 0; ci < nc; ++ci) {
        S.theta[(size_t)ci] = psi_theta(chans[(size_t)ci], bank.fft_len(), f.wc_built);
        psi_convolve(f.PhiB[(size_t)ci], Lf, inv, f.n_chips, frac, cs.prim.data(), P, G, false,
                     S.psi_plain.data() + (size_t)ci * G * P);
        psi_convolve(f.PhiB[(size_t)ci], Lf, inv, f.n_chips, frac, cs.prim.data(), P, G, true,
                     S.psi_prev.data() + (size_t)ci * S.n_prev);
    }
    S.build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return S;
}

PsiSplit build_psi_split(const gnss::ChannelizedReplicaBank& bank, int prn_index,
                         const std::vector<int>& chans, double doppler_hz, double sample_rate,
                         int G, int P, bool frac) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto& code = bank.full_code(prn_index);
    const PsiCodeSplit cs = psi_split_code(code.data(), bank.eff_code_length(), P);
    PsiSplit S = build_psi_split_from(bank, prn_index, chans, doppler_hz, sample_rate, G, cs, frac);
    S.build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return S;
}

} // namespace gnss_cuda
