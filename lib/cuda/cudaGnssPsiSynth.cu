#include "cudaGnssPsiSynth.hpp"
#include "cudaGnssReplicaDevice.cuh" // for cmulf
#include "gnssChannelizedReplica.hpp"

#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>

// The prototype of docs/gnss_gpu_search.md's code-phase-table synthesis. Nothing here is shared
// with the production kernels: cudaGnssChordDespread.cu stays as it is, and this kernel is
// judged against it at the correlation level by cudaGnssPsiBench.cpp.

namespace {

using gnss_cuda::cmulf;

/// Table element load: float2, or __half2 widened to float2 (DespreadParams::phi_half).
__device__ inline float2 tab_ld(const float2* t, int i) {
    return t[i];
}
__device__ inline float2 tab_ld(const __half2* t, int i) {
    return __half22float2(t[i]);
}

/// Table value at absolute grid index @c a in [0, G*L): the plain table, or for a tiled code
/// the period-sign split sec[q] Psi_plain[m'] + (sec[q-1] - sec[q]) Psi_prev[m'] (see
/// PsiJob). @c Ns, @c GP, @c n_prev are block-uniform, so the branches do not diverge.
template<class TAB>
__device__ inline float2 psi_at(const TAB* __restrict__ tab, const TAB* __restrict__ prev,
                                const int8_t* __restrict__ sec, int Ns, int GP, int n_prev, int a) {
    if (Ns == 1)
        return tab_ld(tab, a);
    const int q = a / GP;
    const int mp = a - q * GP;
    const float sq = (float)sec[q];
    float2 v = tab_ld(tab, mp);
    v.x *= sq;
    v.y *= sq;
    if (mp < n_prev) {
        const float sp = (float)sec[q == 0 ? Ns - 1 : q - 1] - sq;
        const float2 pv = tab_ld(prev, mp);
        v.x = fmaf(sp, pv.x, v.x);
        v.y = fmaf(sp, pv.y, v.y);
    }
    return v;
}

__device__ inline double warp_sum(double v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

/// Cubic Lagrange weights at fraction f in [0, 1) for the stencil j = -1, 0, 1, 2.
__device__ inline void lagrange4(float f, float w[4]) {
    const float fm1 = f - 1.f, fm2 = f - 2.f, fp1 = f + 1.f;
    w[0] = -f * fm1 * fm2 * (1.f / 6.f);
    w[1] = fp1 * fm1 * fm2 * 0.5f;
    w[2] = -fp1 * f * fm2 * 0.5f;
    w[3] = fp1 * f * fm1 * (1.f / 6.f);
}

/// grid.x = job (PRN), grid.y = covering channel, block.x = hop lane (grid-stride): the shape
/// of gnss_waveform_kernel, so the output rows, energy rows and hop order are identical.
///
/// Per hop and trial (code phase C in grid units, q = floor(C), f = C - q):
///     S = e^{-i th f} sum_{j=-1..2} w_j(f) e^{+i th j} Psi[(q + j) mod GL],
///     r = 0.5 conj(pa) S,
/// th = theta_B inv_cps / G (float, per (job, channel)), pa the hop's carrier phasor. The four
/// stencil phasors e^{+i th j} are hoisted per block; the per-trial cost is one sincosf for the
/// fraction ramp, four table loads and ~40 flops.
///
/// FIXED: the per-hop code phase and carrier are 64-bit fixed-point accumulations (PsiJob's
/// cg_ref/dcg/dsg in 2^-shift grid units, ang0_t/dturn_t in 2^-64 turns) instead of the fp64
/// DFMA + fmod of the production kernel. On GA102 fp64 runs at 1/32 of fp32, and the per-hop
/// fp64 fmod alone is a sizeable share of a 10-20 us launch.
template<int MAXT, bool ROT, class TAB, bool FIXED>
__global__ __launch_bounds__(MAXT) void gnss_psi_waveform_kernel(
    const gnss_cuda::DespreadJob* __restrict__ jobs, const gnss_cuda::PsiJob* __restrict__ pjobs,
    const float* __restrict__ theta, // [n_job][nchan]
    gnss_cuda::DespreadParams p,
    const float* __restrict__ rot, // ROT only: [n_job][nchan]
    float2* __restrict__ wave,     // [3*n_job][nchan][n_hops]
    double* __restrict__ energy) { // [4*n_job][nchan]
    const int b = blockIdx.x;
    const int ci = blockIdx.y;
    const int m = threadIdx.x;
    const int n_chan = gridDim.y;
    const gnss_cuda::DespreadJob job = jobs[b];
    const gnss_cuda::PsiJob pj = pjobs[b];

    // E, P, L, P_HEAD. The fixed-point path sums its few hops per thread in float (2 at CHORD
    // geometry, 6e-8 relative) and widens once: a double add per hop runs on GA102's 1/32-rate
    // fp64 pipe, which the profile showed busy 40% of the launch.
    double e[4] = {0.0, 0.0, 0.0, 0.0};
    float ef[4] = {0.f, 0.f, 0.f, 0.f};
    const bool covered = (job.chan_mask >> ci) & 1ULL;

    // Hoisted: invariant across hops and trials.
    const int G = pj.G;
    const int GP = G * pj.P;
    const int GL = G * job.code_len;
    const TAB* tab = (const TAB*)pj.psi + (size_t)ci * GP;
    const TAB* prev = pj.psi_prev ? (const TAB*)pj.psi_prev + (size_t)ci * pj.n_prev : nullptr;
    const float th = theta[(size_t)b * n_chan + ci];
    float2 Ej[4]; // e^{+i th j}, j = -1, 0, 1, 2
    {
        float s1, c1;
        sincosf(th, &s1, &c1);
        Ej[1] = make_float2(1.f, 0.f);
        Ej[2] = make_float2(c1, s1);
        Ej[0] = make_float2(c1, -s1);
        Ej[3] = cmulf(Ej[2], Ej[2]);
    }
    float2 rotE = make_float2(1.f, 0.f), rotL = rotE;
    if (ROT) {
        float sr, cr;
        sincosf(rot[(size_t)b * n_chan + ci], &sr, &cr);
        rotE = make_float2(cr, sr);
        rotL = make_float2(cr, -sr);
    }
    const size_t row_stride = (size_t)n_chan * p.n_hops; // wave rows E, P, L
    // fp64 path constants (grid units).
    const double dsG = job.ds * (double)G;
    const double GLd = (double)GL;
    // fixed-point path constants. The fraction is read from the top 24 bits below the point
    // (32-bit conversions only: a 64-bit int to float conversion runs on the fp64 pipe).
    const long long GLf = (long long)GL << pj.shift;
    const int fsh = pj.shift - 24;

    for (int mh = m; mh < p.n_hops; mh += blockDim.x) {
        // An uncovered channel still writes zero replica rows (launch_correlate_nm and the
        // pack read every channel's rows) and must not touch a table not built for it.
        if (!covered) {
#pragma unroll
            for (int t = 0; t < 3; ++t)
                wave[((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh] = make_float2(0.f, 0.f);
            continue;
        }

        // Prompt code phase in grid units reduced to [0, GL), and the carrier phasor.
        double CGr = 0.0;
        long long Cfr = 0;
        float2 pa;
        if (FIXED) {
            // The wrap count (<= ~11 per record at CHORD) from a float estimate, then one
            // conditional fix-up each way: a 64-bit modulo is emulated through fp64.
            const int q = (int)(fmaf((float)mh, pj.dcg_r, pj.cg_r));
            Cfr = pj.cg_ref + (long long)mh * pj.dcg - (long long)q * GLf;
            Cfr += (Cfr < 0) ? GLf : 0;
            Cfr -= (Cfr >= GLf) ? GLf : 0;
            const unsigned long long at = pj.ang0_t + (unsigned long long)mh * pj.dturn_t;
            // top 24 bits of the turn fraction -> [0, 2) half-turns for sincospif
            const float x = (float)(unsigned)(at >> 40) * (2.f / 16777216.f);
            float sn, cn;
            sincospif(x, &sn, &cn);
            pa = make_float2(cn, sn);
        } else {
            // As the production kernel: references at p.n0 reduced on the host, the rates
            // multiply only the intra-record offset dn (cudaGnssChordDespread.cu:69-85).
            const double dn = (double)((long long)mh * (long long)p.fft_len);
            const double C_P = job.cp_ref + dn * job.cps;
            const double CG = C_P * (double)G;
            CGr = CG - floor(CG / GLd) * GLd;
            const double ang = job.ang0 + fmod(job.wc * dn, 2.0 * M_PI);
            float sn, cn;
            sincosf((float)ang, &sn, &cn);
            pa = make_float2(cn, sn);
        }
        const float2 pb = make_float2(pa.x, -pa.y);
        const bool in_head = (mh < job.m_head);

        // Trial order {P, E, L} as the production kernel: rows 1, 0, 2 = C, C-ds, C+ds.
        constexpr int NT = ROT ? 1 : 3;
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
            const int t = (tt == 0) ? 1 : (tt == 1) ? 0 : 2;
            int qi;
            float f;
            if (FIXED) {
                long long Cf = Cfr + (tt == 0 ? 0 : (tt == 1 ? -pj.dsg : pj.dsg));
                Cf += (Cf < 0) ? GLf : 0;
                Cf -= (Cf >= GLf) ? GLf : 0;
                qi = (int)(Cf >> pj.shift);
                f = (float)((unsigned)(Cf >> fsh) & 0xffffffu) * (1.f / 16777216.f);
            } else {
                const double Cg = CGr + (tt == 0 ? 0.0 : (tt == 1 ? -dsG : dsG));
                const double qd = floor(Cg);
                f = (float)(Cg - qd);
                qi = (int)qd;
                qi += (qi < 0) ? GL : 0;
                qi -= (qi >= GL) ? GL : 0;
            }
            float w[4];
            lagrange4(f, w);
            float2 S = make_float2(0.f, 0.f);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int a = qi + (j - 1);
                a += (a < 0) ? GL : 0;
                a -= (a >= GL) ? GL : 0;
                const float2 v = psi_at<TAB>(tab, prev, pj.sec, pj.Ns, GP, pj.n_prev, a);
                const float2 wE = make_float2(w[j] * Ej[j].x, w[j] * Ej[j].y);
                const float2 c = cmulf(wE, v);
                S.x += c.x;
                S.y += c.y;
            }
            // The fraction ramp e^{-i th f}.
            float sf, cf;
            sincosf(-th * f, &sf, &cf);
            S = cmulf(S, make_float2(cf, sf));
            const float2 t2 = cmulf(pb, S);
            const float2 r = make_float2(0.5f * t2.x, 0.5f * t2.y);
            const size_t o = ((size_t)(3 * b + t) * n_chan + ci) * p.n_hops + mh;
            const float eef = r.x * r.x + r.y * r.y;
            const double ee = (double)eef;
            if (ROT) {
                wave[o - row_stride] = cmulf(r, rotE);
                wave[o] = r;
                wave[o + row_stride] = cmulf(r, rotL);
                if (FIXED) {
                    ef[1] += eef;
                    if (in_head)
                        ef[3] += eef;
                } else {
                    e[1] += ee;
                    if (in_head)
                        e[3] += ee;
                }
            } else {
                wave[o] = r;
                if (FIXED) {
                    ef[t] += eef;
                    if (t == 1 && in_head)
                        ef[3] += eef;
                } else {
                    e[t] += ee;
                    if (t == 1 && in_head)
                        e[3] += ee;
                }
            }
        }
    }

    // Block reduction over hop lanes: warp shuffles, one barrier, warp 0 finishes. The
    // production kernel's shared tree takes 44 barriers for the four rows, which at two hops
    // per thread was the largest stall in the profile. blockDim.x must be a multiple of 32.
    if (FIXED) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            e[j] = (double)ef[j];
    }
    if (ROT) {
        e[0] = e[1];
        e[2] = e[1];
    }
    __shared__ double sh_w[MAXT / 32][4];
    const int lane = m & 31, warp = m >> 5, n_warp = blockDim.x >> 5;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        e[j] = warp_sum(e[j]);
        if (lane == 0)
            sh_w[warp][j] = e[j];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const double v = warp_sum(lane < n_warp ? sh_w[lane][j] : 0.0);
            if (lane == 0)
                energy[(size_t)(4 * b + j) * n_chan + ci] = v;
        }
    }
}

int block_ceiling_psi(const void* fn) {
    cudaFuncAttributes at{};
    return cudaFuncGetAttributes(&at, fn) == cudaSuccess ? at.maxThreadsPerBlock : 256;
}

template<bool ROT, class TAB, bool FIXED>
cudaError_t launch_psi_t(const gnss_cuda::DespreadJob* jobs, const gnss_cuda::PsiJob* pjobs,
                         const float* theta, dim3 grid, int threads,
                         const gnss_cuda::DespreadParams& p, const float* rot, float2* wave,
                         double* energy, cudaStream_t stream) {
    if (threads > 512) {
        static const int k10 =
            block_ceiling_psi((const void*)gnss_psi_waveform_kernel<1024, ROT, TAB, FIXED>);
        if (threads <= k10) {
            gnss_psi_waveform_kernel<1024, ROT, TAB, FIXED>
                <<<grid, threads, 0, stream>>>(jobs, pjobs, theta, p, rot, wave, energy);
            return cudaGetLastError();
        }
        threads = 512;
    }
    static const int k5 =
        block_ceiling_psi((const void*)gnss_psi_waveform_kernel<512, ROT, TAB, FIXED>);
    if (threads > k5)
        threads = k5;
    gnss_psi_waveform_kernel<512, ROT, TAB, FIXED>
        <<<grid, threads, 0, stream>>>(jobs, pjobs, theta, p, rot, wave, energy);
    return cudaGetLastError();
}

template<class TAB, bool FIXED>
cudaError_t launch_psi_rot(const gnss_cuda::DespreadJob* jobs, const gnss_cuda::PsiJob* pjobs,
                           const float* theta, dim3 grid, int threads,
                           const gnss_cuda::DespreadParams& p, const float* rot, float2* wave,
                           double* energy, cudaStream_t stream) {
    return rot ? launch_psi_t<true, TAB, FIXED>(jobs, pjobs, theta, grid, threads, p, rot, wave,
                                                energy, stream)
               : launch_psi_t<false, TAB, FIXED>(jobs, pjobs, theta, grid, threads, p, rot, wave,
                                                 energy, stream);
}

double reduce_pi(double x) {
    double t = std::fmod(x + M_PI, 2.0 * M_PI);
    if (t < 0.0)
        t += 2.0 * M_PI;
    return t - M_PI; // [-pi, pi)
}

} // namespace

namespace gnss_cuda {

cudaError_t launch_psi_waveform(const DespreadJob* jobs, const PsiJob* pjobs, const float* theta,
                                int n_job, int n_chan, const DespreadParams& p, const float* rot,
                                int fixed, float2* wave, double* energy, cudaStream_t stream) {
    int threads = p.n_hops < 1024 ? p.n_hops : 1024;
    int pow2 = 1;
    while (pow2 * 2 <= threads)
        pow2 *= 2;
    threads = pow2;
    if (threads < 32)
        return cudaErrorInvalidValue; // the warp-shuffle reduction needs whole warps
    const dim3 grid(n_job, n_chan);
    if (p.phi_half)
        return fixed ? launch_psi_rot<__half2, true>(jobs, pjobs, theta, grid, threads, p, rot,
                                                     wave, energy, stream)
                     : launch_psi_rot<__half2, false>(jobs, pjobs, theta, grid, threads, p, rot,
                                                      wave, energy, stream);
    return fixed ? launch_psi_rot<float2, true>(jobs, pjobs, theta, grid, threads, p, rot, wave,
                                                energy, stream)
                 : launch_psi_rot<float2, false>(jobs, pjobs, theta, grid, threads, p, rot, wave,
                                                 energy, stream);
}

void psi_to_half(const float2* src, void* dst, size_t n) {
    __half2* d = (__half2*)dst;
    for (size_t i = 0; i < n; ++i)
        d[i] = __floats2half2_rn(src[i].x, src[i].y);
}

double psi_theta(int chan, int fft_len, double wc, double* theta_far) {
    const double off = 2.0 * M_PI * (double)chan / (double)fft_len;
    const double thB = reduce_pi(off - wc);
    const double thA = reduce_pi(off + wc);
    if (theta_far)
        *theta_far = std::fabs(thA);
    return thB;
}

void psi_fixed_setup(const DespreadJob& j, PsiJob& pj, int fft_len, int n_hops) {
    // Grid-unit fixed point with the largest shift that keeps the record's whole code-phase
    // range below 2^62: the rounding of the per-hop advance dcg is then <= 2^-(shift+1) grid
    // units per hop, accumulated over n_hops.
    const long double G = (long double)pj.G;
    const long double dC = (long double)fft_len * (long double)j.cps * G;
    const long double maxC =
        ((long double)j.code_len + (long double)j.ds + 2.0L) * G + (long double)n_hops * dC;
    int shift = (int)std::floor(62.0L - std::log2(maxC));
    if (shift > 60)
        shift = 60;
    if (shift < 0)
        throw std::runtime_error("psi_fixed_setup: code phase range does not fit int64");
    const long double scale = std::ldexp(1.0L, shift);
    pj.shift = shift;
    pj.cg_ref = (long long)std::llround((long double)j.cp_ref * G * scale);
    pj.dcg = (long long)std::llround(dC * scale);
    pj.dsg = (long long)std::llround((long double)j.ds * G * scale);
    const long double GLf = (long double)j.code_len * G * scale;
    pj.dcg_r = (float)((long double)pj.dcg / GLf);
    pj.cg_r = (float)((long double)pj.cg_ref / GLf);
    constexpr long double TWO_PI_L = 6.283185307179586476925286766559005768L;
    const long double two64 = 18446744073709551616.0L;
    long double fr0 = std::fmod((long double)j.ang0 / TWO_PI_L, 1.0L);
    if (fr0 < 0.0L)
        fr0 += 1.0L;
    long double frd = std::fmod((long double)j.wc * (long double)fft_len / TWO_PI_L, 1.0L);
    if (frd < 0.0L)
        frd += 1.0L;
    long double a0 = fr0 * two64, ad = frd * two64;
    if (a0 >= two64)
        a0 = two64 - 1.0L;
    if (ad >= two64)
        ad = two64 - 1.0L;
    pj.ang0_t = (unsigned long long)a0;
    pj.dturn_t = (unsigned long long)ad;
}

PsiTable build_psi_table(const gnss::ChannelizedReplicaBank& bank, int prn_index,
                         const std::vector<int>& chans, double doppler_hz, double sample_rate,
                         int G, bool frac) {
    using cd = std::complex<double>;
    const auto t0 = std::chrono::steady_clock::now();
    PsiTable T;
    const auto f = bank.hoprate_filter(chans, doppler_hz, prn_index);
    const int Lf = bank.fft_len() * bank.num_taps();
    const double cps = bank.eff_chip_rate() / sample_rate
                       * (1.0 + bank.code_doppler_sign * doppler_hz / bank.carrier_hz());
    const int n_full = (int)std::ceil((double)(Lf - 1) * cps) + 2;
    if (f.d_first != 0 || f.n_chips != n_full)
        throw std::runtime_error("build_psi_table: the bank truncates the PFB span (max_chips "
                                 "must be 0): a truncated gather is not interpolable");
    const double inv = 1.0 / cps;
    const int L = (int)bank.eff_code_length();
    const auto& code = bank.full_code(prn_index);
    const int nc = (int)chans.size();
    const int n_chips = f.n_chips;
    T.G = G;
    T.L = L;
    T.n_chips = n_chips;
    T.inv_cps = inv;
    T.wc = f.wc_built;
    T.theta.resize(nc);
    T.psi.assign((size_t)nc * G * L, make_float2(0.f, 0.f));

    std::vector<double> Kr((size_t)n_chips), Ki((size_t)n_chips);
    std::vector<double> acc_r((size_t)L), acc_i((size_t)L);
    for (int ci = 0; ci < nc; ++ci) {
        T.theta[(size_t)ci] = psi_theta(chans[(size_t)ci], bank.fft_len(), f.wc_built);
        const std::vector<cd>& Phi = f.PhiB[(size_t)ci];
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
        for (int g = 0; g < G; ++g) {
            const double phi = (double)g / (double)G;
            // The smooth edge sits at x + 0.5 (tap units): the rect model's exclusive edge
            // floor(x) + 1 averages to x + 0.5 over the sub-sample phase, so this is the
            // zero-mean-delay smoothing of it. psi_interp_proto.py's x + 1.0 is half a sample
            // (1.6e-3 chips, 4.7 cm) late against production, which the sky test resolves.
            cd prevv = frac ? phi_lin((phi - 1.0) * inv + 0.5) : phi_snap((phi - 1.0) * inv);
            for (int d = 0; d < n_chips; ++d) {
                const cd cur = frac ? phi_lin((phi + (double)d) * inv + 0.5)
                                    : phi_snap((phi + (double)d) * inv);
                const cd K = cur - prevv;
                Kr[(size_t)d] = K.real();
                Ki[(size_t)d] = K.imag();
                prevv = cur;
            }
            // Psi[m] = sum_d code[(m - d) mod L] K_d: a circular convolution over the period.
            std::fill(acc_r.begin(), acc_r.end(), 0.0);
            std::fill(acc_i.begin(), acc_i.end(), 0.0);
            for (int d = 0; d < n_chips; ++d) {
                const double kr = Kr[(size_t)d], ki = Ki[(size_t)d];
                int idx = (L - d % L) % L; // (0 - d) mod L, then walks with m
                for (int mm = 0; mm < L; ++mm) {
                    const double cv = (double)code[(size_t)idx];
                    acc_r[(size_t)mm] += cv * kr;
                    acc_i[(size_t)mm] += cv * ki;
                    if (++idx == L)
                        idx = 0;
                }
            }
            for (int mm = 0; mm < L; ++mm)
                T.psi[((size_t)ci * L + (size_t)mm) * G + (size_t)g] =
                    make_float2((float)acc_r[(size_t)mm], (float)acc_i[(size_t)mm]);
        }
    }
    T.build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return T;
}

} // namespace gnss_cuda
