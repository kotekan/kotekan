// Standalone GPU check of the cudaDtvSubtract kernels on a synthetic 4+4b frame: one
// transmitter timestream through a known impulse response per input (a direct path and two
// reflections at different lags per input), plus noise and a weak second source.
//  - decode and the per-input powers against a CPU sum (exact)
//  - the beam and the lag correlations against a CPU sum with the same weights
//  - the eigen start: the covariance against a CPU sum, the eigenvector's coherence with the
//    dominant arrival's steering vector
//  - the host-side least-squares fit of the impulse response from the correlations (the
//    Toeplitz normal equations, as scripts/gnss/dtv_state.py solves them) against the truth
//  - the subtraction with the true response: the residual power per input against the noise,
//    and the re-quantised bytes against a CPU subtraction
// Exits 0 when every check passes.   ./cuda_dtv_subtract_test [n_times]

#include "cudaDtvSubtractKernel.hpp"
#include "cudaGnssBeamDumpKernel.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using cf = std::complex<float>;
using cd = std::complex<double>;
using gnss_cuda::DTV_HIST;
using gnss_cuda::DTV_INPUTS;

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,           \
                    __LINE__);                                                                     \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

static unsigned char enc44(float re, float im) {
    int r = (int)std::lround(re), i = (int)std::lround(im);
    r = r < -7 ? -7 : (r > 7 ? 7 : r);
    i = i < -7 ? -7 : (i > 7 ? 7 : i);
    return (unsigned char)(((r + 8) << 4) | (i + 8));
}
static cf dec44(unsigned char b) {
    return cf((float)((b >> 4) & 0x0f) - 8.0f, (float)(b & 0x0f) - 8.0f);
}

static int fails = 0;
static void check(bool ok, const char* what, double got, double want) {
    printf("%s %s: %.6g (want %.6g)\n", ok ? "  ok  " : "FAIL  ", what, got, want);
    if (!ok)
        fails++;
}

int main(int argc, char** argv) {
    const int NT = argc > 1 ? atoi(argv[1]) : 8192;
    const int NI = DTV_INPUTS, NF = 3, NC = 2, NSRC = 2, NLAG = 49, LAG_MIN = -8;
    const int sel_h[NC] = {0, 2};
    std::mt19937 rng(11);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // Truth per channel: timestream s (unit power, 2 history + NT samples so the previous
    // frame's tail exists), impulse response g[i][lag] with a direct path at lag 0 for every
    // input and two reflections at random lags 1..30, noise 1.0 rms per component.
    const int NS = DTV_HIST + NT;
    std::vector<std::vector<cf>> s_true(NC, std::vector<cf>(NS));
    std::vector<std::vector<cf>> g_true(NC, std::vector<cf>((size_t)NLAG * NI));
    std::vector<std::vector<cf>> steer(NC, std::vector<cf>(NI));
    std::vector<unsigned char> frame((size_t)NT * NF * NI);
    std::vector<std::vector<cf>> x_true(NC, std::vector<cf>((size_t)NT * NI)); // decoded frame
    for (int c = 0; c < NC; c++) {
        for (int t = 0; t < NS; t++)
            s_true[c][t] = cf(gauss(rng), gauss(rng)) * 0.7071f;
        for (int i = 0; i < NI; i++) {
            const float amp = (c == 0 ? 2.5f : 1.8f) * (0.6f + 0.8f * uni(rng));
            const float ph = 6.2831853f * uni(rng);
            steer[c][i] = cf(amp * cosf(ph), amp * sinf(ph));
            g_true[c][(size_t)(0 - LAG_MIN) * NI + i] = steer[c][i];
            for (int r = 0; r < 2; r++) {
                const int lag = 1 + (int)(uni(rng) * 30.0f);
                const float a = 0.5f * amp * uni(rng), p2 = 6.2831853f * uni(rng);
                g_true[c][(size_t)(lag - LAG_MIN) * NI + i] += cf(a * cosf(p2), a * sinf(p2));
            }
        }
    }
    // the frame: selected channels = sources + noise (input 5 dead on channel 1); the other
    // channel = noise
    for (int t = 0; t < NT; t++)
        for (int f = 0; f < NF; f++)
            for (int i = 0; i < NI; i++) {
                cf v(gauss(rng), gauss(rng));
                for (int c = 0; c < NC; c++) {
                    if (sel_h[c] != f)
                        continue;
                    for (int j = 0; j < NLAG; j++) {
                        const int idx = DTV_HIST + t - (LAG_MIN + j);
                        if (idx >= 0 && idx < NS)
                            v += g_true[c][(size_t)j * NI + i] * s_true[c][idx];
                    }
                    if (c == 1 && i == 5)
                        v = cf(0.0f, 0.0f);
                }
                frame[((size_t)t * NF + f) * NI + i] = enc44(v.real(), v.imag());
            }
    for (int c = 0; c < NC; c++)
        for (int t = 0; t < NT; t++)
            for (int i = 0; i < NI; i++)
                x_true[c][(size_t)t * NI + i] = dec44(frame[((size_t)t * NF + sel_h[c]) * NI + i]);

    // device buffers
    unsigned char* d_frame;
    int *d_sel, *d_act, *d_ref;
    float2 *d_x, *d_s, *d_w, *d_g, *d_cov, *d_q1, *d_q2, *d_wtmp, *d_rxs, *d_rss;
    float *d_pw_raw, *d_pw_res, *d_ps, *d_pwtmp, *d_lam;
    CK(cudaMalloc(&d_frame, frame.size()));
    CK(cudaMalloc(&d_sel, NC * sizeof(int)));
    CK(cudaMalloc(&d_act, NC * NSRC * sizeof(int)));
    CK(cudaMalloc(&d_ref, NC * sizeof(int)));
    CK(cudaMalloc(&d_x, (size_t)NC * NT * NI * sizeof(float2)));
    CK(cudaMalloc(&d_s, (size_t)NC * NSRC * NS * sizeof(float2)));
    CK(cudaMalloc(&d_w, (size_t)NC * NSRC * NI * sizeof(float2)));
    CK(cudaMalloc(&d_g, (size_t)NC * NSRC * NLAG * NI * sizeof(float2)));
    CK(cudaMalloc(&d_cov, (size_t)NC * NI * NI * sizeof(float2)));
    CK(cudaMalloc(&d_q1, (size_t)NC * NI * sizeof(float2)));
    CK(cudaMalloc(&d_q2, (size_t)NC * NI * sizeof(float2)));
    CK(cudaMalloc(&d_wtmp, (size_t)NC * NI * sizeof(float2)));
    CK(cudaMalloc(&d_rxs, (size_t)NC * NSRC * NLAG * NI * sizeof(float2)));
    CK(cudaMalloc(&d_rss, (size_t)NC * NSRC * NLAG * sizeof(float2)));
    CK(cudaMalloc(&d_pw_raw, NC * NI * sizeof(float)));
    CK(cudaMalloc(&d_pw_res, NC * NI * sizeof(float)));
    CK(cudaMalloc(&d_ps, NC * NSRC * sizeof(float)));
    CK(cudaMalloc(&d_pwtmp, NC * NI * sizeof(float)));
    CK(cudaMalloc(&d_lam, NC * 4 * sizeof(float)));
    CK(cudaMemcpy(d_frame, frame.data(), frame.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sel, sel_h, NC * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_s, 0, (size_t)NC * NSRC * NS * sizeof(float2)));
    CK(cudaMemset(d_g, 0, (size_t)NC * NSRC * NLAG * NI * sizeof(float2)));
    CK(cudaMemset(d_cov, 0, (size_t)NC * NI * NI * sizeof(float2)));
    CK(cudaMemset(d_q1, 0, (size_t)NC * NI * sizeof(float2)));
    CK(cudaMemset(d_q2, 0, (size_t)NC * NI * sizeof(float2)));
    CK(cudaMemset(d_ps, 0, NC * NSRC * sizeof(float)));
    CK(cudaMemset(d_rss, 0, (size_t)NC * NSRC * NLAG * sizeof(float2)));
    CK(cudaMemset(d_rxs, 0, (size_t)NC * NSRC * NLAG * NI * sizeof(float2)));
    const cudaStream_t st = 0;

    // 1. decode + raw powers
    gnss_cuda::launch_dtv_decode(d_frame, d_sel, NC, NF, NT, d_x, d_pw_raw, st);
    std::vector<float2> x_dev((size_t)NC * NT * NI);
    std::vector<float> pw_raw(NC * NI);
    CK(cudaMemcpy(x_dev.data(), d_x, x_dev.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pw_raw.data(), d_pw_raw, pw_raw.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double dmax = 0.0, pmax = 0.0;
    for (int c = 0; c < NC; c++) {
        std::vector<double> p(NI, 0.0);
        for (int t = 0; t < NT; t++)
            for (int i = 0; i < NI; i++) {
                const cf v = x_true[c][(size_t)t * NI + i];
                const float2 d = x_dev[((size_t)c * NT + t) * NI + i];
                dmax = std::max(dmax, (double)std::abs(cf(d.x, d.y) - v));
                p[i] += std::norm(v);
            }
        for (int i = 0; i < NI; i++)
            pmax = std::max(pmax, std::abs(p[i] - pw_raw[c * NI + i]) / p[i]);
    }
    check(dmax == 0.0, "decode max |error|", dmax, 0.0);
    check(pmax < 1e-5, "raw power max rel error", pmax, 0.0);

    // 2. eigen start on channel 0/1 (source 0): covariance vs CPU, eigenvector vs the steering
    //    vector of the direct path (the reflections make it only approximately rank-1)
    gnss_cuda::launch_dtv_cov(d_x, NC, NT, d_cov, st);
    std::vector<float2> cov((size_t)NC * NI * NI);
    CK(cudaMemcpy(cov.data(), d_cov, cov.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    double cmax = 0.0;
    for (int c = 0; c < NC; c++)
        for (int i = 0; i < NI; i += 7)
            for (int j = 0; j < NI; j += 5) {
                cd acc(0.0, 0.0);
                for (int t = 0; t < NT; t++)
                    acc += cd(x_true[c][(size_t)t * NI + i])
                           * std::conj(cd(x_true[c][(size_t)t * NI + j]));
                const float2 d = cov[((size_t)c * NI + i) * NI + j];
                cmax = std::max(cmax, std::abs(acc - cd(d.x, d.y)) / (1.0 + std::abs(acc)));
            }
    check(cmax < 1e-4, "covariance max rel error (sampled)", cmax, 0.0);
    gnss_cuda::launch_dtv_pick_ref(d_cov, NC, d_ref, st);
    gnss_cuda::launch_beam_eig(d_cov, NC, NT, 0, 24, d_q1, d_q2, d_wtmp, d_pwtmp, d_lam, st, d_ref);
    std::vector<float2> w0((size_t)NC * NI);
    std::vector<float> lam(NC * 4);
    std::vector<int> ref(NC);
    CK(cudaMemcpy(w0.data(), d_wtmp, w0.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(lam.data(), d_lam, lam.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref.data(), d_ref, NC * sizeof(int), cudaMemcpyDeviceToHost));
    for (int c = 0; c < NC; c++) {
        // coherence of conj(w) (the beam points along conj(w)) with the direct-path steering
        cd num(0.0, 0.0);
        double n1 = 0.0, n2 = 0.0;
        for (int i = 0; i < NI; i++) {
            const cd wi(w0[(size_t)c * NI + i].x, -w0[(size_t)c * NI + i].y);
            num += wi * std::conj(cd(steer[c][i]));
            n1 += std::norm(wi);
            n2 += std::norm(steer[c][i]);
        }
        const double coh = std::abs(num) / std::sqrt(n1 * n2);
        printf("   channel %d: reference input %d, lambda1/lambda2 %.1f/%.1f\n", c, ref[c],
               lam[c * 4], lam[c * 4 + 1]);
        check(coh > 0.9, "eigen start: coherence with the direct path", coh, 1.0);
        check(ref[c] != 5 || c == 0, "eigen start: reference input is not the dead one", ref[c],
              -1);
    }

    // 3. beam, lag correlations and the fit with the eigen-start weights (source 0 active)
    std::vector<int> act(NC * NSRC, 0);
    for (int c = 0; c < NC; c++)
        act[c * NSRC] = 1;
    CK(cudaMemcpy(d_act, act.data(), act.size() * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy2D(d_w, (size_t)NSRC * NI * sizeof(float2), d_wtmp, NI * sizeof(float2),
                    NI * sizeof(float2), NC, cudaMemcpyDeviceToDevice));
    // a previous frame's tail of s: the test's history is zero, as after a cold start
    gnss_cuda::launch_dtv_shift(d_s, NC, NSRC, NT, st);
    gnss_cuda::launch_dtv_beam(d_x, d_w, d_act, 0, NC, NSRC, NT, d_s, d_ps, st);
    gnss_cuda::launch_dtv_lagcorr(d_x, d_s, d_act, 0, NC, NSRC, NT, NLAG, LAG_MIN, d_rxs, d_rss,
                                  st);
    CK(cudaDeviceSynchronize());
    std::vector<float2> s_dev((size_t)NC * NSRC * NS), rxs((size_t)NC * NSRC * NLAG * NI),
        rss((size_t)NC * NSRC * NLAG);
    std::vector<float> ps(NC * NSRC);
    CK(cudaMemcpy(s_dev.data(), d_s, s_dev.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rxs.data(), d_rxs, rxs.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rss.data(), d_rss, rss.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ps.data(), d_ps, ps.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double smax = 0.0, rmax = 0.0, rsmax = 0.0, psmax = 0.0;
    std::vector<std::vector<cf>> s_est(NC, std::vector<cf>(NS, cf(0.0f, 0.0f)));
    for (int c = 0; c < NC; c++) {
        double pss = 0.0;
        for (int t = 0; t < NT; t++) {
            cd acc(0.0, 0.0);
            for (int i = 0; i < NI; i++)
                acc += cd(w0[(size_t)c * NI + i].x, w0[(size_t)c * NI + i].y)
                       * cd(x_true[c][(size_t)t * NI + i]);
            const float2 d = s_dev[((size_t)c * NSRC + 0) * NS + DTV_HIST + t];
            smax = std::max(smax, std::abs(acc - cd(d.x, d.y)) / (1.0 + std::abs(acc)));
            s_est[c][DTV_HIST + t] = cf(d.x, d.y);
            pss += std::norm(acc);
        }
        psmax = std::max(psmax, std::abs(pss - ps[c * NSRC]) / pss);
        for (int j = 0; j < NLAG; j++) {
            const int lag = LAG_MIN + j;
            cd ass(0.0, 0.0); // R_ss at lag difference d = j
            for (int t = 0; t < NT; t++)
                ass += cd(s_est[c][DTV_HIST + t]) * std::conj(cd(s_est[c][DTV_HIST + t - j]));
            const float2 d = rss[((size_t)c * NSRC + 0) * NLAG + j];
            rsmax = std::max(rsmax, std::abs(ass - cd(d.x, d.y)) / (1.0 + std::abs(ass)));
            for (int i = 0; i < NI; i += 3) {
                cd acc(0.0, 0.0);
                for (int t = 0; t < NT; t++) {
                    const int idx = DTV_HIST + t - lag;
                    if (idx < NS)
                        acc += cd(x_true[c][(size_t)t * NI + i]) * std::conj(cd(s_est[c][idx]));
                }
                const float2 d = rxs[(((size_t)c * NSRC + 0) * NLAG + j) * NI + i];
                rmax = std::max(rmax, std::abs(acc - cd(d.x, d.y)) / (1.0 + std::abs(acc)));
            }
        }
    }
    check(smax < 1e-4, "beam max rel error", smax, 0.0);
    check(psmax < 1e-4, "beam power rel error", psmax, 0.0);
    check(rsmax < 1e-3, "R_ss max rel error", rsmax, 0.0);
    check(rmax < 1e-3, "R_xs max rel error (sampled)", rmax, 0.0);

    // 4. the host fit: Toeplitz normal equations T g_i = r_i with T[j][m] = R_ss(lag_j - lag_m)
    //    (from R_ss at the lag differences, conjugate for negative ones), solved by Gaussian
    //    elimination; the fitted g against the truth, expressed through the beam's own scale:
    //    the beam is a linear filter of s, so g_fit * s_est should reproduce g_true * s_true.
    //    Judge by the residual power after subtracting with the fit (step 5 does it on the GPU).
    std::vector<std::vector<cf>> g_fit(NC, std::vector<cf>((size_t)NLAG * NI));
    for (int c = 0; c < NC; c++) {
        // R_ss at lag difference d in [-(NLAG-1), NLAG-1]: rss[d] for d >= 0, its conjugate for
        // d < 0 (R_ss(-d) = conj(R_ss(d)))
        std::vector<cd> rl(2 * NLAG - 1);
        for (int d = 0; d < NLAG; d++) {
            const float2 q = rss[((size_t)c * NSRC + 0) * NLAG + d];
            rl[(size_t)(d + NLAG - 1)] = cd(q.x, q.y);
            rl[(size_t)(-d + NLAG - 1)] = std::conj(cd(q.x, q.y));
        }
        // T[j][m] = sum_t s(t - lag_j) conj(s(t - lag_m)) ~ R_ss(lag_m - lag_j)
        std::vector<cd> T((size_t)NLAG * NLAG);
        for (int j = 0; j < NLAG; j++)
            for (int m = 0; m < NLAG; m++)
                T[(size_t)j * NLAG + m] = rl[(size_t)((m - j) + NLAG - 1)];
        for (int i = 0; i < NI; i++) {
            std::vector<cd> A(T), b(NLAG);
            for (int j = 0; j < NLAG; j++) {
                const float2 q = rxs[(((size_t)c * NSRC + 0) * NLAG + j) * NI + i];
                b[(size_t)j] = cd(q.x, q.y);
            }
            // r_i(lag_j) = sum_m g_i(lag_m) T[m][j]^*? Derive: x_i(t) = sum_m g_m s(t - lag_m);
            // R_xs,i(lag_j) = sum_t x_i(t) conj(s(t - lag_j)) = sum_m g_m sum_t s(t - lag_m)
            // conj(s(t - lag_j)) = sum_m g_m T[m][j]. Solve T^T g = b.
            std::vector<cd> M((size_t)NLAG * NLAG);
            for (int j = 0; j < NLAG; j++)
                for (int m = 0; m < NLAG; m++)
                    M[(size_t)j * NLAG + m] = A[(size_t)m * NLAG + j];
            for (int p = 0; p < NLAG; p++) {
                int piv = p;
                for (int q = p + 1; q < NLAG; q++)
                    if (std::abs(M[(size_t)q * NLAG + p]) > std::abs(M[(size_t)piv * NLAG + p]))
                        piv = q;
                if (piv != p) {
                    for (int m = 0; m < NLAG; m++)
                        std::swap(M[(size_t)p * NLAG + m], M[(size_t)piv * NLAG + m]);
                    std::swap(b[(size_t)p], b[(size_t)piv]);
                }
                for (int q = p + 1; q < NLAG; q++) {
                    const cd f = M[(size_t)q * NLAG + p] / M[(size_t)p * NLAG + p];
                    for (int m = p; m < NLAG; m++)
                        M[(size_t)q * NLAG + m] -= f * M[(size_t)p * NLAG + m];
                    b[(size_t)q] -= f * b[(size_t)p];
                }
            }
            for (int p = NLAG - 1; p >= 0; p--) {
                cd acc = b[(size_t)p];
                for (int m = p + 1; m < NLAG; m++)
                    acc -= M[(size_t)p * NLAG + m] * cd(g_fit[c][(size_t)m * NI + i]);
                g_fit[c][(size_t)p * NI + i] = cf(acc / M[(size_t)p * NLAG + p]);
            }
        }
    }

    // 5. subtract with the fitted response on the GPU: residual power per input vs the noise
    //    (the second source on channel 1 stays), and the bytes vs a CPU subtraction
    std::vector<float2> g_dev((size_t)NC * NSRC * NLAG * NI, make_float2(0.0f, 0.0f));
    for (int c = 0; c < NC; c++)
        for (int j = 0; j < NLAG; j++)
            for (int i = 0; i < NI; i++)
                g_dev[(((size_t)c * NSRC + 0) * NLAG + j) * NI + i] = make_float2(
                    g_fit[c][(size_t)j * NI + i].real(), g_fit[c][(size_t)j * NI + i].imag());
    CK(cudaMemcpy(d_g, g_dev.data(), g_dev.size() * sizeof(float2), cudaMemcpyHostToDevice));
    for (int c = 0; c < NC; c++)
        act[c * NSRC] = 2;
    CK(cudaMemcpy(d_act, act.data(), act.size() * sizeof(int), cudaMemcpyHostToDevice));
    gnss_cuda::launch_dtv_subtract(d_x, d_s, d_g, d_act, 0, NC, NSRC, NT, NLAG, LAG_MIN, st);
    gnss_cuda::launch_dtv_encode(d_x, d_sel, d_act, NC, NSRC, NF, NT, d_frame, d_pw_res, st);
    CK(cudaDeviceSynchronize());
    std::vector<float> pw_res(NC * NI);
    std::vector<unsigned char> frame_out(frame.size());
    CK(cudaMemcpy(pw_res.data(), d_pw_res, pw_res.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(frame_out.data(), d_frame, frame.size(), cudaMemcpyDeviceToHost));
    for (int c = 0; c < NC; c++) {
        double praw = 0.0, pres = 0.0, pres_cpu = 0.0;
        long nbyte = 0, nbad = 0;
        for (int i = 0; i < NI; i++) {
            praw += pw_raw[c * NI + i];
            pres += pw_res[c * NI + i];
        }
        for (int t = 0; t < NT; t++)
            for (int i = 0; i < NI; i++) {
                cf r = x_true[c][(size_t)t * NI + i];
                for (int j = 0; j < NLAG; j++) {
                    const int idx = DTV_HIST + t - (LAG_MIN + j);
                    if (idx < NS)
                        r -= g_fit[c][(size_t)j * NI + i] * s_est[c][idx];
                }
                pres_cpu += std::norm(r);
                const unsigned char want = enc44(r.real(), r.imag());
                const unsigned char got = frame_out[((size_t)t * NF + sel_h[c]) * NI + i];
                nbyte++;
                if (got != want) {
                    // a rounding tie can differ by one LSB in one component
                    const cf a = dec44(got), b = dec44(want);
                    const bool tie =
                        std::abs(std::abs(r.real() - std::trunc(r.real())) - 0.5f) < 0.05f
                        || std::abs(std::abs(r.imag() - std::trunc(r.imag())) - 0.5f) < 0.05f;
                    if (std::abs(a.real() - b.real()) > 1.0f || std::abs(a.imag() - b.imag()) > 1.0f
                        || !tie)
                        nbad++;
                }
            }
        const double noise = 2.0 * NT * (c == 1 ? NI - 1 : NI); // per-component variance 1
        printf(
            "   channel %d: power raw %.3g, residual %.3g (CPU %.3g), noise %.3g, %ld of %ld bytes "
            "differ beyond a rounding tie\n",
            c, praw, pres, pres_cpu, noise, nbad, nbyte);
        check(std::abs(pres - pres_cpu) / pres_cpu < 1e-3, "residual power vs CPU", pres, pres_cpu);
        check(pres < 1.35 * noise, "residual power within 35 % of the noise", pres / noise, 1.0);
        check(nbad == 0, "re-quantised bytes vs CPU", (double)nbad, 0.0);
    }
    // the untouched channel's bytes are unchanged
    long changed = 0;
    for (int t = 0; t < NT; t++)
        for (int i = 0; i < NI; i++)
            changed +=
                frame_out[((size_t)t * NF + 1) * NI + i] != frame[((size_t)t * NF + 1) * NI + i];
    check(changed == 0, "unselected channel untouched", (double)changed, 0.0);

    printf(fails ? "FAILED (%d)\n" : "all checks passed\n", fails);
    return fails ? 1 : 0;
}
