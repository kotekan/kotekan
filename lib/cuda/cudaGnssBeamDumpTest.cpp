// Standalone GPU check of the cudaGnssBeamDump kernels on a synthetic 4+4b frame: a rank-1
// source with a known steering vector plus noise, with one hot and one dead input.
//  - the covariance against a CPU sum (exact)
//  - the eigenvector against the steering vector (coherence) and the whitened eigenvalue
//    against its expectation
//  - the beam samples against a CPU sum with the GPU's weights (float16 rounding), and the
//    beam's gain over the reference input
// Exits 0 when every check passes.   ./cuda_gnss_beam_dump_test [n_chan] [n_times]

#include "cudaGnssBeamDumpKernel.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using cf = std::complex<float>;
using gnss_cuda::BEAM_INPUTS;

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

int main(int argc, char** argv) {
    const int n_chan = argc > 1 ? atoi(argv[1]) : 3;
    const int n_times = argc > 2 ? atoi(argv[2]) : 8192;
    const int n_freq = 5 * n_chan + 8, ref = 100, n_iter = 6; // the selected channels are 3 + 5c
    const int NI = BEAM_INPUTS;
    std::vector<int> sel(n_chan);
    for (int c = 0; c < n_chan; c++)
        sel[c] = 3 + 5 * c;

    // Frame: noise rms 1.5 per component on every input, a rank-1 source of amplitude 2.5 on the
    // selected channels with a random steering phase per input, input 9 at 3x gain (hot), input
    // 40 dead (0x88). Other channels carry noise only.
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 2.0f * (float)M_PI);
    std::vector<unsigned char> frame((size_t)n_times * n_freq * NI, 0x88);
    std::vector<std::vector<cf>> steer(n_chan, std::vector<cf>(NI));
    std::vector<float> gain(NI, 1.0f);
    gain[9] = 3.0f;
    gain[40] = 0.0f;
    for (int c = 0; c < n_chan; c++)
        for (int i = 0; i < NI; i++)
            steer[c][i] = std::polar(1.0f, uni(rng));
    for (int t = 0; t < n_times; t++)
        for (int f = 0; f < n_freq; f++) {
            int c = -1;
            for (int k = 0; k < n_chan; k++)
                if (sel[k] == f)
                    c = k;
            cf s(0.0f, 0.0f);
            if (c >= 0)
                s = 2.5f * cf(gauss(rng), gauss(rng)) / std::sqrt(2.0f); // a random, white source
            for (int i = 0; i < NI; i++) {
                cf x = cf(1.5f * gauss(rng), 1.5f * gauss(rng));
                if (c >= 0)
                    x += s * steer[c][i];
                x *= gain[i];
                frame[((size_t)t * n_freq + f) * NI + i] =
                    gain[i] == 0.0f ? 0x88 : enc44(x.real(), x.imag());
            }
        }

    unsigned char* d_in;
    int* d_sel;
    float2 *d_cov, *d_q1, *d_q2, *d_w;
    float *d_pw, *d_lam;
    __half2* d_beam;
    CK(cudaMalloc(&d_in, frame.size()));
    CK(cudaMalloc(&d_sel, n_chan * sizeof(int)));
    CK(cudaMalloc(&d_cov, (size_t)n_chan * NI * NI * sizeof(float2)));
    CK(cudaMalloc(&d_q1, (size_t)n_chan * NI * sizeof(float2)));
    CK(cudaMalloc(&d_q2, (size_t)n_chan * NI * sizeof(float2)));
    CK(cudaMalloc(&d_w, (size_t)n_chan * NI * sizeof(float2)));
    CK(cudaMalloc(&d_pw, (size_t)n_chan * NI * sizeof(float)));
    CK(cudaMalloc(&d_lam, (size_t)n_chan * 4 * sizeof(float)));
    CK(cudaMalloc(&d_beam, (size_t)n_chan * n_times * sizeof(__half2)));
    CK(cudaMemcpy(d_in, frame.data(), frame.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sel, sel.data(), n_chan * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_q1, 0, (size_t)n_chan * NI * sizeof(float2)));
    CK(cudaMemset(d_q2, 0, (size_t)n_chan * NI * sizeof(float2)));

    cudaEvent_t e0, e1, e2, e3;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventCreate(&e2));
    CK(cudaEventCreate(&e3));
    // warm-up (module load), then the timed pass; the eigenvector state is reset in between
    gnss_cuda::launch_beam_cov(d_in, d_sel, n_chan, n_freq, n_times, d_cov, 0);
    gnss_cuda::launch_beam_eig(d_cov, n_chan, n_times, ref, n_iter, d_q1, d_q2, d_w, d_pw, d_lam,
                               0);
    gnss_cuda::launch_beam_form(d_in, d_sel, n_chan, n_freq, n_times, d_w, d_beam, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_q1, 0, (size_t)n_chan * NI * sizeof(float2)));
    CK(cudaMemset(d_q2, 0, (size_t)n_chan * NI * sizeof(float2)));
    CK(cudaEventRecord(e0));
    gnss_cuda::launch_beam_cov(d_in, d_sel, n_chan, n_freq, n_times, d_cov, 0);
    CK(cudaEventRecord(e1));
    gnss_cuda::launch_beam_eig(d_cov, n_chan, n_times, ref, n_iter, d_q1, d_q2, d_w, d_pw, d_lam,
                               0);
    CK(cudaEventRecord(e2));
    gnss_cuda::launch_beam_form(d_in, d_sel, n_chan, n_freq, n_times, d_w, d_beam, 0);
    CK(cudaEventRecord(e3));
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    float ms01, ms12, ms23;
    CK(cudaEventElapsedTime(&ms01, e0, e1));
    CK(cudaEventElapsedTime(&ms12, e1, e2));
    CK(cudaEventElapsedTime(&ms23, e2, e3));

    std::vector<float2> cov((size_t)n_chan * NI * NI), w((size_t)n_chan * NI);
    std::vector<float> pw((size_t)n_chan * NI), lam((size_t)n_chan * 4);
    std::vector<__half2> beam((size_t)n_chan * n_times);
    CK(cudaMemcpy(cov.data(), d_cov, cov.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(w.data(), d_w, w.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pw.data(), d_pw, pw.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(lam.data(), d_lam, lam.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(beam.data(), d_beam, beam.size() * sizeof(__half2), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int c = 0; c < n_chan; c++) {
        const int f = sel[c];
        // covariance, exact
        std::vector<cf> V((size_t)NI * NI, cf(0, 0));
        for (int t = 0; t < n_times; t++) {
            const unsigned char* row = &frame[((size_t)t * n_freq + f) * NI];
            for (int i = 0; i < NI; i++) {
                const cf a = dec44(row[i]);
                for (int j = 0; j < NI; j++)
                    V[(size_t)i * NI + j] += a * std::conj(dec44(row[j]));
            }
        }
        double max_err = 0.0;
        for (size_t k = 0; k < V.size(); k++) {
            const float2 g = cov[(size_t)c * NI * NI + k];
            max_err = std::max(max_err, (double)std::abs(V[k] - cf(g.x, g.y)));
        }
        // eigenvector vs the steering vector: the weights are conj(q) wt scale, so the steering
        // estimate is conj(w) / wt; compare to steer by coherence over the live inputs
        cf num(0, 0);
        double d1 = 0, d2 = 0;
        for (int i = 0; i < NI; i++) {
            if (gain[i] == 0.0f)
                continue;
            const float wt = 1.0f / std::sqrt(std::max(pw[(size_t)c * NI + i], 1e-3f));
            const cf q = std::conj(cf(w[(size_t)c * NI + i].x, w[(size_t)c * NI + i].y)) / wt;
            num += q * std::conj(steer[c][i]);
            d1 += std::norm(q);
            d2 += std::norm(steer[c][i]);
        }
        const double coh = std::norm(num) / (d1 * d2);
        // expected whitened eigenvalue: per input, signal fraction = s^2/(s^2 + noise^2) with
        // s^2 = 2.5^2, noise = 2 * 1.5^2 (quantisation ignored); lambda1 ~ 1 + sum over live
        // inputs of the fraction minus one fraction (the eigenvector's own share)
        const double frac = 6.25 / (6.25 + 4.5);
        const double lam_exp = 1.0 + frac * (NI - 2); // 127 live inputs, minus one
        // beam vs a CPU sum with the GPU's weights
        double max_beam_err = 0.0, beam_pow = 0.0, ref_pow = 0.0;
        for (int t = 0; t < n_times; t++) {
            const unsigned char* row = &frame[((size_t)t * n_freq + f) * NI];
            cf y(0, 0);
            for (int i = 0; i < NI; i++)
                y += cf(w[(size_t)c * NI + i].x, w[(size_t)c * NI + i].y) * dec44(row[i]);
            const float2 g = __half22float2(beam[(size_t)c * n_times + t]);
            max_beam_err =
                std::max(max_beam_err, (double)std::abs(y - cf(g.x, g.y)) / (1.0 + std::abs(y)));
            beam_pow += std::norm(y);
            ref_pow += std::norm(dec44(row[ref]));
        }
        const double gain_db = 10.0 * std::log10(beam_pow / ref_pow);
        // a coherent beam of ~127 inputs at signal/noise 6.25/4.5 per input: the beam carries the
        // signal at the reference input's level plus noise / 127, so its power over the
        // reference input's is about 6.25/(6.25+4.5) = -2.4 dB (less than 0 dB: the noise is gone)
        const bool c_ok = max_err < 1e-3 && coh > 0.995
                          && std::abs(lam[(size_t)c * 4] - lam_exp) < 0.08 * lam_exp
                          && lam[(size_t)c * 4 + 1] < 4.0 && max_beam_err < 2e-3 && gain_db > -4.0
                          && gain_db < -1.0;
        printf("channel %d (local %d): cov max err %.2g, eigenvector coherence with the steering "
               "vector %.5f, lambda1 %.1f (expected ~%.1f), lambda2 %.2f, beam max rel err %.2g, "
               "beam/ref power %.2f dB -> %s\n",
               c, f, max_err, coh, lam[(size_t)c * 4], lam_exp, lam[(size_t)c * 4 + 1],
               max_beam_err, gain_db, c_ok ? "OK" : "FAIL");
        ok = ok && c_ok;
    }
    // the hot input gets the same weight magnitude relative to its own noise as the others
    // (whitening): |w_9| * sqrt(pw_9) should be ~ |w_i| * sqrt(pw_i)
    {
        double hot = std::hypot(w[9].x, w[9].y) * std::sqrt(pw[9]);
        double typ = std::hypot(w[10].x, w[10].y) * std::sqrt(pw[10]);
        const bool h_ok = std::abs(hot / typ - 1.0) < 0.15 && std::hypot(w[40].x, w[40].y) < 1e-3;
        printf("whitening: |w| sqrt(pw) hot input / typical = %.3f, dead input |w| = %.2g -> %s\n",
               hot / typ, std::hypot(w[40].x, w[40].y), h_ok ? "OK" : "FAIL");
        ok = ok && h_ok;
    }
    printf("timings for %d channels x %d samples: cov %.3f ms, eig %.3f ms, form %.3f ms\n", n_chan,
           n_times, ms01, ms12, ms23);
    printf("%s\n", ok ? "ALL OK" : "FAILED");
    return ok ? 0 : 1;
}
