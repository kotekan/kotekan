// aggbench -- measure and validate the acquisition SURFACE AGGREGATE, which is the whole cost
// of a search pass (docs/CHORD_GNSS_STATE.md 5o, 5q).
//
// Why this exists: every search speedup so far has been a parameter (fine_step, hint spans,
// prns_per_pass, require_hint) that buys latency by NOT searching. Before moving the aggregate
// onto the GPU there has to be a reference that (a) runs at BLIND dimensions rather than the
// hinted ones the live stage enjoys, and (b) can validate any replacement against the shipped
// gnss::aggregate_accumulate rather than against itself.
//
// It also tests one algorithmic claim that, if it holds, changes the flop balance before any
// GPU work happens:
//
//   THE FINE-LAG AXIS IS AN EXACT DFT WHEN s_step * s_cols == sph.
//   D(q,i) = sum_c P_c(q) e^{+i 2 pi f_c (i*s_step) / sph}. With s_step*s_cols = sph the ramp is
//   e^{+i 2 pi f_c i / s_cols}, which depends on f_c ONLY THROUGH f_c mod s_cols. So folding the
//   channels into an s_cols-length bin array and taking one s_cols-point inverse DFT is not an
//   approximation of the direct sum -- it IS the direct sum, reassociated.
//   Direct: nc*s_cols complex MACs = 79*128*8 = 80896 flop/row. Folded: nc adds + one 128-pt
//   FFT ~ 4480 flop/row.
//
// This is NOT the coarse-axis fold that was tried and reverted (5o). That one assumed the
// replica was periodic in the window and died on the NH overlay. This one assumes nothing about
// the signal at all -- it is an identity about the ramp, and the guard below refuses to run when
// its one arithmetic precondition fails.
//
// build:  ./build_tool.sh aggbench
// usage:  ./aggbench [--nd 321] [--mp 3125] [--nc 79] [--fine-step 128] [--threads 16]
//                    [--reps 1] [--no-check]

#include "gnssChannelizedAcquire.hpp"
#ifdef GNSS_CUDA
#include "cudaGnssAcquireKernel.hpp"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using cf = std::complex<float>;

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// The live CHORD aggregator's covering set: the union of six nodes' stride-16 combs over global
// bins 5972..6076. 79 of 105 bins, gcd of the differences = 1 (so s_stored == sph, the worst
// case for the surface size -- there is no periodicity to exploit on the fine axis).
static std::vector<int> chord_channels(int nc) {
    std::vector<int> ids;
    for (int off = 0; off < 16 && (int)ids.size() < nc; ++off)
        for (int b = 5972 + off; b <= 6076 && (int)ids.size() < nc; b += 16)
            ids.push_back(b);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    ids.resize(std::min<size_t>(ids.size(), (size_t)nc));
    return ids;
}

// ---------------------------------------------------------------------------------------------
// The folded aggregate: identical output to gnss::aggregate_accumulate, via one s_cols-point
// inverse DFT per (Doppler, coarse-lag) row instead of nc*s_cols complex MACs.
// ---------------------------------------------------------------------------------------------
static gnss::AcquisitionSurface aggregate_folded(const std::vector<std::vector<std::vector<cf>>>& P,
                                                 const std::vector<int>& chan_freq,
                                                 int samples_per_hop, std::vector<double>& surf,
                                                 int n_threads, int fine_step, bool* ok) {
    const int nc = (int)P.size();
    const int nd = nc ? (int)P[0].size() : 0;
    const int Mp = (nc && nd) ? (int)P[0][0].size() : 0;
    const int sph = samples_per_hop;

    int g = 0;
    for (int ci = 1; ci < nc; ++ci)
        g = std::gcd(g, std::abs(chan_freq[ci] - chan_freq[0]));
    const int s_stored = (nc > 0 && sph > 0) ? sph / std::gcd(g, sph) : sph;
    const int s_step = std::max(1, fine_step);
    const gnss::AcquisitionSurface dims{nd, Mp, sph, s_stored, s_step};
    const int s_cols = dims.fine();
    if ((long)surf.size() != dims.size())
        surf.assign(dims.size(), 0.0);
    if (nc == 0)
        return dims;

    // THE PRECONDITION. The ramp collapses to a length-s_cols DFT only when stepping s_cols
    // times by s_step walks exactly one full period of the ramp argument. Refuse otherwise --
    // silently falling back to something "close" is how the reverted coarse fold read as
    // nearly working.
    *ok = ((long)s_cols * (long)s_step == (long)s_stored);
    if (!*ok)
        return dims;

    // Channel -> fold bin. Factoring the first channel's ramp out is legitimate because it has
    // unit modulus and only |D|^2 is kept; it also keeps the bin index non-negative.
    std::vector<int> bin(nc);
    for (int ci = 0; ci < nc; ++ci) {
        const long df = (long)(chan_freq[ci] - chan_freq[0]) * (long)(s_stored / s_cols) / 1;
        // ramp arg = 2 pi (f_c - f_0) * (i*s_step) / sph; with s_cols*s_step == s_stored,
        // s_step/sph = 1/(s_cols * (sph/s_stored)), so the bin is (f_c-f_0)/(sph/s_stored)
        // mod s_cols. sph/s_stored == gcd(g,sph) divides every channel difference by
        // construction, so this is exact integer arithmetic.
        const int gg = sph / s_stored;
        const long b = ((long)(chan_freq[ci] - chan_freq[0]) / gg) % s_cols;
        bin[ci] = (int)((b % s_cols + s_cols) % s_cols);
        (void)df;
    }

    const long n_rows = (long)nd * Mp;
    const int TILE = 2048; // rows per FFTW batch: keeps the scratch in cache

    fftwf_plan plan = nullptr;
    {
        std::vector<cf> scratch((size_t)TILE * s_cols);
        int n[] = {s_cols};
        // FFTW_BACKWARD == e^{+i...}, matching the aggregate's ramp sign.
        plan = fftwf_plan_many_dft(1, n, TILE, reinterpret_cast<fftwf_complex*>(scratch.data()),
                                   nullptr, 1, s_cols,
                                   reinterpret_cast<fftwf_complex*>(scratch.data()), nullptr, 1,
                                   s_cols, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    const auto run_rows = [&](long r0, long r1) {
        std::vector<cf> buf((size_t)TILE * s_cols);
        for (long rb = r0; rb < r1; rb += TILE) {
            const long rn = std::min((long)TILE, r1 - rb);
            std::fill(buf.begin(), buf.begin() + (size_t)rn * s_cols, cf(0.0f, 0.0f));
            for (long r = 0; r < rn; ++r) {
                const int d = (int)((rb + r) / Mp), q = (int)((rb + r) % Mp);
                cf* row = buf.data() + (size_t)r * s_cols;
                for (int ci = 0; ci < nc; ++ci)
                    row[bin[ci]] += P[ci][d][q]; // the fold: nc adds, not nc*s_cols MACs
            }
            fftwf_execute_dft(plan, reinterpret_cast<fftwf_complex*>(buf.data()),
                              reinterpret_cast<fftwf_complex*>(buf.data()));
            for (long r = 0; r < rn; ++r) {
                const cf* row = buf.data() + (size_t)r * s_cols;
                double* out = surf.data() + (rb + r) * s_cols;
                for (int s = 0; s < s_cols; ++s)
                    out[s] += (double)(row[s].real() * row[s].real()
                                       + row[s].imag() * row[s].imag());
            }
        }
    };

    const int nt = (int)std::max(1L, std::min((long)std::max(1, n_threads), n_rows / TILE + 1));
    if (nt == 1) {
        run_rows(0, n_rows);
    } else {
        std::vector<std::thread> pool;
        for (int t = 0; t < nt; ++t) {
            const long r0 = ((n_rows / TILE + 1) * t / nt) * TILE;
            const long r1 = std::min(n_rows, ((n_rows / TILE + 1) * (t + 1) / nt) * TILE);
            if (r1 > r0)
                pool.emplace_back(run_rows, r0, r1);
        }
        for (auto& th : pool)
            th.join();
    }
    fftwf_destroy_plan(plan);
    return dims;
}

#ifdef GNSS_CUDA
// ---------------------------------------------------------------------------------------------
// GPU driver. Deliberately does the [c][d][q] -> [d][q][c] transpose HERE, on the host, and
// times only the kernel: in the real pipeline cuFFT writes the [d][q][c] layout directly
// (ostride=nc, odist=1), so charging this benchmark for a transpose that will not exist would
// flatter the CPU. What IS charged is every byte the kernel touches.
// ---------------------------------------------------------------------------------------------
#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

struct GpuAgg {
    float2 *dP = nullptr, *dRamp = nullptr, *dTw = nullptr;
    float* dSurf = nullptr;
    int* dBin = nullptr;
    long n_rows = 0;
    int nc = 0, s_cols = 0, n_dop = 0, mp = 0;

    void setup(const std::vector<std::vector<std::vector<cf>>>& P, const std::vector<int>& ids,
               int sph, int fine_step, int nd, int Mp) {
        nc = (int)P.size();
        const int s_stored = sph; // g == 1 for the CHORD union comb; see the header
        s_cols = s_stored / fine_step;
        n_rows = (long)nd * Mp;
        n_dop = nd;
        mp = Mp;

        std::vector<float2> hP((size_t)n_rows * nc);
        for (int c = 0; c < nc; ++c)
            for (int d = 0; d < nd; ++d)
                for (int q = 0; q < Mp; ++q) {
                    const cf v = P[c][d][q];
                    hP[((size_t)d * Mp + q) * nc + c] = make_float2(v.real(), v.imag());
                }
        CK(cudaMalloc(&dP, hP.size() * sizeof(float2)));
        CK(cudaMemcpy(dP, hP.data(), hP.size() * sizeof(float2), cudaMemcpyHostToDevice));

        std::vector<float2> hR((size_t)nc * s_cols);
        for (int c = 0; c < nc; ++c)
            for (int i = 0; i < s_cols; ++i) {
                const double a = 2.0 * M_PI * ids[c] * (double)(i * fine_step) / sph;
                hR[(size_t)c * s_cols + i] = make_float2((float)std::cos(a), (float)std::sin(a));
            }
        CK(cudaMalloc(&dRamp, hR.size() * sizeof(float2)));
        CK(cudaMemcpy(dRamp, hR.data(), hR.size() * sizeof(float2), cudaMemcpyHostToDevice));

        // Fold bins and twiddles (see cudaGnssAcquireKernel.hpp).
        const int gg = sph / s_stored;
        std::vector<int> hB(nc);
        for (int c = 0; c < nc; ++c) {
            const long b = ((long)(ids[c] - ids[0]) / gg) % s_cols;
            hB[c] = (int)((b % s_cols + s_cols) % s_cols);
        }
        CK(cudaMalloc(&dBin, hB.size() * sizeof(int)));
        CK(cudaMemcpy(dBin, hB.data(), hB.size() * sizeof(int), cudaMemcpyHostToDevice));

        std::vector<float2> hW((size_t)s_cols / 2);
        for (int j = 0; j < s_cols / 2; ++j) {
            const double a = 2.0 * M_PI * j / s_cols; // e^{+2 pi i j / s_cols}: INVERSE
            hW[j] = make_float2((float)std::cos(a), (float)std::sin(a));
        }
        CK(cudaMalloc(&dTw, hW.size() * sizeof(float2)));
        CK(cudaMemcpy(dTw, hW.data(), hW.size() * sizeof(float2), cudaMemcpyHostToDevice));

        CK(cudaMalloc(&dSurf, (size_t)n_rows * s_cols * sizeof(float)));
    }

    // Returns kernel seconds; fills `out` with the surface.
    double run(bool folded, std::vector<double>& out) {
        CK(cudaMemset(dSurf, 0, (size_t)n_rows * s_cols * sizeof(float)));
        CK(cudaDeviceSynchronize());
        const double t0 = now_s();
        gnss_cuda::launch_aggregate(dP, dSurf, dRamp, dBin, dTw, nc, n_dop, mp, s_cols, folded, 0);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        const double dt = now_s() - t0;
        std::vector<float> hS((size_t)n_rows * s_cols);
        CK(cudaMemcpy(hS.data(), dSurf, hS.size() * sizeof(float), cudaMemcpyDeviceToHost));
        out.assign(hS.begin(), hS.end());
        return dt;
    }

    double peak_test(long n_cells, double* pk, long long* idx, double* sum, int reps) {
        void* scratch = nullptr;
        CK(cudaMalloc(&scratch, gnss_cuda::peak_scratch_bytes(n_cells)));
        float2* dVal = nullptr;
        long long* dIdx = nullptr;
        CK(cudaMalloc(&dVal, sizeof(float2)));
        CK(cudaMalloc(&dIdx, sizeof(long long)));
        // Warm up: the first launch of a kernel pays module load, which at these timescales
        // is the whole measurement.
        gnss_cuda::launch_peak(dSurf, n_cells, dVal, dIdx, scratch, 0);
        CK(cudaDeviceSynchronize());
        const double t0 = now_s();
        for (int r = 0; r < reps; ++r)
            gnss_cuda::launch_peak(dSurf, n_cells, dVal, dIdx, scratch, 0);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        const double dt = (now_s() - t0) / reps;
        float2 hv;
        CK(cudaMemcpy(&hv, dVal, sizeof(float2), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(idx, dIdx, sizeof(long long), cudaMemcpyDeviceToHost));
        *pk = hv.x;
        *sum = hv.y;
        cudaFree(scratch);
        cudaFree(dVal);
        cudaFree(dIdx);
        return dt;
    }

    ~GpuAgg() {
        cudaFree(dP);
        cudaFree(dRamp);
        cudaFree(dTw);
        cudaFree(dBin);
        cudaFree(dSurf);
    }
};
#endif // GNSS_CUDA

int main(int argc, char** argv) {
    int nd = 321, Mp = 3125, nc = 79, fine_step = 128, threads = 16, reps = 1;
    bool check = true, gpu = false;
    const int sph = 16384;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::atoi(argv[++i]); };
        if (a == "--nd")
            nd = next();
        else if (a == "--mp")
            Mp = next();
        else if (a == "--nc")
            nc = next();
        else if (a == "--fine-step")
            fine_step = next();
        else if (a == "--threads")
            threads = next();
        else if (a == "--reps")
            reps = next();
        else if (a == "--no-check")
            check = false;
        else if (a == "--gpu")
            gpu = true;
    }

    const std::vector<int> ids = chord_channels(nc);
    nc = (int)ids.size();
    const int s_cols = (sph / fine_step);

    printf("aggbench: nc %d  n_dop %d  Mp %d  sph %d  fine_step %d -> s_cols %d\n", nc, nd, Mp, sph,
           fine_step, s_cols);
    printf("  surface %ld cells (%.2f GB as double, %.2f GB as float)\n", (long)nd * Mp * s_cols,
           1e-9 * (double)nd * Mp * s_cols * 8, 1e-9 * (double)nd * Mp * s_cols * 4);
    printf("  P array %.2f GB\n", 1e-9 * (double)nc * nd * Mp * 8);

    // Deterministic P with a planted peak, so a correctness check has something to localize.
    std::mt19937 rng(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int d0 = nd / 3, q0 = Mp / 2, s0 = 37; // planted cell (fine index, not sample)
    std::vector<std::vector<std::vector<cf>>> P(
        nc, std::vector<std::vector<cf>>(nd, std::vector<cf>(Mp)));
    for (int ci = 0; ci < nc; ++ci)
        for (int d = 0; d < nd; ++d)
            for (int q = 0; q < Mp; ++q)
                P[ci][d][q] = cf(gauss(rng), gauss(rng));
    for (int ci = 0; ci < nc; ++ci) {
        // NOTE the MINUS: the aggregate sums e^{+i 2 pi f_c s / sph}, so a peak AT s0 needs the
        // planted phasor to be the conjugate ramp. Planting +ang instead puts the peak at
        // s_cols - s0, which is a correct answer to a different question.
        const double ang = -2.0 * M_PI * (double)ids[ci] * (double)(s0 * fine_step) / (double)sph;
        P[ci][d0][q0] += 40.0f * cf((float)std::cos(ang), (float)std::sin(ang));
    }

    std::vector<double> surf_ref, surf_fold;
    gnss::AcquisitionSurface dims{};

    double t_ref = 0.0;
    for (int r = 0; r < reps; ++r) {
        surf_ref.assign(0, 0.0);
        const double t0 = now_s();
        dims = gnss::aggregate_accumulate(P, ids, sph, surf_ref, threads, fine_step);
        t_ref += now_s() - t0;
    }
    t_ref /= reps;
    const double flops_ref = 8.0 * (double)nc * nd * Mp * dims.fine();
    printf("\nCPU direct   : %8.3f s   %7.2f GFlop/s   (%.1f GFlop)\n", t_ref,
           1e-9 * flops_ref / t_ref, 1e-9 * flops_ref);

    bool ok = false;
    double t_fold = 0.0;
    for (int r = 0; r < reps; ++r) {
        surf_fold.assign(0, 0.0);
        const double t0 = now_s();
        aggregate_folded(P, ids, sph, surf_fold, threads, fine_step, &ok);
        t_fold += now_s() - t0;
    }
    t_fold /= reps;
    if (!ok) {
        printf("CPU folded   : PRECONDITION FAILED (s_cols*s_step != s_stored) -- not run\n");
        return 1;
    }
    printf("CPU folded   : %8.3f s   %.2fx        (equivalent %.2f GFlop/s)\n", t_fold,
           t_ref / t_fold, 1e-9 * flops_ref / t_fold);

    if (check) {
        // Relative agreement, measured against the surface SCALE rather than per cell: the
        // aggregate is a sum of ~nc random phasors, so individual cells near a null have
        // arbitrarily large relative error while carrying no information.
        double maxabs = 0.0, maxdiff = 0.0, sum_ref = 0.0, sum_fold = 0.0;
        long argmax_ref = -1, argmax_fold = -1;
        double pk_ref = -1, pk_fold = -1;
        for (long i = 0; i < (long)surf_ref.size(); ++i) {
            maxabs = std::max(maxabs, std::fabs(surf_ref[i]));
            maxdiff = std::max(maxdiff, std::fabs(surf_ref[i] - surf_fold[i]));
            sum_ref += surf_ref[i];
            sum_fold += surf_fold[i];
            if (surf_ref[i] > pk_ref) {
                pk_ref = surf_ref[i];
                argmax_ref = i;
            }
            if (surf_fold[i] > pk_fold) {
                pk_fold = surf_fold[i];
                argmax_fold = i;
            }
        }
        const int fcols = dims.fine();
        const long want = ((long)d0 * Mp + q0) * fcols + s0;
        printf("\ncheck: max|diff| %.4g vs surface max %.4g  -> rel %.3g\n", maxdiff, maxabs,
               maxdiff / maxabs);
        printf("check: mean ref %.6g  fold %.6g  -> rel %.3g\n", sum_ref / surf_ref.size(),
               sum_fold / surf_fold.size(),
               std::fabs(sum_ref - sum_fold) / std::fabs(sum_ref));
        printf("check: argmax ref %ld  fold %ld  planted %ld  %s\n", argmax_ref, argmax_fold, want,
               (argmax_ref == argmax_fold && argmax_ref == want) ? "OK" : "MISMATCH");
        const double snr_ref = pk_ref / (sum_ref / surf_ref.size());
        const double snr_fold = pk_fold / (sum_fold / surf_fold.size());
        printf("check: snr ref %.3f  fold %.3f\n", snr_ref, snr_fold);
    }

#ifdef GNSS_CUDA
    if (gpu) {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
            printf("\nno CUDA device -- skipping GPU\n");
            return 0;
        }
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        // Peak bandwidth via the attribute API: cudaDeviceProp lost memoryClockRate in CUDA 13.
        int mclk_khz = 0, buswidth = 0;
        cudaDeviceGetAttribute(&mclk_khz, cudaDevAttrMemoryClockRate, 0);
        cudaDeviceGetAttribute(&buswidth, cudaDevAttrGlobalMemoryBusWidth, 0);
        printf("\nGPU: %s  sm_%d%d  %d SMs  peak %.0f GB/s\n", prop.name, prop.major, prop.minor,
               prop.multiProcessorCount, 2.0 * mclk_khz * (buswidth / 8) / 1.0e6);

        GpuAgg g;
        g.setup(P, ids, sph, fine_step, nd, Mp);

        std::vector<double> surf_gd, surf_gf;
        double t_gd = 0.0, t_gf = 0.0;
        g.run(false, surf_gd); // discard: first launch pays module load
        g.run(true, surf_gf);
        for (int r = 0; r < reps; ++r)
            t_gd += g.run(false, surf_gd);
        for (int r = 0; r < reps; ++r)
            t_gf += g.run(true, surf_gf);
        t_gd /= reps;
        t_gf /= reps;

        // Bytes the kernel must touch: read nc complex per row, write s_cols float per row
        // (read-modify-write, so the surface counts twice).
        const double bytes = (double)dims.size() / dims.fine() * nc * 8.0
                             + 2.0 * (double)dims.size() * 4.0;
        printf("GPU direct   : %8.4f s   %7.2f GFlop/s   %6.0f GB/s   %6.1fx vs CPU direct\n",
               t_gd, 1e-9 * flops_ref / t_gd, 1e-9 * bytes / t_gd, t_ref / t_gd);
        printf("GPU folded   : %8.4f s   %7.2f GFlop/s*  %6.0f GB/s   %6.1fx vs CPU direct\n",
               t_gf, 1e-9 * flops_ref / t_gf, 1e-9 * bytes / t_gf, t_ref / t_gf);
        printf("               (*equivalent flops -- the fold actually does ~18x fewer)\n");

        auto cmp = [&](const char* tag, const std::vector<double>& s) {
            double maxabs = 0, maxdiff = 0, sum = 0;
            long am = -1;
            double pk = -1;
            for (long i = 0; i < (long)s.size(); ++i) {
                maxabs = std::max(maxabs, std::fabs(surf_ref[i]));
                maxdiff = std::max(maxdiff, std::fabs(surf_ref[i] - s[i]));
                sum += s[i];
                if (s[i] > pk) {
                    pk = s[i];
                    am = i;
                }
            }
            const long want = ((long)d0 * Mp + q0) * dims.fine() + s0;
            printf("check %s: rel %.3g   argmax %ld (planted %ld) %s   snr %.3f\n", tag,
                   maxdiff / maxabs, am, want, am == want ? "OK" : "MISMATCH",
                   pk / (sum / s.size()));
        };
        cmp("gpu-direct", surf_gd);
        cmp("gpu-folded", surf_gf);

        // On-device peak: the whole point is that the 1 GB surface never crosses PCIe.
        double pk = 0, sm = 0;
        long long idx = -1;
        const double t_pk = g.peak_test(dims.size(), &pk, &idx, &sm, std::max(3, reps));
        printf("GPU peak     : %8.4f s   %6.0f GB/s   peak %.6g at %lld   mean %.6g\n", t_pk,
               1e-9 * (double)dims.size() * 4.0 / t_pk, pk, idx, sm / (double)dims.size());
    }
#endif
    return 0;
}
