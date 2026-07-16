// CHORD-shape synthetic bench for the GNSS despread-as-GEMM architecture (2026-07-16).
//
// Build (standalone, no kotekan deps):
//   nvcc -arch=sm_121 -O3 chordShapeBench.cu -lcublas -o chord_shape_bench   (GB10; adjust arch)
//
// MEASURED 2026-07-16 on the GX10/GB10 (see memory chord-gemm-bench / commit message):
//   * wire [t][f][p][d] 4+4b -> planar int8 with ZERO transposes: 214 GB/s (bandwidth-bound).
//     Per-freq D_f is col-major [N x T] with lda=F*P*D, batch stride=N -- cuBLAS int8 NN takes
//     it DIRECTLY. Emitting k-major instead (a transpose) ran at 7 GB/s -- 30x slower.
//   * int8 tensor GEMM = 7x fp32 cgemm at Full shape (51-77 TOPS ~ GB10 peak).
//   * GENERATOR/MAC CROSSOVER: at N=1024 inputs the GEMM costs 2-3x the replica gen -- CHORD's
//     shape is MAC-DOMINATED, so 4+4b/int8 is the right lever THERE, while the same despread at
//     N=1 (the GX10 receiver) is replica-generation-bound and int8 buys ~nothing. Both regimes
//     now measured; neither generalizes to the other.
//
// Question: at CHORD node shapes -- Pathfinder (384 freq x 2 pol x 64 dish = 128 inputs/freq),
// Full (48 freq x 2 pol x 512 dish = 1024 inputs/freq) -- what does the GNSS sidecar cost?
//   A[N][M] = D[N][T] x R[T][M]^H   per (GNSS-covering freq, record)
// with D the 4+4b wire voltages and R the M replica trial columns.
//
// THREE measured pieces:
//  1. UNPACK: 4+4b wire bytes [t][f][p][d] -> planar int8 re/im in GEMM layout. Bandwidth-bound.
//     (KEY LAYOUT FACT: the wire layout per freq IS col-major [N x T] with lda = F*P*D --
//      dish-fastest = n-fastest -- so no transpose exists anywhere in this pipeline.)
//  2. GEMM: complex int8 via 4 real cublasGemmEx int8 (tensor-core IMMA), vs fp32 cgemm ref.
//  3. REPLICA GEN at CHORD shape: M trials x F_gnss chans x T hops of the real chip-gather
//     (the ONLY GNSS-specific cost that is not a GEMM; dish-count-independent by construction).
//
// Everything synthetic (values don't affect speed); accuracy is NOT the question here -- the
// quantized despread is validated bit-identical elsewhere (cudaGnssDespreadTest).
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CK(x) do { auto e=(x); if (e!=cudaSuccess){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(2);} } while(0)
#define CB(x) do { auto s=(x); if (s!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS err %d @%d\n",(int)s,__LINE__);exit(2);} } while(0)

// ---- 1. unpack: wire [t][(f)][p*d] 4+4b -> planar int8 re[k-major TxN per freq], im[...] ----
// Emit TN-friendly layout for the GEMM: per freq, Re/Im as [T x N] col-major-N? We emit
// K-MAJOR (t-fastest within a column of N): re[f][(n)*T + t] -- the "T" op side.
__global__ void k_unpack(const unsigned char* __restrict__ wire, int8_t* __restrict__ re,
                         int8_t* __restrict__ im, int T, int F, int N /*=P*D*/) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)T * F * N;
    if (i >= total) return;
    (void)T; (void)F; (void)N;
    const unsigned char b = wire[i];
    // PLANAR, WIRE ORDER: out index == in index. Zero transpose -- the GEMM absorbs the layout
    // (per-freq matrix D_f(n,t) = plane[t*F*N + f*N + n] IS col-major [N x T] with lda = F*N and
    // batch stride N: dish-fastest on the wire == n-fastest in the matrix. The whole claim.)
    re[i] = (int8_t)(((b >> 4) & 0x0f) - 8);
    im[i] = (int8_t)((b & 0x0f) - 8);
}

// ---- 3. replica gen at CHORD shape: the real chip-gather (from cudaGnssDespreadKernel) ----
struct GJob { double cp0, cps, inv_cps, wc; int code_off, code_len, n_chips; };
__global__ void k_repgen(const int8_t* __restrict__ code, const float2* __restrict__ phiA,
                         const float2* __restrict__ phiB, const GJob* __restrict__ jobs,
                         int Lf, int T, long long n0, int fft_len,
                         int8_t* __restrict__ out_re, int8_t* __restrict__ out_im) {
    const int m = blockIdx.x;  // replica column (PRN x trial)
    const int f = blockIdx.y;  // covering channel
    const GJob job = jobs[m];
    const float2* pA = phiA + (size_t)f * (Lf + 1);
    const float2* pB = phiB + (size_t)f * (Lf + 1);
    const int ks = (int)job.inv_cps;
    const float kf = (float)(job.inv_cps - ks);
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        const long long n_m = n0 + (long long)t * fft_len;
        const double C = job.cp0 + (double)n_m * job.cps;
        const long long chip0 = (long long)floor(C);
        const double phi = C - (double)chip0;
        const double ang = fmod(job.wc * (double)n_m, 2.0 * M_PI);
        float sn, cn; sincosf((float)ang, &sn, &cn);
        const double base = phi * job.inv_cps;
        long long i0 = chip0 % (long long)job.code_len; if (i0 < 0) i0 += job.code_len;
        int idx = (int)i0;
        int prev_khi = -ks + (int)floorf((float)base - kf);
        float fr = (float)base; int khi_base = 0;
        float2 sA = make_float2(0,0), sB = make_float2(0,0);
        for (int d = 0; d < job.n_chips; ++d) {
            const int khi = khi_base + (int)floorf(fr);
            const int klo = prev_khi + 1;
            prev_khi = khi; fr += kf; khi_base += ks;
            int kh = khi, kl = klo;
            if (kl < 0) kl = 0;
            if (kh > Lf - 1) kh = Lf - 1;
            if (kh >= kl) {
                const float cv = (float)code[job.code_off + idx];
                sA.x += cv*(pA[kh+1].x - pA[kl].x); sA.y += cv*(pA[kh+1].y - pA[kl].y);
                sB.x += cv*(pB[kh+1].x - pB[kl].x); sB.y += cv*(pB[kh+1].y - pB[kl].y);
            }
            if (--idx < 0) idx += job.code_len;
        }
        const float rr = 0.5f*((cn*sA.x - sn*sA.y) + (cn*sB.x + sn*sB.y));
        const float ri = 0.5f*((cn*sA.y + sn*sA.x) + (-sn*sB.x + cn*sB.y)); // ~conj combine
        // quantize replica to int8 (range use is a design knob; speed unaffected)
        const size_t o = ((size_t)f * gridDim.x + m) * T + t;
        out_re[o] = (int8_t)max(-127.f, min(127.f, rr * 40.f));
        out_im[o] = (int8_t)max(-127.f, min(127.f, ri * 40.f));
    }
}

static double bench_ms(int iters, void (*fn)(void*), void* arg) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    fn(arg); CK(cudaDeviceSynchronize()); // warm
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) fn(arg);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

struct Shape { const char* name; int F_node, D, P, F_gnss, M, T; };

int main() {
    cublasHandle_t h; CB(cublasCreate(&h));
    // shapes: {node} x {covering-freq count} x {replica columns} x {record length}
    std::vector<Shape> shapes = {
        {"Pathfinder", 384, 64, 2, 4, 48, 256},
        {"Pathfinder", 384, 64, 2, 48, 48, 256},
        {"Pathfinder", 384, 64, 2, 48, 384, 256},
        {"Full",       48, 512, 2, 4, 48, 256},
        {"Full",       48, 512, 2, 48, 48, 256},
        {"Full",       48, 512, 2, 48, 384, 256},
        {"Full",       48, 512, 2, 48, 384, 2048},
        {"Full",       48, 512, 2, 4, 384, 2048},
    };
    printf("GNSS sidecar at CHORD node shapes: unpack(all F) + 4x int8 GEMM(F_gnss only) + repgen\n");
    printf("%-11s %5s %4s %6s %4s %5s | %9s %9s %9s %9s | %8s %7s\n",
           "node", "F", "D", "Fgnss", "M", "T", "unpack", "gemm_i8", "gemm_f32", "repgen",
           "i8 TOPS", "GB/s");
    for (const auto& s : shapes) {
        const int N = s.D * s.P;
        const long long wire_bytes = (long long)s.T * s.F_node * N; // one record across ALL freqs
        // buffers
        unsigned char* d_wire; int8_t *d_re, *d_im, *d_rre, *d_rim; int32_t* d_c;
        float2 *d_cf, *d_df, *d_rf;
        CK(cudaMalloc(&d_wire, wire_bytes));
        CK(cudaMemset(d_wire, 0x88, wire_bytes));
        CK(cudaMalloc(&d_re, wire_bytes)); CK(cudaMalloc(&d_im, wire_bytes));
        CK(cudaMalloc(&d_rre, (size_t)s.F_gnss * s.M * s.T));
        CK(cudaMalloc(&d_rim, (size_t)s.F_gnss * s.M * s.T));
        CK(cudaMalloc(&d_c, (size_t)s.F_gnss * N * s.M * 4 * sizeof(int32_t)));
        CK(cudaMalloc(&d_cf, (size_t)s.F_gnss * N * s.M * sizeof(float2)));
        CK(cudaMalloc(&d_df, (size_t)s.F_gnss * N * s.T * sizeof(float2)));
        CK(cudaMalloc(&d_rf, (size_t)s.F_gnss * s.M * s.T * sizeof(float2)));
        CK(cudaMemset(d_rre, 1, (size_t)s.F_gnss * s.M * s.T));
        CK(cudaMemset(d_rim, 1, (size_t)s.F_gnss * s.M * s.T));

        // 1. unpack (ALL node freqs -- the wire arrives whole)
        struct UA { const unsigned char* w; int8_t *re, *im; int T, F, N; long long tot; } ua{
            d_wire, d_re, d_im, s.T, s.F_node, N, wire_bytes};
        auto unpack_fn = [](void* p) {
            auto* a = (UA*)p;
            const int blk = 256;
            k_unpack<<<(unsigned)((a->tot + blk - 1) / blk), blk>>>(a->w, a->re, a->im, a->T,
                                                                    a->F, a->N);
        };
        const double t_unpack = bench_ms(20, unpack_fn, &ua);

        // 2a. int8 GEMM: 4 real gemms per covering freq, strided-batched over F_gnss.
        //     C[N x M] = D[N x T] x R[T x M]: cublas col-major "TN": opA=T on the K-major D.
        struct GA { cublasHandle_t h; const int8_t *re, *im, *rre, *rim; int32_t* c;
                    int N, M, T, F, ldaFN; } ga{h, d_re, d_im, d_rre, d_rim, d_c, N, s.M, s.T,
                                                s.F_gnss, s.F_node * N};
        auto gemm_i8 = [](void* p) {
            auto* a = (GA*)p;
            const int32_t one = 1, zero = 0;
            const int8_t* As[4] = {a->re, a->im, a->im, a->re};
            const int8_t* Bs[4] = {a->rre, a->rim, a->rre, a->rim};
            for (int q = 0; q < 4; ++q) {
                auto st = cublasGemmStridedBatchedEx(a->h, CUBLAS_OP_N, CUBLAS_OP_N, a->N, a->M,
                    a->T,
                    &one, As[q], CUDA_R_8I, a->ldaFN, (long long)a->N /*batch stride = f slice*/,
                          Bs[q], CUDA_R_8I, a->T, (long long)a->M * a->T,
                    &zero, a->c + (size_t)q * a->F * a->N * a->M, CUDA_R_32I, a->N,
                    (long long)a->N * a->M, a->F, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
                if (st != CUBLAS_STATUS_SUCCESS) { printf("int8 NN unsupported (%d)\n", (int)st); exit(3); }
            }
        };
        const double t_i8 = bench_ms(20, gemm_i8, &ga);

        // 2b. fp32 complex reference (cgemm strided batched)
        struct GF { cublasHandle_t h; const float2 *d, *r; float2* c; int N, M, T, F; } gf{
            h, d_df, d_rf, d_cf, N, s.M, s.T, s.F_gnss};
        auto gemm_f32 = [](void* p) {
            auto* a = (GF*)p;
            const cuComplex one = {1, 0}, zero = {0, 0};
            CB(cublasCgemmStridedBatched(a->h, CUBLAS_OP_T, CUBLAS_OP_N, a->N, a->M, a->T, &one,
                (const cuComplex*)a->d, a->T, (long long)a->N * a->T,
                (const cuComplex*)a->r, a->T, (long long)a->M * a->T, &zero,
                (cuComplex*)a->c, a->N, (long long)a->N * a->M, a->F));
        };
        const double t_f32 = bench_ms(20, gemm_f32, &gf);

        // 3. replica gen at shape (M x F_gnss x T), real gather geometry (n_chips 5, Lf 80)
        const int Lf = 80, n_chips = 5, code_len = 10230;
        int8_t* d_code; float2 *d_pA, *d_pB; GJob* d_j;
        CK(cudaMalloc(&d_code, code_len)); CK(cudaMemset(d_code, 1, code_len));
        CK(cudaMalloc(&d_pA, (size_t)s.F_gnss * (Lf + 1) * sizeof(float2)));
        CK(cudaMalloc(&d_pB, (size_t)s.F_gnss * (Lf + 1) * sizeof(float2)));
        CK(cudaMemset(d_pA, 0, (size_t)s.F_gnss * (Lf + 1) * sizeof(float2)));
        CK(cudaMemset(d_pB, 0, (size_t)s.F_gnss * (Lf + 1) * sizeof(float2)));
        std::vector<GJob> hj(s.M);
        for (int m = 0; m < s.M; ++m)
            hj[m] = {100.0 + m, 0.0511575, 19.5475, 1e-3, 0, code_len, n_chips};
        CK(cudaMalloc(&d_j, s.M * sizeof(GJob)));
        CK(cudaMemcpy(d_j, hj.data(), s.M * sizeof(GJob), cudaMemcpyHostToDevice));
        struct RA { const int8_t* code; const float2 *pA, *pB; const GJob* j; int Lf, T, fft, M, F;
                    int8_t *ore, *oim; } ra{d_code, d_pA, d_pB, d_j, Lf, s.T, 20, s.M, s.F_gnss,
                                            d_rre, d_rim};
        auto rep_fn = [](void* p) {
            auto* a = (RA*)p;
            dim3 g(a->M, a->F);
            k_repgen<<<g, 256>>>(a->code, a->pA, a->pB, a->j, a->Lf, a->T, 1000000LL, a->fft,
                                 a->ore, a->oim);
        };
        const double t_rep = bench_ms(20, rep_fn, &ra);

        // derived: int8 TOPS (4 gemms x 2 ops/MAC), unpack GB/s
        const double macs = 4.0 * (double)N * s.M * s.T * s.F_gnss;
        const double tops = macs * 2.0 / (t_i8 * 1e-3) / 1e12;
        const double gbs = (double)wire_bytes * 3.0 / (t_unpack * 1e-3) / 1e9; // r+w2
        printf("%-11s %5d %4d %6d %4d %5d | %7.3fms %7.3fms %7.3fms %7.3fms | %8.1f %7.0f\n",
               s.name, s.F_node, s.D, s.F_gnss, s.M, s.T, t_unpack, t_i8, t_f32, t_rep, tops,
               gbs);
        cudaFree(d_wire); cudaFree(d_re); cudaFree(d_im); cudaFree(d_rre); cudaFree(d_rim);
        cudaFree(d_c); cudaFree(d_cf); cudaFree(d_df); cudaFree(d_rf);
        cudaFree(d_code); cudaFree(d_pA); cudaFree(d_pB); cudaFree(d_j);
    }
    printf("\nper-record costs above; real-time margin = record_period / (unpack+gemm+repgen)\n");
    printf("(record_period = T / channel_width; e.g. T=256 @ 390 kHz chan -> 0.66 ms)\n");
    cublasDestroy(h);
    return 0;
}
