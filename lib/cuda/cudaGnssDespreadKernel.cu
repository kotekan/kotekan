#include "cudaGnssDespreadKernel.hpp"

#include "cudaGnssReplicaDevice.cuh"

#include <cuda_runtime.h>
#include <math.h>

// GPU port of gnss::ChannelizedReplicaBank::hoprate_stream + gnss::channelized_despread,
// FUSED: each thread generates one (channel, hop) replica sample from the precomputed
// cumulative filter tables (PhiA/PhiB, Doppler-bucketed, built on CPU) and immediately
// multiplies it against the data -- the correlate-at-data despread as one batched kernel.
//
// Geometry (one launch = one record window of one signal band):
//   grid.x  = spec index b -- ONE PRN's WHOLE CORRELATOR QUAD (E, P, L, P_HEAD)
//   grid.y  = covering-channel index ci
//   block.x = hops (m), one thread per hop (125 @ L1/5MSPS; <=1024 always for our bands)
// Each block computes partial sums over its hops via shared-memory reduction -> rows 4b+0..4b+3
// of corr/energy; the tiny cross-channel sum happens on the host (or a follow-up 1-block kernel
// when batches grow).
//
// WHY THE QUAD IS ONE BLOCK AND NOT FOUR (2026-07-16): E/P/L differ ONLY in code phase -- they
// share a Doppler, hence one carrier phasor, and they hit the same data sample. Run as four
// independent jobs, every one of them re-derived the carrier (a double fmod over a ~1e10-radian
// argument + a sincosf) and re-loaded the data, for a per-hop cost of 4x on both. Fused, the
// carrier and the load happen ONCE per (PRN, channel, hop) and only the chip gather -- the part
// that genuinely differs -- runs three times. P_HEAD rides along for free: it is the prompt's own
// MAC gated on `mh < m_head`, and the MAC measured at 0% of the runtime (see the gather note).
// Measured on the live job mix: 1.14x (1000-hop L1) to 1.22x (10000-hop B1C) kernel-only, 1.18x
// end-to-end through enqueue_batch_device (7244 -> 8568 rec/s, 32 PRN). ⚠️ The win is ALL from
// sharing the carrier/load: an A/B that only deleted the redundant P_HEAD job -- keeping E/P/L as
// three -- measured 0.98-1.00x, because the half-record it saves is paid straight back by the
// extra reduction pass that then has to run inside the prompt's block.
//
// ⚠️ THE WIN IS A FUNCTION OF HOPS PER THREAD -- it inverts below ~2. Fusing costs registers
// (56 -> 78, no spills), which at 128-256 threads/block is roughly a third of the occupancy.
// Where each thread handles ~1 hop the kernel is LATENCY-bound (one dependent fmod -> sincosf ->
// gather chain per thread, nothing to interleave), occupancy is what hides that latency, and the
// fused kernel LOSES ~10%. Where a thread walks several hops it is THROUGHPUT-bound, the carrier
// and load savings dominate, and it wins. Every deployed band is at 3.9-39 hops/thread (20 MSPS
// N=10: 1000-hop GPS, 4000-hop E1C, 10000-hop B1C; 5 MSPS L2C: 5000-hop) -- see the bench note in
// cudaGnssDespreadTest.cpp, which is why that bench runs the wide front end and not the 125-hop
// geometry the correctness check uses.
//
// Numerics (MIXED PRECISION -- the GB10's FP64 throughput is a small fraction of FP32, and the
// FP64 chip-gather was the measured ceiling at ~75k PRN-despreads/s):
//   double, mandatory: the absolute-anchored code phase C = cp0 + n*cps (n ~1e10 exceeds
//     float's 2^24 integer range), its floor/frac reduction, the tap-boundary indices
//     (floor((phi+d)*inv_cps) -- a boundary flip moves a prototype tap between adjacent chips,
//     so it is kept bit-comparable to the CPU division), and the carrier range reduction
//     fmod(wc*n, 2pi) (~1e10 rad).
//   float, everything per-chip after reduction: Phi tables + telescoped gather (values O(1);
//     cancellation error ~2e-6 relative per chip-delta), carrier phasor (sincosf on the reduced
//     angle, 1e-7 rad), replica sample, data MAC, energy.
//   double again for the cross-hop block reduction (125 signed terms; keeps the sums
//     well-conditioned and the outputs' types unchanged).
// Measured against the all-double CPU reference: ~1e-6 relative (gate 1e-5).

namespace {

// The replica primitives (cmulf / chip_gather / load_v) now live in a shared device header
// so the fused despread, the peel, and the CHORD standalone waveform generator all call the
// SAME gather -- see cudaGnssReplicaDevice.cuh for why that has to be true.
using gnss_cuda::chip_gather;
using gnss_cuda::cmulf;
using gnss_cuda::load_v;

/// ABL: kernel-split ablation, a BENCH-ONLY template parameter. Production instantiates
/// ABL_NONE and is unaffected -- like XC, everything else compiles away.
///
/// Exists because the tracker kernel's cost had to be attributed rather than guessed, and the
/// attribution is a planning input: the correlation MAC is what eventually folds into CHORD's
/// N^2, while the replica SYNTHESIS never can (N^2 knows nothing about GNSS codes). The kernel
/// header records a 2026-07-16 ablation claiming the MAC is 0% of the runtime; this is how that
/// gets re-confirmed at CHORD geometry instead of inherited.
///
/// NO_MAC keeps a single add so the gather cannot be dead-code-eliminated -- measure the
/// difference from ABL_NONE to get the MAC's cost. NO_SYN replaces the gather with a constant,
/// keeping the carrier and the MAC -- the difference gives synthesis.
enum AblMode { ABL_NONE = 0, ABL_NO_MAC = 1, ABL_NO_SYN = 2 };

template <typename T, bool XC, int ABL = ABL_NONE>
__global__ void gnss_despread_kernel(const T* __restrict__ data,          // [nchan][n_hops]
                                     const float* __restrict__ chan_scale, // [nchan], 4+4b only
                                     const int8_t* __restrict__ code, // [batch-shared code table]
                                     const gnss_cuda::DespreadJob* __restrict__ jobs, // [n_spec]
                                     gnss_cuda::DespreadParams p,
                                     double2* __restrict__ corr,   // [4*n_spec][nchan]
                                     double* __restrict__ energy,  // [4*n_spec][nchan]
                                     double2* __restrict__ xcorr) { // [2*n_spec][nchan], XC only
    const int b = blockIdx.x;  // spec: one PRN's E/P/L/P_HEAD quad
    const int ci = blockIdx.y; // covering channel
    const int m = threadIdx.x; // hop lane: handles hops m, m+blockDim, ... (grid-stride over
                               // the record, so records LONGER than the block work -- 1000-hop
                               // L1 / 4000-hop E1C records at the 20 MSPS wide front end)
    const gnss_cuda::DespreadJob job = jobs[b];

    // acc/e rows: 0 = Early, 1 = Prompt, 2 = Late, 3 = P_HEAD (the prompt over [0, m_head)).
    double2 acc[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    double e[4] = {0.0, 0.0, 0.0, 0.0};
    // Reference cross-correlations for the analytic peel add-back (XC only; when XC is false
    // these and their reduction compile away completely).
    //   0 = <R_P, R_E>          1 = <R_P, R_L>            (whole window)
    //   2 = <R_P, R_E>|head     3 = <R_P, R_L>|head       (hops [0, m_head) only)
    // The head-restricted pair exists because the peel subtracts a DIFFERENT gain either side of
    // the code-period boundary (the overlay/nav sign flips there). The add-back is then
    //   V_full[k] = V'[k] + a_head*<R_P 1_head, R_k> + a_tail*<R_P 1_tail, R_k>,
    // and the tail term is (whole - head). With one gain the two collapse to the whole-window
    // value, so this only costs anything in windows that actually straddle a flip -- and in those
    // windows it is the difference between an exact add-back and a biased DLL. Like P_HEAD, it
    // rides along on the replicas already in registers.
    double2 xc[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    const bool covered = (job.chan_mask >> ci) & 1ULL; // channel in this PRN's covering set?

    // Hoisted out of the hop loop: invariant in BOTH mh and the trial index.
    const int ks = (int)job.inv_cps;
    const float kf = (float)(job.inv_cps - ks);
    const float2* phiA = job.phiA + (size_t)ci * (p.Lf + 1);
    const float2* phiB = job.phiB + (size_t)ci * (p.Lf + 1);

    for (int mh = m; covered && mh < p.n_hops; mh += blockDim.x) {
        // Per-hop PROMPT code phase at the hop's reference sample (absolute anchoring): DOUBLE.
        // Early/Late are this +- ds, a double add each -- NOT three independent cp0 + n_m*cps
        // products (n_m ~1e10, so that multiply is the expensive fp64 one). The E/L rounding
        // differs from the unfused form by ~1 ulp of C (~1e-7 chips = 0.03 mm), far below the
        // gather's own float error.
        const long long n_m = p.n0 + (long long)mh * p.fft_len;
        const double dn = (double)((long long)mh * (long long)p.fft_len);
        // CODE PHASE FROM A REFERENCE (task #54). This was cp0 + n_m*cps on the ABSOLUTE sample:
        // n_m*cps reaches 6.1e12 chips at CHORD uptime, where a double's ulp is ~1e-3 chips, and it
        // re-rolls every hop. Measured GPU-vs-CPU replica error on the PROMPT row, carrier already
        // referenced via ang0: 7.9e-08 / 8.2e-07 / 9.6e-04 / 3.6e-02 at 0.007 / 0.068 / 0.678 / 6.8
        // days. Zero at prototype scale, 3.6% at the real thing -- and WITHIN a record, so no
        // per-record correction can reach it. cp_ref is the prompt code phase at p.n0, reduced mod
        // code_len on the host in long double; cps now multiplies only the intra-record offset.
        const double C_P = job.cp_ref + dn * job.cps;

        // Carrier phasor pa = e^{i wc n_m}: range-reduce in DOUBLE, trig in float. ONE per hop
        // for the whole quad -- E/P/L share a Doppler, and P_HEAD *is* the prompt.
        // PHASE FROM job.ang0, NOT FROM n_m -- see the long note in cudaGnssChordDespread.cu
        // (task #52). The two-product this replaces was exact for the product and still left
        // wc's own rounding times the absolute sample: a per-record constant that re-rolls with
        // every Doppler re-propagation, 0.217 rad at 6.8 days of uptime, and per-record
        // constants are precisely what the cross-record estimators integrate.
        double ang;   // A/B arm, task #52/#55 -- see DespreadParams::carrier_phase_from_ref
        if (p.carrier_phase_from_ref) {
            ang = job.ang0 + fmod(job.wc * dn, 2.0 * M_PI);
        } else {
            const double pr_ = job.wc * (double)n_m;
            const double er_ = fma(job.wc, (double)n_m, -pr_);
            ang = fmod(pr_, 2.0 * M_PI) + er_;
        }
        float sn, cn;
        sincosf((float)ang, &sn, &cn);
        const float2 pa = make_float2(cn, sn);
        const float2 pb = make_float2(cn, -sn);

        // ONE load for the quad (fp32 complex, or a 4+4b byte unpacked with this channel's scale)
        const float2 dd = load_v(data, (size_t)ci * p.data_stride + mh, chan_scale, ci);
        const bool in_head = (mh < job.m_head);

        // TRIAL ORDER {P, E, L}, not {E, P, L}. The prompt replica has to be in hand when the
        // Early/Late ones are formed, so <R_P, R_E> / <R_P, R_L> can be accumulated without a
        // second gather. Accumulator indices are unchanged, so every output row keeps its
        // meaning -- only the order the three are computed in moved.
        //
        // MATH no-op, NOT a bitwise one. acc[t]/e[t] sum the same terms in the same per-hop
        // order regardless of XC, so corr/energy are unchanged to the LAST BIT of the algebra
        // -- but not of the float codegen: with XC on, `r_P` must stay live and the xc MACs
        // raise register pressure, so nvcc makes different FMA-contraction/scheduling choices
        // on the identical `dd.x*r.x + dd.y*r.y` float MACs. That shifts corr/energy by ~1 ULP
        // (~1e-7 rel, data-dependent) between the XC=off and XC=on instantiations. The test
        // gates this at the suite's float floor (~1e-6), NOT bit-identity -- a real leak from
        // the xc path into acc/e (there is none: grep this function, xc[] never writes acc/e)
        // would be orders larger.
        float2 r_P = make_float2(0.f, 0.f);
        // ONE walk of the chip loop for all three trials instead of three. A block re-reads its
        // Phi slice once per trial otherwise, and the three trials' taps at a given chip are
        // NEIGHBOURS (base is a fractional part scaled by inv_cps, so the trial offset moves the
        // tap inside a ~625-entry window, not to another part of the slice) -- so one pass serves
        // all three off the same L1 lines. Same three code phases, same {P,E,L} order, and the
        // loop body below is textually unchanged: only the gather moved out of it.
        float2 g3A[3], g3B[3];
        if (ABL != ABL_NO_SYN) {
            const double Cs[3] = {C_P, C_P - job.ds, C_P + job.ds};
            gnss_cuda::chip_gather3(job.inv_cps, job.code_offset, job.code_len, job.n_chips, p.Lf,
                                    code, phiA, phiB, ks, kf, Cs, g3A, g3B);
        }
#pragma unroll
        for (int tt = 0; tt < 3; ++tt) {
            const int t = (tt == 0) ? 1 : (tt == 1) ? 0 : 2; // -> cp0 / cp0-ds / cp0+ds
            float2 sA, sB;
            if (ABL == ABL_NO_SYN) {
                // Constant replica: the gather disappears, the carrier apply and MAC remain.
                sA = make_float2(1.0f, 0.0f);
                sB = make_float2(1.0f, 0.0f);
            } else {
                sA = g3A[tt];
                sB = g3B[tt];
            }
            // Replica channel sample r = 0.5 (pa sA + conj(pa) sB); then the despread MAC:
            // acc += data * conj(r), e += |r|^2 -- float compute, widened once for the reduction.
            const float2 t1 = cmulf(pa, sA);
            const float2 t2 = cmulf(pb, sB);
            const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
            if (ABL == ABL_NO_MAC) {
                // ONE add, purely so the gather above stays live. Anything cheaper and nvcc
                // eliminates the synthesis entirely and the measurement means nothing.
                acc[t].x += (double)r.x;
                continue;
            }
            const double re = (double)(dd.x * r.x + dd.y * r.y); // Re(d * conj(r))
            const double im = (double)(dd.y * r.x - dd.x * r.y); // Im(d * conj(r))
            const double ee = (double)(r.x * r.x + r.y * r.y);
            acc[t].x += re;
            acc[t].y += im;
            e[t] += ee;
            // P_HEAD: the prompt's own MAC, gated on the hop. The replica is already in hand, so
            // the head segment costs one predicated add -- no second gather, no second carrier.
            if (t == 1 && in_head) {
                acc[3].x += re;
                acc[3].y += im;
                e[3] += ee;
            }
            if (XC) {
                // t == 1 is stashed for the other two; its own value <R_P,R_P> = |R_P|^2 is
                // energy row 1 already, so it is not stored again.
                if (t == 1) {
                    r_P = r;
                } else {
                    const int x = (t == 0) ? 0 : 1; // 0 = Early, 1 = Late
                    // <R_P, R_t> = sum_hop r_P conj(r_t) -- the SAME <a,b> = sum a conj(b) the
                    // despread uses for <X, R_k>, and in the SAME order as the add-back formula
                    // V[k] = V'[k] + a*<R_P, R_k>. Accumulating <R_t, R_P> instead would be the
                    // conjugate and every consumer would owe a conj() nobody remembers.
                    const double xre = (double)(r_P.x * r.x + r_P.y * r.y);
                    const double xim = (double)(r_P.y * r.x - r_P.x * r.y);
                    xc[x].x += xre;
                    xc[x].y += xim;
                    if (in_head) { // head-restricted twin, rows 2/3 -- one predicated add
                        xc[x + 2].x += xre;
                        xc[x + 2].y += xim;
                    }
                }
            }
        }
    }

    // Block reduction over hop lanes (each lane already summed its strided hops above), run as
    // four sequential passes over ONE set of shared arrays: four at once would be 24 KB/block.
    __shared__ double sh_re[256], sh_im[256], sh_e[256];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // m_head == 0 (no code-period boundary in this window -- e.g. the plain 3-trial contract
        // of despread_batch) leaves row 3 identically zero: write it and skip the tree. The test
        // is uniform across the block, and the row must still be WRITTEN because the assembler
        // reads all four unconditionally.
        if (j == 3 && job.m_head == 0) {
            if (m == 0) {
                corr[(size_t)(p.out_rows_spec * b + 3) * gridDim.y + ci] = make_double2(0.0, 0.0);
                energy[(size_t)(p.out_rows_spec * b + 3) * gridDim.y + ci] = 0.0;
            }
            break;
        }
        __syncthreads(); // the previous pass's reads must finish before we overwrite
        sh_re[m] = acc[j].x;
        sh_im[m] = acc[j].y;
        sh_e[m] = e[j];
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (m < off) {
                sh_re[m] += sh_re[m + off];
                sh_im[m] += sh_im[m + off];
                sh_e[m] += sh_e[m + off];
            }
            __syncthreads();
        }
        if (m == 0) {
            corr[(size_t)(p.out_rows_spec * b + j) * gridDim.y + ci] = make_double2(sh_re[0], sh_im[0]);
            energy[(size_t)(p.out_rows_spec * b + j) * gridDim.y + ci] = sh_e[0];
        }
    }

    // Reference cross-correlations, same tree over the same shared arrays (sh_e unused here).
    if (XC) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            __syncthreads();
            sh_re[m] = xc[j].x;
            sh_im[m] = xc[j].y;
            __syncthreads();
            for (int off = blockDim.x / 2; off > 0; off >>= 1) {
                if (m < off) {
                    sh_re[m] += sh_re[m + off];
                    sh_im[m] += sh_im[m + off];
                }
                __syncthreads();
            }
            if (m == 0)
                xcorr[(size_t)(4 * b + j) * gridDim.y + ci] = make_double2(sh_re[0], sh_im[0]);
        }
    }
}

} // namespace

namespace gnss_cuda {

namespace {
// Block = min(pow2 >= n_hops, 256); longer records grid-stride inside the kernel (the
// 20 MSPS wide front end runs 1000-hop L1 / 4000-hop E1C records).
inline int despread_block(int n_hops) {
    int block = 1;
    while (block < n_hops && block < 256)
        block <<= 1;
    return block;
}
} // namespace

cudaError_t launch_despread(const float2* data, const int8_t* code, const DespreadJob* jobs,
                            int n_spec, int n_chan, const DespreadParams& p, double2* corr,
                            double* energy, cudaStream_t stream, double2* xcorr) {
    dim3 grid(n_spec, n_chan); // one block per (PRN quad, channel) -- 4 output rows each
    // XC is a TEMPLATE parameter, not a runtime flag: with it false the cross-term accumulators
    // and their reduction disappear at compile time, so the chains that do not peel keep their
    // register count (the fusion win is register-sensitive -- see the hops-per-thread note above).
    if (xcorr != nullptr)
        gnss_despread_kernel<float2, true><<<grid, despread_block(p.n_hops), 0, stream>>>(
            data, nullptr, code, jobs, p, corr, energy, xcorr);
    else
        gnss_despread_kernel<float2, false><<<grid, despread_block(p.n_hops), 0, stream>>>(
            data, nullptr, code, jobs, p, corr, energy, nullptr);
    return cudaGetLastError();
}

/// BENCH ONLY -- kernel-split ablation. `abl` selects AblMode; 0 is exactly the production
/// path. Never called from the pipeline: the production launchers above instantiate ABL_NONE
/// and are unchanged, so this costs the deployed kernel nothing but two extra instantiations
/// in the object file.
cudaError_t launch_despread_abl(const float2* data, const int8_t* code, const DespreadJob* jobs,
                                int n_spec, int n_chan, const DespreadParams& p, double2* corr,
                                double* energy, cudaStream_t stream, int abl) {
    dim3 grid(n_spec, n_chan);
    const int blk = despread_block(p.n_hops);
    if (abl == ABL_NO_MAC)
        gnss_despread_kernel<float2, false, ABL_NO_MAC>
            <<<grid, blk, 0, stream>>>(data, nullptr, code, jobs, p, corr, energy, nullptr);
    else if (abl == ABL_NO_SYN)
        gnss_despread_kernel<float2, false, ABL_NO_SYN>
            <<<grid, blk, 0, stream>>>(data, nullptr, code, jobs, p, corr, energy, nullptr);
    else
        gnss_despread_kernel<float2, false, ABL_NONE>
            <<<grid, blk, 0, stream>>>(data, nullptr, code, jobs, p, corr, energy, nullptr);
    return cudaGetLastError();
}

cudaError_t launch_despread_q(const unsigned char* data, const float* chan_scale,
                              const int8_t* code, const DespreadJob* jobs, int n_spec, int n_chan,
                              const DespreadParams& p, double2* corr, double* energy,
                              cudaStream_t stream, double2* xcorr) {
    dim3 grid(n_spec, n_chan);
    if (xcorr != nullptr)
        gnss_despread_kernel<unsigned char, true><<<grid, despread_block(p.n_hops), 0, stream>>>(
            data, chan_scale, code, jobs, p, corr, energy, xcorr);
    else
        gnss_despread_kernel<unsigned char, false><<<grid, despread_block(p.n_hops), 0, stream>>>(
            data, chan_scale, code, jobs, p, corr, energy, nullptr);
    return cudaGetLastError();
}

} // namespace gnss_cuda

namespace {

/**
 * THE VOLTAGE PEEL. One thread owns one (channel, hop) output sample: load the voltage once,
 * subtract every seeded PRN's prompt replica, store the residual once.
 *
 * Grid is (hop tile, channel) -- the TRANSPOSE of the despread's (PRN, channel), because there is
 * no reduction here: the output is a voltage, not a correlation. Every PRN is handled inside the
 * thread, which is what makes the accumulation deterministic and atomic-free. That is only legal
 * because the gains are FEED-FORWARD: they do not depend on the data, so the subtractions commute
 * and there is no successive-peel ordering to respect (unlike the CPU GnssVoltagePeel, whose
 * in-window gain fit must see the other sats already removed).
 *
 * The replica is generated by the SAME chip_gather as the despread, from the same Phi tables and
 * the same double-precision absolute code phase. That is a correctness requirement, not tidiness:
 * the analytic add-back downstream assumes the subtracted waveform and the correlated reference
 * are the same object.
 */
/// Hops each thread owns. ⚠️ ONE IS OPTIMAL HERE -- the OPPOSITE of the despread kernel above,
/// and the hops-per-thread note at the top of this file does NOT transfer. Measured 2026-07-24,
/// 32 PRN x 10 chan x 1000 hops, as a multiple of the un-peeled despread cost:
///     H = 1 -> 2.08x     H = 2 -> 2.35x     H = 4 -> 2.73x     H = 8 -> 3.58x
/// Strictly monotonic, so this was swept rather than reasoned about -- the first attempt shipped
/// H = 4 on the (wrong) theory that the peel was latency-bound the way the despread is.
///
/// WHY IT INVERTS. The despread gains from extra hops because it has something to AMORTIZE WITHIN
/// a hop: one carrier phasor and one voltage load feed THREE chip gathers (E/P/L). The peel has
/// exactly one gather per hop and nothing to share it with, so extra hops buy no reuse -- they
/// only add live state (v[H] plus the unrolled gather chains), and the register pressure costs
/// occupancy, which is what was hiding the fp64 fmod latency to begin with.
///
/// Left as a knob because it is one line and the next geometry (a wider band, a longer record, a
/// GPU with a different register file) could move the optimum. Re-sweep, do not assume.
#define GNSS_PEEL_HOPS_PER_THREAD 1

template <typename T>
__global__ void gnss_peel_kernel(const T* __restrict__ data,           // [nchan][data_stride]
                                 const float* __restrict__ chan_scale, // [nchan], 4+4b only
                                 const int8_t* __restrict__ code,
                                 const gnss_cuda::PeelJob* __restrict__ jobs, // [n_job]
                                 int n_job, gnss_cuda::DespreadParams p,
                                 float2* __restrict__ resid) { // [nchan][n_hops], PACKED
    constexpr int H = GNSS_PEEL_HOPS_PER_THREAD;
    const int ci = blockIdx.y;
    // Thread's hops are STRIDED by blockDim.x, not consecutive, so each of the H loads and stores
    // stays coalesced across the warp.
    const int base = blockIdx.x * blockDim.x * H + threadIdx.x;

    float2 v[H];
    bool act[H];
#pragma unroll
    for (int t = 0; t < H; ++t) {
        const int mh = base + t * (int)blockDim.x;
        act[t] = (mh < p.n_hops);
        if (act[t])
            v[t] = load_v(data, (size_t)ci * p.data_stride + mh, chan_scale, ci);
    }

    for (int b = 0; b < n_job; ++b) {
        const gnss_cuda::PeelJob job = jobs[b];
        if (!((job.chan_mask >> ci) & 1ULL))
            continue; // carrier not in this channel: nothing of this PRN to remove here
        // Per-segment gain, hoisted: m_head is the code-period boundary the despread's P_HEAD row
        // uses -- shared, never re-derived. Below it the overlay/nav symbol has one sign, at and
        // above it possibly the other; one gain across the pair would model a sign flip with a
        // constant and remove almost nothing.
        const float2 ah = job.a_head[ci];
        const float2 at = job.a_tail[ci];
        if (ah.x == 0.f && ah.y == 0.f && at.x == 0.f && at.y == 0.f)
            continue; // not converged / gated off for this PRN-channel: pass the voltage through
        const int ks = (int)job.inv_cps;
        const float kf = (float)(job.inv_cps - ks);
        const float2* const phiA = job.phiA + (size_t)ci * (p.Lf + 1);
        const float2* const phiB = job.phiB + (size_t)ci * (p.Lf + 1);

#pragma unroll
        for (int t = 0; t < H; ++t) {
            if (!act[t])
                continue;
            const int mh = base + t * (int)blockDim.x;
            const long long n_m = p.n0 + (long long)mh * p.fft_len;
            const double dn = (double)((long long)mh * (long long)p.fft_len);
            // CODE PHASE FROM A REFERENCE (task #54). This was cp0 + n_m*cps on the ABSOLUTE sample:
            // n_m*cps reaches 6.1e12 chips at CHORD uptime, where a double's ulp is ~1e-3 chips, and it
            // re-rolls every hop. Measured GPU-vs-CPU replica error on the PROMPT row, carrier already
            // referenced via ang0: 7.9e-08 / 8.2e-07 / 9.6e-04 / 3.6e-02 at 0.007 / 0.068 / 0.678 / 6.8
            // days. Zero at prototype scale, 3.6% at the real thing -- and WITHIN a record, so no
            // per-record correction can reach it. cp_ref is the prompt code phase at p.n0, reduced mod
            // code_len on the host in long double; cps now multiplies only the intra-record offset.
            const double C_P = job.cp_ref + dn * job.cps;
            // Phase from job.ang0 -- see cudaGnssChordDespread.cu (task #52).
            double ang;   // A/B arm, task #52/#55 (peel: MUST match the despread's arm)
            if (p.carrier_phase_from_ref) {
                ang = job.ang0 + fmod(job.wc * dn, 2.0 * M_PI);
            } else {
                const double pr_ = job.wc * (double)n_m;
                const double er_ = fma(job.wc, (double)n_m, -pr_);
                ang = fmod(pr_, 2.0 * M_PI) + er_;
            }
            float sn, cn;
            sincosf((float)ang, &sn, &cn);
            const float2 pa = make_float2(cn, sn);
            const float2 pb = make_float2(cn, -sn);
            float2 sA, sB;
            chip_gather(job.inv_cps, job.code_offset, job.code_len, job.n_chips, p.Lf, code, phiA,
                        phiB, ks, kf, C_P, sA, sB);
            const float2 t1 = cmulf(pa, sA);
            const float2 t2 = cmulf(pb, sB);
            const float2 r = make_float2(0.5f * (t1.x + t2.x), 0.5f * (t1.y + t2.y));
            const float2 a = (mh < job.m_head) ? ah : at;
            v[t].x -= a.x * r.x - a.y * r.y; // v -= a * r
            v[t].y -= a.x * r.y + a.y * r.x;
        }
    }

#pragma unroll
    for (int t = 0; t < H; ++t)
        if (act[t])
            resid[(size_t)ci * p.n_hops + base + t * (int)blockDim.x] = v[t];
}

} // namespace

namespace gnss_cuda {

namespace {
inline int peel_block(int n_hops) {
    return (n_hops < 256 * GNSS_PEEL_HOPS_PER_THREAD) ? 64 : 256;
}
inline dim3 peel_grid(int n_hops, int block, int n_chan) {
    const int per_block = block * GNSS_PEEL_HOPS_PER_THREAD;
    return dim3((unsigned)((n_hops + per_block - 1) / per_block), (unsigned)n_chan);
}
} // namespace

cudaError_t launch_peel(const float2* data, const int8_t* code, const PeelJob* jobs, int n_job,
                        int n_chan, const DespreadParams& p, float2* resid, cudaStream_t stream) {
    const int block = peel_block(p.n_hops);
    gnss_peel_kernel<float2><<<peel_grid(p.n_hops, block, n_chan), block, 0, stream>>>(
        data, nullptr, code, jobs, n_job, p, resid);
    return cudaGetLastError();
}

cudaError_t launch_peel_q(const unsigned char* data, const float* chan_scale, const int8_t* code,
                          const PeelJob* jobs, int n_job, int n_chan, const DespreadParams& p,
                          float2* resid, cudaStream_t stream) {
    const int block = peel_block(p.n_hops);
    gnss_peel_kernel<unsigned char><<<peel_grid(p.n_hops, block, n_chan), block, 0, stream>>>(
        data, chan_scale, code, jobs, n_job, p, resid);
    return cudaGetLastError();
}

} // namespace gnss_cuda

namespace {

/// One thread per (spec, channel). See launch_peel_addback for the identity and why it is exact.
__global__ void gnss_peel_addback_kernel(double2* __restrict__ corr,
                                         const double* __restrict__ energy,
                                         const double2* __restrict__ xcorr,
                                         const float2* __restrict__ gains, int n_spec, int n_chan,
                                         int rows_spec) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_spec * n_chan)
        return;
    const int b = idx / n_chan;  // spec
    const int ci = idx % n_chan; // channel

    const size_t r0 = (size_t)(rows_spec * b) * n_chan + ci; // row 0 (E) of this spec/channel
    const size_t x0 = (size_t)(4 * b) * n_chan + ci;         // xcorr row 0 of this spec/channel

    const double2 ah = make_double2(gains[(size_t)(2 * b) * n_chan + ci].x,
                                    gains[(size_t)(2 * b) * n_chan + ci].y);
    const double2 at = make_double2(gains[(size_t)(2 * b + 1) * n_chan + ci].x,
                                    gains[(size_t)(2 * b + 1) * n_chan + ci].y);

    // <R_P,R_P> over the window and over the head: the prompt's own energies, already computed.
    const double eP = energy[r0 + (size_t)1 * n_chan];
    const double eH = energy[r0 + (size_t)3 * n_chan];

    // PRESERVE the residual BEFORE overwriting rows 1 and 3 -- rows 4/5 are the peel observable.
    corr[r0 + (size_t)4 * n_chan] = corr[r0 + (size_t)1 * n_chan];
    corr[r0 + (size_t)5 * n_chan] = corr[r0 + (size_t)3 * n_chan];

#pragma unroll
    for (int t = 0; t < 4; ++t) {
        // <R_P, R_t>, whole window (xw) and head-restricted (xh).
        double2 xw, xh;
        if (t == 0 || t == 2) { // Early / Late: measured cross-terms
            const int k = (t == 0) ? 0 : 1;
            xw = xcorr[x0 + (size_t)k * n_chan];
            xh = xcorr[x0 + (size_t)(k + 2) * n_chan];
        } else if (t == 1) { // Prompt: <R_P,R_P> is the energy
            xw = make_double2(eP, 0.0);
            xh = make_double2(eH, 0.0);
        } else { // P_HEAD: R_PH is R_P on the head, so head == whole == eH and the tail adds 0
            xw = make_double2(eH, 0.0);
            xh = xw;
        }
        const double2 xt = make_double2(xw.x - xh.x, xw.y - xh.y); // tail share
        double2 v = corr[r0 + (size_t)t * n_chan];
        v.x += ah.x * xh.x - ah.y * xh.y + at.x * xt.x - at.y * xt.y;
        v.y += ah.x * xh.y + ah.y * xh.x + at.x * xt.y + at.y * xt.x;
        corr[r0 + (size_t)t * n_chan] = v;
    }
}

} // namespace

namespace gnss_cuda {

cudaError_t launch_peel_addback(double2* corr, const double* energy, const double2* xcorr,
                                const float2* gains, int n_spec, int n_chan, int rows_spec,
                                cudaStream_t stream) {
    const int total = n_spec * n_chan;
    if (total <= 0)
        return cudaSuccess;
    const int block = 128;
    gnss_peel_addback_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        corr, energy, xcorr, gains, n_spec, n_chan, rows_spec);
    return cudaGetLastError();
}

__global__ void gnss_chan_ingest_q_kernel(const unsigned char* __restrict__ frame,
                                          unsigned char* __restrict__ ring, int n_hops_f,
                                          int n_chan, long long ring_hops, long long write_hop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hops_f * n_chan)
        return;
    const int m = idx / n_chan; // hop within the frame
    const int c = idx % n_chan; // channel (coalesced across threads: frame is hop-major)
    ring[(size_t)c * ring_hops + (size_t)((write_hop + m) % ring_hops)] = frame[idx];
}

cudaError_t launch_chan_ingest_q(const unsigned char* frame, unsigned char* ring, int n_hops_f,
                                 int n_chan, long long ring_hops, long long write_hop,
                                 cudaStream_t stream) {
    const int total = n_hops_f * n_chan;
    const int block = 256;
    gnss_chan_ingest_q_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        frame, ring, n_hops_f, n_chan, ring_hops, write_hop);
    return cudaGetLastError();
}

__global__ void gnss_ring_zero_q_kernel(unsigned char* __restrict__ ring, int n_chan,
                                        long long ring_hops, long long write_hop,
                                        long long count) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count * n_chan)
        return;
    const long long m = idx / n_chan;
    const int c = (int)(idx % n_chan);
    // Offset-encoded zero, NOT 0x00: 0x00 decodes to (-8, -8), a large DC spike.
    ring[(size_t)c * ring_hops + (size_t)((write_hop + m) % ring_hops)] = gnss44::ZERO;
}

cudaError_t launch_ring_zero_q(unsigned char* ring, int n_chan, long long ring_hops,
                               long long write_hop, long long count, cudaStream_t stream) {
    const long long total = count * n_chan;
    const int block = 256;
    gnss_ring_zero_q_kernel<<<(unsigned)((total + block - 1) / block), block, 0, stream>>>(
        ring, n_chan, ring_hops, write_hop, count);
    return cudaGetLastError();
}


__global__ void gnss_chan_ingest_kernel(const float2* __restrict__ frame,
                                        float2* __restrict__ ring, int n_hops_f, int n_chan,
                                        long long ring_hops, long long write_hop) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hops_f * n_chan)
        return;
    const int m = idx / n_chan; // hop within the frame
    const int c = idx % n_chan; // channel (coalesced across threads: frame is hop-major)
    ring[(size_t)c * ring_hops + (size_t)((write_hop + m) % ring_hops)] = frame[idx];
}

cudaError_t launch_chan_ingest(const float2* frame, float2* ring, int n_hops_f, int n_chan,
                               long long ring_hops, long long write_hop, cudaStream_t stream) {
    const int total = n_hops_f * n_chan;
    const int block = 256;
    gnss_chan_ingest_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        frame, ring, n_hops_f, n_chan, ring_hops, write_hop);
    return cudaGetLastError();
}

__global__ void gnss_ring_zero_kernel(float2* __restrict__ ring, int n_chan,
                                      long long ring_hops, long long write_hop,
                                      long long count) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count * n_chan)
        return;
    const long long m = idx / n_chan;
    const int c = (int)(idx % n_chan);
    ring[(size_t)c * ring_hops + (size_t)((write_hop + m) % ring_hops)] = make_float2(0.f, 0.f);
}

cudaError_t launch_ring_zero(float2* ring, int n_chan, long long ring_hops, long long write_hop,
                             long long count, cudaStream_t stream) {
    const long long total = count * n_chan;
    const int block = 256;
    gnss_ring_zero_kernel<<<(unsigned)((total + block - 1) / block), block, 0, stream>>>(
        ring, n_chan, ring_hops, write_hop, count);
    return cudaGetLastError();
}

} // namespace gnss_cuda
