#ifndef CUDA_GNSS_ACQUIRE_KERNEL_HPP
#define CUDA_GNSS_ACQUIRE_KERNEL_HPP

#include <cuda_runtime.h>

/**
 * GPU acquisition surface aggregate + peak reduction.
 *
 * Port of gnss::aggregate_accumulate (lib/stages/gnss/gnssChannelizedAcquire.cpp), which is
 * ~100% of a blind search pass: 81 GFlop and 0.667 s per NH alignment per window at CHORD's
 * blind dimensions (nc=79, n_dop=321, Mp=3125, s_cols=128), measured on cf06 by
 * scripts/gnss/aggbench. Twenty NH alignments x 32 PRNs is the 427 s snapshot that makes a cold
 * start impossible -- see docs/gnss_gpu_search.md.
 *
 *     D(q,i) = sum_c P_c(q) e^{+i 2 pi f_c (i * s_step) / sph},   surf(q,i) += |D(q,i)|^2
 *
 * TWO PATHS, both provided so they can be measured against each other rather than argued about:
 *
 *  - DIRECT: the literal sum, nc * s_cols complex MACs per row. Needs the [nc][s_cols] ramp
 *    table, which at CHORD scale is 79*128*8 = 81 KB and does NOT fit in shared memory, so it
 *    streams from L2 and the kernel is compute-bound.
 *
 *  - FOLDED: when `s_cols * s_step == s_stored` the ramp depends on the channel only through
 *    `(f_c - f_0)/g mod s_cols`, so the sum IS an s_cols-point inverse DFT of the channels
 *    folded into s_cols bins. Exact, not approximate -- an identity about the ramp that assumes
 *    nothing whatever about the signal. (This is NOT the coarse-axis fold that was tried and
 *    reverted in docs/CHORD_GNSS_STATE.md 5o; that one assumed the replica was periodic in the
 *    window and died on the NH overlay.) The table collapses to nc ints + an s_cols/2 twiddle
 *    table, both of which fit in shared memory, and the flop count drops ~18x.
 *
 * LAYOUT. P arrives as [d][q][c] -- channel FASTEST -- not the [c][d][q] the CPU builds. That
 * is deliberate: the aggregate consumes all nc channels of one (d,q) row together, so channel
 * must be contiguous or every row costs nc separate strided loads. cuFFT writes this layout
 * directly via ostride=nc, odist=1 (one batch of nc transforms per Doppler bin), so the
 * transpose is free at the producer rather than a separate pass here.
 *
 * PRECISION. The CPU reference accumulates the surface in double; this accumulates in float.
 * Over `acquire_windows` accumulations of |D|^2 that is comfortable, but it is a real
 * difference, so the validation in scripts/gnss/aggbench compares against the double reference
 * at BLIND dimensions, not at toy ones.
 */
namespace gnss_cuda {

/// Aggregate one window into `surf` (accumulating, as the CPU path does).
/// @param P       device [n_dop][Mp][nc] complex, channel fastest
/// @param surf    device [n_dop][Mp][s_cols] float, ACCUMULATED into (zero it for window 0)
/// @param ramp    device [nc][s_cols] complex, e^{+i 2 pi f_c i s_step / sph}; direct path only
/// @param bin     device [nc] int fold bins; folded path only
/// @param twiddle device [s_cols/2] complex, e^{+2 pi i j / s_cols}; folded path only
/// @param folded  select the folded path (caller must have verified the precondition)
void launch_aggregate(const float2* P, float* surf, const float2* ramp, const int* bin,
                      const float2* twiddle, int nc, int n_dop, int Mp, int s_cols, bool folded,
                      cudaStream_t stream);

/// Peak + mean of a surface, on device -- so the 1.03 GB surface never crosses PCIe.
/// Writes {peak, sum} to `out_val[0..1]` (float2) and the flat argmax to `out_idx[0]`.
/// `scratch` must hold at least `n_blocks` float2 + int; pass n_blocks from @ref peak_blocks.
void launch_peak(const float* surf, long n_cells, float2* out_val, long long* out_idx,
                 void* scratch, cudaStream_t stream);

/// Blocks @ref launch_peak will use for `n_cells`, so the caller can size `scratch`.
int peak_blocks(long n_cells);

/// Bytes of scratch @ref launch_peak needs for `n_cells`.
size_t peak_scratch_bytes(long n_cells);

} // namespace gnss_cuda

#endif // CUDA_GNSS_ACQUIRE_KERNEL_HPP
