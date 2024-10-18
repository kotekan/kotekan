#ifndef _N2K_PL_KERNELS_HPP
#define _N2K_PL_KERNELS_HPP

#include <gputils/Array.hpp>

namespace n2k {
#if 0
}  // editor auto-indent
#endif


// This header file declares two kernels which compute the "counts" array from the PL (packet loss)
// mask. For a complete description of how the counts array is computed, see the high-level software
// overleaf (the "GPU visibility matrices" section).
//
//    launch_pl_mask_expander(): "expands" the PL mask, by removing downsampling along time/freq axes
//    launch_pl_1bit_correlator(): computes the counts array from the expanded PL mask.


// -------------------------------------------------------------------------------------------------
//
// Kernel 1/2: PL mask expander
//
// Recall that when the boolean PL mask is sent from the CPU to the GPU, it is downsampled by
// a factor 2 in time, by a factor 4 in frequency, and a factor 8 in station (= dish + pol):
//
//     ulong pl_in[T/128][(F+3)/4][S/8];   // each 64-bit 'ulong' represents 128 times
//
// The PL mask expander removes the downsampling in time/freq (by duplicating and shuffling bits
// around), but keeps the factor-8 downsampling along the station axis:
//
//     ulong pl_out[T/64][F][S/8];   // each 64-bit 'ulong' represents 64 times
//
// This "expansion" step is necessary before calling launch_pl_1bit_correlator() (see below).
//
// Constraints:
//
//   - Number of stations after downsampling must be a multiple of 16
//      (i.e. number of stations before downsampling must be a multiple of 128)


// Version 1: bare-pointer interface.
extern void launch_pl_mask_expander(
    ulong *pl_out,           // shape (Tout/128, (Fout+3)/4, Sds)
    const ulong *pl_in,      // shape (Tout/64, F, Sds)
    long Tout,               // number of time samples (no downsampling factor)
    long Fout,               // number of freq channels (no downsampling factor)
    long Sds,                // number of downsampled stations (after downsampling by 8)
    cudaStream_t stream = 0);


// Version 2: gputils::Array<> interface.
extern void launch_pl_mask_expander(
    gputils::Array<ulong> &pl_out,        // shape (Tout/128, (Fout+3)/4, Sds)
    const gputils::Array<ulong> &pl_in,   // shape (Tout/64, F, Sds)
    cudaStream_t stream = 0);


// -------------------------------------------------------------------------------------------------
//
// Kernel 2/2: 1-bit correlator (computes counts array from PL mask and RFI mask).
//
//   - Input array: "Expanded" PL mask, obtained by calling launch_pl_mask_expander() above.
//
//         // Each 64-bit 'ulong' represents 64 times.
//         // Sds=(S/8) is the number of downsampled stations.
//         ulong pl[T/64][F][Sds];
//
//   - Input array: boolean RFI mask, obtained by calling the SK-kernel (see rfi_kernels.hpp)
//
//         // Each 32-bit 'uint' represents 32 times.
//         uint rfimask[F][T/32];    (*)
//
//    - The input parameter 'rfimask_fstride' may be useful if the RFI mask is part of a
//      larger ring buffer. This needs a little explanation.
//
//      First mote that in the RFI mask (*), time is the fastest varying index.
//      This may create complications. For example, suppose that the ring buffer length
//      'T_ringbuf' is longer than a single kernel launch 'T_kernel':
//
//          uint rfimask[F][T_ringbuf / 32];   // T_ringbuf, not T_kernel
//
//      From the perspective of the 1-bit correlator, the rfimask is now a discontiguous subarray
//      of a larger array. This can be handled by setting the 'rfimask_fstride' kernel argument
//      to (T_ringbuf / 32), and the 'T' kernel argument to (T_kernel).
//
//    - The input parameter 'Nds' is the downsampling factor of the counts array, relative to
//      the baseband data. (In CHORD, Nds will be around 6000, since the visibility matrix is
//      computed at 30 ms, and baseband data is sampled at ~5 us.)
//
//    - Output 'counts' array: This is logically an (Sds)-by-(Sds) matrix for every (downsampled
//      time, frequency channel). Since the matrix is symmetric, we save memory by dividing into
//      8-by-8 tiles, and only storing tiles in the lower triangle.
//
//         // Ntiles = ((Sds/8) * (Sds/8+1)) / 2.
//         // See overleaf for details of memory layout.
//         int32 counts[T/Nds][F][Ntiles][8][8];
//
// Constraints:
//
//     - Nds must be a multiple of 128 (convenient for tensor core reasons, could be relaxed).
//
//     - We currently assume either Sds=16 (CHORD pathfinder with S=128) or Sds=128 (full CHORD
//       with S=1024). Later, I'll generalize so that Sds can be an arbitrary multiple of 8.


// Version 1: bare-pointer interface.
extern void launch_pl_1bit_correlator(
    int *counts,             // shape (T/Nds, F, Ntiles, 8, 8), where Ntiles = ((Sds/8) * (Sds/8+1)) / 2.
    const ulong *pl_mask,    // shape (T/64, F, Sds)
    const uint *rfimask,     // shape (F, T/32)
    long rfimask_fstride,    // see above
    long T,                  // number of time samples before correlation
    long F,                  // number of frequency channels
    long Sds,                // number of stations (after downsampling by 8)
    long Nds,                // downsampling factor of counts array, relative to baseband
    cudaStream_t stream = 0);


// Version 2: gputils::Array<> interface.
extern void launch_pl_1bit_correlator(
    gputils::Array<int> &counts,           // shape (T/Nds, F, Ntiles, 8, 8), where Ntiles = ((Sds/8) * (Sds/8+1)) / 2.
    const gputils::Array<ulong> &pl_mask,  // shape (T/64, F, Sds)
    const gputils::Array<uint> &rfimask,   // shape (F, T/32)
    long Nds,                              // downsampling factor of counts array, relative to baseband
    cudaStream_t stream = 0);


} // namespace n2k

#endif // _N2K_PL_KERNELS_HPP
