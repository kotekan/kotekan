#ifndef _N2K_RFI_KERNELS_HPP
#define _N2K_RFI_KERNELS_HPP

#include <gputils/Array.hpp>

namespace n2k {
#if 0
}  // editor auto-indent
#endif


// For a description of the X-engine RFI flagging logic, see the high-level
// software overleaf ("RFI statistics computed on GPU" section). The RFI code
// will be hard to understand unless you're familiar with this document!
// 
// This file declares five GPU kernels which create and downsample S-arrays:
//
//    launch_s0_kernel(): computes S0 from the packet loss mask.
//    launch_s012_kernel(): computes S1, S2 from the E-array.
//    launch_s012_time_downsample_kernel(): downsamples S0,S1,S2 along time-axis.
//    launch_s012_station_downsample_kernel(): downsamples S0,S1,S2 along station-axis.
//    class SkKernel: computes SK-arrays and boolean RFI mask from S0,S1,S2 arrays.
//
// When processing multiple "frames" of data, you should create a persistent SkKernel
// object and call launch() for each frame, rather than creating a new SkKernel for
// each frame.
//
// Review of array layouts (for more details see the overleaf):
//
//    - S-array layout is
//
//         ulong S[Tcoarse][F][3][2*D];   // case 1: single-feed S-array
//         ulong S[Tcoarse][F][3];        // case 2: feed-summed S-array
//
//      where the length-3 axis “packs” (S0, S1, S2) into a single array.
//
//    - The S-arrays contain cumulants of the E-array. Schematically:
//
//         S0 = sum (PL)
//         S1 = sum (PL * |E|^2)
//         S2 = sum (PL * |E|^4)
//
//    - The single-feed S-array (case 1 above) is created by calling
//      launch_s0_kernel() and launch_s012_kernel(), and can be downsampled
//      further in time by calling launch_s012_time_downsample_kernel().
//
//    - The feed-summed S-array (case 2 above) is created by calling
//      launch_s012_station_downsample_kernel() on the single-feed S-array.
//
//     - The SK-arrays (created by 'class SK-kernel') have the following memory layouts:
//
//          // length-3 axes are {SK,b,sigma}
//          float SK[T][F][3][S];   // Case 1: single-feed SK-statistic
//          float SK[T][F][3];      // Case 2:feed-averaged SK-statistic
//
//     - The boolean RFI mask (created by 'class SK-kernel') has the following memory layout:
//
//          int1 rfimask[F][T*Nds];        // factor of Nds since time index is "high-res" (baseband)
//          uint rfimask[F][(T*Nds)/32];   // equivalent representation as uint32[] instead of int1[]


// -------------------------------------------------------------------------------------------------
//
// Kernel 1/5: launch_s0_kernel().
//
// Computes S0 from packet loss mask, downsampling in time by specified factor 'Nds'.
//
// Constraints:
//   - Nds must be even (but note that the SK-kernel requires Nds to be a multiple of 32)
//   - S must be a multiple of 128 (required by packet loss array layout, see s0_kernel.cu)
//   - T must be a multiple of 128 (required by packet loss array layout, see s0_kernel.cu)
//   - T must be a multiple of Nds.
//
// Note: the packet loss mask has a nontrivial array layout. See either the software overleaf
// (section "Data sent from CPU to GPU in each kotekan frame"), or comments in s0_kernel.cu.
//
// A note on 'out_fstride' argument to launch_s0_kernel(): In the larger X-engine context,
// the s0_kernel is used to initialize "one-third" of the S012 array:
//
//   S[Tcoarse][F][3][2*D];   // s0_kernel initializes S[:,:,0,:]
//
// Therefore, from the perspective of the s0_kernel, the output array is discontiguous:
// the "stride" of the frequency axis is (6*D) instead of (2*D). This is implemented by
// passing out_fstride=6*D to launch_s0_kernel().


// Version 1: bare-pointer interface.
extern void launch_s0_kernel(
    ulong *S0,                   // output array, shape (T/Nds, F, S)
    const ulong *pl_mask,        // input array, shape (T/128, (F+3)/4, S/8), see above
    long T,                      // number of time samples in input array (before downsampling)
    long F,                      // number of frequency channels
    long S,                      // number of stations (= 2*D, where D is number of dishes)
    long Nds,                    // time downsampling factor
    long out_fstride,            // frequency stride in 'S0' array, see comment above.
    cudaStream_t stream=0);


// Version 2: gputils::Array<> interface.
// Note that there is no 'out_fstride' arugment, since 'S0' is a gputils::Array, which contains strides.

extern void launch_s0_kernel(
     gputils::Array<ulong> &S0,              // output array, shape (T/Nds, F, S)
     const gputils::Array<ulong> &pl_mask,   // input array, shape (T/128, (F+3)/4, S/8), see above
     long Nds,                               // time downsampling factor
     cudaStream_t stream=0);


// -------------------------------------------------------------------------------------------------
//
// Kernel 2/5: launch_s12_kernel().
//
// Computes S1,S2 from E-array, downsampling in time by specified factor 'Nds'.
//
// Note: the s12_kernel does not need the packet loss mask as an input. This is because
// the E-array is assumed to be zeroed for missing packets. (See overleaf notes, section
// "Data sent from CPU to GPU in each kotekan frame".)
//
// A note on 'out_fstride' argument to launch_s12_kernel(): In the larger X-engine context,
// the s12_kernel is used to initialize "two-thirds" of the S012 array:
//
//   S[Tcoarse][F][3][2*D];   // s12_kernel initializes S[:,:,1:3,:]
//
// Therefore, from the perspective of the s12_kernel, the output array is discontiguous:
// the "stride" of the frequency axis is (6*D) instead of (4*D). This is implemented by
// passing out_fstride=6*D to launch_s12_kernel(). (You'll also add the offset (2*D) to
// the S-array base pointer, in order to write to S[:,:,1:3,:] rather than S[:,:,0:2,:].)


// Version 1: bare-pointer interface.
extern void launch_s12_kernel(
    ulong *S12,           // output array, shape (T/Nds, F, 2, S)
    const uint8_t *E,     // input int4+4 array, shape (T, F, S)
    long T,               // number of time samples in input array (before downsampling)
    long F,               // number of frequency channels
    long S,               // number of stations (= 2*D, where D is number of dishes)
    long Nds,             // time downsampling factor
    long out_fstride,     // frequency stride in 'S12' array, see comment above.
    bool offset_encoded,  // toggle between twos-complement/offset-encoded int4s
    cudaStream_t stream=0);


// Version 2: gputils::Array<> interface.
// Note that there is no 'out_fstride' arugment, since 'S12' is a gputils::Array, which contains strides.

extern void launch_s12_kernel(
    gputils::Array<ulong> &S12,         // output array, shape (T/Nds, F, 2, S)
    const gputils::Array<uint8_t> &E,   // input int4+4 array, shape (T, F, S)
    long Nds,                           // time downsampling factor
    bool offset_encoded,                // toggle between twos-complement/offset-encoded int4s
    cudaStream_t stream=0);


// -------------------------------------------------------------------------------------------------
//
// Kernel 3/5: launch_s012_time_downsample_kernel().
//
// Downsamples S012 along time axis, i.e. input/output arrays are
//
//     Sin[T][F][3][S];        // input array
//     Sout[T/Nds][F][3][S];   // output array
//
// Context: In the overleaf notes, "RFI statistics computed on GPU" section, you'll see
// a block diagram with an arrow labelled "Downsample in time", where the S012 arrays are
// downsampled from ~1 ms to ~30 ms. This kernel implements this step.
//
// Note: the last three indices in the above arrays are "spectator" indices as far as
// this kernel is concerned, and can be replaced by a single spectator index with length
// M = (3*F*S). (As usual, S denotes number of stations, i.e. 2*D, where D = number of dishes.)


// Version 1: bare-pointer interface.
extern void launch_s012_time_downsample_kernel(
    ulong *Sout,        // output array, shape (T/Nds,3,F,S) or equivalently (T/Nds,M)
    const ulong *Sin,   // input array, shape (T,3,F,S) or equivalently (T,M)
    long T,             // number of time samples before downsampling
    long M,             // number of spectator indices (3*F*S), see above
    long Nds,           // time downsampling factor
    cudaStream_t stream=0);

// Version 2: gputils::Array<> interface.
extern void launch_s012_time_downsample_kernel(
    gputils::Array<ulong> &Sout,        // output array, shape (T/Nds,3,F,S)
    const gputils::Array<ulong> &Sin,   // input array, shape (T,3,F,S)
    long Nds,                           // time downsampling factor
    cudaStream_t stream=0);


// -------------------------------------------------------------------------------------------------
//
// Kernel 4/5: launch_s012_station_downsample_kernel().
//
// Downsamples S012 along station axis, i.e. input/output arrays are
//
//     Sin[T][F][3][S];     // input array
//     Sout[T][F][3];       // output array
//
// Context: In the overleaf notes, "RFI statistics computed on GPU" section, you'll see
// a block diagram with an arrow labelled "Sum S0, S1, S2 over feeds", where the feed-summed
// S-array is comptued from the single-feed S-array. This kernel implements this step.
//
// Note: this kernel uses the "bad feed" mask. Logically, this is a boolean 1-d array of
// length S=(2*D). We represent the bad feed mask as a length-S uint8_t array, where a
// zero value means "feed is bad", and any nonzero 8-bit value means "feed is good".
//
// Note: the first three indices in the above arrays are "spectator" indices as far as
// this kernel is concerned, and can be replaced by a single spectator index with length
// M = (3*T*F).


// Version 1: bare-pointer interface.
extern void launch_s012_station_downsample_kernel(
    ulong *Sout,             // output array, shape (T,F,3) or equivalently (M,)
    const ulong *Sin,        // input array, shape (T,F,3,S) or equivalently (M,S)
    const uint8_t *bf_mask,  // bad feed mask, shape (S,), see above
    long M,                  // number of spectator indices (3*T*F), see above
    long S,                  // number of stations
    cudaStream_t stream=0);

// Version 2: gputils::Array<> interface.
extern void launch_s012_station_downsample_kernel(
    gputils::Array<ulong> &Sout,              // output array, shape (T,F,3)
    const gputils::Array<ulong> &Sin,         // input array, shape (T,F,3,S)
    const gputils::Array<uint8_t> &bf_mask,   // bad feed mask, shape (S,), see above
    cudaStream_t stream=0);


// -------------------------------------------------------------------------------------------------
//
// Kernel 5/5:'class SkKernel', a wrapper class for a CUDA kernel with the following
// input/output arrays:
//
//   - input: single-feed S-arrays (S_0, S_1, S_2) indexed by (time, freq, station).
//   - output: feed-averaged SK-statistic and associated (b,sigma).
//   - output (optional): single-feed SK-statistic and associated (b,sigma).
//   - output (optional): boolean RFI mask based on feed-averaged SK-statistic.
//
// Recall that the input array to the SK-kernel (single-feed S-array) has the
// following memory layout:
//
//     ulong S[T][F][3][S];    // length-3 axis is {S0,S1,S2}
//
// The output SK-arrays have the following memory layouts:
//
//     // length-3 axes are {SK,b,sigma}
//     float SK[T][F][3][S];   // Case 1: single-feed SK-statistic
//     float SK[T][F][3];      // Case 2:feed-averaged SK-statistic
//
// The boolean RFI mask has the following memory layout:
//
//     int1 rfimask[F][T*Nds];        // factor of Nds since time index is "high-res" (baseband)
//     uint rfimask[F][(T*Nds)/32];   // equivalent representation as uint32[] instead of int1[]
//
// Note that in the RFI mask, time is the fastest varying index (usually time is
// slowest varying). This may create complications. For example, suppose that the
// ring buffer length 'T_ringbuf' is longer than a single kernel launch 'T_kernel':
//
//     uint rfimask[F][T_ringbuf * Nds / 32];   // T_ringbuf, not T_kernel
//
// From the perspective of the SkKernel, the rfimask is now a discontiguous subarray of a larger
// array. This can be handled by using the 'rfimask_fstride' kernel argument (see below) to the
// 32-bit frequency stride (T_ringbuf * Nds / 32). (If the rfimask array were contiguous, then
// 'rfimask_fstride' would be (T_kernel * Nds / 32).)
//
// The SkKernel uses the "bad feed" mask when computing the feed-averaged SK-statistic
// and the boolean RFI mask. Logically, the bad feed mask is a boolean 1-d array of
// length S=(2*D). We represent the bad feed mask as a length-S uint8_t array, where a
// zero value means "feed is bad", and any nonzero 8-bit value means "feed is good".
//
// In the larger X-engine context, two SkKernels are used (see block diagram in
// the ``RFI statistics computed on GPU'' section of the overleaf). The first
// SkKernel runs at ~1 ms, and computes feed-averaged SK-statistic and RFI mask
// (no single-feed SK-statistic). The second SkKernel runs at ~30 ms, and computes
// single-feed and feed-averaged SK-statistics (no RFI mask).
//
// When processing multiple "frames" of data, you should create a persistent SkKernel
// instance and call launch() for each frame, rather than creating a new SkKernel for
// each frame. This is because The SkKernel constructor is less "lightweight" than you
// might expect. (It allocates a few-KB array on the GPU, copies data from CPU to GPU,
// and blocks until the copy is complete).
//
// Reminder: users of the SK-arrays (either single-feed SK or feed-averaged SK) should
// test for negative values of sigma. There are several reasons that an SK-array element
// can be invalid (masked), and this is indicated by setting sigma to a negative value.


struct SkKernel
{
    // High-level parameters for the SkKernel.
    // See overleaf for precise descriptions.
    // We might define kotekan yaml config parameters which are in one-to-one
    // correspondence with these parameters.
    
    struct Params {
	double sk_rfimask_sigmas = 0.0;             // RFI masking threshold in "sigmas" (only used if out_rfimask != NULL)
	double single_feed_min_good_frac = 0.0;     // For single-feed SK-statistic (threshold for validity)
	double feed_averaged_min_good_frac = 0.0;   // For feed-averaged SK-statistic (threshold for validity)
	double mu_min = 0.0;                        // For single-feed SK-statistic (threshold for validity)
	double mu_max = 0.0;                        // For single-feed SK-statistic (threshold for validity)
	long Nds = 0;                               // Downsampling factor used to construct S012 array (i.e. SK-kernel input array)
    };

    // As noted above, the SkKernel constructor allocates a few-KB array on the GPU,
    // copies data from CPU to GPU, and blocks until the copy is complete.
    //
    // Note: params are specified at construction, but also can be changed freely between calls to launch():
    //    sk_kernel->params.sk_rfimask_sigmas = 3.0;   // this sort of thing is okay at any time

    SkKernel(const Params &params, bool check_params=true);
    
    Params params;

    // Bare-pointer launch() interface.
    // Launches asynchronosly (i.e. does not synchronize stream or device after launching kernel.)
    
    void launch(
        float *out_sk_feed_averaged,          // Shape (T,F,3)
	float *out_sk_single_feed,            // Shape (T,F,3,S), can be NULL
	uint *out_rfimask,                    // Shape (F,T*Nds/32), can be NULL
	const ulong *in_S012,                 // Shape (T,F,3,S)
	const uint8_t *in_bf_mask,            // Length S (bad feed mask)
	long rfimask_fstride,                 // Only used if (out_rfimask != NULL). NOTE: uint32 stride, not bit stride!
	long T,                               // Number of downsampled times in S012 array
	long F,                               // Number of frequency channels
	long S,                               // Number of stations (= 2 * dishes)
	cudaStream_t stream = 0,
	bool check_params = true) const;
    
    // gputils::Array<> interface to launch().
    // Launches asynchronosly (i.e. does not synchronize stream or device after launching kernel.)

    void launch(
        gputils::Array<float> &out_sk_feed_averaged,   // Shape (T,F,3)
	gputils::Array<float> &out_sk_single_feed,     // Either empty array or shape (T,F,3,S)
	gputils::Array<uint> &out_rfimask,             // Either empty array or shape (F,T*Nds/32), need not be contiguous
	const gputils::Array<ulong> &in_S012,          // Shape (T,F,3,S)
	const gputils::Array<uint8_t> &in_bf_mask,     // Length S (bad feed bask)
	cudaStream_t stream = 0) const;

    // Used internally by launch() + constructor.
    // You shouldn't need to call this directly.
    static void check_params(const Params &params);

    // Interpolation table, copied to GPU memory by constructor.
    gputils::Array<float> bsigma_coeffs;
    int device = -1;
};


}  // namespace n2k

#endif // _N2K_RFI_KERNELS_HPP
