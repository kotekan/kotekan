#ifndef _N2K_HPP
#define _N2K_HPP

#include <gputils.hpp>


namespace n2k {
#if 0
}  // editor auto-indent
#endif


struct CorrelatorParams
{
    // User-specified parameters (at construction)
    //   nstations = number of stations. A "station" is a dish polarization pair.
    //   nfreq = number of frequency channels (per GPU)
    //
    // Full CHORD: nfreq=16 and nstations=1024 (corresponding to 512 dual-pol dishes).
    // CHORD pathfinder: nfreq=128 and nstations=128 (corresponding to 64 dual-pol dishes).
    
    CorrelatorParams(int nstations, int nfreq);
    
    const int nstations;
    const int nfreq;

    // Derived parameters start here (either computed in constructor, or compile-time constants).
    // Some of these may become user-specified parameters in the future!
    
    static constexpr int bit_depth = 4;         // correlator operates on signed (4+4) bit data
    static constexpr int nt_divisor = 256;      // nt_inner must be divisible by this (see below)
    static constexpr int ns_divisor = 128;      // Number of stations must be divisible by this

    // E-array strides (int32, not bytes!!)
    const int emat_fstride;
    const int emat_tstride;

    // Visibililty matrix strides are in int32 (not int32+32)
    const int vmat_ntiles;   // M*(M+1)/2   where M = nstations/16
    const int vmat_fstride;  // vmat_ntiles * 16 * 16 * 2
    const int vmat_tstride;  // nfreq * vmat_fstride
    
    // Tiling of visibility matrix by threadblocks.
    const int ntiles_1d;
    const int ntiles_2d_offdiag;
    const int ntiles_2d_tot;

    // Block/thread counts
    static constexpr int warps_per_block = 8;
    static constexpr int threads_per_block = 32 * warps_per_block;
    const int threadblocks_per_freq;

    // Shared memory layout (see overleaf; all strides are int32)
    static constexpr int shmem_t8_stride = 1;     // Delta(t)=8,16
    static constexpr int shmem_s1_stride = 4;     // Delta(s)=1,2,4
    static constexpr int shmem_s8_stride = 33;    // Delta(s)=8,16,32,64
    static constexpr int shmem_reim_stride = 16 * 33;
    static constexpr int shmem_ab_stride = 32 * 33;
    static constexpr int shmem_t32_stride = 64 * 33;   // Delta(t)=32,64
    static constexpr int shmem_nbytes = 8 * shmem_t32_stride * 4;   // 256 times, convert int32 -> bytes

    // Convention for packing an int4+4 (Re,Im) pair into an byte. (CHIME/CHORD use 'false' here.)
    static constexpr bool real_part_in_low_bits = false;
    
    // These switches will artificially remove an important part of the processing, in order
    // to measure computational overhead of key steps (but making the kernel incorrect!)
    //
    //  input_shuffle - transpose input data from natural ordering to tensor core ordering
    //  output_shuffle - transpose visibility matrix to cache-friendly ordering before writing
    //  negate_4bit - minus sign which arises when multiplying complex numbers
    
    static constexpr bool artificially_remove_input_shuffle = false;
    static constexpr bool artificially_remove_output_shuffle = false;
    static constexpr bool artificially_remove_negate_4bit = false;
};


class Correlator
{
public:
    Correlator(const CorrelatorParams &params);

    Correlator(int nstations, int nfreq) :
	Correlator(CorrelatorParams(nstations,nfreq)) { }

    // The launch() function below launches the correlator kernel.
    //
    //  - The electric field input data is assumed to be signed (4+4) bit complex.
    //    In a future update, I may generalize this.
    //
    //  - The time cadence is controlled by two parameters, 'nt_outer' and 'nt_inner'.
    //    The total number of time samples is (nt_outer * nt_inner), and the visibility
    //    matrix is computed every 'nt_inner' samples.
    //
    //  - The 'e_in' array is
    //
    //       int8[nt_outer*nt_inner][nfreq][nstations]  with all axes contiguous    (*)
    // 
    //    where we represent complex (4+4) as int8.
    //
    //    Note in (*) that the real/imaginary axis is fastest varying, followed by the
    //    station axis, and both these axes are contiguous. These assumptions would be
    //    a lot of work to change. On the other hand, it would be easy to swap the ordering
    //    of the time/frequency axes, or to allow non-contiguous strides for these axes.
    //
    //  - The 'vis_out' array is the visibility matrix V_{t,f,i,j}, but the memory layout
    //    needs some explanation. We split the matrix into 16-by-16 "tiles", and store only
    //    tiles in the lower triangle.
    //
    //    To write this out formally, we split indices i,j into upper and lower parts:
    //
    //        i = 16*ihi + ilo        0 <= ihi < (nstations/16)     0 <= ilo < 16
    //        j = 16*jhi + jlo        0 <= jhi < (nstations/16)     0 <= jlo < 16
    //
    //    Then we store only entries of the visibility matrix with ihi >= jhi (i.e. tiles in
    //    the lower triangle). The memory offset of visibility matrix (t,f,i,j) is:
    //
    //        Memory offset in multiples of sizeof(int32)
    //           =   (t * nfreq * tstride)
    //             + (f * fstride)
    //             + 512 * (ihi*(ihi+1)/2 + jhi)
    //             + 32*ilo + 2*jlo
    //
    //    where fstride = 512 * (nstations/16) * (nstations/16+1) / 2
    //                  = nstations * (nstations+16)
    //
    //
    //  - The 'rfimask' array is a boolean array indexed by (freq, time), represented as:
    //
    //       int32 rfimask[nfreq][(nt_outer*nt_inner)/32] with axes contiguous.
    //
    //    Formally, the RFI mask bit corresponding to indices (f,t) is given by the following code:
    //
    //       int i = f*nt_outer*nt_inner + t;
    //       bool unmasked = rfimask[i/32] & (1 << (i % 32));   // where rfimask is a (uint *)
    //
    //  - 'nt_inner' must be a multiple of 256.
    //
    //  - We assume that 4-bit electric field samples are in the range [-7,7], i.e. the value (-8)
    //    is not allowed. Currently, if (-8) does arise, then the output of the computation will
    //    be incorrect (silently!). This behavior would be easy to change (possibly with a little
    //    extra computational cost).
    //
    //  - We define the visibility matrix as V_{ij} = sum_t E_{it} E_{jt}^*, with the complex
    //    conjugate on the second factor. (This would be trivial to change.)
    //
    //  - The kernel will segfault if run on a GPU which is not the cuda default device.
    //    This will be easy to fix later.
    //
    //  - The same Correlator object can be used to launch multiple kernels concurrently (either in
    //    serial using the same cuda stream, or in parallel using different cuda streams).
    //
    //  - The first few calls to launch() sometimes run slow, for reasons I haven't figured out yet.
    //    If you run the 'time-correlator' program, you can see the first few calls take longer, but
    //    the timing quickly settles down.

    void launch(int *vis_out, const int8_t *e_in, const uint *rfimask,
		int nt_outer, int nt_inner, cudaStream_t stream=nullptr, bool sync=false) const;

    // This version of launch() uses gputils::Array objects instead of bare pointers.
    // Both arrays must be allocated on the GPU.
    //
    // The 'vis_out' array must have shape (nt_outer, nfreq, nvtiles, 16, 16, 2).
    // If nt_outer==1, then shape (nfreq, nvtiles, 16, 16, 2) is also okay.
    // Here, nvtiles = (nstations/16) * (nstations/16+1) / 2.
    //
    // The 'e_in' array must have shape (nt_outer * nt_inner, nfreq, nstations).
    //
    // The 'rfimask' array must have shape (nfreq, nt_outer * nt_inner / 32).
    // (See previous long comment for indexing logic.)
    
    void launch(gputils::Array<int> &vis_out, const gputils::Array<int8_t> &e_in, const gputils::Array<uint> &rfimask,
		int nt_outer, int nt_inner, cudaStream_t stream=nullptr, bool sync=false) const;
    
    // Initialized by constructor.
    const CorrelatorParams params;

    // Kernel args are (dst, src, rfimask, ptable, nt_inner).
    using kernel_t = void (*)(int *, const int8_t *, const uint *, const int *, int);

protected:
    // This small (currently 27 KB) array will persist in GPU memory for the lifetime of the Correlator object.
    // Note that the shared_ptr destructor will call cudaFree().
    std::shared_ptr<int> precomputed_offsets;

    kernel_t kernel;
};


// Used internally by Correlator constructor
extern std::shared_ptr<int> precompute_offsets(const CorrelatorParams &params);
extern Correlator::kernel_t get_kernel(int nstations, int nfreq);

// Used internally to "promote" compile-time argument to runtime argument.
extern void register_kernel(int nstations, int nfreq, Correlator::kernel_t kernel);

// For testing -- returns allowed (nstations, nfreq) pairs
extern std::vector<std::pair<int,int>> get_all_kernel_params();


// -------------------------------------------------------------------------------------------------
//
// RFI-related code starts here (I might split into multiple .hpp files later)


// launch_rfimask_maker()
//
// Arguments:
//
//    uint rfimask[nbits/32];      // output array
//    float sk_sigma[nbits/ds];    // input array
//    long nbits;                  // defines dimensions of input/output arrays
//    long ds;                     // downsampling factor, also defines dimensions
//    float thresh;                // masking threshold in 'sk_sigma' array (e.g. 5.0 for 5 sigma)
//
// Constraints:
//
//   - nbits must be a multiple of 128, and a multiple of ds.
//   - ds must be a multiple of 32.
//
// Note: in this kernel, we treat 'rfimask' and 'sk_sigma' as 1-d arrays, even though
// in the larger correlator, they are 2-d arrays:
//
//    uint rfimask[F][T/32];
//    float sk_sigma[F][T/ds];
//
// This is because the frequency index is a "spectator" index, so we can flatten these arrays and
// treat them as 1-d arrays with nbits=F*T.
//
// These functions are defined in make_rfimask.cu.


// Bare pointer interface.
extern void launch_rfimask_maker(uint *rfimask, const float *sk_sigma, long nbits, long ds, float thresh, cudaStream_t stream=0);

// gputils::Array<> interface.
extern void launch_rfimask_maker(gputils::Array<uint> &rfimask, const gputils::Array<float> &sk_sigma, long ds, float thresh, cudaStream_t stream=0);


// launch_s0_kernel()
//
// Arguments:
//
//   uint s0[T/ds][F][S];                  // output array, (downsampled time index, freq channel, station)
//   ulong pl_mask[T/128][(F+3)/4][S/8];   // input array, packet loss mask
//   long T;                               // number of time samples
//   long F;                               // number of freq channels
//   long S;                               // number of stations (= dish+pol pairs)
//   long ds;                              // time downsampling factor
//
// Constraints:
//
//   - ds must be even (but note that the make_rfimask kernel requires ds to be a multiple of 32)
//   - S must be a multiple of 128 (required by packet loss array layout, see s0_kernel.cu)
//   - T must be a multiple of 128 (required by packet loss array layout, see s0_kernel.cu)
//   - T must be a multiple of ds.
//
// Note: see s0_kernel.cu for description and discussion of packet loss mask array layout.


// Bare pointer interface.
extern void launch_s0_kernel(uint *s0, const ulong *pl_mask, long T, long F, long S, long ds, cudaStream_t stream=0);

// gputils::Array<> interface.
extern void launch_s0_kernel(gputils::Array<uint> &s0, const gputils::Array<ulong> &pl_mask, long ds, cudaStream_t stream=0);


}  // namespace n2k

#endif // _N2K_HPP
