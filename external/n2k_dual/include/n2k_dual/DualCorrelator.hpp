#ifndef _N2K_DUAL_CORRELATOR_HPP
#define _N2K_DUAL_CORRELATOR_HPP

#include <ksgpu/Array.hpp>

#include <memory>
#include <utility>
#include <vector>

// n2k_dual: a two-input clone of the n2k correlator (see README.md in this directory).
//
// PROVENANCE: cloned 2026-08-06 from kotekan's vendored n2k (upstream commit in
// external/n2k/COMMIT). n2k itself is under active development by others, so this is a
// deliberately separate, mostly-duplicated module -- n2k is NOT modified. If the two-input
// extension is ever accepted upstream, this module retires.
//
// WHAT IT ADDS over n2k::Correlator:
//
//  - TWO electric-field inputs sharing one launch:
//        A: int8[T][F][nstations_a]   the real antenna voltages, as today
//        B: int8[T][F][nstations_b]   synthetic stations (e.g. GNSS replica waveforms)
//    The kernel computes the visibility triangle over the CONCATENATED station axis
//    (stations [0, nstations_a) come from A, [nstations_a, nstations_a+nstations_b) from B),
//    without requiring the two inputs to be interleaved into one buffer.
//
//  - Optional block-class restriction (block_class_mask): each 128x128-station threadblock
//    is either pure-A (AxA), mixed (AxB), or pure-B (BxB); the host-side ptable can skip
//    whole classes, e.g. compute only the mixed + BxB blocks. Tiles belonging to skipped
//    classes are simply never written -- the caller must not read them.
//
// HOW (the only real kernel change): the per-warp global-load base offset was always
// precomputed on the host (the "ptable"), so the input select is warp-uniform. The ptable
// grows from 6 to 8 rows per thread: row 6 carries the warp's input TIME STRIDE (in int32
// units; the two inputs have different strides), row 7 selects the A or B base pointer.
// The load path takes the stride as a runtime (warp-uniform) argument instead of a
// compile-time constant. Everything else -- shared-memory layout, in-register transposes,
// tensor-core MMAs, RFI masking, the output tiling -- is byte-identical to n2k.
//
// KEY PROPERTIES (used by the CHORD GNSS path-B integration):
//
//  - Since triangular tile packing orders tiles by (ihi, jhi) with ihi >= jhi, and A comes
//    first on the station axis, the pure-A block of the output is a VERBATIM PREFIX of each
//    (t, f) slice: the first  na16*(na16+1)/2  tiles (na16 = nstations_a/16) are internally
//    identical to what n2k::Correlator(nstations_a, nfreq) would produce. Splitting the AxA
//    result back out for the standard pipeline is a strided copy, nothing more.
//
//  - Mixed tiles always have the B station on the ROW (i) side: V_ij = E_i * conj(E_j)
//    with i synthetic, j antenna. Consumers wanting data*conj(replica) must conjugate.
//
//  - The per-(freq, time) RFI mask gates A and B samples identically, so the mixed block's
//    effective integration time matches the pure-A block's.
//
// CONSTRAINTS (inherited from n2k, plus one new):
//
//  - nstations_a and nstations_b must EACH be a multiple of 128 (a threadblock's 128-station
//    load panel must fall wholly inside one input). nstations_b == 0 is allowed and must
//    reproduce n2k::Correlator bit-exactly (this is the regression gate).
//  - nt_inner must be a multiple of 256.
//  - 4-bit samples must be in [-7, +7]; the value -8 SILENTLY corrupts results
//    (see negate_4bit in the kernel header). For synthetic inputs this is a hard
//    quantizer requirement, not a data-quality accident.
//  - The (nstations_a + nstations_b, nfreq) pair must have a compiled instantiation:
//    template_instantiations/dual_kernel_{NS}_{NF}.cu. Note the kernel template depends
//    only on the TOTAL station count -- the A/B split is entirely host-side ptable data,
//    so one instantiation serves every 128-multiple split of the same total.

namespace n2k_dual {
#if 0
}  // editor auto-indent
#endif


// Block classes, for block_class_mask and dual_block_class().
static constexpr int BLOCK_CLASS_AA = 0;
static constexpr int BLOCK_CLASS_MIXED = 1;
static constexpr int BLOCK_CLASS_BB = 2;

static constexpr int BLOCK_MASK_AA = 1 << BLOCK_CLASS_AA;
static constexpr int BLOCK_MASK_MIXED = 1 << BLOCK_CLASS_MIXED;
static constexpr int BLOCK_MASK_BB = 1 << BLOCK_CLASS_BB;
static constexpr int BLOCK_MASK_ALL = BLOCK_MASK_AA | BLOCK_MASK_MIXED | BLOCK_MASK_BB;


struct DualCorrelatorParams
{
    // User-specified parameters (at construction)
    //   nstations_a = stations in the A (antenna) input; multiple of 128
    //   nstations_b = stations in the B (synthetic) input; multiple of 128, may be 0
    //   nfreq = number of frequency channels (per GPU)
    //   block_class_mask = which block classes to compute (default: all)

    /// @param freq_map optional list of INPUT channel indices to compute. Empty = all nfreq,
    ///        i.e. today's behaviour. Non-empty: the launch covers only those channels and
    ///        vis_out carries freq_map.size() slices in THAT order, not nfreq. The GNSS comb is
    ///        7 channels of 384, and the mixed/BB blocks on the other 377 correlate synthetic
    ///        lanes that are identically zero -- this is the lever that takes path B from
    ///        2.97x stock N^2 to ~1.2x (see docs/gnss_gpu_search.md 11.2).
    DualCorrelatorParams(int nstations_a, int nstations_b, int nfreq,
                         int block_class_mask = BLOCK_MASK_ALL,
                         const std::vector<int>& freq_map = {});

    const int nstations_a;
    const int nstations_b;
    const int nstations; // total = nstations_a + nstations_b
    const int nfreq;          ///< INPUT channels (addressing + rfimask); compile-time NF
    const int block_class_mask;
    const std::vector<int> freq_map; ///< input channels computed; empty = all
    const int n_freq_out;     ///< vis_out slices = freq_map.size(), or nfreq if empty

    // Derived parameters (computed in constructor, or compile-time constants).

    static constexpr int bit_depth = 4;    // correlator operates on signed (4+4) bit data
    static constexpr int nt_divisor = 256; // nt_inner must be divisible by this
    static constexpr int ns_divisor = 128; // nstations_a AND nstations_b divisible by this

    // E-array strides (int32, not bytes!!), PER INPUT -- the inputs are separately shaped.
    const int emat_fstride_a;
    const int emat_tstride_a;
    const int emat_fstride_b;
    const int emat_tstride_b;

    // Visibility matrix strides are in int32 (not int32+32); over the TOTAL station count.
    const int vmat_ntiles;  // M*(M+1)/2   where M = nstations/16
    const int vmat_fstride; // vmat_ntiles * 16 * 16 * 2
    const int vmat_tstride; // nfreq * vmat_fstride

    // Tiling of visibility matrix by threadblocks (units of 128 stations).
    const int ntiles_a; // nstations_a / 128
    const int ntiles_1d;
    const int ntiles_2d_offdiag;
    const int ntiles_2d_tot;

    // Block/thread counts. threadblocks_per_freq counts only ENABLED blocks
    // (== ntiles_2d_tot when block_class_mask == BLOCK_MASK_ALL).
    static constexpr int warps_per_block = 8;
    static constexpr int threads_per_block = 32 * warps_per_block;
    const int threadblocks_per_freq;

    // Shared memory layout -- identical to n2k (all strides are int32).
    static constexpr int shmem_t8_stride = 1;   // Delta(t)=8,16
    static constexpr int shmem_s1_stride = 4;   // Delta(s)=1,2,4
    static constexpr int shmem_s8_stride = 33;  // Delta(s)=8,16,32,64
    static constexpr int shmem_reim_stride = 16 * 33;
    static constexpr int shmem_ab_stride = 32 * 33;
    static constexpr int shmem_t32_stride = 64 * 33; // Delta(t)=32,64
    static constexpr int shmem_nbytes = 8 * shmem_t32_stride * 4; // 256 times, int32 -> bytes

    // ptable rows per thread. n2k has 6 (ap, bp, gp, sp, vi_warp, vk_warp); n2k_dual adds
    // row 6 = warp's input time stride (int32 units), row 7 = input select (0=A, 1=B).
    static constexpr int ptable_nrows = 8;

    // Convention for packing an int4+4 (Re,Im) pair into a byte. (CHIME/CHORD use 'false'.)
    static constexpr bool real_part_in_low_bits = false;

    // Is signed int4 offset-encoded (stored value+8) or twos-complement? (CHIME/CHORD: true.)
    static constexpr bool offset_encoded = true;

    // Debug switches, as in n2k (making the kernel incorrect when enabled!).
    static constexpr bool artificially_remove_input_shuffle = false;
    static constexpr bool artificially_remove_output_shuffle = false;
    static constexpr bool artificially_remove_negate_4bit = false;
};


// Class of the 128x128-station block at (atile, btile) (units of 128 stations, atile >= btile):
// BLOCK_CLASS_AA, BLOCK_CLASS_MIXED, or BLOCK_CLASS_BB.
extern int dual_block_class(int atile, int btile, int ntiles_a);

// The (atile, btile) pairs of the enabled blocks, in ptable/blockIdx.x order. The order of
// the FULL enumeration (mask = BLOCK_MASK_ALL) matches n2k's _init_block_data(): the strict
// lower triangle (1,0),(2,0),(2,1),(3,0),... first, then the diagonal (0,0),(1,1),...
// A restricted mask keeps that order and drops the disabled blocks.
extern std::vector<std::pair<int, int>> dual_enumerate_blocks(int ntiles_1d, int ntiles_a,
                                                              int block_class_mask);


class DualCorrelator
{
public:
    DualCorrelator(const DualCorrelatorParams& params);

    DualCorrelator(int nstations_a, int nstations_b, int nfreq) :
        DualCorrelator(DualCorrelatorParams(nstations_a, nstations_b, nfreq)) {}

    // Launches the dual correlator kernel. Semantics as n2k::Correlator::launch(), except:
    //
    //  - 'e_in_a' is int8[nt_outer*nt_inner][nfreq][nstations_a], all axes contiguous,
    //    complex (4+4) as int8 (imag in low nibble, real in high nibble, offset-encoded).
    //  - 'e_in_b' is int8[nt_outer*nt_inner][nfreq][nstations_b], same encoding.
    //    Must be nullptr iff nstations_b == 0.
    //  - 'vis_out' covers the (nstations_a + nstations_b) triangle; tile (t,f,i,j) layout
    //    and offsets exactly as documented in n2k/Correlator.hpp, with
    //    fstride = nstations*(nstations+16). Tiles in DISABLED block classes are not
    //    written (caller must not read them).
    //  - 'rfimask' is uint32[nfreq][(nt_outer*nt_inner)/32] (same ring-buffer re-indexing
    //    as kotekan's n2k), gating A and B samples identically.

    void launch(int* vis_out, const int8_t* e_in_a, const int8_t* e_in_b, const uint* rfimask,
                int nt_outer, int nt_inner, cudaStream_t stream = nullptr,
                bool sync = false) const;

    // ksgpu::Array version; all arrays on the GPU.
    //   vis_out: shape (nt_outer, nfreq, nvtiles, 16, 16, 2), or (nfreq, nvtiles, 16, 16, 2)
    //            if nt_outer == 1, where nvtiles = (nstations/16)*(nstations/16+1)/2.
    //   e_in_a:  shape (nt_outer * nt_inner, nfreq, nstations_a)
    //   e_in_b:  shape (nt_outer * nt_inner, nfreq, nstations_b); ignored if nstations_b==0.
    //   rfimask: shape (nfreq, nt_outer * nt_inner / 32)
    void launch(ksgpu::Array<int>& vis_out, const ksgpu::Array<int8_t>& e_in_a,
                const ksgpu::Array<int8_t>& e_in_b, const ksgpu::Array<uint>& rfimask,
                int nt_outer, int nt_inner, cudaStream_t stream = nullptr,
                bool sync = false) const;

    // Initialized by constructor.
    const DualCorrelatorParams params;

    // Kernel args are (dst, srcA, srcB, rfimask, ptable, nt_inner, nt_outer).
    using kernel_t = void (*)(int*, const int8_t*, const int8_t*, const uint*, const int*,
                              const int*, int, int, int);

protected:
    int cuda_device;

    // Persist in GPU memory for the lifetime of the DualCorrelator object.
    std::shared_ptr<int> precomputed_offsets;
    std::shared_ptr<int> d_freq_map; ///< null when freq_map is empty

    kernel_t kernel;
};


// Used internally by DualCorrelator constructor
extern std::shared_ptr<int> precompute_offsets(const DualCorrelatorParams& params);
extern DualCorrelator::kernel_t get_kernel(int nstations, int nfreq);

// Used internally to "promote" compile-time argument to runtime argument.
extern void register_kernel(int nstations, int nfreq, DualCorrelator::kernel_t kernel);

// For testing -- returns allowed (nstations_total, nfreq) pairs
extern std::vector<std::pair<int, int>> get_all_kernel_params();


} // namespace n2k_dual

#endif // _N2K_DUAL_CORRELATOR_HPP
