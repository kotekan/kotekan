#ifndef _N2K_PL_KERNELS_HPP
#define _N2K_PL_KERNELS_HPP

#include <gputils/Array.hpp>

namespace n2k {
#if 0
}  // editor auto-indent
#endif


// NOTE: this code is still under development -- you shouldn't use it in kotekan just yet!

extern void launch_pl_mask_expander(ulong *pl_out, const ulong *pl_in, long Tout, long Fout, long S, cudaStream_t stream=0);

extern void launch_pl_mask_expander(gputils::Array<ulong> &pl_out, const gputils::Array<ulong> &pl_in, cudaStream_t stream=0);


extern void launch_pl_1bit_correlator(
    int *V_out,
    const ulong *pl_mask,
    const uint *rfimask,
    long rfimask_fstride,
    long T,
    long F,
    long S,
    long Nds,
    cudaStream_t stream = 0);


// pl_mask shape = (T/64, F, S)
// V_out shape = (T/Nds, F, ntiles, 8, 8)
extern void launch_pl_1bit_correlator(
    gputils::Array<int> &V_out,
    const gputils::Array<ulong> &pl_mask,
    const gputils::Array<uint> &rfimask,
    long Nds,
    cudaStream_t stream = 0);


} // namespace n2k

#endif // _N2K_PL_KERNELS_HPP
