/**
 * @file
 * @brief Gather the GNSS (mixed + synthetic) tiles out of the extended N^2 triangle.
 *
 * The extended visibility triangle is 107 MB/frame and stays on the GPU; the GNSS consumer
 * needs ~0.7 MB of it -- the mixed (synthetic x live-antenna) and synthetic x synthetic tiles
 * of the 7 comb channels. The selection list is host-built at construction
 * (cudaCorrelatorDual) and constant for the run, so the kernel is a flat gather over
 * (tile, int32) pairs: consecutive threads copy consecutive int32s of one tile, fully
 * coalesced on both sides.
 */

#include <cuda_runtime.h>

#include <cstdint>

namespace gnss_cuda {

__global__ static void gather_tiles_kernel(const std::int32_t* __restrict__ vis,
                                           const int2* __restrict__ sel, int n_sel,
                                           int vmat_fstride, int vmat_tstride,
                                           std::int32_t* __restrict__ out) {
    // grid.x spans tiles*512/blockDim; grid.y = nt_outer
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sel * 512)
        return;
    const int k = idx >> 9;      // tile index in the selection
    const int w = idx & 511;     // int32 within the tile
    const int t = blockIdx.y;    // outer time (sub-integration)
    const int2 s = sel[k];       // (freq, int32 offset within the (t,f) slice)
    out[((size_t)t * n_sel + k) * 512 + w] =
        vis[(size_t)t * vmat_tstride + (size_t)s.x * vmat_fstride + s.y + w];
}

cudaError_t launch_gather_gnss_tiles(const std::int32_t* vis, const int2* sel, int n_sel,
                                     int nt_outer, int vmat_fstride, int vmat_tstride,
                                     std::int32_t* out, cudaStream_t stream) {
    if (n_sel <= 0)
        return cudaSuccess;
    const int threads = 256;
    dim3 grid((n_sel * 512 + threads - 1) / threads, nt_outer);
    gather_tiles_kernel<<<grid, threads, 0, stream>>>(vis, sel, n_sel, vmat_fstride, vmat_tstride,
                                                      out);
    return cudaGetLastError();
}

} // namespace gnss_cuda
