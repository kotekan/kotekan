#include "../include/n2k/pl_kernels.hpp"
#include "../include/n2k/internals/internals.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;

namespace n2k {
#if 0
}  // editor auto-indent
#endif


// FIXME can be improved.
__device__ inline uint2 double_bits(uint x)
{
    uint2 ret{0U,0U};
    
    for (uint i = 0; i < 16; i++) {
	uint x_shifted = x >> i;
	uint bit_pair = 3U << (2*i);
	
	if (x_shifted & 1U)
	    ret.x |= bit_pair;
	if (x_shifted & 0x10000U)
	    ret.y |= bit_pair;
    }

    return ret;
}


// Kernel args:
//   F = (number of output freqs)
//   M = (number of input times) / 64
//   N = (number of stations) * 2
//   pl_out = uint array of shape (2*M, F, N)
//   pl_in = uint array of shape (M, (F+3)/4, N)

__global__ void pl_mask_expand_kernel(uint *pl_out, const uint *pl_in, int F, int M, int N)
{
    const int Fin = (F+3) >> 2;
    
    // Parallelization: x <-> n, y <-> f, z <-> m
    int n = (blockIdx.x * blockDim.x) + threadIdx.x;
    int f = (blockIdx.y * blockDim.y) + threadIdx.y; 
    int m = (blockIdx.z * blockDim.z) + threadIdx.z;
    
    bool valid = (f < Fin) && (m < M) && (n < N);
    int nf_out = valid ? min(F-4*f,4) : 0;

    // Ensure array accesses are within bounds.
    f = (f < Fin) ? f : (Fin-1);
    m = (m < M) ? m : (M-1);
    n = (n < N) ? n : (N-1);
    
    // pl_in = uint array of shape (M, Fin, N)
    // After these shifts, 'pl_in' points to a scalar.
    
    pl_in += long(m) * long(Fin*N);
    pl_in += (f*N + n);

    // pl_out = uint array of shape (2*M, F, N)
    // After these shifts, 'pl_out' points to an array of shape (nf_out, 2) with strides (N, 1).

    int mout = 2*m + (n & 1);
    int nout = n & ~1;
    pl_out += long(mout) * long(F*N);
    pl_out += (4*f*N + nout);

    // Read input mask.
    uint x = *pl_in;
    uint2 y = double_bits(x);
	
    // Write (expanded) output mask.
    for (int i = 0; i < nf_out; i++)
	*((uint2 *) (pl_out + i*N)) = y;
}


void launch_pl_mask_expander(ulong *pl_out, const ulong *pl_in, long Tout, long Fout, long S, cudaStream_t stream)
{
    // FIXME asserts -> exceptions.
    assert(pl_out != nullptr);
    assert(pl_in != nullptr);
    assert(Tout > 0);
    assert(Fout > 0);
    assert(S > 0);

    assert((Tout % 128) == 0);
    assert((S % 16) == 0);

    // FIXME check for 32-bit overflows.
    
    long M = Tout / 128;
    long N = S * 2;

    dim3 nblocks, nthreads;
    gputils::assign_kernel_dims(nblocks, nthreads, N, Fout, M);  // x <-> n, y <-> f, z <-> m

    pl_mask_expand_kernel <<< nblocks, nthreads, 0, stream >>>
	((uint *) pl_out, (const uint *) pl_in, Fout, M, N);

    CUDA_PEEK("pl_mask_expand_kernel");
}


void launch_pl_mask_expander(Array<ulong> &pl_out, const Array<ulong> &pl_in, cudaStream_t stream)
{
    check_array(pl_out, "launch_pl_mask_expander", "pl_out", 3, true);  // contiguous=true
    check_array(pl_in, "launch_pl_mask_expander", "pl_in", 3, true);    // contiguous=true

    long Tout = 64 * pl_out.shape[0];
    long Fout = pl_out.shape[1];
    long S = pl_out.shape[2];

    if (!pl_in.shape_equals({Tout/128, (Fout+3)/4, S})) {
	stringstream ss;
	ss << "launch_pl_mask_expander: pl_out.shape=" << pl_out.shape_str()
	   << " and pl_in.shape=" << pl_in.shape_str()
	   << " are inconsistent (expected pl_in.shape=("
	   << (Tout/128) << "," << ((Fout+3)/4) << "," << S 
	   << "))";
	throw runtime_error(ss.str());
    }

    launch_pl_mask_expander(pl_out.data, pl_in.data, Tout, Fout, S, stream);
}


}  // namespace n2k
