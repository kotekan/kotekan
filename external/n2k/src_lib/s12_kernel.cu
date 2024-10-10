#include "../include/n2k/rfi_kernels.hpp"
#include "../include/n2k/internals/internals.hpp"

#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


// s12_kernel() written by Nada El-Falou.
//
// Arguments:
//
//   ulong    S12[Tds][F][2][S];     // output array (length-2 axis is for S1+S2 output arrays)
//   uint4+4  E[Tds*Nds][F][S];      // input array (electric field)
//   int      Nds;                    // time downsampling factor
//   int      Tds;                   // number of time samples after downsampling
//   int      F;                      // number of frequency channels
//   int      S;                      // number of stations
//   int      out_fstride;            // output stride (minimal value is 2*S)
//
// Constraints (checked in launch_s12_kernel below):
//
//   - S must be a multiple of 128.
//   - out_fstride must be a multiple of 4.
//   - out_fstride must be >= 2*S
//
// Parallelization:
//
//   - Each thread processes 1 output time sample (Nds input samples), 1 frequency, and 4 stations.
//
//   - Each warp processes 1 output time sample, 1 frequency, and 128 stations.
//
//   - Each threadblock processes Wz output time samples, Wy frequencies, and (128*Wx) stations.
//
//       threadIdx.x, blockIdx.x <-> station quadruple
//       threadIdx.y, blockIdx.y <-> frequency
//       threadIdx.z, blockIdx.z <-> output time sample


__device__ __forceinline__ uint square(int num) {
    return num * num;
}

__device__ __forceinline__ uint cmplx_square(int real, int imaginary) {
    return square(real) + square(imaginary);
}

__device__ __forceinline__ uint cmplx_tesseract(int real, int imaginary) { 
    return square(cmplx_square(real, imaginary));
}

__device__ __forceinline__ void write_ulong4(ulong *p, ulong a, ulong b, ulong c, ulong d)
{
    *((ulong4 *) p) = ulong4{a,b,c,d};
}

__global__ void __launch_bounds__(128, 8)
s12_kernel(ulong *S12, const uint *E, int Tds, int F, int S, int Nds, int out_fstride, bool offset_encoded)
{
    const uint to_offset_encoded = offset_encoded ? 0 : 0x88888888U;
    
    ulong s1_0, s1_1, s1_2, s1_3, s2_0, s2_1, s2_2, s2_3;
    s1_0 = s1_1 = s1_2 = s1_3 = s2_0 = s2_1 = s2_2 = s2_3 = 0;

    // Each thread processes 4 stations, at the following base (time, freq, station_quadruple) indices.
    ulong tout = (blockIdx.z * blockDim.z) + threadIdx.z;    // coarse time index (no factor Nds)
    uint freq = (blockIdx.y * blockDim.y) + threadIdx.y;     // frequency channel
    uint sq = (blockIdx.x * blockDim.x) + threadIdx.x;       // station quadruple
    uint Q = (S >> 2);                                       // number of quadruples

    if ((tout >= Tds) || (freq >= F) || (sq >= Q))
	return;
    
    // Per-thread pointer shifts.
    // uint S12[Tds][F][2][S];
    // uint E[Tds*Nds][F][Q];
    
    S12 += (tout*F + freq) * out_fstride + (4*sq);
    E += (Nds*F*Q)*tout + Q*freq + sq;
    
    for (uint n = 0; n < Nds; n++) {
        // Get 4 stations (packed into one uint32)
        uint e = E[F*Q*n] ^ to_offset_encoded;   // offset-encoded
        
        // Unpack uint32 into 4 complex numbers (each with real and imaginary components)
        int e0_re = int(e & 0xf) - 8;
        int e0_im = int((e >> 4) & 0xf) - 8;
        int e1_re = int((e >> 8) & 0xf) - 8;
        int e1_im = int((e >> 12) & 0xf) - 8;
	int e2_re = int((e >> 16) & 0xf) - 8;
        int e2_im = int((e >> 20) & 0xf) - 8;
	int e3_re = int((e >> 24) & 0xf) - 8;
	int e3_im = int((e >> 28) & 0xf) - 8;

        // Square/tesseract and sum
        s1_0 += cmplx_square(e0_re, e0_im);
        s1_1 += cmplx_square(e1_re, e1_im);
        s1_2 += cmplx_square(e2_re, e2_im);
        s1_3 += cmplx_square(e3_re, e3_im);

        s2_0 += cmplx_tesseract(e0_re, e0_im);
        s2_1 += cmplx_tesseract(e1_re, e1_im);
        s2_2 += cmplx_tesseract(e2_re, e2_im);
        s2_3 += cmplx_tesseract(e3_re, e3_im);
    }

    write_ulong4(S12, s1_0, s1_1, s1_2, s1_3);
    write_ulong4(S12+S, s2_0, s2_1, s2_2, s2_3);
}


void launch_s12_kernel(ulong *S12, const uint8_t *E, long T, long F, long S, long Nds, long out_fstride, bool offset_encoded, cudaStream_t stream)
{
    // ulong    S12[T/Tds][F][2][S];
    // uint4+4  E[T][F][S];
    
    if (!E || !S12)
	throw runtime_error("launch_s12_kernel(): null array pointer was specified");
    
    // Define some reasonable ranges for integer-valued arguments.
    if ((T <= 0) || (T > 1000000))
	throw runtime_error("launch_s12_kernel(): invalid value of T");
    if ((F <= 0) || (F > 10000))
	throw runtime_error("launch_s12_kernel(): invalid value of F");
    if ((S <= 0) || (S > 10000))
	throw runtime_error("launch_s12_kernel(): invalid value of S");
    if ((Nds <= 0) || (Nds > 10000))
	throw runtime_error("launch_s12_kernel(): invalid value of Nds");

    long Tds = T/Nds;

    // Some more substantial checks.
    if (S % 128)
	throw runtime_error("launch_s12_kernel(): S must be a multiple of 128");		
    if (T != Tds*Nds)
	throw runtime_error("launch_s12_kernel(): T must be a multiple of Nds");	
    if (out_fstride < 2*S)
	throw runtime_error("launch_s12_kernel(): out_fstride must be >= 2*S");	
    if (out_fstride % 4)
	throw runtime_error("launch_s12_kernel(): out_fstride must be a multiple of 4");

    dim3 nblocks, nthreads;
    gputils::assign_kernel_dims(nblocks, nthreads, S/4, F, Tds);
    
    s12_kernel <<< nblocks, nthreads, 0, stream >>>
	(S12, (const uint *) E, Tds, F, S, Nds, out_fstride, offset_encoded);

    CUDA_PEEK("s12 kernel launch");
}



void launch_s12_kernel(Array<ulong> &S12, const Array<uint8_t> &E, long Nds, bool offset_encoded, cudaStream_t stream)
{
    // ulong    S12[T/Tds][F][2][S];
    // uint4+4  E[T][F][S];

    check_array(S12, "launch_s12_kernel", "S12", 4, false);  // ndim=4, contiguous=false
    check_array(E, "launch_s12_kernel", "E", 3, true);       // ndim=3, contiguous=true

    long T = E.shape[0];
    long F = E.shape[1];
    long S = E.shape[2];
    long out_fstride = S12.strides[1];

    if (S12.shape[0] * Nds != E.shape[0])
	throw runtime_error("launch_s12_kernel(): inconsistent number of time samples in S12 and E arrays");
    if (S12.shape[1] != E.shape[1])
	throw runtime_error("launch_s12_kernel(): inconsistent number of frequency channels in S12 and E arrays");
    if (S12.shape[2] != 2)
	throw runtime_error("launch_s12_kernel(): expected axis 2 of S12 array to have length 2");
    if (S12.shape[3] != E.shape[2])
	throw runtime_error("launch_s12_kernel(): inconsistent number of stations in S12 and E arrays");
    
    if ((S12.strides[2] != S) || (S12.strides[3] != 1))
	throw runtime_error("launch_s12_kernel(): expected inner two axes (with shape (2,S)) of S12 array to be contiguous");
    if (S12.strides[0] != F*out_fstride)
	throw runtime_error("launch_s12_kernel(): expected time+freq axes of S12 to be contiguous");

    launch_s12_kernel(S12.data, E.data, T, F, S, Nds, out_fstride, offset_encoded, stream);
}


}  // namespace n2k
