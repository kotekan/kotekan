#ifndef _N2K_COORELATOR_KERNEL_HPP
#define _N2K_CORRELATOR_KERNEL_HPP

#include "Correlator.hpp"
#include <gputils/constexpr_functions.hpp>   // constexpr_is_divisible()


namespace n2k {
#if 0
}  // editor auto-indent
#endif


// Compile-time parameters:
//
//    NS = number of stations (one "station" is a (dish,polarization) pair).
//    NF = number of frequencies
//
// The kernel runs faster if (NS,NF) are known at compile time, so we make them template parameters.
// Currently, the template is instantiated for the specific cases:
//
//    NS=128    NF=1,2,4,8,16,32,64,128,256,512,1024   (CHORD pathfinder is NF=128)
//    NS=1024   NF=1,2,4,8,16,32,64,128                (Full CHORD is NF=16)
//
// Additional values of (NS,NF) can easily be instantiated with a one-line change to the Makefile.
// We currently have no way of adding new values of (NS,NF) at runtime (could use nvrtc for this).


template<int NS, int NF>
struct CorrelatorKernel
{
    static_assert(gputils::constexpr_is_divisible(NS,128));
    
    static constexpr int emat_fstride = NS/4;        // int32 stride, not bytes
    static constexpr int emat_tstride = NF*(NS/4);   // int32 stride, not bytes
    static constexpr int vmat_fstride = NS*(NS+16);  // int32 stride, not int32+32

    
    // --------------------------------------   Misc helpers   -----------------------------------------


    // Given a 32-bit register 'x' containing eight signed 4-bit integers x[0:8], negate x[i] -> -x[i].
    // WARNING: assumes x[i] is in the range [-7,7], i.e. fails if any x[i] is (-8)!

    static __device__ int negate_4bit(int x)
    {
	if constexpr (!CorrelatorParams::artificially_remove_negate_4bit) {
	    x ^= 0x77777777;
	    x += 0x11111111;
	    x ^= 0x88888888;
	}
	return x;
    }

    // The nvidia LOP3 instruction.
    // Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3
    
    template<int immLut>
    static __device__ int lop3(int a, int b, int c)
    {
	int d;
	
	asm("lop3.b32 %0, %1, %2, %3, %4;" :
	    "=r"(d) : "r"(a), "r"(b), "r"(c), "n"(immLut)
	);
	
	return d;
    }
    
    // blend(): equivalent to (c & a) | (~c & b), but using a single LOP3 instruction.
    
    static __device__ int blend(int a, int b, int c)
    {
	constexpr int A = 0xf0;
	constexpr int B = 0xcc;
	constexpr int C = 0xaa;
	constexpr int N = (C & A) | (~C & B);
	
	return lop3<N> (a, b, c);
    }

    
    // ------------------------------------   In-register transposes  ----------------------------------

    // The purpose of these functions is to implement transpose_rank8_4bit(), which takes an 8-by-8
    // array of 4-bit integers, represented as eight 32-bit registers, and performs a logical transpose.
    //
    // We build up transpose_rank8_4bit() as a sequence of smaller transposes.
    

    static __device__ void transpose_rank2_4bit(int &x, int &y)
    {
	int xr = (x >> 4);
	int yl = (y << 4);
	const int z = 0x0f0f0f0f;
	
	x = blend(x, yl, z);
	y = blend(xr, y, z);
    }


    static __device__ void transpose_rank2_8bit(int &x, int &y)
    {
	int tmp = __byte_perm(x, y, 0x6240);
	y = __byte_perm(x, y, 0x7351);
	x = tmp;
    }


    static __device__ void transpose_rank2_16bit(int &x, int &y)
    {
	int tmp = __byte_perm(x, y, 0x5410);
	y = __byte_perm(x, y, 0x7632);
	x = tmp;
    }


    static __device__ void transpose_rank4_4bit(int &a, int &b, int &c, int &d)
    {
	transpose_rank2_4bit(a, b);
	transpose_rank2_4bit(c, d);
	transpose_rank2_8bit(a, c);
	transpose_rank2_8bit(b, d);
    }


    static __device__ void transpose_rank8_4bit(int &a, int &b, int &c, int &d, int &e, int &f, int &g, int &h)
    {
	transpose_rank4_4bit(a, b, c, d);
	transpose_rank4_4bit(e, f, g, h);
	
	transpose_rank2_16bit(a, e);
	transpose_rank2_16bit(b, f);
	transpose_rank2_16bit(c, g);
	transpose_rank2_16bit(d, h);
    }
    
    
    static __device__ void transpose_rank8_4bit(int x[8])
    {
	transpose_rank8_4bit(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
    }
    
    
    // -------------------------------------------------------------------------------------------------
    //
    // Single-fragment inlines
    

    static __device__ void mma(int c[4], const int a[4], const int b[2], const int d[4])
    {
	asm("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
	    "{%0, %1, %2, %3}, "
	    "{%4, %5, %6, %7}, "
	    "{%8, %9}, "
	    "{%10, %11, %12, %13};" :
	    "=r" (c[0]), "=r" (c[1]), "=r" (c[2]), "=r" (c[3]) :
	    "r" (a[0]), "r" (a[1]), "r" (a[2]), "r" (a[3]),
	    "r" (b[0]), "r" (b[1]),
	    "r" (d[0]), "r" (d[1]), "r" (d[2]), "r" (d[3])
	);
    }

    
    static __device__ void load_A_fragment(int A[2][4], const int *__restrict__ ap)
    {
	// From overleaf: (r0 r1) <-> (i3 j5) <-> (8 stations, 32 times)
	constexpr int S0 = CorrelatorParams::shmem_s8_stride;
	constexpr int S1 = CorrelatorParams::shmem_t32_stride;
	constexpr int SIm = CorrelatorParams::shmem_reim_stride;
	
	// Real part
	A[0][0] = ap[0];
	A[0][1] = ap[S0];
	A[0][2] = ap[S1];
	A[0][3] = ap[S0+S1];
	
	// Imaginary part
	A[1][0] = ap[SIm];
	A[1][1] = ap[SIm+S0];
	A[1][2] = ap[SIm+S1];
	A[1][3] = ap[SIm+S0+S1];
    }


    static __device__ void load_B_fragment(int B[3][2], const int *__restrict__ bp)
    {
	// From overleaf: r0 <-> j5 <-> (32 times)
	constexpr int S = CorrelatorParams::shmem_t32_stride;
	constexpr int SIm = CorrelatorParams::shmem_reim_stride;
	
	// Real part
	B[0][0] = bp[0];
	B[0][1] = bp[S];
	
	// Imaginary part
	B[1][0] = bp[SIm];
	B[1][1] = bp[SIm+S];
	
	// Negative imaginary part
	B[2][0] = negate_4bit(B[1][0]);
	B[2][1] = negate_4bit(B[1][1]);
    }


    // Correlate one V-fragment
    static __device__ void correlate_fragment(int V[2][4], const int A[2][4], const int B[3][2])
    {
	mma(V[0], A[0], B[0], V[0]);   // (Re A) * (Re B)
	mma(V[0], A[1], B[1], V[0]);   // (Im A) * (Im B)
	mma(V[1], A[0], B[2], V[1]);   // (Re A) * (-Im B)
	mma(V[1], A[1], B[0], V[1]);   // (Im A) * (Re B)
    }
    

    template<int N>
    static __device__ void split_fragment(const int A[2][4], int B[3][2])
    {
	// Real part
	B[0][0] = A[0][N];
	B[0][1] = A[0][N+2];
	
	// Imaginary part
	B[1][0] = A[1][N];
	B[1][1] = A[1][N+2];
	
	// Negative imaginary part
	B[2][0] = negate_4bit(B[1][0]);
	B[2][1] = negate_4bit(B[1][1]);
    }


    // -------------------------------------------------------------------------------------------------
    //
    // Prefetching: (global memory) -> (registers) -> (shared memory)


    // prefetch_chunk():
    //  - each warp reads (either A or B) x (8 times).
    //  - the 8-warp threadblock reads (A and B) x (32 times).
    
    static __device__ const int *prefetch_chunk(const int *__restrict__ gp, int pf_out[8])
    {
	constexpr int TS = emat_tstride;
	
	// I tried using __ldcg() here instead of a normal load (caches in L2 but not L1),
	// but this turned out to make the kernel slightly slower.
	
	#pragma unroll
	for (int i = 0; i < 8; i++)
	    pf_out[i] = gp[i*TS];   // FIXME use wide load instructions!
	
	return gp + (32*TS);
    }

    
    static __device__ void store_prefetched_chunk(int *__restrict__ sp, int pf_in[8], uint rm)
    {
	const int t8 = threadIdx.x >> 6;  // satisfies 0 <= t8 < 4, indicates 8-time chunk for this warp.
	rm >>= (t8 << 3);                 // shift rfimask so that bits [0,8) apply to this warp.

	#pragma unroll
	for (int r = 0; r < 8; r++)
	    pf_in[r] = (rm & (1 << r)) ? pf_in[r] : 0;
	
	// Transpose data ordering in registers.
	if constexpr (!CorrelatorParams::artificially_remove_input_shuffle)
	    transpose_rank8_4bit(pf_in);
	
	// After transpose: (r0, r1, r2) <-> (ReIm, s1, s2)
	constexpr int S0 = CorrelatorParams::shmem_reim_stride;
	constexpr int S1 = CorrelatorParams::shmem_s1_stride;
	
	if constexpr (CorrelatorParams::real_part_in_low_bits) {
	    // Real part in low 4 bits, imaginary part in high 4 bits.
	    sp[0] = pf_in[0];
	    sp[S0] = pf_in[1];
	    sp[S1] = pf_in[2];
	    sp[S1+S0] = pf_in[3];
	    sp[2*S1] = pf_in[4];
	    sp[2*S1+S0] = pf_in[5];
	    sp[3*S1] = pf_in[6];
	    sp[3*S1+S0] = pf_in[7];
	}
	else {
	    // Imaginary part in low 4 bits, real part in high 4 bits.
	    sp[0] = pf_in[1];
	    sp[S0] = pf_in[0];
	    sp[S1] = pf_in[3];
	    sp[S1+S0] = pf_in[2];
	    sp[2*S1] = pf_in[5];
	    sp[2*S1+S0] = pf_in[4];
	    sp[3*S1] = pf_in[7];
	    sp[3*S1+S0] = pf_in[6];
	}
    }

    
    // Called by kernel_body().
    // Caller does not need to initialize 'pf'.
    // On exit, 'sp' will be filled with 128 time samples, and 'pf' will be filled with the next 32 time samples.
    
    static __device__ const int *do_initial_prefetch(const int *__restrict__ gp, int *__restrict__ sp, int pf[8], uint rm)
    {
	constexpr uint ALL_LANES = 0xffffffffU;
	constexpr int S = CorrelatorParams::shmem_t32_stride;
	
	int tmp[8];

	gp = prefetch_chunk(gp, pf);
	
	gp = prefetch_chunk(gp, tmp);
	store_prefetched_chunk(sp, pf, __shfl_sync(ALL_LANES, rm, 0));
	
	gp = prefetch_chunk(gp, pf);
	store_prefetched_chunk(sp+S, tmp, __shfl_sync(ALL_LANES, rm, 1));
	
	gp = prefetch_chunk(gp, tmp);
	store_prefetched_chunk(sp+2*S, pf, __shfl_sync(ALL_LANES, rm, 2));
	
	gp = prefetch_chunk(gp, pf);
	store_prefetched_chunk(sp+3*S, tmp, __shfl_sync(ALL_LANES, rm, 3));

	return gp;
    }


    // -------------------------------------------------------------------------------------------------
    //
    // These core computational routines compute tiles of the visibility matrix, while interleaving
    // prefetch operations.
    

    // Correlate full tile (32-by-64 matrix for each warp) over 64 time samples.
    //
    // The 'P' template parameter controls the level of prefetching.
    //
    //   P=0: Caller should pass old prefetched data in 'pf'.
    //        64 time samples will be written to 'sp'.
    //        New prefetched data is stored in 'pf' (overwriting old data).
    //
    //   P=1: Caller should pass old prefetched data in 'pf'.
    //        64 time samples will be written to 'sp'.
    //        Nothing will be stored in 'pf'.
    //
    //   P=2: No prefetching is performed; nothing is written to 'sp'.

    
    template<int P>
    static __device__ const int *
    correlate_t64(int V[8][2][2][4], const int *__restrict__ ap, const int *__restrict__ bp, const int *__restrict__ gp, int *__restrict__ sp, int pf[8], uint rm, int t0)
    {
	constexpr uint ALL_LANES = 0xffffffffU;
	constexpr int SA = 2 * CorrelatorParams::shmem_s8_stride;  // one A-fragment corresponds to 16 stations
	constexpr int SB = CorrelatorParams::shmem_s8_stride;
	constexpr int SS = CorrelatorParams::shmem_t32_stride;
    
	int A[2][2][4];
	int B[3][2];
	int pf2[8];
	
	load_A_fragment(A[0], ap);
	load_A_fragment(A[1], ap+SA);
	
	// First 32 time samples

	if constexpr (P <= 1)
	    gp = prefetch_chunk(gp, pf2);
	
	#pragma unroll
	for (int fy = 0; fy < 4; fy++) {
	    load_B_fragment(B, bp + fy*SB);
	    correlate_fragment(V[fy][0], A[0], B);
	    correlate_fragment(V[fy][1], A[1], B);
	}

	if constexpr (P <= 1)
	    store_prefetched_chunk(sp, pf, __shfl_sync(ALL_LANES, rm, t0 >> 5));

	// Second 32 time samples
	//   V[fy] -> V[fy+4]
	//   bp -> bp + 4*SB
	//   gp -> gp + 32*SG
	//   sp -> sp + SS
	//   pf <-> pf2

	if constexpr (P == 0)
	    gp = prefetch_chunk(gp, pf);

	#pragma unroll
	for (int fy = 0; fy < 4; fy++) {
	    load_B_fragment(B, bp + (fy+4)*SB);
	    correlate_fragment(V[fy+4][0], A[0], B);
	    correlate_fragment(V[fy+4][1], A[1], B);
	}

	if constexpr (P <= 1)
	    store_prefetched_chunk(sp+SS, pf2, __shfl_sync(ALL_LANES, rm, (t0>>5)+1));

	return gp;
    }


    // Correlate full tile (32-by-64 matrix for each warp) over 128 time samples. This function is a
    // trivial wrapper around two calls to correlate_t64(). The point of defining this wrapper is that
    // 128 time samples is the longest interval over which the __restrict__ keywords are valid.
    //
    // The 'P' template parameter controls the level of prefetching.
    //
    //   P=0: Caller should pass old prefetched data in 'pf'.
    //        128 time samples will be written to 'sp'.
    //        New prefetched data is stored in 'pf' (overwriting old data).
    //
    //   P=1: Caller should pass old prefetched data in 'pf'.
    //        128 time samples will be written to 'sp'.
    //        Nothing will be stored in 'pf'.
    //
    //   P=2: No prefetching is performed; nothing is written to 'sp'.

    template<int P>
    static __device__ const int *
    correlate_t128(int V[8][2][2][4], const int *__restrict__ ap, const int *__restrict__ bp, const int *__restrict__ gp, int *__restrict__ sp, int pf[8], uint rm, int t0)
    {
	constexpr int S = 2 * CorrelatorParams::shmem_t32_stride;
	constexpr int Q = (P < 2) ? 0 : 2;

	gp = correlate_t64<Q> (V, ap, bp, gp, sp, pf, rm, t0);
	gp = correlate_t64<P> (V, ap+S, bp+S, gp, sp+S, pf, rm, t0+64);
	return gp;
    }


    // ----------------------------------   V-matrix write path   --------------------------------------

    
    static __device__ void zero_V(int V[8][2][2][4])  // (Y,X,ReIm,frag)
    {
	#pragma unroll
	for (int fy = 0; fy < 8; fy++)
	    #pragma unroll
	    for (int fx = 0; fx < 2; fx++)
		#pragma unroll
		for (int i = 0; i < 2; i++) 
		    #pragma unroll
		    for (int j = 0; j < 4; j++)
			V[fy][fx][i][j] = 0;
    }

    
    // 'bit' should be a power of two
    static __device__ void warp_transpose(int &in0, int &in1, const int bit)
    {
	int flag = (threadIdx.x & bit);
	int src = flag ? in0 : in1;
	int dst = __shfl_xor_sync(0xffffffff, src, bit);
	(flag ? in0 : in1) = dst;
    }

    
    static __device__ void warp_transpose_4(int in0[4], int in1[4], const int bit)
    {
	warp_transpose(in0[0], in1[0], bit);
	warp_transpose(in0[1], in1[1], bit);
	warp_transpose(in0[2], in1[2], bit);
	warp_transpose(in0[3], in1[3], bit);
    }

    
    static __device__ void write_int4(int *__restrict__ p, int a, int b, int c, int d)
    {
	int4 x = make_int4(a,b,c,d);
	__stcs((int4 *) p, x);  // streaming write is slightly faster than normal store.
    }
    

    // The arrays V0[q][r], V1[q][r] represent a 16-by-16 complex matrix, with register assignment:
    //   V01 <-> k3    q <-> ReIm    r0 r1 <-> k0 i3    t0 t1 t2 t3 t4 <-> k1 k2 i0 i1 i2    (*)
    //
    // Note that the first step in write_V_16_16() is warp-transposing to register assignment:
    //   V01 <-> i0    q <-> ReIm    r0 r1 <-> k0 i3    t0 t1 t2 t3 t4 <-> k1 k2 k3 i1 i2    (**)
    //
    // The 'vp_thread' argument is confusing! It's the output pointer after applying per-thread
    // offsets in register assignment (**) [not (*)]. In particular, pointer offsets for (i1,i2)
    // and (k1,k2,k3) should be applied, but not i0.
    
    static __device__ void write_V_16_16(int *__restrict__ vp_thread, int V0[2][4], int V1[2][4])
    {
	if constexpr (!CorrelatorParams::artificially_remove_output_shuffle) {
	    warp_transpose_4(V0[0], V1[0], 0x04);
	    warp_transpose_4(V0[1], V1[1], 0x04);
	}

	// After the warp transposes, the register assignment is:
	//   V01 <-> i0    q <-> ReIm    r0 r1 <-> k0 i3    t0 t1 t2 t3 t4 <-> k1 k2 k3 i1 i2
	
	write_int4(vp_thread, V0[0][0], V0[1][0], V0[0][1], V0[1][1]);        // (i0,i3) = (0,0)
	write_int4(vp_thread + 32, V1[0][0], V1[1][0], V1[0][1], V1[1][1]);   // (i0,i3) = (1,0)
	write_int4(vp_thread + 256, V0[0][2], V0[1][2], V0[0][3], V0[1][3]);  // (i0,i3) = (0,1)
	write_int4(vp_thread + 288, V1[0][2], V1[1][2], V1[0][3], V1[1][3]);  // (i0,i3) = (1,1)
    }
    

    // Writes full V-matrix tile (32-by-64 matrix for each warp).
    // The 'vp_tf' argument has (t,f) offsets applied, but no (i,j) offsets (even per-block offsets)
    // Note that vi_warp is a multiple of 32, and vk_warp is a multiple of 64.
    //
    // Outer three V indices are V[k//8][i//16][ReIm], and innermost int[4] is a 16-by-8 fragment, with:
    //    r0 r1 <-> k0 i3    t0 t1 t2 t3 t4 <-> k1 k2 i0 i1 i2

    static __device__ void write_V(int *__restrict__ vp_tf, int V[8][2][2][4], int vi_warp, int vk_warp)
    {
	int *vp_thread = vp_tf
	    + vi_warp * (vi_warp+16) + (vk_warp << 5)   // tile
	    + ((threadIdx.x & 0x7) << 2)    // t0 t1 t2 <-> k1 k2 k3
	    + ((threadIdx.x & 0x18) << 3);  // t3 t4 <-> i1 i2

	// Divide 32-by-64 array into a 2-by-4 array of 16-by-16 tiles.
	// Let s = int32 offset between tile (0,0) and (1,0).
	int s = (vi_warp+16) << 5;
	
	if (vi_warp < vk_warp)
	    return;

	write_V_16_16(vp_thread, V[0][0], V[1][0]);               // tile (0,0)
	write_V_16_16(vp_thread + s, V[0][1], V[1][1]);           // tile (1,0)
	write_V_16_16(vp_thread + s + 512, V[2][1], V[3][1]);     // tile (1,1)

	if (vi_warp < vk_warp+32)
	    return;

	write_V_16_16(vp_thread + 512, V[2][0], V[3][0]);         // tile (0,1)
	write_V_16_16(vp_thread + 1024, V[4][0], V[5][0]);        // tile (0,2)
	write_V_16_16(vp_thread + s + 1024, V[4][1], V[5][1]);    // tile (1,2)
	write_V_16_16(vp_thread + s + 1536, V[6][1], V[7][1]);    // tile (1,3)
	
	if (vi_warp < vk_warp+64)
	    return;

	write_V_16_16(vp_thread + 1536, V[6][0], V[7][0]);        // tile (0,3)
    }


    // -------------------------------------------------------------------------------------------------
    
    
    static __device__ void
    kernel_body(int *dst, const int8_t *src, const uint *rfimask, const int *ptable, int nt_inner)
    {
	extern __shared__ int shmem[];

	// Initialize pointers.
	
	const int f = blockIdx.y;   // frequency index
	const int n = blockDim.x;   // ptable stride (for length-6 axis)
	const int i = blockIdx.x * (6*blockDim.x) + threadIdx.x;  // base index in ptable array
	
	const int *ap = shmem + ptable[i];
	const int *bp = shmem + ptable[i+n];
	const int *gp = ((const int *) src) + (f * emat_fstride) + ptable[i+2*n];
	int *sp = shmem + ptable[i+3*n];

	const int touter = blockIdx.z;
	gp += ssize_t(touter * nt_inner) * emat_tstride;

	// RFI mask stored in (f,t) ordering
	rfimask += (f*gridDim.z + blockIdx.z) * (nt_inner >> 5);  // apply (f, touter) offsets
	rfimask += (threadIdx.x & 0x1f);                          // apply laneId
		
	// Initialize correlator state.

	uint rm = *rfimask;
	rfimask += 32;
	
	int V[8][2][2][4];
	zero_V(V);

	// Store first 128 time samples in shared memory, and next 32 time samples in pf[].
	int pf[8];
	gp = do_initial_prefetch(gp, sp, pf, rm);
	__syncthreads();
	
	// Main correlator loop.
	
	constexpr int S = 2 * CorrelatorParams::shmem_t32_stride;

	for (int t = 256; t < nt_inner; t += 256) {
	    // Correlate samples [t-256,t-128)
	    // Write samples [t-128,t) to shared memory.
	    // Prefetch samples [t,t+32) to registers.
	    
	    gp = correlate_t128<0> (V, ap, bp, gp, sp+2*S, pf, rm, t-128);
	    __syncthreads();

	    if ((t & 0x300) == 0) {
		rm = *rfimask;
		rfimask += 32;
	    }
	    
	    // Correlate samples [t-128,t)
	    // Write samples [t,t+128) to shared memory.
	    // Prefetch samples [t+128,t+160) to registers.

	    gp = correlate_t128<0> (V, ap+2*S, bp+2*S, gp, sp, pf, rm, t);
	    __syncthreads();
	}

	gp = correlate_t128<1> (V, ap, bp, gp, sp+2*S, pf, rm, nt_inner-128);
	__syncthreads();
	
	gp = correlate_t128<2> (V, ap+2*S, bp+2*S, gp, sp, pf, rm, nt_inner);
	
	// Write visibility matrix to global memory.
	// The 'vp_tf' pointer has (t,f) offsets applied, but no (i,j) offsets (even per-block offsets)
	int *vp_tf = dst + long(touter*NF+f) * vmat_fstride;
	
	int vi_warp = ptable[i+4*n];
	int vk_warp = ptable[i+5*n];
	write_V(vp_tf, V, vi_warp, vk_warp);
    }


    // We define one non-static member function: a constructor which instantiates
    // the kernel template, and "registers" it (see kernel_table.cu) so that it
    // can be called through Correlator::launch().
    
    __host__ CorrelatorKernel();
};


// This wrapper is only needed because cuda doesn't allow the static class
// member function CorrelatorKernel<NS,NF>::kernel_body() to be __global__.

template<int NS, int NF>
__global__ void __launch_bounds__(CorrelatorParams::threads_per_block, 1)
n2k_kernel(int *dst, const int8_t *src, const uint *rfimask, const int *ptable, int ntime)
{    
    CorrelatorKernel<NS,NF>::kernel_body(dst, src, rfimask, ptable, ntime);
}


// The CorrelatorKernel constructor instantiates the kernel and registers it
// (see kernel_table.cu).

template<int NS, int NF>
CorrelatorKernel<NS,NF>::CorrelatorKernel()
{
    register_kernel(NS, NF, n2k_kernel<NS,NF>);
}


}  // namespace n2k

#endif // _N2K_CORRELATOR_KERNEL_HPP
