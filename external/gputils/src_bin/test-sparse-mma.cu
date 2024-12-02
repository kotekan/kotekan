// This test verifies that the mma.sp.* instruction does what I think it does,
// since the PTX documentation is hard to understand!
//
// Currently, only the following PTX instruction is tested:
//   mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
//
// which gets called via the following C++ wrapper (in device_mma.hpp):
//
//   void mma_sp_f16_m16_n8_k16<F> (__half2 d[2], const __half2 a[2], const __half2 b[2],
//                                  const __half2 c[2], unsigned int e);

#include <iostream>
#include "../include/gputils.hpp"

using namespace std;
using namespace gputils;


// -------------------------------------------------------------------------------------------------


// The 'p' argument points to 128 floats, in ordering (r,t,b)
__device__ void read_fragment(__half2 dst[2], const float *p)
{
    int i = threadIdx.x & 0x1f;
    dst[0] = __floats2half2_rn(p[2*i], p[2*i+1]);
    dst[1] = __floats2half2_rn(p[2*i+64], p[2*i+65]);
}


__device__ void write_fragment(float *p, const __half2 src[2])
{
    int i = threadIdx.x & 0x1f;

    float2 r0 = __half22float2(src[0]);
    p[2*i] = r0.x;
    p[2*i+1] = r0.y;
    
    float2 r1 = __half22float2(src[1]);
    p[2*i+64] = r1.x;
    p[2*i+65] = r1.y;
}


template<unsigned int F>
__global__ void mma_sp_kernel(float *dp, const float *ap, const float *bp, const float *cp, unsigned int *ep)
{
    __half2 a[2], b[2], c[2];
    read_fragment(a, ap);
    read_fragment(b, bp);
    read_fragment(c, cp);

    __half2 d[2];
    mma_sp_f16_m16_n8_k16<F> (d, a, b, c, ep[threadIdx.x & 0x1f]);
    write_fragment(dp, d);
}


static void launch_mma_sp_kernel(float *dp, const float *ap, const float *bp, const float *cp, unsigned int *ep, int f)
{
    if (f == 0)
	mma_sp_kernel<0> <<<1,32>>> (dp, ap, bp, cp, ep);
    else if (f == 1)
	mma_sp_kernel<1> <<<1,32>>> (dp, ap, bp, cp, ep);
    else if (f == 2)
	mma_sp_kernel<2> <<<1,32>>> (dp, ap, bp, cp, ep);
    else if (f == 3)
	mma_sp_kernel<3> <<<1,32>>> (dp, ap, bp, cp, ep);
    else
	throw runtime_error("bad value of f");

    CUDA_PEEK("launch mma_sp_kernel");
}


// -------------------------------------------------------------------------------------------------


static Array<float> unpack_bmat(const Array<float> &b_arr)
{
    // Unpack matrix B_{jk}, with register mapping
    //   b0 <-> j0     r0 <-> j3     t0 t1 t2 t3 t4 <-> j1 j2 k0 k1 k2

    assert(b_arr.shape_equals({2,32,2}));   // (r,t,b)
    
    Array<float> b_mat({16,8}, af_rhost);
    
    for (int j = 0; j < 16; j++) {
	for (int k = 0; k < 8; k++) {
	    int b = j & 0x1;
	    int r = j >> 3;
	    int t = ((j>>1) & 0x3) + (k<<2);
	    b_mat.at({j,k}) = b_arr.at({r,t,b});
	}
    }

    return b_mat;
}


static Array<float> unpack_cmat(const Array<float> &c_arr)
{
    // Unpack matrix C_{ik}, with register mapping
    //   b0 <-> k0     r0 <-> i3     t0 t1 t2 t3 t4 <-> k1 k2 i0 i1 i2

    assert(c_arr.shape_equals({2,32,2}));   // (r,t,b)
    
    Array<float> c_mat({16,8}, af_rhost);

    for (int i = 0; i < 16; i++) {
	for (int k = 0; k < 8; k++) {
	    int b = k & 0x1;
	    int r = i >> 3;
	    int t = (k>>1) + ((i & 0x7) << 2);
	    c_mat.at({i,k}) = c_arr.at({r,t,b});
	}
    }

    return c_mat;
}


static Array<float> unpack_amat(const Array<float> &a_arr, const Array<unsigned int> &e_arr, int f)
{
    // Unpack matrix A_{ij}, with register mapping
    //    [A]  b <-> j0 j1    r <-> i3    t0 t1 t2 t3 t4 <-> j2 j3 i0 i1 i2
    //    [E]  b0 b1 b2 <-> j2 j3 i3      t0 t1 t2 t3 t4 <-> f0 f1 i0 i1 i2

    assert(a_arr.shape_equals({2,32,2}));   // (r,t,b)
    assert(e_arr.shape_equals({32}));
    assert((f >= 0) && (f < 4));

    Array<float> a_mat({16,16}, af_rhost | af_zero);
	
    for (int i = 0; i < 16; i++) {
	for (int j23 = 0; j23 < 4; j23++) {
	    for (int b = 0; b < 2; b++) {
		int ra = i >> 3;
		int ta = ((i & 0x7) << 2) + j23;
		int te = ((i & 0x7) << 2) + f;
		int be = ((i & 0x8) >> 1) + j23;
		
		for (int ba = 0; ba < 2; ba++) {
		    int j01 = (e_arr.at({te}) >> (4*be + 2*ba)) & 0x3;
		    a_mat.at({i,4*j23+j01}) = a_arr.at({ra,ta,ba});
		}
	    }
	}
    }

    return a_mat;
}


static Array<unsigned int> make_random_e_array()
{
    Array<unsigned int> e_arr({32}, af_rhost | af_zero);

    for (int i = 0; i < 32; i++) {
	for (int j = 0; j < 8; j++) {
	    // Random 4-bit selector
	    unsigned int lo = rand_int(0,4);  // low 2 bits
	    unsigned int hi = rand_int(0,3);  // high 2 bits
	    if (hi >= lo)
		hi++;

	    e_arr.at({i}) |= (lo << (4*j));
	    e_arr.at({i}) |= (hi << (4*j+2));
	}
    }

    return e_arr;
}


static void test_sparse_mma()
{
    Array<float> a_arr({2,32,2}, af_rhost | af_random);
    Array<float> b_arr({2,32,2}, af_rhost | af_random);
    Array<float> c_arr({2,32,2}, af_rhost | af_random);
    Array<unsigned int> e_arr = make_random_e_array();

    Array<float> a_gpu = a_arr.to_gpu();
    Array<float> b_gpu = b_arr.to_gpu();
    Array<float> c_gpu = c_arr.to_gpu();
    Array<unsigned int> e_gpu = e_arr.to_gpu();

    Array<float> b_mat = unpack_bmat(b_arr);
    Array<float> c_mat = unpack_cmat(c_arr);

    for (int f = 0; f < 4; f++) {
	Array<float> d_gpu({2,32,2}, af_gpu);
	launch_mma_sp_kernel(d_gpu.data, a_gpu.data, b_gpu.data, c_gpu.data, e_gpu.data, f);

	Array<float> a_mat = unpack_amat(a_arr, e_arr, f);
	Array<float> d_mat = unpack_cmat(d_gpu.to_host());

	// "Expected" D-matrix (C + A*B)
	Array<float> d_exp = c_mat.clone();
	for (int i = 0; i < 16; i++)
	    for (int j = 0; j < 16; j++)
		for (int k = 0; k < 8; k++)
		    d_exp.at({i,k}) += a_mat.at({i,j}) * b_mat.at({j,k});

	// cout << "test_sparse_mma(f=" << f << ")" << endl;
	assert_arrays_equal(d_mat, d_exp, "D_gpu", "D_exp", {"i","k"}, 0.01);
    }
}


int main(int argc, char **argv)
{
    // Implements command-line usage: program [device].
    set_device_from_command_line(argc, argv);

    for (int i = 0; i < 100; i++)
	test_sparse_mma();

    cout << "test-sparse-mma: pass\n";
    return 0;
}
