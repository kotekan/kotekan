#include "../include/n2k/rfi_kernels.hpp"
#include "../include/n2k/internals/internals.hpp"
#include "../include/n2k/internals/interpolation.hpp"
#include "../include/n2k/internals/bad_feed_mask.hpp"

#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/cuda_utils.hpp>
#include <gputils/rand_utils.hpp>
#include <gputils/test_utils.hpp>

using namespace std;
using namespace n2k;
using namespace gputils;



// FIXME improve make_random_s012_array()?
static Array<ulong> make_random_s012_array(int T, int F, int S)
{
    std::mt19937 &rng = gputils::default_rng;
    auto dist = std::uniform_int_distribution<uint>(0, 1000);

    Array<ulong> ret({T,F,3,S}, af_rhost);
    for (long i = 0; i < ret.size; i++)
	ret.data[i] = dist(rng);

    return ret;
}


static Array<uint8_t> make_random_bad_feed_mask(int S)
{
    assert(S > 0);
    assert(S < 32*1024);
    assert(S % 128 == 0);  // assumed by load_bad_feed_mask()
    
    Array<uint8_t> ret({S}, af_rhost);
    
    for (long i = 0; i < S; i++)
	ret.data[i] = max(0L, rand_int(-256,256));
    
    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// test_pack_e_array()


static void test_pack_e_array(int T, int F, int S, bool offset_encoded)
{
    cout << "test_pack_e_array: T=" << T << ", F=" << F << ", S=" << S
	 << ", offset_encoded=" << offset_encoded << endl;
    
    Array<complex<int>> E_src = make_random_unpacked_e_array(T,F,S);
    Array<uint8_t> E_packed = pack_e_array(E_src, offset_encoded);
    Array<complex<int>> E_dst = unpack_e_array(E_packed, offset_encoded);

    assert_arrays_equal(E_src, E_dst, "E_src", "E_dst", {"t","f","s"});
}


// -------------------------------------------------------------------------------------------------
//
// test_transpose_bit_with_lane()


static void reference_transpose_bit_with_lane(uint dst[32], const uint src[32], uint bit, uint lane)
{
    assert((bit == 1) || (bit == 2) || (bit == 4) || (bit == 8) || (bit == 16));
    assert((lane == 1) || (lane == 2) || (lane == 4) || (lane == 8) || (lane == 16));

    for (uint ldst = 0; ldst < 32; ldst++) {
	dst[ldst] = 0;
	
	for (uint bdst = 0; bdst < 32; bdst++) {
	    uint lsrc = ldst & ~lane;
	    uint bsrc = bdst & ~bit;

	    if (ldst & lane)
		bsrc |= bit;
	    if (bdst & bit)
		lsrc |= lane;
	    
	    if (src[lsrc] & (1U << bsrc))
		dst[ldst] |= (1U << bdst);
	}
    }
}


// Launch with nblocks = nwarps = 1.
__global__ void transpose_bit_with_lane_kernel(uint *dst, const uint *src, uint bit, uint lane)
{
    uint x = src[threadIdx.x];

    if (bit == 1)
	x = transpose_bit_with_lane<1> (x, lane);
    else if (bit == 2)
	x = transpose_bit_with_lane<2> (x, lane);
    else if (bit == 4)
	x = transpose_bit_with_lane<4> (x, lane);
    else if (bit == 8)
	x = transpose_bit_with_lane<8> (x, lane);
    else if (bit == 16)
	x = transpose_bit_with_lane<16> (x, lane);
	
    dst[threadIdx.x] = x;
}


static void test_transpose_bit_with_lane(uint bit, uint lane)
{
    cout << "test_transpose_bit_with_lane: bit=" << bit << ", lane=" << lane << endl;
    std::mt19937 &rng = gputils::default_rng;

    Array<uint> src({32}, af_rhost);
    for (int i = 0; i < 32; i++)
	src.data[i] = uint(rng()) ^ (uint(rng()) << 16);
    
    Array<uint> dst_cpu({32}, af_uhost);
    reference_transpose_bit_with_lane(dst_cpu.data, src.data, bit, lane);

    Array<uint> src_gpu = src.to_gpu();
    Array<uint> dst_gpu({32}, af_gpu);
    transpose_bit_with_lane_kernel <<<1,32>>> (dst_gpu.data, src_gpu.data, bit, lane);
    CUDA_PEEK("transpose_bit_with_lane_kernel launch");

    assert_arrays_equal(dst_cpu, dst_gpu, "cpu", "gpu", {"i"});
}


static void test_transpose_bit_with_lane()
{
    for (uint bit = 1; bit <= 16; bit *= 2)
	for (uint lane = 1; lane <= 16; lane *= 2)
	    test_transpose_bit_with_lane(bit, lane);
}


// -------------------------------------------------------------------------------------------------
//
// test_load_bad_feed_mask()


__global__ void bad_feed_mask_kernel(uint *out, const uint8_t *bf_mask, int S)
{
    // Length max(S/32,32).
    extern __shared__ uint shmem[];

    int t0 = threadIdx.z;
    t0 = (t0 * blockDim.y) + threadIdx.y;
    t0 = (t0 * blockDim.x) + threadIdx.x;

#if 0
    int nt = blockDim.z * blockDim.y * blockDim.x;
    for (int t = t0; t < nt; t += nt)
	shmem[t] = 0;
    __syncthreads();
#endif

    out[t0] = load_bad_feed_mask<true> ((const uint *) bf_mask, shmem, S);  // Debug=true
}


static void test_bad_feed_mask(int S, uint Wx, uint Wy, uint Wz)
{
    cout << "test_bad_feed_mask: S=" << S << ", Wx=" << Wx << ", Wy=" << Wy << ", Wz=" <<  Wz << endl;
    
    int nt = 32 * Wx * Wy * Wz;
    int shmem_nbytes = bf_mask_shmem_nbytes(S, Wx);   // contains asserts on (S,Wx)
    
    Array<uint8_t> bf_mask = make_random_bad_feed_mask(S);
    Array<uint> out({nt}, af_gpu);
    
    Array<uint8_t> bf_gpu = bf_mask.to_gpu();
    bad_feed_mask_kernel <<< 1, {32*Wx,Wy,Wz}, shmem_nbytes, 0 >>> (out.data, bf_gpu.data, S);
    CUDA_PEEK("bad_feed_mask_kernel launch");
    out = out.to_host();

    for (uint iyz = 0; iyz < Wy*Wz; iyz++) {
	for (uint ix = 0; ix < 32*Wx; ix++) {
	    int bit = 0;
	    for (int s = ix; s < S; s += 32*Wx) {
		bool bf_cpu = (bf_mask.data[s] != 0);
		bool bf_gpu = out.data[ix] & (1U << bit);
		
		if (bf_cpu != bf_gpu) {
		    cout << "failed! iyz=" << iyz << ", ix=" << ix << ", s=" << s << ", bit=" << "bf_cpu=" << bf_cpu << ", bf_gpu=" << bf_gpu << endl;
		    exit(1);
		}
		    
		bit++;
	    }
	}
    }
}


static void test_bad_feed_mask()
{
    for (int n = 0; n < 500; n++) {
	vector<ssize_t> W = random_integers_with_bounded_product(3, 32);  // (Wx, Wy, Wz)
	int Smax = 1024 * W[0];
	int S = 128 * rand_int(1, Smax/128 + 1);
	test_bad_feed_mask(S, W[0], W[1], W[2]);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Test cubic_interpolate() in interpolation.hpp


struct RandCubic
{
    vector<double> c;

    RandCubic() : c(4)
    {
	gputils::randomize(&c[0], 4);
    }

    double eval(double x) const
    {
	double ret = 0.0;
	for (int i = 0; i < 4; i++)
	    ret += c[i] * pow(x,i);
	return ret;
    }
};


static void test_cubic_interpolate()
{
    cout << "test_cubic_interpolate(): start" << endl;
    
    for (int i = 0; i < 100; i++) {
	RandCubic c;
	vector<double> y(4);

	for (int i = 0; i < 4; i++)
	    y[i] = c.eval(i-1);
	
	double x = gputils::rand_uniform();
	double lhs = cubic_interpolate(x, y[0], y[1], y[2], y[3]);
	double rhs = c.eval(x);
	double eps = std::abs(lhs - rhs);
	assert(eps < 1.0e-12);
    }

    cout << "test_cubic_interpolate(): pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Tests consistency between python interpolation (sk_bias.BiasInterpolator.interpolate_{bias,sigma})
// and CPU-side C++ interpolation (interpolate_{bias,sigma}_cpu in interpolation.hpp)
//
// Note: consistency between CPU-side C++ interpolation and GPU-side cuda interpolation is tested
// later (test_gpu_interpolation() below).


static void test_consistency_with_python_interpolation()
{
    cout << "test_consistency_with_python_interpolation(): start" << endl;
    
    const double *xvec = sk_globals::get_debug_x();
    const double *yvec = sk_globals::get_debug_y();
    const double *bvec = sk_globals::get_debug_b();
    const double *svec = sk_globals::get_debug_s();

    for (int i = 0; i < sk_globals::num_debug_checks; i++) {
	double b = interpolate_bias_cpu(xvec[i], yvec[i]);
	double s = interpolate_sigma_cpu(xvec[i]);

	assert(fabs(b-bvec[i]) < 1.0e-10);
	assert(fabs(s-svec[i]) < 1.0e-10);
    }

    cout << "test_consistency_with_python_interpolation(): pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Test load_sigma_coeffs() in interpolation.hpp


// Launch with 1 block and T threads.
//  out: shape (T,4)
//  sigma_coeffs: shape (N,8)
//  ix: shape (T,)

__global__ void sigma_coeffs_test_kernel(float *out, const float *sigma_coeffs, const int *ix)
{
    int t = threadIdx.x;
    load_sigma_coeffs<true> (sigma_coeffs, ix[t], out[4*t], out[4*t+1], out[4*t+2], out[4*t+3]);   // Debug=true
}


static void test_load_sigma_coeffs(int T, int N)
{
    cout << "test_load_sigma_coeffs(T=" << T << ", N=" << N << ")" << endl;
    
    Array<float> out({T,4}, af_uhost);
    Array<float> coeffs({N,8}, af_rhost);
    Array<int> ix({T}, af_rhost);

    for (int i = 0; i < N; i++) {
	float x = rand_uniform();
	for (int j = 0; j < 8; j++)
	    coeffs.at({i,j}) = x;
    }
    
    for (int t = 0; t < T; t++) {
	int i = rand_int(0, N-3);
	ix.at({t}) = i;
	
	for (int j = 0; j < 4; j++)
	    out.at({t,j}) = coeffs.at({i+j,0});
    }

    Array<float> out_gpu({T,4}, af_gpu | af_zero);
    Array<float> coeffs_gpu = coeffs.to_gpu();
    Array<int> ix_gpu = ix.to_gpu();

    sigma_coeffs_test_kernel<<<1,T>>> (out_gpu.data, coeffs_gpu.data, ix_gpu.data);
    CUDA_PEEK("sigma_coeffs_test_kernel");
    CUDA_CALL(cudaDeviceSynchronize());

    assert_arrays_equal(out, out_gpu, "cpu", "gpu", {"t","j"});
}


static void test_load_sigma_coeffs()
{
    for (int i = 0; i < 10; i++) {
	int T = 32 * rand_int(1,10);
	int N = rand_int(10, 100);
	test_load_sigma_coeffs(T, N);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Test load_bias_coeffs() in interpolation.hpp


// Launch with 1 block and T threads.
//  out: shape (T,4)
//  bias_coeffs: shape (N,4,2)
//  ix: shape (T,)
//  y: shape (T,)

__global__ void bias_coeffs_test_kernel(float *out, const float *bias_coeffs, const int *ix, const float *y)
{
    int t = threadIdx.x;
    load_bias_coeffs<true> (bias_coeffs, ix[t], y[t], out[4*t], out[4*t+1], out[4*t+2], out[4*t+3]);   // Debug=true
}


static void test_load_bias_coeffs(int T, int N)
{
    cout << "test_load_bias_coeffs(T=" << T << ", N=" << N << ")" << endl;
    
    Array<float> out({T,4}, af_uhost);
    Array<float> coeffs({N,4,2}, af_rhost);
    Array<int> ix({T}, af_rhost);
    Array<float> yy({T}, af_rhost);

    for (int i = 0; i < N; i++)
	for (int j = 0; j < 4; j++)
	    coeffs.at({i,j,0}) = coeffs.at({i,j,1}) = rand_uniform();
    
    for (int t = 0; t < T; t++) {
	int i = rand_int(0, N-3);
	float y = rand_uniform();
	ix.at({t}) = i;
	yy.at({t}) = y;
	
	for (int j = 0; j < 4; j++) {
	    float c0 = coeffs.at({i+j,0,0});
	    float c1 = coeffs.at({i+j,1,0});
	    float c2 = coeffs.at({i+j,2,0});
	    float c3 = coeffs.at({i+j,3,0});
	    out.at({t,j}) = c0 + c1*y + c2*y*y + c3*y*y*y;
	}
    }

    Array<float> out_gpu({T,4}, af_gpu | af_zero);
    Array<float> coeffs_gpu = coeffs.to_gpu();
    Array<int> ix_gpu = ix.to_gpu();
    Array<float> yy_gpu = yy.to_gpu();

    bias_coeffs_test_kernel<<<1,T>>> (out_gpu.data, coeffs_gpu.data, ix_gpu.data, yy_gpu.data);
    CUDA_PEEK("bias_coeffs_test_kernel");
    CUDA_CALL(cudaDeviceSynchronize());

    assert_arrays_equal(out, out_gpu, "cpu", "gpu", {"t","j"});
}


static void test_load_bias_coeffs()
{
    for (int i = 0; i < 10; i++) {
	int T = 32 * rand_int(1,10);
	int N = rand_int(10, 100);
	test_load_bias_coeffs(T, N);
    }
}


// -------------------------------------------------------------------------------------------------
//
// test_gpu_interpolation(): interpolate b(mu,N) and sigma(mu) on the CPU/GPU, and compare.
//
// This tests the following chain of __device__ inline functions in interpolation.hpp:
//   unpack_bias_sigma_coeffs()
//   interpolate_bias_gpu()
//   interpolate_sigma_gpu()


// Launch with 1 block and T threads.
//   b_out: shape (T,)
//   s_out: shape (T,)
//   x_in: shape (T,)
//   y_in: shape (T,)
//   gmem_bsigma_coeffs: from SkKernel::bsigma_coeffs


__global__ void gpu_interpolation_test_kernel(float *b_out, float *s_out, const float *x_in, const float *y_in, const float *gmem_bsigma_coeffs)
{
    __shared__ float shmem_bsigma_coeffs[bsigma_shmem_nelts];

    unpack_bias_sigma_coeffs(gmem_bsigma_coeffs, shmem_bsigma_coeffs);
    
    float x = x_in[threadIdx.x];
    float y = y_in[threadIdx.x];

    b_out[threadIdx.x] = interpolate_bias_gpu(shmem_bsigma_coeffs, x, y);
    s_out[threadIdx.x] = interpolate_sigma_gpu(shmem_bsigma_coeffs, x);
}


static void test_gpu_interpolation(int T)
{
    cout << "test_gpu_interpolation: T=" << T << endl;
    
    const double xmin = log(sk_globals::mu_min);   // not sk_globals::xmin
    const double xmax = log(sk_globals::mu_max);   // not sk_globals::xmax
    const double ymax = 1.0 / double(sk_globals::bias_nmin);
    
    Array<float> x_cpu({T}, af_rhost);
    Array<float> y_cpu({T}, af_rhost);
    Array<float> b_cpu({T}, af_uhost);
    Array<float> s_cpu({T}, af_uhost);

    for (int t = 0; t < T; t++) {
	float x = rand_uniform(xmin, xmax);
	float y = rand_uniform(0.0, ymax);
	x_cpu.at({t}) = x;
	y_cpu.at({t}) = y;
	b_cpu.at({t}) = interpolate_bias_cpu(x,y);
	s_cpu.at({t}) = interpolate_sigma_cpu(x);
    }
    
    SkKernel::Params params;
    SkKernel sk_kernel(params, false);  // check_params=false

    Array<float> x_gpu = x_cpu.to_gpu();
    Array<float> y_gpu = y_cpu.to_gpu();
    Array<float> b_gpu({T}, af_gpu | af_random);
    Array<float> s_gpu({T}, af_gpu | af_random);

    gpu_interpolation_test_kernel <<<1,T>>>
	(b_gpu.data, s_gpu.data, x_gpu.data, y_gpu.data, sk_kernel.bsigma_coeffs.data);

    CUDA_PEEK("gpu_interpolation_test_kernel launch");
    CUDA_CALL(cudaDeviceSynchronize());

    assert_arrays_equal(s_cpu, s_gpu, "sigma_cpu", "sigma_gpu", {"i"});
    assert_arrays_equal(b_cpu, b_gpu, "bias_cpu", "bias_gpu", {"i"});
}


static void test_gpu_interpolation()
{
    for (int i = 0; i < 10; i++) {
	int T = 32 * rand_int(1,10);
	test_gpu_interpolation(T);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Test s0_kernel


// Input array:   ulong pl_mask[T/128, (F+3)/4, S/8]
// Output array:  ulong s0[T/Nds, F, S]

static Array<ulong> reference_s0_kernel(const Array<ulong> &pl_mask, long T, long F, long S, long Nds)
{
    long Tds = T/Nds;
    long Fds = (F+3)/4;
    long Sds = S/8;
    
    assert(pl_mask.shape_equals({T/128, Fds, Sds}));
    
    Array<ulong> s0_arr({Tds, F, S}, af_rhost);
    
    for (long tds = 0; tds < Tds; tds++) {
	long t2_lo = tds * (Nds >> 1);
	long t2_hi = (tds+1) * (Nds >> 1);
    
	for (long fds = 0; fds < Fds; fds++) {
	    for (long sds = 0; sds < Sds; sds++) {
		ulong s0_elt = 0;

		for (long t2 = t2_lo; t2 < t2_hi; t2++) {
		    ulong x = pl_mask.at({t2 >> 6, fds, sds});
		    ulong bit = 1UL << (t2 & 63);
		    s0_elt += (x & bit) ? 2 : 0;
		}

		for (long f = 4*fds; f < min(4*fds+4,F); f++)
		    for (long s = 8*sds; s < 8*sds+8; s++)
			s0_arr.at({tds, f, s}) = s0_elt;
	    }
	}
    }

    return s0_arr;
}


static void test_s0_kernel(const Array<ulong> &pl_mask, long T, long F, long S, long Nds, long fstride)
{
    stringstream ss;
    ss << "test_s0_kernel(T=" << T << ", F=" << F << ", S=" << S << ", Nds=" << Nds << ", fstride=" << fstride << ")";
    cout << ss.str() << endl;
    
    Array<ulong> s0_ref = reference_s0_kernel(pl_mask, T, F, S, Nds);
    Array<ulong> s0_gpu({T/Nds,F,S}, {F*fstride,fstride,1}, af_gpu | af_guard);
    Array<ulong> pl_mask_gpu = pl_mask.to_gpu();
    launch_s0_kernel(s0_gpu, pl_mask_gpu, Nds);
    
    gputils::assert_arrays_equal(s0_ref, s0_gpu, "cpu", "gpu", {"tds","f","s"});
}


static void test_s0_kernel(long T, long F, long S, long Nds, long fstride)
{
    Array<ulong> pl_mask({T/128, (F+3)/4, S/8}, af_rhost);
    ulong *pl = pl_mask.data;
    
    for (long i = 0; i < pl_mask.size; i++) {
	pl[i] = default_rng();
	pl[i] ^= (ulong(default_rng()) << 22);
	pl[i] ^= (ulong(default_rng()) << 44);
    }

    test_s0_kernel(pl_mask, T, F, S, Nds, fstride);
}


static void test_s0_kernel()
{
    for (int i = 0; i < 100; i++) {
	long Nds = 2 * rand_int(1, 200);
	long Tdiv = std::lcm(Nds, 128);
	
	vector<ssize_t> v = gputils::random_integers_with_bounded_product(3, (1000*1000)/Tdiv);
	long T = v[0]*Tdiv;
	long F = v[1];
	long S = v[2]*128;
	
	long fstride = 4 * rand_int(S/4, S+1);
	test_s0_kernel(T, F, S, Nds, fstride);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Test s12_kernel


static void test_s12_kernel(int Tout, int F, int S, int Nds, int fstride, bool offset_encoded)
{
    cout << "test_s12_kernel: Tout=" << Tout << ", F=" << F << ", S=" << S << ", Nds=" << Nds
	 << ", fstride=" << fstride << ", offset_encoded=" << offset_encoded << endl;

    long Tin = Tout * Nds;
    Array<complex<int>> e_cpu = make_random_unpacked_e_array(Tin,F,S);  // shape (Tin,F,S)
    Array<uint8_t> e_gpu = pack_e_array(e_cpu, offset_encoded);
    e_gpu = e_gpu.to_gpu();
    
    Array<ulong> s_cpu({Tout,F,2,S}, af_uhost | af_zero);
    Array<ulong> s_gpu({Tout,F,2,S}, {F*fstride,fstride,S,1}, af_gpu | af_guard);

    launch_s12_kernel(s_gpu, e_gpu, Nds, offset_encoded);
    CUDA_CALL(cudaDeviceSynchronize());

    // e_cpu -> s_cpu
    for (int tout = 0; tout < Tout; tout++) {
	for (int tin = tout*Nds; tin < (tout+1)*Nds; tin++) {
	    for (int f = 0; f < F; f++) {
		for (int s = 0; s < S; s++) {
		    complex<int> e = e_cpu.at({tin,f,s});
		    uint e2 = e.real()*e.real() + e.imag()*e.imag();
		    s_cpu.at({tout,f,0,s}) += e2;
		    s_cpu.at({tout,f,1,s}) += e2*e2;
		}
	    }
	}
    }

    assert_arrays_equal(s_cpu, s_gpu, "cpu", "gpu", {"t","f","n","s"});
}


static void test_s12_kernel()
{
    for (int n = 0; n < 100; n++) {
	// (Tout, F, S/128, Nds)
	vector<ssize_t> v = gputils::random_integers_with_bounded_product(4, 400000);
	long Tout = v[0];
	long F = v[1];
	long S = 128 * v[2];
	long Nds = v[3];
	int fstride = 4 * rand_int(S/2, S+1);
	bool offset_encoded = rand_int(0,2);
	test_s12_kernel(Tout, F, S, Nds, fstride, offset_encoded);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Test s012_time_downsample


static void test_s012_time_downsample(int Tout, int F, int S, int Nds)
{
    cout << "test_s012_time_downsample: Tout=" << Tout << ", F=" << F << ", S=" << S << ", Nds=" << Nds << endl;

    long Tin = Tout * Nds;
    Array<ulong> s_in = make_random_s012_array(Tin,F,S);  // shape (Tin,F,3,S)
    Array<ulong> s_out_cpu({Tout,F,3,S}, af_uhost | af_zero);
    Array<ulong> s_out_gpu({Tout,F,3,S}, af_gpu | af_guard);
        
    for (int tout = 0; tout < Tout; tout++)
	for (int tin = tout*Nds; tin < (tout+1)*Nds; tin++)
	    for (int f = 0; f < F; f++)
		for (int n = 0; n < 3; n++)
		    for (int s = 0; s < S; s++)
			s_out_cpu.at({tout,f,n,s}) += s_in.at({tin,f,n,s});

    Array<ulong> s_in_gpu = s_in.to_gpu();
    launch_s012_time_downsample_kernel(s_out_gpu, s_in_gpu, Nds);
    s_out_gpu = s_out_gpu.to_host();

    assert_arrays_equal(s_out_cpu, s_out_gpu, "cpu", "gpu", {"t","f","n","s"});
}


static void test_s012_time_downsample()
{
    for (int n = 0; n < 100; n++) {
	vector<ssize_t> v = gputils::random_integers_with_bounded_product(4, 100);
	long Tout = v[0];
	long F = v[1];
	long S = 32 * v[2];
	long Nds = v[3];
	test_s012_time_downsample(Tout, F, S, Nds);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Test s012_station_downsample


static void test_s012_station_downsample(int T, int F, int S)
{
    cout << "test_s012_station_downsample: T=" << T << ", F=" << F << ", S=" << S << endl;

    Array<ulong> s_in = make_random_s012_array(T,F,S);  // shape (T,F,3,S)
    Array<uint8_t> bf_mask = make_random_bad_feed_mask(S);
    
    Array<ulong> s_out_cpu({T,F,3}, af_uhost | af_zero);
    Array<ulong> s_out_gpu({T,F,3}, af_gpu | af_guard);

    for (int t = 0; t < T; t++)
	for (int f = 0; f < F; f++)
	    for (int n = 0; n < 3; n++)
		for (int s = 0; s < S; s++)
		    s_out_cpu.at({t,f,n}) += (bf_mask.data[s] ? s_in.at({t,f,n,s}) : 0);

    Array<ulong> s_in_gpu = s_in.to_gpu();
    Array<uint8_t> bf_mask_gpu = bf_mask.to_gpu();
    launch_s012_station_downsample_kernel(s_out_gpu, s_in_gpu, bf_mask_gpu);
    CUDA_CALL(cudaDeviceSynchronize());
    
    assert_arrays_equal(s_out_cpu, s_out_gpu, "cpu", "gpu", {"t","f","n"});
}


static void test_s012_station_downsample()
{
    for (int n = 0; n < 100; n++) {
	long S = 128 * rand_int(1, rfi_max_stations/128);
	vector<ssize_t> v = gputils::random_integers_with_bounded_product(2, 400000/S);
	test_s012_station_downsample(v[0], v[1], S);  // (T,F,S)
    }
}


//-------------------------------------------------------------------------------------------------
//
// Test SkKernel


struct TestInstance
{
    long T = 0;
    long F = 0;
    long S = 0;
    long Nds = 0;

    double sk_rfimask_sigmas = 0.0;
    double single_feed_min_good_frac = 0.0;
    double feed_averaged_min_good_frac = 0.0;
    double mu_min = 0.0;
    double mu_max = 0.0;
    double rfi_mask_frac = 0.0;

    Array<float> out_sk_feed_averaged;    // shape (T,F,3)
    Array<float> out_sk_single_feed;      // shape (T,F,3,S)
    Array<uint> out_rfimask;              // shape (F,T*Nds/32)
    Array<ulong> in_S012;                 // shape (T,F,3,S)
    Array<uint8_t> in_bf_mask;            // length S (bad feed bask)

    // Temp quantities used when generating the test instance.
    // All vectors are length-S.
    vector<long> S0;
    vector<long> S1;
    vector<long> S2;
    vector<double> sf_sk;
    vector<double> sf_bias;
    vector<double> sf_sigma;
    double fsum_sk;
    double fsum_bias;
    double fsum_sigma;
    bool rfimask;
    
    
    TestInstance(long T_, long F_, long S_, long Nds_)
	: T(T_), F(F_), S(S_), Nds(Nds_)
    {
	assert(T > 0);
	assert(F > 0);
	assert(S > 0);
	assert(Nds > 0);
	assert((S % 128) == 0);
	assert((Nds % 32) == 0);
	
	this->out_sk_feed_averaged = Array<float> ({T,F,3}, af_rhost | af_zero);
	this->out_sk_single_feed = Array<float> ({T,F,3,S}, af_rhost | af_zero);
	this->out_rfimask = Array<uint> ({F,(T*Nds)/32}, af_rhost | af_zero);
	this->in_S012 = Array<ulong> ({T,F,3,S}, af_rhost | af_zero);
	this->in_bf_mask = Array<uint8_t> ({S}, af_rhost | af_zero);

	this->sk_rfimask_sigmas = rand_uniform(0.5, 1.5);
	this->single_feed_min_good_frac = rand_uniform(0.7, 0.8);
	this->feed_averaged_min_good_frac = rand_uniform(0.3, 0.4);
	this->mu_min = rand_uniform(3.0, 4.0);
	this->mu_max = rand_uniform(20.0, 30.0);

	this->S0 = vector<long> (S);
	this->S1 = vector<long> (S);
	this->S2 = vector<long> (S);
	this->sf_sk = vector<double> (S);
	this->sf_bias = vector<double> (S);
	this->sf_sigma = vector<double> (S);

	this->_init_bad_feed_mask();
	
	for (int t = 0; t < T; t++) {
	    for (int f = 0; f < F; f++) {
		this->_init_tf_pair();

		for (int s = 0; s < S; s++) {
		    this->in_S012.at({t,f,0,s}) = S0[s];
		    this->in_S012.at({t,f,1,s}) = S1[s];
		    this->in_S012.at({t,f,2,s}) = S2[s];
		    this->out_sk_single_feed.at({t,f,0,s}) = sf_sk[s];
		    this->out_sk_single_feed.at({t,f,1,s}) = sf_bias[s];
		    this->out_sk_single_feed.at({t,f,2,s}) = sf_sigma[s];
		}
		
		this->out_sk_feed_averaged.at({t,f,0}) = fsum_sk;
		this->out_sk_feed_averaged.at({t,f,1}) = fsum_bias;
		this->out_sk_feed_averaged.at({t,f,2}) = fsum_sigma;

		for (int i = t*(Nds/32); i < (t+1)*(Nds/32); i++)
		    this->out_rfimask.at({f,i}) = rfimask ? 0xffffffffU : 0;

		if (rfimask)
		    this->rfi_mask_frac += 1.0 / double(F*T);
	    }
	}
    }

    static inline double _compute_sk(double s0, double s1, double s2, double b)
    {
	double u = (s0 > 1.5) ? ((s0+1)/(s0-1)) : 0.0;
	double v = (s1 > 0.5) ? (s0 / (s1*s1)) : 0.0;
	return u * (v*s2 - 1) - b;
    }

    // Inverts (s2 -> sk) at fixed (s0,s1).
    static inline double _invert_sk(double s0, double s1, double sk, double b)
    {
	double ru = (s0 > 0.5) ? ((s0-1)/(s0+1)) : 0.0;
	double rv = (s0 > 0.5) ? ((s1*s1) / s0) : 0.0;
	return rv * (ru*(sk+b) + 1);
    }


    // Helper function called by constructor.
    void _init_bad_feed_mask()
    {
	// We mask ~5% of the stations.
	for (int s = 0; s < S; s++)
	    in_bf_mask.at({s}) = rand_int(0, 21);
    }


    // Helper function called by _init_tf_pair().
    void _init_valid_S0_S1(int s)
    {
	long S0_edge = round(single_feed_min_good_frac * Nds);
	S0[s] = rand_int(S0_edge+1, Nds+1);
	
	long S1_edge0 = round(mu_min * S0[s]);
	long S1_edge1 = round(mu_max * S0[s]);
	S1[s] = rand_int(S1_edge0+1, S1_edge1);
    }


    // Helper function called by _init_tf_pair().
    void _init_invalid_S0_S1(int s)
    {
	for (;;) {
	    S0[s] = rand_int(-Nds/32, Nds+1);
	    S0[s] = max(S0[s], 0L);
	    S1[s] = rand_int(0, 98*S0[s]+1);
	    
	    long S0_edge = round(single_feed_min_good_frac * Nds);
	    long S1_edge0 = round(mu_min * S0[s]);
	    long S1_edge1 = round(mu_max * S0[s]);

	    if ((S0[s] < S0_edge) || (S1[s] < S1_edge0) || (S1[s] > S1_edge1))
		return;
	}
    }
    
    void _init_tf_pair()
    {
	double p1 = rand_uniform(-0.2, 1.0);
	double p2 = rand_uniform(-0.2, 1.0);
	double prob_sf_valid = max(max(p1,p2), 0.0);
	
	// The purpose of this outer loop is to allow restarts, if we end up in
	// a situation where roundoff error may be an issue for the unit test
	// (because we're close to a boolean threshold).
	
	for (;;) {
	    double sum_w = 0.0;
	    double sum_wsk = 0.0;
	    double sum_wb = 0.0;
	    double sum_wsigma2 = 0.0;
	    
	    for (int s = 0; s < S; s++) {
		bool sf_valid = (rand_uniform() < prob_sf_valid);

		// Init S0[s], S1[s].
		if (sf_valid)
		    _init_valid_S0_S1(s);
		else
		    _init_invalid_S0_S1(s);

		// Code after this point initializes S2[s].

		double s0 = S0[s];
		double s1 = S1[s];
		double mu = (s0 > 0.5) ? (s1/s0) : 0.0;
		double x = sf_valid ? log(mu) : 0.0;
		double y = sf_valid ? (1.0/s0) : 0.0;
		double b = sf_valid ? interpolate_bias_cpu(x,y) : 0.0;
		double sigma = sf_valid ? (interpolate_sigma_cpu(x) * sqrt(y)) : 0.0;
		double target_sk = 1.0 + sigma * sqrt(3.) * rand_uniform(-1.0,1.0);
		double s2 = _invert_sk(s0, s1, target_sk, b);

		s2 = max(s2, s1);
		s2 = min(s2, 98*s1);		
		s2 = round(s2);
		S2[s] = s2;
		
		double actual_sk = _compute_sk(s0, s1, s2, b);

		sf_sk[s] = sf_valid ? actual_sk : 0.0;
		sf_bias[s] = sf_valid ? b : 0.0;
		sf_sigma[s] = sf_valid ? sigma : -1.0;
		
		double w = (sf_valid && in_bf_mask.at({s})) ? s0 : 0.0;
		
		sum_w += w;
		sum_wsk += w * sf_sk[s];
		sum_wb += w * sf_bias[s];
		sum_wsigma2 += w * w * sf_sigma[s] * sf_sigma[s];
	    }

	    double sum_w_threshold = feed_averaged_min_good_frac * S * Nds;

	    if (fabs(sum_w - sum_w_threshold) < 0.1)
		continue;   // Restart (too close to boolean threshold)
	    
	    bool fsum_valid = (sum_w > sum_w_threshold);

	    this->fsum_sk = fsum_valid ? (sum_wsk / sum_w) : 0.0;
	    this->fsum_bias = fsum_valid ? (sum_wb / sum_w) : 0.0;
	    this->fsum_sigma = fsum_valid ? (sqrt(sum_wsigma2) / sum_w) : -1.0;

	    if (!fsum_valid) {
		this->rfimask = 0;
		return;
	    }

	    // RFI mask is determined by thresholding u.
	    double u = fabs(fsum_sk - 1.0);
	    double uthresh = sk_rfimask_sigmas * fsum_sigma;
	    assert(uthresh > 2.0e-4);

	    if (fabs(u-uthresh) < 1.0e-4)
		continue;  // Restart (too close to boolean threshold)
	    
	    this->rfimask = (u < uthresh);
	    return;
	}
    }
};


static void test_sk_kernel(const TestInstance &ti, bool check_sf_sk=true, bool check_rfimask=true, long rfimask_fstride=0)
{    
    long T = ti.T;
    long F = ti.F;
    long S = ti.S;
    long Nds = ti.Nds;

    if (!rfimask_fstride)
	rfimask_fstride = (T*Nds)/32;
    
    cout << "test_sk_kernel: T=" << T << ", F=" << F << ", S=" << S << ", Nds=" << Nds
	 << ", check_sf_sk=" << check_sf_sk << ", check_rfimask=" << check_rfimask
	 << ", rfimask_fstride=" << rfimask_fstride << ", rfi_mask_frac=" << ti.rfi_mask_frac
	 << endl;
    
    // Input arrays
    Array<ulong> gpu_S012 = ti.in_S012.to_gpu();
    Array<uint8_t> gpu_bf_mask = ti.in_bf_mask.to_gpu();

    // Output arrays
    Array<float> gpu_sk_feed_averaged({T,F,3}, af_gpu | af_random);
    Array<float> gpu_sk_single_feed;
    Array<uint> gpu_rfimask;

    if (check_sf_sk)
	gpu_sk_single_feed = Array<float> ({T,F,3,S}, af_gpu | af_random);
    if (check_rfimask)
	gpu_rfimask = Array<uint> ({F,(T*Nds)/32}, {rfimask_fstride,1}, af_gpu | af_random);

    // FIXME to reduce cut-and-paste here, modify definition of 'struct TestInstance'
    // to include an SkKernel::Params.
    SkKernel::Params params;
    params.sk_rfimask_sigmas = ti.sk_rfimask_sigmas;
    params.single_feed_min_good_frac = ti.single_feed_min_good_frac;
    params.feed_averaged_min_good_frac = ti.feed_averaged_min_good_frac;
    params.mu_min = ti.mu_min;
    params.mu_max = ti.mu_max;
    params.Nds = ti.Nds;

    SkKernel sk_kernel(params);
    
    sk_kernel.launch(
        gpu_sk_feed_averaged,
	gpu_sk_single_feed,
	gpu_rfimask,
	gpu_S012,
	gpu_bf_mask);

    CUDA_CALL(cudaDeviceSynchronize());

    if (check_sf_sk)
	gputils::assert_arrays_equal(gpu_sk_single_feed, ti.out_sk_single_feed, "gpu_sf_sk", "ref_sf_sk", {"t","f","n","s"});
	
    gputils::assert_arrays_equal(gpu_sk_feed_averaged, ti.out_sk_feed_averaged, "gpu_fsum_sk", "ref_fsum_sk", {"t","f","n"});

    if (check_rfimask)
	gputils::assert_arrays_equal(gpu_rfimask, ti.out_rfimask, "gpu_rfimask", "ref_rfimask", {"f","t32"});
}


static void test_sk_kernel()
{
    for (int n = 0; n < 500; n++) {
	long T = rand_int(1, 21);
	long F = rand_int(1, 21);
	long S = 128 * rand_int(1, rfi_max_stations/128);
	long Nds = 32 * rand_int(4, 11);
	bool check_sf_sk = (rand_uniform() < 0.9);
	bool check_rfimask = (rand_uniform() < 0.9);
	long rfimask_fstride = rand_int((T*Nds)/32, (T*Nds)/16);
	
	TestInstance ti(T,F,S,Nds);
	test_sk_kernel(ti, check_sf_sk, check_rfimask, rfimask_fstride);
    }
}


//-------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    test_pack_e_array(16, 32, 128, false);  // offset_encoded=false
    test_pack_e_array(16, 32, 128, true);   // offset_encoded=true
    test_transpose_bit_with_lane();
    test_bad_feed_mask();
    test_cubic_interpolate();
    test_consistency_with_python_interpolation();
    test_load_sigma_coeffs();
    test_load_bias_coeffs();
    test_gpu_interpolation();
    
    test_s0_kernel();
    test_s12_kernel();
    test_s012_time_downsample();
    test_s012_station_downsample();
    test_sk_kernel();
    
    return 0;
}
