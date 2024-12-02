#include <complex>
#include <iostream>
#include <gputils.hpp>

#include "../include/n2k/Correlator.hpp"
#include "../include/n2k/internals/CorrelatorKernel.hpp"
#include "../include/n2k/internals/internals.hpp"

using namespace std;
using namespace gputils;
using namespace n2k;


// -------------------------------------------------------------------------------------------------
//
// Test negate_4bit()


// Use 1 threadblock.
__global__ void negate_4bit_test_kernel(int *buf, int nelts)
{
    for (int i = threadIdx.x; i < nelts; i += blockDim.x)
	buf[i] = CorrelatorKernel<128,32>::negate_4bit(buf[i]);
}


__host__ void test_negate_4bit()
{
    const int nelts = 1024*1024;

    Array<complex<int>> src = make_random_unpacked_e_array(nelts,1,1);  // shape (nelts,1,1)
    src = src.reshape_ref({nelts});                                     // shape (nelts,)

    Array<uint8_t> a = pack_e_array(src, false);          // offset_encoded=false
    a = a.to_gpu();
    
    negate_4bit_test_kernel <<<1,1024>>> ((int *) a.data, nelts/4);
    CUDA_PEEK("negate_4bit_test_kernel");
    CUDA_CALL(cudaDeviceSynchronize());

    a = a.to_host();
    Array<complex<int>> dst = unpack_e_array(a, false);   // offset_encoded=false

    for (int i = 0; i < nelts; i++)
	assert(dst.at({i}) == -src.at({i}));

    cout << "test_negate_4bit: pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Test transpose_rank8_4bit()


static void reference_rank8_4bit_kernel(int *buf, int nelts)
{
    assert((nelts % 8) == 0);
    int m[8][8];
    
    for (int i = 0; i < nelts/8; i++) {
	for (int j = 0; j < 8; j++) {
	    int x = buf[8*i+j];
	    for (int k = 0; k < 8; k++)
		m[j][k] = (x >> (4*k)) & 0xf;
	}

	for (int j = 0; j < 8; j++) {
	    int y = 0;
	    for (int k = 0; k < 8; k++)
		y |= (m[k][j] << (4*k));
	    buf[8*i+j] = y;
	}
    }
}


// Use 1 threadblock and (nelts % 8) == 0
__global__ void transpose_rank8_4bit_kernel(int *buf, int nelts)
{
    for (int i = threadIdx.x; i < nelts/8; i += blockDim.x) {
	int x[8];

	#pragma unroll
	for (int j = 0; j < 8; j++)
	    x[j] = buf[8*i+j];

	CorrelatorKernel<128,32>::transpose_rank8_4bit(x);

	#pragma unroll
	for (int j = 0; j < 8; j++)
	    buf[8*i+j] = x[j];
    }
}


__host__ void test_transpose_rank8_4bit()
{
    const int nelts = 1024;

    Array<int> src_cpu({nelts}, af_rhost);

    for (int i = 0; i < nelts; i++) {
	// Make 32 random bits.
	uint x = gputils::default_rng();
	uint y = gputils::default_rng();
	src_cpu.data[i] = x ^ (y << 16);
    }

    Array<int> dst_cpu = src_cpu.clone();
    reference_rank8_4bit_kernel(dst_cpu.data, nelts);

    Array<int> a = src_cpu.to_gpu();
    transpose_rank8_4bit_kernel <<< 1, 1024 >>> (a.data, nelts);
    CUDA_PEEK("transpose_rank8_4bit_kernel");
    CUDA_CALL(cudaDeviceSynchronize());

    Array<int> dst_gpu = a.to_host();

    for (int i = 0; i < nelts; i++)
	assert(dst_gpu.data[i] == dst_cpu.data[i]);

    cout << "test_transpose_rank8_4bit: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


__host__ int8_t pack_complex44(complex<int> z)
{
    constexpr int8_t bit = CorrelatorParams::offset_encoded ? 0x88 : 0;
    
    assert((z.real() >= -7) && (z.real() <= 7));
    assert((z.imag() >= -7) && (z.imag() <= 7));

    if constexpr (CorrelatorParams::real_part_in_low_bits)
	return ((z.real() & 0xf) | (z.imag() << 4)) ^ bit;
    else
	return ((z.imag() & 0xf) | (z.real() << 4)) ^ bit;
}


__host__ void minimal_correlator_test(int nstations, int nfreq, int f, int sa, int sb, int t, complex<int> za, complex<int> zb, bool rfi_bit=true)
{
    // Currently hardcoded
    const int nt_outer = 4;
    const int nt_inner = 1024;
    
    const int nt_tot = nt_outer * nt_inner;
    const int touter = int(t / nt_inner);
    const int nvtiles = ((nstations/16) * (nstations/16+1)) / 2;
    
    assert(!CorrelatorParams::artificially_remove_input_shuffle);
    assert(!CorrelatorParams::artificially_remove_output_shuffle);
    assert(!CorrelatorParams::artificially_remove_negate_4bit);
    
    assert((f >= 0) && (f < nfreq));
    assert((sa >= 0) && (sa < nstations));
    assert((sb >= 0) && (sb < nstations));
    assert((t >= 0) && (t < nt_outer*nt_inner));
    assert((za.real() >= -7) && (za.real() <= 7));
    assert((za.imag() >= -7) && (za.imag() <= 7));
    assert((zb.real() >= -7) && (zb.real() <= 7));
    assert((zb.imag() >= -7) && (zb.imag() <= 7));
    
    assert(sa >= sb);
    assert((sa > sb) || (za == zb));
    
    int ea = pack_complex44(za);
    int eb = pack_complex44(zb);
    int ahi = sa >> 4;
    int bhi = sb >> 4;
    int alo = sa & 0xf;
    int blo = sb & 0xf;
    int taa = (ahi*(ahi+1))/2 + ahi;
    int tab = (ahi*(ahi+1))/2 + bhi;
    int tbb = (bhi*(bhi+1))/2 + bhi;
    
    cout << "\nminimal_correlator_test(): start:"
	 << " nstations=" << nstations << ", nfreq=" << nfreq
	 << ", f=" << f << ", sa=" << sa << ", sb=" << sb << ", t=" << t
	 << ", za=" << za << ", Ea=" << ea << ", zb=" << zb << ", Eb=" << eb
	 << ", rfi_bit=" << rfi_bit << endl;

    Correlator corr(nstations, nfreq);
    
    Array<int8_t> emat({nt_tot,nfreq,nstations}, af_rhost);
    memset(emat.data, (CorrelatorParams::offset_encoded ? 0x88 : 0), emat.size);
    
    emat.at({t,f,sa}) = ea;
    emat.at({t,f,sb}) = eb;

    assert((nt_tot % 32) == 0);
    Array<uint> rfimask({nfreq,nt_tot/32}, af_rhost | af_zero);
    Array<int> vmat_cpu({nt_outer,nfreq,nvtiles,16,16,2}, af_rhost | af_zero);
    Array<int> vmat_gpu({nt_outer,nfreq,nvtiles,16,16,2}, af_random | af_gpu);

    // Compute rfimask and vmat_cpu.
    if (rfi_bit) {
	int j = (f*nt_outer*nt_inner + t) / 32;
	rfimask.data[j] = (1U << (t % 32));
	
	vmat_cpu.at({touter,f,taa,alo,alo,0}) = (za * conj(za)).real();
	vmat_cpu.at({touter,f,tbb,blo,blo,0}) = (zb * conj(zb)).real();
	vmat_cpu.at({touter,f,tab,alo,blo,0}) = (za * conj(zb)).real();
	vmat_cpu.at({touter,f,tab,alo,blo,1}) = (za * conj(zb)).imag();
	
	if (ahi == bhi) {
	    vmat_cpu.at({touter,f,tab,blo,alo,0}) = (zb * conj(za)).real();
	    vmat_cpu.at({touter,f,tab,blo,alo,1}) = (zb * conj(za)).imag();
	}
    }

    // Compute vmat_gpu.
    emat = emat.to_gpu();
    rfimask = rfimask.to_gpu();
    corr.launch(vmat_gpu, emat, rfimask, nt_outer, nt_inner, nullptr, true);  // sync=true
    vmat_gpu = vmat_gpu.to_host();

    // Compare results.
    
    int nfail = 0;

    for (int to = 0; to < nt_outer; to++) {
	for (int f = 0; f < nfreq; f++) {
	    for (int vt = 0; vt < nvtiles; vt++) {
		for (int i = 0; i < 16; i++) {
		    for (int k = 0; k < 16; k++) {
			int cpu_re = vmat_cpu.at({to,f,vt,i,k,0});
			int cpu_im = vmat_cpu.at({to,f,vt,i,k,1});
			int gpu_re = vmat_gpu.at({to,f,vt,i,k,0});
			int gpu_im = vmat_gpu.at({to,f,vt,i,k,1});

			if ((cpu_re == 0) && (cpu_im == 0) && (gpu_re == 0) && (gpu_im == 0))
			    continue;

			bool fail = (cpu_re != gpu_re) || (cpu_im != gpu_im);

			cout << "    " << (fail ? "FAILED" : "looks good")
			     << ": at touter=" << to << ", f=" << f << ", vt=" << vt << ", i=" << i << ", k=" << k << ": "
			     << "expected (re,im)=(" << cpu_re << "," << cpu_im << "), "
			     << "got (re,im)=(" << gpu_re << "," << gpu_im << ")" << endl;
			
			if (fail)
			    nfail++;
		
			if (nfail >= 32) {
			    cout << "    Reached threshold failure count; aborting test" << endl;
			    exit(1);
			}
		    }
		}
	    }
	}
    }

    if (nfail > 0) {
	cout << "    Test failed" << endl;
	exit(1);
    }

    cout << "minimal_correlator_test(): pass" << endl;
}


void test_correlator(int nstations, int nfreq, int nt_outer, int nt_inner, int M=10)
{
    const int nt_tot = (nt_outer * nt_inner);
    const int nvtiles = ((nstations/16) * (nstations/16+1)) / 2;
    
    assert(!CorrelatorParams::artificially_remove_input_shuffle);
    assert(!CorrelatorParams::artificially_remove_output_shuffle);
    assert(!CorrelatorParams::artificially_remove_negate_4bit);

    cout << "\ntest_correlator("
	 << "nstations=" << nstations
	 << ", nfreq=" << nfreq
	 << ", nt_outer=" << nt_outer
	 << ", nt_inner=" << nt_inner
	 << ")" << endl;

    Array<int> vmat_cpu({nt_outer,nfreq,nvtiles,16,16,2}, af_rhost | af_zero);
    Array<int8_t> emat({nt_tot,nfreq,nstations}, af_rhost | af_zero);
    memset(emat.data, (CorrelatorParams::offset_encoded ? 0x88 : 0), emat.size);
    
    assert((nt_tot % 32) == 0);
    Array<uint> rfimask({nfreq,nt_tot/32}, af_rhost | af_zero);

    // Randomize rfimask.
    // Note: I checked that gputils::default_rng() returns a uint in which all bits are random.
    
    for (int i = 0; i < rfimask.size; i++)
	rfimask.data[i] = gputils::default_rng();
    
    vector<int> ix(nstations);
    for (int i = 0; i < nstations; i++)
	ix[i] = i;

    vector<complex<int>> z(M);

    for (int touter = 0; touter < nt_outer; touter++) {
	for (int t = touter*nt_inner; t < (touter+1)*nt_inner; t++) {
	    for (int f = 0; f < nfreq; f++) {
		// Generate M random indices in [0:nstations).
		for (int i = 0; i < M; i++) {
		    int j = rand_int(i, nstations);
		    std::swap(ix[i], ix[j]);
		}
	    
		// Generate M random E-array values.
		for (int i = 0; i < M; i++) {
		    z[i] = { int(rand_int(-7,8)), int(rand_int(-7,8)) };
		    emat.at({t,f,ix[i]}) = pack_complex44(z[i]);
		}

		// Skip updating vmat_cpu if RFI mask bit is zero.
		uint rm = rfimask.data[(f*nt_tot + t) / 32];
		uint bit = 1U << (t % 32);
		if ((rm & bit) == 0)
		    continue;

		// Update vmat_cpu.
		for (int i = 0; i < M; i++) {
		    int ahi = ix[i] >> 4;
		    int alo = ix[i] & 0xf;
		    
		    for (int j = 0; j < M; j++) {
			int bhi = ix[j] >> 4;
			int blo = ix[j] & 0xf;

			if (ahi < bhi)
			    continue;

			int vt = (ahi*(ahi+1))/2 + bhi;			
			complex<int> zz = z[i] * conj(z[j]);
			vmat_cpu.at({touter,f,vt,alo,blo,0}) += zz.real();
			vmat_cpu.at({touter,f,vt,alo,blo,1}) += zz.imag();
		    }
		}
	    }
	}
    }

    emat = emat.to_gpu();
    rfimask = rfimask.to_gpu();
    Array<int> vmat_gpu({nt_outer,nfreq,nvtiles,16,16,2}, af_random | af_gpu);

    Correlator corr(nstations, nfreq);
    corr.launch(vmat_gpu, emat, rfimask, nt_outer, nt_inner, nullptr, true);  // sync=true
    vmat_gpu = vmat_gpu.to_host();

    for (int touter = 0; touter < nt_outer; touter++) {
	for (int f = 0; f < nfreq; f++) {
	    for (int vt = 0; vt < nvtiles; vt++) {
		for (int i = 0; i < 16; i++) {
		    for (int k = i; k < 16; k++) {
			complex<int> vcpu = complex<int> (vmat_cpu.at({touter,f,vt,i,k,0}), vmat_cpu.at({touter,f,vt,i,k,1}));
			complex<int> vgpu = complex<int> (vmat_gpu.at({touter,f,vt,i,k,0}), vmat_gpu.at({touter,f,vt,i,k,1}));
		    
			if (vcpu != vgpu) {
			    cout << "test_correlator() failed at touter=" << touter
				 << ", f=" << f << ", vt=" << vt << ", i=" << i << ", k=" << k
				 << ": vcpu = " << vcpu << ", vgpu = " << vgpu
				 << endl;
			    
			    exit(1);
			}
		    }
		}
	    }
	}
    }
    
    cout << "test_correlator(): pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    test_negate_4bit();
    test_transpose_rank8_4bit();

    // In the "minimal" test, only two entries in the E-array are nonzero, and
    // the test output is verbose. This can be useful for tracking down bugs,
    // but we don't run it by default.

    // (nstations, nfreq, f, sa, sb, t, za, ab)
    minimal_correlator_test(128, 128, 3, 37, 23, 183, {1,2}, {3,4});
    minimal_correlator_test(1024, 8, 5, 335, 159, 137, {1,2}, {3,4});

    // Full end-to-end test starts here.
    
    std::mt19937 rng(137);
    const double maxbytes = 2.0e9;  // 2 GB

    // List of pairs (nstations, nfreq)
    vector<pair<int,int>> kparams = get_all_kernel_params();
    
    for (auto p: kparams) {
	int nstations = p.first;
	int nfreq = p.second;
	
	int nvtiles = ((nstations/16) * (nstations/16+1)) / 2;
	double nbytes_e = nfreq * nstations;           // multiply by (nt_inner * nt_outer)
	double nbytes_v = 2048.0 * nfreq * nvtiles;    // multiply by (nt_outer)
	
	int max_multiplier = int((0.9999*maxbytes - nbytes_v) / (256. * nbytes_e));
	int nt_inner = 256 * gputils::rand_int(1, min(max_multiplier,10)+1, rng);
	
	int max_nt_outer = int(maxbytes / (nt_inner*nbytes_e + nbytes_v));
	int nt_outer = gputils::rand_int(1, max_nt_outer+1, rng);
	test_correlator(nstations, nfreq, nt_outer, nt_inner);
    }

    return 0;
}
