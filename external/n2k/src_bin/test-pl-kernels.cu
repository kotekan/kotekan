#include "../include/n2k/pl_kernels.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/rand_utils.hpp>
#include <gputils/test_utils.hpp>

using namespace std;
using namespace n2k;
using namespace gputils;


inline uint bit_count(ulong x)
{
    int ret = 0;
    
    for (int i = 0; i < 64; i++)
	ret += ((x & (1UL << i)) ? 1 : 0);

    return ret;
}


inline void double_bits(ulong &y0, ulong &y1, ulong x)
{
    y0 = y1 = 0;

    for (int i = 0; i < 64; i++) {
	if ((x & (1UL << i)) == 0)
	    continue;
	if (i < 32)
	    y0 |= (3UL << (2*i));
	else
	    y1 |= (3UL << (2*i-64));
    }
}


inline uint rand_uint()
{
    uint x = uint(gputils::default_rng());
    x ^= (uint(gputils::default_rng()) << 16);
    return x;
}


inline ulong rand_ulong()
{
    ulong x = ulong(gputils::default_rng());
    x ^= (ulong(gputils::default_rng()) << 22);
    x ^= (ulong(gputils::default_rng()) << 44);
    return x;
}


// ------------------------------------------------------------------------------------------------


static void test_pl_mask_expander(long Tout, long Fout, long Sds)
{
    cout << "test_pl_mask_expander: Tout=" << Tout << ", Fout=" << Fout << ", Sds=" << Sds << endl;

    long Tin = Tout/2;
    long Fin = (Fout+3)/4;
    Array<ulong> pl_in_cpu({Tin/64,Fin,Sds}, af_rhost);
    Array<ulong> pl_out_cpu({Tout/64,Fout,Sds}, af_uhost);
    Array<ulong> pl_out_gpu({Tout/64,Fout,Sds}, af_gpu | af_guard);

    for (long i = 0; i < pl_in_cpu.size; i++)
	pl_in_cpu.data[i] = rand_ulong();

    Array<ulong> pl_in_gpu = pl_in_cpu.to_gpu();
    launch_pl_mask_expander(pl_out_gpu, pl_in_gpu);
    CUDA_CALL(cudaDeviceSynchronize());

    // CPU implementation of PL mask expander starts here.

    for (long tin64 = 0; tin64 < (Tin/64); tin64++) {
	for (long fin = 0; fin < Fin; fin++) {
	    for (long s = 0; s < Sds; s++) {
		ulong x = pl_in_cpu.at({tin64,fin,s});
		
		ulong y0, y1;
		double_bits(y0, y1, x);

		for (long fout = 4*fin; fout < min(4*fin+4,Fout); fout++) {
		    pl_out_cpu.at({2*tin64,fout,s}) = y0;
		    pl_out_cpu.at({2*tin64+1,fout,s}) = y1;
		}
	    }
	}
    }

    gputils::assert_arrays_equal(pl_out_cpu, pl_out_gpu, "cpu", "gpu", {"t64","f","s"});
}


static void test_pl_mask_expander()
{
    for (int n = 0; n < 200; n++) {
	// (Tout/128, Fout, Sds/16)
	vector<ssize_t> v = random_integers_with_bounded_product(3, 20000);
	long Tout = 128 * v[0];
	long Fout = v[1];
	long Sds = 16 * v[2];
	test_pl_mask_expander(Tout, Fout, Sds);
    }
}


// ------------------------------------------------------------------------------------------------


static void test_pl_1bit_correlator(long T, long F, long Sds, long Nds, long rfimask_fstride)
{
    cout << "test_pl_1bit_correlator: T=" << T << ", F=" << F << ", Sds=" << Sds
	 << ", Nds=" << Nds << ", rfimask_fstride=" << rfimask_fstride << endl;
    
    assert(rfimask_fstride >= T/32);

    long Tout = T / Nds;
    long ntiles = ((Sds/8) * ((Sds/8)+1)) / 2;
    
    Array<ulong> pl_cpu({T/64,F,Sds}, af_rhost | af_zero);
    Array<uint> rfimask_cpu({F,T/32}, af_rhost | af_zero);
    Array<int> counts_cpu({Tout,F,ntiles,8,8}, af_uhost);
    Array<int> counts_gpu({Tout,F,ntiles,8,8}, af_gpu | af_guard);

    for (long i = 0; i < pl_cpu.size; i++)
	pl_cpu.data[i] = rand_ulong();

    for (long f = 0; f < F; f++)
	for (long t32 = 0; t32 < T/32; t32++)
	    rfimask_cpu.data[f*rfimask_fstride + t32] = rand_uint();
    
    Array<ulong> pl_gpu = pl_cpu.to_gpu();
    Array<uint> rfimask_gpu({F,T/32}, {rfimask_fstride,1}, af_gpu | af_guard);  // note fstride
    rfimask_gpu.fill(rfimask_cpu);
    launch_pl_1bit_correlator(counts_gpu, pl_gpu, rfimask_gpu, Nds);
    CUDA_CALL(cudaDeviceSynchronize());

    // CPU implementation of correlate_pl_kernel() starts here.

    int N64 = Nds / 64;
    int t64_stride = F*Sds;
    
    for (long tout = 0; tout < Tout; tout++) {
	for (long f = 0; f < F; f++) {
	    const ulong *pl_tf = &pl_cpu.at({tout*N64,f,0});  // shape (N64,Sds), strides (t64_stride, 1)
	    const ulong *rfi_tf = (const ulong *) &rfimask_cpu.at({f,tout*N64*2});        // shape (N64,)
	    int *counts_tf = &counts_cpu.at({tout,f,0,0,0});            // shape (ntiles,8,8), contiguous
	    
	    for (long ixtile = 0; ixtile < (Sds/8); ixtile++) {
		for (long iytile = 0; iytile <= ixtile; iytile++) {
		    long itile = (ixtile*(ixtile+1))/2 + iytile;
		    const ulong *plx = pl_tf + 8*ixtile;   // shape (N64,8), strides (t64_stride, 1)
		    const ulong *ply = pl_tf + 8*iytile;   // shape (N64,8), strides (t64_stride, 1)
		    int *ctile = counts_tf + 64*itile;     // shape (8,8), contiguous

		    for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
			    int v = 0;
			    for (int t64 = 0; t64 < N64; t64++) {
				ulong x = plx[t64*t64_stride + i];
				ulong y = ply[t64*t64_stride + j];
				ulong rfi = rfi_tf[t64];
				v += bit_count(x & y & rfi);
			    }
			    ctile[8*i+j] = v;
			}
		    }
		}
	    }
	}
    }

    gputils::assert_arrays_equal(counts_cpu, counts_gpu, "cpu", "gpu", {"tout","f","tile","i","j"});
}


static void test_pl_1bit_correlator()
{
    for (int n = 0; n < 100; n++) {
	// For now, only Sds=16 and Sds=128 are implemented.
	long Sds = rand_int(0,2) ? 16 : 128;
	
	// v = (T/Nds, F, Nds/128)
	vector<ssize_t> v = gputils::random_integers_with_bounded_product(3, 10*1000*1000/(Sds*Sds));
	long Nds = v[2] * 128;
	long F = v[1];
	long T = v[0] * Nds;
	long rfimask_fstride = rand_int(T/32, T/16);
	
	test_pl_1bit_correlator(T, F, Sds, Nds, rfimask_fstride);
    }
}


// ------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    test_pl_mask_expander();
    test_pl_1bit_correlator();
    return 0;
}
