#include "../include/n2k/rfi_kernels.hpp"

#include <gputils/CudaStreamPool.hpp>

using namespace std;
using namespace gputils;
using namespace n2k;


struct TimingParams
{
    string name;

    long T = 0;
    long F = 0;
    long S = 0;
    long Nds1 = 0;
    long Nds2 = 0;
    float baseband_dt = 0.0;  // seconds
};


static void time_s0_kernel(const TimingParams &tp)
{
    const long ninner = 1000;
    const long nouter = 10;
    
    long T = tp.T;
    long F = tp.F;
    long S = tp.S;
    long Nds = tp.Nds1;
    
    Array<ulong> S0({T/Nds, F, S}, af_gpu | af_zero);
    Array<ulong> pl_mask({T/128, (F+3)/4, S/8}, af_gpu | af_zero);

    string name = tp.name + " s0_kernel";
    double gb = 8.0e-9 * ninner * (S0.size + pl_mask.size);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	for (int i = 0; i < ninner; i++)
	    launch_s0_kernel(S0, pl_mask, Nds, stream);  // calls CUDA_PEEK()
    };

    CudaStreamPool sp(callback, nouter, 1, name);
    sp.monitor_throughput("Global memory BW (GB/s)", gb);
    sp.monitor_time("Real-time fraction", ninner * tp.T * tp.baseband_dt);
    sp.run();
}


static void time_s12_kernel(const TimingParams &tp)
{
    const long ninner = 100;
    const long nouter = 10;
    const bool offset_encoded = true;
    
    long T = tp.T;
    long F = tp.F;
    long S = tp.S;
    long Nds = tp.Nds1;

    Array<ulong> S12({T/Nds,F,2,S}, af_gpu | af_zero);
    Array<uint8_t> E({T,F,S}, af_gpu | af_zero);

    string name = tp.name + " s12_kernel";
    double gb = 1.0e-9 * ninner * (8*S12.size + E.size);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	for (int i = 0; i < ninner; i++)
	    launch_s12_kernel(S12, E, Nds, offset_encoded, stream);  // calls CUDA_PEEK()
    };

    CudaStreamPool sp(callback, nouter, 1, name);
    sp.monitor_throughput("Global memory BW (GB/s)", gb);
    sp.monitor_time("Real-time fraction", ninner * tp.T * tp.baseband_dt);
    sp.run();
}


static void time_s012_time_downsample(const TimingParams &tp)
{
    const long ninner = 100;
    const long nouter = 10;
    
    long T = tp.T / tp.Nds1;
    long F = tp.F;
    long S = tp.S;
    long Nds = tp.Nds2;

    Array<ulong> Sout({T/Nds,F,3,S}, af_gpu | af_zero);
    Array<ulong> Sin({T,F,3,S}, af_gpu | af_zero);

    string name = tp.name + " s012_time_downsample";
    double gb = 8.0e-9 * ninner * (Sout.size + Sin.size);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	for (int i = 0; i < ninner; i++)
	    launch_s012_time_downsample_kernel(Sout, Sin, Nds, stream);  // calls CUDA_PEEK()
    };

    CudaStreamPool sp(callback, nouter, 1, name);
    sp.monitor_throughput("Global memory BW (GB/s)", gb);
    sp.monitor_time("Real-time fraction", ninner * tp.T * tp.baseband_dt);
    sp.run();
}


static void time_s012_station_downsample(const TimingParams &tp)
{
    const long ninner = 100;
    const long nouter = 10;
    
    long T = tp.T / tp.Nds1;
    long F = tp.F;
    long S = tp.S;

    Array<ulong> Sout({T,F,3}, af_gpu | af_zero);
    Array<ulong> Sin({T,F,3,S}, af_gpu | af_zero);
    Array<uint8_t> bf_mask({S}, af_gpu | af_zero);

    string name = tp.name + " s012_station_downsample";
    double gb = 8.0e-9 * ninner * (Sout.size + Sin.size);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	for (int i = 0; i < ninner; i++)
	    launch_s012_station_downsample_kernel(Sout, Sin, bf_mask, stream);  // calls CUDA_PEEK()
    };

    CudaStreamPool sp(callback, nouter, 1, name);
    sp.monitor_throughput("Global memory BW (GB/s)", gb);
    sp.monitor_time("Real-time fraction", ninner * tp.T * tp.baseband_dt);
    sp.run();
}


static void time_sk_kernel(const TimingParams &tp, bool first_flag)
{
    const long ninner = 100;
    const long nouter = 10;

    SkKernel::Params params;
    params.sk_rfimask_sigmas = 10.0;
    params.single_feed_min_good_frac = 0.5;
    params.feed_averaged_min_good_frac = 0.5;
    params.mu_min = 2.0;
    params.mu_max = 50.0;
    params.Nds = first_flag ? tp.Nds1 : (tp.Nds1 * tp.Nds2);

    SkKernel kernel(params);

    long T = tp.T / params.Nds;
    long F = tp.F;
    long S = tp.S;
    
    Array<float> out_sk_feed_averaged({T,F,3}, af_zero | af_gpu);
    Array<ulong> in_S012({T,F,3,S}, af_zero | af_gpu);
    Array<uint8_t> in_bf_mask({S}, af_zero | af_gpu);
    
    Array<float> out_sk_single_feed;
    Array<uint> out_rfimask;

    if (!first_flag)
	out_sk_single_feed = Array<float> ({T,F,3,S}, af_zero | af_gpu);
    if (first_flag)
	out_rfimask = Array<uint> ({F,(T*params.Nds)/32}, af_zero | af_gpu);
    
    string name = tp.name + (first_flag ? " first SkKernel" : " second SkKernel");
    double gb = 4.0e-9 * ninner * out_sk_feed_averaged.size;
    gb += 8.0e-9 * ninner * in_S012.size;
    gb += 1.0e-9 * ninner * in_bf_mask.size;
    gb += 4.0e-9 * ninner * out_sk_single_feed.size;
    gb += 4.0e-9 * ninner * out_rfimask.size;

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	for (int i = 0; i < ninner; i++)
	    kernel.launch(out_sk_feed_averaged, out_sk_single_feed, out_rfimask, in_S012, in_bf_mask, stream);  // calls CUDA_PEEK()
    };
    
    CudaStreamPool sp(callback, nouter, 1, name);
    sp.monitor_throughput("Global memory BW (GB/s)", gb);
    sp.monitor_time("Real-time fraction", ninner * tp.T * tp.baseband_dt);
    sp.run();
}


static void time_all(const TimingParams &tp)
{
    time_s0_kernel(tp);
    time_s12_kernel(tp);
    time_s012_time_downsample(tp);
    time_s012_station_downsample(tp);
    time_sk_kernel(tp, true);
    time_sk_kernel(tp, false);
}


int main(int argc, char **argv)
{
    const double chord_dt = 8192 / 1.5e9;
    
    TimingParams tp_chord_pathfinder;
    tp_chord_pathfinder.name = "CHORD pathfinder";
    tp_chord_pathfinder.T = 128 * 48 * 8;
    tp_chord_pathfinder.F = 410;
    tp_chord_pathfinder.S = 128;
    tp_chord_pathfinder.Nds1 = 128;
    tp_chord_pathfinder.Nds2 = 48;
    tp_chord_pathfinder.baseband_dt = chord_dt;
    time_all(tp_chord_pathfinder);

    TimingParams tp_full_chord;
    tp_full_chord.name = "Full CHORD";
    tp_full_chord.T = 128 * 48 * 8;
    tp_full_chord.F = 51;
    tp_full_chord.S = 1024;
    tp_full_chord.Nds1 = 128;
    tp_full_chord.Nds2 = 48;
    tp_full_chord.baseband_dt = chord_dt;
    time_all(tp_full_chord);
    
    return 0;
}
