#include <iostream>
#include <gputils.hpp>

#include "../include/n2k/Correlator.hpp"
#include "../argparse/argparse.hpp"

using namespace std;
using namespace gputils;
using namespace n2k;


struct TimingParams
{
    long nstations = 1024;
    long nfreq = 16;
    long nt_inner = 256 * 1024;
    long nt_tot = 256 * 1024;
    long nstreams = 1;
    long ncallbacks = 300;
    bool randomize = false;

    // Defaults
    TimingParams() { }

    // Parse command line.
    TimingParams(int argc, const char **argv)
    {
	argparse::ArgumentParser program("time-correlator");

	program.add_argument("--nstations")
	    .help("A 'station' is a (dish,polarization) pair. Number of stations is 1024 for full CHORD [default], or 128 for CHORD pathfinder.")
	    .default_value(std::string("1024"));

	program.add_argument("--nfreq")
	    .help("Number of freqency channels per GPU. Default is 16K/nstations [correct for both CHORD pathfinder and full CHORD]");
	
	program.add_argument("--nt-inner")
	    .help("Number of E-field time samples per visibility matrix sample [default 256K]")
	    .default_value(std::string("262144"));
	
	program.add_argument("--nt-tot")
	    .help("Number of E-field time samples per kernel launch [default 256K]")
	    .default_value(std::string("262144"));

	program.add_argument("--nstreams")
	    .help("Number of CUDA streams used for timing [default 1]")
	    .default_value(std::string("1"));

	program.add_argument("--ncallbacks")
	    .help("Number of callbacks used for timing [default 300]")
	    .default_value(std::string("300"));

	program.add_argument("--randomize")
	    .help("initalize E-field array using random data [default is zeroed data]")
	    .default_value(false)
	    .implicit_value(true);

	program.parse_args(argc, argv);

	// Note: we use gputils::from_str<long>(program.get(xx)) instead of program.get<long>(xx), since the latter segfaults!
	nstations = gputils::from_str<long> (program.get("--nstations"));
	assert(nstations > 0);
	assert((16384 % nstations) == 0);
	
	nfreq = program.is_used("--nfreq") ? gputils::from_str<long> (program.get("--nfreq")) : (16384/nstations);
	assert(nfreq > 0);
	
	nt_inner = gputils::from_str<long> (program.get("--nt-inner"));
	nt_tot = gputils::from_str<long> (program.get("--nt-tot"));	
	nstreams = gputils::from_str<long> (program.get("--nstreams"));
	ncallbacks = gputils::from_str<long> (program.get("--ncallbacks"));
	randomize = (program["--randomize"] == true);
	
	assert(nt_tot > 0);
	assert(nt_inner > 0);
	assert((nt_tot % nt_inner) == 0);
	assert(nstreams > 0);	
	assert(ncallbacks > 0);
    }
};


static Array<uint> make_rfimask(long nfreq, long nt_tot)
{
    assert((nt_tot % 32) == 0);
    Array<uint> rfimask({nfreq, nt_tot/32}, af_rhost);
    memset(rfimask.data, 0xff, rfimask.size * sizeof(uint));
    return rfimask.to_gpu();
}


static void time_correlator(const TimingParams &params)
{
    long nstat = params.nstations;
    long nfreq = params.nfreq;
    long nt_inner = params.nt_inner;
    long nt_tot = params.nt_tot;
    long nstreams = params.nstreams;
    long ncallbacks = params.ncallbacks;
    long nvtiles = ((nstat/16) * (nstat/16+1)) / 2;
    
    cout << "time-correlator:"
	 << " nstations=" << nstat
	 << ", nfreq=" << nfreq
	 << ", nt_inner=" << nt_inner
	 << ", nt_tot=" << nt_tot
	 << ", nstreams=" << nstreams
	 << ", ncallbacks=" << ncallbacks
	 << ", randomize=" << params.randomize
	 << endl;

    assert(nstat > 0);
    assert(nfreq > 0);
    assert(nt_inner > 0);
    assert(nt_tot > 0);
    assert((nt_tot % nt_inner) == 0);
    
    long nt_outer = nt_tot / nt_inner;
    double varr_gb = nstreams * nt_outer * nfreq * nvtiles * 2048. / pow(2,30.);
    double earr_gb = nstreams * nt_tot * double(nfreq*nstat) / pow(2,30);
    // double rt_sec = (nfreq/16.) * (nt_tot * 1.707e-6);  // CHORD: 16 freqs per gpu, 1.707 usec samples

    // In CHORD, time samples are 1.707 usec
    // (FIXME no longer true -- need to change some numbers here)
    double chord_tsamp = 1.707e-6;      // CHORD: 1.707 usec time samples
    double chord_samp_nbytes = 16384.;  // CHORD: 16K E-array bytes/sample
    double chord_vsamp = nt_inner * 1.707e-6;
    double chord_datavol = (nstat * nfreq * nt_tot) / chord_samp_nbytes * chord_tsamp;
    
    cout << "GPU memory usage will be " << (varr_gb + earr_gb) << " GB\n"
	 << "In CHORD, this would correspond to visibility sampling rate: " << chord_vsamp << " sec\n"
	 << "In CHORD, this data volume would correspond to " << chord_datavol << " seconds of real-time data"
	 << endl;
    
    Correlator corr(nstat, nfreq);
    vector<Array<int>> varr(nstreams);
    vector<Array<int8_t>> earr(nstreams);
    vector<Array<uint>> rfimask(nstreams);
    int earr_flags = params.randomize ? (af_random | af_gpu) : (af_zero | af_gpu);

    for (int i = 0; i < nstreams; i++) {
	varr[i] = Array<int> ({nt_outer, nfreq, nvtiles, 16, 16, 2}, af_zero | af_gpu);
	earr[i] = Array<int8_t> ({nt_tot, nfreq, nstat}, earr_flags);
	rfimask[i] = make_rfimask(nfreq, nt_tot);
    }

    cout << "The first few kernels run slow, for reasons I haven't understood yet!\n"
	 << "To mitigate this, the first kernel will be untimed.\n" << endl;
    
    corr.launch(varr[0], earr[0], rfimask[0], nt_outer, nt_inner, nullptr, true);

    cout << "Now running timed kernels. The first few kernels will be a few percent\n"
	 << "slower than the long-run average, but the timing will quickly settle down.\n"
	 << endl;

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    corr.launch(varr[istream], earr[istream], rfimask[istream], nt_outer, nt_inner, stream);
	};
    
    stringstream sp_name;
    sp_name << "n2k (chord vsamp=" << chord_vsamp << ")";
    
    CudaStreamPool sp(callback, ncallbacks, nstreams, sp_name.str());
    sp.monitor_time("CHORD real-time fraction", chord_datavol);
    sp.run();

    // Wrapper script 'run.sh' will capture this last line with 'tail -1'.
    cout << chord_vsamp << " " << (sp.time_per_callback / chord_datavol) << endl;
}


int main(int argc, const char **argv)
{
    TimingParams params(argc, argv);
    time_correlator(params);
    return 0;
}
