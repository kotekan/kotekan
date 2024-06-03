#include <gputils.hpp>
#include "../include/n2k.hpp"

using namespace std;
using namespace gputils;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


CorrelatorParams::CorrelatorParams(int nstations_, int nfreq_) :
    nstations(nstations_),
    nfreq(nfreq_),
    emat_fstride(nstations/4),                // int32 stride, not bytes
    emat_tstride(nfreq * emat_fstride),       // int32 stride, not bytes
    vmat_istride(2 * nstations),              // int32 stride, not bytes (factor 2 is ReIm)
    vmat_fstride(2 * nstations * nstations),  // int32 stride, not bytes (factor 2 is ReIm)
    ntiles_1d(nstations / CorrelatorParams::ns_divisor),
    ntiles_2d_offdiag((ntiles_1d * (ntiles_1d-1)) / 2),
    ntiles_2d_tot(ntiles_2d_offdiag + ntiles_1d),
    threadblocks_per_freq(ntiles_2d_tot)
{
    assert(nstations > 0);
    assert(nfreq > 0);

    if (nstations % CorrelatorParams::ns_divisor)
	throw runtime_error("n2k: expected nstations(=" + to_str(nstations) + ") to be a multiple of " + to_str(CorrelatorParams::ns_divisor));
}

    
Correlator::Correlator(const CorrelatorParams &params_) : params(params_)
{
    this->precomputed_offsets = precompute_offsets(params);
    this->kernel = get_kernel(params.nstations, params.nfreq);
}


void Correlator::launch(int *vis_out, const int8_t *e_in, int nt_outer, int nt_inner, cudaStream_t stream, bool sync) const
{
    assert(nt_outer > 0);
    assert(nt_inner > 0);
    assert(vis_out != nullptr);
    assert(e_in != nullptr);

    if (nt_inner % CorrelatorParams::nt_divisor)
	throw runtime_error("n2k::Correlator::launch: expected nt_inner(=" + to_str(nt_inner) + ") to be a multiple of " + to_str(CorrelatorParams::nt_divisor));
    
    dim3 nblocks;
    nblocks.x = params.threadblocks_per_freq;
    nblocks.y = params.nfreq;
    nblocks.z = nt_outer;
    
    int nthreads = CorrelatorParams::threads_per_block;
    int shmem_nbytes = CorrelatorParams::shmem_nbytes;
    const int *poffsets = this->precomputed_offsets.get();
    
    kernel <<<nblocks, nthreads, shmem_nbytes, stream >>> (vis_out, e_in, poffsets, nt_inner);
    CUDA_PEEK("Correlator::launch");

    if (sync)
	CUDA_CALL(cudaStreamSynchronize(stream));
}


void Correlator::launch(Array<int> &vis_out, const Array<int8_t> &e_in, int nt_outer, int nt_inner, cudaStream_t stream, bool sync) const
{
    int nt_expected = nt_outer * nt_inner;
    int nstat = params.nstations;
    int nfreq = params.nfreq;
    
    if (!e_in.shape_equals({nt_expected,nfreq,nstat})) {
	stringstream ss;
	ss << "Correlator::launch(nfreq=" << nfreq << ", nt_outer=" << nt_outer << ", nt_inner=" << nt_inner << ")"
	   << ": expected emat shape=(" << nt_expected << "," << nfreq << "," << nstat << ")"
	   << ", got shape=" << e_in.shape_str();
	throw runtime_error(ss.str());
    }
			   
    bool vflag1 = vis_out.shape_equals({nt_outer, nfreq, nstat, nstat, 2});
    bool vflag2 = vis_out.shape_equals({nfreq, nstat, nstat, 2});
    bool vshape_ok = vflag1 || (vflag2 && (nt_outer == 1));

    if (!vshape_ok) {
	stringstream ss;
	ss << "Correlator::launch(nfreq=" << nfreq << ", nt_outer=" << nt_outer << ", nt_inner=" << nt_inner << ")"
	   << ": expected vmat shape=(" << nt_outer << "," << nfreq << "," << nstat << "," << nstat << ",2" << ")";

	if (nt_outer == 1)
	    ss << " or shape=(" << nfreq << "," << nstat << "," << nstat << ",2" << ")";

	ss << ", got shape=" << vis_out.shape_str();
	throw runtime_error(ss.str());
    }

    assert(vis_out.is_fully_contiguous());
    assert(vis_out.on_gpu());
    assert(e_in.is_fully_contiguous());
    assert(e_in.on_gpu());
    
    this->launch(vis_out.data, e_in.data, nt_outer, nt_inner, stream, sync);
}


}  // namespace n2k
