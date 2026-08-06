// Cloned from n2k/src_lib/Correlator.cu (see include/n2k_dual/DualCorrelator.hpp for
// provenance and the list of divergences).

#include "../include/n2k_dual/DualCorrelator.hpp"

#include <cassert>
#include <ksgpu.hpp>

using namespace std;
using namespace ksgpu;


namespace n2k_dual {
#if 0
}  // editor auto-indent
#endif


// How many threadblocks (per frequency) survive the class mask.
static int count_enabled_blocks(int ntiles_1d, int ntiles_a, int block_class_mask)
{
    return (int)dual_enumerate_blocks(ntiles_1d, ntiles_a, block_class_mask).size();
}


DualCorrelatorParams::DualCorrelatorParams(int nstations_a_, int nstations_b_, int nfreq_,
                                           int block_class_mask_,
                                           const std::vector<int>& freq_map_) :
    nstations_a(nstations_a_),
    nstations_b(nstations_b_),
    nstations(nstations_a_ + nstations_b_),
    nfreq(nfreq_),
    block_class_mask(block_class_mask_),
    freq_map(freq_map_),
    n_freq_out(freq_map_.empty() ? nfreq_ : (int)freq_map_.size()),
    emat_fstride_a(nstations_a / 4),               // int32 stride, not bytes
    emat_tstride_a(nfreq * emat_fstride_a),        // int32 stride, not bytes
    emat_fstride_b(nstations_b / 4),               // int32 stride, not bytes
    emat_tstride_b(nfreq * emat_fstride_b),        // int32 stride, not bytes
    vmat_ntiles(((nstations / 16) * (nstations / 16 + 1)) / 2),
    vmat_fstride(vmat_ntiles * 16 * 16 * 2), // int32 stride, not int32+32
    vmat_tstride(nfreq * vmat_fstride),      // int32 stride, not int32+32
    ntiles_a(nstations_a / DualCorrelatorParams::ns_divisor),
    ntiles_1d(nstations / DualCorrelatorParams::ns_divisor),
    ntiles_2d_offdiag((ntiles_1d * (ntiles_1d - 1)) / 2),
    ntiles_2d_tot(ntiles_2d_offdiag + ntiles_1d),
    threadblocks_per_freq(count_enabled_blocks(ntiles_1d, ntiles_a, block_class_mask))
{
    assert(nstations_a > 0);
    assert(nstations_b >= 0); // B input is optional (nstations_b == 0 reproduces n2k)
    assert(nfreq > 0);

    if (nstations_a % DualCorrelatorParams::ns_divisor)
        throw runtime_error("n2k_dual: expected nstations_a(=" + to_str(nstations_a)
                            + ") to be a multiple of " + to_str(DualCorrelatorParams::ns_divisor));
    if (nstations_b % DualCorrelatorParams::ns_divisor)
        throw runtime_error("n2k_dual: expected nstations_b(=" + to_str(nstations_b)
                            + ") to be a multiple of " + to_str(DualCorrelatorParams::ns_divisor));
    for (int f : freq_map)
        if (f < 0 || f >= nfreq)
            throw runtime_error("n2k_dual: freq_map entry " + to_str(f)
                                + " outside [0, nfreq=" + to_str(nfreq) + ")");
    if ((block_class_mask & BLOCK_MASK_ALL) == 0)
        throw runtime_error("n2k_dual: block_class_mask(=" + to_str(block_class_mask)
                            + ") enables no block classes");
    if (threadblocks_per_freq == 0)
        throw runtime_error("n2k_dual: block_class_mask(=" + to_str(block_class_mask)
                            + ") leaves no threadblocks for (nstations_a,nstations_b)=("
                            + to_str(nstations_a) + "," + to_str(nstations_b) + ")");
}


DualCorrelator::DualCorrelator(const DualCorrelatorParams& params_) : params(params_)
{
    CUDA_CALL(cudaGetDevice(&this->cuda_device));
    this->precomputed_offsets = precompute_offsets(params);
    if (!params.freq_map.empty()) {
        Array<int> fm({(int)params.freq_map.size()}, af_rhost);
        for (size_t i = 0; i < params.freq_map.size(); i++)
            fm.at({(int)i}) = params.freq_map[i];
        fm = fm.to_gpu();
        this->d_freq_map = std::static_pointer_cast<int>(fm.base);
    }
    this->kernel = get_kernel(params.nstations, params.nfreq);
}


void DualCorrelator::launch(int* vis_out, const int8_t* e_in_a, const int8_t* e_in_b,
                            const uint* rfimask, int nt_outer, int nt_inner, cudaStream_t stream,
                            bool sync) const
{
    int current_device;
    CUDA_CALL(cudaGetDevice(&current_device));

    if (current_device != this->cuda_device)
        throw runtime_error("n2k_dual::DualCorrelator::launch: current cuda device (="
                            + to_str(current_device)
                            + ") does not match device at construction time (="
                            + to_str(this->cuda_device) + ")");

    assert(nt_outer > 0);
    assert(nt_inner > 0);
    assert(vis_out != nullptr);
    assert(e_in_a != nullptr);
    assert((e_in_b != nullptr) == (params.nstations_b > 0));
    assert(rfimask != nullptr);

    if (nt_inner % DualCorrelatorParams::nt_divisor)
        throw runtime_error("n2k_dual::DualCorrelator::launch: expected nt_inner(="
                            + to_str(nt_inner) + ") to be a multiple of "
                            + to_str(DualCorrelatorParams::nt_divisor));

    dim3 nblocks;
    nblocks.x = params.threadblocks_per_freq;
    nblocks.y = params.n_freq_out;
    nblocks.z = nt_outer;

    int nthreads = DualCorrelatorParams::threads_per_block;
    int shmem_nbytes = DualCorrelatorParams::shmem_nbytes;
    const int* poffsets = this->precomputed_offsets.get();

    kernel<<<nblocks, nthreads, shmem_nbytes, stream>>>(vis_out, e_in_a, e_in_b, rfimask, poffsets,
                                                        this->d_freq_map.get(),
                                                        params.n_freq_out, nt_inner, nt_outer);
    CUDA_PEEK("DualCorrelator::launch");

    if (sync)
        CUDA_CALL(cudaStreamSynchronize(stream));
}


void DualCorrelator::launch(Array<int>& vis_out, const Array<int8_t>& e_in_a,
                            const Array<int8_t>& e_in_b, const Array<uint>& rfimask, int nt_outer,
                            int nt_inner, cudaStream_t stream, bool sync) const
{
    int nt_expected = nt_outer * nt_inner;
    int nrfi_expected = (nt_expected / 32);
    int vmat_ntiles = params.vmat_ntiles;
    int nstat_a = params.nstations_a;
    int nstat_b = params.nstations_b;
    int nfreq = params.nfreq;

    if (!e_in_a.shape_equals({nt_expected, nfreq, nstat_a})) {
        stringstream ss;
        ss << "DualCorrelator::launch(nfreq=" << nfreq << ", nt_outer=" << nt_outer
           << ", nt_inner=" << nt_inner << ")"
           << ": expected emat_a shape=(" << nt_expected << "," << nfreq << "," << nstat_a << ")"
           << ", got shape=" << e_in_a.shape_str();
        throw runtime_error(ss.str());
    }

    if ((nstat_b > 0) && !e_in_b.shape_equals({nt_expected, nfreq, nstat_b})) {
        stringstream ss;
        ss << "DualCorrelator::launch(nfreq=" << nfreq << ", nt_outer=" << nt_outer
           << ", nt_inner=" << nt_inner << ")"
           << ": expected emat_b shape=(" << nt_expected << "," << nfreq << "," << nstat_b << ")"
           << ", got shape=" << e_in_b.shape_str();
        throw runtime_error(ss.str());
    }

    const int nfout = params.n_freq_out;
    bool vflag1 = vis_out.shape_equals({nt_outer, nfout, vmat_ntiles, 16, 16, 2});
    bool vflag2 = vis_out.shape_equals({nfout, vmat_ntiles, 16, 16, 2});
    bool vshape_ok = vflag1 || (vflag2 && (nt_outer == 1));

    if (!vshape_ok) {
        stringstream ss;
        ss << "DualCorrelator::launch(nfreq=" << nfreq << ", nt_outer=" << nt_outer
           << ", nt_inner=" << nt_inner << ")"
           << ": expected vmat shape=(" << nt_outer << "," << nfreq << "," << vmat_ntiles
           << ",16,16,2" << ")";

        if (nt_outer == 1)
            ss << " or shape=(" << nfreq << "," << vmat_ntiles << ",16,16,2" << ")";

        ss << ", got shape=" << vis_out.shape_str();
        throw runtime_error(ss.str());
    }

    if (!rfimask.shape_equals({nfreq, nrfi_expected})) {
        stringstream ss;
        ss << "DualCorrelator::launch(nfreq=" << nfreq << ", nt_outer=" << nt_outer
           << ", nt_inner=" << nt_inner << ")"
           << ": expected rfimask shape=(" << nfreq << "," << nrfi_expected << ")"
           << ", got shape=" << rfimask.shape_str();
        throw runtime_error(ss.str());
    }

    assert(vis_out.is_fully_contiguous());
    assert(vis_out.on_gpu());
    assert(e_in_a.is_fully_contiguous());
    assert(e_in_a.on_gpu());
    if (nstat_b > 0) {
        assert(e_in_b.is_fully_contiguous());
        assert(e_in_b.on_gpu());
    }
    assert(rfimask.is_fully_contiguous());
    assert(rfimask.on_gpu());

    const int8_t* b_data = (nstat_b > 0) ? e_in_b.data : nullptr;
    this->launch(vis_out.data, e_in_a.data, b_data, rfimask.data, nt_outer, nt_inner, stream,
                 sync);
}


} // namespace n2k_dual
