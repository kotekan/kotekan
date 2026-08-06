// Cloned from n2k/src_lib/kernel_table.cu (see include/n2k_dual/DualCorrelator.hpp for
// provenance). The registration key is (nstations_total, nfreq): the kernel template
// depends only on the TOTAL station count, so one instantiation serves every 128-multiple
// A/B split of the same total.

#include <mutex>
#include <stdexcept>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/string_utils.hpp>

#include "../include/n2k_dual/DualCorrelator.hpp"
#include "../include/n2k_dual/internals/DualCorrelatorKernel.hpp"

using namespace std;
using namespace ksgpu;

namespace n2k_dual {
#if 0
}  // editor auto-indent
#endif


struct KernelTableEntry
{
    int nstations = 0; // TOTAL station count (nstations_a + nstations_b)
    int nfreq = 0;

    DualCorrelator::kernel_t kernel = nullptr;
};


static vector<KernelTableEntry> kernel_table;
static mutex kernel_table_lock;


DualCorrelator::kernel_t get_kernel(int nstations, int nfreq)
{
    assert(nfreq > 0);
    assert(nstations > 0);

    if (nstations % DualCorrelatorParams::ns_divisor)
        throw runtime_error("n2k_dual::get_kernel(): expected nstations(=" + to_str(nstations)
                            + ") to be a multiple of "
                            + to_str(DualCorrelatorParams::ns_divisor));

    unique_lock<mutex> ul(kernel_table_lock);

    for (const auto& e : kernel_table) {
        if ((e.nstations != nstations) || (e.nfreq != nfreq))
            continue;

        // Called unconditionally so it runs at least once per cuda device (the shmem
        // request exceeds the 48 KB default).
        CUDA_CALL(cudaFuncSetAttribute(e.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       DualCorrelatorParams::shmem_nbytes));

        return e.kernel;
    }

    stringstream ss;
    ss << "n2k_dual: You have requested (nstations_total,nfreq)=(" << nstations << "," << nfreq
       << "). This pair of values is not supported. You will need to add"
       << " 'dual_kernel_" << nstations << "_" << nfreq << ".cu' to KERNELFILES in"
       << " external/n2k_dual/CMakeLists.txt and recompile.";

    throw runtime_error(ss.str());
}


void register_kernel(int nstations, int nfreq, DualCorrelator::kernel_t kernel)
{
    KernelTableEntry e;
    e.nstations = nstations;
    e.nfreq = nfreq;
    e.kernel = kernel;

    unique_lock<mutex> ul(kernel_table_lock);
    kernel_table.push_back(e);
}


vector<pair<int, int>> get_all_kernel_params()
{
    vector<pair<int, int>> ret;
    unique_lock<mutex> ul(kernel_table_lock);

    for (const auto& k : kernel_table)
        ret.push_back({k.nstations, k.nfreq});

    return ret;
}


} // namespace n2k_dual
