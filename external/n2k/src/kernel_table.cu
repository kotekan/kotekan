#include "../include/n2k.hpp"
#include "../include/n2k_kernel.hpp"

using namespace std;
using namespace gputils;

namespace n2k {
#if 0
}  // editor auto-indent
#endif


// FIXME it would be slightly better to have a per-KernelTableEntry lock, for setting shmem size.
struct KernelTableEntry
{
    int nstations = 0;
    int nfreq = 0;
    
    Correlator::kernel_t kernel = nullptr;
    mutable bool shmem_attr_set = false;
};


static vector<KernelTableEntry> kernel_table;
static mutex kernel_table_lock;


Correlator::kernel_t get_kernel(int nstations, int nfreq)
{
    assert(nfreq > 0);
    assert(nstations > 0);

    if (nstations % CorrelatorParams::ns_divisor)
	throw runtime_error("n2k::Correlator::get_kernel(): expected nstations(=" + to_str(nstations) + ") to be a multiple of " + to_str(CorrelatorParams::ns_divisor));
    
    unique_lock<mutex> ul(kernel_table_lock);

    for (const auto &e: kernel_table) {
	if ((e.nstations != nstations) || (e.nfreq != nfreq))
	    continue;
    
	// Reference for cudaFuncSetAttribute()
	// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g422642bfa0c035a590e4c43ff7c11f8d
    
	if (!e.shmem_attr_set) {
	    CUDA_CALL(cudaFuncSetAttribute(
	        e.kernel, 
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		CorrelatorParams::shmem_nbytes
	    ));
	    e.shmem_attr_set = true;
	}

	return e.kernel;
    }
    
    stringstream ss;
    ss << "n2k: You have requested (nstations,nfreq)=(" << nstations << "," << nfreq << "). This pair of values is not supported."
       << " Sadly, you will need to add 'template_instantiations/kernel_" << nstations << "_" << nfreq << ".o' to the Makefile and recompile."
       << " In the future, I may do this automatically with nvrtc.";
    
    throw runtime_error(ss.str());
}


void register_kernel(int nstations, int nfreq, Correlator::kernel_t kernel)
{
    KernelTableEntry e;
    e.nstations = nstations;
    e.nfreq = nfreq;
    e.kernel = kernel;
    
    unique_lock<mutex> ul(kernel_table_lock);
    kernel_table.push_back(e);
}


vector<pair<int,int>> get_all_kernel_params()
{
    vector<pair<int,int>> ret;
    unique_lock<mutex> ul(kernel_table_lock);
    
    for (const auto &k: kernel_table)
	ret.push_back({k.nstations, k.nfreq});

    return ret;
}


}  // namespace n2k
