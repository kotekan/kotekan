#ifndef GPU_HSA_DEVICE_INTERFACE_H
#define GPU_HSA_DEVICE_INTERFACE_H

#include "Config.hpp"
#include "gpuDeviceInterface.hpp" // for Config, gpuDeviceInterface
#include "hsa/hsa.h"              // for hsa_agent_t, hsa_signal_t, hsa_status_t, hsa_region_t
#include "hsa/hsa_ext_amd.h"      // for hsa_amd_memory_pool_t

#include <stdint.h> // for uint32_t, uint64_t, int32_t


// Parameters for the get_gpu_agent function
struct gpu_config_t {
    int gpu_id;
    hsa_agent_t* agent;
};

// Parameters for the get_device_memory_region function
struct gpu_mem_config_t {
    int gpu_id;
    hsa_amd_memory_pool_t* region;
};


class hsaDeviceInterface : public gpuDeviceInterface {
public:
    hsaDeviceInterface(kotekan::Config& config, const std::string& unique_name, int32_t gpu_id,
                       int gpu_buffer_depth, uint32_t numa_node);
    virtual ~hsaDeviceInterface();

    // Note, if precede_signal is 0, then we don't wait on any signal.
    // These functions should only be called once per command, and
    // the returned signal should be stored in signals[]
    // These functions are reentrant/thread-safe.
    hsa_signal_t async_copy_host_to_gpu(void* dst, void* src, int len, hsa_signal_t precede_signal,
                                        hsa_signal_t copy_signal);
    hsa_signal_t async_copy_gpu_to_host(void* dst, void* src, int len, hsa_signal_t precede_signal,
                                        hsa_signal_t copy_signal);

    // Not used for data copies, this should only be used for static tables, etc.
    // at start up.  Will block until all memory has been copied.
    // These functions are reentrant/thread-safe.
    void sync_copy_gpu_to_host(void* dst, void* src, int length);
    void sync_copy_host_to_gpu(void* dst, void* src, int length);

    hsa_agent_t get_gpu_agent();
    hsa_region_t get_kernarg_region();
    hsa_agent_t get_cpu_agent();
    hsa_queue_t* get_queue();
    uint64_t get_hsa_timestamp_freq();
    uint32_t get_gpu_numa_node();

protected:
    void* alloc_gpu_memory(size_t len) override;
    void free_gpu_memory(void*) override;

    // GPU HSA variables
    hsa_agent_t gpu_agent;
    hsa_amd_memory_pool_t global_region;
    hsa_region_t kernarg_region;

    // This might need to be more than one queue
    hsa_queue_t* queue;

    // HSA profiling time stamp resolution
    uint64_t timestamp_frequency_hz;

    // GPU Information
    char agent_name[64];

    // CPU HSA variables
    hsa_agent_t cpu_agent;
    hsa_amd_memory_pool_t host_region;

    // The NUMA node this GPU is attached too.
    uint32_t numa_node;

private:
    static hsa_status_t get_gpu_agent(hsa_agent_t agent, void* data);
    static hsa_status_t get_cpu_agent(hsa_agent_t agent, void* data);
    static hsa_status_t get_kernarg_memory_region(hsa_region_t region, void* data);
    static hsa_status_t get_device_memory_region(hsa_amd_memory_pool_t region, void* data);
};

#endif // GPU_HSA_DEVICE_INTERFACE_H
