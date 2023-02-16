#include "hsaDeviceInterface.hpp"

#include "hsa/hsa.h"          // for hsa_agent_get_info, hsa_status_t, hsa_signal_t, HSA_STATUS...
#include "hsa/hsa_ext_amd.h"  // for hsa_amd_memory_async_copy, hsa_amd_agents_allow_access
#include "hsaBase.h"          // for HSA_CHECK
#include "kotekanLogging.hpp" // for DEBUG, INFO, INFO_NON_OO

#include <assert.h> // for assert

using kotekan::Config;

void error_callback(hsa_status_t status, hsa_queue_t* queue, void* data) {
    // Unused parameter, suppress warning.
    (void)data;

    const char* message;
    hsa_status_string(status, &message);
    INFO_NON_OO("ERROR *********** ERROR at queue {:d}: {:s} ************* ERROR\n", queue->id,
                message);
}

hsaDeviceInterface::hsaDeviceInterface(Config& config_, const std::string& unique_name,
                                       int32_t gpu_id_, int gpu_buffer_depth_,
                                       uint32_t numa_node_) :
    gpuDeviceInterface(config_, unique_name, gpu_id_, gpu_buffer_depth_) {
    (void)unique_name;
    numa_node = numa_node_;

    hsa_status_t hsa_status;

    // Function parameters
    gpu_config_t gpu_config;
    gpu_config.agent = &gpu_agent;
    gpu_config.gpu_id = gpu_id_;

    // Get the CPU agent
    hsa_status = hsa_iterate_agents(get_cpu_agent, &cpu_agent);
    if (hsa_status == HSA_STATUS_INFO_BREAK)
        hsa_status = HSA_STATUS_SUCCESS;
    HSA_CHECK(hsa_status);
    // Get the CPU memory region.
    hsa_amd_agent_iterate_memory_pools(cpu_agent, get_device_memory_region, &host_region);

    // Get GPU agent
    hsa_status = hsa_iterate_agents(get_gpu_agent, &gpu_config);
    if (hsa_status == HSA_STATUS_INFO_BREAK)
        hsa_status = HSA_STATUS_SUCCESS;
    HSA_CHECK(hsa_status);

    // Get GPU agent name and number
    hsa_status = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, agent_name);
    HSA_CHECK(hsa_status);
    int num;
    hsa_status = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NODE, &num);
    HSA_CHECK(hsa_status);

    INFO("Initializing HSA GPU type {:s} at index {:d}.", agent_name, num - 1);

    global_region.handle = (uint64_t)-1;
    hsa_amd_agent_iterate_memory_pools(gpu_agent, get_device_memory_region, &global_region);
    hsa_status = (global_region.handle == (uint64_t)-1) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
    HSA_CHECK(hsa_status);

    // Find a memory region that supports kernel arguments
    kernarg_region.handle = (uint64_t)-1;
    hsa_agent_iterate_regions(gpu_agent, get_kernarg_memory_region, &kernarg_region);
    hsa_status = (kernarg_region.handle == (uint64_t)-1) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
    HSA_CHECK(hsa_status);

    // Query the maximum size of the queue.
    // Create a queue using the maximum size.
    uint32_t queue_size = 0;
    hsa_status = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    HSA_CHECK(hsa_status);
    hsa_status = hsa_queue_create(gpu_agent, queue_size, HSA_QUEUE_TYPE_MULTI, error_callback,
                                  nullptr, UINT32_MAX, UINT32_MAX, &queue);
    HSA_CHECK(hsa_status);

    hsa_status = hsa_amd_profiling_set_profiler_enabled(queue, 1);
    HSA_CHECK(hsa_status);

    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_frequency_hz);
}

hsaDeviceInterface::~hsaDeviceInterface() {
    hsa_status_t hsa_status = hsa_queue_destroy(queue);
    HSA_CHECK(hsa_status);
}

void* hsaDeviceInterface::alloc_gpu_memory(int len) {
    void* ptr;
    hsa_status_t hsa_status = hsa_amd_memory_pool_allocate(global_region, len, 0, &ptr);
    HSA_CHECK(hsa_status);
    return ptr;
}

void hsaDeviceInterface::free_gpu_memory(void* ptr) {
    hsa_status_t hsa_status = hsa_amd_memory_pool_free(ptr);
    HSA_CHECK(hsa_status);
}


hsa_signal_t hsaDeviceInterface::async_copy_host_to_gpu(void* dst, void* src, int len,
                                                        hsa_signal_t precede_signal,
                                                        hsa_signal_t copy_signal) {
    hsa_status_t hsa_status;
    int num_precede_signals = 0;

    if (precede_signal.handle != 0)
        num_precede_signals = 1;

    //    hsa_signal_store_release(copy_signal, 1);
    while (0 < hsa_signal_cas_release(copy_signal, 0, 1))
        ;

    hsa_status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, src);
    HSA_CHECK(hsa_status);
    // hsa_status = hsa_amd_agents_allow_access(1, &cpu_agent, nullptr, dst);
    // HSA_CHECK(hsa_status);

    if (num_precede_signals > 0) {
        hsa_status = hsa_amd_memory_async_copy(dst, gpu_agent, src, cpu_agent, len,
                                               num_precede_signals, &precede_signal, copy_signal);
    } else {
        hsa_status =
            hsa_amd_memory_async_copy(dst, gpu_agent, src, cpu_agent, len, 0, nullptr, copy_signal);
    }
    HSA_CHECK(hsa_status);

    DEBUG("ASync host->gpu[{:d}] copy {:p} -> {:p}, len {:d}, precede_signal: {:d}, post_signal: "
          "{:d}",
          gpu_id, src, dst, len, precede_signal.handle, copy_signal.handle);

    return copy_signal;
}

hsa_signal_t hsaDeviceInterface::async_copy_gpu_to_host(void* dst, void* src, int len,
                                                        hsa_signal_t precede_signal,
                                                        hsa_signal_t copy_signal) {

    hsa_status_t hsa_status;
    int num_precede_signals = 0;

    if (precede_signal.handle != 0)
        num_precede_signals = 1;

    //    hsa_signal_store_release(copy_signal, 1);
    while (0 < hsa_signal_cas_release(copy_signal, 0, 1))
        ;

    // hsa_status = hsa_amd_agents_allow_access(1, &cpu_agent, nullptr, src);
    // HSA_CHECK(hsa_status);
    hsa_status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, dst);
    HSA_CHECK(hsa_status);

    if (num_precede_signals > 0) {
        hsa_status = hsa_amd_memory_async_copy(dst, cpu_agent, src, gpu_agent, len,
                                               num_precede_signals, &precede_signal, copy_signal);
    } else {
        hsa_status =
            hsa_amd_memory_async_copy(dst, cpu_agent, src, gpu_agent, len, 0, nullptr, copy_signal);
    }
    HSA_CHECK(hsa_status);

    DEBUG(
        "ASync gpu[{:d}]->host copy {:p} -> {:p}, len: {:d}, precede_signal {:d}, post_signal {:d}",
        gpu_id, src, dst, len, precede_signal.handle, copy_signal.handle);

    return copy_signal;
}

void hsaDeviceInterface::sync_copy_host_to_gpu(void* dst, void* src, int length) {
    hsa_signal_t sig;
    hsa_status_t hsa_status;

    DEBUG("Sync host->gpu[{:d}] copy {:p} -> {:p}, len: {:d}", gpu_id, src, dst, length);

    hsa_status = hsa_signal_create(1, 0, nullptr, &sig);
    HSA_CHECK(hsa_status);

    hsa_status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, src);
    HSA_CHECK(hsa_status);
    // hsa_status = hsa_amd_agents_allow_access(1, &cpu_agent, nullptr, dst);
    // HSA_CHECK(hsa_status);

    hsa_status = hsa_amd_memory_async_copy(dst, gpu_agent, src, cpu_agent, length, 0, nullptr, sig);
    HSA_CHECK(hsa_status);

    while (
        hsa_signal_wait_acquire(sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE))
        ;
    hsa_status = hsa_signal_destroy(sig);
    HSA_CHECK(hsa_status);
}

void hsaDeviceInterface::sync_copy_gpu_to_host(void* dst, void* src, int length) {
    hsa_signal_t sig;
    hsa_status_t hsa_status;

    DEBUG("Sync gpu[{:d}]->host copy {:p} -> {:p}, len: {:d}", gpu_id, src, dst, length);

    hsa_status = hsa_signal_create(1, 0, nullptr, &sig);
    HSA_CHECK(hsa_status);

    // hsa_status = hsa_amd_agents_allow_access(1, &cpu_agent, nullptr, src);
    // HSA_CHECK(hsa_status);
    hsa_status = hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, dst);
    HSA_CHECK(hsa_status);

    hsa_status = hsa_amd_memory_async_copy(dst, cpu_agent, src, gpu_agent, length, 0, nullptr, sig);
    HSA_CHECK(hsa_status);

    while (
        hsa_signal_wait_acquire(sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE))
        ;
    hsa_status = hsa_signal_destroy(sig);
    HSA_CHECK(hsa_status);
}

hsa_status_t hsaDeviceInterface::get_cpu_agent(hsa_agent_t agent, void* data) {
    if (data == nullptr) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    hsa_device_type_t hsa_device_type;
    hsa_status_t hsa_error_code =
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type);
    if (hsa_error_code != HSA_STATUS_SUCCESS) {
        return hsa_error_code;
    }

    if (hsa_device_type == HSA_DEVICE_TYPE_CPU) {
        *((hsa_agent_t*)data) = agent;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t hsaDeviceInterface::get_gpu_agent(hsa_agent_t agent, void* data) {
    hsa_device_type_t device_type;
    hsa_status_t status;
    int num;

    gpu_config_t* gpu_config = (gpu_config_t*)data;

    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    assert(status == HSA_STATUS_SUCCESS);
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &num);
    assert(status == HSA_STATUS_SUCCESS);
    if ((HSA_DEVICE_TYPE_GPU == device_type) && (gpu_config->gpu_id == (num - 1))) {
        uint32_t features = 0;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE, &features);
        if (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH) {
            hsa_queue_type_t queue_type;
            hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_TYPE, &queue_type);
            if (queue_type == HSA_QUEUE_TYPE_MULTI) {
                *gpu_config->agent = agent;
                return HSA_STATUS_INFO_BREAK;
            }
        }
    }
    return HSA_STATUS_SUCCESS;
}

hsa_status_t hsaDeviceInterface::get_kernarg_memory_region(hsa_region_t region, void* data) {
    hsa_region_segment_t segment;
    hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (HSA_REGION_SEGMENT_GLOBAL != segment) {
        return HSA_STATUS_SUCCESS;
    }

    hsa_region_global_flag_t flags;
    hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
        hsa_region_t* ret = (hsa_region_t*)data;
        *ret = region;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t hsaDeviceInterface::get_device_memory_region(hsa_amd_memory_pool_t region,
                                                          void* data) {
    hsa_amd_segment_t segment;
    hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if (HSA_AMD_SEGMENT_GLOBAL != segment) {
        return HSA_STATUS_SUCCESS;
    }

    hsa_amd_memory_pool_global_flag_t flags;
    hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)
        || (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)) {
        // INFO("Found device region, flags={:x}", flags);
        hsa_amd_memory_pool_t* ret = (hsa_amd_memory_pool_t*)data;
        *ret = region;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

hsa_agent_t hsaDeviceInterface::get_gpu_agent() {
    return gpu_agent;
}

hsa_region_t hsaDeviceInterface::get_kernarg_region() {
    return kernarg_region;
}

hsa_queue_t* hsaDeviceInterface::get_queue() {
    return queue;
}

uint64_t hsaDeviceInterface::get_hsa_timestamp_freq() {
    return timestamp_frequency_hz;
}

uint32_t hsaDeviceInterface::get_gpu_numa_node() {
    return numa_node;
}
