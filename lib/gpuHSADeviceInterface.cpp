#include "gpuHSADeviceInterface.hpp"
#include "gpu_command.h"
#include "callbackdata.h"
#include "math.h"
#include <errno.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_finalize.h"
#include "hsa/hsa_ext_amd.h"

gpuHSADeviceInterface::gpuHSADeviceInterface(Config& config_, int gpu_id_) :
    config(config_), gpu_id(gpu_id_){

    hsa_status_t hsa_status;

    // Function parameters
    gpu_config_t gpu_config;
    gpu_config.agent = &gpu_agent;
    gpu_config.gpu_id = gpu_id;

    gpu_mem_config_t gpu_mem_config;
    gpu_mem_config.region = &global_region;
    gpu_mem_config.gpu_id = gpu_id;

    // Get the CPU agent
    hsa_status = hsa_iterate_agents(get_cpu_agent, &cpu_agent);
    if(hsa_status == HSA_STATUS_INFO_BREAK)
        hsa_status = HSA_STATUS_SUCCESS;
    assert(hsa_status == HSA_STATUS_SUCCESS);
    // Get the CPU memory region.
    hsa_amd_agent_iterate_memory_pools(cpu_agent,
            get_device_memory_region, &host_region);

    // Get GPU agent
    hsa_status = hsa_iterate_agents(get_gpu_agent, &gpu_config);
    if(hsa_status == HSA_STATUS_INFO_BREAK)
        hsa_status = HSA_STATUS_SUCCESS;
    assert(hsa_status == HSA_STATUS_SUCCESS);

    // Get GPU agent name and number
    hsa_status = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, agent_name);
    assert(hsa_status == HSA_STATUS_SUCCESS);
    int num;
    hsa_status = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NODE, &num);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    INFO("Initializing HSA GPU type %s at index %i.", agent_name, num-1);

    global_region.handle = (uint64_t)-1;
    hsa_amd_agent_iterate_memory_pools(gpu_agent, get_device_memory_region, &global_region);
    hsa_status = (global_region.handle == (uint64_t)-1) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
    assert(hsa_status == HSA_STATUS_SUCCESS);

    // Find a memory region that supports kernel arguments
    kernarg_region.handle=(uint64_t)-1;
    hsa_agent_iterate_regions(gpu_agent, get_kernarg_memory_region, &kernarg_region);
    hsa_status = (kernarg_region.handle == (uint64_t)-1) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
    assert(hsa_status == HSA_STATUS_SUCCESS);

    // Query the maximum size of the queue.
    // Create a queue using the maximum size.
    uint32_t queue_size = 0;
    hsa_status = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    assert(hsa_status == HSA_STATUS_SUCCESS);
    hsa_status = hsa_queue_create(gpu_agent, queue_size, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    _gpu_buffer_depth = config.get_int("/gpu/buffer_depth");

}

gpuHSADeviceInterface::~gpuHSADeviceInterface() {
    hsa_status_t hsa_status = hsa_queue_destroy(queue);
    assert(hsa_status == HSA_STATUS_SUCCESS);
}

hsa_signal_t gpuHSADeviceInterface::async_copy_host_to_gpu(void* dst, void* src, int len, hsa_signal_t precede_signal) {
    hsa_signal_t post_sig;
    hsa_status_t hsa_status;
    int num_precede_signals = 0;

    INFO("ASync host->gpu[%d] copy %p -> %p", gpu_id, src, dst);

    if (precede_signal.handle != 0)
        num_precede_signals = 1;

    hsa_status = hsa_signal_create(1, 0, NULL, &post_sig);
    assert(hsa_status == HSA_STATUS_SUCCESS);


    hsa_status = hsa_amd_agents_allow_access(1, &gpu_agent, NULL, src);
    assert(hsa_status == HSA_STATUS_SUCCESS);
    //hsa_status = hsa_amd_agents_allow_access(1, &cpu_agent, NULL, dst);
    //assert(hsa_status == HSA_STATUS_SUCCESS);

    if (num_precede_signals > 0) {
        hsa_status = hsa_amd_memory_async_copy(dst, gpu_agent,
                                               src, cpu_agent,
                                               len, num_precede_signals,
                                               &precede_signal, post_sig);
    } else {
        hsa_status = hsa_amd_memory_async_copy(dst, gpu_agent,
                                               src, cpu_agent,
                                               len, 0,
                                               NULL, post_sig);
    }
    assert(hsa_status == HSA_STATUS_SUCCESS);

    return post_sig;
}

hsa_signal_t gpuHSADeviceInterface::async_copy_gpu_to_host(void* dst, void* src, int len, hsa_signal_t precede_signal) {
    hsa_signal_t post_sig;
    hsa_status_t hsa_status;
    int num_precede_signals = 0;

    INFO("ASync gpu[%d]->host copy %p -> %p", gpu_id, src, dst);

    if (precede_signal.handle != 0)
        num_precede_signals = 1;

    hsa_status = hsa_signal_create(1, 0, NULL, &post_sig);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    //hsa_status = hsa_amd_agents_allow_access(1, &cpu_agent, NULL, src);
    //assert(hsa_status == HSA_STATUS_SUCCESS);
    hsa_status = hsa_amd_agents_allow_access(1, &gpu_agent, NULL, dst);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    if (num_precede_signals > 0) {
        hsa_status = hsa_amd_memory_async_copy(dst, cpu_agent,
                                               src, gpu_agent,
                                               len, num_precede_signals,
                                               &precede_signal, post_sig);
    } else {
        hsa_status = hsa_amd_memory_async_copy(dst, cpu_agent,
                                               src, gpu_agent,
                                               len, 0,
                                               NULL, post_sig);
    }
    assert(hsa_status == HSA_STATUS_SUCCESS);

    return post_sig;
}

void gpuHSADeviceInterface::sync_copy_host_to_gpu(void *dst, void *src, int length) {
    hsa_signal_t sig;
    hsa_status_t hsa_status;

    INFO("Sync host->gpu[%d] copy %p -> %p", gpu_id, src, dst);

    hsa_status = hsa_signal_create(1, 0, NULL, &sig);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    hsa_status = hsa_amd_agents_allow_access(1, &gpu_agent, NULL, src);
    assert(hsa_status == HSA_STATUS_SUCCESS);
    //hsa_status = hsa_amd_agents_allow_access(1, &cpu_agent, NULL, dst);
    //assert(hsa_status == HSA_STATUS_SUCCESS);

    hsa_status = hsa_amd_memory_async_copy(dst, gpu_agent,
                                           src, cpu_agent,
                                           length, 0, NULL, sig);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    while (hsa_signal_wait_acquire(sig, HSA_SIGNAL_CONDITION_LT, 1,
                                   UINT64_MAX, HSA_WAIT_STATE_ACTIVE));
    hsa_status = hsa_signal_destroy(sig);
    assert(hsa_status == HSA_STATUS_SUCCESS);
}

void gpuHSADeviceInterface::sync_copy_gpu_to_host(void *dst, void *src, int length) {
    hsa_signal_t sig;
    hsa_status_t hsa_status;

    INFO("Sync gpu[%d]->host copy %p -> %p", gpu_id, src, dst);

    hsa_status = hsa_signal_create(1, 0, NULL, &sig);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    //hsa_status = hsa_amd_agents_allow_access(1, &cpu_agent, NULL, src);
    //assert(hsa_status == HSA_STATUS_SUCCESS);
    hsa_status = hsa_amd_agents_allow_access(1, &gpu_agent, NULL, dst);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    hsa_status = hsa_amd_memory_async_copy(dst, cpu_agent,
                                           src, gpu_agent,
                                           length, 0, NULL, sig);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    while (hsa_signal_wait_acquire(sig, HSA_SIGNAL_CONDITION_LT, 1,
                                   UINT64_MAX, HSA_WAIT_STATE_ACTIVE));
    hsa_status = hsa_signal_destroy(sig);
    assert(hsa_status == HSA_STATUS_SUCCESS);
}

hsa_status_t gpuHSADeviceInterface::get_cpu_agent(hsa_agent_t agent, void* data) {
    if (data == NULL) {
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

hsa_status_t gpuHSADeviceInterface::get_gpu_agent(hsa_agent_t agent, void* data) {
    hsa_device_type_t device_type;
    hsa_status_t status;
    int num;

    gpu_config_t *gpu_config = (gpu_config_t*)data;

    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    assert(status == HSA_STATUS_SUCCESS);
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &num);
    assert(status == HSA_STATUS_SUCCESS);
    if ((HSA_DEVICE_TYPE_GPU == device_type) &&
        (gpu_config->gpu_id == (num-1)))
    {
        *gpu_config->agent = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

hsa_status_t gpuHSADeviceInterface::get_kernarg_memory_region(hsa_region_t region, void* data) {
    hsa_region_segment_t segment;
    hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (HSA_REGION_SEGMENT_GLOBAL != segment) {
        return HSA_STATUS_SUCCESS;
    }

    hsa_region_global_flag_t flags;
    hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
        hsa_region_t* ret = (hsa_region_t*) data;
        *ret = region;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t gpuHSADeviceInterface::get_device_memory_region(hsa_amd_memory_pool_t region, void* data) {
    hsa_amd_segment_t segment;
    hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if (HSA_AMD_SEGMENT_GLOBAL != segment) {
        return HSA_STATUS_SUCCESS;
    }

    hsa_amd_memory_pool_global_flag_t flags;
    hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) ||
        (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED))
    {
        INFO("Found device region, flags=%x", flags);
        hsa_amd_memory_pool_t* ret = (hsa_amd_memory_pool_t*) data;
        *ret = region;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

void* gpuHSADeviceInterface::get_gpu_memory_array(const string& name, const int index, const int len) {
    assert(index < _gpu_buffer_depth);
    hsa_status_t hsa_status;
    // Check if the memory isn't yet allocated
    if (gpu_memory.count(name) == 0) {
        for (int i = 0; i < _gpu_buffer_depth; ++i) {
            void * ptr;
            hsa_status=hsa_amd_memory_pool_allocate(global_region, len, 0, &ptr);
            assert(hsa_status == HSA_STATUS_SUCCESS);
            gpu_memory[name].len = len;
            gpu_memory[name].gpu_pointers.push_back(ptr);
        }
    }

    // The size must match what has already been allocated.
    assert(len == gpu_memory[name].len);
    // Make sure we aren't asking for an index past the end of the array.
    assert(index < gpu_memory[name].gpu_pointers.size());

    // Return the requested memory.
    return gpu_memory[name].gpu_pointers[index];
}

void* gpuHSADeviceInterface::get_gpu_memory(const string& name, const int len) {
    hsa_status_t hsa_status;
    // Check if the memory isn't yet allocated
    if (gpu_memory.count(name) == 0) {
        void * ptr;
        hsa_status=hsa_amd_memory_pool_allocate(global_region, len, 0, &ptr);
        assert(hsa_status == HSA_STATUS_SUCCESS);
        gpu_memory[name].len = len;
        gpu_memory[name].gpu_pointers.push_back(ptr);
    }

    // The size must match what has already been allocated.
    assert(len == gpu_memory[name].len);
    assert(gpu_memory[name].gpu_pointers.size() == 1);

    // Return the requested memory.
    return gpu_memory[name].gpu_pointers[0];
}

int gpuHSADeviceInterface::get_gpu_id() {
    return gpu_id;
}
int gpuHSADeviceInterface::get_gpu_buffer_depth() {
    return _gpu_buffer_depth;
}

hsa_agent_t gpuHSADeviceInterface::get_gpu_agent() {
    return gpu_agent;
}

hsa_region_t gpuHSADeviceInterface::get_kernarg_region() {
    return kernarg_region;
}

hsa_queue_t* gpuHSADeviceInterface::get_queue() {
    return queue;
}

gpuMemoryBlock::~gpuMemoryBlock()  {
    for (auto&& gpu_pointer : gpu_pointers) {
        hsa_amd_memory_pool_free(gpu_pointer);
    }
    // TODO delete queue
}
