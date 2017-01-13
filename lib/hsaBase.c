#include "hsaBase.h"

hsa_agent_t cpu_agent;
hsa_amd_memory_pool_t host_region;

// Internal healer function.
static hsa_status_t get_cpu_agent(hsa_agent_t agent, void* data) {
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

// Internal helper function
static hsa_status_t get_device_memory_region(hsa_amd_memory_pool_t region, void* data) {
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

void kotekan_hsa_start() {
    hsa_status_t hsa_status = hsa_init();
    assert(hsa_status == HSA_STATUS_SUCCESS);

    // Get the CPU agent
    hsa_status = hsa_iterate_agents(get_cpu_agent, &cpu_agent);
    if(hsa_status == HSA_STATUS_INFO_BREAK) {
        hsa_status = HSA_STATUS_SUCCESS;
    }
    assert(hsa_status == HSA_STATUS_SUCCESS);

    // Get the CPU memory region
    hsa_status = hsa_amd_agent_iterate_memory_pools(cpu_agent, get_device_memory_region, &host_region);
    if (hsa_status == HSA_STATUS_INFO_BREAK) {
        hsa_status = HSA_STATUS_SUCCESS;
    }
    assert(hsa_status == HSA_STATUS_SUCCESS);
}

void * hsa_host_malloc(size_t len) {
    void * ptr;

    hsa_status_t hsa_status;
    hsa_status = hsa_amd_memory_pool_allocate(host_region, len, 0, &ptr);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    return ptr;
}

void hsa_host_free(void *ptr) {
    hsa_status_t hsa_status;
    hsa_status = hsa_amd_memory_pool_free(ptr);
    assert(hsa_status == HSA_STATUS_SUCCESS);
}

void kotekan_hsa_stop() {
    hsa_shut_down();
}

