#include "hsaBase.h"

#include "hsa/hsa_ext_amd.h" // for hsa_amd_memory_pool_get_info, hsa_amd_memory_pool_t, hsa_am...

#include <sys/mman.h> // for mlock

#define MAX_NUMA 4

hsa_agent_t cpu_agent[MAX_NUMA];
hsa_amd_memory_pool_t host_region[MAX_NUMA];
uint32_t num_numa_nodes;

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

    uint32_t numa_node = 255;
    hsa_error_code = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &numa_node);
    if (hsa_error_code != HSA_STATUS_SUCCESS) {
        return hsa_error_code;
    }

    if (hsa_device_type == HSA_DEVICE_TYPE_CPU) {
        assert(numa_node < MAX_NUMA);
        cpu_agent[numa_node] = agent;
        *(uint32_t*)(data) += 1;
        assert(*(uint32_t*)(data) <= MAX_NUMA);
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
    if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)
        || (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)) {
        INFO_F("Found device region, flags=%x", flags);
        hsa_amd_memory_pool_t* ret = (hsa_amd_memory_pool_t*)data;
        *ret = region;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

void kotekan_hsa_start() {
    hsa_status_t hsa_status = hsa_init();
    HSA_CHECK(hsa_status);

    hsa_status = hsa_amd_profiling_async_copy_enable(1);
    HSA_CHECK(hsa_status);

    num_numa_nodes = 0;

    // Get the CPU agent
    hsa_status = hsa_iterate_agents(get_cpu_agent, &num_numa_nodes);
    HSA_CHECK(hsa_status);

    INFO_F("HSA Found: %d CPU memory agents (NUMA areas)", num_numa_nodes);
    assert(num_numa_nodes <= MAX_NUMA);
    // Get the CPU memory region
    for (uint32_t i = 0; i < num_numa_nodes; ++i) {
        hsa_status = hsa_amd_agent_iterate_memory_pools(cpu_agent[i], get_device_memory_region,
                                                        &host_region[i]);
        if (hsa_status == HSA_STATUS_INFO_BREAK) {
            hsa_status = HSA_STATUS_SUCCESS;
        }
        HSA_CHECK(hsa_status);
    }
}

void* hsa_host_malloc(size_t len, uint32_t numa_node) {
    void* ptr;

    assert(numa_node < MAX_NUMA);

    hsa_status_t hsa_status;
    hsa_status = hsa_amd_memory_pool_allocate(host_region[numa_node], len, 0, &ptr);
    HSA_CHECK(hsa_status);

    if (mlock(ptr, len) != 0) {
        ERROR_F("Error locking memory - check ulimit -a to check memlock limits");
        return NULL;
    }

    return ptr;
}

void hsa_host_free(void* ptr) {
    hsa_status_t hsa_status;
    hsa_status = hsa_amd_memory_pool_free(ptr);
    HSA_CHECK(hsa_status);
}

void kotekan_hsa_stop() {
    hsa_shut_down();
}
