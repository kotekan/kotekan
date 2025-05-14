#include "hsaBeamformReorder.hpp"

#include "Config.hpp"             // for Config
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaBase.h"              // for hsa_host_free, hsa_host_malloc
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config

#include <cstdint>     // for int32_t
#include <exception>   // for exception
#include <regex>       // for match_results<>::_Base_type
#include <string.h>    // for memcpy, memset
#include <sys/types.h> // for uint

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaBeamformReorder);

hsaBeamformReorder::hsaBeamformReorder(Config& config, const std::string& unique_name,
                                       bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "reorder" KERNEL_EXT, "reorder.hsaco") {
    command_type = gpuCommandType::KERNEL;

    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _reorder_map = config.get<std::vector<int32_t>>(unique_name, "reorder_map");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");

    input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    output_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;

    // Create a C style array for backwards compatibility.
    map_len = 512 * sizeof(int);
    _reorder_map_c = (int*)hsa_host_malloc(map_len, device.get_gpu_numa_node());
    for (uint i = 0; i < 512; ++i) {
        _reorder_map_c[i] = _reorder_map[i];
    }
    void* device_map = device.get_gpu_memory("reorder_map", map_len);
    device.sync_copy_host_to_gpu(device_map, (void*)_reorder_map_c, map_len);
}

hsaBeamformReorder::~hsaBeamformReorder() {
    hsa_host_free(_reorder_map_c);
}

hsa_signal_t hsaBeamformReorder::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    struct __attribute__((aligned(16))) args_t {
        void* input_buffer;
        void* map_buffer;
        void* output_buffer;
    } args;
    memset(&args, 0, sizeof(args));
    args.input_buffer =
        device.get_gpu_memory_array("input", gpu_frame_id, _gpu_buffer_depth, input_frame_len);
    args.map_buffer = device.get_gpu_memory("reorder_map", map_len);
    args.output_buffer = device.get_gpu_memory("input_reordered", output_frame_len);

    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    kernelParams params;
    params.workgroup_size_x = 256;
    params.workgroup_size_y = 1;
    params.grid_size_x = 256;
    params.grid_size_y = _samples_per_data_set;
    params.num_dims = 2;

    params.private_segment_size = 0;
    params.group_segment_size = 8192;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}
