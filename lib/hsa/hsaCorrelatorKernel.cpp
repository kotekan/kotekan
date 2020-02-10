#include "hsaCorrelatorKernel.hpp"

#include "Config.hpp"             // for Config
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaBase.h"              // for hsa_host_free, hsa_host_malloc
#include "hsaCommand.hpp"         // for kernelParams, REGISTER_HSA_COMMAND, _factory_aliashsaC...
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config
#include "kotekanLogging.hpp"     // for DEBUG2

#include "fmt.hpp" // for format, fmt

#include <cstdint>   // for int32_t
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <string.h>  // for memcpy, memset
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaCorrelatorKernel);

hsaCorrelatorKernel::hsaCorrelatorKernel(Config& config, const std::string& unique_name,
                                         bufferContainer& host_buffers,
                                         hsaDeviceInterface& device) :
    hsaSubframeCommand(config, unique_name, host_buffers, device, "CHIME_X", "N2.hsaco") {
    command_type = gpuCommandType::KERNEL;


    // N_INTG is the number summed in each workitem
    // if there are more, they get split across multiple workitems
    // time slice id is identified by the Y group.
    _n_intg = config.get<int32_t>(unique_name, "n_intg");

    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    int block_size = config.get<int>(unique_name, "block_size");
    _num_blocks = (int32_t)(_num_elements / block_size) * (_num_elements / block_size + 1) / 2.;
    input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    presum_len = _num_elements * _num_local_freq * 2 * sizeof(int32_t);
    // I don't really like this way of getting to correlator output size (AR)
    corr_frame_len = _num_blocks * block_size * block_size * 2 * sizeof(int32_t);
    block_map_len = _num_blocks * 2 * sizeof(uint32_t);


    // Allocate and copy the block map
    host_block_map = (uint32_t*)hsa_host_malloc(block_map_len, device.get_gpu_numa_node());
    int block_id = 0;
    for (int y = 0; block_id < _num_blocks; y++) {
        for (int x = y; x < _num_elements / block_size; x++) {
            host_block_map[2 * block_id + 0] = x;
            host_block_map[2 * block_id + 1] = y;
            block_id++;
        }
    }

    // Copy it to the GPU
    void* device_block_map = device.get_gpu_memory("block_map", block_map_len);
    device.sync_copy_host_to_gpu(device_block_map, host_block_map, block_map_len);

    // Create the extra kernel args object.
    host_kernel_args = (corr_kernel_config_t*)hsa_host_malloc(sizeof(corr_kernel_config_t),
                                                              device.get_gpu_numa_node());
    host_kernel_args->n_elem = _num_elements;
    host_kernel_args->n_intg = _n_intg;
    host_kernel_args->n_iter = _sub_frame_samples;
    host_kernel_args->n_blk = _num_blocks;

    void* device_kernel_args =
        device.get_gpu_memory("corr_kernel_config", sizeof(corr_kernel_config_t));
    device.sync_copy_host_to_gpu(device_kernel_args, host_kernel_args,
                                 sizeof(corr_kernel_config_t));

    // pre-allocate GPU memory
    device.get_gpu_memory_array("input", 0, input_frame_len);
    device.get_gpu_memory_array("presum", 0, presum_len);
    device.get_gpu_memory_array("corr", 0, corr_frame_len);
    device.get_gpu_memory("block_map", block_map_len);
    device.get_gpu_memory("corr_kernel_config", sizeof(corr_kernel_config_t));
}

hsaCorrelatorKernel::~hsaCorrelatorKernel() {
    hsa_host_free((void*)host_block_map);
    hsa_host_free((void*)host_kernel_args);
}

hsa_signal_t hsaCorrelatorKernel::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Unused parameter, suppress warning
    (void)precede_signal;

    struct __attribute__((aligned(16))) args_t {
        void* input_buffer;
        void* presum_buffer;
        void* corr_buffer;
        void* blk_map;
        void* config;
    } args;
    memset(&args, 0, sizeof(args));
    // Index into the sub frame.
    args.input_buffer =
        (void*)((uint8_t*)device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len)
                + _num_elements * _num_local_freq * _sub_frame_samples * _sub_frame_index);
    args.presum_buffer = device.get_gpu_memory_array(
        fmt::format(fmt("presum_{:d}"), _sub_frame_index), gpu_frame_id, presum_len);
    args.corr_buffer = device.get_gpu_memory_array(fmt::format(fmt("corr_{:d}"), _sub_frame_index),
                                                   gpu_frame_id, corr_frame_len);
    args.blk_map = device.get_gpu_memory("block_map", block_map_len);
    args.config = device.get_gpu_memory("corr_kernel_config", sizeof(corr_kernel_config_t));
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    DEBUG2("correlatorKernel: gpu[{:d}][{:d}], input_buffer: {:p}, presum_buffer: {:p}, "
           "corr_buffer: {:p}, "
           "blk_map: {:p}, config: {:p}, sizeof(args) = {:d}, kernels_args[{:d}] = {:p}",
           device.get_gpu_id(), gpu_frame_id, args.input_buffer, args.presum_buffer,
           args.corr_buffer, args.blk_map, args.config, (int)sizeof(args), gpu_frame_id,
           kernel_args[gpu_frame_id]);

    DEBUG2("correlatorKernel: gpu[{:d}][{:d}], wgx {:d}, wgy {:d}, wgz {:d}, gsx {:d}, gsy {:d}, "
           "gsz {:d}",
           device.get_gpu_id(), gpu_frame_id, 16, 4, 1, 16, 4 * _sub_frame_samples / _n_intg,
           _num_blocks);

    // Set kernel dims
    kernelParams params;
    params.workgroup_size_x = 16;
    params.workgroup_size_y = 4;
    params.workgroup_size_z = 1;
    params.grid_size_x = 16;
    params.grid_size_y = 4 * _sub_frame_samples / _n_intg;
    params.grid_size_z = _num_blocks;
    params.num_dims = 3;

    params.private_segment_size = 0;
    params.group_segment_size = 3136;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}
