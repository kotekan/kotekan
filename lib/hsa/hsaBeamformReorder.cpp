#include "hsaBeamformReorder.hpp"
#include "hsaBase.h"

hsaBeamformReorder::hsaBeamformReorder(const string& kernel_name, const string& kernel_file_name,
			    hsaDeviceInterface& device, Config& config,
			    bufferContainer& host_buffers,
			    const string &unique_name ) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name) {
    command_type = CommandType::KERNEL;
    apply_config(0);

    // Create a C style array for backwards compatiably.
    map_len = 512 * sizeof(int);
    _reorder_map_c = (int *)hsa_host_malloc(map_len);
    for (uint i=0;i<512;++i){
        _reorder_map_c[i] = _reorder_map[i];
    }
    void * device_map = device.get_gpu_memory("reorder_map", map_len);
    device.sync_copy_host_to_gpu(device_map, (void*)_reorder_map_c, map_len);
}

hsaBeamformReorder::~hsaBeamformReorder() {
    hsa_host_free(_reorder_map_c);
}

void hsaBeamformReorder::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);

    _num_elements = config.get_int(unique_name, "num_elements");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _reorder_map = config.get_int_array(unique_name, "reorder_map");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");

    input_frame_len  = _num_elements * _num_local_freq * _samples_per_data_set ;
    output_frame_len = _num_elements * _num_local_freq * _samples_per_data_set ;

}

hsa_signal_t hsaBeamformReorder::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    struct __attribute__ ((aligned(16))) args_t {
        void *input_buffer;
	void *map_buffer;
        void *output_buffer;
    } args;
    memset(&args, 0, sizeof(args));
    args.input_buffer = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.map_buffer = device.get_gpu_memory("reorder_map", map_len);
    args.output_buffer = device.get_gpu_memory_array("input", gpu_frame_id, output_frame_len);

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

