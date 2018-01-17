#include "hsaBeamformPulsarOneFeed.hpp"
#include "hsaBase.h"

hsaBeamformPulsarOneFeed::hsaBeamformPulsarOneFeed(const string& kernel_name, const string& kernel_file_name,
                            hsaDeviceInterface& device, Config& config,
                            bufferContainer& host_buffers,
                            const string &unique_name) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name) {

    apply_config(0);

    phase_len = _num_elements*_num_pulsar*2*sizeof(float);
    host_phase = (float *)hsa_host_malloc(phase_len);

    int index = 0;
    INFO("+++++ Pulsar beamforming with one feed (Feed %d)", _one_feed);
    for (int b=0; b < _num_pulsar; b++){
        for (int n=0; n<_num_elements; n++){
	  if ((n == _one_feed_p0) or (n == _one_feed_p1)) {
	        host_phase[index++] = 1;
		host_phase[index++] = 1;
	    }
	    else {
	        host_phase[index++] = 0;
		host_phase[index++] = 0;
	    }
	}
    }

    void * device_phase = device.get_gpu_memory("beamform_phase", phase_len);
    device.sync_copy_host_to_gpu(device_phase, (void *)host_phase, phase_len);
}

hsaBeamformPulsarOneFeed::~hsaBeamformPulsarOneFeed() {
    hsa_host_free(host_phase);
}

void hsaBeamformPulsarOneFeed::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);

    _num_elements = config.get_int(unique_name, "num_elements");
    _num_pulsar = config.get_int(unique_name, "num_pulsar");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _num_pol = config.get_int(unique_name, "num_pol");
    _one_feed_p0 = config.get_int(unique_name, "psr-test_one_feed_p0");
    _one_feed_p1 = config.get_int(unique_name, "psr-test_one_feed_p1");

    input_frame_len = _num_elements * _samples_per_data_set;
    output_frame_len =  _samples_per_data_set * _num_pulsar * _num_pol *  sizeof(uint8_t);
}

hsa_signal_t hsaBeamformPulsarOneFeed::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {
  //INFO("##[BeamformPulsar]###  gpu_frame_id=%d, input_frame_len=%lu ", gpu_frame_id, input_frame_len);


    struct __attribute__ ((aligned(16))) args_t {
        void *input_buffer;
        void *phase_buffer;
        void *output_buffer;
    } args;
    memset(&args, 0, sizeof(args));
    args.input_buffer = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.phase_buffer = device.get_gpu_memory("beamform_phase", phase_len);
    args.output_buffer = device.get_gpu_memory_array("bf_output", gpu_frame_id, output_frame_len);

    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));


    kernelParams params;
    params.workgroup_size_x = 256;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = 512;
    params.grid_size_y = _num_pulsar;
    params.grid_size_z = _samples_per_data_set/32;
    params.num_dims = 3;

    params.private_segment_size = 0;
    params.group_segment_size = 2048;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}

