#include "hsaBeamformKernel.hpp"
#include "hsaBase.h"

hsaBeamformKernel::hsaBeamformKernel(const string& kernel_name, const string& kernel_file_name,
                            hsaDeviceInterface& device, Config& config,
                            bufferContainer& host_buffers,
                            const string &unique_name) :
    hsaCommand(kernel_name, kernel_file_name, device, config, host_buffers, unique_name) {

    apply_config(0);

    map_len = 256 * sizeof(int);
    host_map = (uint32_t *)hsa_host_malloc(map_len);

    float t, delta_t, beam_ref;
    int cl_index;
    float D2R = PI/180.;
    int pad = 2 ;

    for (int b = 0; b < 256; ++b){
        beam_ref = asin(LIGHT_SPEED*(b-256/2.) / (FREQ_REF*1.e6) / (256) /FEED_SEP) * 180./ PI;
        t = 256*pad*(FREQ_REF*1.e6)*(FEED_SEP/LIGHT_SPEED*sin(beam_ref*D2R)) + 0.5;
        delta_t = 256*pad*(FREQ1*1e6-FREQ_REF*1e6) * (FEED_SEP/LIGHT_SPEED*sin(beam_ref*D2R));
        cl_index = (int) floor(t + delta_t) + 256*pad/2.;

        if (cl_index < 0)
            cl_index = 256*pad + cl_index;
        else if (cl_index > 256*pad)
            cl_index = cl_index - 256*pad;

        cl_index = cl_index - 256;
        if (cl_index < 0){
            cl_index = 256*pad + cl_index;
        }
        host_map[b] = cl_index;
    }

    void * device_map = device.get_gpu_memory("beamform_map", map_len);
    device.sync_copy_host_to_gpu(device_map, (void *)host_map, map_len);

    coeff_len = 32*sizeof(float);
    host_coeff = (float *)hsa_host_malloc(coeff_len);

    for (int angle_iter=0; angle_iter < 4; angle_iter++){
        float anglefrac = sin(0.4*angle_iter*PI/180.);   //say 0, 0.4, 0.8 and 1.2 for now.
        for (int cylinder=0; cylinder < 4; cylinder++) {
            host_coeff[angle_iter*4*2 + cylinder*2] = cos(2*PI*anglefrac*cylinder*22*FREQ1*1.e6/LIGHT_SPEED);
            host_coeff[angle_iter*4*2 + cylinder*2 + 1] = sin(2*PI*anglefrac*cylinder*22*FREQ1*1.e6/LIGHT_SPEED);
        }
    }

    void * device_coeff_map = device.get_gpu_memory("beamform_coeff_map", coeff_len);
    device.sync_copy_host_to_gpu(device_coeff_map, (void*)host_coeff, coeff_len);
}

hsaBeamformKernel::~hsaBeamformKernel() {
    hsa_host_free(host_map);
    hsa_host_free(host_coeff);
    // TODO Free device memory allocations.
}

void hsaBeamformKernel::apply_config(const uint64_t& fpga_seq) {
    hsaCommand::apply_config(fpga_seq);

    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    output_frame_len = _num_elements * _samples_per_data_set * 2 * sizeof(float);
}

hsa_signal_t hsaBeamformKernel::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    struct __attribute__ ((aligned(16))) args_t {
        void *input_buffer;
        void *map_buffer;
        void *coeff_buffer;
        void *output_buffer;
    } args;
    memset(&args, 0, sizeof(args));
    args.input_buffer = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.map_buffer = device.get_gpu_memory("beamform_map", map_len);
    args.coeff_buffer = device.get_gpu_memory("beamform_coeff_map", coeff_len);
    args.output_buffer = device.get_gpu_memory_array("beamform_output", gpu_frame_id, output_frame_len);
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    hsa_status_t hsa_status = hsa_signal_create(1, 0, NULL, &signals[gpu_frame_id]);
    assert(hsa_status == HSA_STATUS_SUCCESS);

    // Obtain the current queue write index.
    uint64_t index = hsa_queue_load_write_index_acquire(device.get_queue());
    hsa_kernel_dispatch_packet_t* dispatch_packet = (hsa_kernel_dispatch_packet_t*)device.get_queue()->base_address +
                                                            (index % device.get_queue()->size);
    dispatch_packet->setup  |= 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    dispatch_packet->workgroup_size_x = (uint32_t)256;
    dispatch_packet->workgroup_size_y = (uint16_t)1;
    dispatch_packet->workgroup_size_z = (uint16_t)1;
    dispatch_packet->grid_size_x = (uint32_t)256;
    dispatch_packet->grid_size_y = (uint16_t)2;
    dispatch_packet->grid_size_z = (uint32_t)_samples_per_data_set;
    dispatch_packet->completion_signal = signals[gpu_frame_id];
    dispatch_packet->kernel_object = kernel_object;
    dispatch_packet->kernarg_address = (void*) kernel_args[gpu_frame_id];
    dispatch_packet->private_segment_size = 0;
    dispatch_packet->group_segment_size = (uint32_t)16384; //Not sure if I need that
    dispatch_packet-> header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

    hsa_queue_add_write_index_acquire(device.get_queue(), 1);
    hsa_signal_store_relaxed(device.get_queue()->doorbell_signal, index);

    return signals[gpu_frame_id];
}

