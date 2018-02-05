#include "hsaBeamformKernel.hpp"
#include "hsaBase.h"
#include <unistd.h>

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
        float anglefrac = sin(0.1*angle_iter*PI/180.);   //say 0, 0.1, 0.2, 0.3
        for (int cylinder=0; cylinder < 4; cylinder++) {
            host_coeff[angle_iter*4*2 + cylinder*2] = cos(2*PI*anglefrac*cylinder*22*FREQ1*1.e6/LIGHT_SPEED);
            host_coeff[angle_iter*4*2 + cylinder*2 + 1] = sin(2*PI*anglefrac*cylinder*22*FREQ1*1.e6/LIGHT_SPEED);
        }
    }

    void * device_coeff_map = device.get_gpu_memory("beamform_coeff_map", coeff_len);
    device.sync_copy_host_to_gpu(device_coeff_map, (void*)host_coeff, coeff_len);

    //Figure out which frequency, is there a better way that doesn't involve reading in the whole thing? Check later
    metadata_buf = host_buffers.get_buffer("network_buf");

    gain_len = 2*2048*sizeof(float);
    host_gain = (float *)hsa_host_malloc(gain_len);

}

hsaBeamformKernel::~hsaBeamformKernel() {
    hsa_host_free(host_map);
    hsa_host_free(host_coeff);
    hsa_host_free(host_gain);
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

int hsaBeamformKernel::wait_on_precondition(int gpu_frame_id)
{
  uint8_t * frame = wait_for_full_frame(metadata_buf, unique_name.c_str(), 0);
  if (frame == NULL) return -1;
}

hsa_signal_t hsaBeamformKernel::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {

    stream_id_t stream_id = get_stream_id_t(metadata_buf, 0);
    freq_now = bin_number_chime(&stream_id); 

    FILE *ptr_myfile;
    char filename[sizeof "/root/FRB-GainFiles/quick_gains_0000_reordered.bin"];
    sprintf(filename, "/root/FRB-GainFiles/quick_gains_%04d_reordered.bin",freq_now);
    if( access( filename, F_OK ) == -1 ) {
      // file doesn't exists (since some freq are missing), for those freq we just read in 1+0j
      sprintf(filename, "/root/FRB-GainFiles/dummy.bin");
    }

    ptr_myfile=fopen(filename,"rb");
    fread(host_gain,sizeof(float)*2*2048,1,ptr_myfile);
    fclose(ptr_myfile);
    INFO("!!!!!!---------GPU got gains gains[0]=(%.2f %.2f) gains[2047]=(%.2f %.2f)", host_gain[0], host_gain[1], host_gain[4094], host_gain[4095])
    void * device_gain = device.get_gpu_memory("beamform_gain", gain_len);
    device.sync_copy_host_to_gpu(device_gain, (void*)host_gain, gain_len);



    struct __attribute__ ((aligned(16))) args_t {
        void *input_buffer;
        void *map_buffer;
        void *coeff_buffer;
        void *output_buffer;
	void *gain_buffer;
    } args;
    memset(&args, 0, sizeof(args));
    args.input_buffer = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    args.map_buffer = device.get_gpu_memory("beamform_map", map_len);
    args.coeff_buffer = device.get_gpu_memory("beamform_coeff_map", coeff_len);
    args.output_buffer = device.get_gpu_memory("beamform_output", output_frame_len);
    args.gain_buffer = device.get_gpu_memory("beamform_gain", gain_len);
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));


    kernelParams params;
    params.workgroup_size_x = 256;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = 256;
    params.grid_size_y = 2;
    params.grid_size_z = _samples_per_data_set;
    params.num_dims = 3;

    params.private_segment_size = 0;
    params.group_segment_size = 16384;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}

