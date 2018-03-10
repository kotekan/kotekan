#include "hsaBeamformKernel.hpp"

REGISTER_HSA_COMMAND(hsaBeamformKernel);

hsaBeamformKernel::hsaBeamformKernel(Config& config, const string &unique_name,
                            bufferContainer& host_buffers,
                            hsaDeviceInterface& device) :
    hsaCommand("zero_padded_FFT512","unpack_shift_beamform_flip.hsaco", config, unique_name, host_buffers, device) {
    command_type = CommandType::KERNEL;

    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _gain_dir = config.get_string(unique_name, "gain_dir");

    scaling = config.get_float_default(unique_name, "frb_scaling", 1.0);
    vector<float> dg = {0.0,0.0}; //re,im
    default_gains = config.get_float_array_default(unique_name,"frb_missing_gains",dg);

    input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    output_frame_len = _num_elements * _samples_per_data_set * 2 * sizeof(float);


    map_len = 256 * sizeof(int);
    host_map = (uint32_t *)hsa_host_malloc(map_len);

    coeff_len = 32*sizeof(float);
    host_coeff = (float *)hsa_host_malloc(coeff_len);

    gain_len = 2*2048*sizeof(float);
    host_gain = (float *)hsa_host_malloc(gain_len);

    //Figure out which frequency, is there a better way that doesn't involve reading in the whole thing? Check later
    metadata_buf = host_buffers.get_buffer("network_buf");
    metadata_buffer_id = 0;
    metadata_buffer_precondition_id = 0;
    freq_now = 0;

    first_pass=true;
}

hsaBeamformKernel::~hsaBeamformKernel() {
    hsa_host_free(host_map);
    hsa_host_free(host_coeff);
    hsa_host_free(host_gain);
    // TODO Free device memory allocations.
}

int hsaBeamformKernel::wait_on_precondition(int gpu_frame_id) {
    uint8_t * frame = wait_for_full_frame(metadata_buf, unique_name.c_str(), metadata_buffer_precondition_id);
    if (frame == NULL) return -1;
    metadata_buffer_precondition_id = (metadata_buffer_precondition_id + 1) % metadata_buf->num_frames;
    return 0;
}


void hsaBeamformKernel::calculate_cl_index(uint32_t *host_map, float FREQ1, float *host_coeff) {
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

    for (int angle_iter=0; angle_iter < 4; angle_iter++){
        float anglefrac = sin(0.5*angle_iter*PI/180.);   //EW beam separation 0.5 deg
        for (int cylinder=0; cylinder < 4; cylinder++) {
            host_coeff[angle_iter*4*2 + cylinder*2] = cos(2*PI*anglefrac*cylinder*22*FREQ1*1.e6/LIGHT_SPEED);
            host_coeff[angle_iter*4*2 + cylinder*2 + 1] = sin(2*PI*anglefrac*cylinder*22*FREQ1*1.e6/LIGHT_SPEED);
        }
    }
}


hsa_signal_t hsaBeamformKernel::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {
    if (first_pass){
        stream_id_t stream_id = get_stream_id_t(metadata_buf, metadata_buffer_id);
        freq_now = bin_number_chime(&stream_id);
        float freq_MHz = freq_from_bin(freq_now);
        FILE *ptr_myfile;
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/quick_gains_%04d_reordered.bin",_gain_dir.c_str(),freq_now);
        ptr_myfile=fopen(filename,"rb");
        if (ptr_myfile == NULL) {
            ERROR("GPU Cannot open gain file %s", filename);
            for (int i=0;i<2048;i++){
                host_gain[i*2]   = default_gains[0] * scaling;
                host_gain[i*2+1] = default_gains[1] * scaling;
            }
        }
        else {
            if (_num_elements != fread(host_gain,sizeof(float)*2,_num_elements,ptr_myfile)) {
                ERROR("Gain file (%s) wasn't long enough! Something went wrong, breaking...", filename);
            }
            fclose(ptr_myfile);
            for (uint32_t i=0; i<2048; i++){
                host_gain[i*2  ] = host_gain[i*2  ] * scaling;
                host_gain[i*2+1] = host_gain[i*2+1] * scaling;
            }
        }
        void * device_gain = device.get_gpu_memory("beamform_gain", gain_len);
        device.sync_copy_host_to_gpu(device_gain, (void*)host_gain, gain_len);

        calculate_cl_index(host_map, freq_MHz, host_coeff);
        void * device_map = device.get_gpu_memory("beamform_map", map_len);
        device.sync_copy_host_to_gpu(device_map, (void *)host_map, map_len);

        void * device_coeff_map = device.get_gpu_memory("beamform_coeff_map", coeff_len);
        device.sync_copy_host_to_gpu(device_coeff_map, (void*)host_coeff, coeff_len);

        metadata_buffer_id = (metadata_buffer_id + 1) % metadata_buf->num_frames;
        first_pass=false;
    }

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

