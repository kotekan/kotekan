#include "hsaBeamformKernel.hpp"
#include "configUpdater.hpp"

#include <signal.h>

REGISTER_HSA_COMMAND(hsaBeamformKernel);

// Request gain file re-parse with e.g.
// curl localhost:12048/frb_gain -X POST -H 'Content-Type: appication/json' -d '{"frb_gain_dir":"the_new_path"}'
// Update NS beam
// curl localhost:12048/gpu/gpu_<gpu_id>/frb/update_NS_beam/<gpu_id> -X POST -H 'Content-Type: application/json' -d '{"northmost_beam":<value>}'
// Update EW beam
// curl localhost:12048/gpu/gpu_<gpu_id>/frb/update_EW_beam/<gpu_id> -X POST -H 'Content-Type: application/json' -d '{"ew_id":<value>,"ew_beam":<value>}'

hsaBeamformKernel::hsaBeamformKernel(Config& config, const string &unique_name,
                            bufferContainer& host_buffers,
                            hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "zero_padded_FFT512","unpack_shift_beamform_flip.hsaco") {
    command_type = gpuCommandType::KERNEL;

    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");

    scaling = config.get_default<float>(unique_name, "frb_scaling", 1.0);
    vector<float> dg = {0.0,0.0}; //re,im
    default_gains = config.get_default<std::vector<float>>(
                unique_name, "frb_missing_gains", dg);

    _northmost_beam = config.get<float>(unique_name, "northmost_beam");
    freq_ref = (LIGHT_SPEED*(128) / (sin(_northmost_beam *PI/180.) * FEED_SEP *256))/1.e6;

    _ew_spacing = config.get<std::vector<float>>(unique_name, "ew_spacing");
    _ew_spacing_c = (float *)hsa_host_malloc(4*sizeof(float));
    for (int i=0;i<4;i++){
        _ew_spacing_c[i] = _ew_spacing[i];
    }

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
    freq_idx = -1;
    freq_MHz = -1;

    update_gains=true;
    update_NS_beam=true;
    update_EW_beam=true;
    first_pass=true;

    using namespace std::placeholders;
    restServer &rest_server = restServer::instance();
    endpoint_NS_beam = unique_name + "/frb/update_NS_beam/" + std::to_string(device.get_gpu_id());
    rest_server.register_post_callback(endpoint_NS_beam,
            std::bind(&hsaBeamformKernel::update_NS_beam_callback, this, _1, _2));
    endpoint_EW_beam = unique_name + "/frb/update_EW_beam/" + std::to_string(device.get_gpu_id());
    rest_server.register_post_callback(endpoint_EW_beam,
            std::bind(&hsaBeamformKernel::update_EW_beam_callback, this, _1, _2));
    //listen for gain updates
    _gain_dir = config.get_default<std::string>(unique_name,"updatable_gain_frb","");
    if (_gain_dir.length() > 0)
        configUpdater::instance().subscribe(config.get<std::string>(unique_name,"updatable_gain_frb"),
                                            std::bind(&hsaBeamformKernel::update_gains_callback, this, _1));
}

hsaBeamformKernel::~hsaBeamformKernel() {
    restServer::instance().remove_json_callback(endpoint_NS_beam);
    restServer::instance().remove_json_callback(endpoint_EW_beam);
    hsa_host_free(host_map);
    hsa_host_free(host_coeff);
    hsa_host_free(host_gain);
    hsa_host_free(_ew_spacing_c);
    // TODO Free device memory allocations.
}


bool hsaBeamformKernel::update_gains_callback(nlohmann::json &json) {
    //we're not fussy about exactly when the gains update, so no need for a lock here
    update_gains=true;
    try {
        _gain_dir = json.at("frb_gain_dir");
    } catch (std::exception& e) {
        WARN("[FRB] Fail to read gain_dir %s", e.what());
        return false;
    }
    INFO("[FRB] updated gain with %s", _gain_dir.c_str());
    return true;
}

void hsaBeamformKernel::update_EW_beam_callback(connectionInstance& conn, json& json_request) {
    int ew_id;
    try {
        ew_id = json_request["ew_id"];
    } catch (...) {
        conn.send_error("could not parse FRB E-W beam update", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    _ew_spacing_c[ew_id] = json_request["ew_beam"];
    update_EW_beam=true;
    config.update_value(unique_name, "ew_spacing/" + std::to_string(ew_id), json_request["ew_beam"]);
    conn.send_empty_reply(HTTP_RESPONSE::OK);

}

void hsaBeamformKernel::update_NS_beam_callback(connectionInstance& conn, json& json_request) {
    try {
        _northmost_beam = json_request["northmost_beam"];
    } catch (...) {
        conn.send_error("could not parse FRB N-S beam update", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    freq_ref = (LIGHT_SPEED*(128) / (sin(_northmost_beam *PI/180.) * FEED_SEP *256))/1.e6;
    update_NS_beam=true;

    config.update_value(unique_name, "northmost_beam", json_request["northmost_beam"]);
    conn.send_empty_reply(HTTP_RESPONSE::OK);
    config.update_value(unique_name, "gain_dir", _gain_dir);
}

int hsaBeamformKernel::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    uint8_t * frame = wait_for_full_frame(metadata_buf, unique_name.c_str(), metadata_buffer_precondition_id);
    if (frame == NULL) return -1;
    metadata_buffer_precondition_id = (metadata_buffer_precondition_id + 1) % metadata_buf->num_frames;
    return 0;
}


void hsaBeamformKernel::calculate_cl_index(uint32_t *host_map, float freq_now, double freq_ref) {
    float t, delta_t, beam_ref;
    int cl_index;
    float D2R = PI/180.;
    int pad = 2 ;

    for (int b = 0; b < 256; ++b){
        beam_ref = asin(LIGHT_SPEED*(b-256/2.) / (freq_ref*1.e6) / (256) /FEED_SEP) * 180./ PI;
        t = 256*pad*(freq_ref*1.e6)*(FEED_SEP/LIGHT_SPEED*sin(beam_ref*D2R)) + 0.5;
        delta_t = 256*pad*(freq_now*1e6-freq_ref*1e6) * (FEED_SEP/LIGHT_SPEED*sin(beam_ref*D2R));
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
}

void hsaBeamformKernel::calculate_ew_phase(float freq_now, float *host_coeff, float *_ew_spacing_c) {
    for (int angle_iter=0; angle_iter < 4; angle_iter++){
        float anglefrac = sin(_ew_spacing_c[angle_iter]*PI/180.);
        for (int cylinder=0; cylinder < 4; cylinder++) {
            host_coeff[angle_iter*4*2 + cylinder*2] = cos(2*PI*anglefrac*cylinder*22*freq_now*1.e6/LIGHT_SPEED);
            host_coeff[angle_iter*4*2 + cylinder*2 + 1] = sin(2*PI*anglefrac*cylinder*22*freq_now*1.e6/LIGHT_SPEED);
        }
    }
}


hsa_signal_t hsaBeamformKernel::execute(int gpu_frame_id, hsa_signal_t precede_signal) {
    if (first_pass) {
        first_pass = false;
        stream_id_t stream_id = get_stream_id_t(metadata_buf, metadata_buffer_id);
        freq_idx = bin_number_chime(&stream_id);
        freq_MHz = freq_from_bin(freq_idx);

        metadata_buffer_id = (metadata_buffer_id + 1) % metadata_buf->num_frames;
    }

    if (update_gains) {
        //brute force wait to make sure we don't clobber memory
        if (hsa_signal_wait_scacquire(precede_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                        UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0) {
            ERROR("***** ERROR **** Unexpected signal value **** ERROR **** ");
        }
        update_gains=false;
        FILE *ptr_myfile;
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/quick_gains_%04d_reordered.bin",_gain_dir.c_str(),freq_idx);
        INFO("Loading gains from %s",filename);
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
                raise(SIGINT);
                return precede_signal;
            }
            fclose(ptr_myfile);
            for (uint32_t i=0; i<2048; i++){
                host_gain[i*2  ] = host_gain[i*2  ] * scaling;
                host_gain[i*2+1] = host_gain[i*2+1] * scaling;
            }
        }
        void * device_gain = device.get_gpu_memory("beamform_gain", gain_len);
        device.sync_copy_host_to_gpu(device_gain, (void*)host_gain, gain_len);
    }

    if (update_NS_beam) {
        calculate_cl_index(host_map, freq_MHz, freq_ref);
        void * device_map = device.get_gpu_memory("beamform_map", map_len);
        device.sync_copy_host_to_gpu(device_map, (void *)host_map, map_len);
        update_NS_beam = false;
    }
    if (update_EW_beam) {
        calculate_ew_phase(freq_MHz, host_coeff, _ew_spacing_c);
        void * device_coeff_map = device.get_gpu_memory("beamform_coeff_map", coeff_len);
        device.sync_copy_host_to_gpu(device_coeff_map, (void*)host_coeff, coeff_len);
        update_EW_beam = false;
    }

    struct __attribute__ ((aligned(16))) args_t {
        void *input_buffer;
        void *map_buffer;
        void *coeff_buffer;
        void *output_buffer;
        void *gain_buffer;
    } args;
    memset(&args, 0, sizeof(args));

    args.input_buffer = device.get_gpu_memory("input_reordered", input_frame_len);
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
