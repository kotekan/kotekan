#include "hsaRfiInputSum.hpp"
#include "hsaBase.h"
#include <math.h>
#include <mutex>

REGISTER_HSA_COMMAND(hsaRfiInputSum);

hsaRfiInputSum::hsaRfiInputSum(Config& config,
                       const string &unique_name,
                       bufferContainer& host_buffers,
                       hsaDeviceInterface& device) :
    //Note, the rfi_chime_inputsum_private.hsaco kernel may be used in the future.
    hsaCommand("rfi_chime_inputsum", "rfi_chime_inputsum.hsaco", config, unique_name, host_buffers, device) {
    command_type = CommandType::KERNEL;
    //Retrieve parameters from kotekan config
    //Data Parameters
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    //RFI Config Parameters
    _sk_step = config.get_int_default(unique_name, "sk_step", 256);
    //Compute Buffer lengths
    input_frame_len = sizeof(float)*_num_elements*_num_local_freq*_samples_per_data_set/_sk_step;
    output_frame_len = sizeof(float)*_num_local_freq*_samples_per_data_set/_sk_step;
    mask_len = sizeof(uint8_t)*_num_elements;
    correction_frame_len = sizeof(uint32_t)*_samples_per_data_set/_sk_step;
    _bad_inputs = config.get_int_array(unique_name, "bad_inputs");
    _num_bad_inputs = _bad_inputs.size();
    //Local Parameters
    rebuildInputMask = true;
    //Register rest server endpoint
    using namespace std::placeholders;
    restServer &rest_server = restServer::instance();
    endpoint = unique_name + "/update_bad_inputs";
    rest_server.register_post_callback(endpoint,
            std::bind(&hsaRfiInputSum::rest_callback, this, _1, _2));
}

hsaRfiInputSum::~hsaRfiInputSum() {
    restServer::instance().remove_json_callback(endpoint);
    //Free allocated memory
    hsa_host_free(InputMask);
}

void hsaRfiInputSum::rest_callback(connectionInstance& conn, json& json_request) {
    //Lock Mutex
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    INFO("RFI Input Sum Callback Received... Changing Parameters")
    //Change internal parameters
    _bad_inputs.clear();
    for(uint32_t i = 0; i < json_request["bad_inputs"].size(); i++){
        _bad_inputs.push_back(json_request["bad_inputs"][i].get<int>());
    }
    //Update relevant variables
    _num_bad_inputs = _bad_inputs.size();
    //Flag for input mask rebuild
    rebuildInputMask = true;
    //Send reply
    conn.send_empty_reply(HTTP_RESPONSE::OK);
    config.update_value(unique_name, "bad_inputs", _bad_inputs);
}

hsa_signal_t hsaRfiInputSum::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {
    //Lock rest server callback mutex
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    //Build Input mask when needed (after rest callback or on first execution)
    if (rebuildInputMask) {
        rebuildInputMask = false;
        //Allocate memory for input mask
        InputMask = (uint8_t *)hsa_host_malloc(mask_len);
        //Fill input mask based on config parameters
        uint32_t j = 0;
        for(uint32_t i = 0; i < mask_len/sizeof(uint8_t); i++){
            InputMask[i] = (uint8_t)0;
            if(_bad_inputs.size() > 0 && (int32_t)i == _bad_inputs[j]){
                InputMask[i] = (uint8_t)1;
                j++;
            }
        }
        //Copy to gpu memory
        void * input_mask_map = device.get_gpu_memory("input_mask", mask_len);
        device.sync_copy_host_to_gpu(input_mask_map, (void *)InputMask, mask_len);
    }
    //Struct for hsa arguments
    struct __attribute__ ((aligned(16))) args_t {
        void *input;
        void *output;
        void *InputMask;
        void *LostSampleCorrection;
        uint32_t num_elements;
        uint32_t num_bad_inputs;
        uint32_t sk_step;

    } args;
    //Initialize arguments
    memset(&args, 0, sizeof(args));
    //Set arguments
    args.input = device.get_gpu_memory("timesum", input_frame_len);
    args.output = device.get_gpu_memory_array("rfi_output",gpu_frame_id, output_frame_len);
    args.InputMask = device.get_gpu_memory("input_mask", mask_len);
    args.LostSampleCorrection = device.get_gpu_memory("lost_sample_correction", correction_frame_len);
    args.num_elements = _num_elements;
    args.num_bad_inputs = _num_bad_inputs;
    args.sk_step = _sk_step;
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));
    //Set kernel execution parameters
    kernelParams params;
    params.workgroup_size_x = 256;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = 256;
    params.grid_size_y = _num_local_freq;
    params.grid_size_z = _samples_per_data_set/_sk_step;
    params.num_dims = 3;
    params.private_segment_size = 0;
    params.group_segment_size = 16384;
    //Parameters for rfi_chime_inputsum_private.hsaco, for easy switching if needed in future
/*    params.workgroup_size_x = _num_local_freq;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = _num_local_freq;
    params.grid_size_y = (_samples_per_data_set/_sk_step)/24;
    params.grid_size_z = 24;
    params.num_dims = 3;
    params.private_segment_size = 0;
    params.group_segment_size = 16384;*/
    //Execute kernel
    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);
    //Return signal
    return signals[gpu_frame_id];
}
