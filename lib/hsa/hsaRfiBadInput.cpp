#include "hsaRfiBadInput.hpp"
#include "hsaBase.h"
#include <math.h>
#include <unistd.h>
#include <mutex>

REGISTER_HSA_COMMAND(hsaRfiBadInput);

hsaRfiBadInput::hsaRfiBadInput(Config& config,const string &unique_name,
                         bufferContainer& host_buffers,
                         hsaDeviceInterface& device):
    hsaCommand("rfi_bad_input", "rfi_bad_input.hsaco", config, unique_name, host_buffers, device){
    command_type = CommandType::KERNEL;
    //Retrieve parameters from kotekan confint(unique_name, "num_elements");
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    //RFI Config Parameters
    _sk_step = config.get_int_default(unique_name, "sk_step", 256);
    //Compute Buffer lengths
    input_frame_len = sizeof(float)*_num_local_freq*_num_elements*_samples_per_data_set/_sk_step;
    output_frame_len = sizeof(float)*_num_local_freq*_num_elements;
    //Local Parameters
    //Register rest server endpoint
    using namespace std::placeholders;
    restServer &rest_server = restServer::instance();
    endpoint = unique_name + "/rfi_bad_input_callback/" + std::to_string(device.get_gpu_id());
    rest_server.register_post_callback(endpoint,
            std::bind(&hsaRfiBadInput::rest_callback, this, _1, _2));
}

hsaRfiBadInput::~hsaRfiBadInput() {
    restServer::instance().remove_json_callback(endpoint);
    //Free allocated memory
}

void hsaRfiBadInput::rest_callback(connectionInstance& conn, json& json_request) {
    //Lock mutex
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    INFO("RFI Callbak Received... Changing Parameters")
    //Change internal parameters
    //Update Config
    //Send reply
    conn.send_empty_reply(HTTP_RESPONSE::OK);
}

hsa_signal_t hsaRfiBadInput::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    //Structure for gpu arguments
    struct __attribute__ ((aligned(16))) args_t {
	void *input;
	void *output;
	uint32_t M;
        uint32_t num_sk;
    } args;
    //Initialize arguments
    memset(&args, 0, sizeof(args));
    //Set argumnets to correct values
    args.input = device.get_gpu_memory("timesum", input_frame_len);
    args.output = device.get_gpu_memory_array("rfi_bad_input", gpu_frame_id, output_frame_len);
    args.M = _sk_step;
    args.num_sk = _samples_per_data_set/_sk_step;
    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));
    // Apply correct kernel parameters
    kernelParams params;
    params.workgroup_size_x = 256;
    params.workgroup_size_y = 1;
    params.grid_size_x = _num_elements;
    params.grid_size_y = _num_local_freq;
    params.num_dims = 2;
    // Should this be zero?
    params.private_segment_size = 0;
    params.group_segment_size = 0;

    //Execute kernel
    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);
    //return signal
    return signals[gpu_frame_id];
}
