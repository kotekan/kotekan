#include "hsaRfiInputSum.hpp"
#include "hsaBase.h"
#include <math.h>
#include <mutex>

REGISTER_HSA_COMMAND(hsaRfiInputSum);

hsaRfiInputSum::hsaRfiInputSum(Config& config,
                       const string &unique_name,
                       bufferContainer& host_buffers,
                       hsaDeviceInterface& device) :
    hsaCommand("rfi_chime_inputsum", "rfi_chime_inputsum.hsaco", config, unique_name, host_buffers, device) {
//    hsaCommand("rfi_chime_inputsum", "rfi_chime_inputsum_private.hsaco", config, unique_name, host_buffers, device) {

    command_type = CommandType::KERNEL;
    //Retrieve parameters from kotekan config
    //Data Parameters
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    //RFI Config Parameters
    _sk_step = config.get_int(unique_name, "sk_step");

    //Compute Buffer lengths
    input_frame_len = sizeof(float)*_num_elements*_num_local_freq*_samples_per_data_set/_sk_step;
    output_frame_len = sizeof(float)*_num_local_freq*_samples_per_data_set/_sk_step;
    _num_bad_inputs = config.get_int_array(unique_name, "bad_inputs").size();
    _M = (_num_elements - _num_bad_inputs)*_sk_step;

    using namespace std::placeholders;
    restServer &rest_server = restServer::instance();
    string endpoint = "/rfi_input_sum_callback/" + std::to_string(device.get_gpu_id());
    rest_server.register_post_callback(endpoint,
            std::bind(&hsaRfiInputSum::rest_callback, this, _1, _2));
}

hsaRfiInputSum::~hsaRfiInputSum() {
}

void hsaRfiInputSum::rest_callback(connectionInstance& conn, json& json_request) {
    std::lock_guard<std::mutex> lock(rest_callback_mutex);
    INFO("RFI Input Sum Callback Received... Changing Parameters")

    _num_bad_inputs = json_request["num_bad_inputs"];
    _M = (_num_elements - _num_bad_inputs)*_sk_step;

    conn.send_empty_reply(HTTP_RESPONSE::OK);
}


hsa_signal_t hsaRfiInputSum::execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) {
    std::lock_guard<std::mutex> lock(rest_callback_mutex);

    struct __attribute__ ((aligned(16))) args_t {
	void *input;
	void *output;
	uint32_t num_elements;
	uint32_t M;
    } args;

    memset(&args, 0, sizeof(args));
    args.input = device.get_gpu_memory("timesum", input_frame_len);
    args.output = device.get_gpu_memory_array("rfi_output",gpu_frame_id, output_frame_len);
    args.num_elements = _num_elements;
    args.M = _M;

    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

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

/*    params.workgroup_size_x = _num_local_freq;
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;
    params.grid_size_x = _num_local_freq;
    params.grid_size_y = (_samples_per_data_set/_sk_step)/24;
    params.grid_size_z = 24;
    params.num_dims = 3;
    params.private_segment_size = 0;
    params.group_segment_size = 16384;*/

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}
