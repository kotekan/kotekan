#include "clOutputBeamformResult.hpp"

clOutputBeamformResult::clOutputBeamformResult(Config& config, const string &unique_name,
                            bufferContainer& host_buffers, clDeviceInterface& device) :
    clCommand("", "", config, unique_name, host_buffers, device)
{
//    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
//    _block_size = config.get_int(unique_name, "block_size");
    _num_data_sets = config.get_int(unique_name, "num_data_sets");
//    _num_blocks = config.get_int(unique_name,"num_blocks");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    beam_out_buf = host_buffers.get_buffer("beam_out_buf");

}

clOutputBeamformResult::~clOutputBeamformResult()
{
}

void clOutputBeamformResult::build()
{
    apply_config(0);
    clCommand::build();
}

cl_event clOutputBeamformResult::execute(int gpu_frame_id, const uint64_t& fpga_seq, cl_event pre_event)
{
    clCommand::execute(gpu_frame_id, 0, pre_event);

    uint32_t output_len = _samples_per_data_set * _num_data_sets * _num_local_freq * 2;
    cl_mem output_memory_frame = device.get_gpu_memory_array("output",gpu_frame_id, output_len);

    CHECK_CL_ERROR( clEnqueueReadBuffer(device.getQueue(2),
                                        output_memory_frame,
                                        CL_FALSE,
                                        0,
                                        output_len,
                                        beam_out_buf,
                                        1,
                                        &pre_event,
                                        &post_event[gpu_frame_id]) );

    return post_event[gpu_frame_id];
}



