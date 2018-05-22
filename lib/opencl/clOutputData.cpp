#include "clOutputData.hpp"

REGISTER_CL_COMMAND(clOutputData);

clOutputData::clOutputData(Config& config, const string &unique_name,
                            bufferContainer& host_buffers, clDeviceInterface& device) :
    clCommand("", "", config, unique_name, host_buffers, device)
{
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _block_size = config.get_int(unique_name, "block_size");
    _num_data_sets = config.get_int(unique_name, "num_data_sets");
    _num_blocks = config.get_int(unique_name,"num_blocks");

    output_buf = host_buffers.get_buffer("output_buf");
    output_buffer_id = 0;
    output_buffer_precondition_id = 0;
    output_buffer_finalize_id = 0;
}

clOutputData::~clOutputData()
{
}

int clOutputData::wait_on_precondition(int gpu_frame_id)
{
    // Wait for there to be data in the input (output) buffer.
    uint8_t * frame = wait_for_empty_frame(output_buf, unique_name.c_str(), output_buffer_precondition_id);
    if (frame == NULL) return -1;
    //INFO("Got full buffer %s[%d], gpu[%d][%d]", output_buf->buffer_name, output_buffer_precondition_id,
    //        device.get_gpu_id(), gpu_frame_id);

    output_buffer_precondition_id = (output_buffer_precondition_id + 1) % output_buf->num_frames;
    return 0;
}


cl_event clOutputData::execute(int buf_frame_id, const uint64_t& fpga_seq, cl_event pre_event)
{
    DEBUG2("CLOUTPUTDATA::EXECUTE");

    clCommand::execute(buf_frame_id, 0, pre_event);

    uint32_t output_len = _num_local_freq * _num_blocks * (_block_size*_block_size) * 2 * _num_data_sets  * sizeof(int32_t);

    cl_mem gpu_output_frame = device.get_gpu_memory_array("output",
                                                buf_frame_id, output_len);
    void * host_output_frame = (void *)output_buf->frames[output_buffer_id];

    // Read the results
    CHECK_CL_ERROR( clEnqueueReadBuffer(device.getQueue(2),
                                            gpu_output_frame,
                                            CL_FALSE,
                                            0,
                                            output_len,
                                            host_output_frame,
                                            (pre_event==NULL)?0:1,
                                            (pre_event==NULL)?NULL:&pre_event,
                                            &post_event[buf_frame_id]) );

    return post_event[buf_frame_id];
}


