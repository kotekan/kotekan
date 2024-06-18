#include "clInputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clInputData);

clInputData::clInputData(Config& config, const std::string& unique_name,
                         bufferContainer& host_buffers, clDeviceInterface& device,
                         int instance_num) :
    clCommand(config, unique_name, host_buffers, device, instance_num, no_cl_command_state, "clInputData", "") {

    _gpu_memory = config.get_default<std::string>(unique_name, "gpu_memory", "input");

    // Get the input buffers and register as a consumer if we are the first instance.
    in_bufs = get_buffer_array("in_bufs", instance_num == 0, false);

    command_type = gpuCommandType::COPY_IN;
}

clInputData::~clInputData() {}

int clInputData::wait_on_precondition() {

    Buffer * buf = in_bufs.at(gpu_frame_id % in_bufs.size());
    int frame_index = gpu_frame_id / in_bufs.size() % buf->num_frames;
    INFO("wait_for_full_frame called with buf: {}, buf index: {}, unique_name: {}, frame_index: {}, gpu_frame_id: {}", buf->buffer_name, gpu_frame_id % in_bufs.size(), unique_name, frame_index, gpu_frame_id);
    uint8_t* frame = wait_for_full_frame(buf, unique_name.c_str(), frame_index);
    if (frame == nullptr) {
        return -1;
    }
    return 0;
}

cl_event clInputData::execute(cl_event pre_event) {
    pre_execute();

    Buffer * buf = in_bufs[gpu_frame_id % in_bufs.size()];
    int frame_index = gpu_frame_id / in_bufs.size() % buf->num_frames;

    void* host_memory_frame = (void*)buf->frames[frame_index];
    // Data transfer to GPU
    device.async_copy_host_to_gpu(_gpu_memory, gpu_frame_id, host_memory_frame,
                                  buf->aligned_frame_size, pre_event, post_event);

    return post_event;
}

void clInputData::finalize_frame() {
    clCommand::finalize_frame();
    
    Buffer * buf = in_bufs[gpu_frame_id % in_bufs.size()];
    int frame_index = gpu_frame_id / in_bufs.size() % buf->num_frames;
    INFO("Mark_frame_empty called with: buf: {}, buf index {}, unique_name: {}, frame_index: {}, gpu_frame_id: {}", buf->buffer_name, gpu_frame_id % in_bufs.size(), unique_name, frame_index, gpu_frame_id);
    mark_frame_empty(buf, unique_name.c_str(), frame_index);
}
