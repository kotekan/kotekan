#include "clInputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clInputData);

clInputData::clInputData(Config& config, const std::string& unique_name,
                         bufferContainer& host_buffers, clDeviceInterface& device) :
    clCommand(config, unique_name, host_buffers, device, "clInputData", ""),
    in_bufs(config, unique_name, host_buffers, "in_bufs", false) {

    _gpu_memory = config.get_default<std::string>(unique_name, "gpu_memory", "input");

    command_type = gpuCommandType::COPY_IN;
}

clInputData::~clInputData() {}

int clInputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    NextFrameCollection in_frame_collection = in_bufs.get_next_frame_precondition();
    if (in_frame_collection.frame == nullptr)
        return -1;

    return 0;
}

cl_event clInputData::execute(int gpu_frame_id, cl_event pre_event) {
    pre_execute(gpu_frame_id);

    NextFrameCollection in_frame_collection = in_bufs.get_next_frame_execute();

    cl_mem gpu_memory_frame = device.get_gpu_memory_array(
        _gpu_memory, gpu_frame_id, in_frame_collection.buf->aligned_frame_size);
    void* host_memory_frame = (void*)in_frame_collection.frame;
    // Data transfer to GPU
    CHECK_CL_ERROR(clEnqueueWriteBuffer(device.getQueue(0), gpu_memory_frame, CL_FALSE,
                                        0, // offset
                                        in_frame_collection.buf->aligned_frame_size,
                                        host_memory_frame, (pre_event == nullptr) ? 0 : 1,
                                        (pre_event == nullptr) ? nullptr : &pre_event,
                                        &post_events[gpu_frame_id]));

    return post_events[gpu_frame_id];
}

void clInputData::finalize_frame(int frame_id) {
    clCommand::finalize_frame(frame_id);
    in_bufs.release_frame_finalize();
}
