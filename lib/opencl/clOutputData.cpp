#include "clOutputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clOutputData);

clOutputData::clOutputData(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, clDeviceInterface& device,
                           int instance_num) :
    clCommand(config, unique_name, host_buffers, device, instance_num, "clOutputData", ""),
    in_bufs(config, unique_name, host_buffers, "in_bufs", false),
    out_bufs(config, unique_name, host_buffers, "out_bufs", true) {

    _gpu_memory = config.get_default<std::string>(unique_name, "gpu_memory", "output");

    command_type = gpuCommandType::COPY_OUT;
}

clOutputData::~clOutputData() {}

int clOutputData::wait_on_precondition() {
    NextFrameCollection in_frame_collection = in_bufs.get_next_frame_precondition();
    if (in_frame_collection.frame == nullptr)
        return -1;

    NextFrameCollection out_frame_collection = out_bufs.get_next_frame_precondition();
    if (out_frame_collection.frame == nullptr)
        return -1;

    return 0;
}


cl_event clOutputData::execute(cl_event pre_event) {
    pre_execute();

    in_bufs.get_next_frame_execute();
    NextFrameCollection out_frame_collection = out_bufs.get_next_frame_execute();

    cl_mem gpu_output_frame = device.get_gpu_memory_array(
        _gpu_memory, gpu_frame_id, out_frame_collection.buf->aligned_frame_size);
    void* host_output_frame = (void*)out_frame_collection.frame;

    // Transfer data from the device to host
    CHECK_CL_ERROR(clEnqueueReadBuffer(device.getQueue(2), gpu_output_frame, CL_FALSE, 0,
                                       out_frame_collection.buf->aligned_frame_size,
                                       host_output_frame, 1, &pre_event,
                                       &post_event));

    return post_event;
}

void clOutputData::finalize_frame() {
    clCommand::finalize_frame();

    NextFrameCollection in_frame_collection = in_bufs.get_next_frame_finalize();
    NextFrameCollection out_frame_collection = out_bufs.get_next_frame_finalize();

    pass_metadata(in_frame_collection.buf, in_frame_collection.frame_id, out_frame_collection.buf,
                  out_frame_collection.frame_id);

    in_bufs.release_frame_finalize();
    out_bufs.release_frame_finalize();
}
