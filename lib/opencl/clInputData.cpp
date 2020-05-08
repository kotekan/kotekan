#include "clInputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;
using std::string;

REGISTER_CL_COMMAND(clInputData);

clInputData::clInputData(Config& config, const string& unique_name, bufferContainer& host_buffers,
                         clDeviceInterface& device) :
    clCommand(config, unique_name, host_buffers, device, "clInputData", ""),
    in_buf(host_buffers.get_buffer(config.get<std::string>(unique_name, "in_buf"))),
    in_buf_id(in_buf),
    in_buf_precondition_id(in_buf),
    in_buf_finalize_id(in_buf),
    _gpu_memory_name(config.get<std::string>(unique_name, "gpu_memory_name")) {

    command_type = gpuCommandType::COPY_IN;

    register_consumer(in_buf, unique_name.c_str());

    // Register the host memory in in_buf with the OpenCL run time.
    for (int i = 0; i < in_buf->num_frames; i++) {
        cl_int err;
        cl_mem cl_mem_prt = clCreateBuffer(device.get_context(),
                                    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                    in_buf->frame_size,
                                    in_buf->frames[i],
                                    &err);
        CHECK_CL_ERROR(err);
        void* pinned_ptr = clEnqueueMapBuffer(device.getQueue(0), cl_mem_prt, CL_TRUE,
                                                CL_MAP_READ, 0, in_buf->frame_size, 0, nullptr, nullptr, &err);
        CHECK_CL_ERROR(err);

        // As far as I can tell pinned_ptr should always be the same as the host pointer,
        // if it's not this as implications for upstream processes and so the system should fail
        // when the condition isn't meet.
        if ((void*)in_buf->frames[i] != pinned_ptr) {
            ERROR("OpenCL registered pointer is different from normal host pointer: {:p}, opencl pointer: {:p}",
                  (void*)in_buf->frames[i],
                  pinned_ptr);
            throw std::runtime_error("Something wrong with the registration of host memory in OpenCL");
        }

        opencl_in_buf_frames.push_back(std::make_tuple(cl_mem_prt, pinned_ptr));
    }

}

clInputData::~clInputData() {

    // Unregister the host memory from the OpenCL runtime
    for (auto &opencl_frame: opencl_in_buf_frames) {
        cl_event wait_event;
        clEnqueueUnmapMemObject(device.getQueue(0),
                                std::get<0>(opencl_frame),
                                std::get<1>(opencl_frame), 0, nullptr, &wait_event);
        // Block here to make sure the memory actually gets unmapped.
        clWaitForEvents(1, &wait_event);
    }
}

int clInputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Wait for there to be data in the input (network) buffer.
    uint8_t* frame =
        wait_for_full_frame(in_buf, unique_name.c_str(), in_buf_precondition_id);
    if (frame == nullptr)
        return -1;

    in_buf_precondition_id++;
    return 0;
}

cl_event clInputData::execute(int gpu_frame_id, cl_event pre_event) {
    pre_execute(gpu_frame_id);

    cl_mem gpu_memory_frame = device.get_gpu_memory_array(_gpu_memory_name,
                                                          gpu_frame_id, in_buf->frame_size);
    void* host_memory_frame = (void*)in_buf->frames[in_buf_id];

    // Data transfer to GPU
    CHECK_CL_ERROR(clEnqueueWriteBuffer(
        device.getQueue(0), gpu_memory_frame, CL_FALSE,
        0, // offset
        in_buf->frame_size, host_memory_frame, (pre_event == nullptr) ? 0 : 1,
        (pre_event == nullptr) ? nullptr : &pre_event, &post_events[gpu_frame_id]));

    in_buf_id++;
    return post_events[gpu_frame_id];
}

void clInputData::finalize_frame(int frame_id) {
    clCommand::finalize_frame(frame_id);
    mark_frame_empty(in_buf, unique_name.c_str(), in_buf_finalize_id);
    in_buf_finalize_id++;
}

std::string clInputData::get_performance_metric_string() {
    double transfer_speed = (double)in_buf->frame_size
                            / (double)get_last_gpu_execution_time() / 1000000000;
    return fmt::format("Speed: {:.2f} GB/s ({:.2f} Gb/s)", transfer_speed, transfer_speed * 8);
}