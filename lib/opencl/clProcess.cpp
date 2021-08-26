#include "clProcess.hpp"

#include "StageFactory.hpp"
#include "unistd.h"
#include "util.h"

#include <iostream>
#include <sys/time.h>

using kotekan::bufferContainer;
using kotekan::Config;

using namespace std;

REGISTER_KOTEKAN_STAGE(clProcess);

// TODO Remove the GPU_ID from this constructor
clProcess::clProcess(Config& config_, const std::string& unique_name,
                     bufferContainer& buffer_container) :
    gpuProcess(config_, unique_name, buffer_container) {
    device = new clDeviceInterface(config_, gpu_id, _gpu_buffer_depth);
    dev = device;
    device->prepareCommandQueue(true); // yes profiling
    init();
}

clProcess::~clProcess() {
    // Unregister the host memory from the OpenCL runtime
    for (auto& opencl_frame : opencl_host_frames) {
        cl_event wait_event;
        clEnqueueUnmapMemObject(device->getQueue(0), std::get<0>(opencl_frame),
                                std::get<1>(opencl_frame), 0, nullptr, &wait_event);
        // Block here to make sure the memory actually gets unmapped.
        clWaitForEvents(1, &wait_event);
    }
}

gpuEventContainer* clProcess::create_signal() {
    return new clEventContainer();
}

gpuCommand* clProcess::create_command(const std::string& cmd_name, const std::string& unique_name) {
    auto cmd = FACTORY(clCommand)::create_bare(cmd_name, config, unique_name,
                                               local_buffer_container, *device);
    // TODO Why is this not in the constructor?
    cmd->build();
    DEBUG("Command added: {:s}", cmd_name);
    return cmd;
}

void clProcess::queue_commands(int gpu_frame_id) {
    cl_event signal = nullptr;
    for (auto& command : commands) {
        // Feed the last signal into the next operation
        signal = ((clCommand*)command)->execute(gpu_frame_id, signal);
    }
    final_signals[gpu_frame_id]->set_signal(signal);
    DEBUG("Commands executed.");
}

void clProcess::register_host_memory(struct Buffer* host_buffer) {
    // Register the host memory in in_buf with the OpenCL run time.
    for (int i = 0; i < host_buffer->num_frames; i++) {
        cl_int err;
        cl_mem cl_mem_prt =
            clCreateBuffer(device->get_context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                           host_buffer->frame_size, host_buffer->frames[i], &err);
        CHECK_CL_ERROR(err);
        void* pinned_ptr =
            clEnqueueMapBuffer(device->getQueue(0), cl_mem_prt, CL_TRUE, CL_MAP_READ, 0,
                               host_buffer->frame_size, 0, nullptr, nullptr, &err);
        CHECK_CL_ERROR(err);

        // As far as I can tell pinned_ptr should always be the same as the host pointer,
        // if it's not this has implications for upstream processes and so the system should fail
        // when the condition isn't met.
        if ((void*)host_buffer->frames[i] != pinned_ptr) {
            ERROR("OpenCL registered pointer is different from normal host pointer: {:p}, opencl "
                  "pointer: {:p}",
                  (void*)host_buffer->frames[i], pinned_ptr);
            throw std::runtime_error(
                "Something wrong with the registration of host memory in OpenCL");
        }

        DEBUG("Registed frame: {:s}[{:d}]", host_buffer->buffer_name, i);
        opencl_host_frames.push_back(std::make_tuple(cl_mem_prt, pinned_ptr));
    }
}