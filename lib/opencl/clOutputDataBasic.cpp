#include "clOutputDataBasic.hpp"

#include "buffer.hpp"
#include "chimeMetadata.hpp"
#include "errors.h"

#include <iostream>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clOutputDataBasic);

clOutputDataBasic::clOutputDataBasic(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, clDeviceInterface& device,
                           int gpu_id) :
    clCommand(config, unique_name, host_buffers, device, gpu_id, no_cl_command_state, "", "") {

    // Get the buffer this stage reads from
    in_buf = host_buffers.get_buffer(config.get<std::string>(unique_name, "in_bufs")[0]);
}

cl_event clOutputDataBasic::execute(cl_event pre_event) {

    // For simplicity, always read frame 0
    uint32_t frame_id = 0;
    size_t num_elements = in_buf->num_elements;

    // Get the GPU memory pointer
    cl_mem gpu_buf = device.get_gpu_memory_array("output", frame_id, num_elements * sizeof(float));

    // Copy GPU buffer to CPU
    std::vector<float> host_data(num_elements);
    CHECK_CL_ERROR(clEnqueueReadBuffer(device.getQueue(0), gpu_buf, CL_TRUE, 0,
                                       num_elements * sizeof(float), host_data.data(),
                                       0, nullptr, nullptr));

    // Print first few elements
    std::cout << "[clOutputDataBasic] First 10 elements of frame " << frame_id << ": ";
    for (size_t i = 0; i < std::min(num_elements, size_t(10)); ++i) {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;

    return pre_event;
}
