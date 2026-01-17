#include "buffer.hpp"
#include "clCommand.hpp"
#include "chimeMetadata.hpp"

REGISTER_CL_COMMAND(clFakeInput);

clFakeInput::clFakeInput(Config& config, const std::string& unique_name,
                         bufferContainer& host_buffers, clDeviceInterface& device,
                         int gpu_id) :
    clCommand(config, unique_name, host_buffers, device, gpu_id, no_cl_command_state, "", "") {

    out_buf = host_buffers.get_buffer("fake_input_buffer");
}

cl_event clFakeInput::execute(cl_event pre_event) {
    // Get the GPU buffer pointer
    cl_mem gpu_buf = device.get_gpu_memory_array("input", 0, 1024 * sizeof(float));

    // Fill with e.g., 1.0
    std::vector<float> fake_data(1024, 1.0);
    clEnqueueWriteBuffer(device.getQueue(0), gpu_buf, CL_TRUE, 0,
                         1024 * sizeof(float), fake_data.data(), 0, nullptr, nullptr);

    return nullptr;
}
