REGISTER_CL_COMMAND(clMultiplyZero);

clMultiplyZero::clMultiplyZero(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, clDeviceInterface& device,
                               int gpu_id) :
    clCommand(config, unique_name, host_buffers, device, gpu_id, no_cl_command_state, "", "") {

    in_buf = host_buffers.get_buffer("fake_input_buffer");
    out_buf = host_buffers.get_buffer("fake_output_buffer");
}

cl_event clMultiplyZero::execute(cl_event pre_event) {
    cl_mem input_gpu = device.get_gpu_memory_array("input", 0, 1024 * sizeof(float));
    cl_mem output_gpu = device.get_gpu_memory_array("output", 0, 1024 * sizeof(float));

    // Read input back to CPU (or you could do OpenCL kernel multiply if you want)
    std::vector<float> data(1024);
    clEnqueueReadBuffer(device.getQueue(0), input_gpu, CL_TRUE, 0,
                        1024 * sizeof(float), data.data(), 0, nullptr, nullptr);

    // Multiply by 0
    for (auto& v : data)
        v = 34.0;

    clEnqueueWriteBuffer(device.getQueue(0), output_gpu, CL_TRUE, 0,
                         1024 * sizeof(float), data.data(), 0, nullptr, nullptr);

    return nullptr;
}
