#include "gpuCommand.hpp"

#include <iostream>

using std::string;
using std::to_string;

gpuCommand::gpuCommand(Config& config_, const string& unique_name_, bufferContainer& host_buffers_,
                       gpuDeviceInterface& device_, const string& default_kernel_command,
                       const string& default_kernel_file_name) :
    kernel_command(default_kernel_command),
    kernel_file_name(default_kernel_file_name),
    config(config_),
    unique_name(unique_name_),
    host_buffers(host_buffers_),
    dev(device_) {
    _gpu_buffer_depth = config.get<int>(unique_name, "buffer_depth");

    // Set the local log level.
    string s_log_level = config.get<string>(unique_name, "log_level");
    set_log_level(s_log_level);
    set_log_prefix(unique_name);

    // Load the kernel if there is one.
    if (default_kernel_file_name != "") {
        kernel_file_name =
            config.get_default<string>(unique_name, "kernel_path", ".") + "/"
            + config.get_default<string>(unique_name, "kernel", default_kernel_file_name);
        kernel_command = config.get_default<string>(unique_name, "command", default_kernel_command);
    }
}

gpuCommand::~gpuCommand() {}

void gpuCommand::finalize_frame(int gpu_frame_id) {
    (void)gpu_frame_id;
}

int gpuCommand::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    return 0;
}


string& gpuCommand::get_name() {
    return kernel_command;
}

void gpuCommand::pre_execute(int gpu_frame_id) {
    assert(gpu_frame_id < _gpu_buffer_depth);
    assert(gpu_frame_id >= 0);
}

double gpuCommand::get_last_gpu_execution_time() {
    return last_gpu_execution_time;
}

gpuCommandType gpuCommand::get_command_type() {
    return command_type;
}
