#include "gpuCommand.hpp"

#include "Config.hpp" // for Config

#include "fmt.hpp"

#include <assert.h>  // for assert
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

using std::string;
using std::to_string;

gpuCommand::gpuCommand(Config& config_, const std::string& unique_name_,
                       bufferContainer& host_buffers_, gpuDeviceInterface& device_,
                       int instance_num_, std::shared_ptr<gpuCommandState> state_,
                       const std::string& default_kernel_command,
                       const std::string& default_kernel_file_name) :
    kernel_file_name(default_kernel_file_name),
    config(config_), unique_name(unique_name_), host_buffers(host_buffers_), dev(device_),
    instance_num(instance_num_), command_state(state_) {
    _gpu_buffer_depth = config.get<int>(unique_name, "buffer_depth");

    // Set the local log level.
    std::string s_log_level = config.get<string>(unique_name, "log_level");
    set_log_level(s_log_level);
    set_name(default_kernel_command);

    // Load the kernel if there is one.
    if (default_kernel_file_name != "") {
        kernel_file_name =
            config.get_default<string>(unique_name, "kernel_path", ".") + "/"
            + config.get_default<string>(unique_name, "kernel", default_kernel_file_name);
        kernel_command = config.get_default<string>(unique_name, "command", default_kernel_command);
    }

    profiling = config.get_default<bool>(unique_name, "profiling", true);
    if (profiling) {
        frame_arrival_period = config.get<double>(unique_name, "frame_arrival_period");
    }

    excute_time = kotekan::KotekanTrackers::instance().add_tracker(
        unique_name, get_name() + "_execute_time", "seconds");
    utilization =
        kotekan::KotekanTrackers::instance().add_tracker(unique_name, get_name() + "_u", "");
}

gpuCommand::~gpuCommand() {}

void gpuCommand::start_frame(int64_t _gpu_frame_id) {
    gpu_frame_id = _gpu_frame_id;
}

void gpuCommand::finalize_frame() {}

int gpuCommand::wait_on_precondition() {
    return 0;
}

void gpuCommand::set_name(const std::string& c) {
    kernel_command = c;
    set_log_prefix(fmt::format("{:s} ({:30s})", unique_name, kernel_command));
}

string& gpuCommand::get_name() {
    return kernel_command;
}

std::string gpuCommand::get_unique_name() const {
    return unique_name;
}

void gpuCommand::pre_execute() {}

double gpuCommand::get_last_gpu_execution_time() {
    return excute_time->get_current();
}

gpuCommandType gpuCommand::get_command_type() {
    return command_type;
}
