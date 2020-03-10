#include "clProcess.hpp"

#include "unistd.h"
#include "util.h"

#include <iostream>
#include <sys/time.h>

using kotekan::bufferContainer;
using kotekan::Config;

using namespace std;

REGISTER_KOTEKAN_STAGE(clProcess);

// TODO Remove the GPU_ID from this constructor
clProcess::clProcess(Config& config_, const string& unique_name,
                     bufferContainer& buffer_container) :
    gpuProcess(config_, unique_name, buffer_container) {
    device = new clDeviceInterface(config_, gpu_id, _gpu_buffer_depth);
    dev = device;
    device->prepareCommandQueue(true); // yes profiling
    init();
}

clProcess::~clProcess() {}

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
    cl_event signal = NULL;
    for (auto& command : commands) {
        // Feed the last signal into the next operation
        signal = ((clCommand*)command)->execute(gpu_frame_id, signal);
    }
    final_signals[gpu_frame_id]->set_signal(signal);
    DEBUG("Commands executed.");
}
