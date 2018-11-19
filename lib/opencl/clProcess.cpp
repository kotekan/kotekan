#include "clProcess.hpp"
#include "unistd.h"
#include "util.h"

#include <iostream>
#include <sys/time.h>

using namespace std;

REGISTER_KOTEKAN_PROCESS(clProcess);

// TODO Remove the GPU_ID from this constructor
clProcess::clProcess(Config& config_,
        const string& unique_name,
        bufferContainer &buffer_container):
    gpuProcess(config_, unique_name, buffer_container)
{
    device = new clDeviceInterface(config_, gpu_id, _gpu_buffer_depth);
    dev = device;
    device->prepareCommandQueue(true); //yes profiling
    init();
}

clProcess::~clProcess() {
}

gpuEventContainer *clProcess::create_signal(){
    return new clEventContainer();
}

gpuCommand *clProcess::create_command(json cmd_info){
    auto cmd = FACTORY(clCommand)::create_bare(cmd_info["name"], config,
                                               unique_name, local_buffer_container,
                                               *device);
    cmd->build();
    std::string name = cmd_info["name"];
    DEBUG("Command added: %s",name.c_str());
    return cmd;
}

void clProcess::queue_commands(int gpu_frame_id)
{
    cl_event signal = NULL;
    for (auto &command : commands) {
        // Feed the last signal into the next operation
        signal = ((clCommand*)command)->execute(gpu_frame_id, 0, signal);
    }
    final_signals[gpu_frame_id]->set_signal(signal);
    INFO("Commands executed.");
}
