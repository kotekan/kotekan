#include "hsaProcess.hpp"
#include "unistd.h"
#include "vdif_functions.h"
#include "fpga_header_functions.h"
#include "KotekanProcess.hpp"
#include "util.h"

#include <iostream>
#include <sys/time.h>
#include <memory>

REGISTER_KOTEKAN_PROCESS(hsaProcess);

hsaProcess::hsaProcess(Config& config, const string& unique_name,
                     bufferContainer &buffer_container):
    gpuProcess(config, unique_name, buffer_container)
{
    device = new hsaDeviceInterface(config, gpu_id, _gpu_buffer_depth);
    dev = device;
    init();
}

gpuEventContainer *hsaProcess::create_signal(){
    return new hsaEventContainer();
}

hsaProcess::~hsaProcess() {
    delete device;
}

gpuCommand *hsaProcess::create_command(json cmd_info)
{
    auto cmd = FACTORY(hsaCommand)::create_bare(cmd_info["name"], config, unique_name, local_buffer_container, *device);
    return cmd;
}

void hsaProcess::queue_commands(int gpu_frame_id)
{
    hsa_signal_t signal;
    signal.handle = 0;
    for (auto &command : commands) {
        // Feed the last signal into the next operation
        signal = ((hsaCommand*)command)->execute(gpu_frame_id, signal);
    }
    final_signals[gpu_frame_id]->set_signal(&signal);
}

