#include "hsaProcess.hpp"

#include "KotekanProcess.hpp"
#include "fpga_header_functions.h"
#include "unistd.h"
#include "util.h"
#include "vdif_functions.h"

#include <iostream>
#include <memory>
#include <sys/time.h>

REGISTER_KOTEKAN_PROCESS(hsaProcess);

hsaProcess::hsaProcess(Config& config, const string& unique_name,
                       bufferContainer& buffer_container) :
    gpuProcess(config, unique_name, buffer_container) {
    dev = (gpuDeviceInterface*)new hsaDeviceInterface(config, gpu_id, _gpu_buffer_depth);
    init();
}

gpuEventContainer* hsaProcess::create_signal() {
    return new hsaEventContainer();
}

hsaProcess::~hsaProcess() {}

gpuCommand* hsaProcess::create_command(const std::string& cmd_name,
                                       const std::string& unique_name) {
    auto cmd = FACTORY(hsaCommand)::create_bare(cmd_name, config, unique_name,
                                                local_buffer_container, *(hsaDeviceInterface*)dev);
    return cmd;
}

void hsaProcess::queue_commands(int gpu_frame_id) {
    hsa_signal_t signal;
    signal.handle = 0;
    for (auto& command : commands) {
        // Feed the last signal into the next operation
        signal = ((hsaCommand*)command)->execute(gpu_frame_id, signal);
    }
    final_signals[gpu_frame_id]->set_signal(&signal);
}
