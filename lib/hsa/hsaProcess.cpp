#include "hsaProcess.hpp"

#include "Config.hpp"             // for Config
#include "StageFactory.hpp"       // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "factory.hpp"            // for FACTORY
#include "gpuDeviceInterface.hpp" // for gpuDeviceInterface
#include "gpuEventContainer.hpp"  // for gpuEventContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand, _factory_aliashsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config
#include "hsaEventContainer.hpp"  // for hsaEventContainer

#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <stdint.h>  // for uint32_t
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_KOTEKAN_STAGE(hsaProcess);

hsaProcess::hsaProcess(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    gpuProcess(config, unique_name, buffer_container) {
    uint32_t numa_node = config.get_default<uint32_t>(unique_name, "numa_node", 0);
    dev = (gpuDeviceInterface*)new hsaDeviceInterface(config, unique_name, gpu_id,
                                                      _gpu_buffer_depth, numa_node);
    init();
}

gpuEventContainer* hsaProcess::create_signal() {
    return new hsaEventContainer();
}

hsaProcess::~hsaProcess() {}

std::vector<gpuCommand*> hsaProcess::create_command(const std::string& cmd_name,
                                                    const std::string& unique_name) {
    auto cmd = FACTORY(hsaCommand)::create_bare(cmd_name, config, unique_name,
                                                local_buffer_container, *(hsaDeviceInterface*)dev);
    std::vector<gpuCommand*> v;
    v.push_back(cmd);
    return v;
}

void hsaProcess::queue_commands(int gpu_frame_counter) {
    int gpu_frame_id = gpu_frame_counter % _gpu_buffer_depth;
    hsa_signal_t signal;
    signal.handle = 0;
    for (auto& command : commands) {
        // Feed the last signal into the next operation
        signal = ((hsaCommand*)command[0])->execute(gpu_frame_id, signal);
    }
    final_signals[gpu_frame_id]->set_signal(&signal);
}

void hsaProcess::register_host_memory(struct Buffer* host_buffer) {
    // Since we use hsa_host_malloc for all memory with HSA configurations
    // we do not need to register the memory with the GPU drivers.
    (void)host_buffer;
}
