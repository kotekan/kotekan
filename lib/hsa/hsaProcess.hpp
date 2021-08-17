#ifndef HSA_PROCESS_H
#define HSA_PROCESS_H

#include "Config.hpp"            // for Config
#include "bufferContainer.hpp"   // for bufferContainer
#include "gpuCommand.hpp"        // for gpuCommand
#include "gpuEventContainer.hpp" // for gpuEventContainer
#include "gpuProcess.hpp"        // for gpuProcess

#include <string> // for string

class hsaProcess final : public gpuProcess {
public:
    hsaProcess(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    virtual ~hsaProcess();

protected:
    gpuCommand* create_command(const std::string& cmd_name,
                               const std::string& unique_name) override;
    gpuEventContainer* create_signal() override;
    void queue_commands(int gpu_frame_id) override;
    void register_host_memory(struct Buffer * host_buffer) override;
};

#endif // HSA_PROCESS_H
