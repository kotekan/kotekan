#ifndef CL_PROCESS_H
#define CL_PROCESS_H

#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b)&0x0F)

#include "clCommand.hpp"
#include "clDeviceInterface.hpp"
#include "clEventContainer.hpp"
#include "gpuProcess.hpp"

class clProcess final : public gpuProcess {
public:
    clProcess(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& buffer_container);
    virtual ~clProcess();

    gpuCommand* create_command(const std::string& cmd_name,
                               const std::string& unique_name) override;
    gpuEventContainer* create_signal() override;
    void queue_commands(int gpu_frame_id) override;

    clDeviceInterface* device;
};

#endif // CL_PROCESS_H
