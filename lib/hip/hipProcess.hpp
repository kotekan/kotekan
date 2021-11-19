/**
 * @file
 * @brief Stage for running a set of HIP commands
 *  - hipProcess : public gpuProcess
 */

#ifndef HIP_PROCESS_H
#define HIP_PROCESS_H

#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b)&0x0F)

#include "hipCommand.hpp"
#include "hipDeviceInterface.hpp"
#include "hipEventContainer.hpp"
#include "gpuProcess.hpp"

#include <string>

class hipProcess final : public gpuProcess {
public:
    hipProcess(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    virtual ~hipProcess();

    gpuCommand* create_command(const std::string& cmd_name,
                               const std::string& unique_name) override;
    gpuEventContainer* create_signal() override;
    void queue_commands(int gpu_frame_id) override;

    hipDeviceInterface* device;
};

#endif // HIP_PROCESS_H
