#ifndef HSA_PROCESS_H
#define HSA_PROCESS_H

#include <condition_variable>
#include <mutex>
#include <string>          // for string
#include <thread>

#include "Config.hpp"
#include "StageFactory.hpp"
#include "gpuProcess.hpp"  // for gpuProcess
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_finalize.h"
#include "hsaCommand.hpp"
#include "hsaDeviceInterface.hpp"
#include "hsaEventContainer.hpp"

class gpuCommand;
class gpuEventContainer;
namespace kotekan {
class Config;
class bufferContainer;
}  // namespace kotekan

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
};

#endif // HSA_PROCESS_H
