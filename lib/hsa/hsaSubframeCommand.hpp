#ifndef GPU_HSA_CORRELATOR_SUBFRAME_COMMAND_H
#define GPU_HSA_CORRELATOR_SUBFRAME_COMMAND_H

#include <stdint.h>        // for uint32_t
#include <string>          // for string

#include "hsaCommand.hpp"  // for hsaCommand

class hsaDeviceInterface;
namespace kotekan {
class Config;
class bufferContainer;
}  // namespace kotekan

class hsaSubframeCommand : public hsaCommand {
public:
    /**
     * @brief pass through constructor which added the subframe
     *        values from the config and possibly other common
     *        variables.
     */
    hsaSubframeCommand(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device,
                       const std::string& default_kernel_command = "",
                       const std::string& default_kernel_file_name = "");
    virtual ~hsaSubframeCommand() = default;

protected:
    uint32_t _sub_frame_samples;
    uint32_t _sub_frame_index;
    uint32_t _num_sub_frames;
};

#endif // GPU_HSA_CORRELATOR_SUBFRAME_COMMAND_H
