#ifndef GPU_HSA_CORRELATOR_SUBFRAME_COMMAND_H
#define GPU_HSA_CORRELATOR_SUBFRAME_COMMAND_H

#include "hsaCommand.hpp"

class hsaCorrelatorSubframeCommand: public hsaCommand
{
public:
    /**
     * @brief pass through constructor which added the subframe
     *        values from the config and possibly other common
     *        variables.
     */
    hsaCorrelatorSubframeCommand(Config &config, const string &unique_name,
               bufferContainer &host_buffers, hsaDeviceInterface &device,
               const string &default_kernel_command="",
               const string &default_kernel_file_name="");
    virtual ~hsaCorrelatorSubframeCommand() = default;

protected:
    uint32_t _sub_frame_samples;
    uint32_t _sub_frame_index;
    uint32_t _num_sub_frames;
};

#endif // GPU_HSA_CORRELATOR_SUBFRAME_COMMAND_H

