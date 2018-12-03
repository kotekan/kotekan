#include "hsaCorrelatorSubframeCommand.hpp"

hsaCorrelatorSubframeCommand::hsaCorrelatorSubframeCommand(
                            Config& config, const string &unique_name,
                            bufferContainer& host_buffers, hsaDeviceInterface& device,
                            const string &default_kernel_command,
                            const string &default_kernel_file_name) :
    hsaCommand(config, unique_name, host_buffers, device,
               default_kernel_command,default_kernel_file_name) {

    _sub_frame_samples = config.get<uint32_t>(unique_name, "sub_frame_samples");
    _sub_frame_index = config.get_default<uint32_t>(unique_name, "sub_frame_index", 0);
    INFO("sub_frame_index: %d", _sub_frame_index);
    _num_sub_frames = config.get<uint32_t>(unique_name, "num_sub_frames");
}