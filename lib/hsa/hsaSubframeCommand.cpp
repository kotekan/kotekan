#include "hsaSubframeCommand.hpp"
#include <exception>

hsaSubframeCommand::hsaSubframeCommand(
                            Config& config, const string &unique_name,
                            bufferContainer& host_buffers, hsaDeviceInterface& device,
                            const string &default_kernel_command,
                            const string &default_kernel_file_name) :
    hsaCommand(config, unique_name, host_buffers, device,
               default_kernel_command,default_kernel_file_name) {

    _sub_frame_index = config.get_default<uint32_t>(unique_name, "sub_frame_index", 0);
    _num_sub_frames = config.get_default<uint32_t>(unique_name, "num_sub_frames", 1);
    uint32_t samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _sub_frame_samples = samples_per_data_set / _num_sub_frames;

    if (_sub_frame_index >= _num_sub_frames) {
        throw std::runtime_error("Index cannot be larger the number of subframes");
    }

    if (samples_per_data_set % _num_sub_frames != 0) {
        throw std::runtime_error("The number of subframes must divide the number of samples");
    }

    DEBUG2("sub_frame_index: %i, num_sub_frames: %i, sub_frame_samples: %i",
            _sub_frame_index, _num_sub_frames,_sub_frame_samples);
}