#include "hsaSubframeCommand.hpp"

#include "Config.hpp"         // for Config
#include "kotekanLogging.hpp" // for DEBUG2

#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

hsaSubframeCommand::hsaSubframeCommand(Config& config, const std::string& unique_name,
                                       bufferContainer& host_buffers, hsaDeviceInterface& device,
                                       const std::string& default_kernel_command,
                                       const std::string& default_kernel_file_name) :
    hsaCommand(config, unique_name, host_buffers, device, default_kernel_command,
               default_kernel_file_name) {

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

    DEBUG2("sub_frame_index: {:d}, num_sub_frames: {:d}, sub_frame_samples: {:d}", _sub_frame_index,
           _num_sub_frames, _sub_frame_samples);
}
