#include "CHIMETelescope.hpp"

#include "ICETelescope.hpp"
#include "Telescope.hpp"


REGISTER_TELESCOPE(CHIMETelescope, "chime");


CHIMETelescope::CHIMETelescope(const kotekan::Config& config, const std::string& path) :
    ICETelescope(config.get<std::string>(path, "log_level")) {

    // TODO: rename this parameter to `num_freq_per_stream` in the config
    _num_freq_per_stream = config.get<uint32_t>(path, "num_local_freq");

    set_sampling_params(800.0, 2048, 2);
    set_gps(config, path);
}
