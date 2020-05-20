#include "CHIMETelescope.hpp"

#include "ICETelescope.hpp"
#include "Telescope.hpp"


REGISTER_TELESCOPE(CHIMETelescope, "chime");


CHIMETelescope::CHIMETelescope(const kotekan::Config& config, const std::string& path) :
    ICETelescope(config.get<std::string>(path, "log_level")) {
    set_sampling_params(800.0, 2048, 2);
    set_gps({0, 0});
}
