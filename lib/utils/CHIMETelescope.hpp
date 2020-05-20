#ifndef CHIME_TELESCOPE_HPP
#define CHIME_TELESCOPE_HPP

#include "Config.hpp" // for Config
#include "ICETelescope.hpp"

#include <stdint.h> // for int32_t, uint32_t
#include <string>   // for string
#include <time.h>


/**
 * @brief A telescope class to represent CHIME.
 **/
class CHIMETelescope : public ICETelescope {
public:
    CHIMETelescope(const kotekan::Config& config, const std::string& path);
};


#endif // CHIME_TELESCOPE_HPP