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

    freq_id_t to_freq_id(stream_t stream, uint32_t ind) const override;

private:
    // Maps the post-final shuffle stream_ids to frequency bins
    std::map<uint64_t, freq_id_t> frequency_table;

    void set_frequency_map(const kotekan::Config& config, const std::string& path);
};


#endif // CHIME_TELESCOPE_HPP