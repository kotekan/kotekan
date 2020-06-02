#ifndef CHIME_TELESCOPE_HPP
#define CHIME_TELESCOPE_HPP

#include "Config.hpp" // for Config
#include "ICETelescope.hpp"

#include <stdint.h> // for int32_t, uint32_t
#include <string>   // for string
#include <time.h>


/**
 * @brief A telescope class to represent CHIME.
 *
 * @conf    require_frequency_map        Require a frequency map in the config file, or generate
 *                                       an error and exit if none exists. Default: false
 * @conf    allow_default_frequency_map  If @c require_frequency_map is set to false and
 *                                       @c allow_default_frequency_map is true allow queries from
 *                                       the default map.  If set to false, then do not generate
 *                                       a default map, and throw an exception when lookups are
 *                                       done for stream_ids.
 *                                       This can be useful for systems not expected to
 *                                       lookup stream_ids. This option has no effect if
 *                                       @c require_frequency_map is set to true.  Default: true
 **/
class CHIMETelescope : public ICETelescope {
public:
    CHIMETelescope(const kotekan::Config& config, const std::string& path);

    /// Override the default table to account for remapping and 4-way CPU shuffle
    freq_id_t to_freq_id(stream_t stream, uint32_t ind) const override;

private:
    /// Require a frequency remapping table in the config, or exit with an error.
    bool _require_frequency_map;

    /// Allow a default map to be generated if one isn't aviable in the config.
    bool _allow_default_frequency_map;

    /// Maps the post-final shuffle stream_ids to frequency bins
    std::map<uint64_t, freq_id_t> frequency_table;

    /**
     * @brief Sets the CHIME specific post 4-way CPU shuffle stream ID to
     *        frequency bin mapping table.  Uses the table in the config if one exists
     *        otherwise it generates the FPGA default mapping table.
     * @param config The current config
     * @param path The location in the config of the @c fmap table from the FPGAs
     */
    void set_frequency_map(const kotekan::Config& config, const std::string& path);
};


#endif // CHIME_TELESCOPE_HPP
