#ifndef CHIME_TELESCOPE_HPP
#define CHIME_TELESCOPE_HPP

#include "Config.hpp"       // for Config
#include "ICETelescope.hpp" // for ICETelescope
#include "Telescope.hpp"    // for freq_id_t, stream_t

#include <json.hpp> // for json
#include <map>      // for map
#include <stdint.h> // for uint32_t, uint64_t
#include <string>   // for string


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
 * @conf    query_frequency_map  bool.   Should we get the frequency map from a remote source.
 * @conf    frequency_map_host   string. The frequency map server IP address.
 * @conf    frequency_map_port   uint.   The port number on the frequency map server.
 * @conf    frequency_map_endpoint string. The endpoint with the frequency map.
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

    /// Query the frequency map directly from fpga_master (only used for testing)
    bool _query_frequency_map;

    /// The frequency map server IP address
    std::string _frequency_map_host;

    /// The port number on the frequency map server.
    uint32_t _frequency_map_port;

    /// The endpoint with the frequency map.
    std::string _frequency_map_endpoint;

    /// Maps the post-final shuffle stream_ids to frequency bins
    std::map<uint64_t, freq_id_t> frequency_table;

    /**
     * @brief Fetch the frequency map from an endpoint
     *
     * @param host    String.  The IP address of the server with the map
     * @param port    Int.     The port of the server
     * @param path    String   The endpoint name. e.g. (e.g. /get-frequency-map)
     *
     * @return json object with frequency map.
     */
    nlohmann::json fetch_frequency_map(const std::string& host, const uint32_t port,
                                       const std::string& path);

    /**
     * @brief Sets the CHIME specific post 4-way CPU shuffle stream ID to
     *        frequency bin mapping table.  Uses the table in the config if one exists
     *        otherwise it generates the FPGA default mapping table.
     * @param fpga_freq_map_json  JSON The frequency map from the FPGAs
     */
    void set_frequency_map(nlohmann::json& fpga_freq_map_json);
};


#endif // CHIME_TELESCOPE_HPP
