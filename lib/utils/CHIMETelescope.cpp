#include "CHIMETelescope.hpp"

#include "ICETelescope.hpp"   // for ice_stream_id_t, ice_encode_stream_id, ice_extract_stream_id
#include "Telescope.hpp"      // for freq_id_t, REGISTER_TELESCOPE, _factory_aliasTelescope
#include "kotekanLogging.hpp" // for ERROR, DEBUG2, WARN
#include "restClient.hpp"

#include "fmt.hpp"  // for format
#include "json.hpp" // for json, basic_json<>::object_t, basic_json

#include <cstdint>   // for uint64_t
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <utility>   // for tuple_element<>::type
#include <vector>    // for vector


REGISTER_TELESCOPE(CHIMETelescope, "CHIMETelescope");

CHIMETelescope::CHIMETelescope(const kotekan::Config& config, const std::string& path) :
    ICETelescope(config.get<std::string>(path, "log_level")) {

    // This is always 1 for CHIME
    _num_freq_per_stream = 1;

    set_sampling_params(800.0, 2048, 2);

    // Get the GPS time, either from the config or fpga_master
    bool require_gps = config.get_default<bool>(path, "require_gps", true);
    _query_gps = config.get_default<bool>(path, "query_gps", false);
    _gps_host = config.get_default<std::string>(path, "gps_host", "10.1.13.1");
    _gps_port = config.get_default<uint32_t>(path, "gps_port", 54321);
    _gps_endpoint = config.get_default<std::string>(path, "gps_endpoint", "/get-frame0-time");
    if (_query_gps)
        set_gps(_gps_host, _gps_port, _gps_endpoint);
    if (!gps_enabled)
        set_gps(config);

    _require_frequency_map = config.get_default<bool>(path, "require_frequency_map", false);
    _allow_default_frequency_map =
        config.get_default<bool>(path, "allow_default_frequency_map", true);

    // Get the frequency map, either from config or fpga_master
    _query_frequency_map = config.get_default<bool>(path, "query_frequency_map", false);
    _frequency_map_host = config.get_default<std::string>(path, "frequency_map_host", "10.1.13.1");
    _frequency_map_port = config.get_default<uint32_t>(path, "frequency_map_port", 54321);
    _frequency_map_endpoint =
        config.get_default<std::string>(path, "frequency_map_endpoint", "/get-frequency-map");

    nlohmann::json fpga_freq_map_json;
    if (_query_frequency_map)
        fpga_freq_map_json =
            fetch_frequency_map(_frequency_map_host, _frequency_map_port, _frequency_map_endpoint);
    else
        fpga_freq_map_json = config.get_default<nlohmann::json>("/", "fpga_frequency_map", {});

    set_frequency_map(fpga_freq_map_json);

    if (require_gps && !gps_enabled) {
        throw std::runtime_error("The system requires a GPS time, but none was found.");
    }
}

nlohmann::json CHIMETelescope::fetch_frequency_map(const std::string& host, const uint32_t port,
                                                   const std::string& path) {
    INFO("Requesting frequency map from server: {:s}:{:d}{:s} This might take some time...", host,
         port, path);
    auto reply =
        restClient::instance().make_request_blocking(path, {{"format", "s:b"}}, host, port, 0, 30);

    if (!reply.first) {
        WARN("Failed to get frequency map from {:s}:{:d}{:s}");
        return nlohmann::json();
    }

    auto json_reply = nlohmann::json::parse(reply.second);
    return json_reply;
}

void CHIMETelescope::set_frequency_map(nlohmann::json& fpga_freq_map_json) {
    if (fpga_freq_map_json.empty()) {
        if (_require_frequency_map) {
            std::string msg = fmt::format("Frequency map required but not found");
            ERROR("{:s}", msg);
            throw std::runtime_error(msg);
        }

        if (!_allow_default_frequency_map) {
            WARN("The generation of a default frequency map is disabled, "
                 "calls to convert stream IDs will generate an exception");
            return;
        }

        WARN("No frequency map provided, generating default map");

        // Default FPGA mapping will be incorrect for any dynamic map
        // This function generates the default FPGA table mapping in the CHIME
        // configuration 4 pairs of ICEBoards, won't work for any other config.
        ice_stream_id_t stream_id = {0, 0, 0, 0};
        for (uint32_t i = 0; i < nfreq; ++i) {

            stream_id.slot_id = i % 16;
            stream_id.link_id = (i / 32) % 8;
            stream_id.crate_id = (i / 16) % 2;
            stream_id.unused = (i / 256) % 4;

            DEBUG2("Adding table entry: stream_id: {:d} : crate: {:d}, "
                   "slot: {:d}, link: {:d}, unused: {:d} = {:d}",
                   ice_encode_stream_id(stream_id).id, stream_id.crate_id, stream_id.slot_id,
                   stream_id.link_id, stream_id.unused, i);
            frequency_table[ice_encode_stream_id(stream_id).id] = i;
        }
    } else {
        // Extract the frequency map
        std::map<std::string, std::vector<int>> fpga_freq_map;
        try {
            fpga_freq_map =
                fpga_freq_map_json.at("fmap").get<std::map<std::string, std::vector<int>>>();
        } catch (std::exception const& e) {
            std::string msg = fmt::format("Could not read the fmap table: {:s}", e.what());
            ERROR("{:s}", msg);
            throw std::runtime_error(msg);
        }

        // We do not use the FPGA table directly because we do a final corner-turn
        // to generate streams of data with 1 frequency per-stream instead of the
        // 4 frequencies per-stream the FPGAs send on each of the 4 links.
        ice_stream_id_t fpga_stream_id = {0, 0, 0, 0};
        ice_stream_id_t post_shuffle_id = {0, 0, 0, 0};
        for (auto& [fpga_encoded_stream_id, bin_numbers] : fpga_freq_map) {
            int index = 0;
            for (auto& bin : bin_numbers) {
                fpga_stream_id = ice_extract_stream_id({std::stoull(fpga_encoded_stream_id)});

                // We assume that all 4 crate pairs are generating the same map
                if (fpga_stream_id.crate_id != 0 && fpga_stream_id.crate_id != 1)
                    break;

                post_shuffle_id.crate_id = fpga_stream_id.crate_id;
                post_shuffle_id.slot_id = fpga_stream_id.slot_id;
                post_shuffle_id.link_id = fpga_stream_id.link_id;
                post_shuffle_id.unused = index++;

                DEBUG2("Adding table entry: stream_id: {:d} : crate: {:d}, "
                       "slot: {:d}, link: {:d}, unused: {:d} = {:d}",
                       ice_encode_stream_id(post_shuffle_id).id, post_shuffle_id.crate_id,
                       post_shuffle_id.slot_id, post_shuffle_id.link_id, post_shuffle_id.unused,
                       bin);
                frequency_table[ice_encode_stream_id(post_shuffle_id).id] = bin;
            }
        }
        INFO("Using custom frequency map.");
    }
}

freq_id_t CHIMETelescope::to_freq_id(stream_t stream, uint32_t /* ind */) const {
    try {
        return frequency_table.at(stream.id);
    } catch (std::exception const& e) {
        ice_stream_id_t stream_id = ice_extract_stream_id(stream);
        std::string msg = fmt::format("Failed to lookup stream_id: {:d} : crate: {:d}, "
                                      "slot: {:d}, link: {:d}, unused: {:d}",
                                      stream.id, stream_id.crate_id, stream_id.slot_id,
                                      stream_id.link_id, stream_id.unused);
        ERROR("{}", msg);
        throw std::runtime_error(msg);
    }
}
