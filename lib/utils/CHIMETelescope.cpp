#include "CHIMETelescope.hpp"

#include <exception>           // for exception
#include <stdexcept>           // for runtime_error
#include <utility>             // for pair
#include <vector>              // for vector

#include "ICETelescope.hpp"    // for ice_stream_id_t, ice_encode_stream_id, ice_extract_stream_id
#include "Telescope.hpp"       // for stream_t, Telescope, REGISTER_TELESCOPE, _factory_aliasTel...
#include "geoUtil.hpp"         // for GeoFrame
#include "kotekanLogging.hpp"  // for ERROR, INFO, WARN, DEBUG2, FATAL_ERROR_NON_OO
#include "restClient.hpp"      // for restClient
#include "fmt.hpp"             // for compile_string_to_view, format, format_string
#include "json.hpp"            // for json, json_ref, basic_json, input_adapter


REGISTER_TELESCOPE(CHIMETelescope, "CHIMETelescope");

CHIMETelescope::CHIMETelescope(const kotekan::Config& config, const std::string& path) :
    ICETelescope(path, config.get<std::string>(path, "log_level"),
                 config.get_default<bool>(path, "require_eop", false),
                 config.get_default<std::string>(path, "eop_updatable_config", ""),
                 grid_frame_from_config(config, path),
                 config.get_default<double>(path, "feed_sep_EW", 22.0),
                 config.get_default<double>(path, "feed_sep_NS", 0.3048)) {
    INFO("Building CHIMETelescope");

    // This is always 1 for CHIME
    _num_freq_per_stream = 1;

    set_sampling_params(800.0, 2048, 2);

    // Get the GPS time, either from the config or fpga_master. gps_host_info
    // names a sibling block (typically /fpga_controller) holding ``host``,
    // ``port``, and optionally ``timing_endpoint`` (the default for
    // gps_endpoint). Required when query_gps is true.
    bool require_gps = config.get_default<bool>(path, "require_gps", true);
    _query_gps = config.get_default<bool>(path, "query_gps", false);
    std::string default_gps_endpoint = "/get-frame0-time";
    if (config.exists(path, "gps_host_info")) {
        const std::string ref = config.get<std::string>(path, "gps_host_info");
        _gps_host = config.get_default<std::string>(ref, "host", "10.1.13.1");
        _gps_port = config.get_default<uint32_t>(ref, "port", 54321);
        if (config.exists(ref, "timing_endpoint"))
            default_gps_endpoint = config.get<std::string>(ref, "timing_endpoint");
    } else if (_query_gps) {
        FATAL_ERROR_NON_OO("CHIMETelescope ({}): query_gps is true but gps_host_info is not set; "
                           "point it at the FPGA controller block (e.g. /fpga_controller).",
                           path);
    }
    _gps_endpoint = config.get_default<std::string>(path, "gps_endpoint", default_gps_endpoint);
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

station_id_t CHIMETelescope::element_index_to_station_id(uint64_t el_idx, ElementOrder ord) const {
    if (ord == ElementOrder::CHIMECorrelator) {
        return el_idx;
    } else if (ord == ElementOrder::CHIMECylinder) {
        return el_idx;
    } else if (ord == ElementOrder::CHIMEBeamformer) {
        const uint64_t polarization = el_idx / 1024;
        const uint64_t cylinder = (el_idx % 1024) / 256;
        const uint64_t dish = (el_idx % 1024) % 256;
        return encode_station_id(cylinder, polarization, dish);
    }

    FATAL_ERROR("Cannot handle element order {}.", ord);
}

uint64_t CHIMETelescope::station_id_to_element_index(station_id_t st_id, ElementOrder ord) const {
    if (ord == ElementOrder::CHIMECorrelator) {
        return st_id;
    } else if (ord == ElementOrder::CHIMECylinder) {
        return st_id;
    } else if (ord == ElementOrder::CHIMEBeamformer) {
        uint64_t cylinder, polarization, dish;
        decode_station_id(st_id, cylinder, polarization, dish);
        return polarization * 1024 + cylinder * 256 + dish;;
    }

    FATAL_ERROR("Cannot handle element order {}.", ord);
}

grid_idx_2d_t CHIMETelescope::station_id_to_grid_indices([[maybe_unused]] station_id_t st_id) const {
    
    uint64_t cylinder, polarization, dish;
    decode_station_id(st_id, cylinder, polarization, dish);

    return grid_idx_2d_t{static_cast<int32_t>(cylinder), static_cast<int32_t>(dish)};
} 

vec3d_t CHIMETelescope::station_id_to_feed_position_m([[maybe_unused]] station_id_t st_id) const {
    uint64_t cylinder, polarization, dish;
    decode_station_id(st_id, cylinder, polarization, dish);

    return vec3d_t{cylinder * _feed_sep_x_m, dish * _feed_sep_y_m, 0.0};
}

void CHIMETelescope::decode_station_id(station_id_t st_id, uint64_t& cylinder, uint64_t& polarization, uint64_t& dish) {
    cylinder = st_id / 512;
    polarization = (st_id % 512) / 256;
    dish = (st_id % 512) % 256;
}

station_id_t CHIMETelescope::encode_station_id(uint64_t cylinder, uint64_t polarization, uint64_t dish) {
    return cylinder * 512 + polarization * 256 + dish;
}

GeoFrame CHIMETelescope::grid_frame_from_config(const kotekan::Config& config,
                                                const std::string& path) {

    vec3d_t grid_x_axis = config.get_default<vec3d_t>(path, "grid_x_axis", {1.0, 0.0, 0.0});
    vec3d_t grid_y_axis = config.get_default<vec3d_t>(path, "grid_y_axis", {0.0, 1.0, 0.0});
    vec3d_t grid_z_axis = vec3d_cross(grid_x_axis, grid_y_axis);

    GeoFrame grid_frame(config.get<std::string>(path, "log_level"), "grid",
                        config.get_default<double>(path, "inst_lat", 0.0),
                        config.get_default<double>(path, "inst_long", 0.0), {0.0, 0.0, 0.0},
                        grid_x_axis, grid_y_axis, grid_z_axis);

    return grid_frame;
}
