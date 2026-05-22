#include "ICETelescope.hpp"

#include <cmath>               // for abs
#include <stdexcept>           // for invalid_argument, runtime_error

#include "Telescope.hpp"       // for Telescope, stream_t, freq_id_t, REGISTER_TELESCOPE, _facto...
#include "geoUtil.hpp"         // for GeoFrame
#include "kotekanLogging.hpp"  // for WARN, INFO, FATAL_ERROR, FATAL_ERROR_NON_OO
#include "restClient.hpp"      // for restClient
#include "visUtil.hpp"         // for input_ctype
#include "fmt.hpp"             // for compile_string_to_view, format, format_string
#include "json.hpp"            // for basic_json, input_adapter, json


REGISTER_TELESCOPE(ICETelescope, "ICETelescope");

#define GIGA 1000000000

ICETelescope::ICETelescope(const kotekan::Config& config, const std::string& path) :
    Telescope(path, config.get<std::string>(path, "log_level"),
              config.get_default<bool>(path, "require_eop", false),
              config.get_default<std::string>(path, "eop_updatable_config", ""),
              grid_frame_from_config(config, path),
              config.get_default<double>(path, "feed_sep_EW", 0.0),
              config.get_default<double>(path, "feed_sep_NS", 0.0)) {
    INFO("Building ICETelescope");

    // TODO: rename this parameter to `num_freq_per_stream` in the config
    _num_freq_per_stream = config.get_default<uint32_t>(path, "num_local_freq", 128);

    set_sampling_params(config, path);

    bool require_gps = config.get_default<bool>(path, "require_gps", false);
    _query_gps = config.get_default<bool>(path, "query_gps", false);
    // gps_host_info names a sibling block (typically /fpga_controller)
    // holding ``host``, ``port``, and optionally ``timing_endpoint`` (the
    // default for gps_endpoint). Required when query_gps is true.
    std::string default_gps_endpoint = "/get-frame0-time";
    if (config.exists(path, "gps_host_info")) {
        const std::string ref = config.get<std::string>(path, "gps_host_info");
        _gps_host = config.get<std::string>(ref, "host");
        _gps_port = config.get_default<uint32_t>(ref, "port", 54321);
        if (config.exists(ref, "timing_endpoint"))
            default_gps_endpoint = config.get<std::string>(ref, "timing_endpoint");
    } else if (_query_gps) {
        FATAL_ERROR_NON_OO("ICETelescope ({}): query_gps is true but gps_host_info is not set; "
                           "point it at the FPGA controller block (e.g. /fpga_controller).",
                           path);
    }
    _gps_endpoint = config.get_default<std::string>(path, "gps_endpoint", default_gps_endpoint);
    if (_query_gps)
        set_gps(_gps_host, _gps_port, _gps_endpoint);
    if (!gps_enabled)
        set_gps(config);

    if (require_gps && !gps_enabled) {
        throw std::runtime_error("The system requires a GPS time, but none was found.");
    }

    _num_elements = config.get<uint64_t>(path, "num_elements");
    const auto input_tuple = ICETelescope::parse_reorder_default(config, path, _num_elements);
    _input_reorder = std::get<0>(input_tuple);
    _input_map = std::get<1>(input_tuple);
    if (_input_reorder.size() != _num_elements) {
        FATAL_ERROR("input_reorder has size {:d}, should be num_elements = {:d}",
                _input_reorder.size(), _num_elements);
    }
    if (_input_reorder.size() != _num_elements) {
        FATAL_ERROR("input_map (from input_reorder) has size {:d}, should be num_elements = {:d}",
                _input_map.size(), _num_elements);
    }

    _correlator_stations = invert_reorder_table(_input_reorder);
}


void ICETelescope::set_sampling_params(const kotekan::Config& config, const std::string& path) {
    set_sampling_params(config.get_default<double>(path, "sampling_rate", 800.0),
                        config.get_default<uint32_t>(path, "fft_length", 2048),
                        config.get_default<uint8_t>(path, "nyquist_zone", 2));
}

void ICETelescope::set_sampling_params(double sample_rate, uint32_t fft_length, uint8_t zone) {
    // Set the physical frequency of id=0, and the spacing, taking into account
    // the aliasing of each Nyquist zone
    freq0_MHz = (zone / 2) * sample_rate;
    df_MHz = (zone % 2 ? 1 : -1) * sample_rate / fft_length;
    nfreq = fft_length / 2;
    ny_zone = zone;

    // TODO: revisit this if we think the length might ever not be an integer
    // number of ns
    dt_ns = 1e3 / sample_rate * fft_length;
}

void ICETelescope::set_gps(const kotekan::Config& config) {
    if (!config.exists("/", "gps_time")) {
        WARN("No GPS time section found. Ignoring.");
        return;
    }

    if (config.exists("/gps_time", "error")) {
        auto error_message = config.get<std::string>("/gps_time", "error");
        WARN("GPS time lookup failed with reason: \n {:s}\n", error_message);
        return;
    }

    if (!config.exists("/gps_time", "frame0_nano")) {
        WARN("No GPS frame0 time found in config.");
        return;
    }

    time0_ns = config.get<uint64_t>("/gps_time", "frame0_nano");
    gps_enabled = true;
}

void ICETelescope::set_gps(const std::string& host, const uint32_t port, const std::string& path) {

    INFO("Requesting GPS time from server: {:s}:{:d}{:s} This might take some time...", host, port,
         path);
    auto reply = restClient::instance().make_request_blocking(path, {}, host, port, 0, 30);

    if (!reply.first) {
        WARN("Failed to get GPS time, using system time");
        return;
    }

    auto json_reply = nlohmann::json::parse(reply.second);

    if (json_reply.count("error") == 1) {
        std::string error_message = json_reply["error"];
        WARN("Error returned by GPS server, error: {:s}", error_message);
        return;
    }

    if (json_reply.count("frame0_nano") == 0) {
        WARN("No `frame0_nano` value returned by GPS server, the server reply was: {:s} - {:s}",
             reply.second, json_reply.dump());
        return;
    }

    time0_ns = json_reply["frame0_nano"].get<uint64_t>();
    INFO("GPS frame0 time set to {:d}", time0_ns);
    gps_enabled = true;
}

freq_id_t ICETelescope::to_freq_id(stream_t stream, uint32_t ind) const {
    (void)ind;

    auto stream_id = ice_extract_stream_id(stream);

    // The default mapping is directly related to the number of ICEBoards in the system,
    // and as a result the number of frequencies in each data stream, so we can select on that.
    switch (_num_freq_per_stream) {
        case 1: // 128 ICEBoards (2048 elements) e.g. CHIME
            return stream_id.crate_id * 16 + stream_id.slot_id + stream_id.link_id * 32
                   + stream_id.unused * 256;
        case 4: // 32 ICEBoards (512 elements) e.g. HIRAX-256
            return stream_id.slot_id + stream_id.crate_id * 16 + stream_id.link_id * 32 + ind * 256;
        case 8: // 16 ICEBoards (256 elements) e.g. Pathfinder, HIRAX-128, GBO, Hat Creek
            return stream_id.slot_id + stream_id.link_id * 16 + ind * 128;
        case 16: // 8 ICEBoards (128 elements) e.g. Princeton
            return stream_id.slot_id + stream_id.link_id * 8 + ind * 64;
        case 128: // 1 ICEBoard (16 elements) e.g. ARO, Synthesis telescope, etc.
            return stream_id.link_id + ind * 8;
        default:
            FATAL_ERROR("No known frequency mapping for num_freq_per_stream = {:d}",
                        _num_freq_per_stream);
            return 0;
    }
}


double ICETelescope::to_freq_MHz(freq_id_t freq_id) const {

    if (freq_id >= nfreq) {
        throw std::invalid_argument(
            fmt::format("Invalid frequency ID={}, accepted ranged 0 <= id < {}", freq_id, nfreq));
    }

    return freq0_MHz + freq_id * df_MHz;
}


size_t ICETelescope::num_freq_per_stream() const {
    return _num_freq_per_stream;
}

size_t ICETelescope::num_freq() const {
    return nfreq;
}

double ICETelescope::freq_width_MHz(freq_id_t freq_id) const {

    if (freq_id >= nfreq) {
        throw std::invalid_argument(
            fmt::format("Invalid frequency ID={}, accepted ranged 0 <= id < {}", freq_id, nfreq));
    }

    return std::abs(df_MHz);
}

timespec ICETelescope::to_time(uint64_t seq) const {
    auto time_ns = time0_ns + seq * dt_ns;
    return {(time_t)(time_ns / GIGA), (long)(time_ns % GIGA)};
}

int64_t ICETelescope::to_time_ns(uint64_t seq) const {
    int64_t time_ns = time0_ns + seq * dt_ns;
    return time_ns;
}

uint64_t ICETelescope::to_seq(timespec time) const {
    return (time.tv_sec * GIGA + time.tv_nsec - time0_ns) / dt_ns;
}


bool ICETelescope::gps_time_enabled() const {
    return gps_enabled;
}

uint64_t ICETelescope::seq_length_nsec() const {
    return dt_ns;
}

uint8_t ICETelescope::nyquist_zone() const {
    return ny_zone;
}

GeoFrame ICETelescope::grid_frame_from_config(const kotekan::Config& config,
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

station_id_t ICETelescope::element_index_to_station_id(uint64_t el_idx, ElementOrder ord) const {
    if (el_idx >= _num_elements) 
        FATAL_ERROR("Element idx {:d} >= num_elements {:d}", el_idx, _num_elements);

    if (ord == ElementOrder::CHIMECorrelator) {
        return _correlator_stations.at(el_idx);
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

uint64_t ICETelescope::station_id_to_element_index(station_id_t st_id, ElementOrder ord) const {
    if (st_id >= _num_elements) 
        FATAL_ERROR("Station ID {:d} >= num_elements {:d}", st_id, _num_elements);

    if (ord == ElementOrder::CHIMECorrelator) {
        return _input_reorder.at(st_id);
    } else if (ord == ElementOrder::CHIMECylinder) {
        return st_id;
    } else if (ord == ElementOrder::CHIMEBeamformer) {
        uint64_t cylinder, polarization, dish;
        decode_station_id(st_id, cylinder, polarization, dish);
        return polarization * 1024 + cylinder * 256 + dish;;
    }

    FATAL_ERROR("Cannot handle element order {}.", ord);
}

grid_idx_2d_t ICETelescope::station_id_to_grid_indices(station_id_t st_id) const {
    if (st_id >= _num_elements) 
        FATAL_ERROR("Station ID {:d} >= num_elements {:d}", st_id, _num_elements);
    
    uint64_t cylinder, polarization, dish;
    decode_station_id(st_id, cylinder, polarization, dish);

    return grid_idx_2d_t{static_cast<int32_t>(cylinder), static_cast<int32_t>(dish)};
} 

vec3d_t ICETelescope::station_id_to_feed_position_m(station_id_t st_id) const {
    if (st_id >= _num_elements) 
        FATAL_ERROR("Station ID {:d} >= num_elements {:d}", st_id, _num_elements);

    uint64_t cylinder, polarization, dish;
    decode_station_id(st_id, cylinder, polarization, dish);

    return vec3d_t{cylinder * _feed_sep_x_m, dish * _feed_sep_y_m, 0.0};
}


ice_stream_id_t ice_extract_stream_id(const stream_t encoded_stream_id) {
    ice_stream_id_t stream_id;

    uint16_t encoded_id = (uint16_t)(encoded_stream_id.id);

    stream_id.link_id = encoded_id & 0x000F;
    stream_id.slot_id = (encoded_id & 0x00F0) >> 4;
    stream_id.crate_id = (encoded_id & 0x0F00) >> 8;
    stream_id.unused = (encoded_id & 0xF000) >> 12;

    return stream_id;
}

stream_t ice_encode_stream_id(const ice_stream_id_t s_stream_id) {
    uint16_t stream_id;

    stream_id = (s_stream_id.link_id & 0xF) + ((s_stream_id.slot_id & 0xF) << 4)
                + ((s_stream_id.crate_id & 0xF) << 8) + ((s_stream_id.unused & 0xF) << 12);

    return {(uint64_t)stream_id};
}
void ICETelescope::decode_station_id(station_id_t st_id, uint64_t& cylinder, uint64_t& polarization, uint64_t& dish) {
    cylinder = st_id / 512;
    polarization = (st_id % 512) / 256;
    dish = (st_id % 512) % 256;
}

station_id_t ICETelescope::encode_station_id(uint64_t cylinder, uint64_t polarization, uint64_t dish) {
    return cylinder * 512 + polarization * 256 + dish;
}

std::tuple<uint32_t, uint32_t, std::string> ICETelescope::parse_reorder_single(nlohmann::json j) {
    if (!j.is_array() || j.size() != 3) {
        throw std::runtime_error("Could not parse json item for input reordering: " + j.dump());
    }

    uint32_t adc_id = j[0].get<int>();
    uint32_t chan_id = j[1].get<int>();
    std::string serial = j[2].get<std::string>();

    return std::make_tuple(adc_id, chan_id, serial);
}

std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> ICETelescope::parse_reorder(nlohmann::json& j) {

    uint32_t adc_id, chan_id;
    std::string serial;

    std::vector<uint32_t> adc_ids;
    std::vector<input_ctype> inputmap;

    if (!j.is_array()) {
        throw std::runtime_error("Was expecting list of input orders.");
    }

    for (auto& element : j) {
        std::tie(adc_id, chan_id, serial) = parse_reorder_single(element);

        adc_ids.push_back(adc_id);
        inputmap.emplace_back(chan_id, serial);
    }

    return std::make_tuple(adc_ids, inputmap);
}

std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> ICETelescope::default_reorder(size_t num_elements) {

    std::vector<uint32_t> adc_ids;
    std::vector<input_ctype> inputmap;

    for (uint32_t i = 0; i < num_elements; i++) {
        adc_ids.push_back(i);
        inputmap.emplace_back(i, "INVALID");
    }

    return std::make_tuple(adc_ids, inputmap);
}

std::tuple<std::vector<uint32_t>, std::vector<input_ctype>>
ICETelescope::parse_reorder_default(const kotekan::Config& config, const std::string& path, uint64_t num_elements) {

    try {
        nlohmann::json reorder_config = config.get<std::vector<nlohmann::json>>(path, "input_reorder");

        return parse_reorder(reorder_config);
    } catch (const std::exception& e) {
        return default_reorder(num_elements);
    }
}
std::vector<station_id_t> ICETelescope::invert_reorder_table(const std::vector<uint32_t>& input_reorder) 
{
    std::vector<station_id_t> correlator_stations(input_reorder.size());

    for (station_id_t st_id = 0; st_id < input_reorder.size(); st_id++) {
        uint32_t corr_idx = input_reorder.at(st_id);
        correlator_stations.at(corr_idx) = st_id;
    }

    return correlator_stations;
}

