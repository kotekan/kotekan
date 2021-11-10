#include "ICETelescope.hpp"

#include "Telescope.hpp"      // for stream_t, freq_id_t, REGISTER_TELESCOPE, Telescope, _facto...
#include "kotekanLogging.hpp" // for WARN, INFO, FATAL_ERROR
#include "restClient.hpp"     // for restClient

#include "fmt.hpp"  // for format
#include "json.hpp" // for basic_json, basic_json<>::object_t, basic_json<>::value_type

#include <cstdint>   // for uint64_t
#include <exception> // for exception
#include <math.h>    // for abs
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error, invalid_argument
#include <vector>    // for vector


REGISTER_TELESCOPE(ICETelescope, "ICETelescope");

#define GIGA 1000000000

ICETelescope::ICETelescope(const kotekan::Config& config, const std::string& path) :
    Telescope(config.get<std::string>(path, "log_level")) {

    // TODO: rename this parameter to `num_freq_per_stream` in the config
    _num_freq_per_stream = config.get_default<uint32_t>(path, "num_local_freq", 128);

    set_sampling_params(config, path);

    bool require_gps = config.get_default<bool>(path, "require_gps", false);
    _query_gps = config.get_default<bool>(path, "query_gps", false);
    _gps_host = config.get_default<std::string>(path, "gps_host", "127.0.0.1");
    _gps_port = config.get_default<uint32_t>(path, "gps_host", 54321);
    _gps_endpoint = config.get_default<std::string>(path, "gps_endpoint", "/get-frame0-time");
    if (_query_gps)
        set_gps(_gps_host, _gps_port, _gps_endpoint);
    if (!gps_enabled)
        set_gps(config);

    if (require_gps && !gps_enabled) {
        throw std::runtime_error("The system requires a GPS time, but none was found.");
    }
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
        case 8: // 16 ICEBoards (256 elements) e.g. Pathfinder/HIRAX-128
            return stream_id.slot_id + stream_id.link_id * 16 + ind * 128;
        case 16: // 8 ICEBoards (128 elements) e.g. Allenby
            // TODO: Check this mapping
            return stream_id.slot_id + stream_id.link_id * 32 + ind * 64;
        case 128: // 1 ICEBoard (16 elements) e.g. ARO, Synthesis telescope
            return stream_id.link_id + ind * 8;
        default:
            FATAL_ERROR("No known frequency mapping for num_freq_per_stream = {:d}",
                        _num_freq_per_stream);
            return 0;
    }
}


double ICETelescope::to_freq(freq_id_t freq_id) const {

    if (freq_id >= nfreq) {
        throw std::invalid_argument(
            fmt::format("Invalid frequency ID={}, accepted ranged 0 <= id < {}", freq_id, nfreq));
    }

    return freq0_MHz + freq_id * df_MHz;
}


uint32_t ICETelescope::num_freq_per_stream() const {
    return _num_freq_per_stream;
}

uint32_t ICETelescope::num_freq() const {
    return nfreq;
}

double ICETelescope::freq_width(freq_id_t freq_id) const {

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
