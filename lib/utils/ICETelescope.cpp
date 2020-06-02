#include "ICETelescope.hpp"

#include "Telescope.hpp"
#include "chimeMetadata.h"
#include "visUtil.hpp"

#include <cmath>
#include <fmt.hpp>


REGISTER_TELESCOPE(ICETelescope, "ice");

#define GIGA 1000000000

ICETelescope::ICETelescope(const kotekan::Config& config, const std::string& path) :
    Telescope(config.get<std::string>(path, "log_level")) {

    // TODO: rename this parameter to `num_freq_per_stream` in the config
    _num_freq_per_stream = config.get<uint32_t>(path, "num_local_freq");

    set_sampling_params(config, path);
    set_gps(config, path);
}


void ICETelescope::set_sampling_params(const kotekan::Config& config, const std::string& path) {
    set_sampling_params(config.get<double>(path, "sampling_rate"),
                        config.get<uint32_t>(path, "frame_length"),
                        config.get<uint8_t>(path, "nyquist_zone"));
}

void ICETelescope::set_sampling_params(double sample_rate, uint32_t length, uint8_t zone) {
    // Set the physical frequency of id=0, and the spacing, taking into account
    // the aliasing of each Nyquist zone
    freq0_MHz = (zone / 2) * sample_rate;
    df_MHz = (zone % 2 ? 1 : -1) * sample_rate / length;
    nfreq = length / 2;

    // TODO: revisit this if we think the length might ever not be an integer
    // number of ns
    dt_ns = 1e3 / sample_rate * length;
}

void ICETelescope::set_gps(const kotekan::Config& config, const std::string& path) {

    auto require_gps = config.get_default<bool>(path, "require_gps", false);

    if (!config.exists("/", "gps_time")) {
        if (require_gps) {
            FATAL_ERROR("GPS time section is required, but was not found in the config.");
        } else {
            WARN("No GPS time section found. Ignoring.");
        }
        return;
    }

    if (config.exists("/gps_time", "error")) {
        auto error_message = config.get<std::string>("/gps_time", "error");

        if (require_gps) {
            FATAL_ERROR("Required GPS time lookup failed with reason: \n {:s}\n", error_message);
        } else {
            WARN("GPS time lookup failed with reason: \n {:s}\n", error_message);
        }
        return;
    }

    if (!config.exists("/gps_time", "frame0_nano")) {
        if (require_gps) {
            FATAL_ERROR("GPS frame0 time is required, but was not found in the config.");
        } else {
            WARN("No GPS frame0 time found. Ignoring.");
        }
        return;
    }

    time0_ns = config.get<uint64_t>("/gps_time", "frame0_nano");
    gps_enabled = true;
}

freq_id_t ICETelescope::to_freq_id(stream_t stream, uint32_t ind) const {
    (void)ind;

    auto stream_id = ice_extract_stream_id(stream);

    // CHIME bin number
    return stream_id.crate_id * 16 + stream_id.slot_id + stream_id.link_id * 32
           + stream_id.unused * 256;

    /*
    // 16 element version
    return stream_id.link_id + index * 8;

    // Pathfinder version
    return stream_id.slot_id + stream_id.link_id * 16 + ind * 128;
    */
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
    return true;
}

uint64_t ICETelescope::seq_length_nsec() const {
    return dt_ns;
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