#include "CHORDTelescope.hpp"

#include "Telescope.hpp"        // for REGISTER_TELESCOPE, Telescope, ...
#include "kotekanLogging.hpp"   // for WARN, INFO, FATAL_ERROR
#include "restClient.hpp"       // for restClient
#include "configUpdater.hpp"   // for ConfigUpdater

#include "fmt.hpp"  // for format
#include "json.hpp" //for basic_json, basic_json<>::object_t, basic_jason<>::value_type

#include <cstdint>      // for uint64_t  TODO: why not stdint.h?
#include <exception>    // for exception
#include <math.h>       // for abs
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for runtime_error, invalid_argument
#include <vector>       // for vector


REGISTER_TELESCOPE(CHORDTelescope, "CHORDTelescope");

#define GIGA 1000000000

CHORDTelescope::CHORDTelescope(const kotekan::Config& config,
                               const std::string& path) :
    Telescope(config.get<std::string>(path, "log_level")) {

    bool require_gps = config.get_default<uint32_t>(path, "require_gps", false);
    _query_gps = config.get_default<bool>(path, "query_gps", false);
    _gps_host = config.get_default<std::string>(path, "gps_host", "127.0.0.1");
    _gps_port = config.get_default<uint32_t>(path, "gps_port", 54321);
    _gps_endpoint = config.get_default<std::string>(path, "gps_endpoint", "/get-frame0-time");
    if (_query_gps)
        set_gps(_gps_host, _gps_port, _gps_endpoint);
    if (!gps_enabled)
        set_gps(config);

    if (require_gps && !gps_enabled) {
        throw std::runtime_error("The system requires a GPS time, but none was found.");
    }

    _inst_long_deg = config.get<double>(path, "inst_long_deg");
    _inst_lat_deg = config.get<double>(path, "inst_lat_deg");
    _inst_alt_deg = config.get<double>(path, "inst_alt_deg");
    INFO("Telescope configured with longitude: {:f} deg", _inst_long_deg);
    INFO("Telescope configured with latitude:  {:f} deg", _inst_lat_deg);
    INFO("Telescope configured with altitude: {:f} deg", _inst_alt_deg);
    INFO("Telescope targetting declination: {:f} deg",
            _inst_lat_deg + 90-_inst_alt_deg);
    if (gps_enabled)
        INFO("Telescope configured with GPS time0: {:d} ns", time0_ns);
    else
        INFO("Telescope GPS time not enabled.");

    std::vector<double> orientation_vec = config.get<std::vector<double>>(path,
                                                        "inst_orientation");
    if (orientation_vec.size() != 9){
        throw std::runtime_error("The instrument orienation must be 9 elements specifying a 3x3 matrix.");
    }

    for(int i=0; i < 3; i++)
        for(int j=0; j<3; j++)
            _inst_orientation[i][j] = orientation_vec[3*i+j];

    _dish_positions = config.get<std::vector<std::array<double, 3>>>(path,
                                                        "dish_positions");

    using namespace std::placeholders;

    kotekan::configUpdater::instance().subscribe(
            config.get<std::string>(path, "updatable_config"),
            std::bind(&CHORDTelescope::receive_ut1_updates, this, _1));
    
}


void CHORDTelescope::set_gps(const kotekan::Config& config) {
    if (!config.exists("/", "gps_time")) {
        WARN("No GPS time section found. Ignoring.");
        return;
    }

    if (config.exists("/gps_time", "error")) {
        auto error_message = config.get<std::string>("/gps_time", "error");
        WARN("GPS time lookup failed with reason: \n {:s}\n", error_message);
        return;
    }

    if(!config.exists("/gps_time", "frame0_nano")) {
        WARN("No GPS frame0 time found in config.");
        return;
    }

    time0_ns = config.get<uint64_t>("/gps_time", "frame0_nano");
    gps_enabled = true;
}

void CHORDTelescope::set_gps(const std::string& host, const uint32_t port,
                             const std::string& path) {

    INFO("Requesting GPS time from server: {:s}.{:d}{:s} This might take some time...",
            host, port, path);
    auto reply = restClient::instance().make_request_blocking(path, {}, host,
                                                              port, 0, 30);

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
    }

    time0_ns = json_reply["frame0_nano"].get<uint64_t>();
    INFO("GPS frame0 time set to {:d}", time0_ns);
    gps_enabled = true;
}
    
bool CHORDTelescope::receive_ut1_updates(nlohmann::json& json) {
    std::lock_guard<std::mutex> lock(_ut1_lock);
    try {
        _dut1 = json.at("DUT1").get<double>();
        _dtai = json.at("DTAI").get<double>();
    } catch (std::exception& e) {
        WARN("CHORDTelescope failed to read DUT1 update: {:s}", e.what());
        return false;
    }

    return true;
}

timespec CHORDTelescope::to_time(uint64_t seq) const {
    auto time_ns = time0_ns + seq * dt_ns;
    return {(time_t)(time_ns / GIGA), (long)(time_ns % GIGA)};
}

uint64_t CHORDTelescope::to_seq(timespec time) const {
    return (time.tv_sec * GIGA + time.tv_nsec - time0_ns) / dt_ns;
}

bool CHORDTelescope::gps_time_enabled() const {
    return gps_enabled;
}

uint64_t CHORDTelescope::seq_length_nsec() const {
    return dt_ns;
}

double CHORDTelescope::get_inst_long_deg() const {
    return _inst_long_deg;
}

double CHORDTelescope::get_inst_lat_deg() const {
    return _inst_lat_deg;
}

std::array<double, 3> CHORDTelescope::get_sky_vec_in_dish_coords(
        double ra, double dec, double era) const {

    double phi = M_PI * (ra - era);
    double theta = M_PI * (90 - dec);

    // unit vector pointing to ra/dec in spherical coordinates
    // fixed to the Earth.  phi=0 ~ Greenwich
    double n_geocen[3] = {cos(phi) * sin(theta),
                          sin(phi) * sin(theta),
                          cos(theta)};

    double clon = cos(M_PI * _inst_long_deg);
    double slon = sin(M_PI * _inst_long_deg);
    double clat = cos(M_PI * (90 - _inst_lat_deg));
    double slat = sin(M_PI * (90 - _inst_lat_deg));

    double R_geocen_to_local[3][3] = {
        {-slon,       clon,      0},      // x: local East  in topocen coords.
        {-clon*clat, -slon*clat, slat},   // y: local North in topocen coords.
        { clon*slat,  slon*slat, clat}};  // z: local Up in topocen coords.

    double n_local_geoid[3] = {0, 0, 0};
    for(int i = 0; i<3; i++)
        for(int j = 0; j<3; j++) 
            n_local_geoid[i] = R_geocen_to_local[i][j] * n_geocen[j];

    std::array<double, 3> n_local = {0, 0, 0};

    for(int i = 0; i<3; i++)
        for(int j = 0; j<3; j++) 
            n_local[i] = _inst_orientation[i][j] * n_local_geoid[j];

    return n_local;
}

double CHORDTelescope::get_orientation_el(int i, int j) const {
    return _inst_orientation[i][j];
}

double CHORDTelescope::get_dish_coord(int i, int j) const {
    return _dish_positions[i][j];
}

int CHORDTelescope::get_num_dishes() const {
    return _dish_positions.size();
}

double CHORDTelescope::get_dut1() const {
    std::lock_guard<std::mutex> lock(_ut1_lock);
    return _dut1;
}

double CHORDTelescope::get_dtai() const {
    std::lock_guard<std::mutex> lock(_ut1_lock);
    return _dtai;
}

//TODO: This is a stub to satisfy inheritance and should not be used.
freq_id_t CHORDTelescope::to_freq_id(stream_t stream, uint32_t ind) const {
    return 0;
}
    
//TODO: This is a stub to satisfy inheritance and should not be used.
double CHORDTelescope::to_freq(freq_id_t freq_id) const {
    return freq_id * 1.6e9/8192;
}
    
uint32_t CHORDTelescope::num_freq_per_stream() const {
//TODO: This is a stub to satisfy inheritance and should not be used.
    return 0;
}
    
//TODO: This is a stub to satisfy inheritance and should not be used.
uint32_t CHORDTelescope::num_freq() const {
    return 0;
}
    
//TODO: This is a stub to satisfy inheritance and should not be used.
double CHORDTelescope::freq_width(freq_id_t freq_id) const {
    return 0;
}
    
//TODO: This is a stub to satisfy inheritance and should not be used.
uint8_t CHORDTelescope::nyquist_zone() const {
    return 0;
}
