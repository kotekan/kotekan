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

static constexpr double C = 2.99792458e8;

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

    double sampling_rate = config.get_default<double>(path, "sampling_rate_Hz",
                                                      3.2e9);
    uint64_t fft_length = config.get_default<uint64_t>(path, "fft_length",
                                                       16384);
    dt_ns = (GIGA * fft_length) / sampling_rate;

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

    double deg2rad = M_PI/180;
    double cos_lon = cos(deg2rad * _inst_long_deg);
    double sin_lon = sin(deg2rad * _inst_long_deg);
    double cos_lat = cos(deg2rad * _inst_lat_deg);
    double sin_lat = sin(deg2rad * _inst_lat_deg);

    // Topocentric X (East) in Geocentric (Earth-centered, Earth-fixed) coords
    _R_geo_to_topo[0][0] = -sin_lon;
    _R_geo_to_topo[0][1] =  cos_lon;
    _R_geo_to_topo[0][2] =  0.0;

    // Topocentric Y (North) in Geocentric (Earth-centered, Earth-fixed) coords
    _R_geo_to_topo[1][0] = -sin_lat * cos_lon;
    _R_geo_to_topo[1][1] = -sin_lat * sin_lon;
    _R_geo_to_topo[1][2] =  cos_lat;

    // Topocentric Z (Up) in Geocentric (Earth-centered, Earth-fixed) coords
    _R_geo_to_topo[2][0] =  cos_lat * cos_lon;
    _R_geo_to_topo[2][1] =  cos_lat * sin_lon;
    _R_geo_to_topo[2][2] =  sin_lat;


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

    double deg2rad = M_PI / 180;

    double phi = deg2rad * (ra - era);
    double theta = deg2rad * (90 - dec);

    // unit vector pointing to ra/dec in spherical coordinates
    // fixed to the Earth.  phi=0 ~ Greenwich
    std::array<double, 3> n_geocen= {
                          cos(phi) * sin(theta),
                          sin(phi) * sin(theta),
                          cos(theta)};

    INFO("n_geocen: {} {} {}", n_geocen[0], n_geocen[1], n_geocen[2]);

    std::array<double, 3> n_topo = geocen_vec_to_topocen_vec(n_geocen);

    return topocen_vec_to_tel_vec(n_topo);
}

std::array<double, 3> CHORDTelescope::get_pointing_vec_in_tel_coords() const {

    double deg2rad = 180/M_PI;
    std::array<double, 3> n = {0.0, cos(deg2rad * _inst_alt_deg),
                                    sin(deg2rad * _inst_alt_deg)};

    return n;
}

std::array<double, 3> CHORDTelescope::topocen_vec_to_tel_vec(
        const std::array<double, 3>& v_topocen) const {
    
    std::array<double, 3> v_tel = {0, 0, 0};
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            v_tel[i] += _inst_orientation[i][j] * v_topocen[j];

    return v_tel;
}

std::array<double, 3> CHORDTelescope::tel_vec_to_topocen_vec(
        const std::array<double, 3>& v_tel) const {
    
    std::array<double, 3> v_topocen = {0, 0, 0};
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            v_topocen[i] += _inst_orientation[j][i] * v_tel[j];

    return v_topocen;
}


std::array<double, 3> CHORDTelescope::geocen_vec_to_topocen_vec(
        const std::array<double, 3>& v_geo) const {

    std::array<double, 3> v_topo = {0, 0, 0};
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            v_topo[i] += _R_geo_to_topo[i][j] * v_geo[j];

    return v_topo;
}

std::array<double, 3> CHORDTelescope::topocen_vec_to_geocen_vec(
        const std::array<double, 3>& v_topo) const {

    std::array<double, 3> v_geo = {0, 0, 0};
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            v_geo[i] += _R_geo_to_topo[j][i] * v_topo[j];

    return v_geo;
}

std::array<double, 3> CHORDTelescope::cirs_vec_to_geocen_vec(
        const std::array<double, 3>& v_cirs, double era_deg) const {

    double deg2rad = M_PI/180;
    double cos_era = cos(deg2rad * era_deg);
    double sin_era = sin(deg2rad * era_deg);

    std::array<double, 3> v_geocen = {
         v_cirs[0] * cos_era + v_cirs[1] * sin_era,
        -v_cirs[0] * sin_era + v_cirs[1] * cos_era,
         v_cirs[2]};

    return v_geocen;
}

std::array<double, 3> CHORDTelescope::geocen_vec_to_cirs_vec(
        const std::array<double, 3>& v_geocen, double era_deg) const {

    double deg2rad = M_PI/180;
    double cos_era = cos(deg2rad * era_deg);
    double sin_era = sin(deg2rad * era_deg);

    std::array<double, 3> v_cirs = {
        v_geocen[0] * cos_era - v_geocen[1] * sin_era,
        v_geocen[0] * sin_era + v_geocen[1] * cos_era,
        v_geocen[2]};

    return v_cirs;
}

void CHORDTelescope::fringestop_phases_1d(double freq_Hz, double era_deg,
        double era_deg0, std::vector<std::complex<double>>& phases) const {

    //Take the pointing vector for the telescope (constant in time),
    //and find it in the CIRS frame at ERA0.  This is the point we are
    //attempting to stop the fringes at.
    std::array<double, 3> n_tel0 = get_pointing_vec_in_tel_coords();
    std::array<double, 3> n_topo0 = tel_vec_to_topocen_vec(n_tel0);
    std::array<double, 3> n_geo0 = topocen_vec_to_geocen_vec(n_topo0);
    std::array<double, 3> n_cirs = geocen_vec_to_cirs_vec(n_geo0, era_deg0);

    //Now, given this CIRS vector, find its components in the telescope
    //frame at the requested ERA
    std::array<double, 3> n_geo = cirs_vec_to_geocen_vec(n_cirs, era_deg);
    std::array<double, 3> n_topo = geocen_vec_to_topocen_vec(n_geo);
    std::array<double, 3> n_tel = topocen_vec_to_tel_vec(n_topo);

    // n_tel is now (at ERA) the point on the sky which will be at the
    // phase center (n_tel0) at ERA0.
   
    double wavelength_m = C / freq_Hz;

    // wavenumber for this frequency
    double k = 2*M_PI * C / freq_Hz;

    for(int i = 0; i < _dish_positions.size(); i++)
    {
        double phase = -k * (
                  _dish_positions[i][0] * (n_tel[0] - n_tel0[0])
                + _dish_positions[i][1] * (n_tel[1] - n_tel0[1])
                + _dish_positions[i][2] * (n_tel[2] - n_tel0[2]));

        phases[i] = {cos(phase) , sin(phase)};
    }
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
