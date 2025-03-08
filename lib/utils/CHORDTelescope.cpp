#include "CHORDTelescope.hpp"

#include "Telescope.hpp"        // for REGISTER_TELESCOPE, Telescope, ...
#include "kotekanLogging.hpp"   // for WARN, INFO, FATAL_ERROR
#include "restClient.hpp"       // for restClient
#include "restServer.hpp"       // for restServer, connectionInstance
#include "configUpdater.hpp"   // for ConfigUpdater
#include "timeUtil.hpp"

#include "fmt.hpp"  // for format
#include "json.hpp" //for basic_json, basic_json<>::object_t, basic_jason<>::value_type

#include <cstdint>      // for uint64_t  TODO: why not stdint.h?
#include <exception>    // for exception
#include <math.h>       // for abs
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for runtime_error, invalid_argument
#include <vector>       // for vector


REGISTER_TELESCOPE(CHORDTelescope, "CHORDTelescope");

#define GIGA 1'000'000'000

static constexpr double C = 2.99792458e8;

using kotekan::connectionInstance;
using kotekan::restServer;

CHORDTelescope::CHORDTelescope(const kotekan::Config& config,
                               const std::string& path) :
    Telescope(config.get<std::string>(path, "log_level")) {

    _unique_name = path;
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
    _R_itrs_to_topo[0][0] = -sin_lon;
    _R_itrs_to_topo[0][1] =  cos_lon;
    _R_itrs_to_topo[0][2] =  0.0;

    // Topocentric Y (North) in Geocentric (Earth-centered, Earth-fixed) coords
    _R_itrs_to_topo[1][0] = -sin_lat * cos_lon;
    _R_itrs_to_topo[1][1] = -sin_lat * sin_lon;
    _R_itrs_to_topo[1][2] =  cos_lat;

    // Topocentric Z (Up) in Geocentric (Earth-centered, Earth-fixed) coords
    _R_itrs_to_topo[2][0] =  cos_lat * cos_lon;
    _R_itrs_to_topo[2][1] =  cos_lat * sin_lon;
    _R_itrs_to_topo[2][2] =  sin_lat;


    using namespace std::placeholders;

    kotekan::configUpdater::instance().subscribe(
            config.get<std::string>(path, "updatable_config"),
            std::bind(&CHORDTelescope::receive_eop_updates, this, _1));

    restServer &rest_server = restServer::instance();

    rest_server.register_get_callback(path + "/time0_ns",
                        std::bind(&CHORDTelescope::send_time0_ns, this, _1));
}

CHORDTelescope::~CHORDTelescope() {
    restServer &rest_server = restServer::instance();
    rest_server.remove_json_callback(_unique_name + "/time0_ns");
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
    
bool CHORDTelescope::receive_eop_updates(nlohmann::json& json) {
    std::lock_guard<std::mutex> lock(_eop_lock);
    try {
        std::vector<struct EOP> tmp_eop_table;
        for(const auto& elem : json.at("earth_orientation_parameter_table")) {
            INFO("CHORDTelescope EOP update: {:s}", elem.dump());
            uint64_t t_ns = elem.at("time_inst_ns").get<uint64_t>();
            double dut1 = elem.at("delta_UT1_inst").get<double>();
            double x_pm = elem.at("x_pm").get<double>();
            double y_pm = elem.at("y_pm").get<double>();
            tmp_eop_table.push_back(
                    build_EOP_from_update(t_ns, dut1, x_pm, y_pm));
        }

        std::sort(tmp_eop_table.begin(), tmp_eop_table.end(), EOP_comp_time);

        _eop_table = tmp_eop_table;

    } catch (std::exception& e) {
        WARN("CHORDTelescope failed to read EOP update: {:s}", e.what());
        return false;
    }

    return true;
}

void CHORDTelescope::send_time0_ns(connectionInstance& conn) {
    nlohmann::json reply;
    reply["time0_ns"] = time0_ns;
    conn.send_json_reply(reply);
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
        double ra, double dec, const struct EOP &eop) const {

    // Taking the ra & dec to be in CIRS frame

    double deg2rad = M_PI / 180;

    double phi = deg2rad * ra;
    double theta = deg2rad * (90 - dec);

    // unit vector pointing to ra/dec in spherical coordinates
    // fixed to the Earth.  phi=0 ~ Greenwich
    std::array<double, 3> n_cirs= {
                          cos(phi) * sin(theta),
                          sin(phi) * sin(theta),
                          cos(theta)};

    INFO("n_cirs: {} {} {}", n_cirs[0], n_cirs[1], n_cirs[2]);

    std::array<double, 3> n_itrs = cirs_vec_to_itrs_vec(n_cirs, eop);
    std::array<double, 3> n_topo = itrs_vec_to_topocen_vec(n_itrs);

    return topocen_vec_to_tel_vec(n_topo);
}

std::array<double, 3> CHORDTelescope::get_pointing_vec_in_tel_coords() const {

    double deg2rad = M_PI/180;
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

std::array<double, 3> CHORDTelescope::vec_axes_rotation_R1(
        const std::array<double, 3>& v, double theta) const {
    // Return coordinates of vector v in frame rotated by theta about x-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    std::array<double, 3> v_rot = {
         v[0],
         cos_th * v[1] + sin_th * v[2],
        -sin_th * v[1] + cos_th * v[2]};

    return v_rot;
}

std::array<double, 3> CHORDTelescope::vec_axes_rotation_R2(
        const std::array<double, 3>& v, double theta) const {
    // Return coordinates of vector v in frame rotated by theta about y-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    std::array<double, 3> v_rot = {
        cos_th * v[0] - sin_th * v[2],
        v[1],
        sin_th * v[0] + cos_th * v[2]};

    return v_rot;
}

std::array<double, 3> CHORDTelescope::vec_axes_rotation_R3(
        const std::array<double, 3>& v, double theta) const {
    // Return coordinates of vector v in frame rotated by theta about z-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    std::array<double, 3> v_rot = {
         cos_th * v[0] + sin_th * v[1],
        -sin_th * v[0] + cos_th * v[1],
         v[2]};

    return v_rot;
}


std::array<double, 3> CHORDTelescope::itrs_vec_to_topocen_vec(
        const std::array<double, 3>& v_itrs) const {

    std::array<double, 3> v_topo = {0, 0, 0};
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            v_topo[i] += _R_itrs_to_topo[i][j] * v_itrs[j];

    return v_topo;
}

std::array<double, 3> CHORDTelescope::topocen_vec_to_itrs_vec(
        const std::array<double, 3>& v_topo) const {

    std::array<double, 3> v_itrs = {0, 0, 0};
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            v_itrs[i] += _R_itrs_to_topo[j][i] * v_topo[j];

    return v_itrs;
}

std::array<double, 3> CHORDTelescope::cirs_vec_to_itrs_vec(
        const std::array<double, 3>& v_cirs, const struct EOP &eop) const {

    double deg2rad = M_PI/180;
    double as2rad = deg2rad / 3600;
    
    double era = deg2rad * eop.ERA_deg;
    double xp = as2rad * eop.xp_as;
    double yp = as2rad * eop.yp_as;

    std::array<double, 3> v1     = vec_axes_rotation_R3(v_cirs, era);
    std::array<double, 3> v2     = vec_axes_rotation_R2(v1,    -xp);
    std::array<double, 3> v_itrs = vec_axes_rotation_R1(v2,    -yp);

    return v_itrs;
}

std::array<double, 3> CHORDTelescope::itrs_vec_to_cirs_vec(
        const std::array<double, 3>& v_itrs, const struct EOP &eop) const {

    double deg2rad = M_PI/180;
    double as2rad = deg2rad / 3600;
    double era = deg2rad * eop.ERA_deg;
    double xp = as2rad * eop.xp_as;
    double yp = as2rad * eop.yp_as;

    std::array<double, 3> v1     = vec_axes_rotation_R1(v_itrs, yp);
    std::array<double, 3> v2     = vec_axes_rotation_R2(v1,     xp);
    std::array<double, 3> v_cirs = vec_axes_rotation_R3(v2,    -era);

    return v_cirs;
}

void CHORDTelescope::fringestop_phases_1d(double freq_Hz, 
        const struct EOP &eop, const struct EOP &eop0,
        std::vector<std::complex<double>>& phases) const {

    //Take the pointing vector for the telescope (constant in time),
    //and find it in the CIRS frame at ERA0.  This is the point we are
    //attempting to stop the fringes at.
    std::array<double, 3> n_tel0 = get_pointing_vec_in_tel_coords();
    std::array<double, 3> n_topo0 = tel_vec_to_topocen_vec(n_tel0);
    std::array<double, 3> n_itrs0 = topocen_vec_to_itrs_vec(n_topo0);
    std::array<double, 3> n_cirs = itrs_vec_to_cirs_vec(n_itrs0, eop0);

    //Now, given this CIRS vector, find its components in the telescope
    //frame at the requested ERA
    std::array<double, 3> n_itrs = cirs_vec_to_itrs_vec(n_cirs, eop);
    std::array<double, 3> n_topo = itrs_vec_to_topocen_vec(n_itrs);
    std::array<double, 3> n_tel = topocen_vec_to_tel_vec(n_topo);

    // n_tel is now (at ERA) the point on the sky which will be at the
    // phase center (n_tel0) at ERA0.
   
    // wavenumber for this frequency
    double k = 2*M_PI * freq_Hz / C;

    for(uint64_t i = 0; i < _dish_positions.size(); i++)
    {
        double phase = -k * (
                  _dish_positions[i][0] * (n_tel[0] - n_tel0[0])
                + _dish_positions[i][1] * (n_tel[1] - n_tel0[1])
                + _dish_positions[i][2] * (n_tel[2] - n_tel0[2]));

        phases[i] = {cos(phase) , sin(phase)};
    }

    /*
    DEBUG("Fringestop Phases: [{}+{}i, {}+{}i, {}+{}i, {}+{}i, {}+{}i, {}+{}i]",
            phases[0].real(), phases[0].imag(),
            phases[1].real(), phases[1].imag(),
            phases[2].real(), phases[2].imag(),
            phases[3].real(), phases[3].imag(),
            phases[4].real(), phases[4].imag(),
            phases[5].real(), phases[5].imag());
    */
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

int CHORDTelescope::get_EOP_table_len() const {
    std::lock_guard<std::mutex> lock(_eop_lock);
    return _eop_table.size();
}

struct EOP CHORDTelescope::get_EOP_at_idx(uint64_t i) const {
    std::lock_guard<std::mutex> lock(_eop_lock);

    if(i < _eop_table.size()) {
        struct EOP eop = _eop_table[i];
        return eop;
    }

    return eop_null;
}

struct EOP CHORDTelescope::get_EOP_at_time(const timespec &t_target) const {
    //Interpolate on the EOP table to find EOP for the given instrument time.

    std::lock_guard<std::mutex> lock(_eop_lock);

    struct EOP eop;
    eop.t_inst = t_target;

    // _eop_table is always sorted by instrument time. Do a quick search
    // for the first table entry with larger time than the target.
    auto eop_b = std::lower_bound(_eop_table.begin(), _eop_table.end(),
                                  eop, EOP_comp_time);
    
    // DUT1, xp_as, and yp_as evolve slowly, on secular time scales, so we
    // interpolate these, and calculate ERA after.

    if(eop_b == _eop_table.begin()) {
        // Time is earlier than covered by the table, use the first value.
        eop.delta_UT1_inst = eop_b->delta_UT1_inst;
        eop.xp_as = eop_b->xp_as;
        eop.yp_as = eop_b->yp_as;
    }
    else if(eop_b == _eop_table.end()) {
        // Time is later than covered by the table, use the last value.
        auto eop_last = eop_b - 1;
        eop.delta_UT1_inst = eop_last->delta_UT1_inst;
        eop.xp_as = eop_last->xp_as;
        eop.yp_as = eop_last->yp_as;
    }
    else {
        // Interpolate!
        auto eop_a = eop_b - 1;
        int64_t diff_ns_a = GIGA * (t_target.tv_sec - eop_a->t_inst.tv_sec)
                             + t_target.tv_nsec - eop_a->t_inst.tv_nsec;
        int64_t diff_ns_b = GIGA * (t_target.tv_sec - eop_b->t_inst.tv_sec)
                             + t_target.tv_nsec - eop_b->t_inst.tv_nsec;

        int64_t diff_ns = diff_ns_a - diff_ns_b;

        double wb = diff_ns_a / ((double) diff_ns);
        double wa = 1.0 - wb;

        eop.delta_UT1_inst = wa * eop_a->delta_UT1_inst
                            + wb * eop_b->delta_UT1_inst;
        eop.xp_as = wa * eop_a->xp_as + wb * eop_b->xp_as;
        eop.yp_as = wa * eop_a->yp_as + wb * eop_b->yp_as;
    }

    timespec t_ut1 = get_UT1_from_time(t_target, eop.delta_UT1_inst);
    double era = get_ERA_from_UT1(t_ut1, nullptr);

    eop.t_ut1 = t_ut1;
    eop.ERA_deg = era;
    
    return eop;
}

struct EOP CHORDTelescope::get_EOP_at_UT1(const timespec &t_ut1) const {
    //Interpolate on the EOP table to find EOP for the given UT1 time.

    std::lock_guard<std::mutex> lock(_eop_lock);

    struct EOP eop;
    eop.t_ut1 = t_ut1;

    // _eop_table is always sorted by instrument time. UT1 is monotonic 
    // with instrument time, unless the Earth has been met with catastrophe.
    // Do a quick search for the first table entry with larger UT1 time than
    // the target.
    auto eop_b = std::lower_bound(_eop_table.begin(), _eop_table.end(),
                                  eop, EOP_comp_ut1);

    // DUT1, xp_as, and yp_as evolve slowly, on secular time scales, so we
    // interpolate these, and calculate ERA after.

    if(eop_b == _eop_table.begin()) {
        // Time is earlier than covered by the table, use the first value.
        eop.delta_UT1_inst = eop_b->delta_UT1_inst;
        eop.xp_as = eop_b->xp_as;
        eop.yp_as = eop_b->yp_as;
    }
    else if(eop_b == _eop_table.end()) {
        // Time is later than covered by the table, use the last value.
        auto eop_last = eop_b - 1;
        eop.delta_UT1_inst = eop_last->delta_UT1_inst;
        eop.xp_as = eop_last->xp_as;
        eop.yp_as = eop_last->yp_as;
    }
    else {
        // Interpolate!
        auto eop_a = eop_b - 1;
        int64_t diff_ns_a = GIGA * (t_ut1.tv_sec - eop_a->t_ut1.tv_sec)
                             + t_ut1.tv_nsec - eop_a->t_ut1.tv_nsec;
        int64_t diff_ns_b = GIGA * (t_ut1.tv_sec - eop_b->t_ut1.tv_sec)
                             + t_ut1.tv_nsec - eop_b->t_ut1.tv_nsec;

        int64_t diff_ns = diff_ns_a - diff_ns_b;

        double wb = diff_ns_a / ((double) diff_ns);
        double wa = 1.0 - wb;

        eop.delta_UT1_inst = wa * eop_a->delta_UT1_inst
                            + wb * eop_b->delta_UT1_inst;
        eop.xp_as = wa * eop_a->xp_as + wb * eop_b->xp_as;
        eop.yp_as = wa * eop_a->yp_as + wb * eop_b->yp_as;
    }

    timespec t_inst = get_time_from_UT1(t_ut1, eop.delta_UT1_inst);
    double era = get_ERA_from_UT1(t_ut1, nullptr);

    eop.t_inst = t_inst;
    eop.ERA_deg = era;
    
    return eop;
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

struct EOP CHORDTelescope::build_EOP_from_update(uint64_t time_ns,
                                                 double delta_ut1_inst,
                                                 double xp_as,
                                                 double yp_as) const {

    struct timespec t {
        .tv_sec=(time_t)(time_ns / GIGA),
        .tv_nsec=(long)(time_ns % GIGA)};

    struct timespec ut1 = get_UT1_from_time(t, delta_ut1_inst);
    double era = get_ERA_from_UT1(ut1, nullptr);

    struct EOP eop {
        .t_inst = t,
        .t_ut1 = ut1,
        .delta_UT1_inst = delta_ut1_inst,
        .ERA_deg = era,
        .xp_as = xp_as,
        .yp_as = yp_as};
    
    return eop;
}

bool EOP_comp_time(const struct EOP &eop1, const struct EOP &eop2)
{
    if(eop1.t_inst.tv_sec < eop2.t_inst.tv_sec)
        return true;
    if(eop1.t_inst.tv_sec == eop2.t_inst.tv_sec
            && eop1.t_inst.tv_nsec < eop2.t_inst.tv_nsec)
        return true;

    return false;
}

bool EOP_comp_ut1(const struct EOP &eop1, const struct EOP &eop2)
{
    if(eop1.t_ut1.tv_sec < eop2.t_ut1.tv_sec)
        return true;
    if(eop1.t_ut1.tv_sec == eop2.t_ut1.tv_sec
            && eop1.t_ut1.tv_nsec < eop2.t_ut1.tv_nsec)
        return true;

    return false;
}
