#include "CHORDTelescope.hpp"

#include "Telescope.hpp"      // for Telescope, freq_id_t, REGISTER_TELESCOPE, _factory_aliasTe...
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for WARN, INFO, DEBUG
#include "restClient.hpp"     // for restClient
#include "restServer.hpp"     // for restServer, connectionInstance
#include "timeUtil.hpp" // for EOP, get_ERA_from_UT1, get_UT1_from_time, nanosec_i64_to_timespec

#include "fmt.hpp"  // for compile_string_to_view
#include "json.hpp" // for basic_json, json, iter_impl, input_adapter

#include <algorithm>  // for lower_bound, copy, sort, max
#include <assert.h>   // for assert
#include <exception>  // for exception
#include <functional> // for bind, _1, function
#include <math.h>     // for sin, cos, M_PI
#include <stdexcept>  // for runtime_error
#include <vector>     // for vector


REGISTER_TELESCOPE(CHORDTelescope, "CHORDTelescope");

#define GIGA 1'000'000'000L

static constexpr double C = 2.99792458e8;
static constexpr double deg2rad = M_PI / 180.0;
static constexpr double arcsec2rad = M_PI / (180.0 * 3600);

using kotekan::connectionInstance;
using kotekan::restServer;

CHORDTelescope::CHORDTelescope(const kotekan::Config& config, const std::string& path) :
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

    if (gps_enabled)
        INFO("Telescope configured with GPS time0: {:d} ns", time0_ns);
    else
        INFO("Telescope GPS time not enabled.");

    // Set the time- and frequency-sampling parameters from the config.
    set_sampling_params(config, path);

    _origin_itrs_lon_deg = config.get_default<double>(path, "origin_itrs_lon_deg", 0.0);
    _origin_itrs_lat_deg = config.get_default<double>(path, "origin_itrs_lat_deg", 0.0);
    _dish_coelev_deg = config.get_default<double>(path, "dish_coelev_deg", 0.0);

    INFO("Telescope configured with longitude:    {:f} deg", _origin_itrs_lon_deg);
    INFO("Telescope configured with latitude:     {:f} deg", _origin_itrs_lat_deg);
    INFO("Telescope configured with co-elevation: {:f} deg", _dish_coelev_deg);
    INFO("Telescope targetting approximate declination: {:f} deg",
         _origin_itrs_lat_deg + 90 - _dish_coelev_deg);

    // Read in the Telescope Coord axes. Must be normalized and orthogonal.
    std::array<double, 3> sep_x =
        config.get_default<std::array<double, 3>>(path, "grid_x_axis", {1.0, 0.0, 0.0});
    std::array<double, 3> sep_y =
        config.get_default<std::array<double, 3>>(path, "grid_y_axis", {0.0, 1.0, 0.0});
    // Compute the Z axis as X x Y
    std::array<double, 3> sep_z = {sep_x[1] * sep_y[2] - sep_x[2] * sep_y[1],
                                   sep_x[2] * sep_y[0] - sep_x[0] * sep_y[2],
                                   sep_x[0] * sep_y[1] - sep_x[1] * sep_y[0]};

    // Construct the topocentric -> tel rotation matrix.
    // We assume the inverse is the transpose.
    for (int i = 0; i < 3; i++) {
        _R_topo_to_tel[0][i] = sep_x[i];
        _R_topo_to_tel[1][i] = sep_y[i];
        _R_topo_to_tel[2][i] = sep_z[i];
    }

    // Read in the Dish Coord axes. Must be normalized and orthogonal.
    std::array<double, 3> dish_x =
        config.get_default<std::array<double, 3>>(path, "dish_elev_axis", {1.0, 0.0, 0.0});
    std::array<double, 3> dish_z =
        config.get_default<std::array<double, 3>>(path, "dish_vert_axis", {0.0, 0.0, 1.0});
    // Compute the Y axis as Z x X
    std::array<double, 3> dish_y = {dish_z[1] * dish_x[2] - dish_z[2] * dish_x[1],
                                    dish_z[2] * dish_x[0] - dish_z[0] * dish_x[2],
                                    dish_z[0] * dish_x[1] - dish_z[1] * dish_x[0]};

    // Construct the topocentric -> dish rotation matrix.
    // We assume the inverse is the transpose.
    for (int i = 0; i < 3; i++) {
        _R_topo_to_dish[0][i] = dish_x[i];
        _R_topo_to_dish[1][i] = dish_y[i];
        _R_topo_to_dish[2][i] = dish_z[i];
    }

    // Set all dish input data: num_dishes, dish_info_table, dish_position, ...
    set_dish_info(config, path);

    double cos_lon = cos(deg2rad * _origin_itrs_lon_deg);
    double sin_lon = sin(deg2rad * _origin_itrs_lon_deg);
    double cos_lat = cos(deg2rad * _origin_itrs_lat_deg);
    double sin_lat = sin(deg2rad * _origin_itrs_lat_deg);

    // Topocentric X (East) in ITRS (Earth-centered, Earth-fixed) coords
    _R_itrs_to_topo[0][0] = -sin_lon;
    _R_itrs_to_topo[0][1] = cos_lon;
    _R_itrs_to_topo[0][2] = 0.0;

    // Topocentric Y (North) in ITRS (Earth-centered, Earth-fixed) coords
    _R_itrs_to_topo[1][0] = -sin_lat * cos_lon;
    _R_itrs_to_topo[1][1] = -sin_lat * sin_lon;
    _R_itrs_to_topo[1][2] = cos_lat;

    // Topocentric Z (Up) in ITRS (Earth-centered, Earth-fixed) coords
    _R_itrs_to_topo[2][0] = cos_lat * cos_lon;
    _R_itrs_to_topo[2][1] = cos_lat * sin_lon;
    _R_itrs_to_topo[2][2] = sin_lat;


    // Set up callbacks for updating EOP and sending time0_ns
    using namespace std::placeholders;

    kotekan::configUpdater::instance().subscribe(
        config.get<std::string>(path, "updatable_config"),
        std::bind(&CHORDTelescope::receive_eop_updates, this, _1));

    restServer& rest_server = restServer::instance();

    rest_server.register_get_callback(path + "/time0_ns",
                                      std::bind(&CHORDTelescope::send_time0_ns, this, _1));
    rest_server.register_get_callback(path + "/eop_table",
                                      std::bind(&CHORDTelescope::send_eop_table, this, _1));
}

CHORDTelescope::~CHORDTelescope() {
    // Must manually remove the GET callbacks
    restServer& rest_server = restServer::instance();
    rest_server.remove_get_callback(_unique_name + "/time0_ns");
    rest_server.remove_get_callback(_unique_name + "/eop_table");
}

void CHORDTelescope::set_sampling_params(const kotekan::Config& config, const std::string& path) {

    double sampling_rate_MHz = config.get_default<double>(path, "sampling_rate_MHz", 3.2e3);
    uint64_t fft_length = config.get_default<uint64_t>(path, "fft_length", 16384);
    ny_zone = config.get_default<uint8_t>(path, "nyquist_zone", 1);

    // Find the time in nanoseconds between fpga_seq_nums (ie. the time between fft_length raw ADC
    // samples)
    dt_ns = (GIGA * fft_length) / (1.0e6 * sampling_rate_MHz);

    // Set the physical frequency of id=0, and the spacing, taking into account
    // the aliasing of each Nyquist zone

    // the freq0 mode jumps in frequency every 2 nyquist zones. The first zone (zone = 1) is the
    // textbook FFT and has freq0 = 0.
    freq0_MHz = (ny_zone / 2) * sampling_rate_MHz;
    // Odd zones count up from freq0, even zones count down.
    df_MHz = (ny_zone % 2 == 1 ? 1 : -1) * sampling_rate_MHz / fft_length;
    // Total number of frequency channels (input data is Real)
    nfreq_total = fft_length / 2;
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

    if (!config.exists("/gps_time", "frame0_nano")) {
        WARN("No GPS frame0 time found in config.");
        return;
    }

    time0_ns = config.get<uint64_t>("/gps_time", "frame0_nano");
    gps_enabled = true;
}

void CHORDTelescope::set_gps(const std::string& host, const uint32_t port,
                             const std::string& path) {

    INFO("Requesting GPS time from server: {:s}.{:d}{:s} This might take some time...", host, port,
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
    }

    time0_ns = json_reply["frame0_nano"].get<uint64_t>();
    INFO("GPS frame0 time set to {:d}", time0_ns);
    gps_enabled = true;
}

bool CHORDTelescope::receive_eop_updates(nlohmann::json& json) {
    // Make sure no one is using the EOP table while we're updating it.
    try {
        // Fill a temporary table with the updated values.
        std::vector<EOP> tmp_eop_table;
        for (const auto& elem : json.at("earth_orientation_parameter_table")) {
            INFO("CHORDTelescope EOP update: {:s}", elem.dump());
            int64_t t_ns = elem.at("time_inst_ns").get<int64_t>();
            double dut1 = elem.at("delta_UT1_inst").get<double>();
            double x_pm = elem.at("x_pm").get<double>();
            double y_pm = elem.at("y_pm").get<double>();
            tmp_eop_table.push_back(build_EOP_from_update(t_ns, dut1, x_pm, y_pm));
        }

        if (tmp_eop_table.empty()) {
            FATAL_ERROR_NON_OO(
                "CHORDTelescope {}: earth_orientation_parameter_table update contained no entries.",
                _unique_name);
        }

        // Sort chronologically
        std::sort(tmp_eop_table.begin(), tmp_eop_table.end(), EOP_comp_time);

        // Replace old table with new.
        {
            std::unique_lock lock(_eop_lock);
            _eop_table = tmp_eop_table;
            INFO("Updated EOP Table with {:d} entries", _eop_table.size());
        }

    } catch (std::exception& e) {
        WARN("CHORDTelescope failed to read EOP update: {:s}", e.what());
        return false;
    }

    return true;
}

void CHORDTelescope::send_eop_table(connectionInstance& conn) {
    nlohmann::json reply;
    {
        std::shared_lock lock(_eop_lock);
        reply["eop_table"] = _eop_table;
    }
    conn.send_json_reply(reply);
}

void CHORDTelescope::send_time0_ns(connectionInstance& conn) {
    nlohmann::json reply;
    reply["time0_ns"] = time0_ns;
    conn.send_json_reply(reply);
}

timespec CHORDTelescope::to_time(uint64_t seq) const {
    return nanosec_i64_to_timespec(to_time_ns(seq));
}

int64_t CHORDTelescope::to_time_ns(uint64_t seq) const {
    return time0_ns + seq * dt_ns;
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

double CHORDTelescope::get_origin_itrs_lon_deg() const {
    return _origin_itrs_lon_deg;
}

double CHORDTelescope::get_origin_itrs_lat_deg() const {
    return _origin_itrs_lat_deg;
}

double CHORDTelescope::get_dish_coelev_deg() const {
    return _dish_coelev_deg;
}

std::array<double, 3> CHORDTelescope::get_sky_vec_in_tel_coords(double ra, double dec,
                                                                const EOP& eop) const {

    // Taking the ra & dec to be in CIRS frame

    double phi = deg2rad * ra;
    double theta = deg2rad * (90 - dec);

    // unit vector pointing to ra/dec in spherical coordinates
    // fixed to the Earth.  phi=0 ~ Greenwich
    std::array<double, 3> n_cirs = {cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)};

    DEBUG("n_cirs: {} {} {}", n_cirs[0], n_cirs[1], n_cirs[2]);

    // Transform CIRS -> ITRS -> TOPO -> Telescope.
    std::array<double, 3> n_itrs = vec_cirs_to_itrs(n_cirs, eop);
    std::array<double, 3> n_topo = vec_itrs_to_topocen(n_itrs);

    return vec_topocen_to_tel(n_topo);
}

std::array<double, 3> CHORDTelescope::get_pointing_vec_in_dish_coords() const {

    // Dish coordinates are fixed with z "up" (co-elevation 0 degrees) and x
    // along the elevation axis of the dish mount.  In this frame the pointing
    // vector is just given by the current elevation.

    double coelev = deg2rad * _dish_coelev_deg;

    // coelev=90 ==> North (y), coelev=0 => Up (z), coelev=-90 -> South (-y)
    std::array<double, 3> n_point = {0.0, sin(coelev), cos(coelev)};

    return n_point;
}

std::array<double, 3>
CHORDTelescope::vec_topocen_to_dish(const std::array<double, 3>& v_topocen) const {

    // Just multiply by known Rotation matrix.
    std::array<double, 3> v_dish = {0, 0, 0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            v_dish[i] += _R_topo_to_dish[i][j] * v_topocen[j];

    return v_dish;
}

std::array<double, 3>
CHORDTelescope::vec_dish_to_topocen(const std::array<double, 3>& v_dish) const {

    // Inverse transform, use R transpose.
    std::array<double, 3> v_topo = {0, 0, 0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            v_topo[i] += _R_topo_to_dish[j][i] * v_dish[j];

    return v_topo;
}

std::array<double, 3>
CHORDTelescope::vec_topocen_to_tel(const std::array<double, 3>& v_topocen) const {

    // Just multiply by known Rotation matrix.
    std::array<double, 3> v_tel = {0, 0, 0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            v_tel[i] += _R_topo_to_tel[i][j] * v_topocen[j];

    return v_tel;
}

std::array<double, 3> CHORDTelescope::vec_tel_to_topocen(const std::array<double, 3>& v_tel) const {

    // Inverse transform, use R transpose.
    std::array<double, 3> v_topocen = {0, 0, 0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            v_topocen[i] += _R_topo_to_tel[j][i] * v_tel[j];

    return v_topocen;
}

std::array<double, 3> CHORDTelescope::vec_axes_rotation_R1(const std::array<double, 3>& v,
                                                           double theta) const {
    // Return coordinates of vector v in frame rotated by theta about x-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    std::array<double, 3> v_rot = {v[0], cos_th * v[1] + sin_th * v[2],
                                   -sin_th * v[1] + cos_th * v[2]};

    return v_rot;
}

std::array<double, 3> CHORDTelescope::vec_axes_rotation_R2(const std::array<double, 3>& v,
                                                           double theta) const {
    // Return coordinates of vector v in frame rotated by theta about y-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    std::array<double, 3> v_rot = {cos_th * v[0] - sin_th * v[2], v[1],
                                   sin_th * v[0] + cos_th * v[2]};

    return v_rot;
}

std::array<double, 3> CHORDTelescope::vec_axes_rotation_R3(const std::array<double, 3>& v,
                                                           double theta) const {
    // Return coordinates of vector v in frame rotated by theta about z-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    std::array<double, 3> v_rot = {cos_th * v[0] + sin_th * v[1], -sin_th * v[0] + cos_th * v[1],
                                   v[2]};

    return v_rot;
}


std::array<double, 3>
CHORDTelescope::vec_itrs_to_topocen(const std::array<double, 3>& v_itrs) const {

    // Just multiply by known Rotation matrix.
    std::array<double, 3> v_topo = {0, 0, 0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            v_topo[i] += _R_itrs_to_topo[i][j] * v_itrs[j];

    return v_topo;
}

std::array<double, 3>
CHORDTelescope::vec_topocen_to_itrs(const std::array<double, 3>& v_topo) const {

    // Inverse transform, use R transpose.
    std::array<double, 3> v_itrs = {0, 0, 0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            v_itrs[i] += _R_itrs_to_topo[j][i] * v_topo[j];

    return v_itrs;
}

std::array<double, 3> CHORDTelescope::vec_cirs_to_itrs(const std::array<double, 3>& v_cirs,
                                                       const EOP& eop) const {

    // IERS Conventions (2010) Chapter 5, Eq 5.1-5.3, and 5.5 give the
    // ITRS -> CIRS Transformation:
    //
    // [CIRS] = R(t) W(t) [ITRS]        Eq. 5.1
    //
    // W(t) = R3(s') R2(x') R1(y')      Eq. 5.3
    // R(t) = R3(-ERA)                  Eq. 5.5
    //
    // We ignore the s' contribution here, it's magnitude is only
    // microarcsecond.
    //
    // The inverse transformation reverses this, taking the negative of each
    // argument.
    double era = deg2rad * eop.ERA_deg;
    double xp = arcsec2rad * eop.xp_as;
    double yp = arcsec2rad * eop.yp_as;

    // 5.5 inverse
    std::array<double, 3> v1 = vec_axes_rotation_R3(v_cirs, era);
    // 5.3, second factor, inverse
    std::array<double, 3> v2 = vec_axes_rotation_R2(v1, -xp);
    // 5.3, first factor, inverse
    std::array<double, 3> v_itrs = vec_axes_rotation_R1(v2, -yp);

    return v_itrs;
}

std::array<double, 3> CHORDTelescope::vec_itrs_to_cirs(const std::array<double, 3>& v_itrs,
                                                       const EOP& eop) const {

    // IERS Conventions (2010) Chapter 5, Eq 5.1-5.3, and 5.5 give the
    // ITRS -> CIRS Transformation:
    //
    // [CIRS] = R(t) W(t) [ITRS]        Eq. 5.1
    //
    // W(t) = R3(s') R2(x') R1(y')      Eq. 5.3
    // R(t) = R3(-ERA)                  Eq. 5.5
    //
    // We ignore the s' contribution here, it's magnitude is only
    // microarcsecond.

    double era = deg2rad * eop.ERA_deg;
    double xp = arcsec2rad * eop.xp_as;
    double yp = arcsec2rad * eop.yp_as;

    // 5.3 (First factor in W)
    std::array<double, 3> v1 = vec_axes_rotation_R1(v_itrs, yp);
    // 5.3 (Second factor in W)
    std::array<double, 3> v2 = vec_axes_rotation_R2(v1, xp);
    // 5.5
    std::array<double, 3> v_cirs = vec_axes_rotation_R3(v2, -era);

    return v_cirs;
}

void CHORDTelescope::fringestop_phases_1d(double freq_MHz, const EOP& eop, const EOP& eop0,
                                          std::vector<std::complex<double>>& phases) const {

    // Get the pointing vector (phase center) for the telescope in dish coordinates. This is
    // constant in time.
    std::array<double, 3> n_dish0 = get_pointing_vec_in_dish_coords();

    // Transform the pointing vector into topocentric coordinates (from which we can
    // transform to the sky), and telescope coordinates (where the dish locations live).
    // These are also constant in time.
    std::array<double, 3> n_topo0 = vec_dish_to_topocen(n_dish0);
    std::array<double, 3> n_tel0 = vec_topocen_to_tel(n_topo0);

    // Take the pointing vector for the telescope and find it in the CIRS frame at ERA0.
    // This is the point we are attempting to stop the fringes at.
    std::array<double, 3> n_itrs0 = vec_topocen_to_itrs(n_topo0);
    std::array<double, 3> n_cirs = vec_itrs_to_cirs(n_itrs0, eop0);

    // Now, given this CIRS vector, find its components in the telescope
    // frame at the requested (current) ERA
    std::array<double, 3> n_itrs = vec_cirs_to_itrs(n_cirs, eop);
    std::array<double, 3> n_topo = vec_itrs_to_topocen(n_itrs);
    std::array<double, 3> n_tel = vec_topocen_to_tel(n_topo);

    // n_tel is now (at ERA) the point on the sky which will be at the
    // phase center (n_tel0) at ERA0.

    // wavenumber for this frequency
    double k = 2 * M_PI * 1e6 * freq_MHz / C;

    for (uint64_t i = 0; i < _dish_positions.size(); i++) {
        double phase = -k
                       * (_dish_positions[i][0] * (n_tel[0] - n_tel0[0])
                          + _dish_positions[i][1] * (n_tel[1] - n_tel0[1])
                          + _dish_positions[i][2] * (n_tel[2] - n_tel0[2]));

        phases[i] = {cos(phase), sin(phase)};
    }
}

void CHORDTelescope::get_input_maps(dishInputFields& input) const {

    // Ensure fields have the correct size
    input.ew_idx.reserve(_num_dishes);
    input.ns_idx.reserve(_num_dishes);
    input.feed_pos_disp_m.reserve(_num_dishes);
    input.coelev_disp_deg.reserve(_num_dishes);
    input.type.reserve(_num_dishes);
    input.label.reserve(_num_dishes);

    input.ew_idx.clear();
    input.ns_idx.clear();
    input.feed_pos_disp_m.clear();
    input.coelev_disp_deg.clear();
    input.type.clear();
    input.label.clear();

    // Fill them from our internal table.
    for (int i = 0; i < _num_dishes; i++) {
        input.ew_idx.push_back(_dish_info_table[i].ew_idx);
        input.ns_idx.push_back(_dish_info_table[i].ns_idx);
        input.feed_pos_disp_m.push_back(_dish_info_table[i].feed_pos_disp_m);
        input.coelev_disp_deg.push_back(_dish_info_table[i].coelev_disp_deg);
        input.type.push_back(static_cast<int64_t>(_dish_info_table[i].type));
        input.label.push_back(_dish_info_table[i].label);
    }
}

uint64_t CHORDTelescope::get_num_stacks() const {
    FATAL_ERROR("get_num_stacks() has not been implemented in CHORDTelescope yet.");
    return 0;
}

double CHORDTelescope::get_tel_orientation_el(int i, int j) const {
    return _R_topo_to_tel[i][j];
}

double CHORDTelescope::get_dish_orientation_el(int i, int j) const {
    return _R_topo_to_dish[i][j];
}

std::array<double, 3> CHORDTelescope::get_dish_position(int i) const {
    return _dish_positions[i];
}

int CHORDTelescope::get_num_dishes() const {
    return _num_dishes;
}

int CHORDTelescope::get_EOP_table_len() const {
    std::shared_lock lock(_eop_lock);
    return _eop_table.size();
}

EOP CHORDTelescope::get_EOP_at_idx(uint64_t i) const {
    std::shared_lock lock(_eop_lock);

    if (i < _eop_table.size()) {
        EOP eop = _eop_table[i];
        return eop;
    }

    return eop_null;
}

EOP CHORDTelescope::get_EOP_at_time(const timespec& ts_target) const {
    // Interpolate on the EOP table to find EOP for the given instrument time.

    EOP eop;

    int64_t t_target = timespec_to_nanosec_i64(ts_target);
    eop.t_inst = t_target;

    {
        std::shared_lock lock(_eop_lock);
        // _eop_table is always sorted by instrument time. Do a quick search
        // for the first table entry with larger time than the target.
        auto eop_b = std::lower_bound(_eop_table.begin(), _eop_table.end(), eop, EOP_comp_time);

        // DUT1, xp_as, and yp_as evolve slowly, on secular time scales, so we
        // interpolate these, and calculate ERA after.

        if (eop_b == _eop_table.begin()) {
            // Time is earlier than covered by the table, use the first value.
            eop.delta_UT1_inst = eop_b->delta_UT1_inst;
            eop.xp_as = eop_b->xp_as;
            eop.yp_as = eop_b->yp_as;
            WARN(
                "Requesting EOP earlier than in table. Requested time = {:d} s + {:d} ns. Earliest "
                "time = {:d} s + {:d} ns.",
                t_target / GIGA, t_target % GIGA, eop_b->t_inst / GIGA, eop_b->t_inst % GIGA);
        } else if (eop_b == _eop_table.end()) {
            // Time is later than covered by the table, use the last value.
            auto eop_last = eop_b - 1;
            eop.delta_UT1_inst = eop_last->delta_UT1_inst;
            eop.xp_as = eop_last->xp_as;
            eop.yp_as = eop_last->yp_as;
            WARN("Requesting EOP later than in table. Requested time = {:d} s + {:d} ns. Latest "
                 "UT1 = "
                 "{:d} s + {:d} ns.",
                 t_target / GIGA, t_target % GIGA, eop_last->t_inst / GIGA,
                 eop_last->t_inst % GIGA);
        } else {
            // Interpolate!
            auto eop_a = eop_b - 1;
            // t - ta in ns. Should be > 0
            int64_t diff_ns_a = t_target - eop_a->t_inst;
            // t - tb in ns. Should be < 0
            int64_t diff_ns_b = t_target - eop_b->t_inst;
            // tb - ta in ns.
            int64_t diff_ns = diff_ns_a - diff_ns_b;

            // weights for points a and b.
            double wb = diff_ns_a / ((double)diff_ns);
            double wa = 1.0 - wb;

            // interpolate
            eop.delta_UT1_inst = wa * eop_a->delta_UT1_inst + wb * eop_b->delta_UT1_inst;
            eop.xp_as = wa * eop_a->xp_as + wb * eop_b->xp_as;
            eop.yp_as = wa * eop_a->yp_as + wb * eop_b->yp_as;
        }
    }

    // now that we have a delta_UT1, can compute UT1 and ERA
    int64_t ut1 = get_UT1_from_time(ts_target, eop.delta_UT1_inst);
    double era = get_ERA_from_UT1(ut1, nullptr);

    eop.t_ut1 = ut1;
    eop.ERA_deg = era;

    return eop;
}

EOP CHORDTelescope::get_EOP_at_UT1(int64_t t_ut1) const {
    // Interpolate on the EOP table to find EOP for the given UT1 time.

    EOP eop;
    eop.t_ut1 = t_ut1;

    {
        std::shared_lock lock(_eop_lock);
        // _eop_table is always sorted by instrument time. UT1 is monotonic
        // with instrument time, unless the Earth has been met with catastrophe.
        // Do a quick search for the first table entry with larger UT1 time than
        // the target.
        auto eop_b = std::lower_bound(_eop_table.begin(), _eop_table.end(), eop, EOP_comp_ut1);

        // DUT1, xp_as, and yp_as evolve slowly, on secular time scales, so we
        // interpolate these, and calculate ERA after.

        if (eop_b == _eop_table.begin()) {
            // Time is earlier than covered by the table, use the first value.
            eop.delta_UT1_inst = eop_b->delta_UT1_inst;
            eop.xp_as = eop_b->xp_as;
            eop.yp_as = eop_b->yp_as;
            WARN("Requesting EOP earlier than in table. Requested UT1 = {:d} s + {:d} ns. Earliest "
                 "UT1 "
                 "= {:d} s + {:d} ns.",
                 t_ut1 / GIGA, t_ut1 % GIGA, eop_b->t_ut1 / GIGA, eop_b->t_ut1 % GIGA);
        } else if (eop_b == _eop_table.end()) {
            // Time is later than covered by the table, use the last value.
            auto eop_last = eop_b - 1;
            eop.delta_UT1_inst = eop_last->delta_UT1_inst;
            eop.xp_as = eop_last->xp_as;
            eop.yp_as = eop_last->yp_as;
            WARN("Requesting EOP later than in table. Requested UT1 = {:d} s + {:d} ns. Latest UT1 "
                 "= "
                 "{:d} s + {:d} ns.",
                 t_ut1 / GIGA, t_ut1 % GIGA, eop_last->t_ut1 / GIGA, eop_last->t_ut1 % GIGA);
        } else {
            // Interpolate! Target time = t, in table interval [ta, tb]
            auto eop_a = eop_b - 1;

            // t - ta in ns. Should be > 0
            int64_t diff_ns_a = t_ut1 - eop_a->t_ut1;
            // t - tb in ns. Should be < 0
            int64_t diff_ns_b = t_ut1 - eop_b->t_ut1;

            // tb - ta in ns.
            int64_t diff_ns = diff_ns_a - diff_ns_b;

            // weight for b point
            double wb = diff_ns_a / ((double)diff_ns);
            // weight for a point.
            double wa = 1.0 - wb;

            // interpolate.
            eop.delta_UT1_inst = wa * eop_a->delta_UT1_inst + wb * eop_b->delta_UT1_inst;
            eop.xp_as = wa * eop_a->xp_as + wb * eop_b->xp_as;
            eop.yp_as = wa * eop_a->yp_as + wb * eop_b->yp_as;
        }
    }

    // Now that we have a delta_UT1, can get t_inst and the ERA
    timespec ts_inst = get_time_from_UT1(t_ut1, eop.delta_UT1_inst);
    double era = get_ERA_from_UT1(t_ut1, nullptr);

    eop.t_inst = timespec_to_nanosec_i64(ts_inst);
    eop.ERA_deg = era;

    return eop;
}

const struct dishInfo& CHORDTelescope::get_dish_at_idx(int64_t idx) const {
    return _dish_info_table[idx];
}

// Get the frequency in MHz corresponding to the given freq_id.
double CHORDTelescope::to_freq_MHz(freq_id_t freq_id) const {
    if (freq_id >= nfreq_total) {
        FATAL_ERROR("Invalid frequency ID={:d}, accepted range [0, {:d})", freq_id, nfreq_total);
    }
    // In even Nyquist zones df_MHz < 0 so the freq_ids count down from freq0.
    return freq0_MHz + freq_id * df_MHz;
}

// Get the configured frequency spacing
double CHORDTelescope::freq_width_MHz(freq_id_t) const {
    // In even Nyquist zones df_MHz < 0, so need to take absolute value.
    return std::abs(df_MHz);
}

// Return the configured Nyquist Zone
uint8_t CHORDTelescope::nyquist_zone() const {
    return ny_zone;
}

// Return the total number of frequency channels
uint32_t CHORDTelescope::num_freq() const {
    return nfreq_total;
}

// Stream logic has been moved to the packet capture code in dpdk
// This stub remains to satisfy inheritance and will likely be removed
// in the future, it will abort if called.
freq_id_t CHORDTelescope::to_freq_id(stream_t, uint32_t) const {
    FATAL_ERROR("CHORDTelesope does not support to_freq_id(stream_t)");
    std::abort();
    return 0;
}

// Stream logic has been moved to the packet capture code in dpdk
// This stub remains to satisfy inheritance and will likely be removed
// in the future, it will abort if called.
uint32_t CHORDTelescope::num_freq_per_stream() const {
    FATAL_ERROR("CHORDTelesope does not support num_freq_per_stream()");
    std::abort();
    return 0;
}

EOP CHORDTelescope::build_EOP_from_update(int64_t time_ns, double delta_ut1_inst, double xp_as,
                                          double yp_as) const {

    struct timespec ts_inst = nanosec_i64_to_timespec(time_ns);
    int64_t ut1 = get_UT1_from_time(ts_inst, delta_ut1_inst);
    double era = get_ERA_from_UT1(ut1, nullptr);

    EOP eop{.t_inst = time_ns,
            .t_ut1 = ut1,
            .delta_UT1_inst = delta_ut1_inst,
            .ERA_deg = era,
            .xp_as = xp_as,
            .yp_as = yp_as};

    return eop;
}
void CHORDTelescope::set_dish_info(const kotekan::Config& config, const std::string& path) {

    // Get the number of dishes, make sure its positive.
    _num_dishes = config.get<int32_t>(path, "num_dishes");
    if (_num_dishes <= 0) {
        FATAL_ERROR("CHORDTelescope: num_dishes must be > 0, got: {:d}", _num_dishes);
        std::abort();
    }
    assert(_num_dishes > 0);

    // Get the grid separation distances.
    _dish_separation_ew_m = config.get_default<double>(path, "dish_separation_ew_m", 6.3);
    _dish_separation_ns_m = config.get_default<double>(path, "dish_separation_ns_m", 8.5);

    // Load the dish_inputs table into temporary storage
    std::vector<dishInfo> cfg_tab =
        config.get_default<std::vector<dishInfo>>(path, "dish_inputs", std::vector<dishInfo>());

    // Make real dish table full of Fake dishes.
    _dish_info_table = std::vector<dishInfo>();

    // Set indices for NULL dishes.
    for (int i = 0; i < _num_dishes; i++)
        _dish_info_table.push_back(dishInfo(i));

    // Load the dishes from the config into the table. Make sure dish indices are consistent.
    for (const dishInfo& dish : cfg_tab) {
        int idx = dish.idx;
        if (idx < 0) {
            FATAL_ERROR("dish {:s} has dish_idx {:d}, which mush be >= 0", dish.label, dish.idx);
            std::abort();
        }
        assert(idx >= 0);
        if (idx >= _num_dishes) {
            FATAL_ERROR("dish {:s} has dish_idx {:d}, which mush be < num_dishes ({:d})",
                        dish.label, dish.idx, _num_dishes);
            std::abort();
        }
        assert(idx < _num_dishes);

        if (_dish_info_table[idx].type != InputType::Fake) {
            FATAL_ERROR("dish {:s} has dish_idx {:d}, which is duplicated in `dish_inputs`",
                        dish.label, dish.idx);
        }

        _dish_info_table[dish.idx] = dish;
    }

    // Make dish positions table.
    _dish_positions = std::vector<std::array<double, 3>>();

    // Calculate and fill the dish positions table.
    for (const dishInfo& d : _dish_info_table) {
        _dish_positions.push_back({_dish_separation_ew_m * d.ew_idx + d.feed_pos_disp_m[0],
                                   _dish_separation_ns_m * d.ns_idx + d.feed_pos_disp_m[1],
                                   d.feed_pos_disp_m[2]});
    }
}


bool EOP_comp_time(const EOP& eop1, const EOP& eop2) {
    return eop1.t_inst < eop2.t_inst;
}

bool EOP_comp_ut1(const EOP& eop1, const EOP& eop2) {
    return eop1.t_ut1 < eop2.t_ut1;
}

void to_json(nlohmann::json& j, const EOP& m) {
    assert(j.empty());

    j.emplace("t_inst", m.t_inst);                 // Instrument time, nanoseconds, UNIX epoch.
    j.emplace("t_ut1", m.t_ut1);                   // UT1 time, nanoseconds, J2000(UT1) epoch.
    j.emplace("delta_UT1_inst", m.delta_UT1_inst); // Diff between UT1 and Instrument time, seconds
    j.emplace("ERA_deg", m.ERA_deg);               // Earth Rotation Angle, degrees
    j.emplace("xp_as", m.xp_as);                   // Polar Motion x', in arcseconds.
    j.emplace("yp_as", m.yp_as);                   // Polar Motion y', in arcseconds.
}

void from_json(const nlohmann::json& j, EOP& m) {
    m.t_inst = j.at("t_inst");                 // Instrument time, nanoseconds, UNIX epoch.
    m.t_ut1 = j.at("t_ut1");                   // UT1 time, nanoseconds, J2000(UT1) epoch.
    m.delta_UT1_inst = j.at("delta_UT1_inst"); // Diff between UT1 and Instrument time, seconds
    m.ERA_deg = j.at("ERA_deg");               // Earth Rotation Angle, degrees
    m.xp_as = j.at("xp_as");                   // Polar Motion x', in arcseconds.
    m.yp_as = j.at("yp_as");                   // Polar Motion y', in arcseconds.
}

void to_json(nlohmann::json& j, const dishInfo& d) {
    j = {};
    j.emplace("dish_idx", d.idx);
    j.emplace("ew_idx", d.ew_idx);
    j.emplace("ns_idx", d.ns_idx);
    j.emplace("feed_pos_disp_m", d.feed_pos_disp_m);
    j.emplace("coelev_disp_deg", d.coelev_disp_deg);
    j.emplace("type", d.type);
    j.emplace("label", d.label);
}

void from_json(const nlohmann::json& j, dishInfo& d) {
    d.idx = j.at("dish_idx");
    d.ew_idx = j.at("ew_idx");
    d.ns_idx = j.at("ns_idx");
    d.feed_pos_disp_m = j.at("feed_pos_disp_m");
    d.coelev_disp_deg = j.at("coelev_disp_deg");
    d.type = j.at("type");
    d.label = j.at("label");
}
