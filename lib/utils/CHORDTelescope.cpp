#include "CHORDTelescope.hpp"

#include <assert.h>            // for assert
#include <algorithm>           // for max, min
#include <stdexcept>           // for runtime_error
#include <vector>              // for vector
#include <cmath>               // for sin, cos, floor, ceil, M_PI, abs
#include <chrono>              // for duration_cast, duration, nanoseconds, system_clock

#include "Telescope.hpp"       // for freq_id_t, Telescope, nyquist_zone_t, stream_t, REGISTER_T...
#include "geoUtil.hpp"         // for GeoFrame
#include "kotekanLogging.hpp"  // for FATAL_ERROR_NON_OO, WARN_NON_OO, INFO_NON_OO, DEBUG, FATAL...
#include "restClient.hpp"      // for restClient
#include "timeUtil.hpp"        // for EOP, nanosec_i64_to_timespec
#include "fmt.hpp"             // for compile_string_to_view, format, format_string
#include "json.hpp"            // for basic_json, operator==, json, input_adapter


REGISTER_TELESCOPE(CHORDTelescope, "CHORDTelescope");

#define GIGA 1'000'000'000L
#define SECONDS_PER_DAY 86'400L
#define DAYS_PER_WEEK 7L
#define GPS_WEEK_ROLLOVER 1024L
#define GPS_EPOCH_S                                                                                \
    315964800L // This is the UNIX timestamp of the GPS epoch:
               // 00:00:00, Jan 6, 1980 UTC.
               // It is equal to:
               // (365 * 8   (days in 197[01345789])
               //  + 366 * 2 (days in 197[26])
               //  + 5 )     (Jan 1 midnight to Jan 6 midnight)
               //  * 86400   (UTC seconds per UTC day)
               //  = 315964800

static constexpr double C = 2.99792458e8;
static constexpr double deg2rad = M_PI / 180.0;
static constexpr double arcsec2rad = M_PI / (180.0 * 3600);


// Dish parameters calculation/initialization

GeographicParams GeographicParams::from_config(const kotekan::Config& config,
                                               const std::string& path, double dish_separation_x_m,
                                               double dish_separation_y_m) {
    GeographicParams dish;

    // Instrument geographic coordinates
    dish.dish_coelev_deg = config.get_default<double>(path, "dish_coelev_deg", 0.0);

    // Whether to check for duplicate dish grid locations
    dish.check_duplicate_dish_grid =
        config.get_default<bool>(path, "check_duplicate_dish_grid", true);

    // Set all dish input data: num_dishes, dish_info_table, dish_position, ...
    dish.set_dish_info(config, path, dish_separation_x_m, dish_separation_y_m);

    return dish;
}

void GeographicParams::set_dish_info(const kotekan::Config& config, const std::string& path,
                                     double dish_separation_x_m, double dish_separation_y_m) {
    // Get the number of dishes, make sure its positive.
    num_dishes = config.get<size_t>(path, "num_dishes");
    num_dishes_x = config.get<size_t>(path, "num_dishes_x");
    num_dishes_y = config.get<size_t>(path, "num_dishes_y");

    if (num_dishes == 0) {
        FATAL_ERROR_NON_OO("CHORDTelescope num_dishes must be > 0");
    }
    if (num_dishes > num_dishes_x * num_dishes_y) {
        FATAL_ERROR_NON_OO("CHORDTelescope num_dishes ({:d}) exceeds grid size "
                           "num_dishes_x*num_dishes_y ({:d}*{:d})",
                           num_dishes, num_dishes_x, num_dishes_y);
    }

    // Load the dish_inputs table into temporary storage
    std::vector<dishInfo> cfg_tab =
        // config.get_default<std::vector<dishInfo>>(path, "dish_inputs",
        //                                            std::vector<dishInfo>());
        config.get<std::vector<dishInfo>>(path, "dish_inputs");

    // Make real dish table full of Fake dishes.
    dish_info_table = std::vector<dishInfo>();

    // Set indices for NULL dishes.
    for (size_t i = 0; i < num_dishes; i++)
        dish_info_table.push_back(dishInfo(i));

    // Load the dishes from the config into the table. Make sure dish indices are consistent.
    for (const dishInfo& dish : cfg_tab) {
        dish_index_t idx = dish.idx;
        if (idx < 0) {
            FATAL_ERROR_NON_OO("dish {:s} has dish_idx {:d}, which mush be >= 0", dish.label,
                               dish.idx);
        }
        if (size_t(idx) >= num_dishes) {
            FATAL_ERROR_NON_OO("dish {:s} has dish_idx {:d}, which mush be < num_dishes ({:d})",
                               dish.label, dish.idx, num_dishes);
        }
        if (dish.type != DishType::Fake) {
            if (dish.grid_x_idx < 0 || size_t(dish.grid_x_idx) >= num_dishes_x) {
                FATAL_ERROR_NON_OO("dish {:s} has grid_x_idx {:d}, which must be in [0, {:d})",
                                   dish.label, dish.grid_x_idx, num_dishes_x);
            }
            if (dish.grid_y_idx < 0 || size_t(dish.grid_y_idx) >= num_dishes_y) {
                FATAL_ERROR_NON_OO("dish {:s} has grid_y_idx {:d}, which must be in [0, {:d})",
                                   dish.label, dish.grid_y_idx, num_dishes_y);
            }
        }
        if (dish_info_table.at(idx).type != DishType::Fake) {
            FATAL_ERROR_NON_OO("dish {:s} has dish_idx {:d}, which is duplicated in `dish_inputs`",
                               dish.label, dish.idx);
        }

        dish_info_table[dish.idx] = dish;
    }

    INFO_NON_OO("CHORDTelescope configured with {:d} dishes.  Loaded {:d} from 'dish_inputs'.",
                num_dishes, cfg_tab.size());

    // Make dish positions table.
    dish_positions = std::vector<vec3d_t>();

    // Calculate and fill the dish positions table.
    for (const dishInfo& d : dish_info_table) {
        dish_positions.push_back({dish_separation_x_m * d.grid_x_idx + d.feed_pos_disp_m.at(0),
                                  dish_separation_y_m * d.grid_y_idx + d.feed_pos_disp_m.at(1),
                                  d.feed_pos_disp_m.at(2)});
    }

    using std::max;

    // Create and fill dish grid
    dish_grid = dishGrid(num_dishes_x, num_dishes_y);
    for (dish_index_t dish = 0; dish < dish_index_t(num_dishes); ++dish) {
        const dishInfo& dish_info = dish_info_table.at(dish);
        if (dish_info.type == DishType::Fake)
            continue;
        // Catch inconsistencies
        assert(dish_info.idx == dish);
        dish_index_t& dish_index = dish_grid.dish_index(dish_info.grid_x_idx, dish_info.grid_y_idx);
        if (check_duplicate_dish_grid && dish_index != -1) {
            FATAL_ERROR_NON_OO("dish {:s} has duplicate grid location ({:d},{:d})", dish_info.label,
                               dish_info.grid_x_idx, dish_info.grid_y_idx);
        }
        dish_index = dish;
    }
}


// Frequency params calculation / initialization

freq_id_t FreqParams::max_science_freq_id() const {
    double id = (df_MHz > 0) ? floor((max_science_freq_MHz - freq0_MHz) / df_MHz)
                             : floor((min_science_freq_MHz - freq0_MHz) / df_MHz);

    return static_cast<freq_id_t>(id);
}

freq_id_t FreqParams::min_science_freq_id() const {
    double id = (df_MHz > 0) ? ceil((min_science_freq_MHz - freq0_MHz) / df_MHz)
                             : ceil((max_science_freq_MHz - freq0_MHz) / df_MHz);

    return static_cast<freq_id_t>(id);
}

size_t FreqParams::num_science_freqs() const {
    return max_science_freq_id() - min_science_freq_id() + 1;
}

inline uint64_t FreqParams::compute_dt_ns(size_t fft_length, double sampling_rate_MHz) noexcept {
    const double dt_s = fft_length / (sampling_rate_MHz * 1.0e6);
    return (uint64_t)(dt_s * 1.0e9);
}

inline double FreqParams::compute_freq0_MHz(nyquist_zone_t nz, double sampling_rate_MHz) noexcept {
    // freq0 = floor(nz / 2) * fs (zone 1 => 0, zones 2–3 => fs, 4–5 => 2fs)
    return static_cast<unsigned>(nz / 2u) * sampling_rate_MHz;
}

inline double FreqParams::compute_df_MHz(nyquist_zone_t nz, double sampling_rate_MHz,
                                         size_t fft_length) noexcept {
    // Odd zones count up from freq0, even zones count down.
    const double sign = (nz % 2u == 1u) ? +1.0 : -1.0;
    return sign * sampling_rate_MHz / (double)(fft_length);
}

inline FreqParams::FreqParams(double sampling_rate_MHz_, size_t fft_length_,
                              nyquist_zone_t nyquist_zone_, double min_science_freq_MHz_,
                              double max_science_freq_MHz_) :
    sampling_rate_MHz{sampling_rate_MHz_}, fft_length{fft_length_}, nyquist_zone{nyquist_zone_},
    dt_ns{compute_dt_ns(fft_length_, sampling_rate_MHz_)},
    freq0_MHz{compute_freq0_MHz(nyquist_zone_, sampling_rate_MHz_)},
    df_MHz{compute_df_MHz(nyquist_zone_, sampling_rate_MHz_, fft_length_)},
    num_freqs{static_cast<size_t>(fft_length_ / 2u)}, min_science_freq_MHz{min_science_freq_MHz_},
    max_science_freq_MHz{max_science_freq_MHz_} {}

inline FreqParams FreqParams::from_config(const kotekan::Config& config, const std::string& path) {
    const double sampling_rate_MHz = config.get_default<double>(path, "sampling_rate_MHz", 3.2e3);
    const size_t fft_length = config.get_default<size_t>(path, "fft_length", 16384);
    const nyquist_zone_t nyquist_zone = config.get_default<nyquist_zone_t>(path, "nyquist_zone", 1);

    double max_science_freq_MHz = config.get_default<double>(path, "max_science_freq_MHz", 1500.0);
    double min_science_freq_MHz = config.get_default<double>(path, "min_science_freq_MHz", 300.0);

    if (min_science_freq_MHz > max_science_freq_MHz) {
        FATAL_ERROR_NON_OO(
            "Requested science band has min_science_freq_MHz ({:f}) > max_science_freq_MHz "
            "({:f}).",
            min_science_freq_MHz, max_science_freq_MHz);
    }

    FreqParams freq_params{sampling_rate_MHz, fft_length, nyquist_zone, min_science_freq_MHz,
                           max_science_freq_MHz};

    // Force the requested science band to be inside the physical band covered by the FPGA's FFT.
    const double freq_end =
        freq_params.freq0_MHz + freq_params.df_MHz * double(freq_params.num_freqs - 1);
    const double physical_low = std::min(freq_params.freq0_MHz, freq_end);
    const double physical_high = std::max(freq_params.freq0_MHz, freq_end);

    if (freq_params.min_science_freq_MHz < physical_low
        || freq_params.max_science_freq_MHz > physical_high) {
        ERROR_NON_OO("Requested science band [{:f} MHz, {:f} MHz] exceeds physical band [{:f} MHz, "
                     "{:f} MHz]. "
                     "Clamping to physical band edges.",
                     freq_params.min_science_freq_MHz, freq_params.max_science_freq_MHz,
                     physical_low, physical_high);

        freq_params.min_science_freq_MHz = std::max(freq_params.min_science_freq_MHz, physical_low);
        freq_params.max_science_freq_MHz =
            std::min(freq_params.max_science_freq_MHz, physical_high);
    }

    if (freq_params.min_science_freq_id() >= freq_params.max_science_freq_id()) {
        FATAL_ERROR_NON_OO(
            "Telescope minimum used science frequency ({:f} MHz) is greater than or equal to "
            "maximum used science frequency ({:f} MHz)!",
            freq_params.min_science_freq_MHz, freq_params.max_science_freq_MHz);
    }

    return freq_params;
}


// GPS params initialization

GPSTimeParams GPSTimeParams::from_config(const kotekan::Config& config, const std::string& path) {
    GPSTimeParams gps;

    gps.require_gps = config.get_default<bool>(path, "require_gps", false);
    gps.query_gps = config.get_default<bool>(path, "query_gps", false);
    // gps_host_info names a sibling block (typically /fpga_controller)
    // holding ``host``, ``port``, and optionally ``timing_endpoint`` (the
    // default for gps_endpoint). Required when query_gps is true.
    std::string default_gps_endpoint = "/get-frame0-time";
    if (config.exists(path, "gps_host_info")) {
        const std::string ref = config.get<std::string>(path, "gps_host_info");
        gps.gps_host = config.get<std::string>(ref, "host");
        gps.gps_port = config.get_default<uint32_t>(ref, "port", 54321);
        if (config.exists(ref, "timing_endpoint"))
            default_gps_endpoint = config.get<std::string>(ref, "timing_endpoint");
    } else if (gps.query_gps) {
        FATAL_ERROR_NON_OO("CHORDTelescope ({}): query_gps is true but gps_host_info is not set; "
                           "point it at the FPGA controller block (e.g. /fpga_controller).",
                           path);
    }
    gps.gps_endpoint = config.get_default<std::string>(path, "gps_endpoint", default_gps_endpoint);
    gps.auto_correct_gps_week_rollover =
        config.get_default<bool>(path, "auto_correct_gps_week_rollover", false);

    if (gps.query_gps)
        set_gps_time_params_from_remote(gps); // sets gps_enabled, time0_ns
    if (!gps.gps_enabled)
        set_gps_time_params_from_config(gps, config); // sets gps_enabled, time0_ns
    if (gps.require_gps && !gps.gps_enabled)
        FATAL_ERROR_NON_OO("The system requires a GPS time, but none was found.");

    if (gps.gps_enabled)
        INFO_NON_OO("Telescope configured with GPS time0: {:d} ns", gps.time0_ns);
    else {
        INFO_NON_OO("Telescope GPS time not enabled, using system time.");
        set_gps_time_params_from_system(gps); // sets gps_enabled, time0_ns
    }

    return gps;
}

void GPSTimeParams::set_gps_time_params_from_config(GPSTimeParams& gps,
                                                    const kotekan::Config& config) {
    if (!config.exists("/", "gps_time")) {
        WARN_NON_OO("No GPS time section found. Ignoring.");
        return;
    }

    if (config.exists("/gps_time", "error")) {
        auto error_message = config.get<std::string>("/gps_time", "error");
        WARN_NON_OO("GPS time lookup failed with reason: \n {:s}\n", error_message);
        return;
    }

    if (!config.exists("/gps_time", "frame0_nano")) {
        WARN_NON_OO("No GPS frame0 time found in config.");
        return;
    }

    gps.time0_ns = config.get<uint64_t>("/gps_time", "frame0_nano");
    gps.gps_enabled = true;
}

// static
void GPSTimeParams::set_gps_time_params_from_remote(GPSTimeParams& gps) {

    uint64_t time0_ns = 0;

    // The REST request might fail. The gps struct for this call is `const`,
    // so we provide a different variable to write the time to if successful.
    bool success = get_gps_time0_ns_from_remote(gps, time0_ns, 30);

    // Get out of here if there was a problem.
    if (!success) {
        gps.gps_enabled = false;
        return;
    }

    // Otherwise, we successfully have the time!
    gps.time0_ns = time0_ns;
    gps.gps_enabled = true;
    INFO_NON_OO("GPS frame0 time set to {:d}", gps.time0_ns);
}

void GPSTimeParams::set_gps_time_params_from_system(GPSTimeParams& gps) {
    gps.time0_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
    gps.gps_enabled = false;
}

bool GPSTimeParams::get_gps_time0_ns_from_remote(const GPSTimeParams& gps, uint64_t& time0_ns,
                                                 int timeout) {
    INFO_NON_OO("Requesting GPS time from server: {:s}:{:d}{:s} This might take some time...",
                gps.gps_host, gps.gps_port, gps.gps_endpoint);

    auto reply = restClient::instance().make_request_blocking(gps.gps_endpoint, {}, gps.gps_host,
                                                              gps.gps_port, 0, timeout);

    if (!reply.first) {
        WARN_NON_OO("Failed to get GPS time, using system time");
        return false;
    }

    auto json_reply = nlohmann::json::parse(reply.second);

    if (json_reply.count("error") == 1) {
        std::string error_message = json_reply["error"];
        WARN_NON_OO("Error returned by GPS server, error: {:s}", error_message);
        return false;
    }

    if (json_reply.count("frame0_nano") == 0) {
        WARN_NON_OO(
            "No `frame0_nano` value returned by GPS server, the server reply was: {:s} - {:s}",
            reply.second, json_reply.dump());
        return false;
    }

    uint64_t raw_time0_ns = json_reply["frame0_nano"].get<uint64_t>();

    uint64_t week_rollover_offset_ns = 0;

    if (gps.auto_correct_gps_week_rollover) {
        // Number of nanoseconds to adjust for the 1024 week rollover.

        // Can only do this if the GPS Server (fpga_master) sent a "start_ctime"
        if (json_reply.count("start_ctime") == 0) {
            WARN_NON_OO(
                "No `start_ctime` value returned by GPS server, the server reply was: {:s} - {:s}",
                reply.second, json_reply.dump());
            return false;
        }

        // FPGA start time is not subject to the GPS rollover, so that is
        // our baseline.
        uint64_t fpga_start_s = json_reply["start_ctime"].get<uint64_t>();
        uint64_t fpga_start_ns = GIGA * fpga_start_s;

        // nanoseconds in the rollover period
        uint64_t dt_rollover_ns = GPS_WEEK_ROLLOVER * DAYS_PER_WEEK * SECONDS_PER_DAY * GIGA;

        // GPS epoch (when GPS time started) in ns.
        uint64_t gps0_ns = GPS_EPOCH_S * GIGA;

        // Compute the (correct) number of rollovers at the fpga start time
        // and the apparent number of rollovers in the sent GPS time0.
        uint64_t rollovers_at_start = (fpga_start_ns - gps0_ns) / dt_rollover_ns;
        uint64_t rollovers_in_time0 = (raw_time0_ns - gps0_ns) / dt_rollover_ns;

        INFO_NON_OO("Correcting GPS time by {} 1024 week rollover periods.",
                    rollovers_at_start - rollovers_in_time0);

        // add 1 rollover of nanoseconds for each rollover time0 is off fpga_start by.
        week_rollover_offset_ns = dt_rollover_ns * (rollovers_at_start - rollovers_in_time0);
    }

    // Set final time0, raw from GPS plus optional rollover correction.
    time0_ns = raw_time0_ns + week_rollover_offset_ns;

    // Success!
    return true;
}


// CHORD telescope

CHORDTelescope::CHORDTelescope(const kotekan::Config& config, const std::string& path) :
    Telescope(path, config.get<std::string>(path, "log_level"),
              config.get_default<bool>(path, "require_eop", false),
              config.get_default<std::string>(path, "eop_updatable_config", ""),
              grid_frame_from_config(config, path),
              config.get_default<double>(path, "dish_separation_x_m", 6.3),
              config.get_default<double>(path, "dish_separation_y_m", 8.5)),
    // Frequency sampling parameters
    _freq_params(FreqParams::from_config(config, path)),
    // GPS time configuration parameters
    _gps_time_params(GPSTimeParams::from_config(config, path)),
    // Instrument geographic coordinates
    _geographic_params(GeographicParams::from_config(config, path, _feed_sep_x_m, _feed_sep_y_m)),
    _dish_frame(dish_frame_from_config(config, path)) {

    DEBUG("Building CHORDTelescope");
}

CHORDTelescope::~CHORDTelescope() {
    DEBUG("Removing CHORDTelescope");
}

GeoFrame CHORDTelescope::grid_frame_from_config(const kotekan::Config& config,
                                                const std::string& path) {

    vec3d_t grid_x_axis = config.get_default<vec3d_t>(path, "grid_x_axis", {1.0, 0.0, 0.0});
    vec3d_t grid_y_axis = config.get_default<vec3d_t>(path, "grid_y_axis", {0.0, 1.0, 0.0});
    vec3d_t grid_z_axis = vec3d_cross(grid_x_axis, grid_y_axis);

    GeoFrame grid_frame(config.get<std::string>(path, "log_level"), "grid",
                        config.get_default<double>(path, "origin_itrs_lat_deg", 0.0),
                        config.get_default<double>(path, "origin_itrs_lon_deg", 0.0),
                        {0.0, 0.0, 0.0}, grid_x_axis, grid_y_axis, grid_z_axis);

    return grid_frame;
}

GeoFrame CHORDTelescope::dish_frame_from_config(const kotekan::Config& config,
                                                const std::string& path) {

    vec3d_t dish_x = config.get_default<vec3d_t>(path, "dish_elev_axis", {1.0, 0.0, 0.0});
    vec3d_t dish_z = config.get_default<vec3d_t>(path, "dish_vert_axis", {0.0, 0.0, 1.0});
    vec3d_t dish_y = vec3d_cross(dish_z, dish_x);

    GeoFrame dish_frame(config.get<std::string>(path, "log_level"), "dish",
                        config.get_default<double>(path, "origin_itrs_lat_deg", 0.0),
                        config.get_default<double>(path, "origin_itrs_lon_deg", 0.0),
                        {0.0, 0.0, 0.0}, dish_x, dish_y, dish_z);

    return dish_frame;
}

timespec CHORDTelescope::to_time(uint64_t seq) const {
    return nanosec_i64_to_timespec(to_time_ns(seq));
}

int64_t CHORDTelescope::to_time_ns(uint64_t seq) const {
    return _gps_time_params.time0_ns + seq * _freq_params.dt_ns;
}

uint64_t CHORDTelescope::to_seq(timespec time) const {
    return (time.tv_sec * GIGA + time.tv_nsec - _gps_time_params.time0_ns) / _freq_params.dt_ns;
}

bool CHORDTelescope::gps_time_enabled() const {
    return _gps_time_params.gps_enabled;
}

bool CHORDTelescope::query_gps_time0_ns(uint64_t& time0_ns, int timeout) const {

    if (_gps_time_params.query_gps) {
        // If we were set to query GPS, then query the GPS and return.
        return _gps_time_params.get_gps_time0_ns_from_remote(_gps_time_params, time0_ns, timeout);
    }

    // Otherwise, we were set from config, which cannot change. Simply return that value.
    time0_ns = _gps_time_params.time0_ns;
    return true;
}

uint64_t CHORDTelescope::seq_length_nsec() const {
    return _freq_params.dt_ns;
}

double CHORDTelescope::get_dish_coelev_deg() const {
    return _geographic_params.dish_coelev_deg;
}

vec3d_t CHORDTelescope::get_sky_vec_in_grid_coords(double ra, double dec, const EOP& eop) const {

    // Taking the ra & dec to be in CIRS frame

    double phi = deg2rad * ra;
    double theta = deg2rad * (90 - dec);

    // unit vector pointing to ra/dec in spherical coordinates
    // fixed to the Earth.  phi=0 ~ Greenwich
    vec3d_t n_cirs = {cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)};

    DEBUG("n_cirs: {} {} {}", n_cirs[0], n_cirs[1], n_cirs[2]);

    // Transform CIRS -> ITRS -> TOPO -> Telescope.
    vec3d_t n_itrs = vec_cirs_to_itrs(n_cirs, eop);
    vec3d_t n_topo = vec_itrs_to_topo(n_itrs);

    return vec_topo_to_grid(n_topo);
}

vec3d_t CHORDTelescope::get_pointing_vec_in_dish_coords() const {

    // Dish coordinates are fixed with z "up" (co-elevation 0 degrees) and x
    // along the elevation axis of the dish mount.  In this frame the pointing
    // vector is just given by the current elevation.

    double coelev = deg2rad * _geographic_params.dish_coelev_deg;

    // coelev=90 ==> North (y), coelev=0 => Up (z), coelev=-90 -> South (-y)
    vec3d_t n_point = {0.0, sin(coelev), cos(coelev)};

    return n_point;
}

vec3d_t CHORDTelescope::vec_topo_to_dish(const vec3d_t& v_topo) const {
    return _dish_frame.vec_topo_to_frame(v_topo);
}
vec3d_t CHORDTelescope::vec_dish_to_topo(const vec3d_t& v_dish) const {
    return _dish_frame.vec_frame_to_topo(v_dish);
}

vec3d_t CHORDTelescope::vec_axes_rotation_R1(const vec3d_t& v, double theta) const {
    // Return coordinates of vector v in frame rotated by theta about x-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    vec3d_t v_rot = {v[0], cos_th * v[1] + sin_th * v[2], -sin_th * v[1] + cos_th * v[2]};

    return v_rot;
}

vec3d_t CHORDTelescope::vec_axes_rotation_R2(const vec3d_t& v, double theta) const {
    // Return coordinates of vector v in frame rotated by theta about y-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    vec3d_t v_rot = {cos_th * v[0] - sin_th * v[2], v[1], sin_th * v[0] + cos_th * v[2]};

    return v_rot;
}

vec3d_t CHORDTelescope::vec_axes_rotation_R3(const vec3d_t& v, double theta) const {
    // Return coordinates of vector v in frame rotated by theta about z-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    vec3d_t v_rot = {cos_th * v[0] + sin_th * v[1], -sin_th * v[0] + cos_th * v[1], v[2]};

    return v_rot;
}


vec3d_t CHORDTelescope::vec_cirs_to_itrs(const vec3d_t& v_cirs, const EOP& eop) const {

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
    vec3d_t v1 = vec_axes_rotation_R3(v_cirs, era);
    // 5.3, second factor, inverse
    vec3d_t v2 = vec_axes_rotation_R2(v1, -xp);
    // 5.3, first factor, inverse
    vec3d_t v_itrs = vec_axes_rotation_R1(v2, -yp);

    return v_itrs;
}

vec3d_t CHORDTelescope::vec_itrs_to_cirs(const vec3d_t& v_itrs, const EOP& eop) const {

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
    vec3d_t v1 = vec_axes_rotation_R1(v_itrs, yp);
    // 5.3 (Second factor in W)
    vec3d_t v2 = vec_axes_rotation_R2(v1, xp);
    // 5.5
    vec3d_t v_cirs = vec_axes_rotation_R3(v2, -era);

    return v_cirs;
}

void CHORDTelescope::fill_fringestop_phases_1d(double freq_MHz, const EOP& eop, const EOP& eop0,
                                               std::vector<std::complex<float>>& phases) const {

    // Get the pointing vector (phase center) for the telescope in dish coordinates. This is
    // constant in time.
    vec3d_t n_dish0 = get_pointing_vec_in_dish_coords();

    // Transform the pointing vector into topotric coordinates (from which we can
    // transform to the sky), and grid coordinates (where the dish locations live).
    // These are also constant in time.
    vec3d_t n_topo0 = vec_dish_to_topo(n_dish0);
    vec3d_t n_grid0 = vec_topo_to_grid(n_topo0);

    // Take the pointing vector for the telescope and find it in the CIRS frame at ERA0.
    // This is the point we are attempting to stop the fringes at.
    vec3d_t n_itrs0 = vec_topo_to_itrs(n_topo0);
    vec3d_t n_cirs = vec_itrs_to_cirs(n_itrs0, eop0);

    // Now, given this CIRS vector, find its components in the telescope
    // frame at the requested (current) ERA
    vec3d_t n_itrs = vec_cirs_to_itrs(n_cirs, eop);
    vec3d_t n_topo = vec_itrs_to_topo(n_itrs);
    vec3d_t n_grid = vec_topo_to_grid(n_topo);

    // n_grid is now (at ERA) the point on the sky which will be at the
    // phase center (n_grid0) at ERA0.

    // wavenumber for this frequency
    double k = 2 * M_PI * 1e6 * freq_MHz / C;

    for (uint64_t i = 0; i < _geographic_params.dish_positions.size(); i++) {
        double phase = -k
                       * (_geographic_params.dish_positions[i][0] * (n_grid[0] - n_grid0[0])
                          + _geographic_params.dish_positions[i][1] * (n_grid[1] - n_grid0[1])
                          + _geographic_params.dish_positions[i][2] * (n_grid[2] - n_grid0[2]));

        phases[i] = {static_cast<float>(cos(phase)), static_cast<float>(sin(phase))};
    }
}

void CHORDTelescope::fill_input_maps(dishInputFields& input) const {

    auto& num_dishes = _geographic_params.num_dishes;
    auto& dish_info_table = _geographic_params.dish_info_table;

    // Ensure fields have the correct size
    input.grid_x_idx.reserve(num_dishes);
    input.grid_y_idx.reserve(num_dishes);
    input.feed_pos_disp_m.reserve(num_dishes);
    input.coelev_disp_deg.reserve(num_dishes);
    input.type.reserve(num_dishes);
    input.label.reserve(num_dishes);

    input.grid_x_idx.clear();
    input.grid_y_idx.clear();
    input.feed_pos_disp_m.clear();
    input.coelev_disp_deg.clear();
    input.type.clear();
    input.label.clear();

    // Fill them from our internal table.
    for (size_t i = 0; i < num_dishes; i++) {
        input.grid_x_idx.push_back(dish_info_table.at(i).grid_x_idx);
        input.grid_y_idx.push_back(dish_info_table.at(i).grid_y_idx);
        input.feed_pos_disp_m.push_back(dish_info_table.at(i).feed_pos_disp_m);
        input.coelev_disp_deg.push_back(dish_info_table.at(i).coelev_disp_deg);
        input.type.push_back(dish_info_table.at(i).type);
        input.label.push_back(dish_info_table.at(i).label);
    }
}

uint64_t CHORDTelescope::get_num_stacks() const {
    FATAL_ERROR("get_num_stacks() has not been implemented in CHORDTelescope yet.");
    return 0;
}

double CHORDTelescope::get_dish_orientation_el(int i, int j) const {
    return _dish_frame.get_R_topo_to_frame()[i][j];
}

vec3d_t CHORDTelescope::get_dish_position_in_grid_coords(int i) const {
    return _geographic_params.dish_positions[i];
}

size_t CHORDTelescope::get_num_dishes() const {
    return _geographic_params.num_dishes;
}

size_t CHORDTelescope::get_num_dishes_x() const {
    return _geographic_params.num_dishes_x;
}
size_t CHORDTelescope::get_num_dishes_y() const {
    return _geographic_params.num_dishes_y;
}

const dishInfo& CHORDTelescope::get_dish_at_idx(dish_index_t idx) const {
    return _geographic_params.dish_info_table.at(idx);
}

const dishGrid& CHORDTelescope::get_dish_grid() const {
    return _geographic_params.dish_grid;
}

// Get the frequency in MHz corresponding to the given freq_id.
double CHORDTelescope::to_freq_MHz(freq_id_t freq_id) const {
    if (freq_id >= _freq_params.num_freqs) {
        FATAL_ERROR("Invalid frequency ID={:d}, accepted range [0, {:d})", freq_id,
                    _freq_params.num_freqs);
    }
    // In even Nyquist zones _df_MHz < 0 so the freq_ids count down from freq0.
    return _freq_params.freq0_MHz + freq_id * _freq_params.df_MHz;
}

// Get the configured frequency spacing
double CHORDTelescope::freq_width_MHz(freq_id_t) const {
    // In even Nyquist zones _df_MHz < 0, so need to take absolute value.
    return std::abs(_freq_params.df_MHz);
}

// Return the configured Nyquist Zone
nyquist_zone_t CHORDTelescope::nyquist_zone() const {
    return _freq_params.nyquist_zone;
}

// Return the total number of FPGA frequency channels
size_t CHORDTelescope::num_freq() const {
    return _freq_params.num_freqs;
}

freq_id_t CHORDTelescope::max_science_freq_id() const {
    return _freq_params.max_science_freq_id();
}

freq_id_t CHORDTelescope::min_science_freq_id() const {
    return _freq_params.min_science_freq_id();
}

size_t CHORDTelescope::num_science_freqs() const {
    return _freq_params.num_science_freqs();
}

freq_id_t CHORDTelescope::to_freq_id(stream_t stream_id, uint32_t index) const {
    // The frequency ID is currently defined by the following formula:
    // freq_id = min_science_freq_id() + index * 128  + stream_id
    return min_science_freq_id() + index * 128 + stream_id.id;
}

// Currently hard coded in the F-engine packet format.
size_t CHORDTelescope::num_freq_per_stream() const {
    return 48;
}
    
station_id_t CHORDTelescope::element_index_to_station_id(uint64_t el_idx, ElementOrder ord) const {
    if (ord == ElementOrder::CHORDEarly) {
        return el_idx;
    } else if (ord == ElementOrder::CHORDBeamformer) {
        return el_idx;
    }
    FATAL_ERROR("Cannot handle element order {}.", ord);
}

uint64_t CHORDTelescope::station_id_to_element_index(station_id_t st_id, ElementOrder ord) const {
    if (ord == ElementOrder::CHORDEarly) {
        return st_id;
    } else if (ord == ElementOrder::CHORDBeamformer) {
        return st_id;
    }
    FATAL_ERROR("Cannot handle element order {}.", ord);
}

grid_idx_2d_t CHORDTelescope::station_id_to_grid_indices(station_id_t st_id) const {
    return grid_idx_2d_t{static_cast<int32_t>(st_id), static_cast<int32_t>(st_id)};
} 

vec3d_t CHORDTelescope::station_id_to_feed_position_m(station_id_t st_id) const {
    return vec3d_t{static_cast<double>(st_id), static_cast<double>(st_id), static_cast<double>(st_id)};
}

void to_json(nlohmann::json& j, const dishInfo& d) {
    j = {};
    j.emplace("dish_idx", d.idx);
    j.emplace("grid_x_idx", d.grid_x_idx);
    j.emplace("grid_y_idx", d.grid_y_idx);
    j.emplace("feed_pos_disp_m", d.feed_pos_disp_m);
    j.emplace("coelev_disp_deg", d.coelev_disp_deg);
    j.emplace("type", d.type);
    j.emplace("label", d.label);
}

void from_json(const nlohmann::json& j, dishInfo& d) {
    d.idx = j.at("dish_idx");
    d.grid_x_idx = j.at("grid_x_idx");
    d.grid_y_idx = j.at("grid_y_idx");
    d.feed_pos_disp_m = j.at("feed_pos_disp_m");
    d.coelev_disp_deg = j.at("coelev_disp_deg");
    d.type = j.at("type");
    d.label = j.at("label");
}


void to_json(nlohmann::json& j, const DishType& t) {
    switch (t) {
        case DishType::Fake:
            j = "Fake";
            break;
        case DishType::ArrayDish:
            j = "ArrayDish";
            break;
        default:
            throw std::runtime_error(
                fmt::format("to_json - unknown DishType, value: {:d}", static_cast<int32_t>(t)));
            break;
    }
}

void from_json(const nlohmann::json& j, DishType& t) {
    if (j == "Fake")
        t = DishType::Fake;
    else if (j == "ArrayDish")
        t = DishType::ArrayDish;
    else
        throw std::runtime_error(fmt::format("from_json - unknown DishType: {}", j.dump()));
}
