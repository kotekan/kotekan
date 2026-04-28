#ifndef CHORD_TELESCOPE_HPP
#define CHORD_TELESCOPE_HPP

#include "Config.hpp"     // for Config
#include "Telescope.hpp"  // for freq_id_t, Telescope, stream_t
#include "restServer.hpp" // for connectionInstance
#include "timeUtil.hpp"   // for EOP

#include "json.hpp" // for json

#include <array>    // for array
#include <complex>  // for complex
#include <mutex>    // for mutex
#include <stdint.h> // for uint64_t, uint32_t, int64_t, uint8_t
#include <string>   // for string, basic_string
#include <time.h>   // for timespec
#include <utility>  // for forward
#include <vector>   // for vector

// Types for frequency and dish indexing
// (Not necessarily logical IDs)
using freq_index_t = size_t;
using dish_index_t =
    int64_t; // int to allow negative values. TODO: separate types / avoid ad-hoc -1?

/**
 * @brief Enum for denoting the type of dish input into Kotekan
 */
enum class DishType : int32_t {
    Fake = -1,     // Not a real dish
    ArrayDish = 0, // A standard dish in the main array.
};

void to_json(nlohmann::json& j, const DishType& t);
void from_json(const nlohmann::json& j, DishType& t);


/**
 * @brief   Simple struct with needed dish info.
 *
 * @param   idx             dish_index_t Index of this dish (row or column), x polarization, in the
 *                                  standard visibility matrix. The y polarization channel will
 *                                  be at index + num_dishes.
 * @param   grid_x_idx          dish_index_t Grid location E/W (x) index. 0 = westmost column,
 * increases east
 * @param   grid_y_idx          dish_index_t Grid location N/S (y) index. 0 = southmost row,
 * increases north
 * @param   feed_pos_disp_m      std::array<double, 3>   Feed position displacement from grid
 * location, meters, Telescope coordinates: X = dish E/W separation, Y = dish N/S separation.
 * actual_pos = grid_pos + disp
 * @param   coelev_disp_deg double  Co-elevation displacement from target, in degrees.
 *                                  actual_coelev = target_coelev + disp.
 * @param   type            Type of dish input (enum DishType)
 * @param   label           std::string Label for dish. Future: key for layout DB?
 */
struct dishInfo {
    dish_index_t idx;
    dish_index_t grid_x_idx;
    dish_index_t grid_y_idx;
    std::array<double, 3> feed_pos_disp_m;
    double coelev_disp_deg;
    DishType type;
    std::string label;

    /**
     * @brief   Default constructor for a Fake, un-indexed dishInfo.
     */
    dishInfo() :
        idx(-1), grid_x_idx(0), grid_y_idx(0), feed_pos_disp_m({0.0, 0.0, 0.0}),
        coelev_disp_deg(0.0), type(DishType::Fake), label("Fake") {}

    /**
     * @brief   Constructor for a Fake dishInfo with an index.
     */
    dishInfo(dish_index_t idx) :
        idx(idx), grid_x_idx(0), grid_y_idx(0), feed_pos_disp_m({0.0, 0.0, 0.0}),
        coelev_disp_deg(0.0), type(DishType::Fake), label("Fake") {}

    /**
     * @brief   Constructor for dishInfo with all fields.
     */
    dishInfo(dish_index_t idx, dish_index_t grid_x_idx, dish_index_t grid_y_idx,
             const std::array<double, 3>& feed_pos_disp_m, double coelev_disp_deg, DishType type,
             const std::string& label) :
        idx(idx), grid_x_idx(grid_x_idx), grid_y_idx(grid_y_idx), feed_pos_disp_m(feed_pos_disp_m),
        coelev_disp_deg(coelev_disp_deg), type(type), label(label) {}
};

inline bool operator==(const dishInfo& lhs, const dishInfo& rhs) {
    return (lhs.idx == rhs.idx) && (lhs.grid_x_idx == rhs.grid_x_idx)
           && (lhs.grid_y_idx == rhs.grid_y_idx)
           && (lhs.feed_pos_disp_m[0] == rhs.feed_pos_disp_m[0])
           && (lhs.feed_pos_disp_m[1] == rhs.feed_pos_disp_m[1])
           && (lhs.feed_pos_disp_m[2] == rhs.feed_pos_disp_m[2])
           && (lhs.coelev_disp_deg == rhs.coelev_disp_deg) && (lhs.type == rhs.type)
           && (lhs.label == rhs.label);
}

void to_json(nlohmann::json& j, const dishInfo& d);
void from_json(const nlohmann::json& j, dishInfo& d);

/**
 * @brief   Struct containing "input" data fields for file writers. Fields are ordered by their
 * appearance in the standard visibility matrix, ie, the "dish_idx" field in "dish_input" in the
 * config.
 *
 * @param   grid_x_idx          dish_index_t Grid location E/W (x) index. 0 = westmost column,
 * increases east
 * @param   grid_y_idx          dish_index_t Grid location N/S (y) index. 0 = southmost row,
 * increases north
 * @param   feed_pos_disp_m      std::array<double, 3>   Feed position displacement from grid
 * location, meters, Telescope coordinates: X = dish E/W separation, Y = dish N/S separation.
 * actual_pos = grid_pos + disp
 * @param   coelev_disp_deg double  Co-elevation displacement from target, in degrees.
 *                                  actual_coelev = target_coelev + disp.
 * @param   type            DishType Type of dish input. DishType::Fake, DishType::ArrayDish
 * @param   label           std::string Label for dish. Future: key for layout DB?
 */
struct dishInputFields {
    std::vector<dish_index_t> grid_x_idx;
    std::vector<dish_index_t> grid_y_idx;
    std::vector<std::array<double, 3>> feed_pos_disp_m;
    std::vector<double> coelev_disp_deg;
    std::vector<DishType> type;
    std::vector<std::string> label;
};

/**
 * @class dishGrid
 * @brief 2D logical grid mapping dish indices to grid coordinates.
 *
 * This class represents a regular grid of dish positions with
 * dimensions @c _num_dishes_x by @c _num_dishes_y. The grid is stored
 * as a flat array where the x index is the fast index and the y index
 * is the slow index:
 *
 *   index = x + _num_dishes_x * y
 *
 * Each cell stores a @c dish_index_t giving the dish index at that
 * grid location, or @c -1 if the grid cell is unoccupied.
 *
 * The class provides bounds-checked access to entries via the
 * @c dish_index(grid_x_idx, grid_y_idx) accessors and throws
 * @c std::runtime_error on out-of-range indices or when constructed
 * with an array of the wrong size.
 */
class dishGrid {
    size_t _num_dishes_x, _num_dishes_y;
    // dish index, or -1 if unoccupied.
    // array layout has x as fast and y as slow index.
    std::vector<dish_index_t> _dish_index;

public:
    dishGrid(const dishGrid&) = default;
    dishGrid(dishGrid&&) = default;
    dishGrid& operator=(const dishGrid&) = default;
    dishGrid& operator=(dishGrid&&) = default;

    dishGrid() : dishGrid(0, 0) {}

    dishGrid(size_t num_dishes_x, size_t num_dishes_y) :
        _num_dishes_x(num_dishes_x), _num_dishes_y(num_dishes_y),
        _dish_index(_num_dishes_x * _num_dishes_y, -1) {}

    dishGrid(size_t num_dishes_x, size_t num_dishes_y, std::vector<dish_index_t> dish_index) :
        _num_dishes_x(num_dishes_x), _num_dishes_y(num_dishes_y),
        _dish_index(std::move(dish_index)) {
        if (_dish_index.size() != _num_dishes_x * _num_dishes_y)
            throw std::runtime_error("wrong array size");
    }

    size_t get_num_dishes_x() const {
        return _num_dishes_x;
    }
    size_t get_num_dishes_y() const {
        return _num_dishes_y;
    }

    dish_index_t dish_index(dish_index_t grid_x_idx, dish_index_t grid_y_idx) const {
        if (grid_x_idx < 0 || size_t(grid_x_idx) >= _num_dishes_x)
            throw std::runtime_error("index error");
        if (grid_y_idx < 0 || size_t(grid_y_idx) >= _num_dishes_y)
            throw std::runtime_error("index error");
        return _dish_index.at(grid_x_idx + _num_dishes_x * grid_y_idx);
    }

    dish_index_t& dish_index(dish_index_t grid_x_idx, dish_index_t grid_y_idx) {
        if (grid_x_idx < 0 || size_t(grid_x_idx) >= _num_dishes_x)
            throw std::runtime_error("index error");
        if (grid_y_idx < 0 || size_t(grid_y_idx) >= _num_dishes_y)
            throw std::runtime_error("index error");
        return _dish_index.at(grid_x_idx + _num_dishes_x * grid_y_idx);
    }

    const std::vector<dish_index_t>& get_dish_indices() const {
        return _dish_index;
    }
};

/**
 * @brief Struct containing dish parameters.
 * @details This struct contains all the necessary parameters to describe the dish
 * positions and orientations in a CHORD-like telescope. Generally, this is constructed
 * from configuration parameters using the from_config() static method.
 **/
struct GeographicParams {
    /// Instument geographic coordinates in degrees.
    double origin_itrs_lon_deg = 0.0;
    double origin_itrs_lat_deg = 0.0;

    /// Matrix to transform from local topocentric coordinates to the
    /// grid (ie. dish position) coordinate system.
    double R_topo_to_grid[3][3] = {};

    /// Dish pointing angle.  Measured in degrees from vertical.
    double dish_coelev_deg = 0.0;

    /// Matrix to transform from local topocentric coordinates to the
    /// dish (ie. z is dish zenith, x is elevation axis) coordinate system.
    double R_topo_to_dish[3][3] = {};

    /// Matrix to transform vectors from ITRS geocentric coordinates (ECEF) to
    /// local topocentric coordinates
    double R_itrs_to_topo[3][3] = {};

    /// Total number of dishes in the telescope, each provides 2 polarizations,
    /// so num_elements = 2 * num_dishes.
    size_t num_dishes = 0;
    size_t num_dishes_x = 0;
    size_t num_dishes_y = 0;

    /// Dish-dish grid spacing in the E/W (x) and N/S (y) directions in meters.
    double dish_separation_x_m = 0.0;
    double dish_separation_y_m = 0.0;

    /// Dish positions in dish coordinate system.
    std::vector<std::array<double, 3>> dish_positions;

    /// Full dish info table (Fake + real dishes), used to build dish_positions.
    std::vector<dishInfo> dish_info_table;

    /// Description of the grid of dishes
    dishGrid dish_grid;

    /// Whether to fatal on duplicate dish grid locations (default: true).
    bool check_duplicate_dish_grid = true;

    /**
     * @brief Build full dish parameters (including rotation matrices and positions)
     * from the config for this telescope.
     *
     * @conf   origin_itrs_lon_deg  double. Longitude of the telescope origin in ITRS coords.
     * @conf   origin_itrs_lat_deg  double. Latitude of the telescope origin in ITRS coords.
     * @conf   dish_coelev_deg      double. Dish pointing co-elevation angle.
     * @conf   dish_separation_x_m  double. Dish separation in the E/W (x) direction, meters.
     * @conf   dish_separation_y_m  double. Dish separation in the N/S (y) direction, meters.
     * @conf   grid_x_axis          array<double,3>. Unit vector giving the grid x axis in
     *                              topocentric coords.
     * @conf   grid_y_axis          array<double,3>. Unit vector giving the grid y axis in
     *                              topocentric coords.
     * @conf   dish_elev_axis       array<double,3>. Unit vector giving the dish elevation axis in
     *                              topocentric coords.
     * @conf   dish_vert_axis       array<double,3>. Unit vector giving the dish vertical axis in
     *                              topocentric coords.
     * @conf   dish_inputs          array<dishInfo>. List of dish inputs, with all fields populated.
     *                              See struct dishInfo for field descriptions.
     * @conf   num_dishes           size_t. Total number of dishes.
     * @conf   num_dishes_x         size_t. Number of dishes in the E/W (x) direction.
     * @conf   num_dishes_y         size_t. Number of dishes in the N/S (y) direction.
     * @conf   check_duplicate_dish_grid  bool. Fatal on duplicate dish grid locations.
     *                              Default: true.
     *
     * @param   config  The config.
     * @param   path    This telescope's path in the config.
     *
     * @return  The built GeographicParams.
     **/
    static GeographicParams from_config(const kotekan::Config& config, const std::string& path);

    /**
     * @brief   Set dish information about dish inputs from the config.
     * See from_config() for configuration options.
     *
     * @param   config  The config.
     * @param   path    This telescope's path in the config.
     **/
    void set_dish_info(const kotekan::Config& config, const std::string& path);
};


/**
 * @brief   Struct containing "frequency" data fields.
 **/
struct FreqParams {

    // basic config values
    double sampling_rate_MHz;
    size_t fft_length;
    nyquist_zone_t nyquist_zone; // CHORD is 1, CHIME is 2

    // derived timing/frequency quantities
    uint64_t dt_ns;   /// time in nanoseconds between fpga_seq_nums (ie. the time between
                      /// fft_length raw ADC samples)
    double freq0_MHz; /// freq0 = floor(nyquist_zone / 2) * sampling_rate_MHz (matches ICETelescope
                      /// convention: zone 1 => 0, zones 2–3 => fs, zones 4–5 => 2fs, ...).
    double df_MHz;    /// Odd zones count up from freq0, even zones count down.
    size_t num_freqs; /// Total number of frequency channels (input data is Real)

    // telescope science use limits (in MHz, physical band of interest)
    double min_science_freq_MHz; /// Minimum frequency used for science, in MHz (setting this using
                                 /// from_config will clamp the value to be inside the physical band
                                 /// allowed by fpga frequencies)
    double max_science_freq_MHz; /// Maximum frequency used for science, in MHz (setting this using
                                 /// from_config will clamp the value to be inside the physical band
                                 /// allowed by fpga frequencies)

    /**
     * @brief Build from kotekan::Config
     *
     * @conf   sampling_rate_MHz   double. ADC Sampling Rate (default: 3.2 GHz for CHORD)
     * @conf   fft_length          size_t. F-engine FFT length (default: 16384 for CHORD)
     * @conf   nyquist_zone        nyquist_zone_t.  Nyquist Zone we're operating in (default: 1 for
     *CHORD)
     * @conf   max_science_freq_MHz double. Maximum frequency used for science, in MHz (default:
     *1500.0). Values exceeding the physical band allowed by fpga frequencies will be clamped.
     * @conf   min_science_freq_MHz double. Minimum frequency used for science, in MHz (default:
     *300.0). Values exceeding the physical band allowed by fpga frequencies will be clamped.
     *
     * @param   config  The config.
     * @param   path    This telescope's path in the config.
     *
     * @return  The built FreqParams.
     **/
    static FreqParams from_config(const kotekan::Config& config, const std::string& path);

    freq_id_t max_science_freq_id() const; /// Get the maximum science frequency ID accounting for
                                           /// nyquist zones. Called by telescope object.
    freq_id_t min_science_freq_id() const; /// Get the minimum science frequency ID accounting for
                                           /// nyquist zones. Called by telescope object.
    size_t num_science_freqs()
        const; /// Get the number of science frequencies between min and max science freq IDs.

private:
    // Constructor using primitive values
    FreqParams(double sampling_rate_MHz_, size_t fft_length_, nyquist_zone_t nyquist_zone_,
               double min_science_freq_MHz_, double max_science_freq_MHz_);

    /// Find the time in nanoseconds between fpga_seq_nums (ie. the time between fft_length raw ADC
    /// samples)
    static uint64_t compute_dt_ns(size_t fft_length, double sampling_rate_MHz) noexcept;

    /// Set the physical frequency of id=0, and the spacing, taking into account
    /// the aliasing of each Nyquist zone.
    // freq0 = floor(nz / 2) * fs (zone 1 => 0, zones 2–3 => fs, 4–5 => 2fs)
    static double compute_freq0_MHz(nyquist_zone_t nz, double sampling_rate_MHz) noexcept;

    // Odd nyquist zones count up from freq0, even zones count down.
    static double compute_df_MHz(nyquist_zone_t nz, double sampling_rate_MHz,
                                 size_t fft_length) noexcept;
};

/**
 * @brief   Struct containing "GPS" data fields.
 * @details This struct contains all the necessary parameters to describe the GPS
 * time configuration in a CHORD-like telescope. Generally, this is constructed
 * from configuration parameters using the from_config() static method.
 **/
struct GPSTimeParams {
    /// Should we require GPS time to be available
    bool require_gps = false;

    /// Should we try to get the GPS time from remote server
    bool query_gps = false;

    /// The GPS server IP address
    std::string gps_host;

    /// The port number on the GPS server
    uint32_t gps_port = 0;

    /// The endpoint with the GPS time
    std::string gps_endpoint;

    /// Whether to correct gps week rollover (1024 weeks) on time0
    bool auto_correct_gps_week_rollover = false;

    /// Whether GPS time has been enabled
    bool gps_enabled = false;

    /// The time of FPGA seq num = 0,
    /// time0_ns is a UNIX timestamp, in nanoseconds. It does not include
    /// leap seconds.
    uint64_t time0_ns = 0;

    /**
     * @brief Build GPS parameters from config, including (optionally) querying
     * a remote server and enforcing _require_gps.
     *
     * @conf    require_gps         bool.   If true, exception is thrown if GPS
     *                                      unavailable.
     * @conf    query_gps           bool.   Should the telescope object get the GPS
     *                                      from a remote source. If not available,
     *                                      or false, will try to retrieve from
     *                                      config.
     * @conf    gps_host            string. The GPS server IP address.
     * @conf    gps_port            uint.   The port number on the GPS server.
     * @conf    gps_endpoint        string. The enpoint with the GPS time.
     * @conf    auto_correct_gps_week_rollover  bool.   If true, correct GPS time0
     *                                      for the 1024 week GPS rollover using
     *                                      start_ctime from the GPS server.
     **/
    static GPSTimeParams from_config(const kotekan::Config& config, const std::string& path);

    /// Retrieve the time0_ns from a remote server (fpga_master)
    static bool get_gps_time0_ns_from_remote(const GPSTimeParams& gps, uint64_t& time0_ns,
                                             int timeout);

private:
    /// Set the GPS time parameters (gps_enabled, time0_ns) from the config.
    static void set_gps_time_params_from_config(GPSTimeParams& gps, const kotekan::Config& config);

    /// Set the GPS time parameters (gps_enabled, time0_ns) from a remote server (fpga_master)
    static void set_gps_time_params_from_remote(GPSTimeParams& gps);

    /// Set the GPS time parameters (gps_enabled, time0_ns) from the system clock.
    static void set_gps_time_params_from_system(GPSTimeParams& gps);
};


/**
 * @brief Implementation for a CHORD-like telescope.
 *
 * Configuration is split across dedicated helper structs:
 * @see GPSTimeParams::from_config (GPS requirements and time source)
 * @see FreqParams::from_config (frequency sampling)
 * @see GeographicParams::from_config (dish and array geometry)
 *
 * @conf    eop_updatable_config    string. ConfigUpdater path for dynamic EOP updates. For more
 *information see Telescope.hpp
 * @conf    log_level           string. Optional log level for this telescope instance.
 *
 * @author Geoffrey Ryan
 **/

/*
 * 2024/10/25: Initial version copied from ICETelescope. Frequency logic
 *              stripped out. GR
 * 2025/11/10: Required frequency logic re-added (no stream_t behaviour). Dish input table
 *              introduced with per-dish grid placement, positioning, pointing, and labels. GR
 * 2025/12/04: Factor telescope members into logical groups: frequency parameters,
 *              gps time parameters, geographic/dish parameters. JM (PR #1373)
 * 2026/01/28:  Remove EOP functionality and move to Telescope.
 */

class CHORDTelescope : public Telescope {
public:
    CHORDTelescope(const kotekan::Config& config, const std::string& path);

    /**
     * @brief Is the GPS time source enabled?
     *
     * @return  True if the GPS time is available.
     **/
    bool gps_time_enabled() const override;

    /**
     * Convert an FPGA sequence number into an instrument time (UNIX epoch time at start plus TAI
     *time elapsed since start).
     *
     * @param  seq  The FPGA sequence number.
     *
     * @return  The corresponding instrument time (UNIX epoch time at start plus TAI time elapsed
     * since start).
     **/
    timespec to_time(uint64_t seq) const override;

    /**
     * @brief Convert an instrument time (UNIX epoch time at start plus TAI time elapsed since
     * start) into the nearest sequence number.
     *
     * @note When there is not an exact correspondence between the given time
     *       and FPGA sequence numbers, this routine will return the latest valid
     *       FPGA sequence number before the given timestamp.
     *
     * @param  time  The instrument time.
     *
     * @return  The corresponding FPGA sequence number.
     **/
    uint64_t to_seq(timespec time) const override;

    /**
     * @brief Get the length in nanoseconds of an FPGA sequence number tick.
     *
     * @return  Length of an FPGA sequence number tick in nanoseconds.
     **/
    uint64_t seq_length_nsec() const override;

    /**
     * @brief   Return the time corresponding to the given fpga sequence number as an int64_t.
     *          Uses the epoch of time0_ns.
     */
    int64_t to_time_ns(uint64_t seq) const override;

    /**
     * @brief   Queries the source of the GPS time0_ns value. Returns true on success, and updates
     *          the value of the given reference. This is a BLOCKING call, may take several seconds
     * to resolve, and should NOT be used by time-critical functions.
     *
     *  @param  time0_ns    The value of time0_ns, updated on succes, otherwise untouched.
     *  @param  timeout     Timeout length for REST call in seconds, -1 for no timeout.
     *
     *  @return true if query was successful and time0_ns is valid. false otherwise.
     */
    bool query_gps_time0_ns(uint64_t& time0_ns, int timeout) const;

    /**
     * @brief   Return the longitude of the instrument.
     **/
    double get_origin_itrs_lon_deg() const;

    /**
     * @brief   Return the latitude of the instrument.
     **/
    double get_origin_itrs_lat_deg() const;

    /**
     * @brief   Return the co-elevation angle of the instrument. 0.0 is up,
     *          90.0 is North.
     **/
    double get_dish_coelev_deg() const;

    /**
     * @brief   Return a component of the Topo -> Grid frame rotation
     *          matrix.
     *
     * @param   i   First index, int, 0 <= i < 3, row
     * @param   j   First index, int, 0 <= j < 3, col
     **/
    double get_grid_orientation_el(int i, int j) const;

    /**
     * @brief   Return a component of the Topo -> Dish frame rotation matrix.
     *
     * @param   i   First index, int, 0 <= i < 3, row
     * @param   j   First index, int, 0 <= j < 3, col
     **/
    double get_dish_orientation_el(int i, int j) const;

    /**
     * @brief   Return a dish location, in grid coordinates.
     *
     * @param   i   Dish index, int, 0 <= i < num_dishes
     **/
    std::array<double, 3> get_dish_position_in_grid_coords(int i) const;

    /**
     * @brief   Return the number of dishes.
     **/
    size_t get_num_dishes() const;

    size_t get_num_dishes_x() const;
    size_t get_num_dishes_y() const;

    double get_dish_separation_x_m() const;
    double get_dish_separation_y_m() const;


    /**
     * @brief get information about a specific dish.
     **/
    const dishInfo& get_dish_at_idx(int64_t idx) const;

    const dishGrid& get_dish_grid() const;

    /**
     * @brief   Return an observing vector (normalized vec3) in grid
     *          coordinates, corresponding to the given CIRS RA and DEC.
     * @param   ra  Target Right Ascension in CIRS frame.
     * @param   dec Target Declination in CIRS frame.
     * @param   eop EOP for the time of observation.
     **/
    std::array<double, 3> get_sky_vec_in_grid_coords(double ra, double dec, const EOP& eop) const;

    /**
     * @brief   Return the pointing vector (direction dish is pointing, the
     *          phase center), in Dish coordinates (x is elevation axis (~East),
     *          y is ~North, z is co-elevation = 0 deg (~up).
     **/
    std::array<double, 3> get_pointing_vec_in_dish_coords() const;

    /**
     * @brief   Transform the given vector from topocentric to dish coords.
     *
     * @param   v_topo  Vector in topocentric coordinates.
     **/
    std::array<double, 3> vec_topocen_to_dish(const std::array<double, 3>& v_topo) const;

    /**
     * @brief   Transform the given vector from Dish to Topocentric coords.
     *
     * @param   v_topo  Vector in dish coordinates.
     **/
    std::array<double, 3> vec_dish_to_topocen(const std::array<double, 3>& v_dish) const;

    /**
     * @brief   Transform the given vector from topocentric to grid coords.
     *
     * @param   v_topo  Vector in topocentric coordinates.
     **/
    std::array<double, 3> vec_topocen_to_grid(const std::array<double, 3>& v_topo) const;

    /**
     * @brief   Transform the given vector from grid to topocentric coords.
     *
     * @param   v_grid  Vector in grid coordinates.
     **/
    std::array<double, 3> vec_grid_to_topocen(const std::array<double, 3>& v_grid) const;

    /**
     * @brief   Transform the given vector from ITRS to topocentric coords.
     *
     * @param   v_topo  Vector in ITRS coordinates.
     **/
    std::array<double, 3> vec_itrs_to_topocen(const std::array<double, 3>& v_itrs) const;

    /**
     * @brief   Transform the given vector from topocentric to ITRS coords.
     *
     * @param   v_topo  Vector in topocentric coordinates.
     **/
    std::array<double, 3> vec_topocen_to_itrs(const std::array<double, 3>& v_topo) const;

    /**
     * @brief   Transform the given vector from CIRS to ITRS coords.
     *
     * @param   v_topo  Vector in CIRS coordinates.
     * @param   eop     EOP for time of transformation.
     **/
    std::array<double, 3> vec_cirs_to_itrs(const std::array<double, 3>& v_cirs,
                                           const EOP& eop) const;

    /**
     * @brief   Transform the given vector from ITRS to CIRS coords.
     *
     * @param   v_topo  Vector in ITRS coordinates.
     * @param   eop     EOP for time of transformation.
     **/
    std::array<double, 3> vec_itrs_to_cirs(const std::array<double, 3>& v_itrs,
                                           const EOP& eop) const;

    /**
     * @brief   Transform the given vector to a frame where the basis has
     *          rotated about the x axis by an angle theta.
     *
     * @param   v       Input vector
     * @param   theta   Angle basis is rotated by, in radians.
     **/
    std::array<double, 3> vec_axes_rotation_R1(const std::array<double, 3>& v, double theta) const;

    /**
     * @brief   Transform the given vector to a frame where the basis has
     *          rotated about the y axis by an angle theta.
     *
     * @param   v       Input vector
     * @param   theta   Angle basis is rotated by, in radians.
     **/
    std::array<double, 3> vec_axes_rotation_R2(const std::array<double, 3>& v, double theta) const;

    /**
     * @brief   Transform the given vector to a frame where the basis has
     *          rotated about the z axis by an angle theta.
     *
     * @param   v       Input vector
     * @param   theta   Angle basis is rotated by, in radians.
     **/
    std::array<double, 3> vec_axes_rotation_R3(const std::array<double, 3>& v, double theta) const;

    /**
     * @brief   Compute the fringestopping phases for each dish.
     *
     * @param   freq_MHz Frequency to compute phases for.
     * @param   eop     Current EOP.
     * @param   eop0    EOP of phase reference time. if eop=eop0 all phases are
     *                  1.0
     * @param   phases  Vector of std::complex<double>, with size of at
     *                  least num_dishes. The phases will be written to the
     *                  first num_dishes elements of this vector.
     **/
    void fringestop_phases_1d(double freq_MHz, const EOP& eop, const EOP& eop0,
                              std::vector<std::complex<float>>& phases) const;

    /**
     * @brief   Fill a dishInputFields struct with dish information. Will possibly
     *          reallocate the internal vectors in 'input'.
     *
     * @param   input   struct dishInputFields reference. The struct to fill,
     *                  will reallocate the internal vectors if they are not the
     *                  correct size.
     **/
    void fill_input_maps(dishInputFields& input) const;

    /**
     * @brief Get the number of unique baselines in the array
     */
    size_t get_num_stacks() const;

    /**
     * Get the physical frequency in MHz of the specified freq ID.
     *
     * @param  freq_id  The frequency ID.
     *
     * @returns         The central frequency in MHz.
     **/
    double to_freq_MHz(freq_id_t freq_id) const override;

    /**
     * @brief Get the frequency width of a given channel.  When the frequency spacing is constant,
     *this is the equivalent to the spacing between frequency channels.
     *
     * @return  The width of the frequency channel in MHz.
     **/
    double freq_width_MHz(freq_id_t freq_id) const override;

    /**
     * @brief Get which Nyquist zone we are in.
     *
     * @return  The Nyquist zone.
     **/
    nyquist_zone_t nyquist_zone() const override;

    /**
     * @brief Get the total number of possible FPGA frequencies channels.
     * This does not correspond to frequency channels that are processed,
     * which is given by num_science_freqs().
     *
     * @return  The total number of frequency channels.
     **/
    size_t num_freq() const override;

    /**
     * @brief Get the maximum FPGA frequency ID that will be processed (downstream of FPGA).
     * @returns The maximum frequency ID.
     **/
    freq_id_t max_science_freq_id() const;

    /**
     * @brief Get the minimum FPGA frequency ID that will be processed (downstream of FPGA).
     * @returns The minimum frequency ID.
     **/
    freq_id_t min_science_freq_id() const;

    /**
     * @brief Get the number of frequency channels that will be processed (downstream of FPGA).
     * @returns Number of frequency channels.
     **/
    size_t num_science_freqs() const;

    /**
     * @brief Convert a stream and index within that stream to a global frequency ID.
     */
    freq_id_t to_freq_id(stream_t stream, uint32_t ind) const override;

    /**
     * @brief Return the number of frequencies per F-engine stream.
     */
    size_t num_freq_per_stream() const override;

    /**
     * @brief   Compute the local ERA (eral in SOFA) at the telescope site.
     *
     *  The local ERA is:
     *
     *      ERAL = ERA + longitude(ITRS) + s',
     *
     *  where:
     *      ERA is the Earth Rotation Angle (era00 in SOFA),
     *      longitude(ITRS) is the geodetic longitude of the site in ITRS
     *      s' is the TIO locator (sp00 in SOFA)
     *
     *  This is the equivalent to Local Apparent Sidereal Time (LAST) in the CIO-based
     *  coordinate systems implemented by the IAU in 2000.
     *
     *  The s' is very small, it accrues at 47 microarcseconds per century, and is
     *  ignored in this calculation.
     *
     * @param   eop  An EOP object for the time at which the ERAL is requested.
     *
     * @return The local ERA in degrees.
     **/
    double get_ERAL_deg(EOP& eop) const;

    // A forwarding constructor, such that derived classes can skip the main
    // CHORDTelescope constructor but still construct the Telescope class
    template<typename... Args>
    CHORDTelescope(Args&&... args) : Telescope(std::forward<Args>(args)...){};

    ~CHORDTelescope();

protected:
    // Frequency parameters
    const FreqParams _freq_params;

    // GPS parameters
    const GPSTimeParams _gps_time_params;

    /// Dish / array geometry and coordinate transforms.
    const GeographicParams _geographic_params;
};

#endif // CHORD_TELESCOPE_HPP
