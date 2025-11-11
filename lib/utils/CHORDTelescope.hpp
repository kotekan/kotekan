#ifndef CHORD_TELESCOPE_HPP
#define CHORD_TELESCOPE_HPP

#include "Config.hpp"     // for Config
#include "Telescope.hpp"  // for freq_id_t, Telescope, stream_t
#include "restServer.hpp" // for connectionInstance

#include "json.hpp" // for json

#include <array>    // for array
#include <complex>  // for complex
#include <mutex>    // for mutex
#include <stdint.h> // for uint64_t, uint32_t, int64_t, uint8_t
#include <string>   // for string, basic_string
#include <time.h>   // for timespec
#include <utility>  // for forward
#include <vector>   // for vector


/**
 * @brief   Simple struct for containing Earth Orientation Parameter (EOP) data
 *
 * @param   t_inst          int64_t Instrument time, nanoseconds, UNIX epoch.
 * @param   t_ut1           int64_t UT1 time, nanoseconds, J2000(UT1) epoch.
 * @param   delta_UT1_inst  double  Difference between UT1 and Instrument time, seconds.
 * @param   ERA_deg         double  Earth Rotation Angle, degrees.
 * @param   xp_as           double  Polar Motion x', arcseconds.
 * @param   yp_as           double  Polar Motion y', arcseconds.
 */
struct EOP {
    int64_t t_inst;        // Instrument time, nanoseconds, UNIX epoch.
    int64_t t_ut1;         // UT1 time, nanoseconds, J2000(UT1) epoch.
    double delta_UT1_inst; // Diff between UT1 and Instrument time, seconds
    double ERA_deg;        // Earth Rotation Angle, degrees
    double xp_as;          // Polar Motion x', in arcseconds.
    double yp_as;          // Polar Motion y', in arcseconds.
};

void to_json(nlohmann::json& j, const EOP& m);
void from_json(const nlohmann::json& j, EOP& m);

// A null (all 0) struct EOP;
const static struct EOP eop_null = {
    .t_inst = 0, .t_ut1 = 0, .delta_UT1_inst = 0.0, .ERA_deg = 0.0, .xp_as = 0.0, .yp_as = 0.0};

/**
 * @brief Enum for denoting the type of dish input into Kotekan
 */
enum class InputType : int64_t {
    Fake = -1,          // Not a real dish
    ArrayDish = 0,      // A standard dish in the main array.
};


/**
 * @brief   Simple struct with needed dish info.
 *
 * @param   idx             int64_t Index of this dish (row or column), x polarization, in the
 *                                  standard visibility matrix. The y polarization channel will
 *                                  be at index + num_dishes.
 * @param   ew_idx          int64_t Grid location E/W (x) index. 0 = westmost column, increases east
 * @param   ns_idx          int64_t Grid location N/S (y) index. 0 = southmost row, increases north
 * @param   pos_disp_m      std::array<double, 3>   Position displacement from grid location,
 * meters, Telescope coordinates: X = dish E/W separation, Y = dish N/S separation.  actual_pos =
 * grid_pos + disp
 * @param   coelev_disp_deg double  Co-elevation displacement from target, in degrees.
 *                                  actual_coelev = target_coelev + disp.
 * @param   type            int64_t Type of dish input.  -1 = NULL, 0 = CHORD Dish.
 * @param   label           std::string Label for dish. Future: key for layout DB?
 */
struct dishInfo {
    int64_t idx;
    int64_t ew_idx;
    int64_t ns_idx;
    std::array<double, 3> pos_disp_m;
    double coelev_disp_deg;
    InputType type;
    std::string label;
};

inline bool operator==(const dishInfo& lhs, const dishInfo& rhs) {
    return (lhs.idx == rhs.idx) && (lhs.ew_idx == rhs.ew_idx) && (lhs.ns_idx == rhs.ns_idx)
           && (lhs.pos_disp_m[0] == rhs.pos_disp_m[0]) && (lhs.pos_disp_m[1] == rhs.pos_disp_m[1])
           && (lhs.pos_disp_m[2] == rhs.pos_disp_m[2])
           && (lhs.coelev_disp_deg == rhs.coelev_disp_deg) && (lhs.type == rhs.type)
           && (lhs.label == rhs.label);
}

/**
 * @brief   Function to generate a dishInfo struct from individual members, used in testing.
 */
inline dishInfo make_dishInfo(int64_t idx, int64_t ew_idx, int64_t ns_idx,
                              const std::array<double, 3>& pos_disp_m, double coelev_disp_deg,
                              InputType type, const std::string& label) {
    dishInfo d{.idx = idx,
               .ew_idx = ew_idx,
               .ns_idx = ns_idx,
               .pos_disp_m = {pos_disp_m[0], pos_disp_m[1], pos_disp_m[2]},
               .coelev_disp_deg = coelev_disp_deg,
               .type = type,
               .label = label};
    return d;
}

void to_json(nlohmann::json& j, const dishInfo& d);
void from_json(const nlohmann::json& j, dishInfo& d);

// A null (all 0) struct EOP;
const static struct dishInfo dish_null = {.idx = -1,
                                          .ew_idx = 0,
                                          .ns_idx = 0,
                                          .pos_disp_m = {0.0, 0.0, 0.0},
                                          .coelev_disp_deg = 0.0,
                                          .type = InputType::Fake,
                                          .label = "Fake"};

/**
 * @brief   Struct containing "input" data fields for file writers. Fields are ordered by their
 * appearance in the standard visibility matrix, ie, the "dish_idx" field in "dish_input" in the
 * config.
 *
 * @param   ew_idx          int64_t Grid location E/W (x) index. 0 = westmost column, increases east
 * @param   ns_idx          int64_t Grid location N/S (y) index. 0 = southmost row, increases north
 * @param   pos_disp_m      std::array<double, 3>   Position displacement from grid location,
 * meters, Telescope coordinates: X = dish E/W separation, Y = dish N/S separation.  actual_pos =
 * grid_pos + disp
 * @param   coelev_disp_deg double  Co-elevation displacement from target, in degrees.
 *                                  actual_coelev = target_coelev + disp.
 * @param   type            int64_t Type of dish input.  -1 = NULL, 0 = CHORD Dish.
 * @param   label           std::string Label for dish. Future: key for layout DB?
 */
struct dishInputFields {
    std::vector<int64_t> ew_idx;
    std::vector<int64_t> ns_idx;
    std::vector<std::array<double, 3>> pos_disp_m;
    std::vector<double> coelev_disp_deg;
    std::vector<int64_t> type;
    std::vector<std::string> label;
};


/**
 * @brief Implementation for a CHORD-like telescope.
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
 * @conf    origin_itrs_lon_deg double. Instrument longitude in degrees.
 * @conf    origin_itrs_lat_deg double. Instrument latitude in degrees.
 * @conf    dish_coelev_deg     double. Instrument pointing co-elevation, in
 *                                      degrees from zenith. Positive is North.
 * @conf    sampling_rate_MHz   double. ADC Sampling Rate (default: 3.2 GHz for CHORD)
 * @conf    fft_lenth           double. F-engine FFT length (default: 16384 for CHORD)
 * @conf    nyquist_zone        uint8.  Nyquist Zone we're operating in (default: 1 for CHORD)
 * @conf    origin_itrs_lon_deg double. ITRS longitude of the telescope & topocentric coordinate
 *origin.
 * @conf    origin_itrs_lat_deg double. ITRS latitude of the telescope & topocentric coordinate
 *origin.
 * @conf    grid_x_axis         [double, 3].    The basis vector, measured in
 *                                      the topocentric frame, of the dish-dish
 *                                      E/W separation.  Must be:
 *                                      normalized, orthogonal to the y-axis.
 * @conf    grid_y_axis         [double, 3].    The basis vector, measured in
 *                                      the topocentric frame, of the dish-dish
 *                                      N/S separation.  Must be:
 *                                      normalized, orthogonal to the x-axis.
 * @conf    dish_elev_axis      [double, 3].    The basis vector, measured in
 *                                      the topocentric frame, of the dish
 *                                      elevation axis. East-pointing. Must be:
 *                                      normalized, orthogonal to the vert-axis.
 * @conf    dish_vert_axis      [double, 3].    The basis vector, measured in
 *                                      the topocentric frame, of the dish
 *                                      vertical direction. Up-pointing, 0 deg
 *                                      co-elevation. Must be:
 *                                      normalized, orthogonal to the elev-axis.
 * @conf    num_dishes          int     Total number of dishes in the kotekan data
 *                                      pipeline, each providing 2 polarizations. Equal to
 *                                      the total number of configured dishes, plus possibly
 *                                      some "fake" dishes to keep the number a multiple of
 *                                      32.
 * @conf    dish_separation_ew_m    double.     The separation in meters between dish grid
 *                                      locations in the Telescope x-axis direction
 *                                      (generally, East/West).
 * @conf    dish_separation_ns_m    double.     The separation in meters between dish grid
 *                                      locations in the Telescope y-axis direction
 *                                      (generally, North/South).
 * @conf    dish_inputs         [dishInfo, N]   List of dishInfo structs, each represented
 *                                      by a map with the following keys:
 *                                      - dishIdx   int     Position of this dish in the
 *                                          standard visibility matrix
 *                                      - ew_idx    int     E/W (x) grid position in the
 *                                          main array.
 *                                      - ns_idx    int     N/S (y) grid position in the
 *                                          main array.
 *                                      - pos_disp_m [double, 3]    Displacement from grid
 *                                          position in meters, Telescope frame.
 *                                      - coelev_disp_deg   double  Displacement from
 *                                          target co-elevation, degrees.
 *                                      - type      int64_t     Integer code for type of input,
 *                                          -1: fake "NULL" dish, 0: standard dish.
 *                                      - label     String  Label for input.
 **/

/*
 * 2024/10/25: Initial version copied from ICETelescope. Frequency logic
 *              stripped out. GR
 * 2025/11/10: Required frequency logic re-added (no stream_t behaviour). Dish input table
 *              introduced with per-dish grid placement, positioning, pointing, and labels. GR
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
     * Convert a sequence number into an instrument time (UNIX epoch time at start plus TAI time
     *elapsed since start).
     *
     * @param  seq  The sequence number.
     *
     * @return  The corresponding instrument time (UNIX epoch time at start plus TAI time elapsed
     *since start).
     **/
    timespec to_time(uint64_t seq) const override;

    /**
     * @brief Convert an instrument time (UNIX epoch time at start plus TAI time elapsed since
     *start) into the nearest sequence number.
     *
     * @note When there is not an exact correspondence between the given time
     *       and FPGA sequence numbers, this routine will return the latest valid
     *       FPGA sequence number before the given timestamp.
     *
     * @param  time  The instrument time.
     *
     * @return  The corresponding sequence number.
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
    int64_t to_time_ns(uint64_t seq) const;

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
     * @brief   Return a component of the Topo -> Telescope frame rotation
     *          matrix.
     *
     * @param   i   First index, int, 0 <= i < 3, row
     * @param   j   First index, int, 0 <= j < 3, col
     **/
    double get_tel_orientation_el(int i, int j) const;

    /**
     * @brief   Return a component of the Topo -> Dish frame rotation matrix.
     *
     * @param   i   First index, int, 0 <= i < 3, row
     * @param   j   First index, int, 0 <= j < 3, col
     **/
    double get_dish_orientation_el(int i, int j) const;

    /**
     * @brief   Return a dish location, in the Telescope frame.
     *
     * @param   i   Dish index, int, 0 <= i < num_dishes
     **/
    std::array<double, 3> get_dish_position(int i) const;

    /**
     * @brief   Return the number of dishes.
     **/
    int get_num_dishes() const;

    /**
     * @brief   Return the number of entries in the EOP table.
     **/
    int get_EOP_table_len() const;

    /**
     * @brief   Return the EOP table entry at an index.
     *
     * @param   i   Index of desired EOP entry, 0 <= i < EOP_table_len
     **/
    struct EOP get_EOP_at_idx(uint64_t i) const;

    /**
     * @brief   Return the EOP at the desired instrument time. Will interpolate
     *          over table, use first or last entry if target time is out of
     *          table range.
     *
     * @param   ts  Target instrument time, as a timespec.
     **/
    struct EOP get_EOP_at_time(const timespec& ts) const;

    /**
     * @brief   Return the EOP at the desired UT1 time. Will interpolate
     *          over table, using the first or last entry if target time is
     *          out of table range.
     *
     * @param   ts  Target UT1 time, in nanoseconds since J2000(UT1) int64_t
     **/
    struct EOP get_EOP_at_UT1(int64_t ut1) const;

    const struct dishInfo& get_dish_at_idx(int64_t idx) const;

    /**
     * @brief   Return an observing vector (normalized vec3) in telescope
     *          coordinates, corresponding to the given CIRS RA and DEC.
     * @param   ra  Target Right Ascension in CIRS frame.
     * @param   dec Target Declination in CIRS frame.
     * @param   eop EOP for the time of observation.
     **/
    std::array<double, 3> get_sky_vec_in_tel_coords(double ra, double dec,
                                                    const struct EOP& eop) const;
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
     * @brief   Transform the given vector from topocentric to telescope coords.
     *
     * @param   v_topo  Vector in topocentric coordinates.
     **/
    std::array<double, 3> vec_topocen_to_tel(const std::array<double, 3>& v_topo) const;

    /**
     * @brief   Transform the given vector from telescope to topocentric coords.
     *
     * @param   v_topo  Vector in telescope coordinates.
     **/
    std::array<double, 3> vec_tel_to_topocen(const std::array<double, 3>& v_tel) const;

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
                                           const struct EOP& eop) const;

    /**
     * @brief   Transform the given vector from ITRS to CIRS coords.
     *
     * @param   v_topo  Vector in ITRS coordinates.
     * @param   eop     EOP for time of transformation.
     **/
    std::array<double, 3> vec_itrs_to_cirs(const std::array<double, 3>& v_itrs,
                                           const struct EOP& eop) const;

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
    void fringestop_phases_1d(double freq_MHz, const struct EOP& eop, const struct EOP& eop0,
                              std::vector<std::complex<double>>& phases) const;

    /**
     * @brief   Fill a dishInputFields struct with dish information. Will possibly
     *          reallocate the internal vectors in 'input'.
     *
     * @param   input   struct dishInputFields reference. The struct to fill,
     *                  will reallocate the internal vectors if they are not the
     *                  correct size.
     **/
    void get_dish_inputs(dishInputFields& input) const;

    /**
     * Get the physical frequency in MHz of the specified freq ID.
     *
     * @param  freq_id  The frequency ID.
     *
     * @returns         The central frequency in MHz.
     **/
    double to_freq_MHz(freq_id_t freq_id) const override;

    /**
     * @brief Get the total number of frequencies channels.
     *
     * This is the upper bound for freq_id.
     *
     * @return  The total number of frequency channels.
     **/
    uint32_t num_freq() const override;

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
    uint8_t nyquist_zone() const override;

    /**
     * @brief CHORDTelescope does not implement this function, `stream_t` logic has been moved to
     * dpdk.
     */
    freq_id_t to_freq_id(stream_t stream, uint32_t ind) const override;

    /**
     * @brief CHORDTelescope does not implement this function, `stream_t` logic has been moved to
     * dpdk.
     */
    uint32_t num_freq_per_stream() const override;

    // A forwarding constructor, such that derived classes can skip the main
    // CHORDTelescope constructor but still construct the Telescope class
    template<typename... Args>
    CHORDTelescope(Args&&... args) : Telescope(std::forward<Args>(args)...){};

    ~CHORDTelescope();

protected:
    /**
     * @brief Set the internal parameters `dt_ns`, `freq0_MHz`, `df_MHz`, `nfreq_total`, and
     * `ny_zone` which set the basic time and frequency sampling behaviour.  Reads the
     * `sampling_rate_MHz`, `fft_length`, and `nyquist_zone` Config fields.
     * @param config    The config.
     * @param path      This telescope object's path.
     */
    void set_sampling_params(const kotekan::Config& config, const std::string& path);

    /**
     * @brief Set the GPS time parameters from the config.
     *
     * @param   config  Kotekan config
     **/
    void set_gps(const kotekan::Config& config);

    /**
     * @brief Set the GPS time from a remote server (fpga_master)
     *
     * @param host  The host name of the server with the GPS time information
     * @param port  The port of the server with the GPS time information
     * @param path  The endpoint resource name (e.g. /get-frame0-time)
     **/
    void set_gps(const std::string& host, const uint32_t port, const std::string& path);

    /**
     * @brief Callback to update EOP data
     *
     * @param json JSON reference of the config
     */
    bool receive_eop_updates(nlohmann::json& json);

    /**
     * @brief   Callback to send current EOP table
     *
     * @param   conn    Kotekan connection.
     */
    void send_eop_table(kotekan::connectionInstance& conn);

    /**
     * @brief   Callback to send time0_ns value
     *
     * @param   conn    Kotekan connection.
     */
    void send_time0_ns(kotekan::connectionInstance& conn);

    /**
     * @brief   Build a single EOP struct from config values
     *
     * @param   t_ns    Instrument time in nanoseconds.
     * @param   delta_ut1_inst  Diff between UT1 and Instrument time in seconds
     * @param   xp_as   Polar Motion x' coordinate in arcseconds
     * @param   yp_as   Polar Motion y' coordinate in arcseconds
     **/
    struct EOP build_EOP_from_update(int64_t t_ns, double delta_ut1_inst, double xp_as,
                                     double yp_as) const;

    /**
     * @brief   Load information about dish inputs from the config.
     *
     * @param   config  The config.
     * @param   path    This telescope's path in the config.
     **/
    void set_dish_info(const kotekan::Config& config, const std::string& path);

    // The telescope's name in the config
    std::string _unique_name;

    /// Should we try to get the GPS time from remote server
    bool _query_gps;

    /// The GPS server IP address
    std::string _gps_host;

    /// The port number on the GPS server
    uint32_t _gps_port;

    /// The endpoint with the GPS time
    std::string _gps_endpoint;

    /// Instument geographic coordinates in degrees.
    double _origin_itrs_lon_deg;
    double _origin_itrs_lat_deg;

    // Matrix to transform from local topocentric coordinates to the
    // telescope (ie. dish position) coordinate system.
    double _R_topo_to_tel[3][3];

    // Dish pointing angle.  Measured in degrees from vertical.
    double _dish_coelev_deg;

    // Matrix to transform from local topocentric coordinates to the
    // dish (ie. z is dish zenith, x is elevation axis) coordinate system.
    double _R_topo_to_dish[3][3];

    // Matrix to transform vectors from ITRS geocentric coordinates (ECEF) to
    // local topocentric coordinates
    double _R_itrs_to_topo[3][3];

    // Total number of dishes in the telescope, each provides 2 polarizations,
    // so num_elements = 2 * num_dishes.
    int32_t _num_dishes;

    // Dish-dish grid spacing in the E/W (x) and N/S (y) directions in meters.
    double _dish_separation_ew_m;
    double _dish_separation_ns_m;

    // Dish positions in dish coordinate system.
    std::vector<std::array<double, 3>> _dish_positions;

    // The time of FPGA frame=0, and the time length of each frame (in ns)
    // time0_ns is a UNIX timestamp, in nanoseconds. It does not include
    // leap seconds.
    bool gps_enabled = false;
    uint64_t time0_ns = 0;
    uint64_t dt_ns;
    uint8_t ny_zone;
    uint64_t nfreq_total;
    double freq0_MHz;
    double df_MHz;

    // Earth Orientation Parameters
    mutable std::shared_mutex _eop_lock;
    std::vector<struct EOP> _eop_table;

    // Dish Properties
    std::vector<struct dishInfo> _dish_info_table;
};

/*
 * @brief   Comparison function for searching/sorting the EOP table. Compares
 *          EOP based on t_inst, orders chronologically.
 *
 * @params  eop1    First EOP to compare.
 * @params  eop2    Second EOP to compare.
 **/
bool EOP_comp_time(const struct EOP& eop1, const struct EOP& eop2);

/*
 * @brief   Comparison function for searching/sorting the EOP table. Compares
 *          EOP based on t_ut1, orders by increasing rotation. Will produce the
 *          same order as t_inst, unless something is apocalyptically wrong.
 *
 * @params  eop1    First EOP to compare.
 * @params  eop2    Second EOP to compare.
 **/
bool EOP_comp_ut1(const struct EOP& eop1, const struct EOP& eop2);

#endif // CHORD_TELESCOPE_HPP
