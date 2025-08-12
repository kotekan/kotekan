#ifndef CHORD_TELESCOPE_HPP
#define CHORD_TELESCOPE_HPP

#include "Config.hpp" // for Config
#include "Telescope.hpp"
#include "restServer.hpp"

#include <stdint.h> // for int32_t, uint32_t  TODO: why not cstdint?
#include <string>
#include <time.h>
#include <utility>


/**
 * @brief   Simple struct for containing Earth Orientation Parameter (EOP) data
 */
struct EOP {
    int64_t t_inst;        // Instrument time, nanoseconds, UNIX epoch.
    int64_t t_ut1;         // UT1 time, nanoseconds, J2000(UT1) epoch.
    double delta_UT1_inst; // Diff between UT1 and Instrument time, seconds
    double ERA_deg;        // Earth Rotation Angle, degrees
    double xp_as;          // Polar Motion x', in arcseconds.
    double yp_as;          // Polar Motion y', in arcseconds.
};

// A null (all 0) struct EOP;
const static struct EOP eop_null = {
    .t_inst = 0, .t_ut1 = 0, .delta_UT1_inst = 0.0, .ERA_deg = 0.0, .xp_as = 0.0, .yp_as = 0.0};


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
 * @conf    inst_long_deg       double. Instrument longitude in degrees.
 * @conf    inst_lat_deg        double. Instrument latitude in degrees.
 * @conf    inst_alt_deg        double. Instrument pointing altitude, in
 *                                      degrees from the northern horizon
 *                                      (az=0).
 * @conf    sampling_rate_Hz    double. ADC Sampling Rate (~3.2 GHz)
 * @conf    sampling_rate_Hz    double. F-engine FFT length (~16384)
 * @conf    inst_grid_x_axis    [double, 3].    The basis vector, measured in
 *                                      the topocentric frame, of the dish-dish
 *                                      E/W separation.  Must be:
 *                                      normalized, orthogonal to the y-axis.
 * @conf    inst_grid_y_axis    [double, 3].    The basis vector, measured in
 *                                      the topocentric frame, of the dish-dish
 *                                      N/S separation.  Must be:
 *                                      normalized, orthogonal to the x-axis.
 * @conf    inst_dish_alt_axis  [double, 3].    The basis vector, measured in
 *                                      the topocentric frame, of the dish
 *                                      altitude axis. East-pointing. Must be:
 *                                      normalized, orthogonal to the vert-axis.
 * @conf    inst_dish_vert_axis [double, 3].    The basis vector, measured in
 *                                      the topocentric frame, of the dish
 *                                      vertical direction. Up-pointing, 90 deg
 *                                      altitude. Must be:
 *                                      normalized, orthogonal to the alt-axis.
 * @conf    dish_positions      [[double, 3], N]    List of 3D dish positions,
 *                                      measured in the "Telescope" frame,
 *                                      where the x-axis is dish E/W sep,
 *                                      and the y-axis is dish N/S sep. Should
 *                                      be, within mm, a rectilinear
 *                                      axis-aligned grid in these coordinates.
 **/

/*
 * 2024/10/25: Initial version copied from ICETelescope. Frequency logic
 *              stripped out. GR
 */


class CHORDTelescope : public Telescope {
public:
    CHORDTelescope(const kotekan::Config& config, const std::string& path);

    // Implementations of the required time mapping functions
    bool gps_time_enabled() const override;
    timespec to_time(uint64_t seq) const override;
    uint64_t to_seq(timespec time) const override;
    uint64_t seq_length_nsec() const override;

    /**
     * @brief   Return the longitude of the instrument.
     **/
    double get_inst_long_deg() const;

    /**
     * @brief   Return the latitude of the instrument.
     **/
    double get_inst_lat_deg() const;

    /**
     * @brief   Return the altitude angle of the instrument. 90.0 is up,
     *          0.0 is North.
     **/
    double get_inst_alt_deg() const;

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
     *          phase center), in Dish coordinates (x is altitude axis,
     *          y is ~North, z is altitude = 90deg (~up).
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
     * @param   freq_Hz Frequency to compute phases for.
     * @param   eop     Current EOP.
     * @param   eop0    EOP of phase reference time. if eop=eop0 all phases are
     *                  1.0
     * @param   phases  Vector of std::complex<double>, with size of at
     *                  least num_dishes. The phases will be written to the
     *                  first num_dishes elements of this vector.
     **/
    void fringestop_phases_1d(double freq_Hz, const struct EOP& eop, const struct EOP& eop0,
                              std::vector<std::complex<double>>& phases) const;

    // Implementations of the required frequency mapping functions
    // TODO: Implement these.
    freq_id_t to_freq_id(stream_t stream, uint32_t ind) const override;
    double to_freq(freq_id_t freq_id) const override;
    uint32_t num_freq_per_stream() const override;
    uint32_t num_freq() const override;
    double freq_width(freq_id_t freq_id) const override;
    uint8_t nyquist_zone() const override;

    // A forwarding constructor, such that derived classes can skip the main
    // CHORDTelescope constructor but still construct the Telescope class
    template<typename... Args>
    CHORDTelescope(Args&&... args) : Telescope(std::forward<Args>(args)...){};

    ~CHORDTelescope();

protected:
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
    double _inst_long_deg;
    double _inst_lat_deg;

    // Matrix to transform from local topocentric coordinates to the
    // telescope (ie. dish position) coordinate system.
    double _R_topo_to_tel[3][3];

    // Dish pointing angle.  Measured in degrees from vertical.
    double _inst_alt_deg;

    // Matrix to transform from local topocentric coordinates to the
    // dish (ie. z is dish zenith, x is altitude axis) coordinate system.
    double _R_topo_to_dish[3][3];

    // Matrix to transform vectors from ITRS geocentric coordinates (ECEF) to
    // local topocentric coordinates
    double _R_itrs_to_topo[3][3];

    // Dish positions in dish coordinate system.
    std::vector<std::array<double, 3>> _dish_positions;

    // The time of FPGA frame=0, and the time length of each frame (in ns)
    // time0_ns is a UNIX timestamp, in nanoseconds. It does not include
    // leap seconds.
    bool gps_enabled = false;
    uint64_t time0_ns = 0;
    uint64_t dt_ns;

    // Earth Orientation Parameters
    mutable std::mutex _eop_lock;
    std::vector<struct EOP> _eop_table;
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
