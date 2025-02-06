#ifndef CHORD_TELESCOPE_HPP
#define CHORD_TELESCOPE_HPP

#include "Config.hpp" // for Config
#include "Telescope.hpp"

#include <stdint.h> // for int32_t, uint32_t  TODO: why not cstdint?
#include <string>
#include <time.h>
#include <utility>

/**
 * @brief Implementation for a CHORD-like telescope.
 *
 * @conf    require_gps     bool.   If true, exception is thrown if GPS
 *                                  unavailable.
 * @conf    query_gps       bool.   Should the telescope object get the GPS
 *                                  from a remote source. If not available,
 *                                  or false, will try to retrieve from config.
 * @conf    gps_host        string. The GPS server IP address.
 * @conf    gps_port        uint.   The port number on the GPS server.
 * @conf    gps_endpoint    string. The enpoint with the GPS time.
 * @conf    inst_long_deg   double. Instrument longitude in degrees.
 * @conf    inst_lat_deg    double. Instrument latitude in degrees.
 * @conf    inst_alt_deg    double. Instrument pointing altitude, in degrees
 *                                  from the northern horizon (az=0).
 **/

/*
 * 2024/10/25: Initial version copied from ICETelescope. Frequency logic
 *              stripped out. GR
 */

struct EOP {
    struct timespec t_gps;
    struct timespec t_ut1;
    uint64_t seq;
    double ERA_deg;
    double xp_as;
    double yp_as;
};

class CHORDTelescope : public Telescope {
public:
    CHORDTelescope(const kotekan::Config& config, const std::string& path);

    // Implementations of the required time mapping functions
    bool gps_time_enabled() const override;
    timespec to_time(uint64_t seq) const override;
    uint64_t to_seq(timespec time) const override;
    uint64_t seq_length_nsec() const override; 

    double get_inst_long_deg() const;
    double get_inst_lat_deg() const;
    double get_orientation_el(int i, int j) const;
    double get_dish_coord(int i, int j) const;
    int get_num_dishes() const;
    double get_dut1() const;
    double get_dtai() const;
    struct EOP get_EOP_at_seq(uint64_t seq) const;
    struct EOP get_EOP_at_ERA(double ERA_deg) const;
    std::array<double, 3> get_sky_vec_in_dish_coords(double ra, double dec,                                                 double era) const;

    std::array<double, 3> get_pointing_vec_in_tel_coords() const;

    std::array<double, 3> topocen_vec_to_tel_vec(
            const std::array<double, 3>& v_topo) const;
    std::array<double, 3> tel_vec_to_topocen_vec(
            const std::array<double, 3>& v_tel) const;
    std::array<double, 3> itrs_vec_to_topocen_vec(
            const std::array<double, 3>& v_itrs) const;
    std::array<double, 3> topocen_vec_to_itrs_vec(
            const std::array<double, 3>& v_topo) const;

    std::array<double, 3> vec_axes_rotation_R1(
        const std::array<double, 3>& v, double theta) const;
    std::array<double, 3> vec_axes_rotation_R2(

        const std::array<double, 3>& v, double theta) const;
    std::array<double, 3> vec_axes_rotation_R3(
        const std::array<double, 3>& v, double theta) const;

    std::array<double, 3> cirs_vec_to_itrs_vec(
        const std::array<double, 3>& v_cirs, double era_deg,
        double xp_as, double yp_as) const;
    std::array<double, 3> itrs_vec_to_cirs_vec(
        const std::array<double, 3>& v_itrs, double era_deg,
        double xp_as, double yp_as) const;

    void fringestop_phases_1d(double freq_Hz, double era_deg, double xp_as,
        double yp_as, double era_deg0, double xp_as0, double yp_as0,
        std::vector<std::complex<double>>& phases) const;

    // Implementations of the required frequency mapping functions
    // TODO: These are not necessary for CHORD and should maybe be shunted to
    // a different part of the inheritance tree.
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
    void set_gps(const std::string& host, const uint32_t port,
                 const std::string& path);

    /**
     * @brief Callback to update UT1 data
     *
     * @param json JSON reference of the config
     */
    bool receive_eop_updates(nlohmann::json& json);

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

    // Dish pointing angle.  Measured in degrees from the Northern horizon.
    double _inst_alt_deg;

    // Matrix to transform from local topocentric coordinates to the 
    // dish coordinate system.
    double _inst_orientation[3][3];

    // Matrix to transform vectors from ITRS geocentric coordinates (ECEF) to
    // local topocentric coordinates
    double _R_itrs_to_topo[3][3];

    // Dish positions in dish coordinate system.
    std::vector<std::array<double, 3>> _dish_positions;

    // The time of FPGA frame=0, and the time length of each frame (in ns)
    // TODO: Document precisely what epoch the time0 is measured from, whether
    // it includes leap seconds, etc.
    bool gps_enabled = false;
    uint64_t time0_ns = 0;
    uint64_t dt_ns;

    //Earth Orientation Parameters
    mutable std::mutex _eop_lock;
    double _dut1;  // UT1 - UTC in seconds
    double _dtai;  // TAI - UTC in seconds
    double _x_pm;  // Polar Motion x coordinate (arcseconds)
    double _y_pm;  // Polat Motion y coordinate (arcseconds)

};

#endif // CHORD_TELESCOPE_HPP
