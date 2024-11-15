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
 * @conf    inst_long       double. Instrument longitude.
 * @conf    inst_lat        double. Instrument latitude.
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

    double get_inst_long() const;
    double get_inst_lat() const;
    double get_orientation_el(int i, int j) const;
    double get_dut1() const;

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
    bool receive_ut1_updates(nlohmann::json& json);

    /// Should we try to get the GPS time from remote server
    bool _query_gps;

    /// The GPS server IP address
    std::string _gps_host;

    /// The port number on the GPS server
    uint32_t _gps_port;

    /// The endpoint with the GPS time
    std::string _gps_endpoint;

    /// Instument geographic coordinates
    double _inst_long;
    double _inst_lat;
    double _inst_orientation[3][3];

    // The time of FPGA frame=0, and the time length of each frame (in ns)
    // TODO: Document precisely what epoch the time0 is measured from, whether
    // it includes leap seconds, etc.
    bool gps_enabled = false;
    uint64_t time0_ns = 0;
    uint64_t dt_ns;

    //UT1 time conversion data
    mutable std::mutex _ut1_lock;
    double _dut1;  // UT1 - UTC in seconds
};

#endif // CHORD_TELESCOPE_HPP
