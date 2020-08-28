#ifndef ICE_TELESCOPE_HPP
#define ICE_TELESCOPE_HPP

#include "Config.hpp" // for Config
#include "Telescope.hpp"
#include "chimeMetadata.h"

#include <stdint.h> // for int32_t, uint32_t
#include <string>   // for string
#include <time.h>
#include <utility>

/**
 * @brief Implementation for an ICEboard like telescope with a Casper PFB.
 *
 * @conf  sample_rate     int.     The sampling rate in MHz.
 * @conf  fft_length      int.     The FFT frame length in samples. Note this is the length of a
 *                                 basic FFT, the ICEBoard uses a PFB with a 8096 sample window,
 *                                 but the effective length is 2048 for computing the number
 *                                 of frequency bins.
 * @conf  nyquist_zone    int.     The Nyquist zone we are sampling in (zone=1 is standard
 *                                 sampling, zone>=2 is alias sampling).
 * @conf  query_gps       bool.    Should the telescope object get the GPS from a remote source
 *                                 if not available, or false, will try to retrieve from config.
 * @conf  gps_host        string.  The GPS server IP address
 * @conf  gps_port        uint.    The port number on the GPS server
 * @conf  gps_endpoint    string.  The endpoint with the GPS time
 **/
class ICETelescope : public Telescope {
public:
    ICETelescope(const kotekan::Config& config, const std::string& path);

    // Implementations of the required frequency mapping functions
    freq_id_t to_freq_id(stream_t stream, uint32_t ind) const override;
    double to_freq(freq_id_t freq_id) const override;
    uint32_t num_freq_per_stream() const override;
    uint32_t num_freq() const override;
    double freq_width(freq_id_t freq_id) const override;

    // Implementations of the required time mapping functions
    bool gps_time_enabled() const override;
    timespec to_time(uint64_t seq) const override;
    uint64_t to_seq(timespec time) const override;
    uint64_t seq_length_nsec() const override;

protected:
    /**
     * @brief Set the sampling parameters for the telescope.
     *
     * @param  sample_rate  The ADC sampling rate in MHz.
     * @param  fft_length   The FFT length used in the FPGA.  Note this is the length of a
     *                      basic FFT, the ICEBoard uses a PFB with a 8096 sample window, but the
     *                      effective length is 2048 for computing the number of frequency bins.
     * @param  zone         Which Nyquist zone are we sampling in? zone=1 is
     *                      standard sampling, zone>=2 are alias sampling.
     **/
    void set_sampling_params(double sample_rate, uint32_t fft_length, uint8_t zone);

    /**
     * @brief Set the sampling parameters for the telescope from the config.
     *
     * @param  config  Kotekan config.
     * @param  path    Kotekan config path.
     **/
    void set_sampling_params(const kotekan::Config& config, const std::string& path);

    /**
     * @brief Set the GPS time parameters from the config.
     *
     * @param  config  Kotekan config.
     **/
    void set_gps(const kotekan::Config& config);

    /**
     * @brief Set the GPS time from a remote server (pychfpga/fpga_master)
     *
     * @param host   The host name of the server with the GPS time information
     * @param port   The port of the server with the GPS time information
     * @param path   The endpoint resource name (e.g. /get-frame0-time)
     */
    void set_gps(const std::string& host, const uint32_t port, const std::string& path);

    // The number of frequencies per stream
    uint32_t _num_freq_per_stream;

    // The zeroth frequency and spacing in MHz.
    double freq0_MHz;
    double df_MHz;
    uint32_t nfreq;

    /// Should we try to get the GPS time from remote server
    bool _query_gps;

    /// The GPS server IP address
    std::string _gps_host;

    /// The port number on the GPS server
    uint32_t _gps_port;

    /// The endpoint with the GPS time
    std::string _gps_endpoint;

    // The time of FPGA frame=0, and the time length of each frame (in ns)
    bool gps_enabled = false;
    uint64_t time0_ns = 0;
    uint64_t dt_ns;

    // A forwarding constructor, such that derived classes can skip the main
    // ICETelescope constructor but still construct the Telescope class
    template<typename... Args>
    ICETelescope(Args&&... args) : Telescope(std::forward<Args>(args)...){};
};


// Old style code ported over
struct ice_stream_id_t {
    uint8_t link_id;
    uint8_t slot_id;
    uint8_t crate_id;
    uint8_t unused;
};


ice_stream_id_t ice_extract_stream_id(const stream_t encoded_stream_id);
stream_t ice_encode_stream_id(const ice_stream_id_t s_stream_id);

inline ice_stream_id_t ice_get_stream_id_t(const struct Buffer* buf, int ID) {
    return ice_extract_stream_id(get_stream_id(buf, ID));
}

inline void ice_set_stream_id_t(struct Buffer* buf, int ID, ice_stream_id_t stream_id) {
    set_stream_id(buf, ID, ice_encode_stream_id(stream_id));
}


#endif // ICE_TELESCOPE_HPP