#ifndef ICE_TELESCOPE_HPP
#define ICE_TELESCOPE_HPP

#include "Config.hpp"    // for Config
#include "Telescope.hpp" // for freq_id_t, stream_t, Telescope
#include "geoUtil.hpp"   // for GeoFrame
#include "visUtil.hpp"   // for input_ctype

#include <stdint.h> // for uint32_t, uint8_t, uint64_t
#include <string>   // for string, basic_string
#include <time.h>   // for timespec
#include <utility>  // for forward

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
 * @conf  gps_port        uint.    The port number on the GPS server. Default: 54321
 * @conf  gps_endpoint    string.  The endpoint with the GPS time
 **/
class ICETelescope : public Telescope {
public:
    ICETelescope(const kotekan::Config& config, const std::string& path);

    // Implementations of the required frequency mapping functions
    freq_id_t to_freq_id(stream_t stream, uint32_t ind) const override;
    double to_freq_MHz(freq_id_t freq_id) const override;
    size_t num_freq_per_stream() const override;
    size_t num_freq() const override;
    double freq_width_MHz(freq_id_t freq_id) const override;
    uint8_t nyquist_zone() const override;

    // Implementations of the required time mapping functions
    bool gps_time_enabled() const override;
    timespec to_time(uint64_t seq) const override;
    int64_t to_time_ns(uint64_t seq) const override;
    uint64_t to_seq(timespec time) const override;
    uint64_t seq_length_nsec() const override;
    
    station_id_t element_index_to_station_id(uint64_t el_idx, ElementOrder ord) const override;
    uint64_t station_id_to_element_index(station_id_t st_id, ElementOrder ord) const override;
    grid_idx_2d_t station_id_to_grid_indices(station_id_t st_id) const override; 
    vec3d_t station_id_to_feed_position_m(station_id_t st_id) const override;

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

    static GeoFrame grid_frame_from_config(const kotekan::Config& config, const std::string& path);

    void decode_station_id(station_id_t st_id, uint64_t& cylinder, uint64_t& polarization, uint64_t& dish) const;
    station_id_t encode_station_id(uint64_t cylinder, uint64_t polarization, uint64_t dish) const;

    /**
     * @brief Parse the reordering configuration section
     * @param config    Configuration handle.
     * @param base_path Path into YAML file to search from.
     * @param num_elements Total number of inputs (2048 for production CHIME)
     * @return          Tuple containing a vector of the input reorder map, and a
     *                  vector of the input labels for the index map.
     */
    static std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> parse_reorder_default(const kotekan::Config& config, const std::string& path, uint64_t num_elements);

    static std::vector<station_id_t> invert_reorder_table(const std::vector<uint32_t>& input_reorder);
    
    /// Number of elements (2048 for production CHIME)
    const uint64_t _num_polarizations;
    const uint64_t _num_dishes;
    const uint64_t _num_elements;
    const uint64_t _num_cylinders;
    const uint64_t _num_dishes_per_cylinder;

    // The number of frequencies per stream
    uint32_t _num_freq_per_stream;

    // The zeroth frequency and spacing in MHz.
    double freq0_MHz;
    double df_MHz;
    uint32_t nfreq;
    uint8_t ny_zone;

    /// Should we try to get the GPS time from remote server
    bool _query_gps;

    /// The GPS server IP address. Resolved from the block named by
    /// /<path>/gps_host_info; left empty when that key is absent (in which
    /// case @c _query_gps must also be false).
    std::string _gps_host;

    /// The port number on the GPS server. Same lookup as @c _gps_host.
    uint32_t _gps_port = 0;

    /// The endpoint with the GPS time
    std::string _gps_endpoint;

    // The time of FPGA frame=0, and the time length of each frame (in ns)
    bool gps_enabled = false;
    uint64_t time0_ns = 0;
    uint64_t dt_ns;

    /// Encodes CHIMECorrelator order
    std::vector<uint32_t> _input_reorder;
    /// The inputs index map
    std::vector<input_ctype> _input_map;
    /// Encodes inverse CHIMECorrelator order
    std::vector<station_id_t> _correlator_stations;

    std::vector<vec2d_t> _feed_positions_2d;

    // A forwarding constructor, such that derived classes can skip the main
    // ICETelescope constructor but still construct the Telescope class
    template<typename... Args>
    ICETelescope(uint64_t num_polarizations, uint64_t num_dishes, uint64_t num_cylinders, Args&&... args) : Telescope(std::forward<Args>(args)...),
        _num_polarizations(num_polarizations), _num_dishes(num_dishes),
        _num_elements(num_polarizations * num_dishes), _num_cylinders(num_cylinders),
        _num_dishes_per_cylinder(num_dishes / num_cylinders) {};

private:
    
    static std::tuple<uint32_t, uint32_t, std::string> parse_reorder_single(nlohmann::json j);
    static std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> parse_reorder(nlohmann::json& j);
    static std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> default_reorder(size_t num_elements);
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

#endif // ICE_TELESCOPE_HPP
