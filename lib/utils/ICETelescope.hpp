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
 * @conf  sample_rate   The sampling rate in MHz.
 * @conf  frame_length  The FPGA frame length in samples.
 * @conf  nyquist_zone  The Nyquist zone we are sampling in (zone=1 is standard
 *                      sampling, zone>=2 is alias sampling).
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
    timespec seq_length() const override;

protected:

    /**
     * @brief Set the sampling parameters for the telescope.
     *
     * @param  sample_rate  The sampling rate in MHz.
     * @param  length       The length of each frame.
     * @param  zone         Which Nyquist zone are we sampling in? zone=1 is
     *                      standard sampling, zone>=2 are alias sampling.
     **/
    void set_sampling_params(double sample_rate, uint32_t length, uint8_t zone);

    /**
     * @brief Set the GPS time parameters.
     *
     * @param  time0  The GPS time of seq=0
     **/
    void set_gps(timespec time0);

    // The number of frequencies per stream
    uint32_t _num_freq_per_stream;

    // The zeroth frequency and spacing in MHz.
    double freq0_MHz;
    double df_MHz;
    uint32_t nfreq;

    // The time of FPGA frame=0, and the time length of each frame (in ns)
    bool gps_enabled;
    uint64_t time0_ns;
    uint64_t dt_ns;

    // A forwarding constructor, such that derived classes can skip the main
    // ICETelescope constructor but still construct the Telescope class
    template<typename... Args>
    ICETelescope(Args&&... args) : Telescope(std::forward<Args>(args)...) {};
};


// Old style code ported over
struct ice_stream_id_t {
    uint8_t link_id;
    uint8_t slot_id;
    uint8_t crate_id;
    uint8_t unused;
};


ice_stream_id_t ice_extract_stream_id(const uint16_t encoded_stream_id);
uint16_t ice_encode_stream_id(const ice_stream_id_t s_stream_id);

inline ice_stream_id_t ice_get_stream_id_t(const struct Buffer* buf, int ID) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    return ice_extract_stream_id(chime_metadata->stream_ID);
}

inline void ice_set_stream_id_t(struct Buffer* buf, int ID, ice_stream_id_t stream_id) {
    struct chimeMetadata* chime_metadata = (struct chimeMetadata*)buf->metadata[ID]->metadata;
    chime_metadata->stream_ID = ice_encode_stream_id(stream_id);
}


#endif // ICE_TELESCOPE_HPP