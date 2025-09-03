#ifndef GPU_TRACKING_BEAMFORM_SIMULATE_HPP
#define GPU_TRACKING_BEAMFORM_SIMULATE_HPP

#include "Config.hpp" // for Config
#include "Stage.hpp"  // for Stage
#include "Telescope.hpp"
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for beamCoord

#include <stdint.h>    // for int32_t, uint32_t
#include <string>      // for string
#include <sys/types.h> // for uint
#include <time.h>      // for timespec
#include <vector>      // for vector

/**
 * @class gpuTrackingBeamformSimulate
 * @brief CPU version of tracking beamformer for verification
 *
 * Read hsaTrackingUpdatePhase and hsaTrackingBeamform to find out what
 * the tracking beamformer does. This is basically the same, except
 * without capability to dynamically update gain paths and RA/Dec
 * (no endpoints).
 *
 * @author Cherry Ng
 *
 */

class gpuTrackingBeamformSimulate : public kotekan::Stage {
public:
    /// Constructor
    gpuTrackingBeamformSimulate(kotekan::Config& config, const std::string& unique_name,
                                kotekan::bufferContainer& buffer_container);
    /// Destructor
    ~gpuTrackingBeamformSimulate();
    /// Main loop, read gains, calculate new phase, brute-force beamform
    void main_thread() override;

private:
    /// Input
    Buffer* input_buf;
    /// Input length
    int input_len;
    /// An intermediate array of unpacked data
    double* input_unpacked;
    /// Output in CPU
    float* cpu_output;
    /// Output
    Buffer* output_buf;
    /// Output length
    int output_len;

    /// Config options:
    /// number of elements = 2048
    uint32_t _num_elements;
    /// number of samples = 49152
    int32_t _samples_per_data_set;
    /// number of beams
    int32_t _num_beams;
    /// number of polarizations = 2
    int32_t _num_pol;
    /// Array of reordering index, length 512
    std::vector<int32_t> _reorder_map;
    /// Array of reordering index in c
    int* reorder_map_c;
    /// N-S feed separation in m
    float _feed_sep_NS;
    /// E-W feed separation in m
    int32_t _feed_sep_EW;

    /// gain stuff:
    /// Array of gain paths
    std::vector<std::string> _gain_dir;
    /// Defualt gain if invalid gain paths
    std::vector<float> default_gains;
    /// Array of gain values
    float* cpu_gain;

    /// Buffer for accessing metadata
    Buffer* metadata_buf;
    /// Metadata buffer ID
    int32_t metadata_buffer_id;
    /// Metadata buffer precondition ID
    int32_t metadata_buffer_precondition_id;
    /// Freq bin index, where the 0th is at 800MHz
    freq_id_t freq_now;
    /// Freq in MHz
    float freq_MHz;

    /// Keep track of the passing of time, in order to trigger update phase every second
    uint second_now;
    uint second_last;
    /// Time now in second
    struct timespec time_now_gps;

    /// RA, DEC and scaling factor for the tracking beams
    struct beamCoord beam_coord; // active coordinates to be passed to metatdata
    std::vector<float> _source_ra;
    std::vector<float> _source_dec;

    /// Array of phase for beamforming
    double* phase;
    /// Flag to trigger phase update, either because of endpt message or every second
    bool update_phase;

    /// Brute-force beamform by multiplying each element with a phase shift
    void cpu_tracking_beamformer(double* input, double* phase, float* output, int nsamp, int nelem,
                                 int nbeams, int npol);
    /// Reorder data from correlator order to cylinder order
    void reorder(unsigned char* data, int* map);
    /// Figure our LST at this frame and the Alt-Az of the 10 sources, then calculate phase delays
    /// at each input
    void calculate_phase(struct beamCoord beam_coord, timespec time_now, float freq_now,
                         float* gain, double* output);
};

#endif
