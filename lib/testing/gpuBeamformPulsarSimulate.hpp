#ifndef GPU_BEAMFORM_PULSAR_SIMULATE_HPP
#define GPU_BEAMFORM_PULSAR_SIMULATE_HPP

#include "Config.hpp" // for Config
#include "Stage.hpp"  // for Stage
#include "Telescope.hpp"
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.h"     // for psrCoord

#include <stdint.h>    // for int32_t, uint32_t
#include <string>      // for string
#include <sys/types.h> // for uint
#include <time.h>      // for timespec
#include <vector>      // for vector

/**
 * @class gpuBeamformPulsarSimulate
 * @brief CPU version of pulsar beamformer for verification
 *
 * Read hsaTrackingUpdatePhase and hsaTrackingBeamform to find out what
 * the pulsar beamformer does. This is basically the same, except
 * without capability to dynamically update gain paths and RA/Dec
 * (no endpoints).
 *
 * @author Cherry Ng
 *
 */

class gpuBeamformPulsarSimulate : public kotekan::Stage {
public:
    /// Constructor
    gpuBeamformPulsarSimulate(kotekan::Config& config, const std::string& unique_name,
                              kotekan::bufferContainer& buffer_container);
    /// Destructor
    ~gpuBeamformPulsarSimulate();
    /// Main loop, read gains, calculate new phase, brute-force beamform
    void main_thread() override;

private:
    /// Input
    struct Buffer* input_buf;
    /// Input length
    int input_len;
    /// An intermediate array of unpacked data
    double* input_unpacked;
    /// Output in CPU
    float* cpu_output;
    /// Output
    struct Buffer* output_buf;
    /// Output length
    int output_len;

    /// Config options:
    /// number of elements = 2048
    uint32_t _num_elements;
    /// number of samples = 49152
    int32_t _samples_per_data_set;
    /// number of pulsars = 10
    int32_t _num_pulsar;
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

    /// 10 pulsar RA, DEC and scaling factor
    struct psrCoord psr_coord; // active coordinates to be passed to metatdata
    std::vector<float> _source_ra;
    std::vector<float> _source_dec;

    /// Array of phase for beamforming
    double* phase;
    /// Flag to trigger phase update, either because of endpt message or every second
    bool update_phase;

    /// Brute-force beamform by multiplying each element with a phase shift
    void cpu_beamform_pulsar(double* input, double* phase, float* output, int nsamp, int nelem,
                             int npsr, int npol);
    /// Reorder data from correlator order to cylinder order
    void reorder(unsigned char* data, int* map);
    /// Figure our LST at this frame and the Alt-Az of the 10 sources, then calculate phase delays
    /// at each input
    void calculate_phase(struct psrCoord psr_coord, timespec time_now, float freq_now, float* gain,
                         double* output);
};

#endif
