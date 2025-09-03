/**
 * @file
 * @brief FRB beamformer CPU verification
 */

#ifndef GPU_BEAMFORM_SIMULATE_HPP
#define GPU_BEAMFORM_SIMULATE_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <stdint.h> // for int32_t, uint64_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class gpuBeamformSimulate
 * @brief CPU verification for FRB beamformer
 *
 * This is CPU only pipeline to verify the FRB beamformer.
 * This CPU pipeline does the equivalent of:
 *   - hsaBeamformReorder
 *   - name: hsaBeamformKernel
 *   - name: hsaBeamformTranspose
 *   - name: hsaBeamformUpchan
 *
 * @author Cherry Ng
 **/

class gpuBeamformSimulate : public kotekan::Stage {
public:
    /// Constructor
    gpuBeamformSimulate(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);
    ~gpuBeamformSimulate();
    void main_thread() override;

private:
    /// Initializes internal variables from config, allocates reorder_map, gain, get metadata buffer
    Buffer* input_buf;
    Buffer* output_buf;
    Buffer* hfb_output_buf;

    /// Number of elements, should be 2048
    int32_t _num_elements;
    /// Number of time samples, should be a multiple of 3x128 for FRB, standard ops is 49152
    int32_t _samples_per_data_set;

    /// Upchannelize factor, should be 128
    int32_t _factor_upchan;
    /// Downsampling factor for the time axis, set to 3
    int32_t _downsample_time;
    /// Downsampling factor for the freq axis, set to 8
    int32_t _downsample_freq;
    /// Array of reordering index
    std::vector<int32_t> _reorder_map;
    /// The desired extent (e.g. 90, 60, 45) of the Northmost beam in degree
    float _northmost_beam;
    /// The reference freq for calcating beam spacing, a function of the input _northmost_beam
    double Freq_ref;
    /// The sky angle of the 4 EW beams in degree
    std::vector<float> _ew_spacing;
    float* _ew_spacing_c;
    /// Default gain values if gain file is missing for this freq, currently set to 1+1j
    std::vector<float> default_gains;
    /// No. of beams
    uint32_t _num_frb_total_beams;

    /// Directory path where gain files are
    std::string _gain_dir;

    /// Array of phase delays for E-W brute force beamform, float of size 32
    float* coff;
    /// Array of gains, float size of 2048*2
    float* cpu_gain;

    /// Buffer for accessing metadata
    Buffer* metadata_buf;
    /// Metadata buffer ID
    int32_t metadata_buffer_id;
    /// Freq bin index, where the 0th is at 800MHz
    freq_id_t freq_now;
    /// Freq in MHz
    float freq_MHz;

    /// Unpacked data
    double* input_unpacked;
    /// Unpacked data padded
    double* input_unpacked_padded;
    /// Clamped to 256 n-s beams
    double* clamping_output;
    /// Output from NS-EW beamform (pre-upchannelization)
    double* cpu_beamform_output;
    /// Transpose beamform_output from time-pol-beams to pol-beam-time
    double* transposed_output;
    /// Intermediate array to hold the 128 times for upchannelize
    double* tmp128;
    /// Intermediate array to hold the 512 values for FFT bf
    int* tmp512;
    /// Array of reordering index in C style for backwards compatibility.
    int* reorder_map_c;
    /// Output data
    float* cpu_final_output;
    float* cpu_hfb_final_output;

    /// Input length, should be nsamp x n_elem x 2
    int input_len;
    /// input_len x 2 because we pad by 2
    int input_len_padded;
    /// transpose length: (nsamp+32) x n_elem x 2
    int transposed_len;
    /// output length: n_elem*(nsamp/ds_t/ds_f/2)
    int output_len;
    /// hfb output length: num_frb_total_beams x num_sub_freq
    int hfb_output_len;

    /// Scaling factor to be applied on the gains, currently set to 1.0 and somewhat deprecated?
    float scaling;

    void reorder(unsigned char* data, int* map);
    void cpu_beamform_ns(double* data, uint64_t transform_length, int stop_level);
    void cpu_beamform_ew(double* input, double* output, float* Coeff, int nbeamsNS, int nbeamsEW,
                         int npol, int nsamp_in);
    void clamping(double* input, double* output, float freq, int nbeamsNS, int nbeamsEW,
                  int nsamp_in, int npol);
    void transpose(double* input, double* output, int nbeams, int nsamp_in);
    void upchannelize(double* data, int upchan);
};

#endif
