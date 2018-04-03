/**
 * @file
 * @brief FRB hybrid beamformer (FFT N-S; brute-force E-W)
 *  - hsaBeamformKernel : public hsaCommand
 */

#ifndef HSA_BEAMFORM_KERNEL_H
#define HSA_BEAMFORM_KERNEL_H

#include "hsaCommand.hpp"
#include "restServer.hpp"

#define FREQ_REF 492.125984252
#define LIGHT_SPEED 3.e8
#define FEED_SEP 0.3048
#define PI 3.14159265

/**
 * @class hsaBeamformKernel
 * @brief hsaCommand to beamform for FRB
 *
 *
 * This is an hsaCommand that launches the kernel (unpack_shift_beamform_flip) for
 * FRB beamforming. The kernel unpacks the already reordered input data (shape @c 
 * n_samp x @c n_elem) and multiplies the conjugate of gains to it. We then apply 
 * an FFT beamforming along the N-S direction that is padded by 2. An array of 
 * clamping index is used to tell which of the 512 beams to clamp to, to form 256 
 * N-S beams. A brute force beamform along the E-W direction is calculated using an 
 * array of phase delays to form 4 E-W beams. The output is flipped in the N-S 
 * direction in order to following the L2/3 convention of south beams before north 
 * beams. The ordering of the output data is time-pol-beamEW-beamNS, where beamNS 
 * is the fastest varying.
 *
 * @requires_kernel    unpack_shift_beamform_flip.hasco
 *
 * @par GPU Memory
 * @gpu_mem  input              Input data of size input_frame_len
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c uchar
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  beamform_output    Output data of size output_frame_len
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  beamform_map       Array of clamping index of size 256
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c uint
 *     @gpu_mem_metadata        none
 * @gpu_mem  beamform_coeff_map Array of phase delay size 32
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        none
 * @gpu_mem  beamform_gain      Array of gains size 2048*2
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        none
 * 
 *
 * @conf   num_elements         Int (default 2048). Number of elements
 * @conf   num_local_freq       Int (default 1). Number of local freq.
 * @conf   samples_per_data_set Int (default 49152). Number of time samples in a data set
 * @conf   gain_dir             String - directory path where gain files are
 * @conf   scaling              Float (default 1.0). Scaling factor on gains
 * @conf   default_gains        Float array (default 1+1j). Default gain value if gain file is missing
 *
 * @author Cherry Ng
 *
 */



class hsaBeamformKernel: public hsaCommand
{
public:
    ///Constructor, also initializes internal variables from config, allocates host_map, host_coeff and host_gain, get metadata buffer and register endpoint for gain path. 
    hsaBeamformKernel(Config &config, const string &unique_name, 
                        bufferContainer &host_buffers, hsaDeviceInterface &device);

    /// Destructor, cleans up local allocs.
    virtual ~hsaBeamformKernel();

    /// Wait for full metadata frame and keep track of precondition_id
    int wait_on_precondition(int gpu_frame_id) override;

    /// For a given freq, calculate N-S FFT clamping index (host_map) and E-W phase delays (host_coeff)
    void calculate_cl_index(uint32_t *host_map, float freq1, float *host_coeff);

    /// Figure out freq from metadata, calculate freq-specific param, load gains, allocate kernel argument buffer, set kernel dimensions, enqueue kernel
    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

    /// Endpoint for providing new directory path for gain updates 
    void update_gains_callback(connectionInstance& conn, json& json_request);

private:
    /// Input length, should be nsamp x n_elem x 2 for complex / 2 since we pack two 4-bit in one
    int32_t input_frame_len;
    /// Output length, should be nsamp x 2 pol x 1024 beams x 2 for complex
    int32_t output_frame_len;
    /// map of clamping index, should be of size of 256
    int32_t map_len;
    /// 4 cylinder x 4 beams x 2 for complex; size of 32
    int32_t coeff_len;
    /// 2048 elements x 2 for complex
    int32_t gain_len;
    /// Directory path where gain files are
    string _gain_dir;
    /// Default gain values if gain file is missing for this freq, currently set to 1+1j
    vector<float> default_gains;

    /// Buffer for accessing metadata
    Buffer * metadata_buf;
    /// Metadata buffer ID
    int32_t metadata_buffer_id;
    /// Metadata buffer precondition ID
    int32_t metadata_buffer_precondition_id;
    /// Freq bin index, where the 0th is at 800MHz
    int32_t freq_idx;
    /// Freq in MHz
    float freq_MHz;

    /// Array of clamping index, int of size 256
    uint32_t * host_map;
    /// Array of phase delays for E-W brute force beamform, float of size 32
    float * host_coeff;
    /// Array of gains, float size of 2048*2
    float * host_gain;

    /// Scaling factor to be applied on the gains, currently set to 1.0 and somewhat deprecated?
    float scaling;

    /// Number of elements, should be 2048
    uint32_t _num_elements;
    /// Number of local freq, should be 1
    int32_t _num_local_freq;
    /// Number of time samples, should be a multiple of 3x128 for FRB, currently set to 49152
    int32_t _samples_per_data_set;
    ///Flag to control gains to be only loaded on request.
    bool update_gains; 
    /// Flag to avoid re-calculating freq-specific params except at first pass
    bool first_pass; 
};

#endif
