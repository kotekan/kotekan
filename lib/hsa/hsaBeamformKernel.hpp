/**
 * @file
 * @brief FRB hybrid beamformer (FFT N-S; brute-force E-W)
 *  - hsaBeamformKernel : public hsaCommand
 */

#ifndef HSA_BEAMFORM_KERNEL_H
#define HSA_BEAMFORM_KERNEL_H

#include "Config.hpp"             // for Config
#include "buffer.h"               // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface
#include "restServer.hpp"         // for connectionInstance

#include "json.hpp" // for json

#include <stdint.h> // for int32_t, uint32_t
#include <string>   // for string
#include <vector>   // for vector

#define LIGHT_SPEED 299792458.
#define FEED_SEP 0.3048
#define PI 3.14159265

/**
 * @class hsaBeamformKernel
 * @brief hsaCommand to beamform for FRB
 *
 * This is an hsaCommand that launches the kernel (unpack_shift_beamform_flip) for
 * FRB beamforming. The kernel unpacks the already reordered input data (shape @c
 * n_samp x @c n_elem) and multiplies the conjugate of gains to it. We then apply
 * an FFT beamforming along the N-S direction that is padded by 2. An array of
 * clamping index is used to tell which of the 512 beams to clamp to, to form 256
 * N-S beams. A brute force beamform along the E-W direction is calculated using an
 * array of phase delays to form 4 E-W beams. The N-S beam extent and the 4 E-W beam
 * positions are both tunable via endpoints. The output is flipped in the N-S
 * direction in order to following the L2/3 convention of south beams before north
 * beams. The ordering of the output data is time-pol-beamEW-beamNS, where beamNS
 * is the fastest varying.
 *
 * The gain path is registered as a subscriber to an updatable config block.
 *
 * @requires_kernel    unpack_shift_beamform_flip.hasco
 *
 * @par REST Endpoints
 * @endpoint    /frb/update_NS_beam/\<gpu id\> ``POST`` Trigger re-set of
 *              FFT beam spacing in N-S
 *              requires json values      northmost_beam
 *              update config             northmost_beam
 * @endpoint    /frb/update_EW_beam/\<gpu id\> ``POST`` Trigger re-calculate
 *              of phase delay for the 4 E-W brute-force formed beams
 *              requires json values      ew_id, ew_beam
 *              update config             ew_spacing[ew_id]
 *
 * @par GPU Memory
 * @gpu_mem  input_reordered    Input data of size input_frame_len
 *     @gpu_mem_type            static
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
 * @conf   num_elements         Int (default 2048). Number of elements
 * @conf   num_local_freq       Int (default 1). Number of local freq.
 * @conf   samples_per_data_set Int (default 49152). Number of time samples in a data set
 * @conf   scaling              Float (default 1.0). Scaling factor on gains
 * @conf   default_gains        Float array (default 1+1j). Default gain value if gain file is
 * missing
 * @conf   northmost_beam       Float - Setting the extent in NS of the FFT formed beams.
 *                              Zenith angle of the northmost beam (in deg).
 * @conf   ew_spacing           Float array - 4 sky angles for the columns of E-W beams (in deg).
 *
 * @todo   Better handle of variables that gets updated via endpoint, prevent
 *         read/write conflicts.
 *
 * @author Cherry Ng
 *
 */

class hsaBeamformKernel : public hsaCommand {
public:
    /// Constructor, also initializes internal variables from config, allocates host_map, host_coeff
    /// and host_gain, get metadata buffer and register endpoint for gain path.
    hsaBeamformKernel(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    /// Destructor, cleans up local allocs.
    virtual ~hsaBeamformKernel();

    /// Wait for full metadata frame and keep track of precondition_id
    int wait_on_precondition(int gpu_frame_id) override;

    /// Figure out freq from metadata, calculate freq-specific param, load gains, allocate kernel
    /// argument buffer, set kernel dimensions, enqueue kernel
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    /// Endpoint for setting N-S beam extent
    void update_NS_beam_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);
    /// Endpoint for setting E-W beam sky angle
    void update_EW_beam_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);

private:
    /**
     * @brief  Calculate clamping index for the N-S beams
     * @param host_map    array of output clamping indices
     * @param freq_now    freq of this gpu
     * @param freq_ref    reference freq, which determines the N-S extent of the beams
     */
    void calculate_cl_index(uint32_t* host_map, float freq_now, double freq_ref);

    /**
     * @brief Calculate phase delays for the E-W beams
     * @param freq_now       freq of this gpu
     * @param host_coeff     phase offset
     * @param _ew_spacing_c  Float array size 4 - desired EW sky angles
     */
    void calculate_ew_phase(float freq_now, float* host_coeff, float* _ew_spacing_c);

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

    /// Buffer for accessing metadata
    Buffer* metadata_buf;
    /// Metadata buffer ID
    int32_t metadata_buffer_id;
    /// Metadata buffer precondition ID
    int32_t metadata_buffer_precondition_id;
    /// Freq bin index, where the 0th is at 800MHz
    int32_t freq_idx;
    /// Freq in MHz
    float freq_MHz;

    /// Array of clamping index, int of size 256
    uint32_t* host_map;
    /// Array of phase delays for E-W brute force beamform, float of size 32
    float* host_coeff;

    /// Number of elements, should be 2048
    uint32_t _num_elements;
    /// Number of local freq, should be 1
    int32_t _num_local_freq;
    /// Number of time samples, should be a multiple of 3x128 for FRB, currently set to 49152
    int32_t _samples_per_data_set;

    /// The desired extent (e.g. 90, 60, 45) of the Northmost beam in degree
    float _northmost_beam;
    /// The sky angle of the 4 EW beams in degree
    std::vector<float> _ew_spacing;
    float* _ew_spacing_c;

    /// The reference freq for calcating beam spacing, a function of the input _northmost_beam
    double freq_ref;

    /// Flag to avoid re-calculating freq-specific params except at first pass
    bool first_pass;
    /// Flag to update NS beam
    bool update_NS_beam;
    /// Flag to update EW beam
    bool update_EW_beam;

    /// Endpoint for updating NS beams
    std::string endpoint_NS_beam;
    /// Endpoint for updating EW beams
    std::string endpoint_EW_beam;

    /// Config base (@todo this is a huge hack replace with updatable config)
    std::string config_base;
};

#endif
