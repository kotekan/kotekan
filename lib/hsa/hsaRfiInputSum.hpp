/*
 * @file
 * @brief RFI input sum, computes spectral kurtosis estimates
 *  - hsaRfiInputSum : public hsaCommand
 */
#ifndef HSA_RFI_INPUT_SUM_H
#define HSA_RFI_INPUT_SUM_H

#include "Config.hpp"             // for Config
#include "buffer.hpp"               // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface
#include "restServer.hpp"         // for connectionInstance

#include "json.hpp" // for json

#include <stdint.h> // for uint32_t, int32_t
#include <string>   // for string

/*
 * @class hsaRfiInputSum
 * @brief hsaCommand to compute the input sum and spectral kurtosis for RFI detection
 *
 * This is an hsaCommand that launches the kernel (rfi_chime_input_sum.hsaco) to perform
 * a sum of normalized, time summed square power estimates (see hsaRfiTimeSum.hpp). The
 * sum is then used to calculate a spectral kurtosis estimate. The spectral kurtosis estimate
 * is a measure of the underlying gaussianity of the sample. Thus it can be used as a tool
 * to detect non-gaussian signals in CHIME's incoherent beam (RFI).
 *
 * @requires_kernel    rfi_chime_input_sum.hasco
 *
 * @par REST Endpoints
 * @endpoint    /rfi_input_sum_callback/<gpu_id> ``POST`` Change kernel parameters
 *              requires json values      "num_bad_inputs"
 *              update config             "num_bad_inputs"
 *
 * @par GPU Memory
 * @gpu_mem  time_sum           Input data from the hsaRfiTimeSum command  of size input_frame_len
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  rfi_output          Output data of size output_frame_len
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  input_mask         Inputs to be ignored/masked (1 for mask, 0 for don't mask)
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c uint8_t
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  rfi_mask_output    Mask used to zero input data (1 for RFI, 0 for clean)
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 *
 * @conf   num_elements         Int. Number of elements.
 * @conf   num_local_freq       Int. Number of local freq.
 * @conf   samples_per_data_set Int. Number of time samples in a data set.
 * @conf   sk_step              Int (default 256). Length of time integration in SK estimate.
 * @conf   bad_inputs           Array of Ints. Used to compute the number of faulty inputs
 *
 * @author Jacob Taylor
 */
class hsaRfiInputSum : public hsaCommand {
public:
    /// Constructor, initializes internal variables.
    hsaRfiInputSum(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    /// Destructor, cleans up local allocs
    virtual ~hsaRfiInputSum();
    /// Rest server callback
    void rest_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);

    int wait_on_precondition(int gpu_frame_id) override;

    /// Executes rfi_chime_input_sum.hsaco kernel. Allocates kernel variables.
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;

private:
    /// Main data input, used for metadata access
    Buffer* _network_buf;

    /// IDs for _network_buf
    int32_t _network_buf_finalize_id;
    int32_t _network_buf_execute_id;
    int32_t _network_buf_precondition_id;

    /// Length of the input frame, should be sizeof_float x n_elem x n_freq x nsamp / sk_step
    size_t input_frame_len;
    /// Length of the output frame, should be sizeof_float x n_freq x nsamp / sk_step
    size_t output_frame_len;
    /// Length of the input variance frame, should be sizeof_float x n_elem x n_freq x nsamp /
    /// sk_step
    size_t input_var_frame_len;
    /// Length of the output frame, should be sizeof_float x n_freq x nsamp / sk_step
    size_t output_var_frame_len;
    /// Length of the input mask, should be sizeof_uchar x n_elem
    size_t input_mask_len;
    /// Length of the output mask, should be sizeof_uchar x n_freq x nsamp / sk_step
    size_t output_mask_len;
    /// Length of lost sample correction frame
    size_t correction_frame_len;
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;
    /// Integration length of spectral kurtosis estimate in time
    uint32_t _sk_step;
    /// The number of standard deviations in SK which constitute RFI
    uint32_t _rfi_sigma_cut;
    /// Truncation bias switch.
    float _trunc_bias_switch;
};

#endif
