/**
 * @file
 * @brief RFI input sum, computes spectral kurtosis estimates
 *  - hsaRfiInputSum : public hsaCommand
 */
#ifndef HSA_RFI_INPUT_SUM_H
#define HSA_RFI_INPUT_SUM_H

#include "hsaCommand.hpp"

/**
 * @class hsaRfiInputSum
 * @brief hsaCommand to compute the input sum and spectral kurtosis for RFI detection
 *
 * This is an hsaCommand that launches the kernel (rfi_chime_inputsum.hsaco) to perform 
 * a sum of normalized, time summed square power estimates (see hsaRfiTimeSum.hpp). The 
 * sum is then used to calculate a spectral kurtosis estimate. The spectral kurtosis estimate 
 * is a measure of the underlying gaussianity of the sample. Thus it can be used as a tool 
 * to detect non-gaussian signals in CHIME's incoherent beam (RFI).
 *
 * @requires_kernel    rfi_chime_inputsum.hasco
 *
 * @par GPU Memory
 * @gpu_mem  input              Input data of size input_frame_len
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  output             Output data of size output_frame_len
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  num_elements       The total number of elements
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Constant @c uint32_t
 *     @gpu_mem_metadata        none
 * @gpu_mem  M                  The total SK integration length
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Constant @c uint32_t
 *     @gpu_mem_metadata        none
 *
 * @conf   num_elements         Int (default 2048). Number of elements.
 * @conf   num_local_freq       Int (default 1). Number of local freq.
 * @conf   samples_per_data_set Int (default 32768). Number of time samples in a data set.
 * @conf   sk_step              Int Length of time integration in SK estimate.
 *
 * @author Jacob Taylor
 *
 */
class hsaRfiInputSum: public hsaCommand
{
public:
    /// Constructor, initializes internal variables.
    hsaRfiInputSum(Config &config, const string &unique_name,
               bufferContainer &host_buffers, hsaDeviceInterface &device);

    /// Destructor, cleans up local allocs
    virtual ~hsaRfiInputSum();

    void rest_callback(connectionInstance& conn, json& json_request);

    /// Executes rfi_chime_inputsum.hsaco kernel. Allocates kernel variables.
    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:
    /// Length of the input frame, should be sizeof_float x n_elem x n_freq x nsamp / sk_step
    uint32_t input_frame_len;
    /// Length of the input frame, should be sizeof_float x n_freq x nsamp / sk_step
    uint32_t output_frame_len;

    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;

    /// Integration length of spectral kurtosis estimate in time
    uint32_t _sk_step;
    /// The total number of faulty inputs
    uint32_t _num_bad_inputs;
    /// The total integration length of the spectral kurtosis estimate
    uint32_t _M;
    std::mutex rest_callback_mutex;
};

#endif
