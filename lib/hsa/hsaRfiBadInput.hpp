/*
 * @file
 * @brief RFI time sum, performs prallel sum of power, and square power of incoherent beam.
 *  - hsaRfiBadInput : public hsaCommand
 */
#ifndef HSA_RFI_BAD_INPUT_H
#define HSA_RFI_BAD_INPUT_H

#include "hsaCommand.hpp"
#include "restServer.hpp"
#include <mutex>

/*
 * @class hsaRfiBadInput
 * @brief hsaCommand to performs prallel sum of power and square power across time.
 *
 * This is an hsaCommand that launches the kernel (rfi_chime_timesum.hsaco) to perform 
 * a parallel sum of power and square power estimates across time. The sum is then normalized 
 * by the mean power and sent to the hsaRfiIputSum command. 
 *
 * @requires_kernel    rfi_bad_input.hasco
 *
 * @par REST Endpoints
 * @endpoint    /rfi_bad_input_callback/<gpu_id> ``POST`` Change kernel parameters
 *              requires json values
 *              update config
 *
 * @par GPU Memory
 * @gpu_mem  input              Input data of size input_frame_len
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c uint8_t
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  output             Output data of size output_frame_len
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  sk_step            The time ingration length (samples)
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Constant @c uint32_t
 *     @gpu_mem_metadata        none
 * @gpu_mem  num_elements       The total number of elements
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Constant @c uint32_t
 *     @gpu_mem_metadata        none
 *
 * @conf   num_elements         Int. Number of elements.
 * @conf   num_local_freq       Int. Number of local freq.
 * @conf   samples_per_data_set Int. Number of time samples in a data set.
 * @conf   sk_step              Int (default 256). Length of time integration in SK estimate.
 *
 * @author Jacob Taylor
 */
class hsaRfiBadInput: public hsaCommand
{

public:
    /// Constructor, initializes internal variables.
    hsaRfiBadInput(Config& config, const string &unique_name,
                            bufferContainer& host_buffers, hsaDeviceInterface& device);
    /// Destructor, cleans up local allocs
    virtual ~hsaRfiBadInput();
    /// Rest Server callback function
    void rest_callback(connectionInstance& conn, json& json_request);
    /// Executes rfi_chime_inputsum.hsaco kernel. Allocates kernel variables, initalizes input mask array.
    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;
private:
    /// Length of the input frame, should be sizeof_uchar x n_elem x n_freq x nsamp
    uint32_t input_frame_len;
    /// Length of the output frame, should be sizeof_float x n_elem x n_freq x nsamp / sk_step
    uint32_t output_frame_len;
    /// Length of the input mask, should be sizeof_uchar x n_elem
    uint32_t mask_len;
    /// Array to hold the input mask (which inputs are currently functioning)
    uint8_t *InputMask;
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;
    /// Integration length of spectral kurtosis estimate in time
    uint32_t _sk_step;
    /// Rest Server callback mutex
    std::mutex rest_callback_mutex;
    /// String to hold endpoint name
    string endpoint;
};

#endif
