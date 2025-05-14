/*
 * @file
 * @brief RFI time sum, performs prallel sum of power, and square power of incoherent beam.
 *  - hsaRfiTimeSum : public hsaCommand
 */
#ifndef HSA_RFI_TIME_SUM_H
#define HSA_RFI_TIME_SUM_H

#include "Config.hpp"             // for Config
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include "json.hpp" // for json

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector

/*
 * @class hsaRfiTimeSum
 * @brief hsaCommand to performs prallel sum of power and square power across time.
 *
 * This is an hsaCommand that launches the kernel (rfi_chime_time_sum.hsaco) to perform
 * a parallel sum of power and square power estimates across time. The sum is then normalized
 * by the mean power and sent to the hsaRfiIputSum command.
 *
 * @requires_kernel    rfi_chime_time_sum_private.hasco
 *
 * @par REST Endpoints
 * @endpoint    /rfi_time_sum_callback/<gpu_id> ``POST`` Change kernel parameters
 *              requires json values      "bad_inputs"
 *              update config             "bad_inputs"
 *
 * @par GPU Memory
 * @gpu_mem  input              Input data of size input_frame_len
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c uint8_t
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  time_sum           Output data of size output_frame_len
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        chimeMetadata
 *
 * @conf   num_elements         Int. Number of elements.
 * @conf   num_local_freq       Int. Number of local freq.
 * @conf   samples_per_data_set Int. Number of time samples in a data set.
 * @conf   sk_step              Int (default 256). Length of time integration in SK estimate.
 * @conf   bad_inputs           Array of Int The inputs which are currently malfunctioning
 *
 * @author Jacob Taylor
 */
class hsaRfiTimeSum : public hsaCommand {

public:
    /// Constructor, initializes internal variables.
    hsaRfiTimeSum(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    /// Destructor, cleans up local allocs
    virtual ~hsaRfiTimeSum();

    /// Executes rfi_chime_input_sum.hsaco kernel. Allocates kernel variables, initalizes input mask
    /// array.
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    /// Length of the input frame, should be sizeof_uchar x n_elem x n_freq x nsamp
    size_t input_frame_len;
    /// Length of the output frame, should be sizeof_float x n_elem x n_freq x nsamp / sk_step
    size_t output_frame_len;
    /// Length of the output_var frame, should be sizeof_float x n_freq x nsamp / sk_step
    size_t output_var_frame_len;
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;
    /// Integration length of spectral kurtosis estimate in time
    uint32_t _sk_step;
    /// The mapping from correlator to cylinder element indexing.
    std::vector<uint32_t> input_remap;
};

#endif
