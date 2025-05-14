/*
 * @file
 * @brief Computes and averages single input kurtosis values to detect faulty inputs
 *  - hsaRfiBadInput : public hsaCommand
 */
#ifndef HSA_RFI_BAD_INPUT_H
#define HSA_RFI_BAD_INPUT_H

#include "Config.hpp"             // for Config
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface
#include "restServer.hpp"         // for connectionInstance

#include "json.hpp" // for json

#include <stdint.h> // for uint32_t
#include <string>   // for string

/*
 * @class hsaRfiBadInput
 * @brief hsaCommand to average single input kurtosis values across a packet.
 *
 * This is an hsaCommand that launches the kernel (rfi_bad_input.hsaco) to compute and average
 * spectral kurtosis estimates from individual inputs. These values are then used to determine
 * which inputs are not functioning properly.
 *
 * @requires_kernel    rfi_bad_input.hasco
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
 * @gpu_mem  M                  The time integration length (samples)
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Constant @c uint32_t
 *     @gpu_mem_metadata        none
 * @gpu_mem  num_sk             The total number of SK estimates to be averaged
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
class hsaRfiBadInput : public hsaCommand {

public:
    /// Constructor, initializes internal variables.
    hsaRfiBadInput(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    /// Destructor, cleans up local allocs
    virtual ~hsaRfiBadInput();
    /// Rest Server callback function
    void rest_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);
    /// Executes rfi_bad_input.hsaco kernel. Allocates kernel variables.
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    /// Length of the input frame
    size_t input_frame_len;
    /// Length of the output frame
    size_t output_frame_len;
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;
    /// Integration length of spectral kurtosis estimate in time
    uint32_t _sk_step;
};

#endif
