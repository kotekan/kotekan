/*
 * @file
 * @brief Zeros data flagged by the RFI pipeline
 *  - hsaRfiZeroData : public hsaCommand
 */
#ifndef HSA_RFI_ZERO_DATA_H
#define HSA_RFI_ZERO_DATA_H

#include "Config.hpp"             // for Config
#include "buffer.h"               // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include "json.hpp" // for json

#include <mutex>    // for mutex
#include <stdint.h> // for uint32_t, int32_t
#include <string>   // for string

/*
 * @class hsaRfiZeroData
 * @brief hsaCommand to zero input data marked as containing large amounts of RFI
 *
 * This is an hsaCommand that launches the kernel (rfi_chime_zero.hsaco) to set all
 * values detected to have a significant amount of RFI to zero
 *
 * @requires_kernel    rfi_chime_zero.hasco
 *
 *
 * @par GPU Memory
 * @gpu_mem  input              Input data of size input_frame_len
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c uint8_t
 *     @gpu_mem_metadata        chimeMetadata
 * @gpu_mem  mask               The RFI mask to be aplied
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c uint8_t
 *     @gpu_mem_metadata        chimeMetadata
 *
 * @conf   num_elements         Int. Number of elements.
 * @conf   num_local_freq       Int. Number of local freq.
 * @conf   samples_per_data_set Int. Number of time samples in a data set.
 * @conf   sk_step              Int (default 256). Length of time integration in SK estimate.
 *
 * @author Jacob Taylor
 */
class hsaRfiZeroData : public hsaCommand {

public:
    /// Constructor, initializes internal variables.
    hsaRfiZeroData(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    /// Destructor, cleans up local allocs
    virtual ~hsaRfiZeroData();

    /// Function to hadle updatable config rest server calls
    bool update_rfi_zero_flag(nlohmann::json& json);

    /// Executes rfi_chime_zero.hsaco kernel. Allocates kernel variables.
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    /// The current netowrk buffer frame id
    int32_t network_buffer_id;
    /// The network buffer object (i.e. the host input data buffer)
    Buffer* network_buf;
    /// Length of the input frame, should be sizeof_uchar x n_elem x n_freq x nsamp
    uint32_t input_frame_len;
    /// Length of the RFI mask
    uint32_t mask_len;
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;
    /// Integration length of spectral kurtosis estimate in time
    uint32_t _sk_step;
    /// Boolean to toggle RFI zeroing
    bool _rfi_zeroing;
    /// Rest server callback mutex
    std::mutex rest_callback_mutex;
};

#endif
