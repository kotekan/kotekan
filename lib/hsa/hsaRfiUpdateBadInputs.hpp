/*
 * @file
 * @brief Copy-in command object for updating the list of bad inputs
 *  - hsaRfiTimeSum : public hsaCommand
 */
#ifndef HSA_RFI_UPDATE_BAD_INPUTS_HPP
#define HSA_RFI_UPDATE_BAD_INPUTS_HPP

#include "Config.hpp"             // for Config
#include "buffer.hpp"               // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <mutex>    // for mutex
#include <stdint.h> // for int32_t, uint32_t, uint8_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class hsaRfiUpdateBadInputs
 * @brief hsaCommand copy-in command which updates the bad input list for the RFI system
 *
 * @par Buffers:
 * @buffer in_buf Buffer containing the list of bad inputs
 *     @buffer_format Array of @c uint8_t
 *     @buffer_metadata chime_metadata

 * @buffer network_buf Buffer containing the metadata for
 *     @buffer_format FPGA post PDF data.
 *     @buffer_metadata chime_metadata
 *
 * @author Andre Renard & James Willis
 */
class hsaRfiUpdateBadInputs : public hsaCommand {

public:
    /// Standard constructor
    hsaRfiUpdateBadInputs(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    virtual ~hsaRfiUpdateBadInputs();

    /// Wait for network buffer with metadata
    int wait_on_precondition(int gpu_frame_id) override;

    /// Copy the bad input mask should it need to be updated
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    /// Finalize any copies when they are activated
    void finalize_frame(int frame_id) override;

    /// Copy frame into host_mask and store the no. of bad inputs.
    inline void copy_frame(int gpu_frame_id);

private:
    /// Main data input, list of bad inputs
    Buffer* _in_buf;

    /// Used for metadata access
    Buffer* _network_buf;

    /// IDs for _network_buf
    int32_t _network_buf_finalize_id;
    int32_t _network_buf_execute_id;
    int32_t _network_buf_precondition_id;

    /// ID for _in_buf
    int32_t _in_buf_precondition_id;

    /// The numer of frames to update before stopping to copy the bad input mask
    int frames_to_update;

    /// Tracks which GPU frames have an active copy from the execute stage
    /// Note since for a given frame_id there can only be one active set
    /// of commands as long as finalize_frame() marks this as false
    /// there is no risk of a race condition, since that index will not be
    /// reused until finalize_frame() is finished.
    std::vector<bool> frame_copy_active;

    /// Mutex to lock updates to the bad_input lists and copy state.
    std::mutex update_mutex;

    /// The size of the bad input mask.
    uint32_t input_mask_len;

    /// The host memory region which holds the input mask
    /// Note 1 means the element is good, 0 means flagged.
    uint8_t* host_mask;

    /// The no. of bad inputs.
    uint32_t num_bad_inputs;

    /// Stores whether the first bad input update has occurred
    bool first_pass;
};

#endif // HSA_RFI_UPDATE_BAD_INPUTS_HPP
