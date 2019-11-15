/*
 * @file
 * @brief Copy-in command object for updating the list of bad inputs
 *  - hsaRfiTimeSum : public hsaCommand
 */
#ifndef HSA_RFI_UPDATE_BAD_INPUTS_HPP
#define HSA_RFI_UPDATE_BAD_INPUTS_HPP

#include "hsaCommand.hpp"

#include "json.hpp"

#include <vector>

/**
 * @class hsaRfiUpdateBadInputs
 * @brief hsaCommand copy-in command which updates the bad input list for the RFI system
 *
 * @par Buffers:
 * @buffer network_buf Buffer containing the metadata for
 *     @buffer_format FPGA post PDF data.
 *     @buffer_metadata chime_metadata
 *
 * @conf   updatable_config/bad_inputs  String.  String pointing to the location of the
 *                                      config block containing the following properties:
 *                                      "bad_inputs"  An array of bad inputs in cylinder order.
 *
 * @author Andre Renard
 */
class hsaRfiUpdateBadInputs : public hsaCommand {

public:
    /// Standard constructor
    hsaRfiUpdateBadInputs(kotekan::Config& config, const string& unique_name,
                          kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    virtual ~hsaRfiUpdateBadInputs();

    /// Wait for network buffer with metadata
    int wait_on_precondition(int gpu_frame_id) override;

    /// Copy the bad input mask should it need to be updated
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    /// Finalize any copies when they are activated
    void finalize_frame(int frame_id) override;

    /**
     * @brief Updatable config callback function to handle updates to the list of bad inouts.
     * @param json The new list of bad inputs `{"bad_inputs": [<bad_inputs>]}`
     * @return True if the list of bad inputs was parsed, false otherwise.
     */
    bool update_bad_inputs_callback(nlohmann::json& json);

private:
    /// Main data input, used for metadata access
    Buffer* _network_buf;
    Buffer* _in_buf;

    /// IDs for _network_buf
    int32_t _network_buf_finalize_id;
    int32_t _network_buf_execute_id;
    int32_t _network_buf_precondition_id;

    /// IDs for _in_buf
    int32_t _in_buf_id;
    int32_t _in_buf_len;
    int32_t _in_buf_finalize_id;
    int32_t _in_buf_precondition_id;

    /// State of the update
    bool update_bad_inputs;

    /// The numer of frames to update before stopping to copy the bad input mask
    int frames_to_update;

    /// Counter of the number of copy operations which have to be finalized
    int frames_to_update_finalize;

    /// Mutex to lock updates to the bad_input lists and copy state.
    std::mutex update_mutex;

    /// List of current bad inputs in cylinder order
    std::vector<int> bad_inputs_cylinder;

    /// List of current bad inputs in correlator order.
    std::vector<int> bad_inputs_correlator;

    /// The size of the bad input mask.
    uint32_t input_mask_len;

    /// The host memory region which holds the input mask
    /// Note 1 means the element is good, 0 means flagged.
    uint8_t* host_mask;

    /// The mapping from correlator to cylinder element indexing.
    std::vector<uint32_t> input_remap;

    int32_t frame_to_fill;
    int32_t frame_to_fill_finalize;
    bool filling_frame;
    bool first_pass;
};

#endif // HSA_RFI_UPDATE_BAD_INPUTS_HPP
