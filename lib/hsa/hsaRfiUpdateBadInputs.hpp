#ifndef HSA_RFI_UPDATE_BAD_INPUTS_HPP
#define HSA_RFI_UPDATE_BAD_INPUTS_HPP

#include "hsaCommand.hpp"
#include "json.hpp"

#include <vector>

class hsaRfiUpdateBadInputs : public hsaCommand {

public:
    hsaRfiUpdateBadInputs(kotekan::Config& config, const string& unique_name,
                          kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);
    virtual ~hsaRfiUpdateBadInputs();

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;

    bool update_bad_inputs_callback(nlohmann::json& json);

private:
    /// Main data input, used for metadata access
    Buffer* _network_buf;

    /// ID for _network_buf
    int32_t _network_buf_id;

    bool update_bad_inputs;
    int frames_to_update;
    int frames_to_update_finalize;

    std::vector<int> bad_inputs_cylinder;
    std::vector<int> bad_inputs_correlator;

    uint32_t input_mask_len;

    // The mapping from buffer element order to output file element ordering
    std::vector<uint32_t> input_remap;

    uint8_t* host_mask;

    std::mutex update_mutex;
};

#endif // HSA_RFI_UPDATE_BAD_INPUTS_HPP
