#ifndef PIRATE_FRB_SEND_H
#define PIRATE_FRB_SEND_H

#include "bufferSend.hpp"

class pirateDestination : public networkDestination {
public:
    /// Have we sent the pirate protocol header yet?
    bool sent_header;

    /// Beamset that we're sending to this destination
    int beamset;

    /// Beam index range
    int beam_index_min;
    int beam_index_max;

    // Beam positions
    std::vector<float> beam_positions;
    // Beam ids
    std::vector<int16_t> beam_ids;
};

class pirateFrbSend : public kotekan::Stage {
public:
    /// Standard constructor
    pirateFrbSend(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~pirateFrbSend();

    /// Main loop for sending data
    void main_thread() override;

private:
    /// Threshold to drop frames
    float drop_threshold;

    /// Pirate FRB node destinations
    std::vector<std::shared_ptr<pirateDestination > > dests;

    /// The input buffer containing intensity values
    Buffer* intensity_buf;

    /// The input buffer containing offset & scale values
    Buffer* offset_scale_buf;

    /// The input buffer containing beam positions
    Buffer* beam_position_buf;

    /// The input buffer containing beam ids
    Buffer* beam_id_buf;

    bool send_frame(uint8_t* intensity_frame,
                    uint8_t* offset_scale_frame,
                    std::shared_ptr<pirateDestination> dest);
};

#endif
