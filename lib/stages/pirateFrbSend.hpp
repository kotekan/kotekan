#ifndef PIRATE_FRB_SEND_H
#define PIRATE_FRB_SEND_H

#include "bufferSend.hpp"

struct pirateDestination : public destination {
    /// Have we sent the pirate protocol header yet?
    bool sent_header;
};

class pirateFrbSend : public bufferSend {
public:
    /// Standard constructor
    pirateFrbSend(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~pirateFrbSend();

    /// Main loop for sending data
    void main_thread() override;

    /// Called when a frame has been received -- just after it has been claimed from
    /// kotekan.  If false is returned, the sending will end.
    bool got_frame(uint8_t* frame, int frame_id) override;

    /// Called when a frame has been sent -- just before the buffer frame is released
    /// back to kotekan.
    void done_with_frame(int frame_id) override;

protected:
    std::string get_buffer_name() const override;

    void initialize_destination() override;

private:
    /// Pirate FRB node destinations
    std::vector<std::shared_ptr<pirateDestination > > dests;

    /// The input buffer containing offset & scale values
    Buffer* offset_scale_buf;

    /// The current frame of offsets & scales
    uint8_t* offset_scale_frame;

    /// Send a frame
    bool send_frame(uint8_t* frame, int frame_id, std::shared_ptr<struct pirateDestination> dest);
};

#endif
