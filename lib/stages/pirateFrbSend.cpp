#include "pirateFrbSend.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for DEBUG2, ERROR, INFO, DEBUG, WARN
#include "metadata.hpp"          // for metadataObject

#include "fmt.hpp" // for compile_string_to_view, format, format_string, fmt

#include <memory>        // for __shared_ptr_access, shared_ptr

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(pirateFrbSend);

pirateFrbSend::pirateFrbSend(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    bufferSend(config, unique_name, buffer_container),
    sent_header(false),
    offset_scale_buf(nullptr),
    offset_scale_frame(nullptr)
{
    offset_scale_buf = get_buffer("offset_scale");
    offset_scale_buf->register_consumer(unique_name);
}

pirateFrbSend::~pirateFrbSend() {}

std::string pirateFrbSend::get_buffer_name() const {
    return "intensity";
}

struct pirateNetworkHeader {
    uint32_t magic_number;
    uint32_t config_length;
};

bool pirateFrbSend::send_frame(uint8_t* frame, int frame_id) {
    int32_t n = 0;
    int32_t n_sent = 0;

    if (!sent_header) {
        INFO("Sending header");
        
        std::string config = "yaml";

        struct pirateNetworkHeader header;
        header.magic_number = 0xf4bf4b01;
        header.config_length = config.size();
        size_t header_len = sizeof(struct pirateNetworkHeader);

        DEBUG2("Sending header");
        while ((n = send(socket_fd, &((uint8_t*)&header)[n_sent], header_len - n_sent,
                         MSG_NOSIGNAL))
               > 0) {
            n_sent += n;
        }
        sent_header = true;
    }
    INFO("Sending frame {:d}", frame_id);

    return true;
}

bool pirateFrbSend::got_frame(uint8_t*, int frame_id) {
    // get the next frame of the scale & offset buffer
    offset_scale_frame = offset_scale_buf->wait_for_full_frame(unique_name, frame_id);
    if (offset_scale_frame == nullptr)
        return false;
    return true;
}

void pirateFrbSend::done_with_frame(int frame_id) {
    offset_scale_frame = nullptr;
    offset_scale_buf->mark_frame_empty(unique_name, frame_id);
}
