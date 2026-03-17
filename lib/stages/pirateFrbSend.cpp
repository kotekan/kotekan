#include "pirateFrbSend.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for DEBUG2, ERROR, INFO, DEBUG, WARN
#include "metadata.hpp"          // for metadataObject

#include "fmt.hpp" // for compile_string_to_view, format, format_string, fmt
#include "json.hpp" // for json, basic_json, iter_impl

#include <memory>        // for __shared_ptr_access, shared_ptr
#include <arpa/inet.h>   // for htons, inet_addr

using nlohmann::json;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(pirateFrbSend);

pirateFrbSend::pirateFrbSend(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    bufferSend(config, unique_name, buffer_container),
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

void pirateFrbSend::initialize_destination() {
    /**
beams: [
{beam_id: 0, x: 0.0, y: 0.0, type: Real},
{beam_id: 1, x: 0.1, y: -1.2, type: Real},
{beam_id: 75, x: -0.2, y: 4.8, type: Real},
{beam_id: 4, x: -0.2, y: 4.8, type: Fake},
{beam_id: 5, x: -0.2, y: 4.8, type: Real},
...]

frb_targets: [
{ip: 11.1.1.1, port: 12345, beams: [0,1,4,5]},
{ip: 24.24.24.24, port: 12345, beams: [75, 76, 77, 78]},
... ]
    */

    json frb_targets = config.get<std::vector<json> >(unique_name, "frb_targets");
    for (json target : frb_targets) {
        if (!target.is_object()) {
            throw std::invalid_argument(fmt::format(fmt("Expect 'frb_targets' to contain a list of dicts")));
        }
        std::string ip = target["ip"].get<std::string>();
        int port = target["port"].get<int>();
        std::vector<int> beam_ids = target["beams"].get<std::vector<int> >();
        int beamset = target["beam_set"].get<int>();

        std::shared_ptr<struct pirateDestination> dest = std::make_shared<pirateDestination>();
        dest->connected = false;
        dest->sent_header = false;
        dest->socket_fd = -1;
        dest->server_ip = ip;
        dest->server_port = port;

        bzero(&dest->server_addr, sizeof(dest->server_addr));
        dest->server_addr.sin_family = AF_INET;
        dest->server_addr.sin_addr.s_addr = inet_addr(dest->server_ip.c_str());
        dest->server_addr.sin_port = htons(dest->server_port);

        dests.push_back(dest);
    }
}

bool pirateFrbSend::send_frame(uint8_t* frame, int frame_id, struct destination& _dest) {
    int32_t n = 0;
    int32_t n_sent = 0;

    pirateDestination* dest = reinterpret_cast<pirateDestination*>(&_dest);

    if (!dest->sent_header) {
        INFO("Sending header");

        std::string config = "yaml";

        struct pirateNetworkHeader header;
        header.magic_number = 0xf4bf4b01;
        header.config_length = config.size();
        size_t header_len = sizeof(struct pirateNetworkHeader);

        DEBUG2("Sending header");
        while ((n = send(dest->socket_fd, &((uint8_t*)&header)[n_sent], header_len - n_sent,
                         MSG_NOSIGNAL))
               > 0) {
            n_sent += n;
        }
        dest->sent_header = true;
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
