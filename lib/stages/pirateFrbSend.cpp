#include "pirateFrbSend.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for DEBUG2, ERROR, INFO, DEBUG, WARN
#include "metadata.hpp"          // for metadataObject
#include "XEngineMetadata.hpp"

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

void pirateFrbSend::main_thread() {
    int frame_id = 0;

    // Fire up all the connect threads...
    for (auto dest : dests)
        dest->connect_thread = std::thread(&bufferSend::connect_to_server, std::ref(*this),
                                           std::ref(*dest));

    while (!stop_thread) {
        uint8_t* frame = buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;
        if (!got_frame(frame, frame_id))
            break;

        uint32_t num_full_frames = buf->get_num_full_frames();

        // This logic is from sendBuffer, but with drop_frames assumed true.

        if ((float)num_full_frames / (float)buf->num_frames > drop_threshold) {
            // If the number of full frames is high, then we drop some
            // frames, because we likely aren't sending fast enough to
            // keep up with the data rate.
            INFO("Number of full frames in buffer {:s} is {:d} (total frames: {:d}), dropping "
                 "frame_id {:d}",
                 buf->buffer_name, num_full_frames, buf->num_frames, frame_id);
            dropped_frame_counter.inc();
        } else {
            bool any_connected = false;
            for (auto dest : dests)
                if (dest->connected) {
                    any_connected = true;
                    if (!send_frame(frame, frame_id, dest))
                        close_connection(*dest);
                    else
                        DEBUG("Sent frame: {:s}[{:d}] to {:s}:{:d}", buf->buffer_name, frame_id,
                              dest->server_ip, dest->server_port);
                }
            if (!any_connected) {
                // Mark this frame as dropped
                INFO("Dropping frame {:s}[{:d}], because all connections are down.",
                     buf->buffer_name, frame_id);
                dropped_frame_counter.inc();
            }
        }

        done_with_frame(frame_id);
        buf->mark_frame_empty(unique_name, frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
    }

    for (auto dest : dests)
        close_connection(*dest);
    for (auto dest : dests)
        dest->connect_thread.join();
}

bool pirateFrbSend::send_frame(uint8_t* frame, int frame_id, std::shared_ptr<struct pirateDestination> dest) {
    int32_t n = 0;
    int32_t n_sent = 0;

    //pirateDestination* dest = reinterpret_cast<pirateDestination*>(&_dest);

    if (!dest->sent_header) {
        INFO("Sending header");

        struct XEngineMetadata meta;
        meta.version = 1;

        // We need the upchannelization configuration to figure out the upchan zones.
        /*
          See upchan_pathfinder.j2, for instance:

          upchan_channel_ranges:
          64: [0, 1750]
          32: [1750, 2264]
          16: [2264, 2912]
          8: [2912, 3661]
          4: [3661, 4646]
          2: [4646, 5985]
          1: [5985, 8192]

          Where the channel numbers translate to frequencies via
          fengine_pathfinder.j2:

          #     A channel's frequency is   freq = channel * adc_frequency / num_samples_per_frame

          adc_frequency: 3.2e+9           # [Hz]
          num_samples_per_frame: 16384
          
        */

        double adc_freq = config.get<double>(unique_name, "adc_frequency");
        int samples_per_frame = config.get<int>(unique_name, "num_samples_per_frame");

        int channel_low = 0;
        double freq_low = channel_low * adc_freq / samples_per_frame;
        meta.zone_freq_edges.push_back(freq_low);

        json upchan_ranges = config.get<std::vector<json> >(unique_name, "upchan_channel_ranges");
        if (!upchan_ranges.is_object()) {
            throw std::invalid_argument(fmt::format(fmt("Expect 'upchan_channel_ranges' to contain a dicts")));
        }
        for (json::iterator it : upchan_ranges) {
            int upchan_factor = it.key().get<int>();
            std::vector<int> channel_range = it.value().get<std::vector<int> >();
            if (channel_range.size() != 2) {
                throw std::invalid_argument(fmt::format(fmt("Expect 'upchan_channel_ranges' values to be a pair of integers")));
            }
            int lo = channel_range[0];
            int channel_high = channel_range[1];
            if (lo != channel_low) {
                throw std::invalid_argument(fmt::format(fmt("Expect 'upchan_channel_ranges' values to be contiguous")));
            }
            double freq_high = channel_high * adc_freq / samples_per_frame;
            meta.zone_freq_edges.push_back(freq_high);
            int nfreq = upchan_factor * (channel_high - lo);
            meta.zone_nfreq.push_back(nfreq);

            // next one should have:
            channel_low = channel_high;
        }

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

        size_t config_len = header.config_length;
        n_sent = 0;
        DEBUG2("Sending config yaml");
        while ((n = send(dest->socket_fd, &((uint8_t*)config.c_str())[n_sent], config_len - n_sent,
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
