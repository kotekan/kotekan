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
using pirate::XEngineMetadata;

REGISTER_KOTEKAN_STAGE(pirateFrbSend);

pirateFrbSend::pirateFrbSend(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&pirateFrbSend::main_thread, this))
{
    drop_threshold = config.get_default<float>(unique_name, "drop_threshold", 0.6);

    intensity_buf = get_buffer("intensity");
    intensity_buf->register_consumer(unique_name);
    offset_scale_buf = get_buffer("offset_scale");
    offset_scale_buf->register_consumer(unique_name);
    beam_position_buf = get_buffer("beam_positions");
    beam_position_buf->register_consumer(unique_name);
    beam_id_buf = get_buffer("beam_ids");
    beam_id_buf->register_consumer(unique_name);
    
    /**
beams: [
{beam_id: 0, x: 0.0, y: 0.0, type: Real},
{beam_id: 1, x: 0.1, y: -1.2, type: Real},
{beam_id: 75, x: -0.2, y: 4.8, type: Real},
{beam_id: 4, x: -0.2, y: 4.8, type: Fake},
{beam_id: 5, x: -0.2, y: 4.8, type: Real},
...]

frb_search_nodes: [
  {'ip': '10.0.0.42',
   'port': 9000,
   'beamset': 0,
   'beam_index_min': 0,
   'beam_index_max': 200,
  },
  {'ip': '10.0.0.43',
   'port': 9000,
   'beamset': 1,
   'beam_index_min': 200,
   'beam_index_max': 400,
  }
]
    */
    // ughh duplicated from bufferSend.cpp ...
    uint32_t send_timeout = config.get_default<uint32_t>(unique_name, "send_timeout", 20);
    uint32_t reconnect_time = config.get_default<uint32_t>(unique_name, "reconnect_time", 5);

    int n_beams = config.get<int>(unique_name, "frb2_num_beams");

    json frb_targets = config.get<std::vector<json> >(unique_name, "frb_search_nodes");
    for (json target : frb_targets) {
        if (!target.is_object()) {
            throw std::invalid_argument(fmt::format(fmt("Expect 'frb_targets' to contain a list of dicts")));
        }
        std::string ip = target["ip"].get<std::string>();
        int port = target["port"].get<int>();
        int beamset = target["beamset"].get<int>();
        //std::vector<int> beam_ids = target["beams"].get<std::vector<int> >();
        int beam_index_min = target["beam_index_min"].get<int>();
        int beam_index_max = target["beam_index_max"].get<int>();
        if (beam_index_max > n_beams)
            FATAL_ERROR("beam_index_max ({:d}) must be <= n_beams ({:d})",
                        beam_index_max, n_beams);
        if (beam_index_min >= beam_index_max)
            FATAL_ERROR("beam_index_min ({:d}) must be < beam_index_max ({:d})",
                        beam_index_min, beam_index_max);
        if ((beam_index_min < 0) || (beam_index_max < 0))
            FATAL_ERROR("beam_index_min ({:d}) and beam_index_max ({:d}) must be non-negative",
                        beam_index_min, beam_index_max);

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

        dest->reconnect_time = reconnect_time;
        dest->send_timeout = send_timeout;

        dest->beam_index_min = beam_index_min;
        dest->beam_index_max = beam_index_max;
        dest->beamset = beamset;
        
        dests.push_back(dest);
    }
}

pirateFrbSend::~pirateFrbSend() {}

struct pirateNetworkHeader {
    uint32_t magic_number;
    uint32_t config_length;
};


void pirateFrbSend::main_thread() {
    int frame_id = 0;

    // Fire up all the connect threads...
    for (auto dest : dests) {
        std::string name = "pirateFrbSend";
        dest->connect_thread = std::thread(&networkDestination::connect_to_server, std::ref(*dest),
                                           name, std::ref(stop_thread));
    }

    INFO("Waiting for data in beam_positions buffer...");
    uint8_t* beam_ids_ptr = nullptr;
    uint8_t* beam_pos_ptr = nullptr;
    float*   beam_pos = nullptr;
    int16_t* beam_ids = nullptr;
    size_t expect_size;
    int n_beams = config.get<int>(unique_name, "frb2_num_beams");

    beam_pos_ptr = beam_position_buf->wait_for_full_frame(unique_name, frame_id);
    if (beam_pos_ptr == nullptr)
        goto cleanup;

    INFO("Waiting for data in beam_ids buffer...");
    beam_ids_ptr = beam_id_buf->wait_for_full_frame(unique_name, frame_id);
    if (beam_ids_ptr == nullptr)
        goto cleanup;

    // Pull out beam positions and ids...
    beam_pos = reinterpret_cast<float*>(beam_pos_ptr);
    assert(beam_pos);
    expect_size = sizeof(float) * 2 * n_beams;
    assert(beam_position_buf->frame_size == expect_size);

    beam_ids = reinterpret_cast<int16_t*>(beam_ids_ptr);
    assert(beam_ids);
    expect_size = sizeof(int16_t) * n_beams;
    assert(beam_id_buf->frame_size == expect_size);

    // Save the subsets that we are going to send to each of our destinations.
    for (auto dest : dests) {
        int n_dest = dest->beam_index_max - dest->beam_index_min;
        dest->beam_positions.reserve(2 * n_dest);
        dest->beam_positions.insert(dest->beam_positions.begin(),
                                    beam_pos + 2 * dest->beam_index_min,
                                    beam_pos + 2 * dest->beam_index_max);
        dest->beam_ids.reserve(n_dest);
        dest->beam_ids.insert(dest->beam_ids.begin(),
                              beam_ids + dest->beam_index_min,
                              beam_ids + dest->beam_index_max);
    }

    // Release beam position & id buffers
    beam_position_buf->mark_frame_empty(unique_name, frame_id);
    beam_id_buf->mark_frame_empty(unique_name, frame_id);

    while (!stop_thread) {
        INFO("Waiting for frame {:d}...");
        uint8_t* intensity_frame = intensity_buf->wait_for_full_frame(unique_name, frame_id);
        if (intensity_frame == nullptr)
            break;
        uint8_t* offset_scale_frame = offset_scale_buf->wait_for_full_frame(unique_name, frame_id);
        if (offset_scale_frame == nullptr)
            break;
        INFO("Got data for frame {:d}");

        uint32_t num_full_frames = intensity_buf->get_num_full_frames();

        // This logic is from sendBuffer, but with drop_frames assumed true.

        if ((float)num_full_frames / (float)intensity_buf->num_frames > drop_threshold) {
            // If the number of full frames is high, then we drop some
            // frames, because we likely aren't sending fast enough to
            // keep up with the data rate.
            INFO("Number of full frames in intensity buffer is {:d} (total frames: {:d}), dropping "
                 "frame_id {:d}", num_full_frames, intensity_buf->num_frames, frame_id);
            //dropped_frame_counter.inc();
        } else {
            bool any_connected = false;
            for (auto dest : dests)
                if (dest->connected) {
                    any_connected = true;
                    INFO("Sending frame {:d} to dest {:s}:{:d}",
                         frame_id, dest->server_ip, dest->server_port);
                    if (!send_frame(intensity_frame, offset_scale_frame, dest)) {
                        INFO("Failed to send frame {:d} to dest {:s}:{:d}; closing connection",
                             frame_id, dest->server_ip, dest->server_port);
                        dest->close_connection();
                    } else
                        DEBUG("Sent frame {:d} to {:s}:{:d}", frame_id,
                              dest->server_ip, dest->server_port);
                }
            if (!any_connected) {
                // Mark this frame as dropped
                INFO("Dropping intensity frame {:d}, because all connections are down.",
                     frame_id);
                //dropped_frame_counter.inc();
            }
        }

        intensity_buf->mark_frame_empty(unique_name, frame_id);
        offset_scale_buf->mark_frame_empty(unique_name, frame_id);

        frame_id = (frame_id + 1) % intensity_buf->num_frames;
    }

 cleanup:
    INFO("Cleaning up: closing connections...");
    for (auto dest : dests)
        dest->close_connection();
    INFO("Cleaning up: joining TCP threads...");
    for (auto dest : dests)
        dest->connect_thread.join();
    INFO("Exiting");
}

bool pirateFrbSend::send_frame(uint8_t* intensity_frame, uint8_t* offset_scale_frame,
                               std::shared_ptr<pirateDestination> dest) {
    int32_t n = 0;
    int32_t n_sent = 0;

    //pirateDestination* dest = reinterpret_cast<pirateDestination*>(&_dest);

    if (!dest->sent_header) {
        INFO("Sending header");

        struct XEngineMetadata meta;
        meta.version = 2;

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
        for (auto& it : upchan_ranges.items()) {
            //int upchan_factor = it.key().get<int>();
            int upchan_factor = std::stoi(it.key());
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

        meta.freq_channels = config.get<std::vector<long> >(unique_name, "frequency_channels");

        for (int16_t b : dest->beam_ids)
            meta.beam_ids.push_back(b);

        INFO("XEngineMetadata validate:");
        meta.validate();
        
        //std::string config = "yaml";
        std::string config = meta.to_yaml_string(true);

        INFO("Formatted X-engine metadata for Pirate: {:s}", config);
        
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

    return true;
}

