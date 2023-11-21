#include "rfiBroadcast.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"         // for Telescope
#include "buffer.hpp"            // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp"   // for bufferContainer
#include "chimeMetadata.hpp"     // for stream_t, get_fpga_seq_num, get_stream_id
#include "kotekanLogging.hpp"    // for ERROR, DEBUG, INFO
#include "prometheusMetrics.hpp" // for Counter, Metrics, MetricFamily
#include "restServer.hpp"        // for restServer, connectionInstance, HTTP_RESPONSE, HTTP_RES...
#include "rfi_functions.h"       // for RFIHeader
#include "visUtil.hpp"           // for movingAverage

#ifdef DEBUGGING
#include "util.h" // for e_time
#endif

#include "fmt.hpp" // for format, fmt

#include <algorithm>    // for copy, copy_backward, equal, max
#include <arpa/inet.h>  // for inet_aton
#include <atomic>       // for atomic_bool
#include <deque>        // for deque
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, _Placeholder, bind, _1, _2, function
#include <mutex>        // for mutex, lock_guard
#include <netinet/in.h> // for sockaddr_in, IPPROTO_UDP, htons
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for runtime_error
#include <stdlib.h>     // for free, malloc
#include <string.h>     // for memcpy, memset
#include <string>       // for string, allocator, to_string, operator+, operator==
#include <sys/socket.h> // for sendto, socket, AF_INET, SOCK_DGRAM
#include <vector>       // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(rfiBroadcast);

rfiBroadcast::rfiBroadcast(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rfiBroadcast::main_thread, this)),
    sample_counter(Metrics::instance().add_counter("kotekan_rfibroadcast_sample_total", unique_name,
                                                   {"freq_id"})),
    flagged_sample_counter(Metrics::instance().add_counter(
        "kotekan_rfibroadcast_dropped_sample_total", unique_name, {"freq_id"})) {

    // Get buffer from framework
    rfi_buf = get_buffer("rfi_in");
    // Get buffer from framework
    rfi_mask_buf = get_buffer("rfi_mask");
    // Register stage as consumer
    rfi_buf->register_consumer(unique_name);
    // Register stage as consumer
    rfi_mask_buf->register_consumer(unique_name);

    // Intialize internal config
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _num_total_freq = config.get_default<uint32_t>(unique_name, "num_total_freq", 1024);
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // Rfi paramters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    _rfi_combined = config.get_default<bool>(unique_name, "rfi_combined", true);
    _frames_per_packet = config.get_default<uint32_t>(unique_name, "frames_per_packet", 1);
    // Stage-specific paramters
    total_links = config.get_default<uint32_t>(unique_name, "total_links", 1);
    dest_port = config.get<uint32_t>(unique_name, "destination_port");
    dest_server_ip = config.get<std::string>(unique_name, "destination_ip");
    dest_protocol = config.get_default<std::string>(unique_name, "destination_protocol", "UDP");
    replay = config.get_default<bool>(unique_name, "replay", false);

    // Initialize rest server endpoint
    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    endpoint = unique_name + "/change_params";
    rest_server.register_post_callback(endpoint,
                                       std::bind(&rfiBroadcast::rest_callback, this, _1, _2));
    endpoint_zero = fmt::format(fmt("{:s}/percent_zeroed"), unique_name);
    rest_server.register_get_callback(endpoint_zero, std::bind(&rfiBroadcast::rest_zero, this, _1));
}

rfiBroadcast::~rfiBroadcast() {
    restServer::instance().remove_json_callback(endpoint);
    restServer::instance().remove_json_callback(endpoint_zero);
}

void rfiBroadcast::rest_callback(connectionInstance& conn, nlohmann::json& json_request) {
    // Notify that request was received
    INFO("RFI Callback Received... Changing Parameters")
    // Lock mutex
    rest_callback_mutex.lock();
    // Adjust parameters
    _frames_per_packet = json_request["frames_per_packet"].get<int>();
    config.update_value(unique_name, "frames_per_packet", _frames_per_packet);
    // Send reply indicating success
    conn.send_empty_reply(HTTP_RESPONSE::OK);
    // Unlock mutex
    rest_callback_mutex.unlock();
}

void rfiBroadcast::rest_zero(connectionInstance& conn) {
    std::lock_guard<std::mutex> lock(rest_zero_callback_mutex);
    // Notify that request was received
    INFO("RFI Broadcast: Current Zeroing Percentage Sent")
    nlohmann::json reply;
    reply["percentage_zeroed"] = perc_zeroed.average();
    conn.send_json_reply(reply);
}

void rfiBroadcast::main_thread() {
    // Initialize variables
    uint32_t frame_id = 0;
    uint32_t frame_mask_id = 0;
    uint32_t i, j, f;
    uint32_t bytes_sent = 0;
    uint8_t* frame = nullptr;
    uint8_t* frame_mask = nullptr;
    uint32_t link_id = 0;
    stream_t StreamIDs[total_links];
    uint32_t freq_bins[_num_local_freq];
    memset(freq_bins, (uint8_t)0, sizeof(freq_bins));
    uint64_t fake_seq = 0;
    const uint64_t sk_samples_per_frame = _samples_per_data_set / _sk_step;
    auto& tel = Telescope::instance();

    // Initialize packet header
    struct RFIHeader rfi_header = {.rfi_combined = (uint8_t)_rfi_combined,
                                   .sk_step = _sk_step,
                                   .num_elements = _num_elements,
                                   .samples_per_data_set = _samples_per_data_set,
                                   .num_total_freq = _num_total_freq,
                                   .num_local_freq = _num_local_freq,
                                   .frames_per_packet = _frames_per_packet,
                                   .seq_num = 0,
                                   .streamID = 0};

    // Initialize empty packet
    uint32_t packet_length =
        sizeof(rfi_header) + _num_local_freq * sizeof(uint32_t) + _num_local_freq * sizeof(float);
    char* packet_buffer = (char*)malloc(packet_length);
    // Filter by protocol, currently only UDP supported
    if (dest_protocol == "UDP") {
        // UDP Stuff
        struct sockaddr_in saddr_remote; /* the libc network address data structure */
        socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_fd == -1) {
            ERROR("Could not create UDP socket for output stream");
            free(packet_buffer);
            return;
        }
        memset((char*)&saddr_remote, 0, sizeof(sockaddr_in));
        saddr_remote.sin_family = AF_INET;
        saddr_remote.sin_port = htons(dest_port);
        if (inet_aton(dest_server_ip.c_str(), &saddr_remote.sin_addr) == 0) {
            ERROR("Invalid address given for remote server");
            return;
        }
        // Connection successful
        INFO("UDP Connection: {:d} {:s}", dest_port, dest_server_ip);
        // Endless loop
        while (!stop_thread) {
            // Initialize arrays
            float rfi_data[total_links][_num_local_freq * _samples_per_data_set / _sk_step];
            float rfi_avg[total_links][_num_local_freq];
            // Initialize arrays
            uint64_t mask_total = 0;
            // Zero Average array
            memset(rfi_avg, (float)0, sizeof(rfi_avg));
            // Loop through all frames that should be averages together
            for (f = 0; f < _frames_per_packet * total_links; f++) {
                // Get Frame of Mask
                frame_mask = rfi_mask_buf->wait_for_full_frame(unique_name, frame_mask_id);
                if (frame_mask == nullptr)
                    break;
                // Get Frame
                frame = rfi_buf->wait_for_full_frame(unique_name, frame_id);
                if (frame == nullptr)
                    break;
                // Copy frame data to array
                memcpy(rfi_data[link_id], frame, rfi_buf->frame_size);
                // Adjust Header on initial frame
                if (f == 0) {
                    if (replay) {
                        rfi_header.seq_num = (int64_t)fake_seq;
                    } else {
                        rfi_header.seq_num = get_fpga_seq_num(rfi_buf, frame_id);
                    }
                }
                // Adjust Stream ID's
                if (replay) {
                    // TODO: stream_id - this uses internal knowledge of the structure
                    StreamIDs[link_id].id = link_id;
                } else {
                    StreamIDs[link_id] = get_stream_id(rfi_buf, frame_id);
                }
                // Sum over the whole frame
                for (i = 0; i < _num_local_freq; i++) {
                    for (j = 0; j < _samples_per_data_set / _sk_step; j++) {
                        rfi_avg[link_id][i] += rfi_data[link_id][i + _num_local_freq * j];
                        mask_total += frame_mask[i + _num_local_freq * j];
                        //                        DEBUG("RFI Mask {:d}, Mask Total: {:d}, Frame
                        //                        Size:
                        //                        {:d}",, mask_total,rfi_mask_buf->frame_size);
                    }
                }
                // Mark Frame Empty
                rfi_mask_buf->mark_frame_empty(unique_name, frame_mask_id);
                rfi_buf->mark_frame_empty(unique_name, frame_id);
                frame_mask_id = (frame_mask_id + 1) % rfi_mask_buf->num_frames;
                frame_id = (frame_id + 1) % rfi_buf->num_frames;
                link_id = (link_id + 1) % total_links;
            }
            // Lock callback mutex
            rest_callback_mutex.lock();
            rest_zero_callback_mutex.lock();
            // Compute Current Mask Percentage
            float tmp = 100.0 * (float)mask_total
                        / (rfi_mask_buf->frame_size * _frames_per_packet * total_links);
            perc_zeroed.add_sample(tmp);

            // Increment the prometheus metrics for the total number of samples and the total number
            // of flagged samples
            uint32_t current_freq_bin = tel.to_freq_id(StreamIDs[0]);
            sample_counter.labels({std::to_string(current_freq_bin)}).inc(sk_samples_per_frame);
            flagged_sample_counter.labels({std::to_string(current_freq_bin)}).inc(mask_total);

            // TODO JSW: Handle num_local_freq > 1
            freq_bins[0] = current_freq_bin;

#ifdef DEBUGGING
            // Reset Timer (can't time previous loop due to wait for frame blocking call)
            double start_time = e_time();
#endif
            // Loop through each link to send data seperately
            for (j = 0; j < total_links; j++) {
                // Normalize Sum (Take Average)
                for (i = 0; i < _num_local_freq; i++) {
                    rfi_avg[j][i] /= _frames_per_packet * (_samples_per_data_set / _sk_step);
                    if (i == 0) {
                        // TODO: stream_id - this uses internal knowledge of the structure
                        DEBUG("SK value {:f} for freq {:d}, stream {:d}", rfi_avg[j][i], i,
                              StreamIDs[j].id);
                        DEBUG("Percent Masked {:f} for freq {:d} stream {:d}",
                              100.0 * (float)mask_total / rfi_mask_buf->frame_size, i,
                              StreamIDs[j].id);
                    }
                }
                // Add Stream ID to header
                // TODO: stream_id - this uses internal knowledge of the structure
                rfi_header.streamID = (uint16_t)(StreamIDs[j].id);
                // Add Header to packet
                memcpy(packet_buffer, &rfi_header, sizeof(rfi_header));
                // Add frequency bins to packet
                memcpy(packet_buffer + sizeof(rfi_header), freq_bins,
                       _num_local_freq * sizeof(uint32_t));
                // Add Data to packet
                memcpy(packet_buffer + sizeof(rfi_header) + _num_local_freq * sizeof(uint32_t),
                       rfi_avg[j], _num_local_freq * sizeof(float));
                // Send Packet
                bytes_sent = sendto(socket_fd, packet_buffer, packet_length, 0,
                                    (struct sockaddr*)&saddr_remote, sizeof(sockaddr_in));
                // Check if packet sent properly
                if (bytes_sent != packet_length)
                    ERROR("SOMETHING WENT WRONG IN UDP TRANSMIT");
            }
            // Adjust fake_seq num (only for replay mode)
            fake_seq += _samples_per_data_set * _frames_per_packet;
            // Unlock callback mutex
            rest_callback_mutex.unlock();
            rest_zero_callback_mutex.unlock();
            DEBUG("Frame ID {:d} Successfully Broadcasted {:d} links of {:d} Bytes in {:f}ms",
                  frame_id, total_links, bytes_sent, (e_time() - start_time) * 1000);
        }
    } else {
        ERROR("Bad protocol: {:s} Only UDP currently Supported", dest_protocol);
    }
    free(packet_buffer);
}
