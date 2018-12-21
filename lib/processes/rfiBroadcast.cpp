#include "rfiBroadcast.hpp"

#include "chimeMetadata.h"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "util.h"
#include "visUtil.hpp"

#include <arpa/inet.h>
#include <errno.h>
#include <functional>
#include <mutex>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

REGISTER_KOTEKAN_PROCESS(rfiBroadcast);

rfiBroadcast::rfiBroadcast(Config& config, const string& unique_name,
                           bufferContainer& buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&rfiBroadcast::main_thread, this)) {
    // Get buffer from framework
    rfi_buf = get_buffer("rfi_in");
    // Get buffer from framework
    rfi_mask_buf = get_buffer("rfi_mask");
    // Register process as consumer
    register_consumer(rfi_buf, unique_name.c_str());
    // Register process as consumer
    register_consumer(rfi_mask_buf, unique_name.c_str());

    // Intialize internal config
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _num_total_freq = config.get_default<uint32_t>(unique_name, "num_total_freq", 1024);
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // Rfi paramters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    _rfi_combined = config.get_default<bool>(unique_name, "rfi_combined", true);
    _frames_per_packet = config.get_default<uint32_t>(unique_name, "frames_per_packet", 1);
    // Process specific paramters
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
    endpoint_zero = unique_name + "/percent_zeroed";
    rest_server.register_get_callback(endpoint_zero, std::bind(&rfiBroadcast::rest_zero, this, _1));
}

rfiBroadcast::~rfiBroadcast() {
    restServer::instance().remove_json_callback(endpoint);
    restServer::instance().remove_json_callback(endpoint_zero);
}

void rfiBroadcast::rest_callback(connectionInstance& conn, json& json_request) {
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
    json reply;
    reply["percentage_zeroed"] = perc_zeroed.average();
    conn.send_json_reply(reply);
}

void rfiBroadcast::main_thread() {
    // Intialize variables
    uint32_t frame_id = 0;
    uint32_t frame_mask_id = 0;
    uint32_t i, j, f;
    uint32_t bytes_sent = 0;
    uint8_t* frame = NULL;
    uint8_t* frame_mask = NULL;
    uint32_t link_id = 0;
    uint16_t StreamIDs[total_links];
    uint64_t fake_seq = 0;
    prometheusMetrics& metrics = prometheusMetrics::instance();

    // Intialize packet header
    struct RFIHeader rfi_header = {.rfi_combined = (uint8_t)_rfi_combined,
                                   .sk_step = _sk_step,
                                   .num_elements = _num_elements,
                                   .samples_per_data_set = _samples_per_data_set,
                                   .num_total_freq = _num_total_freq,
                                   .num_local_freq = _num_local_freq,
                                   .frames_per_packet = _frames_per_packet,
                                   .seq_num = 0,
                                   .streamID = 0};

    // Intialize empty packet
    uint32_t packet_length = sizeof(rfi_header) + _num_local_freq * sizeof(float);
    char* packet_buffer = (char*)malloc(packet_length);
    // Filter by protocol, currently only UDP supported
    if (dest_protocol == "UDP") {
        // UDP Stuff
        struct sockaddr_in saddr_remote; /* the libc network address data structure */
        socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_fd == -1) {
            ERROR("Could not create UDP socket for output stream");
            return;
        }
        memset((char*)&saddr_remote, 0, sizeof(sockaddr_in));
        saddr_remote.sin_family = AF_INET;
        saddr_remote.sin_port = htons(dest_port);
        if (inet_aton(dest_server_ip.c_str(), &saddr_remote.sin_addr) == 0) {
            ERROR("Invalid address given for remote server");
            return;
        }
        // Connection succesful
        INFO("UDP Connection: %i %s", dest_port, dest_server_ip.c_str());
        // Endless loop
        while (!stop_thread) {
            // Initialize arrays
            float rfi_data[total_links][_num_local_freq * _samples_per_data_set / _sk_step];
            float rfi_avg[total_links][_num_local_freq];
            // Initialize arrays
            uint32_t mask_total = 0;
            // Zero Average array
            memset(rfi_avg, (float)0, sizeof(rfi_avg));
            // Loop through all frames that should be averages together
            for (f = 0; f < _frames_per_packet * total_links; f++) {
                // Get Frame of Mask
                frame_mask = wait_for_full_frame(rfi_mask_buf, unique_name.c_str(), frame_mask_id);
                if (frame_mask == NULL)
                    break;
                // Get Frame
                frame = wait_for_full_frame(rfi_buf, unique_name.c_str(), frame_id);
                if (frame == NULL)
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
                    StreamIDs[link_id] = link_id;
                } else {
                    StreamIDs[link_id] = get_stream_id(rfi_buf, frame_id);
                }
                // Sum over the whole frame
                for (i = 0; i < _num_local_freq; i++) {
                    for (j = 0; j < _samples_per_data_set / _sk_step; j++) {
                        rfi_avg[link_id][i] += rfi_data[link_id][i + _num_local_freq * j];
                        mask_total += frame_mask[i + _num_local_freq * j];
                        //                        DEBUG("RFI Mask %d, Mask Total: %d, Frame Size:
                        //                        %d",, mask_total,rfi_mask_buf->frame_size);
                    }
                }
                // Mark Frame Empty
                mark_frame_empty(rfi_mask_buf, unique_name.c_str(), frame_mask_id);
                mark_frame_empty(rfi_buf, unique_name.c_str(), frame_id);
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

            // Get current frequency bin and set add the prometheus metric
            stream_id_t current_stream_id = extract_stream_id(StreamIDs[0]);
            uint32_t current_freq_bin = bin_number_chime(&current_stream_id);
            std::string tags = "freq_bin=\"" + std::to_string(current_freq_bin) + "\"";
            metrics.add_process_metric("kotekan_rfi_broadcast_mask_percent", unique_name,
                                       perc_zeroed.average(), tags);

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
                        DEBUG("SK value %f for freq %d, stream %d", rfi_avg[j][i], i, StreamIDs[j]);
                        DEBUG("Percent Masked %f for freq %d stream %d",
                              100.0 * (float)mask_total / rfi_mask_buf->frame_size, i,
                              StreamIDs[j]);
                    }
                }
                // Add Stream ID to header
                rfi_header.streamID = StreamIDs[j];
                // Add Header to packet
                memcpy(packet_buffer, &rfi_header, sizeof(rfi_header));
                // Add Data to packet
                memcpy(packet_buffer + sizeof(rfi_header), rfi_avg[j],
                       _num_local_freq * sizeof(float));
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
            DEBUG("Frame ID %d Succesfully Broadcasted %d links of %d Bytes in %fms", frame_id,
                  total_links, bytes_sent, (e_time() - start_time) * 1000);
        }
    } else {
        ERROR("Bad protocol: %s Only UDP currently Supported", dest_protocol.c_str());
    }
    free(packet_buffer);
}
