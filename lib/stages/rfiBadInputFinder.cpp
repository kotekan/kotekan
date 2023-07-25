#include "rfiBadInputFinder.hpp"

#include "Config.hpp"       // for Config
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"
#include "buffer.hpp"            // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for get_fpga_seq_num, get_stream_id
#include "kotekanLogging.hpp"  // for ERROR, INFO, DEBUG
#include "restServer.hpp"      // for restServer, connectionInstance, HTTP_RESPONSE, HTTP_RESPO...
#include "rfi_functions.h"     // for RFIHeader

#ifdef DEBUGGING
#include "util.h" // for e_time
#endif

#include <arpa/inet.h>  // for inet_aton
#include <atomic>       // for atomic_bool
#include <cmath>        // for pow, sqrt
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, _Placeholder, bind, _1, _2, function
#include <mutex>        // for mutex
#include <netinet/in.h> // for sockaddr_in, IPPROTO_UDP, htons
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for runtime_error
#include <stdlib.h>     // for free, malloc
#include <string.h>     // for memcpy, memset
#include <string>       // for string, allocator, operator+
#include <sys/socket.h> // for sendto, socket, AF_INET, SOCK_DGRAM
#include <vector>       // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(rfiBadInputFinder);

rfiBadInputFinder::rfiBadInputFinder(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rfiBadInputFinder::main_thread, this)) {
    // Get buffer from framework
    rfi_buf = get_buffer("rfi_in");
    // Register stage as consumer
    register_consumer(rfi_buf, unique_name.c_str());

    // Intialize internal config
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _num_total_freq = config.get_default<uint32_t>(unique_name, "num_total_freq", 1024);
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // Rfi paramters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    _rfi_combined = config.get_default<bool>(unique_name, "rfi_combined", true);
    _frames_per_packet = config.get_default<uint32_t>(unique_name, "bi_frames_per_packet", 10);
    // Stage-specific paramters
    total_links = config.get_default<uint32_t>(unique_name, "total_links", 1);
    dest_port = config.get<uint32_t>(unique_name, "destination_port");
    dest_server_ip = config.get<std::string>(unique_name, "destination_ip");
    dest_protocol = config.get_default<std::string>(unique_name, "destination_protocol", "UDP");

    // Set stats variables
    stats_sigma = 3;
    // Initialize rest server endpoint
    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    endpoint = unique_name + "/rfi_bad_input_finder";
    rest_server.register_post_callback(endpoint,
                                       std::bind(&rfiBadInputFinder::rest_callback, this, _1, _2));
}

rfiBadInputFinder::~rfiBadInputFinder() {
    restServer::instance().remove_json_callback(endpoint);
}

void rfiBadInputFinder::rest_callback(connectionInstance& conn, nlohmann::json& json_request) {
    // Notify that request was received
    INFO("RFI Callback Received... Changing Parameters")
    // Lock mutex
    rest_callback_mutex.lock();
    // Adjust parameters
    _frames_per_packet = json_request["frames_per_packet"].get<int>();
    config.update_value(unique_name, "bi_frames_per_packet", _frames_per_packet);
    // Send reply indicating success
    conn.send_empty_reply(HTTP_RESPONSE::OK);
    // Unlock mutex
    rest_callback_mutex.unlock();
}

float rfiBadInputFinder::median(float array[], uint32_t num) {
    float new_array[num];
    for (uint32_t x = 0; x < num; x++) {
        new_array[x] = array[x];
    }
    // sort array (super simply)
    for (uint32_t x = 0; x < num; x++) {
        for (uint32_t y = 0; y < num - 1; y++) {
            if (new_array[y] > new_array[y + 1]) {
                float tmp = new_array[y + 1];
                new_array[y + 1] = new_array[y];
                new_array[y] = tmp;
            }
        }
    }
    if (num % 2 != 0)
        return new_array[((num + 1) / 2) - 1];
    else
        return (new_array[(num / 2) - 1] + new_array[num / 2]) / 2;
}

float rfiBadInputFinder::deviation(float array[], uint32_t num, float outliercut) {
    float total = 0;
    uint32_t N = 0;
    float m = median(array, num);
    for (uint32_t i = 0; i < num; i++) {
        if (array[i] < m + outliercut && array[i] > m - outliercut) {
            total += pow(array[i] - m, 2);
            N += 1;
        }
    }
    return sqrt(total / (N - 1));
}

void rfiBadInputFinder::main_thread() {

    // Intialize frame variables
    uint32_t frame_id = 0;
    uint8_t* frame = nullptr;
    uint32_t frame_counter = 0;
    // Initialize arrays
    float rfi_data[_num_local_freq * _num_elements];
    stream_t StreamIDs[total_links];
    uint32_t freq_bins[_num_local_freq];
    uint8_t faulty_counter[_num_local_freq * _num_elements];
    memset(faulty_counter, (uint8_t)0, sizeof(faulty_counter));
    memset(freq_bins, (uint8_t)0, sizeof(freq_bins));
    auto& tel = Telescope::instance();

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
    uint32_t packet_length =
        sizeof(rfi_header) + _num_local_freq * sizeof(uint32_t) + sizeof(faulty_counter);
    char* packet_buffer = (char*)malloc(packet_length);
    // UDP Stuff
    uint32_t bytes_sent = 0;
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
    // Connection succesful
    INFO("UDP Connection: {:d} {:s}", dest_port, dest_server_ip);
    // Main while loop
    while (!stop_thread) {
        // Get a frame
        frame = wait_for_full_frame(rfi_buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;
#ifdef DEBUGGING
        // Reset Timer
        double start_time = e_time();
#endif
        // Copy frame data to array
        memcpy(rfi_data, frame, rfi_buf->frame_size);
        // Add frame metadata to header
        if (frame_counter == 0) {
            // TODO: stream_id - this uses internal knowledge of the structure
            StreamIDs[0] = get_stream_id(rfi_buf, frame_id);
            rfi_header.streamID = (uint16_t)(StreamIDs[0].id);
            rfi_header.seq_num = get_fpga_seq_num(rfi_buf, frame_id);
        }
        // Compute statistics
        float med = median(rfi_data, _num_local_freq * _num_elements);
        float std = deviation(rfi_data, _num_local_freq * _num_elements, 0.1);
        // Compute number of faulty frames based on stats
        for (uint32_t i = 0; i < _num_local_freq * _num_elements; i++) {
            if (rfi_data[i] > med + stats_sigma * std || rfi_data[i] < med - stats_sigma * std) {
                faulty_counter[i]++;
            }
        }
        // Move to next frame
        mark_frame_empty(rfi_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % rfi_buf->num_frames;
        // Adjust frame counter
        frame_counter++;
        // After 10 frames
        rest_callback_mutex.lock();
        if (frame_counter == _frames_per_packet) {

            // Get current frequency bin
            uint32_t current_freq_bin = tel.to_freq_id(StreamIDs[0]);
            // TODO JSW: Handle num_local_freq > 1
            freq_bins[0] = current_freq_bin;

            // Reset counter
            frame_counter = 0;
            // Add Header to packet
            memcpy(packet_buffer, &rfi_header, sizeof(rfi_header));
            // Add frequency bins to packet
            memcpy(packet_buffer + sizeof(rfi_header), freq_bins,
                   _num_local_freq * sizeof(uint32_t));
            // Add Data to packet
            memcpy(packet_buffer + sizeof(rfi_header) + _num_local_freq * sizeof(uint32_t),
                   faulty_counter, sizeof(faulty_counter));
            // Send Packet
            bytes_sent = sendto(socket_fd, packet_buffer, packet_length, 0,
                                (struct sockaddr*)&saddr_remote, sizeof(sockaddr_in));
            // Check if packet sent properly
            if (bytes_sent != packet_length)
                ERROR("SOMETHING WENT WRONG IN UDP TRANSMIT");
            DEBUG("Frame ID {:d} Succesfully Sent. {:d} Bytes in {:f}ms", frame_id, bytes_sent,
                  (e_time() - start_time) * 1000);
            // Reset Counter
            memset(faulty_counter, (uint8_t)0, sizeof(faulty_counter));
        }
        rest_callback_mutex.unlock();
    }
    free(packet_buffer);
}
