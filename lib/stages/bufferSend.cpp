#include "bufferSend.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"              // for Buffer, get_num_full_frames, mark_frame_empty, register...
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for DEBUG2, ERROR, DEBUG, WARN, INFO
#include "metadata.h"            // for metadataContainer
#include "prometheusMetrics.hpp" // for Metrics, Counter

#include "fmt.hpp" // for format, fmt

#include <arpa/inet.h> // for inet_addr
#include <cerrno>      // for errno
#include <chrono>
#include <cstring>      // for strerror, size_t
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, ref, function
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for runtime_error
#include <strings.h>    // for bzero
#include <sys/socket.h> // for send, MSG_NOSIGNAL, connect, setsockopt, socket, AF_INET
#include <sys/time.h>   // for timeval
#include <thread>       // for thread
#include <unistd.h>     // for close, sleep
#include <vector>       // for vector

// Some systems don't support MSG_NOSIGNAL and don't include it in socket.h
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(bufferSend);

bufferSend::bufferSend(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferSend::main_thread, this)),
    dropped_frame_counter(
        Metrics::instance().add_counter("kotekan_buffer_send_dropped_frame_count", unique_name)) {

    buf = get_buffer("buf");
    register_consumer(buf, unique_name.c_str());

    connected = false;
    server_ip = config.get<std::string>(unique_name, "server_ip");
    server_port = config.get_default<uint32_t>(unique_name, "server_port", 11024);

    send_timeout = config.get_default<uint32_t>(unique_name, "send_timeout", 20);
    reconnect_time = config.get_default<uint32_t>(unique_name, "reconnect_time", 5);
    drop_frames = config.get_default<bool>(unique_name, "drop_frames", true);

    // Publish current dropped frame count.

    bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(server_ip.c_str());
    server_addr.sin_port = htons(server_port);

    socket_fd = -1;
}

bufferSend::~bufferSend() {}

void bufferSend::main_thread() {

    int frame_id = 0;

    std::thread connect_thread = std::thread(&bufferSend::connect_to_server, std::ref(*this));

    while (!stop_thread) {

        uint8_t* frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        uint32_t num_full_frames = get_num_full_frames(buf);

        if (drop_frames && num_full_frames > (((uint32_t)buf->num_frames + 1) / 2)) {
            // If the number of full frames is high, then we drop some frames,
            // because we likely aren't sending fast enough to up with the data rate.
            INFO("Number of full frames in buffer {:s} is {:d} (total frames: {:d}), dropping "
                 "frame_id {:d}",
                 buf->buffer_name, num_full_frames, buf->num_frames, frame_id);
            dropped_frame_counter.inc();
        } else if (drop_frames && !connected) {
            INFO("Dropping frame {:s}[{:d}], because connection to {:s}:{:d} is down.",
                 buf->buffer_name, frame_id, server_ip, server_port);
            dropped_frame_counter.inc();
        } else if (connected) {
            // Send header
            struct bufferFrameHeader header;
            const size_t header_len = sizeof(struct bufferFrameHeader);
            int32_t n = 0;
            int32_t n_sent = 0;

            header.frame_size = buf->frame_size;
            header.metadata_size = buf->metadata[frame_id]->metadata_size;

            DEBUG2("frame_size: {:d}, metadata_size: {:d}", header.frame_size,
                   header.metadata_size);

            // Recover from partial sends
            DEBUG2("Sending header");
            while ((n = send(socket_fd, &((uint8_t*)&header)[n_sent], header_len - n_sent,
                             MSG_NOSIGNAL))
                   > 0) {
                n_sent += n;
            }
            // Handle errors
            if (n < 0) {
                ERROR("Error {:s}, failed to send header to {:s}:{:d}", strerror(errno), server_ip,
                      server_port);
                close_connection();
                continue;
            }
            DEBUG2("Sent header: {:d}", n_sent);

            // Send metadata
            DEBUG2("Sending metadata");
            n_sent = 0;
            while ((n = send(socket_fd, &((uint8_t*)buf->metadata[frame_id]->metadata)[n_sent],
                             header.metadata_size - n_sent, MSG_NOSIGNAL))
                   > 0) {
                n_sent += n;
            }
            if (n < 0) {
                ERROR("Error {:s}, failed to metadata to {:s}:{:d}", strerror(errno), server_ip,
                      server_port);
                close_connection();
                continue;
            }
            DEBUG2("Sent metadata: {:d}", n_sent);

            // Send buffer frame.
            DEBUG2("Sending frame with {:d} bytes", header.frame_size);
            n_sent = 0;
            while ((n = send(socket_fd, &frame[n_sent], (int32_t)header.frame_size - n_sent,
                             MSG_NOSIGNAL))
                   > 0) {
                n_sent += n;
                // DEBUG("Total sent: {:d}", n_sent);
            }
            if (n < 0) {
                ERROR("Error {:s}, failed to frame data to {:s}:{:d}", strerror(errno), server_ip,
                      server_port);
                close_connection();
                continue;
            }
            DEBUG2("Sent frame: {:d}", n_sent);
            DEBUG("Sent frame: {:s}[{:d}] to {:s}:{:d}", buf->buffer_name, frame_id, server_ip,
                  server_port);

        } else {
            // Wait for connection and block
            INFO("Waiting for connection to {:s}:{:d}...", server_ip, server_port);
            std::unique_lock<std::mutex> connection_lock(connection_state_mutex);
            connection_state_cv.wait_for(connection_lock, std::chrono::seconds(1),
                                         [&]() { return (stop_thread || connected); });
            continue;
        }

        mark_frame_empty(buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
    }

    close_connection();
    connect_thread.join();
}

void bufferSend::close_connection() {
    if (socket_fd >= 0)
        close(socket_fd);

    socket_fd = -1;
    {
        std::unique_lock<std::mutex> connection_lock(connection_state_mutex);
        connected = false;
    }
    connection_state_cv.notify_all();
}

void bufferSend::connect_to_server() {

    while (!stop_thread) {

        DEBUG("Trying to connecting to server: {:s}:{:d}", server_ip, server_port);

        socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd == -1) {
            std::string msg = fmt::format(fmt("Could not create socket, errno: {:d} ({:s})"), errno,
                                          std::strerror(errno));
            ERROR("{:s}", msg);
            throw std::runtime_error(msg);
        }

        if (connect(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
            WARN("Could not connect to server {:s}:{:d}, error: {:s}({:d}), waiting {:d} seconds "
                 "to retry...",
                 server_ip, server_port, strerror(errno), errno, reconnect_time);
            close(socket_fd);
            // TODO Add a Stage level "breakable sleep" so this doesn't
            // lock up the shutdown process for upto reconnect_time seconds.
            sleep(reconnect_time);
            continue;
        }

        // Prevent SIGPIPE on send failure.
        // This is used for MacOS, since linux doesn't have SO_NOSIGPIPE
#ifdef SO_NOSIGPIPE
        int set = 1;
        if (setsockopt(socket_fd, SOL_SOCKET, SO_NOSIGPIPE, (void*)&set, sizeof(int)) < 0) {
            ERROR("bufferSend: setsockopt() NOSIGPIPE ");
        }
#endif

        // Set send timeout.
        struct timeval tv_timeout;
        tv_timeout.tv_sec = send_timeout;
        tv_timeout.tv_usec = 0;

        if (setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO, (void*)&tv_timeout, sizeof(tv_timeout))
            < 0) {
            ERROR("bufferSend: setsockopt() timeout failed.");
        }

        INFO("Connected to server {:s}:{:d} for sending buffer {:s}", server_ip, server_port,
             buf->buffer_name);
        {
            std::unique_lock<std::mutex> connection_lock(connection_state_mutex);
            connected = true;
        }

        // Notify that connection is established
        connection_state_cv.notify_one();

        // wait for connection to get closed
        std::unique_lock<std::mutex> connection_lock(connection_state_mutex);
        connection_state_cv.wait(connection_lock, [&]() { return !connected || stop_thread; });
    }
}

std::string bufferSend::dot_string(const std::string& prefix) const {
    std::string dot = Stage::dot_string(prefix);
    std::string target = fmt::format("{:s}:{:d}", server_ip, server_port);
    dot += fmt::format("{:s}\"{:s}\" [shape=doubleoctagon style=filled,color=lightblue]", prefix,
                       target);
    dot += fmt::format("{:s}\"{:s}\" -> \"{:s}\"", prefix, get_unique_name(), target);

    return dot;
}
