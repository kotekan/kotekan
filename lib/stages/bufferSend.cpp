#include "bufferSend.hpp"

#include "Config.hpp"            // for Config
#include "PipelineGraph.hpp"     // for PipelineGraph, GraphNode
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "configTracker.hpp"     // for ConfigTracker
#include "kotekanLogging.hpp"    // for DEBUG2, ERROR, INFO, DEBUG, WARN
#include "metadata.hpp"          // for metadataObject
#include "prometheusMetrics.hpp" // for Counter, Metrics

#include "fmt.hpp" // for compile_string_to_view, format, format_string, fmt

#include <arpa/inet.h>  // for htons, inet_addr
#include <cerrno>       // for errno
#include <chrono>       // for seconds
#include <cstring>      // for strerror, size_t
#include <functional>   // for bind, ref, function
#include <memory>       // for __shared_ptr_access, shared_ptr
#include <stdexcept>    // for runtime_error
#include <strings.h>    // for bzero
#include <sys/socket.h> // for send, MSG_NOSIGNAL, AF_INET, connect, setsockopt, socket
#include <sys/time.h>   // for timeval
#include <thread>       // for thread
#include <unistd.h>     // for close, sleep

// Some systems don't support MSG_NOSIGNAL and don't include it in socket.h
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::ConfigTracker;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(bufferSend);

bufferSend::bufferSend(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferSend::main_thread, this)),
    dropped_frame_counter(
        Metrics::instance().add_counter("kotekan_buffer_send_dropped_frame_count", unique_name)) {

    buf = get_buffer("buf");
    buf->register_consumer(unique_name);

    connected = false;
    server_ip = config.get<std::string>(unique_name, "server_ip");
    server_port = config.get_default<uint32_t>(unique_name, "server_port", 11024);

    send_timeout = config.get_default<uint32_t>(unique_name, "send_timeout", 20);
    reconnect_time = config.get_default<uint32_t>(unique_name, "reconnect_time", 5);
    drop_frames = config.get_default<bool>(unique_name, "drop_frames", true);
    drop_threshold = config.get_default<float>(unique_name, "drop_threshold", 0.6);

    // Publish current dropped frame count.

    bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(server_ip.c_str());
    server_addr.sin_port = htons(server_port);

    socket_fd = -1;

    // Default to the tracker's enabled state (set at startup, before stages).
    // A stage may opt out, but cannot opt in when the tracker is off, so the
    // configured value is ANDed with the tracker state.
    const bool ct_enabled = ConfigTracker::instance().is_enabled();
    use_config_tracker =
        config.get_default<bool>(unique_name, "use_config_tracker", ct_enabled) && ct_enabled;
    config_tracker_combined_hash = "";

    use_frame_desc = config.get_default<bool>(unique_name, "use_frame_desc", false);
    desc_sent = false;
}

bufferSend::~bufferSend() {}

void bufferSend::main_thread() {

    int frame_id = 0;

    std::thread connect_thread = std::thread(&bufferSend::connect_to_server, std::ref(*this));

    while (!stop_thread) {

        uint8_t* frame = buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        uint32_t num_full_frames = buf->get_num_full_frames();

        if (drop_frames && (float)num_full_frames / (float)buf->num_frames > drop_threshold) {
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
            int32_t n = 0;
            int32_t n_sent = 0;

            auto meta = buf->get_metadata(frame_id);
            auto metadata_size = meta->get_serialized_size();
            auto frame_size = buf->frame_size;
            bufferFrameHeader header;
            header.metadata_size = static_cast<uint32_t>(meta->get_serialized_size());
            header.frame_size = static_cast<uint32_t>(buf->frame_size);
            header.config_tracker_update =
                config_tracker_combined_hash != ConfigTracker::instance().getTrackerHash();
            // for legacy CHIME, do not send last field (config_tracker_update)
            size_t header_len = use_config_tracker ? sizeof(bufferFrameHeader)
                                                   : sizeof(bufferFrameHeaderNoConfigTracker);

            if (header.config_tracker_update)
                DEBUG("Config tracker data has been updated, sending new config tracker data.");
            DEBUG2("frame_size: {:d}, metadata_size: {:d}, config_tracker_update: {:d}",
                   header.frame_size, header.metadata_size, header.config_tracker_update);

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


            // Record the tracker state we just signaled to the receiver.
            config_tracker_combined_hash = ConfigTracker::instance().getTrackerHash();
            DEBUG2("Sent header: {:d}", n_sent);

            // Send the frame descriptor once per connection (independent of the
            // config tracker), right after the first frame's header: a 4-byte
            // JSON length followed by that many bytes. Later frames send neither.
            if (use_frame_desc && !desc_sent) {
                auto desc = buf->get_frame_desc();
                const std::string desc_json = desc ? desc->to_json().dump() : std::string();
                uint32_t frame_desc_size = static_cast<uint32_t>(desc_json.size());
                n_sent = 0;
                while ((n = send(socket_fd, &((uint8_t*)&frame_desc_size)[n_sent],
                                 sizeof(frame_desc_size) - n_sent, MSG_NOSIGNAL))
                       > 0)
                    n_sent += n;
                if (n < 0) {
                    ERROR("Error {:s}, failed to send frame_desc size to {:s}:{:d}",
                          strerror(errno), server_ip, server_port);
                    close_connection();
                    continue;
                }
                if (frame_desc_size > 0) {
                    n_sent = 0;
                    while ((n = send(socket_fd, desc_json.data() + n_sent, frame_desc_size - n_sent,
                                     MSG_NOSIGNAL))
                           > 0)
                        n_sent += n;
                    if (n < 0) {
                        ERROR("Error {:s}, failed to send frame_desc to {:s}:{:d}", strerror(errno),
                              server_ip, server_port);
                        close_connection();
                        continue;
                    }
                }
                desc_sent = true;
            }

            // Send metadata
            DEBUG2("Sending metadata");
            n_sent = 0;
            {
                char metabuf[metadata_size];
                meta->serialize(metabuf);
                while ((n = send(socket_fd, metabuf + n_sent, metadata_size - n_sent, MSG_NOSIGNAL))
                       > 0) {
                    n_sent += n;
                }
            }
            if (n < 0) {
                ERROR("Error {:s}, failed to metadata to {:s}:{:d}", strerror(errno), server_ip,
                      server_port);
                close_connection();
                continue;
            }
            DEBUG2("Sent metadata: {:d}", n_sent);

            // Send buffer frame.
            DEBUG2("Sending frame with {:d} bytes", frame_size);
            n_sent = 0;
            while ((n = send(socket_fd, &frame[n_sent], (int32_t)frame_size - n_sent, MSG_NOSIGNAL))
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

            // Set to true since we've sent a transmission.
            first_transmission_sent = true;

        } else {
            // Wait for connection and block
            INFO("Waiting for connection to {:s}:{:d}...", server_ip, server_port);
            std::unique_lock<std::mutex> connection_lock(connection_state_mutex);
            connection_state_cv.wait_for(connection_lock, std::chrono::seconds(1),
                                         [&]() { return (stop_thread || connected); });
            continue;
        }

        buf->mark_frame_empty(unique_name, frame_id);
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
            first_transmission_sent = false; // Reset first transmission flag
            desc_sent = false;               // re-send the frame descriptor on the new connection
        }

        // Notify that connection is established
        connection_state_cv.notify_one();

        // wait for connection to get closed
        std::unique_lock<std::mutex> connection_lock(connection_state_mutex);
        connection_state_cv.wait(connection_lock, [&]() { return !connected || stop_thread; });
    }
}

void bufferSend::add_graph_details(kotekan::PipelineGraph& graph) const {
    // Node id is derived from the stage, not from the address, so that two stages
    // sending to the same host:port stay distinct nodes.
    const std::string dest_id = fmt::format("{:s}/destination", get_unique_name());
    graph.add_node(dest_id)
        .add_line(fmt::format("{:s}:{:d}", server_ip, server_port))
        .set_attr("shape", "doubleoctagon")
        .set_attr("style", "filled")
        .set_attr("color", "lightblue");
    graph.add_edge(get_unique_name(), dest_id);
}
