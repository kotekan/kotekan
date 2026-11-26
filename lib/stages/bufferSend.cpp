#include "bufferSend.hpp"

#include "Config.hpp"            // for Config
#include "PipelineGraph.hpp"     // for PipelineGraph, GraphNode
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "configTracker.hpp"     // for ConfigTracker
#include "kotekanLogging.hpp"    // for DEBUG2, ERROR, FATAL_ERROR, INFO, DEBUG, WARN
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
#include <unistd.h>     // for close

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
                       bufferContainer& buffer_container, std::string buffer_name) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferSend::main_thread, this)),
    dest(config.get<std::string>(unique_name, "server_ip"),
         config.get_default<uint16_t>(unique_name, "server_port", 11024),
         config.get_default<uint32_t>(unique_name, "reconnect_time", 5),
         config.get_default<uint32_t>(unique_name, "send_timeout", 20)),
    dropped_frame_counter(
        Metrics::instance().add_counter("kotekan_buffer_send_dropped_frame_count", unique_name)) {
    buf = get_buffer(buffer_name);
    // Every frame is sent with its metadata object; bufferRecv cannot accept a frame without one.
    if (!buf->metadata_pool) {
        FATAL_ERROR("Buffer {:s} has no metadata_pool: every frame sent needs a metadata object",
                    buf->buffer_name);
    }
    buf->register_consumer(unique_name);

    drop_frames = config.get_default<bool>(unique_name, "drop_frames", true);
    drop_threshold = config.get_default<float>(unique_name, "drop_threshold", 0.6);

    // Publish current dropped frame count.

    // Default to the tracker's enabled state (set at startup, before stages).
    // A stage may opt out, but cannot opt in when the tracker is off, so the
    // configured value is ANDed with the tracker state.
    const bool ct_enabled = ConfigTracker::instance().is_enabled();
    use_config_tracker =
        config.get_default<bool>(unique_name, "use_config_tracker", ct_enabled) && ct_enabled;

    use_frame_desc = config.get_default<bool>(unique_name, "use_frame_desc", false);
}

bufferSend::~bufferSend() {}

void bufferSend::main_thread() {

    int frame_id = 0;

    dest.start_connect_thread(unique_name, stop_thread);

    while (!stop_thread) {

        uint8_t* frame = buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;
        if (!got_frame(frame, frame_id))
            break;

        uint32_t num_full_frames = buf->get_num_full_frames();

        if (drop_frames && (float)num_full_frames / (float)buf->num_frames > drop_threshold) {
            // If the number of full frames is high, then we drop some frames,
            // because we likely aren't sending fast enough to up with the data rate.
            INFO("Number of full frames in buffer {:s} is {:d} (total frames: {:d}), dropping "
                 "frame_id {:d}",
                 buf->buffer_name, num_full_frames, buf->num_frames, frame_id);
            dropped_frame_counter.inc();
        } else if (drop_frames && !dest.connected) {
            INFO("Dropping frame {:s}[{:d}], because connection to {:s}:{:d} is down.",
                 buf->buffer_name, frame_id, dest.server_ip, dest.server_port);
            dropped_frame_counter.inc();
        } else if (dest.connected) {
            if (!send_frame(frame, frame_id, dest)) {
                dest.close_connection();
                continue;
            }
            DEBUG("Sent frame: {:s}[{:d}] to {:s}:{:d}", buf->buffer_name, frame_id, dest.server_ip,
                  dest.server_port);
        } else {
            // Wait for connection and block
            INFO("Waiting for connection to {:s}:{:d}...", dest.server_ip, dest.server_port);
            std::unique_lock<std::mutex> connection_lock(dest.connection_state_mutex);
            dest.connection_state_cv.wait_for(connection_lock, std::chrono::seconds(1),
                                              [&]() { return (stop_thread || dest.connected); });
            continue;
        }

        done_with_frame(frame_id);
        buf->mark_frame_empty(unique_name, frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
    }

    dest.close_connection();
    dest.connect_thread.join();
}

bool bufferSend::send_frame(uint8_t* frame, int frame_id, networkDestination& dest) {
    auto meta = buf->get_metadata(frame_id);
    if (!meta) {
        // FATAL_ERROR throws out of this function, so end the connect thread first.
        stop();
        dest.close_connection();
        dest.connect_thread.join();
        FATAL_ERROR("Frame {:s}[{:d}] has no metadata object: its producer did not "
                    "allocate one",
                    buf->buffer_name, frame_id);
    }
    auto metadata_size = meta->get_serialized_size();
    auto frame_size = buf->frame_size;
    bufferFrameHeader header;
    header.metadata_size = static_cast<uint32_t>(meta->get_serialized_size());
    header.frame_size = static_cast<uint32_t>(buf->frame_size);
    // Flag a tracker update on the first frame of every connection, not
    // only when this side's tracker hash changes: a receiver that
    // (re)started after us holds no upstream configs and has no way to
    // ask for them, so the first frame it sees must carry the flag.
    header.config_tracker_update =
        !dest.first_transmission_sent
        || dest.config_tracker_combined_hash != ConfigTracker::instance().getTrackerHash();
    // for legacy CHIME, do not send last field (config_tracker_update)
    size_t header_len =
        use_config_tracker ? sizeof(bufferFrameHeader) : sizeof(bufferFrameHeaderNoConfigTracker);

    if (header.config_tracker_update)
        DEBUG("Config tracker data has been updated, sending new config tracker data.");
    DEBUG2("frame_size: {:d}, metadata_size: {:d}, config_tracker_update: {:d}", header.frame_size,
           header.metadata_size, header.config_tracker_update);

    // Recover from partial sends
    DEBUG2("Sending header");
    if (!dest.send_data((uint8_t*)&header, header_len)) {
        ERROR("Error {:s}, failed to send header to {:s}:{:d}", strerror(errno), dest.server_ip,
              dest.server_port);
        return false;
    }
    DEBUG("Sent frame header: {:s}[{:d}] to {:s}:{:d}", buf->buffer_name, frame_id, dest.server_ip,
          dest.server_port);

    // If the frame sent successfully,
    if (dest.config_tracker_combined_hash != ConfigTracker::instance().getTrackerHash()) {
        dest.config_tracker_combined_hash = ConfigTracker::instance().getTrackerHash();
    }
    DEBUG2("Sent header: {:d}", header_len);

    // Send the frame descriptor once per connection (independent of the
    // config tracker), right after the first frame's header: a 4-byte
    // JSON length followed by that many bytes. Later frames send neither.
    if (use_frame_desc && !dest.desc_sent) {
        auto desc = buf->get_frame_desc();
        const std::string desc_json = desc ? desc->to_json().dump() : std::string();
        uint32_t frame_desc_size = static_cast<uint32_t>(desc_json.size());
        if (!dest.send_data((uint8_t*)&frame_desc_size, sizeof(frame_desc_size))) {
            ERROR("Error {:s}, failed to send frame_desc size to {:s}:{:d}", strerror(errno),
                  dest.server_ip, dest.server_port);
            dest.close_connection();
            return false;
        }
        if (frame_desc_size > 0) {
            if (!dest.send_data((uint8_t*)desc_json.data(), frame_desc_size)) {
                ERROR("Error {:s}, failed to send frame_desc to {:s}:{:d}", strerror(errno),
                      dest.server_ip, dest.server_port);
                dest.close_connection();
                return false;
            }
        }
        dest.desc_sent = true;
    }

    // Send metadata
    DEBUG2("Sending metadata");
    {
        char metabuf[metadata_size];
        meta->serialize(metabuf);
        if (!dest.send_data((uint8_t*)metabuf, metadata_size)) {
            ERROR("Error {:s}, failed to send metadata to {:s}:{:d}", strerror(errno),
                  dest.server_ip, dest.server_port);
            return false;
        }
    }
    DEBUG2("Sent metadata: {:d}", metadata_size);

    // Send buffer frame.
    DEBUG2("Sending frame with {:d} bytes", frame_size);
    if (!dest.send_data(frame, frame_size)) {
        ERROR("Error {:s}, failed to frame data to {:s}:{:d}", strerror(errno), dest.server_ip,
              dest.server_port);
        return false;
    }
    DEBUG2("Sent frame: {:d}", frame_size);

    // Set to true since we've sent a transmission.
    dest.first_transmission_sent = true;

    return true;
}

networkDestination::networkDestination(const std::string& ip, uint16_t port, uint32_t reconnect_t,
                                       uint32_t send_t) :
    socket_fd(-1), server_port(port), server_ip(ip), connected(false), reconnect_time(reconnect_t),
    send_timeout(send_t), first_transmission_sent(false), desc_sent(false),
    config_tracker_combined_hash("") {
    bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(server_ip.c_str());
    server_addr.sin_port = htons(server_port);
}

bool networkDestination::send_data(const uint8_t* data, int32_t len) {
    int32_t n = 0;
    int32_t n_sent = 0;
    while ((n = send(socket_fd, &data[n_sent], len - n_sent, MSG_NOSIGNAL)) > 0) {
        n_sent += n;
    }
    return (n >= 0);
}

void networkDestination::start_connect_thread(const std::string& name,
                                              const std::atomic_bool& stop_thread) {
    connect_thread = std::thread(&networkDestination::connect_to_server, this, std::ref(name),
                                 std::ref(stop_thread));
}

void networkDestination::close_connection() {
    if (socket_fd >= 0)
        close(socket_fd);

    socket_fd = -1;
    {
        std::unique_lock<std::mutex> connection_lock(connection_state_mutex);
        connected = false;
    }
    connection_state_cv.notify_all();
}

void networkDestination::connect_to_server(const std::string& name,
                                           const std::atomic_bool& stop_thread) {

    while (!stop_thread) {

        DEBUG("Trying to connecting to server: {:s}:{:d}", server_ip, server_port);

        socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd == -1) {
            std::string msg = fmt::format(fmt("Could not create socket, errno: {:d} ({:s})"), errno,
                                          std::strerror(errno));
            ERROR("{:s}", msg);
            throw std::runtime_error(msg);
        }

        if (connect(socket_fd, (struct sockaddr*)&(server_addr), sizeof(server_addr)) == -1) {
            WARN("Could not connect to server {:s}:{:d}, error: {:s}({:d}), waiting {:d} seconds "
                 "to retry...",
                 server_ip, server_port, strerror(errno), errno, reconnect_time);
            close(socket_fd);
            socket_fd = -1;
            // A stop ends this wait early: main_thread() calls close_connection(), which
            // notifies, before it joins this thread.
            {
                std::unique_lock<std::mutex> connection_lock(connection_state_mutex);
                connection_state_cv.wait_for(connection_lock, std::chrono::seconds(reconnect_time),
                                             [&]() { return (bool)stop_thread; });
            }
            continue;
        }

        // Prevent SIGPIPE on send failure.
        // This is used for MacOS, since linux doesn't have SO_NOSIGPIPE
#ifdef SO_NOSIGPIPE
        int set = 1;
        if (setsockopt(socket_fd, SOL_SOCKET, SO_NOSIGPIPE, (void*)&set, sizeof(int)) < 0) {
            ERROR("{:s}: setsockopt() NOSIGPIPE ", name);
        }
#endif

        // Set send timeout.
        struct timeval tv_timeout;
        tv_timeout.tv_sec = send_timeout;
        tv_timeout.tv_usec = 0;

        if (setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO, (void*)&tv_timeout, sizeof(tv_timeout))
            < 0) {
            ERROR("{:s}: setsockopt() timeout failed.", name);
        }

        INFO("{:s}: Connected to server {:s}:{:d}", name, server_ip, server_port);
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

/**
   std::string bufferSend::dot_string(const std::string& prefix) const {
   std::string dot = Stage::dot_string(prefix);
   std::string target = fmt::format("{:s}:{:d}", dest.server_ip, dest.server_port);
   dot += fmt::format("{:s}\"{:s}\" [shape=doubleoctagon style=filled,color=lightblue]", prefix,
   target);
   dot += fmt::format("{:s}\"{:s}\" -> \"{:s}\"", prefix, get_unique_name(), target);
   return dot;
*/

void bufferSend::add_graph_details(kotekan::PipelineGraph& graph) const {
    // Node id is derived from the stage, not from the address, so that two stages
    // sending to the same host:port stay distinct nodes.
    const std::string dest_id = fmt::format("{:s}/destination", get_unique_name());
    graph.add_node(dest_id)
        .add_line(fmt::format(fmt("{:s}:{:d}"), dest.server_ip, dest.server_port))
        .set_category(kotekan::GraphCategory::Endpoint);
    graph.add_edge(get_unique_name(), dest_id);
}

bool bufferSend::got_frame(uint8_t*, int) {
    return true;
}

void bufferSend::done_with_frame(int) {}
