#include "bufferSend.hpp"
#include "util.h"
#include "chimeMetadata.h"

bufferSend::bufferSend(Config& config,
                        const string& unique_name,
                        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&bufferSend::main_thread, this)) {

    buf = get_buffer("buf");
    register_consumer(buf, unique_name.c_str());

    connected = false;
    server_ip = config.get_string(unique_name, "server_ip");
    server_port = config.get_int_default(unique_name, "server_port", 11024);

    integration_len = config.get_int(unique_name, "samples_per_data_set") *
                        config.get_int(unique_name, "num_gpu_frames");

    bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(server_ip.c_str());
    server_addr.sin_port = htons(server_port);
}

bufferSend::~bufferSend() {
}

void bufferSend::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

void bufferSend::main_thread() {

    int frame_id = 0;

    std::thread connect_thread = std::thread(&bufferSend::connect_to_server, std::ref(*this));

    while (!stop_thread) {

        uint8_t * frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        if (connected) {
            // Send header
            // Note this only works with CHIME metadata for now, but
            // it could be extended to work with any metadata providing
            // some kind of stream ID and seq number.
            struct bufferFrameHeader header;
            const size_t header_len = sizeof(struct bufferFrameHeader);
            int32_t n = 0;
            int32_t n_sent = 0;

            //header.stream_id = get_stream_id(buf, frame_id);
            //header.seq_number = get_fpga_seq_num(buf, frame_id) / integration_len;
            header.frame_size = buf->frame_size;
            header.metadata_size = buf->metadata[frame_id]->metadata_size;

            //INFO("Header: stream_id: %d, seq_number: %d, frame_size: %d, metadata_size: %d",
            //        header.stream_id, header.seq_number, header.frame_size, header.metadata_size);

            // Recover from partial sends
            //DEBUG("Sending header");
            while ((n = send(socket_fd, &((uint8_t*)&header)[n_sent],
                                header_len - n_sent, 0)) > 0) {
                n_sent += n;
            }
            // Handle errors
            if (n < 0) {
                ERROR("Error failed to send header to %s:%d", server_ip.c_str(), server_port);
                close_connection();
                continue;
            }
            //DEBUG("Sent header: %d", n_sent);

            // Send metadata
            //DEBUG("Sending metadata");
            n_sent = 0;
            while ((n = send(socket_fd,
                                &((uint8_t*)buf->metadata[frame_id]->metadata)[n_sent],
                                header.metadata_size - n_sent, 0)) > 0) {
                n_sent += n;
            }
            if (n < 0) {
                ERROR("Error failed to metadata to %s:%d", server_ip.c_str(), server_port);
                close_connection();
                continue;
            }
            //DEBUG("Sent metadata: %d", n_sent);

            // Send buffer frame.
            //DEBUG("Sending frame with %d bytes", header.frame_size);
            n_sent = 0;
            while ((n = send(socket_fd, &frame[n_sent],
                                (int32_t)header.frame_size - n_sent, 0)) > 0) {
                n_sent += n;
                //DEBUG("Total sent: %d", n_sent);
            }
            if (n < 0) {
                ERROR("Error failed to frame data to %s:%d", server_ip.c_str(), server_port);
                close_connection();
                continue;
            }
            //DEBUG("Sent frame: %d", n_sent);
            INFO("Sent frame: %s[%d] to %s:%d", buf->buffer_name, frame_id, server_ip.c_str(), server_port);

        } else {
            WARN("Dropping frame %s[%d], because connection to %s:%d is down.",
                  buf->buffer_name, frame_id, server_ip.c_str(), server_port);
        }

        mark_frame_empty(buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
    }

    // Stop thread will be set to true at this point.
    connection_state_cv.notify_all();
    connect_thread.join();
}

void bufferSend::close_connection() {
    close(socket_fd);
    socket_fd = 0;
    {
        std::unique_lock<std::mutex> connection_lock(connection_state_mutex);
        connected = false;
    }
    connection_state_cv.notify_all();
}

void bufferSend::connect_to_server() {

    while (!stop_thread) {

        INFO("Trying to connecting to server: %s:%d", server_ip.c_str(), server_port);

        socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd == -1) {
            ERROR("Could not create socket, errno: %d", errno);
            throw std::runtime_error("Could not create socket");
        }

        if (connect(socket_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
            ERROR("Could not connect to server %s:%d, error: %d, waiting 5 seconds to retry...", server_ip.c_str(), server_port, errno);
            sleep(5);
            continue;
        }

        INFO("Connected to server %s:%d for sending buffer %s",
                server_ip.c_str(), server_port, buf->buffer_name);
        {
            std::unique_lock<std::mutex> connection_lock(connection_state_mutex);
            connected = true;
        }

        std::unique_lock<std::mutex> connection_lock(connection_state_mutex);
        connection_state_cv.wait(connection_lock, [&](){return !connected || stop_thread;});
    }
}
