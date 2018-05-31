/**
 * @file
 * @brief Object for sending buffer frames to another kotekan instance
 * - bufferSend : public KotekanProcess
 */
#ifndef BUFFER_SEND_H
#define BUFFER_SEND_H

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include <unistd.h>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

/**
 * @struct bufferFrameHeader
 * @brief Internal struct for sending the transfer details.
 */
#pragma pack()
struct bufferFrameHeader {
    uint32_t metadata_size;
    uint32_t frame_size;
};

/**
 * @brief Sends a buffer and metadata over TCP.
 *
 * Will attempt to connect to a remote server (likely another kotekan instance)
 * and send frames and metadata as they arrive.
 *
 * If the remote server is down, or the connection breaks, this process will
 * drop incoming frames, and try to reconnect to the server after @c reconnect_time
 * seconds.
 *
 *
 *
 * @par buffers
 * @buffer buf The buffer to send to the remote server.
 *        @buffer_format any
 *        @buffer_metadata any
 *
 * @config server_ip       String, the IP address of the server to send data too.
 * @config server_port     Int, default 11024. The port number on the remote server.
 * @config send_timeout    Int, default 20. The number of seconds
 *                         before @c send() times out and closes the connection.
 * @config reconnect_time  Int, default 5.  The number of seconds between
 *                         connection attempts to the remote server.
 *
 * @par Metrics
 * @metric kotekan_buffer_send_dropped_frame_count
 *         The number of frames dropped because @c send() is running too slow.
 *
 * @todo Add the rest of the comments here.
 * @todo we might also add counters for dropped frames because the connection
 *       is down, and frames that are lost because of connection errors.
 *
 * @author Andre Renard
 */
class bufferSend : public KotekanProcess {
public:
    /// Standard constructor
    bufferSend(Config &config,
                  const string& unique_name,
                  bufferContainer &buffer_container);

    /// Destructor
    ~bufferSend();

    /// Main loop for sending data
    void main_thread();

    /// Deprecated
    virtual void apply_config(uint64_t fpga_seq);
private:
    /// The input buffer to send frames from.
    struct Buffer *buf;

    /// The server port to connect to.
    uint32_t server_port;

    /// The server IP address to connect to.
    std::string server_ip;

    /// The number of seconds before send() times outs and returns and error.
    uint32_t send_timeout;

    /// The number of seconds between connection attempts
    uint32_t reconnect_time;

    /**
     * @brief Number of frame dropped because the send is too slow.
     * Only counts dropped data from caused by the send being too slow,
     * it does not include the number of frames dropped because the server is down.
     */
    uint64_t dropped_frame_count;

    /// Set to true if there is an active connection
    std::atomic<bool> connected;

    /// Internal server address struct
    struct sockaddr_in server_addr;

    /// The connection file handle
    int socket_fd;

    /// Prevent the sending thread and connection thread from contension
    std::mutex connection_state_mutex;

    /// Used to wakeup the connect thread after a change to the connection state
    std::condition_variable connection_state_cv;

    /// Closes the open connection and starts the process of trying to reconnect
    void close_connection();

    /// Thread for connecting to the remote server
    void connect_to_server();

};

#endif