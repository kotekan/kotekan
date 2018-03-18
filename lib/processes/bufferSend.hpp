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

#pragma pack()
struct bufferFrameHeader {
    uint32_t metadata_size;
    uint32_t frame_size;
};

/**
 * @brief Transfers a buffer and metadata over TCP.
 *
 * @config send_timeout  Int, default 20. The number of seconds
 *                       before @c send() times out and closes the connection.
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
    bufferSend(Config &config,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    ~bufferSend();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);
private:
    struct Buffer *buf;
    uint32_t server_port;
    std::string server_ip;

    /// The number of seconds before send() times outs and returns and error.
    uint32_t send_timeout;

    /**
     * @brief Number of frame dropped because the send is too slow.
     * Only counts dropped data from caused by the send being too slow,
     * it does not include the number of frames dropped because the server is down.
     */
    uint64_t dropped_frame_count;

    std::atomic<bool> connected;
    struct sockaddr_in server_addr;
    int socket_fd;

    std::mutex connection_state_mutex;
    std::condition_variable connection_state_cv;

    void close_connection();

    void connect_to_server();

};

#endif