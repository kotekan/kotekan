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

    int32_t integration_len;

    std::atomic<bool> connected;
    struct sockaddr_in server_addr;
    int socket_fd;

    std::mutex connection_state_mutex;
    std::condition_variable connection_state_cv;

    void close_connection();

    void connect_to_server();

};

#endif