#ifndef BUFFER_RECV_H
#define BUFFER_RECV_H

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "bufferSend.hpp"
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

#include <event2/event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>

class bufferRecv : public KotekanProcess {
public:
    bufferRecv(Config &config,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    ~bufferRecv();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);

    // Returns a buffer ID of the next empty buffer, this must be filled
    // and returned promptly.
    int get_next_frame();
private:
    struct Buffer *buf;
    uint32_t listen_port;

    int current_frame_id = 0;
    std::mutex next_frame_lock;

    static void read_callback(struct bufferevent *bev, void *ctx);
    static void error_callback(struct bufferevent *bev, short error, void *ctx);
    static void accept_connection(evutil_socket_t listener, short event, void *arg);

    void internal_read_callback(struct bufferevent *bev, void *ctx);
    void internal_error_callback(struct bufferevent *bev, short error, void *ctx);
    void internal_accept_connection(evutil_socket_t listener, short event, void *arg);

    struct event_base *base;
    void base_thread();

    size_t dropped_frame_count = 0;

};

enum class connState {
    header, metadata, frame, finished
};

struct acceptArgs {
    struct event_base *base;
    struct Buffer *buf;
    bufferRecv * buffer_recv;
    string unique_name;
};

class connInstance {
public:
    connInstance(const string& producer_name,
                 struct Buffer *buf,
                 bufferRecv * buffer_recv,
                 const string &client_ip,
                 int port);
    ~connInstance();

    string producer_name;
    struct Buffer *buf;
    bufferRecv * buffer_recv;

    string client_name;
    string client_ip;
    int port;

    connState state = connState::header;
    size_t bytes_read = 0;

    struct bufferFrameHeader buf_frame_header;
    uint8_t * frame_space;
    uint8_t * metadata_space;
};

#endif