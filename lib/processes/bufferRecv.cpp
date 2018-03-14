#include "bufferRecv.hpp"
#include "util.h"
#include "bufferSend.hpp"
#include "prometheusMetrics.hpp"

#include <exception>
#include <errno.h>
#include <functional>

using namespace std::placeholders;

REGISTER_KOTEKAN_PROCESS(bufferRecv);

bufferRecv::bufferRecv(Config& config,
                        const string& unique_name,
                        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&bufferRecv::main_thread, this)) {

    listen_port = config.get_int_default(unique_name, "listen_port", 11024);

    buf = get_buffer("buf");
    register_producer(buf, unique_name.c_str());
}

bufferRecv::~bufferRecv() {
}

void bufferRecv::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

void bufferRecv::read_callback(bufferevent* bev, void* ctx) {
    connInstance * instance = (connInstance *) ctx;
    instance->buffer_recv->internal_read_callback(bev, ctx);
}

void bufferRecv::internal_read_callback(struct bufferevent *bev, void *ctx)
{
    DEBUG2("Read Callback");
    struct evbuffer *input;
    input = bufferevent_get_input(bev);
    connInstance * instance = (connInstance *) ctx;
    int64_t n = 0;

    while (evbuffer_get_length(input) && !instance->buffer_recv->stop_thread) {
        switch (instance->state) {
        case connState::header:
            n = evbuffer_remove(input,
                       (void*)(((int8_t*)&instance->buf_frame_header) + instance->bytes_read),
                       sizeof(struct bufferFrameHeader) - instance->bytes_read);
            DEBUG2("Header read bytes: %d", n);
            if (n < 0) {
                ERROR("Reading header failed for client %s, with error %d (%s)",
                        instance->client_ip.c_str(), errno, strerror(errno));
                goto end_loop;
            }
            instance->bytes_read += n;
            if (instance->bytes_read >= sizeof(struct bufferFrameHeader)) {
                assert(instance->bytes_read == sizeof(struct bufferFrameHeader));
                instance->state = connState::metadata;
                instance->bytes_read = 0;

                DEBUG2("Got header: metadata_size: %d, frame_size: %d",
                        instance->buf_frame_header.metadata_size, instance->buf_frame_header.frame_size);

                if ((unsigned int)instance->buf->frame_size != instance->buf_frame_header.frame_size) {
                    ERROR("Frame size does not match between server: %d and client: %d",
                            instance->buf->frame_size, instance->buf_frame_header.frame_size);
                    throw std::runtime_error("Frame size does not match between server and client!");
                }
                if (instance->buf->metadata_pool->metadata_object_size != instance->buf_frame_header.metadata_size) {
                    ERROR("Metadata size does not match between server and client!");
                    throw std::runtime_error("Metadata size does not match between server and client!");
                }
            }

            break;
        case connState::metadata:
            n = evbuffer_remove(input,
                    (void*)(instance->metadata_space + instance->bytes_read),
                    instance->buf_frame_header.metadata_size - instance->bytes_read);
            DEBUG2("Metadata read bytes: %d", n);
            if (n < 0) {
                ERROR("Reading metadata failed for client %s, with error %d (%s)",
                        instance->client_ip.c_str(), errno, strerror(errno));
                goto end_loop;
            }
            instance->bytes_read += n;
            if (instance->bytes_read >= instance->buf_frame_header.metadata_size) {
                assert(instance->bytes_read == instance->buf_frame_header.metadata_size);
                instance->state = connState::frame;
                instance->bytes_read = 0;
            }
        case connState::frame:
            n = evbuffer_remove(input,
                    (void*)(instance->frame_space + instance->bytes_read),
                    instance->buf_frame_header.frame_size - instance->bytes_read);
            if (n < 0) {
                ERROR("Reading frame failed for client %s, with error %d (%s)",
                        instance->client_ip.c_str(), errno, strerror(errno));
                goto end_loop;
            }
            instance->bytes_read += n;
            DEBUG("Frame read bytes: %d, total read: %d", n, instance->bytes_read);
            if (instance->bytes_read >= instance->buf_frame_header.frame_size) {
                assert(instance->bytes_read == instance->buf_frame_header.frame_size);
                instance->state = connState::finished;
                instance->bytes_read = 0;
            }
            break;
        case connState::finished:
            throw std::runtime_error("State set to something unexpected!");
            break;
        }

        if (instance->state == connState::finished) {
            // Get empty frame if one exists.
            DEBUG2("Finished state");
            int frame_id = instance->buffer_recv->get_next_frame();
            if (frame_id == -1) {
                WARN("No free buffer frames, dropping data from %s",
                        instance->client_ip.c_str());

                // Update dropped frame count in prometheus
                dropped_frame_count++;
                prometheusMetrics::instance().add_process_metric(
                    "kotekan_buffer_recv_dropped_frame_total", unique_name, dropped_frame_count
                );
            } else {
                // This call cannot be blocking because we checked that
                // the frame is empty in get_next_frame()
                uint8_t * frame = wait_for_empty_frame(instance->buf,
                                    instance->producer_name.c_str(), frame_id);
                if (frame == NULL) return;

                allocate_new_metadata_object(instance->buf, frame_id);

                memcpy((void*)frame, instance->frame_space,
                        instance->buf_frame_header.frame_size);

                void * metadata = get_metadata(instance->buf, frame_id);
                if (metadata != NULL)
                    memcpy(metadata, instance->metadata_space,
                            instance->buf_frame_header.metadata_size);

                mark_frame_full(instance->buf, instance->producer_name.c_str(), frame_id);
                INFO("Received data from client: %s:%d into frame: %s[%d]",
                        instance->client_ip.c_str(), instance->port,
                        instance->buf->buffer_name, frame_id);
            }
            instance->state = connState::header;
        }
    }
    end_loop:; // TODO Close?
}

void bufferRecv::error_callback(struct bufferevent *bev, short error, void *ctx)
{
    connInstance * instance = (connInstance *)ctx;
    instance->buffer_recv->internal_error_callback(bev, error, ctx);
}

void bufferRecv::internal_error_callback(bufferevent* bev, short error, void* ctx) {
    connInstance * instance = (connInstance *)ctx;
    DEBUG("Error Callback");
    if (error & BEV_EVENT_EOF) {
        WARN("Kotekan client: %s closed connection", instance->client_ip.c_str());
    } else if (error & BEV_EVENT_ERROR) {
        ERROR("Connection error with Kotekan client: %s, errno: %d, message %s",
                instance->client_ip.c_str(), errno, strerror(errno));
    } else if (error & BEV_EVENT_TIMEOUT) {
        ERROR("Connection with Kotekan client: %s timed out.", instance->client_ip.c_str());
    } else {
        ERROR("Un-handled error in error_callback %d", error);
    }
    delete instance;
    bufferevent_free(bev);
}

void bufferRecv::accept_connection(int listener, short event, void* arg) {
    struct acceptArgs * accept_args = (struct acceptArgs *)arg;
    accept_args->buffer_recv->internal_accept_connection(listener, event, arg);
}

void bufferRecv::internal_accept_connection(evutil_socket_t listener, short event, void *arg)
{
    //DEBUG("Accept connection");
    struct acceptArgs * accept_args = (struct acceptArgs *)arg;
    struct event_base *base = accept_args->base;
    struct sockaddr_storage ss;
    socklen_t slen = sizeof(ss);
    int fd = accept(listener, (struct sockaddr*)&ss, &slen);
    if (fd < 0) {
        //ERROR("Failed to accept connection.");
    } else if (fd > FD_SETSIZE) {
        close(fd);
    } else {
        struct bufferevent *bev;
        evutil_make_socket_nonblocking(fd);

        // Get the client IP and port
        struct sockaddr_in *s = (struct sockaddr_in *)&ss;
        int port = ntohs(s->sin_port);
        char ip_str[256];
        inet_ntop(AF_INET, &s->sin_addr, ip_str, sizeof(ip_str));

        //INFO("New connection from client: %s:%d", ip_str, port);

        // New connection instance
        connInstance * instance =
                new connInstance(accept_args->unique_name,
                                 accept_args->buf,
                                 accept_args->buffer_recv,
                                 ip_str,
                                 port);

        bev = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE);
        bufferevent_setcb(bev, &bufferRecv::read_callback, NULL,
                               &bufferRecv::error_callback, (void *)instance);
        size_t expected_size = sizeof(struct bufferFrameHeader) +
                                accept_args->buf->metadata_pool->metadata_object_size +
                                accept_args->buf->frame_size;
        bufferevent_setwatermark(bev, EV_READ, expected_size, 0);
        const int timeout_sec = 60;
        struct timeval read_timeout = {timeout_sec, 0};
        struct timeval write_timeout = {timeout_sec, 0};
        bufferevent_set_timeouts(bev, &read_timeout, &write_timeout);
        bufferevent_enable(bev, EV_READ|EV_WRITE);
    }
}

void bufferRecv::base_thread() {
    event_base_dispatch(base);
}

void bufferRecv::main_thread() {

    evutil_socket_t listener;
    struct sockaddr_in server_addr;
    struct event *listener_event;

    base = event_base_new();
    if (!base)
        throw std::runtime_error("Failed to crate libevent base");

    server_addr.sin_family = AF_INET;
    // Bind to every address (might want to change this later)
    server_addr.sin_addr.s_addr = 0;
    server_addr.sin_port = htons(listen_port);

    listener = socket(AF_INET, SOCK_STREAM, 0);
    evutil_make_socket_nonblocking(listener);

    if (bind(listener, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        ERROR("Failed to bind to socket 0.0.0.0:%d, error: %d (%s)",
                listen_port, errno, strerror(errno));
        return;
    }

    if (listen(listener, 128)<0) {
        ERROR("Failed to open listener %d (%s)", errno, strerror(errno));
        return;
    }

    struct acceptArgs args;
    args.base = base;
    args.buf = buf;
    args.buffer_recv = this;
    args.unique_name = unique_name;
    listener_event = event_new(base, listener, EV_READ|EV_PERSIST,
                        bufferRecv::accept_connection, (void*)&args);
    event_add(listener_event, NULL);

    std::thread base_thread_t =
            std::thread(&bufferRecv::base_thread, std::ref(*this));

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (auto &i : config.get_int_array(unique_name, "cpu_affinity"))
        CPU_SET(i, &cpuset);
    pthread_setaffinity_np(base_thread_t.native_handle(), sizeof(cpu_set_t), &cpuset);

    while (!stop_thread) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    INFO("Event base loop break");
    event_base_loopbreak(base);

    base_thread_t.join();
    // TODO Cleanup connInstances
}

int bufferRecv::get_next_frame() {
    std::lock_guard<std::mutex> lock(next_frame_lock);

    // If the frame is full for some reason (items not being consumed fast enough)
    // Then return -1;
    if (is_frame_empty(buf, current_frame_id) == 0) {
        return -1;
    }

    int last_frame_id = current_frame_id;
    current_frame_id = (current_frame_id + 1) % buf->num_frames;

    return last_frame_id;
}

connInstance::connInstance(const string& producer_name,
                           Buffer* buf,
                           bufferRecv* buffer_recv,
                           const string& client_ip,
                           int port) :
                           producer_name(producer_name),
                           buf(buf),
                           buffer_recv(buffer_recv),
                           client_ip(client_ip),
                           port(port)
                    {
    frame_space = (uint8_t *)malloc(buf->frame_size);
    CHECK_MEM(frame_space);
    metadata_space = (uint8_t *)malloc(buf->metadata_pool->metadata_object_size);
    CHECK_MEM(metadata_space);
}

connInstance::~connInstance() {
    free(frame_space);
    free(metadata_space);
}

