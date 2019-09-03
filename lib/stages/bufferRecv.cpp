#include "bufferRecv.hpp"

#include "bufferSend.hpp"
#include "nt_memcpy.h"
#include "prometheusMetrics.hpp"
#include "util.h"
#include "visUtil.hpp"

#include "fmt.hpp"

#include <cstring>
#include <errno.h>
#include <exception>
#include <functional>
#include <memory.h>
#include <signal.h>
#include <stdlib.h>
#include <string>
#include <sys/mman.h>

using namespace std::placeholders;
using std::mutex;
using std::queue;
using std::thread;
using std::vector;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(bufferRecv);

bufferRecv::bufferRecv(Config& config, const string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferRecv::main_thread, this)),
    dropped_frame_counter(
        Metrics::instance().add_counter("kotekan_buffer_recv_dropped_frame_total", unique_name)) {

    listen_port = config.get_default<uint32_t>(unique_name, "listen_port", 11024);
    num_threads = config.get_default<uint32_t>(unique_name, "num_threads", 1);
    connection_timeout = config.get_default<int>(unique_name, "connection_timeout", 60);

    buf = get_buffer("buf");
    register_producer(buf, unique_name.c_str());
}

bufferRecv::~bufferRecv() {}

void bufferRecv::read_callback(evutil_socket_t fd, short what, void* arg) {

    // Unused parameters required by libevent, suppress warnings.
    (void)fd;
    (void)what;

    // Add job to work queue.
    connInstance* instance = (connInstance*)arg;
    std::lock_guard<mutex> lock(instance->buffer_recv->work_queue_lock);
    std::deque<connInstance*>::iterator it =
        std::find(instance->buffer_recv->work_queue.begin(),
                  instance->buffer_recv->work_queue.end(), instance);
    if (it == instance->buffer_recv->work_queue.end()) {
        instance->increment_ref_count();
        instance->buffer_recv->work_queue.push_back(instance);
        instance->buffer_recv->work_cv.notify_all();
    }
}

void bufferRecv::worker_thread() {
    connInstance* instance;
    DEBUG2("Starting worker thread");
    while (!stop_thread) {
        {
            std::unique_lock<mutex> lock(work_queue_lock);
            work_cv.wait(lock, [&] { return (!work_queue.empty() || stop_thread); });
            if (stop_thread)
                return;
            instance = work_queue.front();
            work_queue.pop_front();
        }
        DEBUG2("Starting worker thread job, queue depth: {:d}", work_queue.size());
        instance->internal_read_callback();
    }
}

bool bufferRecv::get_worker_stop_thread() {
    return worker_stop_thread;
}

void bufferRecv::accept_connection(int listener, short event, void* arg) {
    struct acceptArgs* accept_args = (struct acceptArgs*)arg;
    accept_args->buffer_recv->internal_accept_connection(listener, event, arg);
}

void bufferRecv::increment_droped_frame_count() {
    std::lock_guard<mutex> lock(dropped_frame_count_mutex);
    dropped_frame_counter.inc();
}

void bufferRecv::internal_accept_connection(evutil_socket_t listener, short event, void* arg) {

    // Unused parameters required by libevent, suppress warnings.
    (void)listener;
    (void)event;

    DEBUG("Accept connection");
    struct acceptArgs* accept_args = (struct acceptArgs*)arg;
    struct event_base* base = accept_args->base;
    struct sockaddr_storage ss;
    socklen_t slen = sizeof(ss);
    int fd = accept(listener, (struct sockaddr*)&ss, &slen);
    if (fd < 0) {
        ERROR("Failed to accept connection, error {:d} ({:s}).", errno, std::strerror(errno));
        return;
    }
    if (fd > FD_SETSIZE) {
        ERROR("Got invalid FD");
        close(fd);
        return;
    }

    struct timeval read_timeout = {connection_timeout, 0};

    if (evutil_make_socket_nonblocking(fd) < 0) {
        ERROR("Could not make socket nonblocking");
        close(fd);
        return;
    }

    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (char*)&read_timeout, sizeof(read_timeout)) < 0) {
        ERROR("Could not set socket read timeout");
        close(fd);
        return;
    }

    // Get the client IP and port
    struct sockaddr_in* s = (struct sockaddr_in*)&ss;
    int port = ntohs(s->sin_port);
    char ip_str[256];
    inet_ntop(AF_INET, &s->sin_addr, ip_str, sizeof(ip_str));

    INFO("New connection from client: {:s}:{:d}", ip_str, port);

    // New connection instance
    connInstance* instance = new connInstance(accept_args->unique_name, accept_args->buf,
                                              accept_args->buffer_recv, ip_str, port, read_timeout);

    // Setup logging for the instance object.
    instance->set_log_prefix(accept_args->unique_name + "/instance");
    instance->set_log_level(accept_args->log_level);

    struct event* event_read =
        event_new(base, fd, EV_READ | EV_TIMEOUT, &bufferRecv::read_callback, (void*)instance);

    instance->event_read = event_read;
    instance->fd = fd;

    event_add(instance->event_read, &read_timeout);
}

void bufferRecv::timer(evutil_socket_t fd, short event, void* arg) {

    // Unused parameters required by libevent, suppress warnings.
    (void)fd;
    (void)event;

    bufferRecv* buff_recv = (bufferRecv*)arg;
    if (buff_recv->stop_thread) {
        event_base_loopbreak(buff_recv->base);
    }
}

void bufferRecv::main_thread() {

    evutil_socket_t listener;
    struct sockaddr_in server_addr;
    struct event* listener_event;

    INFO("libevent version: {:s}", event_get_version());

    if (evthread_use_pthreads()) {
        ERROR("Cannot use pthreads with libevent!");
        return;
    }

    base = event_base_new();
    if (!base) {
        FATAL_ERROR("Failed to create libevent base");
        return;
    }

    // Create worker threads:
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (auto& i : config.get<std::vector<int>>(unique_name, "cpu_affinity"))
        CPU_SET(i, &cpuset);

    for (uint32_t i = 0; i < num_threads; ++i) {
        thread_pool.push_back(thread(&bufferRecv::worker_thread, std::ref(*this)));
        pthread_setaffinity_np(thread_pool.back().native_handle(), sizeof(cpu_set_t), &cpuset);
#ifndef MAC_OSX
        std::string short_name =
            string_tail(fmt::format(fmt("{:s}/worker_thread/{:d}"), unique_name, i), 15);
        pthread_setname_np(thread_pool.back().native_handle(), short_name.c_str());
#endif
    }

    server_addr.sin_family = AF_INET;
    // Bind to every address (might want to change this later)
    server_addr.sin_addr.s_addr = 0;
    server_addr.sin_port = htons(listen_port);

    listener = socket(AF_INET, SOCK_STREAM, 0);
    evutil_make_socket_nonblocking(listener);

    if (bind(listener, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        FATAL_ERROR("Failed to bind to socket 0.0.0.0:{:d}, error: {:d} ({:s})", listen_port, errno,
                    strerror(errno));
        return;
    }

    if (listen(listener, 256) < 0) {
        FATAL_ERROR("Failed to open listener {:d} ({:s})", errno, strerror(errno));
        return;
    }

    struct acceptArgs args;
    args.base = base;
    args.buf = buf;
    args.buffer_recv = this;
    args.unique_name = unique_name;
    args.log_level = config.get_default<std::string>(
        unique_name, "instance_log_level", config.get<std::string>(unique_name, "log_level"));

    // Note that if the performance still isn't good enough, we could have more than
    // one base object and base loop threads.  In theory there shouldn't be much happening
    // in the base loop thread, but this needs to be tested more.
    listener_event = event_new(base, listener, EV_READ | EV_PERSIST, bufferRecv::accept_connection,
                               (void*)&args);
    event_add(listener_event, NULL);

    // Create a timer to check for the exit condition
    struct event* timer_event;
    timer_event = event_new(base, -1, EV_PERSIST, &bufferRecv::timer, this);
    struct timeval interval;
    interval.tv_sec = 0;
    interval.tv_usec = 100000;
    event_add(timer_event, &interval);

    // The libevent main loop.
    event_base_dispatch(base);

    // Join all the worker threads
    worker_stop_thread = true;
    work_cv.notify_all();
    for (thread& t : thread_pool) {
        if (t.joinable()) {
            t.join();
        }
    }
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

connInstance::connInstance(const string& producer_name, Buffer* buf, bufferRecv* buffer_recv,
                           const string& client_ip, int port, struct timeval read_timeout) :
    producer_name(producer_name),
    buf(buf),
    buffer_recv(buffer_recv),
    client_ip(client_ip),
    port(port),
    read_timeout(read_timeout) {

    frame_space = buffer_malloc(buf->aligned_frame_size);
    CHECK_MEM(frame_space);

    metadata_space = (uint8_t*)malloc(buf->metadata_pool->metadata_object_size);
    CHECK_MEM(metadata_space);
}

connInstance::~connInstance() {
    INFO("Closing FD");
    close(fd);
    event_free(event_read);
    buffer_free(frame_space);
    free(metadata_space);
}

void connInstance::increment_ref_count() {
    std::lock_guard<std::mutex> lock(reference_count_lock);
    reference_count++;
}

void connInstance::decrement_ref_count() {
    std::lock_guard<std::mutex> lock(reference_count_lock);
    reference_count--;
    DEBUG2("decrement reference_count {:d}, {:d}", reference_count, close_flag);
    if (reference_count == 0 && close_flag) {
        event_del(event_read);
        delete this;
    }
}

void connInstance::close_instance() {
    std::lock_guard<std::mutex> lock(reference_count_lock);
    event_del(event_read);
    close_flag = true;
    DEBUG2("close reference_count {:d}", reference_count);
    if (reference_count == 0) {
        delete this;
    }
}

void connInstance::internal_read_callback() {
    DEBUG2("Read Callback");

    // cache for metrics `kotekan_buffer_recv_transfer_time_seconds` because
    // prometheusMap requires a unique (metric_name, stage_name) key, and the
    // "producer_name" that we use as the "stage" comes from (potentially many)
    // external source(s), so there is no guarantee of its uniqueness.
    static std::mutex producer_transfer_time_map_lock;
    static std::unordered_map<
        std::string, std::shared_ptr<kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>>>
        producer_transfer_time_map;

    // Look up the metric for this instance's producer, or create a new one
    std::shared_ptr<kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>>
        transfer_time_seconds_metric;
    {
        std::lock_guard<std::mutex> lock(producer_transfer_time_map_lock);
        if (producer_transfer_time_map.count(producer_name) == 0) {
            auto& m = Metrics::instance().add_gauge("kotekan_buffer_recv_transfer_time_seconds",
                                                    producer_name, {"source"});
            producer_transfer_time_map[producer_name] =
                std::shared_ptr<kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>>(&m);
        }
        transfer_time_seconds_metric = producer_transfer_time_map.at(producer_name);
    }

    // Locking the instance should be equivalent to locking the bufferevent
    // since the callback which includes a given bev has a unique instance
    // attached to it.
    std::lock_guard<std::mutex> lock(instance_lock);

    // Don't process anything if the instance (and the connection attached to it) has been closed.
    if (close_flag) {
        decrement_ref_count();
        return;
    }

    ssize_t n = 0;

    DEBUG2("Read Callback");
    while (!buffer_recv->get_worker_stop_thread()) {
        switch (state) {
            case connState::header:
                start_time = current_time();

                n = read(fd, (void*)(((int8_t*)&buf_frame_header) + bytes_read),
                         sizeof(struct bufferFrameHeader) - bytes_read);
                if (n <= 0) {
                    handle_error("reading header", errno, n);
                    return;
                }
                DEBUG2("Header read bytes: {:d}", n);
                bytes_read += n;
                if (bytes_read >= sizeof(struct bufferFrameHeader)) {
                    assert(bytes_read == sizeof(struct bufferFrameHeader));
                    state = connState::metadata;
                    bytes_read = 0;

                    DEBUG2("Got header: metadata_size: {:d}, frame_size: {:d}",
                           buf_frame_header.metadata_size, buf_frame_header.frame_size);

                    if ((unsigned int)buf->frame_size != buf_frame_header.frame_size) {
                        ERROR("Frame size does not match between server: {:d} and client: {:d}",
                              buf->frame_size, buf_frame_header.frame_size);
                        decrement_ref_count();
                        close_instance();
                        return;
                    }
                    if (buf->metadata_pool->metadata_object_size
                        != buf_frame_header.metadata_size) {
                        ERROR("Metadata size does not match between server and client!");
                        decrement_ref_count();
                        close_instance();
                        return;
                    }
                }

                break;
            case connState::metadata:
                n = read(fd, (void*)(metadata_space + bytes_read),
                         buf_frame_header.metadata_size - bytes_read);
                if (n <= 0) {
                    handle_error("reading header", errno, n);
                    return;
                }
                DEBUG2("Metadata read bytes: {:d}", n);
                bytes_read += n;
                if (bytes_read >= buf_frame_header.metadata_size) {
                    assert(bytes_read == buf_frame_header.metadata_size);
                    state = connState::frame;
                    bytes_read = 0;
                }
                break;
            case connState::frame:
                n = read(fd, (void*)(frame_space + bytes_read),
                         buf_frame_header.frame_size - bytes_read);
                if (n <= 0) {
                    handle_error("reading header", errno, n);
                    return;
                }
                bytes_read += n;
                DEBUG2("Frame read bytes: {:d}, total read: {:d}", n, bytes_read);
                if (bytes_read >= buf_frame_header.frame_size) {
                    assert(bytes_read == buf_frame_header.frame_size);
                    state = connState::finished;
                    bytes_read = 0;
                }
                break;
            case connState::finished:
                throw std::runtime_error("State set to something unexpected!");
                break;
        }

        if (state == connState::finished) {
            // Get empty frame if one exists.
            DEBUG2("Finished state");
            int frame_id = buffer_recv->get_next_frame();
            if (frame_id == -1) {
                WARN("No free buffer frames, dropping data from {:s}", client_ip);

                // Update dropped frame count in prometheus
                buffer_recv->increment_droped_frame_count();
            } else {
                // This call cannot be blocking because we checked that
                // the frame is empty in get_next_frame()
                uint8_t* frame = wait_for_empty_frame(buf, producer_name.c_str(), frame_id);
                if (frame == NULL)
                    return;

                allocate_new_metadata_object(buf, frame_id);

                // Swap the frame pointers
                frame_space = swap_external_frame(buf, frame_id, frame_space);

                // We could also swap the metadata,
                // but this is more complex, and mucher lower overhead to just memcpy here.
                void* metadata = get_metadata(buf, frame_id);
                if (metadata != NULL)
                    memcpy(metadata, metadata_space, buf_frame_header.metadata_size);

                mark_frame_full(buf, producer_name.c_str(), frame_id);

                // Save a prometheus metric of the elapsed time
                double elapsed = current_time() - start_time;
                // TODO: having IP:port as the "source" label is a **bad**
                // Prometheus practice and of dubious usefulness
                std::string source_label = fmt::format(fmt("{:s}:{:d}"), client_ip, port);
                transfer_time_seconds_metric->labels({source_label}).set(elapsed);

                DEBUG("Received data from client: {:s}:{:d} into frame: {:s}[{:d}]", client_ip,
                      port, buf->buffer_name, frame_id);
            }
            state = connState::header;

            // After getting a frame we "yeld" so that we don't
            // starve other connections with data ready to go.
            decrement_ref_count();
            event_add(event_read, &read_timeout);
            return;
        }
    }
    decrement_ref_count();
}
