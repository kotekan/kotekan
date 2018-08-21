/**
 * @file
 * @brief Object for receiving buffer frames from another kotekan instance
 * - bufferRecv : public KotekanProcess
 * - connState
 * - acceptArgs
 * - connInstance : public kotekanLogging
 */
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
#include <queue>
#include <functional>

#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

#include <event2/event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/thread.h>


/**
 * @brief Receives frames and metadata from other networked kotekan buffers,
 *        and puts them into a local kotekan buffer.
 *
 * This process takes frames from one more more sources and places them into the
 * @c buf for use by other local consumer processes. There is no guarantee regarding
 * the order the frames will arrvive in.  However all frames will be valid, which is to
 * say they will contain the full set of data sent by the client, or they will not be
 * added to the output buffer.
 *
 * This system works with libevent to do event driven async I/O with worker threads to support
 * higher bandwidth than one thread alone could support.  In libevent terms there is one base
 * thread, and @c num_threads worker threads which handle the libevent callbacks.
 *
 * @bar buffers
 * @buffer buf The buffer which accepts new frames (producer)
 *        @buffer_format any
 *        @buffer_metadata any
 *
 * @config listen_port         Int, default 11024.  The port to listen for new connections
 * @config num_threads         Int, default 1.  The number of worker threads to use
 * @config connection_timeout  Int, default 60.  Number of seconds before timeout on transfer
 *
 * @todo Possibly factor out the threadpool.
 * @todo Allow for a different log level for workers from the main thread.
 *
 * @author Andre Renard
 */
class bufferRecv : public KotekanProcess {
public:

    /// Constructor
    bufferRecv(Config &config,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    ~bufferRecv();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);

    /**
     * @brief Returns a buffer ID of the next empty buffer, this must be filled
     *        and returned promptly.  Used internally by worker threads.
     *
     * @return A buffer ID that no one else is using, but which must be filled and returned.
     */
    int get_next_frame();

    /**
     * @brief Increases the dropped frame count by one, and updates the
     *        prometheus metric.  Thread safe.  Called only by worker threads
     */
    void increment_droped_frame_count();

    /**
     * @brief Used only by worker threads to check if they should stop.
     * @return True if they should stop, false otherwise.
     */
    bool get_worker_stop_thread();

private:

    /// The output buffer
    struct Buffer *buf;

    /// The port to listen for new connections on
    uint32_t listen_port;

    /// How long to wait before timing out a transfer
    int connection_timeout;

    /// The current frame to use next
    int current_frame_id = 0;

    /// A lock on the current frame, since many systems may ask for the next frame
    std::mutex next_frame_lock;

    static void read_callback(struct bufferevent *bev, void *ctx);
    static void error_callback(struct bufferevent *bev, short error, void *ctx);
    static void accept_connection(evutil_socket_t listener, short event, void *arg);

    /**
     * @brief Internal timer call back to check for thread exit condition
     *
     * @param fd Not used
     * @param event Not used
     * @param arg The bufferRecv object (just `this`, but this is a static function)
     */
    static void timer(evutil_socket_t fd, short event, void *arg);

    /**
     * @brief Callback for processing new connections.
     *
     * Sets up the new connection @c connInstance object and creates
     * the libevent bufferevent objects to go along with it.
     *
     * @param listener The listener socket
     * @param event The event type
     * @param arg Values needed to setup the new connection in an @c acceptArgs struct
     */
    void internal_accept_connection(evutil_socket_t listener, short event, void *arg);

    /// The base event for libevent, run in the main_thread.
    /// This might be increased to more than one if there are performance issues.
    struct event_base *base;

    /// The number of frames dropped
    size_t dropped_frame_count = 0;

    /// A lock on the @c dropped_frame_count counter
    std::mutex dropped_frame_count_mutex;

    // Worker threads (thread pool section)

    /// The number of worker threads to spawn
    uint32_t num_threads;

    /// The pool of worker threads
    std::vector<std::thread> thread_pool;

    /// Queue of functions to be called by the worker threads
    std::queue<std::function<void(void)>> work_queue;

    /// Lock for the work queue
    std::mutex work_queue_lock;

    /// Condition variable for the state (empty or not) of the work queue
    std::condition_variable work_cv;

    /// Set to true to stop the worker threads
    bool worker_stop_thread = false;

    /**
     * @brief The worker thread for handing read callbacks.
     */
    void worker_thread();

};

/**
 * @brief List of valid states for a connection to be in.
 */
enum class connState {
    header, metadata, frame, finished
};

/**
 * @brief Args passed to the accept new connection call back function
 */
struct acceptArgs {
    /// The libevent base to use.
    struct event_base *base;

    /// The output buffer to attach to
    struct Buffer *buf;

    /// The main buffer recv object needs to be called to manage
    /// the frame ID to use (corrdinate between workers)
    bufferRecv * buffer_recv;

    /// Just copy the unique_name of the process
    string unique_name;

    /// The log level to use.
    string log_level;
};

/**
 * @brief Object to store all the details of a connection to a remote
 *        kotekan buffer.
 *
 * Should not be used externally to the @c bufferRecv class.  This is an
 * internal helper class.
 *
 * @author Andre Renard
 */
class connInstance : public kotekanLogging {
public:
    connInstance(const string& producer_name,
                 struct Buffer *buf,
                 bufferRecv * buffer_recv,
                 const string &client_ip,
                 int port);
    ~connInstance();

    void internal_read_callback(struct bufferevent *bev);
    void internal_error_callback(struct bufferevent *bev, short error, void *ctx);

    void increment_ref_count();
    void decrement_ref_count();
    void close_instance();

    void set_bufferevent(struct bufferevent *bev);

    string producer_name;
    struct Buffer *buf;
    bufferRecv * buffer_recv;

    string client_name;
    string client_ip;
    int port;

    struct bufferevent *buffer_event;


    size_t bytes_read = 0;

    double start_time;

    struct bufferFrameHeader buf_frame_header;
    uint8_t * frame_space;
    uint8_t * metadata_space;

    std::mutex instance_lock;

    /// The number of active jobs which hold a pointer to this
    /// object.  Note this doesn't include the libevent pointers.
    uint32_t reference_count = 0;

    /// Set to true if we've encountered an error and need to close
    /// the connection attached to this instance.
    bool close_flag = false;

    std::mutex reference_count_lock;

    connState state = connState::header;
};

#endif