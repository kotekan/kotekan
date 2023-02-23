/**
 * @file
 * @brief Object for receiving buffer frames from another kotekan instance
 * - bufferRecv : public kotekan::Stage
 * - connState
 * - acceptArgs
 * - connInstance : public kotekanLogging
 */
#ifndef BUFFER_RECV_H
#define BUFFER_RECV_H

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "bufferContainer.hpp"   // for bufferContainer
#include "bufferSend.hpp"        // for bufferFrameHeader
#include "kotekanLogging.hpp"    // for DEBUG2, ERROR, INFO, kotekanLogging
#include "prometheusMetrics.hpp" // for Counter, Gauge, MetricFamily

#include <condition_variable> // for condition_variable
#include <deque>              // for deque
#include <event2/event.h>     // for event_add
#include <event2/util.h>      // for evutil_socket_t
#include <mutex>              // for mutex
#include <stdint.h>           // for uint32_t, uint8_t
#include <stdio.h>            // for size_t
#include <string.h>           // for strerror
#include <string>             // for string
#include <sys/time.h>         // for timeval
#include <thread>             // for thread
#include <unistd.h>           // for ssize_t
#include <vector>             // for vector

// Forward declare
class connInstance;

/**
 * @brief Receives frames and metadata from other networked kotekan buffers,
 *        and puts them into a local kotekan buffer.
 *
 * This stage takes frames from one more more sources and places them into the
 * @c buf for use by other local consumer stages. There is no guarantee regarding
 * the order the frames will arrive in.  However all frames will be valid, which is to
 * say they will contain the full set of data sent by the client, or they will not be
 * added to the output buffer.
 *
 * This system works with libevent to do event driven async I/O with worker threads to support
 * higher bandwidth than one thread alone could support.  In libevent terms there is one base
 * thread, and @c num_threads worker threads which handle the libevent callbacks.
 *
 * @par buffers
 * @buffer buf The buffer which accepts new frames (producer)
 *        @buffer_format any
 *        @buffer_metadata any
 *
 * @conf listen_port         Int, default 11024.  The port to listen for new connections
 * @conf num_threads         Int, default 1.  The number of worker threads to use
 * @conf connection_timeout  Int, default 60.  Number of seconds before timeout on transfer
 * @conf drop_frames         Bool, default true.  Whether to drop frames when buffer fills.
 *
 * @par Metrics
 * @metric kotekan_buffer_recv_transfer_time_seconds
 *         The amount of time it took in seconds to transfer the last frame from the
 *         host given by the @c source label
 * @metric kotekan_buffer_recv_dropped_frame_total
 *         The number of times a frame was dropped because the @c buf was full at the time
 *         a block of was aviable to transfer to it.
 *
 * @todo Possibly factor out the threadpool.
 * @todo Allow for a different log level for workers from the main thread.
 *
 * @author Andre Renard
 */
class bufferRecv : public kotekan::Stage {
    friend class connInstance;

public:
    /// Constructor
    bufferRecv(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    ~bufferRecv();
    void main_thread() override;

    /// Adds the source port to the pipeline dot graph
    virtual std::string dot_string(const std::string& prefix) const override;

private:
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
     * @brief Updates the prometheus metric with time it took to receive the data.
     *        Thread safe.  Called only by worker threads
     */
    void set_transfer_time_seconds(const std::string& source_label, const double elapsed);

    /**
     * @brief Used only by worker threads to check if they should stop.
     * @return True if they should stop, false otherwise.
     */
    bool get_worker_stop_thread();

    /// The output buffer
    struct Buffer* buf;

    /// The port to listen for new connections on
    uint32_t listen_port;

    /// How long to wait before timing out a transfer
    int connection_timeout;

    /// The current frame to use next
    int current_frame_id = 0;

    /// Whether to drop frames when buffer starts filling up
    bool drop_frames;

    /// A lock on the current frame, since many systems may ask for the next frame
    std::mutex next_frame_lock;

    static void read_callback(evutil_socket_t fd, short what, void* arg);
    static void accept_connection(evutil_socket_t listener, short event, void* arg);

    /**
     * @brief Internal timer call back to check for thread exit condition
     *
     * @param fd Not used
     * @param event Not used
     * @param arg The bufferRecv object (just `this`, but this is a static function)
     */
    static void timer(evutil_socket_t fd, short event, void* arg);

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
    void internal_accept_connection(evutil_socket_t listener, short event, void* arg);

    /// The base event for libevent, run in the main_thread.
    /// This might be increased to more than one if there are performance issues.
    struct event_base* base;

    /// The number of frames dropped
    kotekan::prometheus::Counter& dropped_frame_counter;

    /// A lock on the @c dropped_frame_counter
    // TODO: move locking to prometheusMetrics?
    std::mutex dropped_frame_count_mutex;

    /// Time to receive the data
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& transfer_time_seconds;

    /// A lock on the `transfer_time_seconds`
    std::mutex transfer_time_seconds_mutex;

    // Worker threads (thread pool section)

    /// The number of worker threads to spawn
    uint32_t num_threads;

    /// The pool of worker threads
    std::vector<std::thread> thread_pool;

    /// Queue of functions to be called by the worker threads
    std::deque<connInstance*> work_queue;

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
enum class connState { header, metadata, frame, finished };

/**
 * @brief Args passed to the accept new connection call back function
 */
struct acceptArgs {
    /// The libevent base to use.
    struct event_base* base;

    /// The output buffer to attach to
    struct Buffer* buf;

    /// The main buffer recv object needs to be called to manage
    /// the frame ID to use (corrdinate between workers)
    bufferRecv* buffer_recv;

    /// Just copy the unique_name of the stage
    std::string unique_name;

    /// The log level to use.
    std::string log_level;
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
class connInstance : public kotekan::kotekanLogging {
public:
    /// Constructor
    connInstance(const std::string& producer_name, struct Buffer* buf, bufferRecv* buffer_recv,
                 const std::string& client_ip, int port, struct timeval read_timeout);

    /// Destructor
    ~connInstance();

    /**
     * @brief Function called to do the socket @c read
     */
    void internal_read_callback();

    // Reference counting is used to track how many outstanding
    // jobs have been queued for this instance.
    // Currently since we only do one READ at a time, this
    // isn't really needed, but might be useful for doing writes or
    // different types of event handing.

    /**
     * @brief Increases the reference count to this object
     */
    void increment_ref_count();

    /**
     * @brief Decreases the reference count to this object.
     * Note this can delete the object if @c close_frag == true
     */
    void decrement_ref_count();

    /**
     * @brief Stops this instance from processing more data
     * Since there might be more outstanding jobs queued for this
     * instance it isn't deleted unless the refernce count is zero.
     */
    void close_instance();

    /// The name of the parent kotekan_stage
    std::string producer_name;

    /// The kotekan buffer to transfer data into
    struct Buffer* buf;

    /// Pointer to the parient kotekan_stage which owns this instance
    bufferRecv* buffer_recv;

    /// The client IP address for this instance
    std::string client_ip;

    /// The port the client is connected on.
    int port;

    /// The event/read timeout
    struct timeval read_timeout;

    /// The libevent event which gets triggered on a read
    struct event* event_read;

    /// The socket associated with this instance
    evutil_socket_t fd;

    /// Tracks how many bytes have been read from the socket for the current read
    size_t bytes_read = 0;

    /// The start time of a new frame read
    double start_time;

    /// The buffer transfer header
    struct bufferFrameHeader buf_frame_header;

    /// Pointer to the local memory space which matching the size of the incoming frame.
    uint8_t* frame_space;

    /// Pointer to local memory for storing the metadata of the incoming frame.
    uint8_t* metadata_space;

    /// Lock to make sure only one instance of this jobs call backs is run at any one time.
    std::mutex instance_lock;

    /// The number of active jobs which hold a pointer to this
    /// object.  Note this doesn't include the libevent pointers.
    uint32_t reference_count = 0;

    /// Lock for updating the reference count
    std::mutex reference_count_lock;

    /// Set to true if we've encountered an error and need to close
    /// the connection attached to this instance.
    bool close_flag = false;

    /// The state of the transfer, starts with the header state
    connState state = connState::header;

    /**
     * @brief Handles the result of a READ which doesn't return a value > 0
     *
     * @param msg
     * @param err_num
     * @param bytes_read
     */
    inline void handle_error(const std::string& msg, int err_num, ssize_t bytes_read) {
        // Resource temporarily unavailable, no need to close connection
        // The two error codes cover MacOS and Linux
        if ((err_num == 35 || err_num == 11) && bytes_read != 0) {
            DEBUG2("Got resource unavailable error, {:d}, read return {:d}", err_num, bytes_read);
            decrement_ref_count();
            // Add the event back to the libevent queue, so we are notified
            // when more data becomes available.
            event_add(event_read, &read_timeout);
            return;
        }

        if (bytes_read == 0) {
            INFO("Connection to {:s} closed", client_ip);
            decrement_ref_count();
            close_instance();
            return;
        }

        // All other errors close the connection
        ERROR("Error with operation '{:s}' for client {:s}, error code {:d} ({:s}). Closing "
              "connection.",
              msg, client_ip, err_num, strerror(err_num));
        decrement_ref_count();
        close_instance();
    }
};

#endif
