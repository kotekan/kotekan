/*****************************************
@file
@brief Send REST requests to a server.
*****************************************/
#ifndef RESTCLIENT_HPP
#define RESTCLIENT_HPP

#include "restServer.hpp" // for PORT_REST_SERVER

#include "json.hpp" // for json

#include <atomic>             // for atomic
#include <condition_variable> // for condition_variable
#include <event2/buffer.h>    // for evbuffer_iovec
#include <event2/http.h>      // for evhttp_connection
#include <event2/util.h>      // for evutil_socket_t
#include <functional>         // for function
#include <mutex>              // for mutex
#include <stddef.h>           // for size_t
#include <string>             // for string, allocator
#include <thread>             // for thread
#include <utility>            // for pair


/**
 * @class restClient
 * @brief REST client: Send REST messages to a server and maybe get a reply.
 *
 * This class supports sending GET messages and POST messages with json data
 * using libevent and provides access to data from the reply of the server.
 *
 * Implementation
 * ==============
 *
 * There is an event loop running in the main_thread() that gets started by the constructor.
 * The event thread is sending out requests, waits for results and calls the assigned callback
 * functions. All this has to be done from the same thread that runs the event loop, which makes
 * class more complicated than you would expect:
 * The function `make_request` does not itself create the request, because it's usually called from
 * another thread. To bring the request to the event loop thread, libevent's `bufferevent_pair` is
 * used (see http://www.wangafu.net/~nickm/libevent-book/Ref6a_advanced_bufferevents.html).
 * One side of the pair (`bev_req_write`) takes new requests in the form
 *
 * | struct restRequest | string host | string path | string json_data |
 *
 * The length of each string is in the `restRequest` struct. Writing all of this data to the
 * bufferevent's output buffer has to be done atomically, so that the read callback
 * (`_bev_req_readcb`) attached to the other side of the pair (`bew_req_read`) is not called more
 * than once. To achieve this, a `evbuffer_iovec` is used to reserve the necessary space in the
 * output buffer, then copying all data in and finally committing the written data once, which will
 * trigger the read callback. The described sequence still has to be protected from concurrent
 * access (`_mtx_bev_buffer`), because it fails if the buffer is written to between reserving and
 * committing.
 *
 * The request is then finally made by the same thread that runs the event loop, in the
 * read callback `_bev_req_readcb` attached to the reading side `bev_req_read` of the
 * bufferevent pair. This is done by the following:
 *
 * - Create a connection base.
 * - Make a new HTTP request. The external callback function as well as the connection base are
 *   attached to the request in order to pass them to the internal callback function.
 * - Write request data into the HTTP request output buffer.
 * - Start the request.
 *
 * When the request is done, the internal callback function `http_request_done` is called.
 * It calls the external callback that was initially given to `make_request` and hands the
 * result of the request to it.
 *
 * @author Rick Nitsche
 */
class restClient {
public:
    /// The reply of a request: a pair with a success boolean and the reply string
    using restReply = std::pair<bool, std::string>;

    /**
     * @brief Returns an instance of the rest client.
     *
     * @return Returns the rest client instance.
     */
    static restClient& instance();

    /**
     * @brief Send GET or POST with json data to an endpoint.
     *
     * To send a GET message, pass an empty JSON object (`{}`) as the parameter
     * `data`. To send a POST message, pass JSON data.
     *
     * @param path      Path to the endpoint
     *                  (e.g. "/endpoint_name")
     * @param request_done_cb   A callback function that when the request is
     *                          complete.
     * @param data      JSON request (`{}` to send a GET request,
     *                  default: `{}`).
     * @param host      Host (default: "127.0.0.1", Prefer numerical, because
     *                  the DNS lookup is blocking).
     * @param port      Port (default: PORT_REST_SERVER).
     * @param retries   Max. retries to send message (default: 0).
     * @param timeout   Timeout in seconds. If -1 is passed, the default value
     * (of 50 seconds) is set (default: -1).
     */
    void make_request(const std::string& path, std::function<void(restReply)>& request_done_cb,
                      const nlohmann::json& data = {}, const std::string& host = "127.0.0.1",
                      const unsigned short port = PORT_REST_SERVER, const int retries = 0,
                      const int timeout = -1);

    /**
     * @brief Send GET or POST with json data to an endpoint. Blocking.
     *
     * To send a GET message, pass an empty JSON object (`{}`) as the parameter
     * `data`. To send a POST message, pass JSON data. This blocks until a reply
     * is received or there was an error.
     *
     * @param path      Path to the endpoint
     *                  (e.g. "/endpoint_name")
     * @param data      JSON request (`{}` to send a GET request,
     *                  default: `{}`).
     * @param host      Host (default: "127.0.0.1", Prefer numerical, because
     *                  the DNS lookup is blocking).
     * @param port      Port (default: PORT_REST_SERVER).
     * @param retries   Max. retries to send message (default: 0).
     * @param timeout   Timeout in seconds. If -1 is passed, the default value
     * (of 50 seconds) is set (default: -1).
     *
     * @return          restReply object.
     */
    restReply make_request_blocking(const std::string& path, const nlohmann::json& data = {},
                                    const std::string& host = "127.0.0.1",
                                    const unsigned short port = PORT_REST_SERVER,
                                    const int retries = 0, const int timeout = -1);

private:
    /// A structure to pass requests around inside the restClient
    struct restRequest {
        // The strings are of dynamic length and kept outside of the struct.
        // But we need to pass their sizes...
        size_t data_len;
        size_t path_len;
        size_t host_len;

        int retries;
        int timeout;
        unsigned short port;
        std::function<void(restReply)>* request_done_cb;
    };

    /// Private constuctor
    restClient();

    /// Private destructor
    virtual ~restClient();

    /// @brief Internal timer call back to check for thread exit condition
    static void timer(evutil_socket_t fd, short event, void* arg);

    /// Do not allow copy or assignment
    restClient(restClient const&);
    void operator=(restClient const&);

    /// Internal thread function which runs the event loop.
    void event_thread();

    /// callback function for http requests
    static void http_request_done(struct evhttp_request* req, void* arg);

    /// cleanup function that deletes evcon and the argument pair
    static void cleanup(std::pair<std::function<void(restReply)>*, evhttp_connection*>* pair);

    /// Only to be called inside event thread
    bool _make_request(const restRequest* request);

    /// Read callback for the request scheduling thread. Called when the socket holding request
    /// pointers has new data.
    static void _bev_req_readcb(struct bufferevent* bev, void* arg);

    /// error callback for the bufferevent pair
    static void _bev_req_errcb(struct bufferevent* bev, short what, void* arg);

    /// Helper function to copy data of any length into iovec structure.
    static inline void _copy_to_iovec(const void* src, size_t len_src, evbuffer_iovec* iovec,
                                      int* i_extends, size_t* i_vec, int n_extends);

    /// Main event thread handle
    std::thread _main_thread;

    /// flag set to true when exit condition is reached
    std::atomic<bool> _stop_thread;

    /// event base
    /// TODO: use the event base of the restServer
    struct event_base* _base;

    /// dns base
    struct evdns_base* _dns;

    /// Condition variable to signal that event thread has started
    std::condition_variable _cv_start;
    bool _event_thread_started;
    std::mutex _mtx_start;

    /// Writing socket event to pass requests to the event thread
    struct bufferevent* bev_req_write;

    /// Reading socket event to pass requests to the event thread
    struct bufferevent* bev_req_read;

    /// Lock to protect writing to the bufferevent_pair's output buffer (the datasetManager does
    /// this in threads for example).
    std::mutex _mtx_bev_buffer;
};

#endif // RESTCLIENT_HPP
