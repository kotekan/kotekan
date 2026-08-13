#ifndef REST_SERVER_HPP
#define REST_SERVER_HPP

#include "Config.hpp"         // for Config
#include "kotekanLogging.hpp" // for INFO_NON_OO

#include "fmt.hpp"  // for compile_string_to_view
#include "json.hpp" // for json

#include <atomic>             // for atomic
#include <condition_variable> // for condition_variable
#include <event2/util.h>      // for evutil_socket_t
#include <evhttp.h>           // for evhttp  // IWYU pragma: keep
#include <functional>         // for function
#include <map>                // for map
#include <mutex>              // for mutex
#include <shared_mutex>       // for shared_timed_mutex
#include <stdint.h>           // for uint8_t
#include <string>             // for string, allocator, basic_string
#include <sys/types.h>        // for u_short
#include <thread>             // for thread
#include <vector>             // for vector (CORS allowlist)

namespace kotekan {


enum class HTTP_RESPONSE {
    OK = 200,
    BAD_REQUEST = 400,
    REQUEST_FAILED = 402,
    NOT_FOUND = 404,
    INTERNAL_ERROR = 500,
    SERVICE_UNAVAILABLE = 503
};

#define PORT_REST_SERVER 12048

/**
 * @brief Contains details of a request (POST or GET), and provides
 *        functions for replying to the request.
 *
 * Used in call back functions to provide a way to reply to
 * request with either an error, json, binary, text, or empty message.
 *
 * The @c send_ functions should called exactly once per connection instance.
 *
 * @author Andre Renard
 */
class connectionInstance {
public:
    connectionInstance(struct evhttp_request* request);
    ~connectionInstance();

    /**
     * @brief Sends an error message with the details in the
     *        HTTP "Error:" header
     *
     * @param message The message to include in the HTTP header
     * @param status The HTTP error code.
     */
    void send_error(const std::string& message, const HTTP_RESPONSE& status);

    /**
     * @brief Sends a json reply to the client
     *
     * @param json_reply The json object to send to the client.
     */
    void send_json_reply(const nlohmann::json& json_reply);

    /**
     * @brief Sends a binary reply to the client
     *
     * @param data Pointer to the data to send
     * @param len The size of the data in bytes to send
     */
    void send_binary_reply(uint8_t* data, int len);

    /**
     * @brief Sents an empty reply with the given status code
     *
     * @param status The HTTP status code to return
     */
    void send_empty_reply(const HTTP_RESPONSE& status);

    /**
     * Sends an HTTP response with a text "content-type" header
     *
     * @param[in] reply The body of the reply
     * @param[in] content_type The MIME type to declare; the default suits plain
     *                         text, but a caller sending a known format (DOT,
     *                         CSV, ...) should name it so clients can act on it.
     *                         Name the charset too: HTTP says a text type with
     *                         no charset is ISO-8859-1, and clients that follow
     *                         that rule (`requests`, among others) will mangle
     *                         any reply holding non-ASCII.
     */
    void send_text_reply(const std::string& reply,
                         const std::string& content_type = "text/plain; charset=utf-8");

    /**
     * @brief Returns the message body.
     *
     * @return The body of the http request message
     */
    std::string get_body();

    /**
     * @brief Get the uri
     *
     * @return The uri of the http request message
     */
    std::string get_uri();

    /**
     * @brief Gets the query args as a map of key value strings
     *
     * Example "/my_endpoint?val=42&myval=hello" would return a map with items:
     * map["val"] == "42"
     * map["myval"] == "hello"
     *
     * In the case there are no URL query args, an empty map is returned.
     *
     * @return A map with string keys and string values with any url query args
     */
    std::map<std::string, std::string> get_query();

private:
    /// The request details
    struct evhttp_request* request;

    /// The buffer with the reply contents
    struct evbuffer* event_buffer;
};

/**
 * @brief Framework level REST server (singleton)
 *
 * Provides the system for any framwork stage to register endpoints.
 *
 * This object uses libevent internally to handle the http requests.
 *
 * See the docs for examples of using this class.
 *
 * @author Andre Renard
 */
class restServer {

public:
    /**
     * @brief Returns an instance of the rest server.
     *
     * @return Returns the rest server instance.
     */
    static restServer& instance();

    /**
     * @brief Check whether the restServer singleton is still alive.
     *
     * Safe to call during static destruction. Returns false after the
     * restServer destructor has begun.
     */
    static bool is_alive();

    /**
     * @brief Validate a bind address.
     *
     * Accepts the following forms:
     *  - IPv4 literals (e.g., "127.0.0.1", "0.0.0.0").
     *  - localhost (only works for self-connections).
     *
     * This is a syntactic check. For hostnames, it does not resolve DNS; use
     * @ref canBindToAddress to check actual bindability.
     *
     * @param address Address string to validate (no port or scheme).
     * @return true if syntactically valid; false otherwise.
     */
    bool isValidAddress(const std::string& address) const;

    /**
     * @brief Checks if the given IP address and port can be bound to.
     *
     * This function checks if the given IP address and port can be bound to
     * by attempting to create a socket and bind it. It does not actually
     * start a server, but checks if the address is available.
     *
     * @param ip The IP address to check.
     * @param port The port to check.
     * @return true if the address can be bound, false otherwise.
     */
    bool canBindToAddress(const std::string& ip, const u_short port) const;

    /**
     * @brief Start the rest server, should only be called once by the framework
     *
     * @param bind_address The address to bind too.  Default: 0.0.0.0
     * @param port The port to bind.  Default: PORT_REST_SERVER
     */
    void start(const std::string& bind_address = "0.0.0.0", u_short port = PORT_REST_SERVER);

    /**
     * @brief Quiesce request handling ahead of process teardown.
     *
     * Stop dispatching to registered endpoint callbacks and block until any
     * handler already running on the server thread has returned. After this
     * call every incoming request is answered with 503 (Service Unavailable)
     * without touching a callback, so it is safe for the caller to destruct
     * the stages that own those callbacks.
     *
     * This closes a use-after-free on shutdown: an external signal (SIGINT /
     * SIGTERM) tears the pipeline down on the main thread while the server
     * thread may be part-way through a handler that captured a stage's @c
     * this. handle_request looks a callback up under a lock but invokes a
     * copy after releasing it (so callbacks may re-register endpoints), so the
     * map lock alone does not keep the stage alive across the call; this
     * barrier does.
     *
     * One-way: requests are refused permanently, so this is only for process
     * teardown -- not for the @c /stop path, which must leave the server
     * usable for a later @c /start. The caller must not hold any lock that
     * request handlers also take (e.g. the kotekan state lock in kotekan.cpp)
     * while this drains, or drain and handler deadlock against each other.
     * The drain is bounded: after 30 seconds an error is logged and teardown
     * proceeds anyway. When called from the server thread itself the drain is
     * skipped, since the single-threaded event loop guarantees no other
     * handler is in flight.
     *
     * The server thread is left running (endpoints stay resolvable, just
     * refused) so this composes with the framework's normal singleton
     * teardown. Idempotent.
     */
    void stop_processing();

    /**
     * @brief Set the server thread CPU affinity
     *
     * Pulls the CPU thread affinity from the config at "/rest_server"
     *
     * @param config The config file currently being used.
     */
    void set_server_affinity(Config& config);

    /**
     * Registers a GET style callback for a specified HTTP endpoint.
     *
     * @param[in] endpoint Path section of the URL that is handled by the callback
     * @param[in] callback Callback function invoked to handle the request on the endpoint
     *
     * @note Re-registering on an endpoint will override the previous
     * callback value.
     */
    void register_get_callback(std::string endpoint,
                               std::function<void(connectionInstance&)> callback);

    /**
     * Registers a POST callback for a specified HTTP endpoint.
     *
     * Systems calling one of these endpoints are expected to provide a JSON string
     * in the POST data.
     *
     * @param[in] endpoint Path section of the URL that is handled by the callback
     * @param[in] callback Callback function invoked to handle the request on the endpoint
     *
     * @note Re-registering on an endpoint will override the previous
     * callback value.
     */
    void register_post_callback(std::string endpoint,
                                std::function<void(connectionInstance&, nlohmann::json&)> callback);

    /**
     * @brief Removes the GET endpoint referenced by @c endpoint
     *
     * @param endpoint The endpoint to remove.
     */
    void remove_get_callback(std::string endpoint);

    /**
     * @brief Removes the JSON POST endpoint referenced by @c endpoint
     *
     * @param endpoint The endpoint to remove.
     */
    void remove_json_callback(std::string endpoint);

    /**
     * @brief Adds an alias for a given endpoint
     *
     * Maps a new endpoint at @c alias to the endpoint given
     * by the end point given by @c target
     *
     * Note: does not check that the endpoint exists,
     * of if an alias for this endpoint already exists.
     *
     * @todo Should there be more error checking here?
     *
     * @param alias The new endpoint
     * @param target The existing endpoint to map to
     */
    void add_alias(std::string alias, std::string target);

    /**
     * @brief Remove the alias endpoint if it exists
     *
     * @param alias The alias to remove
     */
    void remove_alias(std::string alias);

    /**
     * @brief Adds aliases from the /rest_server/aliases part of the config
     *
     * @param config The config file to use
     */
    void add_aliases_from_config(Config& config);

    /**
     * @brief Configures CORS handling from the @c /rest_server config block.
     *
     * Two keys, both optional, CORS off entirely if neither is set:
     *
     *  - @c cors_allow_origins : a list of exact origin strings
     *    (e.g. @c "http://cobalt.lwlab:8080"). When a request carries an
     *    @c Origin header that matches one of these, that exact value is
     *    reflected back in @c Access-Control-Allow-Origin (plus
     *    @c Vary: Origin). Non-matching / origin-less requests get no CORS
     *    headers, so the browser blocks the cross-origin read -- this is
     *    the recommended, scoped setting.
     *  - @c enable_cors : legacy boolean. When @c true (and no allowlist is
     *    given) every reply gets @c Access-Control-Allow-Origin: * . Kept
     *    for backward compatibility; prefer the allowlist.
     *
     * Off by default (secure). Only the browser viewer needs this; pure
     * machine-to-machine REST use never does.
     *
     * @param config The config file to use
     */
    void set_cors_from_config(Config& config);

    /// True if CORS is configured at all (allowlist non-empty or the legacy
    /// boolean set). Used to decide whether to accept OPTIONS preflights.
    inline bool cors_enabled() const {
        return _enable_cors || !_cors_allow_origins.empty();
    }

    /// Resolve the @c Access-Control-Allow-Origin value to send for a
    /// request whose @c Origin header is @p request_origin (may be null).
    /// Returns the empty string when no CORS header should be emitted.
    std::string cors_allow_origin_for(const char* request_origin) const;

    /**
     * @brief Removes all aliases
     */
    void remove_all_aliases();

    // Getter for the port to use
    inline u_short port() const {
        return _port;
    }

    // Getter for the bind address to use
    inline std::string bind_address() const {
        return _bind_address;
    }

private:
    /// Private constuctor
    restServer();

    /// Private destructor
    virtual ~restServer();

    // Do not allow copy or assignment
    restServer(restServer const&);
    void operator=(restServer const&);

    /**
     * @brief Internal thread function which runs the server.
     */
    void http_server_thread();

    /**
     * @brief Internal timer call back to check for thread exit condition
     *
     * @param fd Not used
     * @param event Not used
     * @param arg The bufferRecv object (just `this`, but this is a static function)
     */
    static void timer(evutil_socket_t fd, short event, void* arg);

    /**
     * @brief libevent callback that emits the "started server" log line.
     *
     * Scheduled with a zero-delay timeout so it runs on the first
     * event-loop iteration; the line the python runner waits on then
     * signals a running loop, not just a bound socket.
     *
     * @param fd Not used
     * @param event Not used
     * @param arg The restServer instance (`this` at registration)
     */
    static void log_started(evutil_socket_t fd, short event, void* arg) {
        (void)fd;
        (void)event;
        restServer* server = (restServer*)arg;
        // This INFO line is parsed by the python runner to get the RESTserver port. Don't edit.
        INFO_NON_OO("restServer: started server on address:port {:s}:{:d}", server->_bind_address,
                    server->_port);
    }

    /**
     * @brief Internal callback function for the evhttp server.
     *
     * @param request   The request object
     * @param cb_data   Expects a pointer to the REST server object
     */
    static void handle_request(struct evhttp_request* request, void* cb_data);

    /**
     * @brief Callback which returns list of endpoints to caller.
     *
     * @param conn The connection to return endpoints to.
     */
    void endpoint_list_callback(connectionInstance& conn);

    /**
     * @brief Trys to parse the JSON in a POST message
     *
     * Fills the reference @c json_parse with the json string
     * provided in the message.
     *
     * If this function falls, then don't call @c ms_send
     *
     * @param request The libevent http request object
     * @param json_parse Reference to the JSON object to fill.
     * @return int 0 if the message contains valid JSON, and -1 if not.
     */
    int handle_json(struct evhttp_request* request, nlohmann::json& json_parse);

    /**
     * @brief Returns the http message as a string, or an empty string
     *
     * @param request The libevent http request object
     * @return string The http message if it exists, or an empty string
     */
    static std::string get_http_message(struct evhttp_request* request);

    /**
     * @brief Generates a string message to match one of the response codes
     *        in the @c HTTP_RESPONSE enum
     *
     * @param status The responce code enum
     * @return string The string message matching that code
     */
    static std::string get_http_responce_code_text(const HTTP_RESPONSE& status);

    /// Map of GET callbacks
    std::map<std::string, std::function<void(connectionInstance&)>> get_callbacks;

    /// Map of JSON POST callbacks
    std::map<std::string, std::function<void(connectionInstance&, nlohmann::json&)>> json_callbacks;

    /// Alias map
    std::map<std::string, std::string> aliases;

    /// Mutex to lock changes to the maps while a request is in progress
    std::shared_timed_mutex callback_map_lock;

    /**
     * @brief Waits until no request is executing the callback for @c endpoint.
     *
     * Callbacks are copied out of the maps and invoked with
     * @c callback_map_lock released, so erasing a map entry does not stop a
     * callback that is already running. Endpoints are usually bound to a stage
     * (e.g. @c gpuProcess::profile_callback), and that stage starts tearing
     * itself down as soon as the remove call returns, so unregistration has to
     * wait the running callback out or the callback dereferences freed memory.
     *
     * Waits are bounded: a callback stuck for longer than the drain timeout
     * means the REST thread is wedged and the caller cannot safely free the
     * callback's resources, so kotekan is shut down with a fatal error.
     * Must not be called while holding @c callback_map_lock.
     */
    void drain_endpoint(const std::string& endpoint);

    /// Endpoint whose callback the libevent dispatch thread is currently
    /// executing (empty if none). A single thread runs all callbacks, so one
    /// marker suffices. Protected by @c _inflight_lock.
    std::string _inflight_endpoint;

    /// Guards @c _inflight_endpoint
    std::mutex _inflight_lock;

    /// Signalled whenever an in-flight callback finishes
    std::condition_variable _inflight_cv;

    /// The libevent base
    struct event_base* event_base = nullptr;

    /// The libevent HTTP server object
    struct evhttp* ev_server = nullptr;

    /// Bind address
    std::string _bind_address;

    /// The port to use
    u_short _port;

    /// Main server thread handle
    std::thread main_thread;

    /// Flag set to true when exit condition is reached
    std::atomic<bool> stop_thread;

    /// Held shared for the whole duration of a request handler and exclusively
    /// by @c stop_processing, so shutdown can wait out any in-flight handler
    /// before the stages those handlers reach into are destructed. Distinct
    /// from @c callback_map_lock (which only guards the map structure and is
    /// released before a callback is invoked).
    std::shared_timed_mutex request_lock;

    /// Cleared by @c stop_processing to refuse further callback dispatch. Once
    /// false, handle_request answers 503 without invoking any endpoint.
    std::atomic<bool> accepting_requests{true};

    /// Cached value of /rest_server/enable_cors (legacy "*" mode); see
    /// @c set_cors_from_config.
    bool _enable_cors = false;

    /// Cached /rest_server/cors_allow_origins allowlist. When non-empty,
    /// CORS uses validated Origin reflection instead of "*".
    std::vector<std::string> _cors_allow_origins;

    /// Allow connectionInstance to use internal helper functions
    friend class connectionInstance;
};

} // namespace kotekan

#endif /* REST_SERVER_HPP */
