#ifndef REST_SERVER_HPP
#define REST_SERVER_HPP

#include <event2/util.h>       // for evutil_socket_t
#include <evhttp.h>      // for evhttp  // IWYU pragma: keep
#include <stdint.h>            // for uint8_t
#include <sys/types.h>         // for u_short
#include <atomic>              // for atomic
#include <functional>          // for function
#include <map>                 // for map
#include <shared_mutex>        // for shared_timed_mutex
#include <string>              // for string, allocator, basic_string
#include <thread>              // for thread
#include <vector>              // for vector (CORS allowlist)

#include "Config.hpp"          // for Config
#include "kotekanLogging.hpp"  // for INFO_NON_OO
#include "json.hpp"            // for json
#include "fmt.hpp"             // for compile_string_to_view

namespace kotekan {


enum class HTTP_RESPONSE {
    OK = 200,
    BAD_REQUEST = 400,
    REQUEST_FAILED = 402,
    NOT_FOUND = 404,
    INTERNAL_ERROR = 500
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
     * Sends an HTTP response with "content-type" header set to "text/plain"
     *
     * @param[in] reply The body of the reply
     */
    void send_text_reply(const std::string& reply);

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
