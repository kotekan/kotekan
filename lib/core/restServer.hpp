#ifndef REST_SERVER_HPP
#define REST_SERVER_HPP

#include "Config.hpp" // for Config

#include "json.hpp" // for json

#include "visUtil.hpp"
#include "prometheusMetrics.hpp"

#include <atomic>        // for atomic
#include <event2/util.h> // for evutil_socket_t
#include <evhttp.h>      // for evhttp  // IWYU pragma: keep
#include <functional>    // for function
#include <map>           // for map
#include <shared_mutex>  // for shared_timed_mutex
#include <stdint.h>      // for uint8_t
#include <string>        // for string, allocator
#include <sys/types.h>   // for u_short
#include <thread>        // for thread

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
     * @brief Start the rest server, should only be called once by the framework
     *
     * @param bind_address The address to bind too.  Default: 0.0.0.0
     * @param port The port to bind.  Default: PORT_REST_SERVER
     */
    void start(const std::string& bind_address = "0.0.0.0", u_short port = PORT_REST_SERVER);

    void stop();

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
     * @brief Removes all aliases
     */
    void remove_all_aliases();

    /// The port to use
    const u_short& port;

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

    /**
     * @brief Returns the aliases map
     *
     * @return Alias map
     */
    std::map<std::string, std::string>& get_aliases();

    /// Map of GET callbacks
    std::map<std::string, std::function<void(connectionInstance&)>> get_callbacks;

    /// Map of JSON POST callbacks
    std::map<std::string, std::function<void(connectionInstance&, nlohmann::json&)>> json_callbacks;

    /// Alias map
    std::map<std::string, std::string> aliases;

    /// Map of callback timers
    std::map<std::string, StatTracker> callback_timers;

    /// Callback timer metrics
    prometheus::MetricFamily<kotekan::prometheus::Gauge>* timer_metrics;

    /// Mutex to lock changes to the maps while a request is in progress
    std::shared_timed_mutex callback_map_lock;

    /// The libevent base
    struct event_base* event_base = nullptr;

    /// The libevent HTTP server object
    struct evhttp* ev_server = nullptr;

    /// Bind address
    std::string bind_address;

    /// The port to use
    u_short _port;

    /// Main server thread handle
    std::thread main_thread;

    /// Flag set to true when exit condition is reached
    std::atomic<bool> stop_thread;

    /// Allow connectionInstance to use internal helper functions
    friend class connectionInstance;
};

} // namespace kotekan

#endif /* REST_SERVER_HPP */
