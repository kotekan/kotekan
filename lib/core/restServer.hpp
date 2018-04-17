#ifndef REST_SERVER_HPP
#define REST_SERVER_HPP

#include "Config.hpp"
#include "mongoose.h"
#include "json.hpp"
#include <thread>
#include <functional>
#include <map>

enum class HTTP_RESPONSE {
    OK = 200,
    BAD_REQUEST = 400,
    REQUEST_FAILED = 402,
    NOT_FOUND = 404,
    INTERNAL_ERROR = 500
};

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
    connectionInstance(mg_connection *nc, int ev, void * ev_data);
    ~connectionInstance();

    /**
     * @brief Sends an error message with the details in the
     *        HTTP "Error:" header
     *
     * @param message The message to include in the HTTP header
     * @param status_code The HTTP error code.
     */
    void send_error(const std::string &message, const HTTP_RESPONSE &status);

    /**
     * @brief Sends a json reply to the client
     *
     * @param json_reply The json object to send to the client.
     */
    void send_json_reply(nlohmann::json &json_reply);

    /**
     * @brief Sends a binary reply to the client
     *
     * @param data Pointer to the data to send
     * @param len The size of the data in bytes to send
     */
    void send_binary_reply(uint8_t * data, int len);

    /**
     * @brief Sents an empty reply with the given status code
     *
     * @param status_code The HTTP status code to return
     */
    void send_empty_reply(const HTTP_RESPONSE &status);

    /**
     * Sends an HTTP response with "content-type" header set to "text/plain"
     *
     * @param[in] reply The body of the reply
     * @param[in] status_code HTTP response status code (default = HTTP_OK = 200)
     */
    void send_text_reply(const std::string &reply, const HTTP_RESPONSE &status = HTTP_RESPONSE::OK);

    /**
     * @brief Returns the message body.
     *
     * @return The body of the http request message
     */
    std::string get_body();

    /**
     * @brief Get the full http request message
     *
     * @return The full http request message
     */
    std::string get_full_message();
private:

    /// The connection details
    mg_connection *nc;

    /// Event ID
    int ev;

    /// Event data
    void * ev_data;
};

/**
 * @brief Framework level REST server (singleton)
 *
 * Provides the system for any framwork process to register endpoints.
 *
 * Currently this object uses the mongoose webserver internally to handle
 * the http requests.
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
    static restServer &instance();

    /**
     * @brief Set the server thread CPU affinity
     *
     * Pulls the CPU thread affinity from the config at "/rest_server"
     *
     * @param config The config file currently being used.
     */
    void set_server_affinity(Config &config);

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
                               std::function<void(connectionInstance &)> callback);

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
    void register_json_callback(std::string endpoint,
                        std::function<void(connectionInstance &, nlohmann::json &)> callback);

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
    void mongoose_thread();

    /**
     * @brief Internal callback function for the mongoose server.
     *
     * @param nc Connection struct
     * @param ev Event ID
     * @param ev_data Event data struct
     */
    static void handle_request(struct mg_connection *nc, int ev, void *ev_data);

    /**
     * @brief Callback which returns list of endpoints to caller.
     *
     * @param conn The connection to return endpoints too.
     */
    void endpoint_list_callback(connectionInstance &conn);

    /**
     * @brief Trys to parse the JSON in a POST message
     *
     * Fills the reference @c json_parse with the json string
     * provided in the message.
     *
     * If this function falls, then don't call @c ms_send
     *
     * @param nc connection object
     * @param ev event ID
     * @param ev_data event data
     * @param json_parse Reference to the JSON object to fill.
     * @return int 0 if the message contains valid JSON, and -1 if not.
     */
    int handle_json(struct mg_connection *nc, int ev, void *ev_data, nlohmann::json &json_parse);

    /// Map of GET callbacks
    std::map<string, std::function<void(connectionInstance &)>> get_callbacks;

    /// Map of JSON POST callbacks
    std::map<string, std::function<void(connectionInstance &, json &)>> json_callbacks;

    /// mongoose management object.
    struct mg_mgr mgr;

    /// mongoose main listen connection.
    struct mg_connection *nc;

    /// The port to use, for now this is constant 12048
    const char *port = "12048";

    /// Main server thread handle
    std::thread main_thread;
};

#endif /* REST_SERVER_HPP */
