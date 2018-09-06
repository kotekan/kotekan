/*****************************************
@file
@brief Send REST messages to a server.
*****************************************/
#ifndef RESTCLIENT_HPP
#define RESTCLIENT_HPP

#include "json.hpp"
#include "restServer.hpp"

using namespace std;

/**
 * @class restClient
 * @brief REST client: Send REST messages to a server and maybe get a reply.
 *
 * This class supports sending GET messages and POST messages with json data
 * using libevent. Any reply received is returned.
 *
 * @warning This is not thread save. Don't share an object of this between
 * threads and don't send requests from different threads at the same time.
 *
 * @author Rick Nitsche
 */
class restClient {
public:

    /**
     * @brief Send GET or POST with json data to an endpoint.
     *
     * To send a GET message, pass a `nullptr` as the parameter `data`.
     * To send a POST message, pass JSON data.
     *
     * @param path      Path to the endpoint
     *                  (e.g. "/endpoint_name")
     * @param data      JSON request or a nullptr (default: nullptr).
     * @param host      Host (default: "localhost").
     * @param port      Port (default: PORT_REST_SERVER).
     * @param retries   Max. retries to send message (default: 0).
     * @param timeout   Timeout in seconds. If -1 is passed, the default value
     * (of 50 seconds) is set (default: -1).
     * @return          The servers reply, indicating success, any sent data
     * and its length.
     */
    unique_ptr<struct restReply> send(string path,
                          const nlohmann::json& data = {},
                          const string& host = "localhost",
                          const unsigned short port = PORT_REST_SERVER,
                          const int retries = 0, const int timeout = -1);

    /**
     * @brief Default constructor.
     */
    restClient() = default;

private:

    /// callback function for http request
    static void http_request_done(struct evhttp_request *req, void *arg);

    /// reply struct for callback to store reply
    static struct restReply _reply;
};

/**
 * @brief The REST reply.
 *
 * Indicates if the REST request was successfull and keeps any data attached to
 * the servers response as well as the data size.
 */
struct restReply {
        bool success = false;
        void* data = nullptr;
        size_t datalen = 0;
};

#endif // RESTCLIENT_HPP

