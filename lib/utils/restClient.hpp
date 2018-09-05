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
 * @brief REST client: Send REST messages to a server.
 *
 * This class supports sending json data in a POST message using Mongoose -
 * Embedded Web Server / Embedded Networking Library.
 *
 * @todo Implement send_get() for GET messages.
 *
 * @author Rick Nitsche
 */
class restClient {
public:

    /**
     * @brief Send json data to a POST endpoint.
     * @param s_url     URL of the endpoint
     *                  (e.g. "localhost:12048/endpoint_name")
     * @param request   JSON request
     * @return          False in case of failure, True otherwise.
     */
    unique_ptr<struct restReply> send(string path,
                          const nlohmann::json& data = {},
                          const string& host = "localhost",
                          const unsigned short port = PORT_REST_SERVER,
                          const int retries = 0, const int timeout = -1);

    /// Default constructor.
    restClient() = default;

private:

    /// callback function for http request
    static void http_request_done(struct evhttp_request *req, void *arg);

    /// reply struct for callback to store reply
    static struct restReply _reply;
};

struct restReply {
        bool success = false;
        void* data = nullptr;
        size_t datalen = 0;
};

#endif // RESTCLIENT_HPP

