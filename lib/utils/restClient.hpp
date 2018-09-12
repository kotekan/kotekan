/*****************************************
@file
@brief Send REST messages to a server.
*****************************************/
#ifndef RESTCLIENT_HPP
#define RESTCLIENT_HPP

#include "gsl-lite.hpp"

#include "json.hpp"
#include "restServer.hpp"
#include "errors.h"

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <evhttp.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/bufferevent.h>
#include <event2/dns.h>
#include <cxxabi.h>

/**
 * @brief The REST reply.
 *
 * Keeps data from the servers response.
 */
template<typename T>
struct restReply {
        friend class restClient;
    public:
        gsl::span<T> data;
    private:
        std::unique_ptr<T> raw_data;
};

/**
 * @class restClient
 * @brief REST client: Send REST messages to a server and maybe get a reply.
 *
 * This class supports sending GET messages and POST messages with json data
 * using libevent and provides access to data from the reply of the server.
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
     * To send a GET message, pass an empty JSON object (`{}`) as the parameter
     * `data`. To send a POST message, pass JSON data.
     *
     * @param path      Path to the endpoint
     *                  (e.g. "/endpoint_name")
     * @param data      JSON request or a nullptr (default: nullptr).
     * @param host      Host (default: "localhost").
     * @param port      Port (default: PORT_REST_SERVER).
     * @param retries   Max. retries to send message (default: 0).
     * @param timeout   Timeout in seconds. If -1 is passed, the default value
     * (of 50 seconds) is set (default: -1).
     * @return          `true` if successfull, otherwise `false`.
     */
    bool send(std::string path,
              const nlohmann::json& data = {},
              const std::string& host = "localhost",
              const unsigned short port = PORT_REST_SERVER,
              const int retries = 0, const int timeout = -1);

    /**
     * @brief Get the data attached to the reply of the server.
     *
     * If the server sent data with its reply, it will be returned in a
     * `restReply` struct. If no data was received from the server,
     * `data.empty()` will be true for the field `data` of the struct.
     * @returns Any data received from the server as a reply to a message sent.
     */
    template<typename T>
    struct restReply<T> get_reply();

    /**
     * @brief Get the data attached to the reply of the server as a string.
     *
     * If the server sent data with its reply, it will be returned in a
     * string. If no data was received from the server,
     * a string of length 0 will be returned.
     * @returns Any data received from the server as a reply to a message sent
     * (as a string).
     */
    std::string get_reply();

    /**
     * @brief Default constructor.
     */
    restClient() = default;

private:

    /// callback function for http request
    static void http_request_done(struct evhttp_request *req, void *arg);

    /// this is where the callback stores the reply
    static char* _data;
    static size_t _datalen;
    static bool _success;
};

template<typename T>
restReply<T> restClient::get_reply() {
    struct restReply<T> reply;

    if (_success == false || _datalen == 0) {
        reply.raw_data = std::unique_ptr<T>(nullptr);
        reply.data = gsl::span<T>(nullptr, 0);
        return reply;
    }
    if (_datalen % sizeof(T)) {
        int status;
        WARN("restClient: size of received data (%d) is not a multiple of " \
             "the size of the requested type %s (%d).",
             _datalen,
             abi::__cxa_demangle(typeid(T).name(), NULL, NULL, &status),
             sizeof(T));
        reply.raw_data = std::unique_ptr<T>(nullptr);
        reply.data = gsl::span<T>(nullptr, 0);
        return reply;
    }
    reply.raw_data = std::unique_ptr<T>((T*)_data);
    reply.data = gsl::span<T>((T*)_data, _datalen / sizeof(T));
    return reply;
}

#endif // RESTCLIENT_HPP

