/*****************************************
@file
@brief Send REST messages to a server.
*****************************************/
#ifndef RESTCLIENT_HPP
#define RESTCLIENT_HPP

#include "json.hpp"

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
    struct restReply send_json (const char *s_url,
                                const nlohmann::json *request);

    /// Default constructor.
    restClient() = default;

    /// exit flag set by ev_handler to inform about connection status
    static int _s_exit_flag;

    /// URL
    static const char *_s_url;

    private:

    /// event handler that is called by mongoose
    static void ev_handler(struct mg_connection *nc, int ev, void *ev_data);
};

struct restReply {
        bool success = false;
        void* data = nullptr;
        size_t data_len = 0;
};

#endif // RESTCLIENT_HPP

