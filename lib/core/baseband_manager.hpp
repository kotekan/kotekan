#ifndef BASEBAND_MANAGER_HPP
#define BASEBAND_MANAGER_HPP

#include "json.hpp"
#include "restServer.hpp"
#include <chrono>
#include <deque>


struct BasebandRequest {
    int64_t start_fpga;
    int64_t length_fpga;
    std::chrono::system_clock::time_point received = std::chrono::system_clock::now();
};


class BasebandManager {
public:
    /**
     * @brief Returns the singleton instance of the BasebandManager object.
     * @return A pointer to the BasebandManager object
     */
    static BasebandManager& instance();

    /**
     * @brief Registers this class with the REST server, creating the
     *        /baseband end point
     * @param rest_server The server to register with.
     */
    void register_with_server(restServer * rest_server);

    /**
     * @brief The call back function for GET requests to `/baseband`.
     *
     * This function is never called directly.
     *
     * @param conn The connection instance to send results to
     */
    void status_callback(connectionInstance& conn);

    /**
     * @brief The call back function for POST requests to `/baseband`.
     *
     * This function is never called directly.
     *
     * @param conn The connection instance to send results to
     * @param request JSON dictionary with the request data
     */
    void handle_request_callback(connectionInstance& conn, json& request);


private:
    /// Constructor, not used directly
    BasebandManager() = default;

    /// Queue of unprocessed baseband requests
    std::deque<BasebandRequest> requests;

    /// request updating lock
    std::mutex requests_lock;
};

#endif /* BASEBAND_MANAGER_HPP */
