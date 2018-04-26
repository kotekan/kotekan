#ifndef BASEBAND_REQUEST_MANAGER_HPP
#define BASEBAND_REQUEST_MANAGER_HPP

#include "json.hpp"
#include "restServer.hpp"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>


/**
 * Helper structure to capture a baseband dump request.
 */
struct BasebandRequest {
    int64_t start_fpga;
    int64_t length_fpga;
    std::chrono::system_clock::time_point received = std::chrono::system_clock::now();
};


/**
 * Helper structure to track the progress of a dump request's processing.
 */
struct BasebandDumpStatus {
    const BasebandRequest request;
    const size_t bytes_total = request.length_fpga * 17;
    size_t bytes_remaining = bytes_total;
};


class BasebandRequestManager {
public:
    /**
     * @brief Returns the singleton instance of the BasebandRequestManager object.
     * @return A pointer to the BasebandRequestManager object
     */
    static BasebandRequestManager& instance();

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

    /**
     * @brief Tries to get the next dump request to process.
     *
     * @return a shared_ptr to the `BasebandDumpStatus` object if there is a
     * request available for the readout process handling frequency `freq_id`,
     * or nullptr if the request queue is empty.
     */
    std::shared_ptr<BasebandDumpStatus> get_next_request(const uint32_t freq_id);

private:
    /// Constructor, not used directly
    BasebandRequestManager() = default;

    /// Queue of unprocessed baseband requests for each basebandReadout process,
    /// indexed by `freq_id`
    std::map<uint32_t, std::deque<BasebandRequest>> requests;

    /// Queue of baseband dumps in progress
    std::vector<std::shared_ptr<BasebandDumpStatus>> processing;

    /// request updating lock
    std::mutex requests_lock;

    /// new request notification
    std::condition_variable requests_cv;
};

#endif /* BASEBAND_REQUEST_MANAGER_HPP */
