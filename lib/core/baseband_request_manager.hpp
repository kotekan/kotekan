/**
 * @file
 * @brief Manager for tracking baseband buffer writeout requests
 *  - airspyInput : public KotekanProcess
 */

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
 * @class BasebandRequest
 * @brief Helper structure to capture a baseband dump request.
 */
struct BasebandRequest {
    /// FRB internal unique event ID
    uint64_t event_id;
    /// Starting FPGA frame of the dump
    int64_t start_fpga;
    /// Length of the dump in FPGA frames
    int64_t length_fpga;
    /// destination file (relative to ``base_dir`` from the configuration.)
    std::string file_name;
    /// Time when the request was received
    std::chrono::system_clock::time_point received = std::chrono::system_clock::now();
};


/**
 * @class BasebandRequestState
 * @brief State of the request
 */
enum class BasebandRequestState { WAITING, INPROGRESS, DONE, ERROR };

/**
 * @class BasebandDumpStatus
 * @brief Helper structure to track the progress of a dump request's processing.
 */
struct BasebandDumpStatus {
    /// The request that is being tracked
    const BasebandRequest request;
    /// Amount of the data to dump, in bytes
    size_t bytes_total = 0;
    /// Remaining data to write, in bytes
    size_t bytes_remaining = bytes_total;
    BasebandRequestState state = BasebandRequestState::WAITING;
    /// Description of the failure, when the state is ERROR
    std::string reason = "";
};


/**
 * @class BasebandRequestManager
 * @brief Class for receiving baseband dump requests and sending request status
 *
 * This class must be registered with a kotekan REST server instance,
 * using the @c register_with_server() function.
 *
 * This class is a singleton, and can be accessed with @c instance(). The normal
 * use is for the @c basebandReadout process to call @get_next_request in a
 * loop, and when the result is non-null, use the returned @BasebandDumpStatus
 * to keep track of the data written so far. Once the writing of the data file
 * is completed, the ``state`` of the request should be set to ``DONE``.
 *
 * @author Davor Cubranic
 */
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
