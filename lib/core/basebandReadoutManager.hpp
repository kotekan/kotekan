/**
 * @file
 * @brief Manager for tracking baseband buffer writeout requests
 *  - basebandReadoutManager
 *  - basebandRequest
 *  - basebandDumpStatus
 */

#ifndef BASEBAND_READOUT_MANAGER_HPP
#define BASEBAND_READOUT_MANAGER_HPP

#include <condition_variable>
#include <deque>
#include <forward_list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "json.hpp"

/**
 * @class basebandRequest
 * @brief Helper structure to capture a baseband dump request.
 */
struct basebandRequest {
    /// FRB internal unique event ID
    uint64_t event_id;
    /// Starting FPGA frame of the dump
    int64_t start_fpga;
    /// Length of the dump in FPGA frames
    int64_t length_fpga;
    /// destination directory (relative to ``base_dir`` from the configuration.)
    std::string file_path;
    /// destination file (relative to ``file_path``)
    std::string file_name;
    /// Time when the request was received
    std::chrono::system_clock::time_point received = std::chrono::system_clock::now();
};


/**
 * @class basebandDumpStatus
 * @brief Helper structure to track the progress of a dump request's processing.
 */
struct basebandDumpStatus {
    /**
     * @class basebandDumpStatus::State
     * @brief State of the request
     */
    enum class State { WAITING, INPROGRESS, DONE, ERROR };

    /// The request that is being tracked
    const basebandRequest request;
    /**
     * Amount of the data to dump, in bytes. It can change once the writer
     * thread gets the actual buffer data locked.
     */
    size_t bytes_total = 0;
    /// Remaining data to write, in bytes
    size_t bytes_remaining = bytes_total;
    /// Current state of the request
    basebandDumpStatus::State state = State::WAITING;
    /// Description of the failure, when the state is ERROR
    std::string reason = "";
};


/**
* @class basebandReadoutManager
* @brief Class for managing readout state
*/
class basebandReadoutManager {
public:
    /**
     * @brief Adds a new baseband dump request to the `requests` queue and
     * notifiers threads waiting on `has_request`
     *
     * The element is `inserted_after` the `tail` of the queue, and the
     * `tail` is incremented by one to point to the new element.
     */
    void add(basebandRequest);

    /**
     * @brief Tries to get the next dump request to process.
     *
     * This is the element in the `requests` pointed to by `waiting`, unless
     * it's pointing at the `tail` (past the last element). If the element is
     * found, it is returned to the caller, and `waiting` is moved to the next
     * element in the queue.
     *
     * @return a tuple of:
     *    (1) a pointer to the `basebandDumpStatus` object if there is a request
     *        available or nullptr if there are no new requests in `requests`;
     *        and
     *
     *    (2) a pointer to the mutex that should be acquired while the (mutable)
     *        elements of the request object are being accessed (only valid if
     *        the returned object pointer is non-null).
     */
    std::tuple<basebandDumpStatus*, std::mutex*> get_next_request();

    /**
     * @brief Updates the manager's internal state of which request is currently
     * being written to the disk.
     *
     * The `requests` queue will be searched for a request with the specified
     * `event_id`, and `current` updated to point to it. The `current` pointer
     * is not visible to outside clients, and its only purpose is so the
     * internal code that iterates over the request queue knows when to acquire
     * the `current_mtx` mutex to read the an element.
     *
     * @note If the element is not found, this method will raise a runtime
     * exception.
     *
     * @return a reference to the mutex that should be acquired while the (mutable)
     * elements of the request object are being accessed.
     */
    std::mutex& set_current(uint64_t event_id);

    /// Returns a unique pointer to the copy of the event status for `event_id`,
    /// or nullptr if not known
    std::unique_ptr<basebandDumpStatus> find(uint64_t event_id);

    /// Returns a copy of all event statuses known to this readout manager
    std::vector<basebandDumpStatus> all();

private:
    /**
     * Sequence of baseband dump requests and their status. New events are
     * appended at the end. Events are never removed from the queue, they are
     * passed to worker threads by a non-owning pointer, and their state is
     * updated in-place. Worker threads having a pointer into the elements is
     * safe as long as the readout manager is guaranteed to outlive them, which
     * it will be as the manager is itself owned by the
     * `basebandApiManager`, and is created in the main process.
    */
    std::forward_list<basebandDumpStatus> requests;

    /**
     * `requests`-updating lock. Held only while elements are added to the queue
     * or internal pointers (`waiting`, `tail`) moved around.
     */
    std::mutex requests_mtx;

    /**
     * Condition used to notify that there is a valid new request in `requests`.
     */
    std::condition_variable has_request;

    using iterator = std::forward_list<basebandDumpStatus>::iterator;

    /**
     * Pointer just before the next unprocessed request (i.e., the least
     * recently added element with `state` "waiting") in `requests`.
     */
    iterator waiting = requests.before_begin();
    /// Lock to hold while accessing or updating the `waiting` element.
    std::mutex waiting_mtx;

    /**
     * Pointer the element in `requests` whose data are in the process of being
     * written (by the write thread of the readoutProcess).
     */
    iterator current = requests.before_begin();
    /// Lock to hold while accessing or updating the `current` element.
    std::mutex current_mtx;

    /**
     * Pointer to the last (most recently added) event in `requests`. We need
     * this to append new elements without traversing the queue from the
     * beginning.
     */
    iterator tail = requests.before_begin();
};

#endif /* BASEBAND_READOUT_MANAGER_HPP */
