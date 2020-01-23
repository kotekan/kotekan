/**
 * @file
 * @brief Manager for tracking baseband buffer writeout requests
 *  - basebandReadoutManager
 *  - basebandRequest
 *  - basebandDumpStatus
 */

#ifndef BASEBAND_READOUT_MANAGER_HPP
#define BASEBAND_READOUT_MANAGER_HPP

#include "SynchronizedQueue.hpp" // for SynchronizedQueue

#include "gsl-lite.hpp" // for span

#include <chrono>       // for system_clock, system_clock::time_point
#include <forward_list> // for forward_list
#include <functional>   // for reference_wrapper
#include <memory>       // for unique_ptr, shared_ptr, allocator
#include <mutex>        // for mutex
#include <stdint.h>     // for uint64_t, int64_t, uint32_t, uint8_t
#include <string>       // for string
#include <time.h>       // for size_t, timespec
#include <utility>      // for pair
#include <vector>       // for vector

namespace kotekan {

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
    /// Time when the processing started (null if ``state`` is still WAITING)
    std::shared_ptr<std::chrono::system_clock::time_point> started = nullptr;
    /// Time when the processing finished (null if ``state`` is not DONE or ERROR)
    std::shared_ptr<std::chrono::system_clock::time_point> finished = nullptr;
};


/**
 * @struct basebandDumpData
 * @brief A container for baseband data and metadata.
 *
 * @note This class does not own the underlying data buffer, but provides a view
 *       (i.e., a `gsl::span`) to it. Users are responsible for managing the
 *       memory storage.
 *
 * @author Kiyoshi Masui
 */
struct basebandDumpData {
    /// Indicates the reason why data could not be read, if it is not `Ok`
    enum class Status { Ok, TooLong, Late, ReserveFailed, Cancelled };

    /// Constructor used to indicate error
    basebandDumpData(Status);
    /// Initialize the container with all parameters but does not fill in the data.
    basebandDumpData(uint64_t event_id_, uint32_t freq_id_, uint32_t num_elements_,
                     int64_t data_start_fpga_, uint64_t data_length_fpga_,
                     timespec data_start_ctime_, const gsl::span<uint8_t>&);

    //@{
    /// Metadata.
    const uint64_t event_id;
    const uint32_t freq_id;
    const uint32_t num_elements;
    const int64_t data_start_fpga;
    const uint64_t data_length_fpga;
    const timespec data_start_ctime;
    //@}
    /// Data access. Array has length `num_elements * data_length_fpga` and is aligned on a 16-byte
    /// boundary.
    const gsl::span<uint8_t> data;
    /// Original size of the write reservation, regardless of the boundary
    const size_t reservation_length;

    /// Status::Ok if the `data` is valid, or the reason why it was not read out
    const Status status;

    /**
     * @brief Narrows the span to align it on a 16 byte boundary.
     *
     * Helper for basebandDumpData constructor.
     *
     * @note The original span should be oversized by at least 15 bytes, since that's how much can
     * be lost by aligning.
     */
    static gsl::span<uint8_t> span_from_length_aligned(const gsl::span<uint8_t>&);
};


/**
 * @class basebandReadoutManager
 * @brief Class for managing readout state
 */
class basebandReadoutManager {
public:
    // convenience type alias for keeping a status and mutex required for modifying it
    using requestStatusMutex = std::pair<basebandDumpStatus&, std::mutex&>;

    // convenience type alias for a request ready to be written to a file, together with its
    // read-out data
    using ReadyRequest = std::pair<basebandDumpStatus&, basebandDumpData>;

    // convenience type alias for a ReadyRequest and mutex required for modifying it
    using ReadyRequestMutex = std::pair<ReadyRequest, std::mutex&>;

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
     * @return if there is a request available, a `unique_ptr` to the pair of:
     *    (1) a reference to the `basebandDumpStatus` object; and
     *
     *    (2) a reference to the mutex that should be acquired while the (mutable)
     *        elements of the request object are being accessed.
     * otherwise, a nullptr
     */
    std::unique_ptr<basebandReadoutManager::requestStatusMutex> get_next_waiting_request();

    /**
     * @brief Adds a baseband dump request to the `ready` queue and
     * notifier threads waiting on `has_next_ready_request`
     */
    void ready(const ReadyRequest&);

    /**
     * @brief Interrupts all threads blocked on "waiting" and "ready" dump request queues
     */
    void stop();

    /**
     * @brief Tries to get the next dump request whose data is ready for writing
     *
     * This is the element in the `requests` pointed to by `waiting`, unless
     * it's pointing at the `tail` (past the last element). If the element is
     * found, it is returned to the caller, and `waiting` is moved to the next
     * element in the queue.
     *
     * The `requests` queue following the last processed event (i.e., the
     * `current` position) will be searched for a request in the `INPROGRESS`
     * state. If the element is found, it is returned to the caller, and
     * `current` updated to point to it. If the element is not found, this
     * method will raise a runtime exception.
     *
     * @note The `current` pointer is not visible to outside clients, and its
     * only purpose is so the internal code that iterates over the request queue
     * knows when to acquire the `current_mtx` mutex to read the an element.
     *
     * @return a pair of:
     *    (1) a reference to the `basebandDumpStatus` object; and
     *
     *    (2) a reference to the mutex that should be acquired while the (mutable)
     *        elements of the request object are being accessed.
     *
     */
    std::unique_ptr<ReadyRequestMutex> get_next_ready_request();

    /// Returns a unique pointer to the copy of the event status for `event_id`,
    /// or nullptr if not known
    std::unique_ptr<basebandDumpStatus> find(uint64_t event_id);

    /// Returns a copy of all event statuses known to this readout manager
    std::vector<basebandDumpStatus> all();

private:
    /**
     * Sequence of baseband dump requests and their status. New events are
     * appended at the head. Events are never removed from the queue, they are
     * passed to worker threads by a non-owning pointer, and their state is
     * updated in-place. Worker threads' having a pointer into the elements is
     * safe as long as the readout manager is guaranteed to outlive them, which
     * it will be as the manager is itself owned by the
     * `basebandApiManager`, and is created in the main thread.
     */
    std::forward_list<basebandDumpStatus> requests;

    /**
     * `requests`-updating lock. Held only while elements are added to the queue
     * or internal pointers (`readout_current`, `writeout_current`) moved around.
     */
    std::mutex requests_mtx;

    /// requests that have been received by the baseband API, but that the readout thread hasn't
    /// processed yet
    SynchronizedQueue<std::reference_wrapper<basebandDumpStatus>> waiting_queue;

    /// requests that have been processed by the readout thread, and are now ready to be written out
    SynchronizedQueue<ReadyRequest> ready_queue;

    /**
     * Pointer to the element in `requests` that the readout thread is currently working on
     */
    basebandDumpStatus* readout_current = nullptr;
    /// Lock to hold while accessing or updating the `readout_current` element.
    std::mutex readout_mtx;

    /**
     * Pointer to the element in `requests` that the write thread is currently working on
     */
    basebandDumpStatus* writeout_current = nullptr;
    /// Lock to hold while accessing or updating the `writeout_current` element.
    std::mutex writeout_mtx;
};

} // namespace kotekan

#endif /* BASEBAND_READOUT_MANAGER_HPP */
