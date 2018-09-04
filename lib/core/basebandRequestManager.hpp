/**
 * @file
 * @brief Manager for tracking baseband buffer writeout requests
 *  - airspyInput : public KotekanProcess
 */

#ifndef BASEBAND_REQUEST_MANAGER_HPP
#define BASEBAND_REQUEST_MANAGER_HPP

#include "json.hpp"
#include "gpsTime.h"
#include "restServer.hpp"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>


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
void to_json(json& j, const basebandRequest& r);

struct basebandSlice {
    /// Starting FPGA frame of the dump
    int64_t start_fpga;
    /// Length of the dump in FPGA frames
    int64_t length_fpga;
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
void to_json(json& j, const basebandDumpStatus& s);


/**
 * @class basebandRequestManager
 * @brief Class for receiving baseband dump requests and sending request status
 *
 * This class must be registered with a kotekan REST server instance,
 * using the @c register_with_server() function.
 *
 * This class is a singleton, and can be accessed with @c instance(). The normal
 * use is for the @c basebandReadout process to call @get_next_request in a
 * loop, and when the result is non-null, use the returned @c basebandDumpStatus
 * to keep track of the data written so far. Once the writing of the data file
 * is completed, the ``state`` of the request should be set to ``DONE``.
 *
 * @author Davor Cubranic
 */
class basebandRequestManager {
public:
    /**
     * @brief Returns the singleton instance of the ``basebandRequestManager`` object.
     * @return A pointer to the ``basebandRequestManager`` object
     */
    static basebandRequestManager& instance();

    /**
     * @brief Registers this class with the REST server, creating the
     *        ``/baseband`` end point
     * @param rest_server The server to register with.
     */
    void register_with_server(restServer * rest_server);

    /**
     * @brief The call back function for GET requests to `/baseband`.
     *
     * The function sends an HTTP response with the status of all
     * baseband dumps received since this instance started running.
     *
     * @note This function is never called directly.
     *
     * @param conn The connection instance to send results to
     */
    void status_callback_all(connectionInstance& conn);

    /**
     * @brief The call back function for GET requests to `/baseband/:event_id`.
     *
     * The function sends an HTTP response with the status of a specified
     * baseband dump.
     *
     * @note This function is never called directly.
     *
     * @param event_id unique identifier for the event
     * @param conn The connection instance to send results to
     */
    void status_callback_single_event(const uint64_t event_id, connectionInstance& conn);

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
     * @brief Register a readout process for specified frequency
     *
     * @return a shared_ptr to the mutex used to guard access to the baseband
     * dump currently in progress.
     */
    std::shared_ptr<std::mutex> register_readout_process(const uint32_t freq_id);

    /**
     * @brief Tries to get the next dump request to process.
     *
     * @return a shared_ptr to the `basebandDumpStatus` object if there is a
     * request available for the readout process handling frequency `freq_id`,
     * or nullptr if the request queue is empty.
     */
    std::shared_ptr<basebandDumpStatus> get_next_request(const uint32_t freq_id);

private:
    /// Constructor, not used directly
    basebandRequestManager() = default;

    /// Sampling frequency (Hz)
    static constexpr double ADC_SAMPLE_RATE = 800e6;

    /// Number of samples in the inital FFT in the F-engine.
    static constexpr double FPGA_NSAMP_FFT = 2048;

    /// FPGA clock rate (Hz)
    static constexpr double FPGA_FRAME_RATE = 1. / (FPGA_PERIOD_NS * 1E-9); // =390,625
    // Can also be done as FPGA_FRAME_RATE = ADC_SAMPLE_RATE / FPGA_NSAMP_FFT

    /// Reference frequency in the L0 subsystem
    static constexpr double FPGA_FREQ0 = 800e6;

    /// Width of frequency bin, used to calculate frequency of an index, relative to FPGA_FREQ0
    static constexpr double FPGA_DELTA_FREQ  = - ADC_SAMPLE_RATE / FPGA_NSAMP_FFT;

    /// Physical constant: elementary charge (C)
    static constexpr double ELEMENTARY_CHARGE = 1.6021766208e-19;
    /// Physical constant: the electric constant (vacuum permittivity, F/m)
    static constexpr double EPSILON_0 = 8.854187817620389e-12;
    /// Physical constant: mass of electron (kg)
    static constexpr double ELECTRON_MASS = 9.10938356e-31;
    /// Physical constant: speed of light in vacuum (m/s)
    static constexpr double C = 299792458.0;

    /// Unit definition: parsec (m)
    static constexpr double PARSEC = 3.0856775813057292e+16;
    /// Unit definition: Centimeters (m)
    static constexpr double CM = 1e-2;

    /// Physical constant: Dispersion measure, in Hz**2 s / (pc cm^-3)
    static constexpr double K_DM = (
        ELEMENTARY_CHARGE * ELEMENTARY_CHARGE
        / 2 / M_PI
        / 4 / M_PI / EPSILON_0
        / ELECTRON_MASS / C
        * (PARSEC / (CM * CM * CM)));

    /// Reference frequency in the L1 subsystem. (TODO verify. May be off by 1 bin.)
    static constexpr double L1_REFERERENCE_FREQ = 400e6;

    /// TODO verify. I'm assuming dm_error is 1-sigma.
    static constexpr double N_DM_ERROR_TOL = 3;

    /// Utility function that adjusts the trigger times given for the reference
    /// frequency to those for the frequency `freq_id`
    static basebandSlice translate_trigger(int64_t fpga_time0, int64_t fpga_width,
                                           const double dm, const double dm_error,
                                           const int64_t freq_id,
                                           const double ref_freq_hz=L1_REFERERENCE_FREQ);

    void status_callback_single_event(connectionInstance& conn);

    /**
    * @class basebandReadoutRegistryEntry
    * @brief Helper structure to track registered basebandReadout processes
    */
    struct basebandReadoutRegistryEntry {
        /// request updating lock
        std::mutex requests_lock;

        /// new request notification
        std::condition_variable requests_cv;

        /// Queue of unprocessed baseband requests for this frequency
        std::deque<basebandRequest> request_queue;

        /// Status of the baseband dump currently in progress
        std::shared_ptr<basebandDumpStatus> current_status = nullptr;

        /// Lock to update the current basebandDumpStatus object
        std::shared_ptr<std::mutex> current_lock = std::make_shared<std::mutex>();

        /// Queue of completed baseband requests for this frequency
        std::vector<basebandDumpStatus> processing;
    };

    /**
     * @class basebandReadoutRegistry
     * @brief encapsulation of a lock-protected map to registered readout processes
     */
    class basebandReadoutRegistry {
    public:
        using iterator = std::map<uint32_t, basebandReadoutRegistryEntry>::iterator;
        iterator begin() noexcept;
        iterator end() noexcept;
        basebandReadoutRegistryEntry& operator[]( const uint32_t& key );

    private:
        std::mutex map_lock;
        std::map<uint32_t, basebandReadoutRegistryEntry> readout_map;
    };

    /// Map of registered readout processes, indexed by `freq_id`
    basebandReadoutRegistry readout_registry;
};

#endif /* BASEBAND_REQUEST_MANAGER_HPP */
