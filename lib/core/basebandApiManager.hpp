/**
 * @file
 * @brief Manager for tracking baseband readout stages and handling REST API
 *  - basebandApiManager
 */

#ifndef BASEBAND_API_MANAGER_HPP
#define BASEBAND_API_MANAGER_HPP

#include "basebandReadoutManager.hpp"
#include "gpsTime.h"
#include "prometheusMetrics.hpp"
#include "restServer.hpp"

#include "json.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>


namespace kotekan {

/// Implicit conversion for constructing `nlohmann::json` from a `basebandDumpStatus`
void to_json(json& j, const basebandDumpStatus& s);


/**
 * @class basebandApiManager
 * @brief Class for receiving baseband dump requests and sending request status
 *
 * This class must be registered with a kotekan REST server instance,
 * using the @c register_with_server() function.
 *
 * This class is a singleton, and can be accessed with @c instance(). The normal
 * use is for the @c basebandReadout stage to call @get_next_request in a
 * loop, and when the result is non-null, use the returned @c basebandDumpStatus
 * to keep track of the data written so far. Once the writing of the data file
 * is completed, the ``state`` of the request should be set to ``DONE``.
 *
 * @par Metrics
 * @metric kotekan_baseband_requests_total
 *         The count of dump requests received by the REST endpoint ``/baseband``
 *
 * @author Davor Cubranic
 */
class basebandApiManager {
public:
    /**
     * @brief Returns the singleton instance of the ``basebandApiManager`` object.
     * @return A reference to the ``basebandApiManager`` object
     */
    static basebandApiManager& instance();

    /**
     * @brief Registers this class with the REST server, creating the
     *        ``/baseband`` end point
     * @param rest_server The server to register with.
     */
    void register_with_server(restServer* rest_server);

    /**
     * @brief The call back function for GET requests to `/baseband`.
     *
     * The function sends over `conn` an HTTP response with the status of all
     * baseband dumps received since this instance started running.
     *
     * The response is a JSON dictionary, with keys of unique ids of the
     * baseband events. The key's value is the status of that baseband dump, and
     * is a list with an element for each frequency index handled on the node.
     * These elements are dictionaries with the following structure:
     *   - freq_id : int
     *     Channel's frequency index
     *   - status : str
     *     Channel's dump progress (`waiting/inprogress/done/error`)
     *   - file_name : str
     *     Destination file (relative to the request's `file_path`)
     *   - length : int
     *     The total number of bytes that needs to be saved for the frequency
     *   - transferred : int
     *     The number of bytes that has been saved so far
     *   - reason : str
     *     Description of the error (only present if it occurred).
     *
     * @note This function is never called directly.
     *
     * @param conn The connection instance to send results to
     */
    void status_callback_all(connectionInstance& conn);

    /**
     * @brief The call back function for GET requests to `/baseband/:event_id`.
     *
     * The function sends over `conn` an HTTP response with the status of a
     * specified baseband dump. The response is a JSON list, with an element for
     * each frequency index handled on the node. Each element is a dictionary
     * with elements:
     *   - freq_id : int
     *     Channel's frequency index
     *   - status : str
     *     Channel dump progress (`waiting/inprogress/done/error`)
     *   - file_name : str
     *     Destination file (relative to the request's `file_path`)
     *   - length : int
     *     The total number of bytes that needs to be saved for the frequency
     *   - transferred : int
     *     The number of bytes that has been saved so far
     *   - reason : str
     *     Description of the error (only present if it occurred).
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
     * The request MUST be a JSON dictionary, with the following elements:
     *   - event_id : int
     *     Unique identifier of the event
     *   - start_unix_seconds : integer
     *     Whole part of the start time of the dump (seconds since Unix
     *     epoch) at the reference frequency channel
     *   - start_unix_nano : integer
     *     Fractional part of the start time of the dump (seconds since Unix
     *     epoch) at the reference frequency channel, in nanoseconds
     *   - duration_nano : integer
     *     Duration of the dump at the reference frequency channel, in
     *     nanoseconds
     *   - dm : number
     *     Dispersion measure, in pc cm-3.
     *   - dm_error : number
     *     Uncertainty on dispersion measure, in pc cm-3.
     *   - file_path : string
     *     the path relative to the archiver root where the baseband files should be saved
     *
     * On successful completion, sends over `conn` a JSON dictionary with keys
     * frequency indexes handled on the node, with the values another dictionary
     * with elements:
     *   - file_name : str
     *     Destination file (relative to the request's `file_path`)
     *   - start_fpga : int
     *     Starting FPGA time of the dump
     *   - length_fpga : int
     *     Length of the dump in FPGA time
     *
     * @note This function is never called directly, but as a callback
     * registered with the `restServer`.
     *
     * @param conn The connection instance to send results to
     * @param request JSON dictionary with the request data
     */
    void handle_request_callback(connectionInstance& conn, json& request);

    /**
     * @brief Register a readout stage for specified frequency
     *
     * @return a shared_ptr to the mutex used to guard access to the baseband
     * dump currently in progress.
     */
    basebandReadoutManager& register_readout_stage(const uint32_t freq_id);

private:
    /// Constructor, not used directly
    basebandApiManager();

    /// Sampling frequency (Hz)
    static constexpr double ADC_SAMPLE_RATE = 800e6;

    /// Number of samples in the inital FFT in the F-engine.
    static constexpr double FPGA_NSAMP_FFT = 2048;

    /// FPGA clock rate (Hz)
    static constexpr double FPGA_FRAME_RATE = 1. / (FPGA_PERIOD_NS * 1E-9); // =390,625
    // Can also be done as FPGA_FRAME_RATE = ADC_SAMPLE_RATE / FPGA_NSAMP_FFT

    /// Width of frequency bin, used to calculate frequency of an index, relative to FPGA_FREQ0
    static constexpr double FPGA_DELTA_FREQ = -ADC_SAMPLE_RATE / FPGA_NSAMP_FFT;

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
    static constexpr double K_DM = (ELEMENTARY_CHARGE * ELEMENTARY_CHARGE / 2 / M_PI / 4 / M_PI
                                    / EPSILON_0 / ELECTRON_MASS / C * (PARSEC / (CM * CM * CM)));

    /// Reference frequency in the L1 subsystem. (TODO verify. May be off by 1 bin.)
    static constexpr double L1_REFERERENCE_FREQ = 400e6;

    /// TODO verify. I'm assuming dm_error is 1-sigma.
    static constexpr double N_DM_ERROR_TOL = 3;

    /// convenience wrapper for a pair of starting FPGA frame and length of the dump in FPGA frames
    struct basebandSlice {
        int64_t start_fpga;
        int64_t length_fpga;
    };

    /// Utility function that adjusts the trigger times given for the reference
    /// frequency to those for the frequency `freq_id`
    static basebandSlice translate_trigger(int64_t fpga_time0, int64_t fpga_width, const double dm,
                                           const double dm_error, const uint32_t freq_id,
                                           const double ref_freq_hz = L1_REFERERENCE_FREQ);

    void status_callback_single_event(connectionInstance& conn);

    /**
     * @class basebandReadoutRegistry
     * @brief encapsulation of a lock-protected map to registered readout stage
     */
    class basebandReadoutRegistry {
    public:
        using iterator = std::map<uint32_t, basebandReadoutManager>::iterator;
        iterator begin() noexcept;
        iterator end() noexcept;
        basebandReadoutManager& operator[](const uint32_t& key);

    private:
        std::mutex map_lock;
        std::map<uint32_t, basebandReadoutManager> readout_map;
    };

    /// Map of registered readout stages, indexed by `freq_id`
    basebandReadoutRegistry readout_registry;

    prometheusMetrics& metrics;
    uint32_t request_count = 0;
};

} // namespace kotekan

#endif /* BASEBAND_API_MANAGER_HPP */
