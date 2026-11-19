/**
 * @file
 * @brief Contains the bladeRF 2.0 producer for kotekan.
 *  - bladeRFInput : public kotekan::Stage
 */

#ifndef BLADERF_INPUT_HPP
#define BLADERF_INPUT_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "restServer.hpp"      // for connectionInstance

#include "json.hpp" // for json

#include <atomic>             // for atomic
#include <condition_variable> // for condition_variable
#include <libbladeRF.h>       // for bladerf, bladerf_gain_mode
#include <mutex>              // for mutex
#include <stddef.h>           // for size_t
#include <stdint.h>           // for int8_t, uint64_t
#include <string>             // for string

/**
 * @class bladeRFInput
 * @brief Producer ``kotekan::Stage`` which streams complex 8-bit samples from one RX
 *        channel of a Nuand bladeRF 2.0 micro into a ``Buffer``.
 *
 * Above 61.44 MSPS the stage enables libbladeRF's OVERSAMPLE feature, which runs the
 * AD9361 at up to 122.88 MSPS, beyond its specification. In that mode libbladeRF fixes
 * the AD9361 analog baseband filters wide open and an analog bandwidth cannot be set, so
 * @c bandwidth is ignored and the passband is whatever that configuration gives.
 *
 * The FPGA packs each 12-bit sample into 8 bits by keeping the top 8 bits, so one count
 * here is 16 ADC counts, and the truncation adds a -0.5 count bias to I and Q.
 *
 * Samples are read with timestamps (@c BLADERF_FORMAT_SC8_Q7_META). When the timestamps
 * show that the device dropped samples, the stage writes zeros in their place, so frame
 * @c k always holds samples @c k*N to @c (k+1)*N-1 of the stream, where N is the frame
 * length in samples. A gap longer than @c max_gap_fill_ms stops kotekan.
 *
 * This producer depends on ``libbladeRF`` 2.5 or later, and needs FPGA 0.15.1 or later
 * for correct timestamps with 8-bit samples.
 *
 * @par Buffers
 * @buffer out_buf The kotekan buffer which will be fed; any size holding whole samples.
 *     @buffer_format Array of @c int8_t I/Q pairs (cint8)
 *     @buffer_metadata none
 *
 * @par REST endpoints
 * Registered under `<unique_name>/`.
 *  - GET  @c /get_config — returns the current tuning, gains and device state as JSON.
 *  - GET  @c /adcstat    — per-component mean and RMS, and the fraction of samples at
 *                          full scale, from the next complete frame, in 8-bit counts.
 *  - POST @c /set_config — updates @c freq, @c gain or @c bias_tee. Whether retuning
 *                          disturbs the OVERSAMPLE register settings is not yet known.
 *
 * @conf   serial          String (default ""). Serial number of the bladeRF to open; any
 *                         unique prefix works. Empty opens the first one found.
 * @conf   fpga_image      String (default ""). FPGA bitstream to load at startup. Empty
 *                         uses whatever is loaded, normally libbladeRF's autoload image.
 * @conf   freq            Float (default 1225.0). RX LO frequency, in MHz.
 * @conf   sample_bw       Float (default 122.88). Sample rate, in MSPS. Above 61.44 the
 *                         OVERSAMPLE feature is enabled.
 * @conf   bandwidth       Float (default: the sample rate, at most 56). Analog bandwidth,
 *                         in MHz. Ignored with OVERSAMPLE.
 * @conf   gain_mode       String (default "manual"). One of "manual", "default",
 *                         "fast_attack", "slow_attack" or "hybrid". The AGC modes change
 *                         the gain under the data, which upsets anything that integrates.
 * @conf   gain            Int (default 30). Overall RX gain in dB, in manual mode.
 * @conf   bias_tee        Bool (default false). Power an active antenna on the RX port.
 * @conf   ref_in_freq     Float (default 0). Frequency of a reference on the REF IN
 *                         port, in MHz. Above 0, the VCTCXO is phase-locked to it; the stage
 *                         stops kotekan if it does not lock at startup, and logs an ERROR if
 *                         the lock is lost later. 0 leaves the VCTCXO free running.
 * @conf   num_buffers     Int (default 64). libbladeRF sync-interface buffers.
 * @conf   buffer_size     Int (default 32768). Samples per libbladeRF buffer, and per read.
 * @conf   num_transfers   Int (default 32). USB transfers in flight; less than
 *                         @c num_buffers.
 * @conf   timeout_ms      Int (default 1000). How long a read may wait for samples.
 * @conf   max_gap_fill_ms Float (default 1000). Longest dropout filled with zeros.
 * @conf   startup_discard_ms Float (default 100). Samples discarded after enabling RX. The
 *                         first reads can hold stale samples from the previous session,
 *                         followed by a jump in the timestamps.
 * @conf   adcstat_timeout_ms Int (default 250). How long a GET @c /adcstat waits for the
 *                         next frame before failing.
 *
 * @par Metrics
 * @metric kotekan_bladerfinput_dropped_samples_total
 *         Samples the device dropped, which the stage replaced with zeros.
 * @metric kotekan_bladerfinput_discontinuities_total
 *         Number of separate dropouts.
 * @metric kotekan_bladerfinput_pll_locked
 *         1 while the VCTCXO is locked to the reference; only with @c ref_in_freq.
 *
 * Stopping kotekan when the device fails needs nothing more: a failed or timed-out read
 * stops it.
 *
 * @author Keith Vanderlinde
 */
class bladeRFInput : public kotekan::Stage {
public:
    bladeRFInput(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~bladeRFInput() override;

    void main_thread() override;

    /// REST callbacks.
    void get_config_callback(kotekan::connectionInstance& conn);
    void set_config_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);
    void adcstat_callback(kotekan::connectionInstance& conn);

private:
    /// Opens the device, loads the FPGA if asked to, and checks it. False after a FATAL_ERROR.
    bool open_device();
    /// Sets the clock, sample rate, tuning and gains. False after a FATAL_ERROR.
    bool configure_device();
    /// Reads from the device until the stage stops, writing samples and zero-fill to frames.
    void stream();

    /// Appends @p n samples (or zeros, when @p src is null) to the frame stream.
    /// False once the buffer has shut down.
    bool write_samples(const int8_t* src, size_t n);
    /// Computes and publishes ADC statistics on the current frame, if a request is waiting.
    void publish_adcstat(const int8_t* frame, size_t n);

    /// Kotekan buffer object which will be fed.
    Buffer* buf;
    /// Handle to the bladeRF.
    struct bladerf* dev = nullptr;

    /// Pointer to the current frame's memory, or null between frames.
    int8_t* frame_ptr = nullptr;
    /// Samples already written to the current frame.
    size_t frame_loc = 0;
    /// Samples per frame.
    size_t frame_samples;
    /// Index of the current frame.
    int frame_id = 0;

    // options
    std::string _serial;
    std::string _fpga_image;
    /// LO frequency, in Hz.
    uint64_t _freq;
    /// Requested and actual sample rate, in samples/s.
    uint32_t _sample_rate;
    uint32_t _actual_sample_rate = 0;
    /// Requested analog bandwidth, in Hz; unused with OVERSAMPLE.
    uint32_t _bandwidth;
    /// Whether the AD9361 runs with libbladeRF's OVERSAMPLE feature.
    bool _oversample;
    std::string _gain_mode_name;
    bladerf_gain_mode _gain_mode;
    int _gain;
    bool _bias_tee;
    /// REF IN frequency in Hz, or 0 for a free-running VCTCXO.
    uint64_t _ref_in_freq;
    int _num_buffers;
    int _buffer_size;
    int _num_transfers;
    int _timeout_ms;
    /// Longest gap filled with zeros, in samples.
    uint64_t _max_gap_fill;
    /// Samples discarded at startup.
    uint64_t _startup_discard;
    /// Longest wait in @c adcstat_callback for the producer, in ms.
    int _adcstat_timeout_ms;

    /// Serial number and versions of the opened device, for get_config.
    std::string dev_serial;
    std::string fpga_version;
    std::string fw_version;
    /// Timestamp of the first sample of the stream, for get_config.
    std::atomic<uint64_t> first_timestamp{0};

    /// ADC statistics, published like airspyInput's: the REST handler requests a dump and
    /// waits on @c adcstat_cv until the producer fills the stats on the next frame.
    std::mutex adcstat_mutex;
    std::condition_variable adcstat_cv;
    std::atomic<bool> dump_adcstat{false};
    bool adcstat_ready = false;
    float adc_mean_i = 0, adc_mean_q = 0, adc_rms_i = 0, adc_rms_q = 0, adc_railfrac = 0;
};

#endif
