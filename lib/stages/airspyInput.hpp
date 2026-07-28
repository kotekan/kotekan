/**
 * @file
 * @brief Contains airspy producer for kotekan.
 *  - airspyInput : public kotekan::Stage
 */

#ifndef AIRSPY_INPUT_HPP
#define AIRSPY_INPUT_HPP

#include <airspy.h>             // for airspy_transfer_t
#include <pthread.h>            // for pthread_mutex_t
#include <stdint.h>             // for uint32_t
#include <atomic>               // for atomic
#include <condition_variable>   // for condition_variable
#include <mutex>                // for mutex
#include <string>               // for string, basic_string

#include "Config.hpp"           // for Config
#include "Stage.hpp"            // for Stage
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "restServer.hpp"       // for connectionInstance
#include "json.hpp"             // for json

#define BYTES_PER_SAMPLE 2

/**
 * @class airspyInput
 * @brief Producer ``kotekan::Stage`` which streams radio data from an AirSpy SDR device
 *        into a ``Buffer``.
 *
 * This is a simple producer which initializes an AirSpy dongle (https://airspy.com)
 * in streaming mode, filling samples into the provided kotekan buffer.
 * It is agnostic about buffer framesize, simply filling in its data until a frame is full,
 * marking it so, requesting a new one, and continuing.
 * An internal sub-frame pointer is used to position data in subsequent callbacks within frames.
 * A pthread mutex is used to ensure callbacks don't clobber one another.
 *
 * Samples are read in @c AIRSPY_SAMPLE_RAW mode (real 12-bit ADC values, packed in 16-bit
 * shorts). On each completed frame the producer subtracts the 2048-count ADC offset and
 * applies an alternating-sign multiplication to shift the spectrum into the first Nyquist
 * zone, then marks the frame full.
 *
 * This producer depends on ``libairspy``.
 *
 * @par Buffers
 * @buffer out_buf The kotekan buffer which will be fed; any size.
 *     @buffer_format Array of @c shorts (real samples, DC-corrected and Nyquist-shifted)
 *     @buffer_metadata none
 *
 * @par REST endpoints (registered under @c <unique_name>/)
 *  - GET  @c /get_config  — returns current gain/freq/samplerate as JSON.
 *  - GET  @c /adcstat     — triggers a one-shot ADC stats sample (mean, RMS, rail fraction).
 *  - POST @c /set_config  — updates @c freq, @c gain_lna, @c gain_mix, @c gain_if, or @c add_lag.
 *
 * @conf   freq         Float (default 1420.0). LO tuning frequency, in MHz.
 * @conf   sample_bw    Float (default 2.5). Sample rate (samples/s), in MHz. Must be a value
 *                      reported by @c airspy_get_samplerates() for the connected device
 *                      (typically 2.5 or 10).
 * @conf   gain_lna     Int (default 5). Gain setting of the LNA, in the range 0-14.
 * @conf   gain_mix     Int (default 5). Gain setting of the mixer amplifier, from 0-15.
 * @conf   gain_if      Int (default 5). Gain setting of the IF amplifier, from 0-15.
 * @conf   biast_power  Bool (default false). Enable 4.5V DC bias on the RF input.
 * @conf   dither_disable Bool (default false). Disable the R820T tuner's fractional-N PLL
 *                      dither (reg 0x12 bit 4) for coherent multi-unit operation: eliminates
 *                      per-restart inter-unit LO drift at fractional tunes, at the cost of
 *                      deterministic fractional-N spurs (common-mode in cross-products and
 *                      absorbable into a per-unit complex g-cal). Leave at default (dither on)
 *                      for single-dongle / autocorr work where the spurs are undesirable.
 * @conf   serial       Long (default 0). Specific airspy serial-number to open; 0 = any.
 * @conf   airspy_file  String (default ""). Read from this file instead of a real device; for
 *                      offline testing. Ignored if @c serial is set.
 *
 * @warning If incoming USB transfers ever overlap, sample ordering becomes undefined.
 *
 * @author Keith Vanderlinde
 */
class airspyInput : public kotekan::Stage {
public:
    airspyInput(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~airspyInput() override;

    void main_thread() override;

    /// Initialises the airspy device. Also configures gains, LO tuning,
    /// samplerate, and sample type.
    struct airspy_device* init_device();

    /// libairspy DMA-callback trampoline; forwards to @c airspy_producer on the @c ctx instance.
    static int airspy_callback(airspy_transfer_t* transfer);

    /// Fills kotekan buffer frames from one USB transfer. See class doc.
    void airspy_producer(airspy_transfer_t* transfer);

    /// Watches the sample stream and says so, loudly, when it is not running. Runs on the
    /// stage thread for the life of the stage; see the definition for why it has to exist.
    /// ⚠️ THREAD SIGNATURE CHANGED (2026-07-27): because main_thread() no longer returns after
    /// airspy_start_rx(), a healthy airspy stage is now **3** threads, not 2 -- measured
    /// 3/3/3 across l1/l2c/l5. The old field diagnostic
    ///   cat /proc/$(pgrep -x kotekan)/task/*/comm | grep airspy_in | sort | uniq -c
    /// therefore reads 3 = healthy, 2 = half-open (the libusb poll thread is the one that is
    /// never created). Prefer kotekan_airspy_streaming from /metrics -- it states the answer
    /// instead of encoding it in a count that shifts whenever this file changes.
    void stream_watchdog();

    /// REST callbacks.
    void get_config_callback(kotekan::connectionInstance& conn);
    void set_config_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);
    void adcstat_callback(kotekan::connectionInstance& conn);

private:
    /// Kotekan buffer object which will be fed.
    Buffer* buf;
    /// Handle to the airspy device.
    struct airspy_device* a_device = nullptr;

    /// Pointer to the current frame's memory.
    unsigned char* frame_ptr = nullptr;
    /// Memory offset from the start of the current frame.
    uint32_t frame_loc;
    /// Index of the current frame.
    int frame_id;
    /// Absolute sample index since stream start, counting BOTH received samples and
    /// libairspy-dropped ones (transfer->dropped_samples). When the output buffer has a GNSS
    /// metadata pool, each completed frame is stamped with sample_seq = _frame_seq0 (its first
    /// sample's index); the F-engine propagates it (else derives its own), so this is opt-in per
    /// config and a no-op for pool-less buffers. Folding in dropped_samples makes sample_seq
    /// track TRUE time across USB drops -- a drop becomes a clean, KNOWN gap downstream (the
    /// search/tracker absolute-hop referencing handles it) instead of silent counter divergence.
    /// On a drop the partial frame is abandoned so no frame straddles the gap.
    int64_t _samples_seq = 0; ///< running true sample index (received + dropped)
    /// Stream-integrity counters, exposed via /adcstat (drop fraction = dropped/total).
    /// The producer already ACCOUNTS drops (clean sample_seq gaps); these make the rate
    /// a standing metric instead of a grep through WARN logs. Atomics: the REST thread
    /// reads them against the producer's writes.
    std::atomic<uint64_t> _samples_total{0};   ///< mirror of _samples_seq for lock-free reads
    std::atomic<uint64_t> _samples_dropped{0}; ///< cumulative libairspy FIFO drops (samples)
    std::atomic<uint64_t> _drop_events{0};     ///< number of distinct drop episodes
    int64_t _frame_seq0 = 0;  ///< true sample index of the current frame's first sample
    /// Wall-clock UTC (unix seconds) of TRUE sample 0, anchored at the first producer callback
    /// (~ms accuracy: USB delivery latency; the tiny alignment lag is ignored). 0 = not yet
    /// anchored. Served in /adcstat as "utc0_sample0" -- the absolute-time anchor for the
    /// broker's L2C CL time-assist: the CL 1.5 s code epoch is locked to GPS time, so an
    /// absolute timeline good to ~10 ms pins the CL segment (the measured CM code phase
    /// supplies the fine time below that).
    std::atomic<double> _utc0_sample0{0.0};
    /// Pending byte-count of samples to skip before next copy (for inter-device alignment).
    uint32_t lag = 0;
    /// Whether @c airspy_init() succeeded (gates @c airspy_exit() in the destructor).
    bool airspy_opened = false;
    /// Mutex serialising producer-callback invocations.
    pthread_mutex_t recv_busy;

    // options
    /// Frequency of the LO, in Hz.
    uint32_t freq;
    /// Sample rate, in samples/s. Should be 2500000 or 10000000.
    uint32_t _sample_rate;
    /// Gain of the LNA, 0-14.
    int _gain_lna;
    /// Gain of the mixer amp, 0-15.
    int _gain_mix;
    /// Gain of the IF amp, 0-15.
    int _gain_if;
    /// 4.5V DC bias on the RF input (0/1).
    int _biast_power;
    /// Disable R820T fractional-N PLL dither (0/1) for coherent multi-unit ops.
    int _dither_disable;
    /// Optional specific device serial-number; 0 means open any.
    long _airspy_sn;
    /// Optional file path to read raw samples from instead of a device.
    std::string _airspy_fn;

    /// How long /adcstat waits for the producer before failing the request. The wait MUST
    /// stay bounded: the handler runs on restServer's single libevent thread, so an
    /// unbounded wait on a non-streaming (half-open) device takes down the REST plane for
    /// every band at once -- THE WEDGE, see adcstat_callback().
    int _adcstat_timeout_ms = 3000;

    /// Stream watchdog thresholds -- see stream_watchdog().
    /// @c _stream_start_timeout_ms: how long after a SUCCESSFUL airspy_start_rx() to wait for
    /// the first sample before declaring the device half-open. Generous (USB enumeration +
    /// R820T PLL lock + the first bulk transfers); a healthy unit delivers in tens of ms.
    /// @c _stream_stall_ms: how long an already-running stream may go silent before the same
    /// verdict. Both are diagnosis-only -- the watchdog never touches the device.
    int _stream_start_timeout_ms = 5000;
    int _stream_stall_ms = 2000;

    /// ADC statistics. The REST adcstat handler requests a dump and waits on
    /// @c adcstat_cv until the producer fills @c adc{rms,mean,railfrac} on the
    /// next frame and flips @c adcstat_ready (published under @c adcstat_mutex).
    /// @c dump_adcstat is atomic so the producer's hot-path check is a
    /// well-defined lock-free read rather than a (technically UB) plain-bool
    /// race against the REST writer; @c adcstat_mutex still guards the
    /// multi-field stats publish/consume.
    std::mutex adcstat_mutex;
    std::condition_variable adcstat_cv;
    std::atomic<bool> dump_adcstat{false};
    bool adcstat_ready = false;
    float adcrms = 0;
    float adcmean = 0;
    float adcrailfrac = 0;
};

#endif
