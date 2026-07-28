#include "airspyInput.hpp"
#include <chrono>

#include "Config.hpp"           // for Config
#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata, metadata_is_gnss_chan (opt-in
                                // sample_seq)
#include "NDArray.hpp"           // for GenericNDArray
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp"   // for make_input_desc
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for ERROR, INFO, DEBUG, FATAL_ERROR
#include "prometheusMetrics.hpp" // for Metrics, Gauge, Counter
#include "restServer.hpp"        // for connectionInstance, HTTP_RESPONSE, restServer

#include "fmt.hpp" // for compile_string_to_view, format

#include <algorithm>          // for min
#include <cmath>              // for sqrt
#include <condition_variable> // for condition_variable
#include <fcntl.h>            // for open, O_RDWR
#include <functional>         // for bind, function, _1, _2
#include <json.hpp>           // for basic_json, json
#include <memory>             // for shared_ptr
#include <mutex>              // for mutex, unique_lock, lock_guard
#include <stdint.h>           // for uint32_t, uint8_t
#include <stdlib.h>           // for free, abs, malloc
#include <string.h>           // for memcpy
#include <thread>             // for this_thread::sleep_for
#include <unistd.h>           // for size_t, close, usleep

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(airspyInput);

airspyInput::airspyInput(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&airspyInput::main_thread, this)) {

    buf = get_buffer("out_buf");
    buf->register_producer(unique_name);

    // The DC-subtract / Nyquist-flip pass in main_thread walks samples two
    // at a time (fr[i], fr[i+1]); an odd sample count would read one past
    // the frame. Sample count = frame_size / BYTES_PER_SAMPLE, so the frame
    // must be a multiple of 2*BYTES_PER_SAMPLE.
    if ((buf->frame_size / BYTES_PER_SAMPLE) % 2 != 0) {
        FATAL_ERROR("airspyInput: out_buf frame_size ({:d} B) must yield an even sample "
                    "count (multiple of {:d} B); got {:d} samples.",
                    buf->frame_size, 2 * BYTES_PER_SAMPLE, buf->frame_size / BYTES_PER_SAMPLE);
        return;
    }

    // Buffer carries int16 1-D samples. ensure_frame_desc, NOT require_frame_desc:
    // this stage can ORIGINATE the descriptor (n_samples follows from frame_size), so
    // it works whether or not config declares the buffer. Where config declares it
    // (kotekan_buffer: ndarray, the CHORD convention) the two are reconciled and a
    // mismatch is fatal; where it does not (the GNSS node's `standard` buffers) the
    // descriptor is attached here, as it was before the chord merge. `require` makes a
    // missing config declaration fatal, which took the whole GNSS node down on the
    // first launch after the merge (2026-07-28). The even-sample-count constraint
    // above is enforced separately: the descriptor cannot express it.
    buf->ensure_frame_desc(kotekan_airspy::make_input_desc(buf->frame_size / BYTES_PER_SAMPLE));

    freq = config.get_default<float>(unique_name, "freq", 1420) * 1e6;             // MHz
    _sample_rate = config.get_default<float>(unique_name, "sample_bw", 2.5) * 1e6; // MSPS
    _gain_lna = config.get_default<int>(unique_name, "gain_lna", 5);               // 0-14
    _gain_if = config.get_default<int>(unique_name, "gain_if", 5);                 // 0-15
    _gain_mix = config.get_default<int>(unique_name, "gain_mix", 5);               // 0-15
    _biast_power = config.get_default<bool>(unique_name, "biast_power", false) ? 1 : 0;
    _dither_disable = config.get_default<bool>(unique_name, "dither_disable", false) ? 1 : 0;

    _airspy_sn = config.get_default<long>(unique_name, "serial", 0);
    _airspy_fn = config.get_default<std::string>(unique_name, "airspy_file", "");
    // 250 ms, not 3000 (2026-07-27). The bounded wait fixed THE WEDGE -- an unbounded
    // cv.wait() here on restServer's single libevent thread killed every endpoint on every
    // band when one airspy came up half-open. But 3 s is still far too long for an endpoint
    // the VIEWER POLLS EVERY 1.5 s: a dead producer would block the whole REST plane ~2/3 of
    // the time, which is a wedge in all but name. A healthy device fills the buffer in one
    // frame period (2.5-10 ms), so 250 ms is ~25x margin and 12x less worst-case blocking.
    _adcstat_timeout_ms = config.get_default<int>(unique_name, "adcstat_timeout_ms", 250);
    _stream_start_timeout_ms =
        config.get_default<int>(unique_name, "stream_start_timeout_ms", 5000);
    _stream_stall_ms = config.get_default<int>(unique_name, "stream_stall_ms", 2000);
}

airspyInput::~airspyInput() {
    if (a_device != nullptr) {
        airspy_stop_rx(a_device);
        airspy_close(a_device);
    }
    if (airspy_opened)
        airspy_exit();
    // Wake any REST callback waiting on a pending adcstat request.
    adcstat_cv.notify_all();
}

void airspyInput::get_config_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply;
    reply["lna_gain"] = _gain_lna;
    reply["mix_gain"] = _gain_mix;
    reply["if_gain"] = _gain_if;
    reply["samplerate"] = _sample_rate;
    reply["freq"] = freq;
    reply["airspy_sn"] = _airspy_sn;
    conn.send_json_reply(reply);
}

void airspyInput::adcstat_callback(kotekan::connectionInstance& conn) {
    std::unique_lock<std::mutex> lock(adcstat_mutex);
    dump_adcstat = true;
    // BOUNDED wait -- this handler runs on restServer's SINGLE libevent thread, so blocking
    // here does not fail one request, it kills the ENTIRE REST plane for every band.
    //
    // THE WEDGE, diagnosed 2026-07-26 (3rd sample, data/wedge/wedge_20260726_224903): a
    // marginal USB link left /l1_airspy_in HALF-OPEN -- device claimed, libusb poll thread
    // never created, so the producer never runs and adcstat_ready never flips. The launcher's
    // own startup "front end" probe hits this endpoint, the unbounded wait() never returned,
    // and the whole node went dark -- INCLUDING the two airspys that were streaming perfectly.
    // Symptom: kotekan alive, no errors logged, every endpoint hanging; SIGTERM will not
    // clear it (shutdown cannot sequence through a dead REST plane) -- it needs SIGKILL.
    //
    // A frame arrives every few ms on a healthy device, so this timeout only ever expires
    // when the device is not delivering at all. Failing ONE request loudly beats hanging
    // every request silently: the cost of a false timeout is a retry, the cost of no timeout
    // is the node.
    const bool got = adcstat_cv.wait_for(lock, std::chrono::milliseconds(_adcstat_timeout_ms),
                                         [this] { return adcstat_ready || stop_thread; });
    if (!got) {
        dump_adcstat = false; // withdraw the request; the producer must not service a corpse
        lock.unlock();
        ERROR("/{:s}: adcstat timed out after {:d} ms -- no frames from this airspy. The "
              "device is enumerated but not streaming (half-open); check the USB link.",
              unique_name, _adcstat_timeout_ms);
        conn.send_error("adcstat timed out: airspy not delivering frames (half-open device)",
                        kotekan::HTTP_RESPONSE::REQUEST_FAILED);
        return;
    }
    if (stop_thread) {
        lock.unlock();
        conn.send_error("Stage shutting down.", kotekan::HTTP_RESPONSE::INTERNAL_ERROR);
        return;
    }

    nlohmann::json reply;
    reply["rms"] = adcrms;
    reply["mean"] = adcmean;
    reply["railfrac"] = adcrailfrac;
    // Absolute-time anchor for the CL time-assist (0.0 until the first producer callback).
    reply["utc0_sample0"] = _utc0_sample0.load(std::memory_order_relaxed);
    // Stream integrity: cumulative REAL samples (delivered + accounted gaps), cumulative
    // libairspy FIFO drops, and drop episodes. Fraction = samples_dropped / samples_total.
    // The producer accounts every drop as a clean sample_seq gap, so these are exact.
    reply["samples_total"] = _samples_total.load(std::memory_order_relaxed);
    reply["samples_dropped"] = _samples_dropped.load(std::memory_order_relaxed);
    reply["drop_events"] = _drop_events.load(std::memory_order_relaxed);
    adcstat_ready = false;
    lock.unlock();

    conn.send_json_reply(reply);
}

void airspyInput::set_config_callback(kotekan::connectionInstance& conn,
                                      nlohmann::json& json_request) {
    int err;
    bool success = false;

    // Each setting is optional; missing keys throw and are caught silently.
    try {
        freq = ((float)json_request["freq"]) * 1e6;
        INFO("Updating airspy LO frequency to {:d}", freq);
        err = airspy_set_freq(a_device, freq);
        if (err != AIRSPY_SUCCESS)
            ERROR("airspy_set_freq() failed: {:s} ({:d})",
                  airspy_error_name((enum airspy_error)err), err);
        else
            success = true;
    } catch (...) {
    }
    try {
        _gain_lna = json_request["gain_lna"];
        INFO("Updating airspy LNA gain to {:d}", _gain_lna);
        err = airspy_set_lna_gain(a_device, _gain_lna);
        if (err != AIRSPY_SUCCESS)
            ERROR("airspy_set_lna_gain() failed: {:s} ({:d})",
                  airspy_error_name((enum airspy_error)err), err);
        else
            success = true;
    } catch (...) {
    }
    try {
        _gain_mix = json_request["gain_mix"];
        INFO("Updating airspy mixer gain to {:d}", _gain_mix);
        err = airspy_set_mixer_gain(a_device, _gain_mix);
        if (err != AIRSPY_SUCCESS)
            ERROR("airspy_set_mixer_gain() failed: {:s} ({:d})",
                  airspy_error_name((enum airspy_error)err), err);
        else
            success = true;
    } catch (...) {
    }
    try {
        _gain_if = json_request["gain_if"];
        INFO("Updating airspy IF gain to {:d}", _gain_if);
        err = airspy_set_vga_gain(a_device, _gain_if);
        if (err != AIRSPY_SUCCESS)
            ERROR("airspy_set_vga_gain() failed: {:s} ({:d})",
                  airspy_error_name((enum airspy_error)err), err);
        else
            success = true;
    } catch (...) {
    }
    try {
        int add_lag = json_request["add_lag"];
        INFO("Updating airspy lag by {:d}", add_lag);
        pthread_mutex_lock(&recv_busy);
        lag += add_lag * BYTES_PER_SAMPLE;
        pthread_mutex_unlock(&recv_busy);
        success = true;
    } catch (...) {
    }

    if (success) {
        usleep(10000);
        conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
    } else {
        conn.send_error("Couldn't parse airspy rx parameters.",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
    }
}

void airspyInput::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer& rest_server = kotekan::restServer::instance();
    rest_server.register_post_callback(unique_name + "/set_config",
                                       std::bind(&airspyInput::set_config_callback, this, _1, _2));
    rest_server.register_get_callback(unique_name + "/adcstat",
                                      std::bind(&airspyInput::adcstat_callback, this, _1));
    rest_server.register_get_callback(unique_name + "/get_config",
                                      std::bind(&airspyInput::get_config_callback, this, _1));

    frame_id = 0;
    frame_loc = 0;
    recv_busy = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;

    int err = airspy_init();
    if (err != AIRSPY_SUCCESS) {
        INFO("airspy_init() failed: {:s} ({:d}); proceeding without it.",
             airspy_error_name((enum airspy_error)err), err);
        airspy_opened = false;
    } else {
        airspy_opened = true;
    }

    // init_device() already FATAL_ERROR'd with the specific reason on any
    // failure; just exit main_thread quietly here.
    a_device = init_device();
    if (a_device == nullptr)
        return;

    err = airspy_start_rx(a_device, airspy_callback, static_cast<void*>(this));
    if (err != AIRSPY_SUCCESS) {
        FATAL_ERROR("airspy_start_rx() failed: {:s} ({:d})",
                    airspy_error_name((enum airspy_error)err), err);
        return;
    }

    // start_rx() returning SUCCESS does NOT mean the device is streaming -- see
    // stream_watchdog(). This call does not return until the stage stops.
    stream_watchdog();
}

/**
 * Fix (a) of THE WEDGE post-mortem (2026-07-27): "a half-open device should emit an ERROR at
 * open time, not be inferred from a thread count."
 *
 * THE GAP THIS FILLS. Everything above returns SUCCESS on a half-open device: libusb claims
 * the interface, every control transfer (samplerate, gains, PLL, board id, serial) is answered
 * by firmware, and airspy_start_rx() returns AIRSPY_SUCCESS -- but the transfer thread never
 * delivers a callback, so airspy_producer() never runs. Historically main_thread() RETURNED
 * right here, which meant that on the one failure that matters there was nothing left running
 * to notice it: no frames, no error, no log line, and a band that is simply absent. The only
 * evidence was indirect (one thread instead of two, /adcstat hanging) and the outage that
 * taught us this ran for hours behind an "=== 3-band node running ===" banner.
 *
 * Measured trigger (2026-07-26, data/wedge/wedge_20260726_224903): a marginal USB link. The
 * SAME link failed loudly when it let go DURING start_rx (AIRSPY_ERROR_LIBUSB, kotekan exits --
 * correct) and silently when it let go just after. Only the silent branch needed fixing.
 *
 * WHAT IT DOES. Polls the producer's own sample counter -- no device calls, so a sick USB link
 * cannot be made sicker by the diagnosis -- and publishes three things to /metrics:
 *
 *   kotekan_airspy_streaming{stage_name}             1 = delivering, 0 = not
 *   kotekan_airspy_samples_total{stage_name}         cumulative true samples
 *   kotekan_airspy_samples_dropped_total{stage_name} cumulative libairspy FIFO drops
 *
 * /metrics is deliberately the vehicle: it is served without touching adcstat_mutex, so it
 * answers even for a device whose /adcstat is failing -- the endpoint that wedged the node is
 * not the endpoint you have to poll to find out that it wedged. `streaming` is also what
 * run_3band.sh gates its startup banner on, so a half-open unit now reports DEGRADED at launch
 * instead of hours later.
 *
 * The stall branch (stream was up, then stopped) fires for a device that dies mid-run AND for
 * a producer blocked in wait_for_empty_frame() with a full input buffer -- distinct causes with
 * the same observable, so the message names both rather than guessing.
 */
void airspyInput::stream_watchdog() {
    auto& m_streaming = Metrics::instance().add_gauge("kotekan_airspy_streaming", unique_name);
    auto& m_samples = Metrics::instance().add_counter("kotekan_airspy_samples_total", unique_name);
    auto& m_dropped =
        Metrics::instance().add_counter("kotekan_airspy_samples_dropped_total", unique_name);

    const auto tick = std::chrono::milliseconds(100);
    const double report_every_ms = 30000.0; // keep saying it -- an operator may grep in at any time

    auto t_progress = std::chrono::steady_clock::now();
    uint64_t last_total = 0, last_dropped = 0;
    bool ever_streamed = false;
    double next_report_ms = (double)_stream_start_timeout_ms;

    while (!stop_thread) {
        std::this_thread::sleep_for(tick);

        const uint64_t n_total = _samples_total.load(std::memory_order_relaxed);
        const uint64_t n_dropped = _samples_dropped.load(std::memory_order_relaxed);
        m_samples.inc(n_total - last_total);
        m_dropped.inc(n_dropped - last_dropped);
        last_dropped = n_dropped;

        const double silent_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - t_progress)
                                     .count();

        if (n_total != last_total) { // the producer ran since the last tick
            last_total = n_total;
            t_progress = std::chrono::steady_clock::now();
            m_streaming.set(1);
            if (!ever_streamed) {
                ever_streamed = true;
                // WARN, not INFO, and it is not a warning: the node runs at log_level "warn",
                // so INFO never reaches the log that actually gets archived. This is the
                // POSITIVE CONTROL for the silent failure above -- three lines per run that
                // say each front end really did start delivering, and how long it took. Its
                // absence is the evidence when a post-mortem asks "did L1 ever stream?".
                WARN("streaming CONFIRMED -- first samples {:.0f} ms after start_rx.", silent_ms);
            } else if (next_report_ms > (double)_stream_stall_ms) {
                // we had complained; say that it came back, or the log strands the operator
                // at the last ERROR with no idea it self-healed
                ERROR("stream RESUMED after {:.1f} s silent.", silent_ms / 1000.0);
            }
            next_report_ms = (double)_stream_stall_ms;
            continue;
        }

        const double limit = ever_streamed ? (double)_stream_stall_ms
                                           : (double)_stream_start_timeout_ms;
        if (silent_ms < limit)
            continue;

        m_streaming.set(0);
        if (silent_ms < next_report_ms)
            continue;
        next_report_ms = silent_ms + report_every_ms;

        if (!ever_streamed)
            ERROR("HALF-OPEN DEVICE -- airspy_start_rx() succeeded {:.1f} s ago but the device "
                  "has never delivered a single sample (serial {:#016x}). This band is "
                  "producing NOTHING. Check the USB link/cable; the last one to do this was a "
                  "marginal micro-B adapter.",
                  silent_ms / 1000.0, (unsigned long)_airspy_sn);
        else
            ERROR("stream STALLED -- no samples for {:.1f} s ({:d} delivered so far). Either "
                  "the device stopped (USB link) or the producer is blocked on a full input "
                  "buffer (check kotekan_valve_dropped_frames_total and the 'behind realtime' "
                  "warnings). This band is producing NOTHING meanwhile.",
                  silent_ms / 1000.0, n_total);
    }
}

int airspyInput::airspy_callback(airspy_transfer_t* transfer) {
    airspyInput* proc = static_cast<airspyInput*>(transfer->ctx);
    proc->airspy_producer(transfer);
    return 0;
}

void airspyInput::airspy_producer(airspy_transfer_t* transfer) {
    // Serialise overlapping callbacks; libairspy can in principle deliver them concurrently.
    pthread_mutex_lock(&recv_busy);

    // One-shot absolute-time anchor (see the header doc): this transfer's samples span the last
    // sample_count / real_rate seconds ending ~now (real stream = 2x the complex device rate;
    // USB delivery latency ~1-3 ms). Refer back to TRUE sample 0 so downstream can turn any
    // sample_seq/hop into wall-clock UTC: utc0_sample0 + seq / real_rate.
    if (_utc0_sample0.load(std::memory_order_relaxed) == 0.0) {
        const double now = std::chrono::duration<double>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
        const double real_rate = 2.0 * (double)_sample_rate;
        const double t_first = now - (double)transfer->sample_count / real_rate;
        _utc0_sample0.store(
            t_first
            - (double)(_samples_seq + (int64_t)transfer->dropped_samples) / real_rate);
    }

    void* in = transfer->samples;
    size_t bt = transfer->sample_count * BYTES_PER_SAMPLE;

    // Account for samples libairspy dropped (FIFO overflow) just before this transfer's data.
    // The gap sits between the current partial frame and this transfer's samples, so abandon the
    // partial frame (it would straddle the gap) and advance the true sample index past the gap.
    // sample_seq then tracks TRUE time -> the drop is a clean, KNOWN jump downstream (handled by
    // the absolute-hop referencing) rather than the silent counter divergence that randomises
    // code phase. (Only meaningful when the buffer carries GNSS metadata; harmless otherwise.)
    if (transfer->dropped_samples > 0) {
        _samples_seq += (int64_t)transfer->dropped_samples;
        _samples_dropped.fetch_add((uint64_t)transfer->dropped_samples,
                                   std::memory_order_relaxed);
        _drop_events.fetch_add(1, std::memory_order_relaxed);
        if (frame_loc != 0) { // discard the straddling partial frame; refill it post-gap
            frame_loc = 0;
            _frame_seq0 = _samples_seq;
        }
        WARN("airspy: {:d} samples dropped -- partial frame discarded, sample_seq gap inserted "
             "(pipeline behind realtime)",
             transfer->dropped_samples);
    }
    while (bt > 0) {
        // Skip past any pending alignment lag.
        if (lag > 0) {
            if (lag >= bt) {
                lag -= bt;
                bt = 0;
                continue;
            }
            bt -= lag;
            in = (void*)((char*)in + lag);
            lag = 0;
        }

        // Fetch a fresh frame only when we don't already hold one. frame_ptr is nulled on a
        // completed frame; a drop-discard keeps it (frame_loc reset to 0) so the held frame is
        // refilled from the top with the post-gap sample index.
        if (frame_loc == 0 && frame_ptr == nullptr) {
            DEBUG("Airspy waiting for frame_id {:d}", frame_id);
            auto _w0 = std::chrono::steady_clock::now();
            frame_ptr = (unsigned char*)buf->wait_for_empty_frame(unique_name, frame_id);
            if (frame_ptr == nullptr)
                break;
            // DIAG: blocking here means input_buf is full -> pipeline behind realtime, libairspy
            // FIFO overflowing. The drops are now accounted (transfer->dropped_samples -> a clean
            // sample_seq gap), but they still cost data, so the warning still matters.
            const double _wms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _w0)
                    .count();
            if (_wms > 2.0)
                WARN("airspy: blocked {:.1f} ms waiting for an empty frame -- pipeline behind "
                     "realtime, USB FIFO likely dropping samples",
                     _wms);
            _frame_seq0 = _samples_seq; // this frame's first sample is at the current true index
        }

        size_t copy_length = std::min<size_t>(bt, buf->frame_size - frame_loc);
        DEBUG("Filling Buffer {:d} With {:d} Data Samples ({})", frame_id,
              copy_length / BYTES_PER_SAMPLE, transfer->sample_count);

        memcpy(frame_ptr + frame_loc, in, copy_length);
        bt -= copy_length;
        in = (void*)((char*)in + copy_length);
        _samples_seq += (int64_t)(copy_length / BYTES_PER_SAMPLE); // advance true index by received samples
        frame_loc = (frame_loc + copy_length) % buf->frame_size;

        if (frame_loc == 0) {
            DEBUG("Airspy Buffer {:d} Full", frame_id);

            short* fr = (short*)frame_ptr;
            const uint32_t n_samples = buf->frame_size / BYTES_PER_SAMPLE;

            // Lock-free atomic read on the hot path (dump_adcstat is
            // std::atomic<bool>); if a dump was requested, compute stats and
            // publish them under adcstat_mutex below.
            if (dump_adcstat) {
                float mean = 0, rms = 0, rail = 0;
                for (uint32_t i = 0; i < n_samples; i++)
                    if (abs(fr[i] - 2048) >= (2 << 10))
                        rail++;
                rail /= n_samples;
                for (uint32_t i = 0; i < n_samples; i++)
                    mean += (float)fr[i];
                mean /= n_samples;
                for (uint32_t i = 0; i < n_samples; i++)
                    rms += ((float)fr[i] - mean) * ((float)fr[i] - mean);
                rms = sqrt(rms / n_samples);

                {
                    std::lock_guard<std::mutex> lock(adcstat_mutex);
                    adcrailfrac = rail;
                    adcrms = rms;
                    adcmean = mean - 2048;
                    adcstat_ready = true;
                    dump_adcstat = false;
                }
                adcstat_cv.notify_one();
                INFO("Airspy ADC mean: {:f}, RMS: {:f}, rail fraction {:f}", adcmean, adcrms,
                     adcrailfrac);
            }

            // RAW samples are unsigned 12-bit centred on 2048: subtract DC bias.
            // Also flip the sign on every other sample to shift the spectrum into
            // the first Nyquist zone (multiply by (-1)^idx).
            for (uint32_t i = 0; i < n_samples; i += 2) {
                fr[i] = fr[i] - 2048;
                fr[i + 1] = 2048 - fr[i + 1];
            }

            // If the output buffer carries GNSS channel metadata, stamp the absolute sample
            // index of this frame's first sample (sample_seq), moving the timing origin to the
            // true source. The F-engine propagates it downstream; a no-op for buffers without
            // a GNSS pool (existing configs), so it changes nothing there.
            if (metadata_is_gnss_chan(buf)) {
                buf->allocate_new_metadata_object(frame_id);
                get_gnss_chan_metadata(buf, frame_id)->sample_seq = _frame_seq0;
            }
            buf->mark_frame_full(unique_name, frame_id);
            frame_id = (frame_id + 1) % buf->num_frames;
            frame_ptr = nullptr; // release: next iteration fetches a fresh frame
        }
    }
    // Publish the true sample index for lock-free /adcstat reads (once per transfer).
    _samples_total.store((uint64_t)_samples_seq, std::memory_order_relaxed);
    pthread_mutex_unlock(&recv_busy);
}

struct airspy_device* airspyInput::init_device() {
    int result;
    struct airspy_device* dev = nullptr;
    uint8_t board_id = AIRSPY_BOARD_ID_INVALID;

    if (_airspy_sn) {
        result = airspy_open_sn(&dev, _airspy_sn);
    } else if (!_airspy_fn.empty()) {
        int airspy_fd = open(_airspy_fn.c_str(), O_RDWR);
        if (airspy_fd == -1) {
            FATAL_ERROR("Error opening file: {:s}", _airspy_fn);
            return nullptr;
        }
        // libairspy accepts a file descriptor through the same _sn entrypoint.
        result = airspy_open_sn(&dev, airspy_fd);
        close(airspy_fd);
    } else {
        result = airspy_open(&dev);
    }
    if (result != AIRSPY_SUCCESS) {
        FATAL_ERROR("airspy_open() failed: {:s} ({:d})",
                    airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }

    { // pick a supported samplerate that matches the config
        uint32_t supported_samplerate_count;
        result = airspy_get_samplerates(dev, &supported_samplerate_count, 0);
        if (result != AIRSPY_SUCCESS) {
            FATAL_ERROR("airspy_get_samplerates() failed: {:s} ({:d})",
                        airspy_error_name((enum airspy_error)result), result);
            return nullptr;
        }
        uint32_t* supported_samplerates =
            (uint32_t*)malloc(supported_samplerate_count * sizeof(uint32_t));
        result = airspy_get_samplerates(dev, supported_samplerates, supported_samplerate_count);
        if (result != AIRSPY_SUCCESS) {
            FATAL_ERROR("airspy_get_samplerates() failed: {:s} ({:d})",
                        airspy_error_name((enum airspy_error)result), result);
            free(supported_samplerates);
            return nullptr;
        }
        int samplerate_idx = -1;
        for (uint32_t i = 0; i < supported_samplerate_count; i++) {
            INFO("Samplerate: idx {:d} = {:d} Hz", i, supported_samplerates[i]);
            if (supported_samplerates[i] == _sample_rate)
                samplerate_idx = i;
        }
        free(supported_samplerates);
        if (samplerate_idx < 0) {
            FATAL_ERROR("Unsupported sample rate: {:d} Hz", _sample_rate);
            return nullptr;
        }
        INFO("Selected sample rate: {:d} Hz -> idx {:d}", _sample_rate, samplerate_idx);
        result = airspy_set_samplerate(dev, samplerate_idx);
        if (result != AIRSPY_SUCCESS) {
            FATAL_ERROR("airspy_set_samplerate() failed: {:s} ({:d})",
                        airspy_error_name((enum airspy_error)result), result);
            return nullptr;
        }
    }

    result = airspy_set_sample_type(dev, AIRSPY_SAMPLE_RAW);
    if (result != AIRSPY_SUCCESS) {
        FATAL_ERROR("airspy_set_sample_type() failed: {:s} ({:d})",
                    airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }

    result = airspy_set_freq(dev, freq);
    if (result != AIRSPY_SUCCESS) {
        FATAL_ERROR("airspy_set_freq() failed: {:s} ({:d})",
                    airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }

    result = airspy_set_vga_gain(dev, _gain_if);
    if (result != AIRSPY_SUCCESS) {
        FATAL_ERROR("airspy_set_vga_gain() failed: {:s} ({:d})",
                    airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }

    result = airspy_set_mixer_gain(dev, _gain_mix);
    if (result != AIRSPY_SUCCESS) {
        FATAL_ERROR("airspy_set_mixer_gain() failed: {:s} ({:d})",
                    airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }
    result = airspy_set_mixer_agc(dev, 0); // disable mixer AGC
    if (result != AIRSPY_SUCCESS) {
        FATAL_ERROR("airspy_set_mixer_agc() failed: {:s} ({:d})",
                    airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }

    result = airspy_set_lna_gain(dev, _gain_lna);
    if (result != AIRSPY_SUCCESS) {
        FATAL_ERROR("airspy_set_lna_gain() failed: {:s} ({:d})",
                    airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }

    result = airspy_set_rf_bias(dev, _biast_power);
    if (result != AIRSPY_SUCCESS) {
        FATAL_ERROR("airspy_set_rf_bias() failed: {:s} ({:d})",
                    airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }

    // R820T fractional-N PLL dither: reg 0x12 bit 4 (0 = dither on, 1 = off).
    // Between airspy_open and airspy_start_rx the R820T is in standby, so the
    // write only updates the firmware's register shadow; airspy_start_rx later
    // bulk-writes that shadow before locking the PLL, so the lock happens with
    // the chosen dither setting baked in. Disabling dither eliminates the
    // per-restart inter-unit LO drift at fractional tunes (coherent mode) at
    // the cost of deterministic fractional-N spurs.
    {
        uint8_t v;
        result = airspy_r820t_read(dev, 0x12, &v);
        if (result != AIRSPY_SUCCESS) {
            FATAL_ERROR("airspy_r820t_read(0x12) failed: {:s} ({:d})",
                        airspy_error_name((enum airspy_error)result), result);
            return nullptr;
        }
        v = _dither_disable ? (v | 0x10) : (v & ~0x10);
        result = airspy_r820t_write(dev, 0x12, v);
        if (result != AIRSPY_SUCCESS) {
            FATAL_ERROR("airspy_r820t_write(0x12) failed: {:s} ({:d})",
                        airspy_error_name((enum airspy_error)result), result);
            return nullptr;
        }
        INFO("R820T fractional-N PLL dither {:s}",
             _dither_disable ? "disabled (coherent mode)" : "enabled (stock)");
    }

    result = airspy_board_id_read(dev, &board_id);
    if (result != AIRSPY_SUCCESS) {
        FATAL_ERROR("airspy_board_id_read() failed: {:s} ({:d})",
                    airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }
    INFO("Board ID Number: {:d} ({:s})", board_id,
         airspy_board_id_name((enum airspy_board_id)board_id));

    airspy_read_partid_serialno_t read_partid_serialno;
    result = airspy_board_partid_serialno_read(dev, &read_partid_serialno);
    if (result != AIRSPY_SUCCESS) {
        FATAL_ERROR("airspy_board_partid_serialno_read() failed: {:s} ({:d})",
                    airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }
    INFO("Part ID Number: {:#08X} {:#08X}", read_partid_serialno.part_id[0],
         read_partid_serialno.part_id[1]);
    INFO("Serial Number: {:#08X}{:08X}", read_partid_serialno.serial_no[2],
         read_partid_serialno.serial_no[3]);
    _airspy_sn =
        (((long)read_partid_serialno.serial_no[2]) << 32) + read_partid_serialno.serial_no[3];

    return dev;
}
