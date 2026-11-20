#include "bladeRFInput.hpp"

#include "Config.hpp"            // for Config
#include "GnssChanMetadata.hpp"  // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"         // for Telescope
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for INFO, WARN, ERROR, FATAL_ERROR
#include "prometheusMetrics.hpp" // for Metrics, Counter, Gauge
#include "restServer.hpp"        // for connectionInstance, HTTP_RESPONSE, restServer

#include "fmt.hpp" // for format

#include <algorithm>  // for min
#include <bladeRF2.h> // for bladerf_set_bias_tee, bladerf_set_pll_enable
#include <chrono>     // for steady_clock, system_clock, milliseconds, seconds
#include <cmath>      // for llround, sqrt
#include <exception>  // for exception
#include <functional> // for bind, _1, _2
#include <future>     // for promise, future
#include <json.hpp>   // for basic_json, json
#include <memory>     // for make_shared
#include <string.h>   // for memcpy, memset
#include <thread>     // for sleep_for, thread
#include <utility>    // for pair
#include <vector>     // for vector

#if !defined(LIBBLADERF_API_VERSION) || LIBBLADERF_API_VERSION < 0x02050000
#error "bladeRFInput needs libbladeRF 2.5 or later, for 8-bit samples and OVERSAMPLE"
#endif

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(bladeRFInput);

namespace {
/// The AD9361's highest sample rate without libbladeRF's OVERSAMPLE feature.
constexpr uint32_t MAX_STANDARD_RATE = 61440000;
/// The AD9361's widest analog bandwidth.
constexpr uint32_t MAX_BANDWIDTH = 56000000;
/// The receive channel used.
const bladerf_channel RX_CH = BLADERF_CHANNEL_RX(0);
/// Bytes per complex 8-bit sample.
constexpr size_t BYTES_PER_SAMPLE = 2;

std::string version_string(const struct bladerf_version& v) {
    return fmt::format("{:d}.{:d}.{:d}", v.major, v.minor, v.patch);
}
} // namespace

bladeRFInput::bladeRFInput(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bladeRFInput::main_thread, this)) {

    buf = get_buffer("out_buf");
    buf->register_producer(unique_name);

    if (buf->frame_size % BYTES_PER_SAMPLE != 0) {
        FATAL_ERROR("bladeRFInput: out_buf frame_size ({:d} B) must hold whole I/Q pairs.",
                    buf->frame_size);
        return;
    }
    frame_samples = buf->frame_size / BYTES_PER_SAMPLE;
    // The buffer's cint8 descriptor is declared in config (kotekan_buffer: ndarray).
    buf->require_frame_desc();

    _serial = config.get_default<std::string>(unique_name, "serial", "");
    _fpga_image = config.get_default<std::string>(unique_name, "fpga_image", "");
    _freq = config.get_default<double>(unique_name, "freq", 1225.0) * 1e6;
    _sample_rate = config.get_default<double>(unique_name, "sample_bw", 122.88) * 1e6;
    _oversample = _sample_rate > MAX_STANDARD_RATE;
    _bandwidth = config.get_default<double>(unique_name, "bandwidth",
                                            std::min(_sample_rate, MAX_BANDWIDTH) / 1e6)
                 * 1e6;
    _gain = config.get_default<int>(unique_name, "gain", 30);
    _bias_tee = config.get_default<bool>(unique_name, "bias_tee", false);
    _ref_in_freq = config.get_default<double>(unique_name, "ref_in_freq", 0) * 1e6;
    _num_buffers = config.get_default<int>(unique_name, "num_buffers", 64);
    _buffer_size = config.get_default<int>(unique_name, "buffer_size", 32768);
    _num_transfers = config.get_default<int>(unique_name, "num_transfers", 32);
    _timeout_ms = config.get_default<int>(unique_name, "timeout_ms", 1000);
    _max_gap_fill =
        config.get_default<double>(unique_name, "max_gap_fill_ms", 1000) * 1e-3 * _sample_rate;
    _startup_discard =
        config.get_default<double>(unique_name, "startup_discard_ms", 100) * 1e-3 * _sample_rate;
    // As the GNSS stages take it: the config's value, else the telescope's time of sample 0.
    _frame0_utc = config.get_default<double>(unique_name, "frame0_utc", 0.0);
    if (_frame0_utc <= 0 && Telescope::instance().gps_time_enabled())
        _frame0_utc = Telescope::instance().to_time_ns(0) * 1e-9;
    _adcstat_timeout_ms = config.get_default<int>(unique_name, "adcstat_timeout_ms", 250);
    _force_filter_open = config.get_default<bool>(unique_name, "force_filter_open", false);
    _data_port_delay = config.get_default<int>(unique_name, "data_port_delay", -1);
    _max_stream_restarts = config.get_default<int>(unique_name, "max_stream_restarts", 3);
    if (_data_port_delay > 0xff) {
        FATAL_ERROR("bladeRFInput: data_port_delay must be -1 or a register value 0-255, not {:d}",
                    _data_port_delay);
        return;
    }
    if (_force_filter_open && !_oversample) {
        FATAL_ERROR("bladeRFInput: force_filter_open needs a sample_bw above {:.2f} (OVERSAMPLE); "
                    "below that the digital filters set the band.",
                    MAX_STANDARD_RATE / 1e6);
        return;
    }

    _gain_mode_name = config.get_default<std::string>(unique_name, "gain_mode", "manual");
    if (_gain_mode_name == "manual")
        _gain_mode = BLADERF_GAIN_MGC;
    else if (_gain_mode_name == "default")
        _gain_mode = BLADERF_GAIN_DEFAULT;
    else if (_gain_mode_name == "fast_attack")
        _gain_mode = BLADERF_GAIN_FASTATTACK_AGC;
    else if (_gain_mode_name == "slow_attack")
        _gain_mode = BLADERF_GAIN_SLOWATTACK_AGC;
    else if (_gain_mode_name == "hybrid")
        _gain_mode = BLADERF_GAIN_HYBRID_AGC;
    else {
        FATAL_ERROR("bladeRFInput: unknown gain_mode '{:s}'; expected 'manual', 'default', "
                    "'fast_attack', 'slow_attack' or 'hybrid'.",
                    _gain_mode_name);
        return;
    }

    if (_sample_rate == 0 || _buffer_size <= 0 || _num_transfers <= 0
        || _num_transfers >= _num_buffers || _timeout_ms <= 0 || _adcstat_timeout_ms <= 0) {
        FATAL_ERROR("bladeRFInput: sample_bw, buffer_size, num_transfers, timeout_ms and "
                    "adcstat_timeout_ms must be positive, and num_transfers less than "
                    "num_buffers.");
        return;
    }
    // Forcing the filter open was measured from the filter libbladeRF tunes for 56 MHz.
    _set_bandwidth = !_oversample || _force_filter_open || config.exists(unique_name, "bandwidth");
    if (_force_filter_open && !config.exists(unique_name, "bandwidth"))
        _bandwidth = MAX_BANDWIDTH;
}

bladeRFInput::~bladeRFInput() {
    // Wake any REST callback waiting on a pending adcstat request, then remove the callbacks
    // (which waits for running ones) before the device goes away.
    adcstat_cv.notify_all();
    kotekan::restServer& rest_server = kotekan::restServer::instance();
    rest_server.remove_json_callback(unique_name + "/set_config");
    rest_server.remove_get_callback(unique_name + "/adcstat");
    rest_server.remove_get_callback(unique_name + "/get_config");
    if (dev != nullptr) {
        // libbladeRF can hang for good stopping a stream that has failed: it cancels its
        // worker thread after 3 s, then waits forever for the stream to finish. Close on a
        // separate thread, and abandon the device if that does not finish; it is released
        // when kotekan exits.
        struct bladerf* d = dev;
        auto closed = std::make_shared<std::promise<void>>();
        std::future<void> done = closed->get_future();
        std::thread([d, closed] {
            bladerf_enable_module(d, RX_CH, false);
            bladerf_close(d);
            closed->set_value();
        }).detach();
        if (done.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
            ERROR("Closing the bladeRF timed out: libbladeRF hung stopping the stream. Abandoning "
                  "the device; it is released when kotekan exits.");
    }
}

bool bladeRFInput::write_rfic_register(uint16_t addr, uint8_t val) {
    int status = bladerf_set_rfic_register(dev, addr, val);
    uint8_t readback = 0;
    if (status == 0)
        status = bladerf_get_rfic_register(dev, addr, &readback);
    if (status != 0 || readback != val) {
        FATAL_ERROR(
            "Writing 0x{:02x} to AD9361 register 0x{:03x} failed ({:s}); it reads 0x{:02x}.", val,
            addr, status ? bladerf_strerror(status) : "no error", readback);
        return false;
    }
    return true;
}

bool bladeRFInput::open_device() {
    const std::string id = _serial.empty() ? "" : "*:serial=" + _serial;
    int status = bladerf_open(&dev, id.empty() ? nullptr : id.c_str());
    if (status != 0) {
        dev = nullptr;
        FATAL_ERROR("bladerf_open({:s}) failed: {:s}", id.empty() ? "any device" : id,
                    bladerf_strerror(status));
        return false;
    }

    const std::string board = bladerf_get_board_name(dev);
    if (board != "bladerf2") {
        FATAL_ERROR("bladeRFInput needs a bladeRF 2.0, but opened a {:s}.", board);
        return false;
    }

    struct bladerf_serial sn;
    if (bladerf_get_serial_struct(dev, &sn) == 0)
        dev_serial = sn.serial;
    struct bladerf_version ver;
    if (bladerf_fw_version(dev, &ver) == 0)
        fw_version = version_string(ver);
    bladerf_fpga_size size = BLADERF_FPGA_UNKNOWN;
    bladerf_get_fpga_size(dev, &size);
    INFO("Opened bladeRF 2.0 serial {:s}, {:s} FPGA, FX3 firmware {:s}", dev_serial,
         size == BLADERF_FPGA_A4   ? "xA4"
         : size == BLADERF_FPGA_A9 ? "xA9"
                                   : "unknown",
         fw_version);

    if (!_fpga_image.empty()) {
        INFO("Loading FPGA image {:s}", _fpga_image);
        status = bladerf_load_fpga(dev, _fpga_image.c_str());
        if (status != 0) {
            FATAL_ERROR("bladerf_load_fpga({:s}) failed: {:s}", _fpga_image,
                        bladerf_strerror(status));
            return false;
        }
    }
    status = bladerf_is_fpga_configured(dev);
    if (status <= 0) {
        FATAL_ERROR("The bladeRF has no FPGA image loaded. Set fpga_image, or install one "
                    "where libbladeRF autoloads it (e.g. /usr/local/share/Nuand/bladeRF/"
                    "hostedxA4.rbf).");
        return false;
    }
    status = bladerf_fpga_version(dev, &ver);
    if (status != 0) {
        FATAL_ERROR("bladerf_fpga_version() failed: {:s}", bladerf_strerror(status));
        return false;
    }
    fpga_version = version_string(ver);
    // 8-bit samples arrived with FPGA 0.15.0, and their RX timestamps were fixed in 0.15.1.
    if (ver.major == 0 && (ver.minor < 15 || (ver.minor == 15 && ver.patch < 1))) {
        FATAL_ERROR("FPGA {:s} is loaded; bladeRFInput needs 0.15.1 or later for timestamps "
                    "on 8-bit samples.",
                    fpga_version);
        return false;
    }
    INFO("FPGA {:s}", fpga_version);

    if (bladerf_device_speed(dev) != BLADERF_DEVICE_SPEED_SUPER)
        WARN("The bladeRF is not on a SuperSpeed (USB 3) port; expect dropped samples at "
             "{:.2f} MSPS.",
             _sample_rate / 1e6);
    return true;
}

bool bladeRFInput::configure_device() {
    int status;
    if (_ref_in_freq > 0) {
        status = bladerf_set_pll_refclk(dev, _ref_in_freq);
        if (status == 0)
            status = bladerf_set_pll_enable(dev, true);
        if (status != 0) {
            FATAL_ERROR("Setting up the {:.3f} MHz reference failed: {:s}", _ref_in_freq / 1e6,
                        bladerf_strerror(status));
            return false;
        }
        bool locked = false;
        for (int i = 0; i < 20 && !locked && !stop_thread; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            bladerf_get_pll_lock_state(dev, &locked);
        }
        if (!locked) {
            FATAL_ERROR("The VCTCXO did not lock to the {:.3f} MHz reference on REF IN within "
                        "2 s.",
                        _ref_in_freq / 1e6);
            return false;
        }
        INFO("VCTCXO locked to the {:.3f} MHz reference", _ref_in_freq / 1e6);
    }

    // OVERSAMPLE must be on before the sample rate is set: libbladeRF applies its register
    // settings as part of setting the rate.
    if (_oversample) {
        status = bladerf_enable_feature(dev, BLADERF_FEATURE_OVERSAMPLE, true);
        if (status != 0) {
            FATAL_ERROR("Enabling OVERSAMPLE failed: {:s}", bladerf_strerror(status));
            return false;
        }
    }
    bladerf_sample_rate actual_rate;
    status = bladerf_set_sample_rate(dev, RX_CH, _sample_rate, &actual_rate);
    if (status != 0) {
        FATAL_ERROR("bladerf_set_sample_rate({:d}) failed: {:s}", _sample_rate,
                    bladerf_strerror(status));
        return false;
    }
    // With OVERSAMPLE, libbladeRF can report success after the rate change failed, so read
    // the rate back.
    bladerf_sample_rate readback_rate = 0;
    status = bladerf_get_sample_rate(dev, RX_CH, &readback_rate);
    if (status != 0 || readback_rate != actual_rate || actual_rate == 0) {
        FATAL_ERROR("The sample rate did not take: bladerf_set_sample_rate reported {:d}, and it "
                    "reads back as {:d}.",
                    actual_rate, readback_rate);
        return false;
    }
    _actual_sample_rate = actual_rate;
    if (actual_rate != _sample_rate)
        WARN("Asked for {:d} samples/s, got {:d}", _sample_rate, actual_rate);
    if (_set_bandwidth) {
        bladerf_bandwidth actual_bw;
        status = bladerf_set_bandwidth(dev, RX_CH, _bandwidth, &actual_bw);
        if (status != 0) {
            FATAL_ERROR("bladerf_set_bandwidth({:d}) failed: {:s}", _bandwidth,
                        bladerf_strerror(status));
            return false;
        }
        _bandwidth = actual_bw;
        INFO("Sample rate {:d} samples/s{:s}, analog bandwidth {:d} Hz", actual_rate,
             _oversample ? " with OVERSAMPLE" : "", actual_bw);
    } else {
        INFO("Sample rate {:d} samples/s with OVERSAMPLE, analog bandwidth left to libbladeRF",
             actual_rate);
    }

    status = bladerf_set_frequency(dev, RX_CH, _freq);
    if (status != 0) {
        FATAL_ERROR("bladerf_set_frequency({:d}) failed: {:s}", _freq, bladerf_strerror(status));
        return false;
    }
    status = bladerf_set_gain_mode(dev, RX_CH, _gain_mode);
    if (status != 0) {
        FATAL_ERROR("bladerf_set_gain_mode({:s}) failed: {:s}", _gain_mode_name,
                    bladerf_strerror(status));
        return false;
    }
    if (_gain_mode == BLADERF_GAIN_MGC) {
        status = bladerf_set_gain(dev, RX_CH, _gain);
        if (status != 0) {
            FATAL_ERROR("bladerf_set_gain({:d}) failed: {:s}", _gain, bladerf_strerror(status));
            return false;
        }
    }
    status = bladerf_set_bias_tee(dev, RX_CH, _bias_tee);
    if (status != 0) {
        FATAL_ERROR("bladerf_set_bias_tee() failed: {:s}", bladerf_strerror(status));
        return false;
    }
    INFO("Tuned to {:d} Hz, gain mode {:s}, gain {:d} dB, bias tee {:s}", _freq, _gain_mode_name,
         _gain, _bias_tee ? "on" : "off");

    // The RX mux is an FPGA register that survives closing and re-opening the device.
    status = bladerf_set_rx_mux(dev, BLADERF_RX_MUX_BASEBAND);
    if (status != 0) {
        FATAL_ERROR("bladerf_set_rx_mux() failed: {:s}", bladerf_strerror(status));
        return false;
    }
    if (_data_port_delay >= 0) {
        if (!write_rfic_register(0x006, _data_port_delay))
            return false;
        INFO("AD9361 data port delay (0x006) set to 0x{:02x}", _data_port_delay);
    }
    if (_force_filter_open) {
        // Bit 7 of 0x1E6 (TUNE_OVERRIDE) makes the BBF use the capacitors written to
        // 0x1E7-0x1EC instead of the calibrated ones, and bits 1 and 3 of 0x1DB make the TIAs
        // use the capacitors in 0x1DC-0x1DF. libbladeRF's OVERSAMPLE settings clear these
        // override bits, so they have no effect.
        uint8_t r2346 = 0, tia_config = 0, tia1_c = 0, tia2_c = 0;
        if (bladerf_get_rfic_register(dev, 0x1e6, &r2346) != 0
            || bladerf_get_rfic_register(dev, 0x1db, &tia_config) != 0
            || bladerf_get_rfic_register(dev, 0x1dc, &tia1_c) != 0
            || bladerf_get_rfic_register(dev, 0x1de, &tia2_c) != 0) {
            FATAL_ERROR("Reading the AD9361 filter registers failed.");
            return false;
        }
        const std::pair<uint16_t, uint8_t> writes[] = {
            {0x1e6, (uint8_t)(r2346 | 0x80)}, // BBF tune override, keeping R2346
            {0x1e7, 0},                       // BBF C1, C2, C3 at their minimum
            {0x1e8, 0},
            {0x1e9, 0},
            {0x1ea, 0},
            {0x1eb, 0},
            {0x1ec, 0},
            {0x1db, (uint8_t)(tia_config | 0x0a)}, // TIA1 and TIA2 capacitor override
            {0x1dc, (uint8_t)(tia1_c & 0xc0)},     // TIA C at its minimum, keeping RF
            {0x1dd, 0},
            {0x1de, (uint8_t)(tia2_c & 0xc0)},
            {0x1df, 0},
        };
        for (const auto& [addr, val] : writes)
            if (!write_rfic_register(addr, val))
                return false;
        INFO("Forced the RX baseband filter and TIA capacitors to their minimum");
    }

    status = bladerf_sync_config(dev, BLADERF_RX_X1, BLADERF_FORMAT_SC8_Q7_META, _num_buffers,
                                 _buffer_size, _num_transfers, _timeout_ms);
    if (status != 0) {
        FATAL_ERROR("bladerf_sync_config() failed: {:s}", bladerf_strerror(status));
        return false;
    }
    return true;
}

void bladeRFInput::get_config_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply;
    reply["serial"] = dev_serial;
    reply["fpga_version"] = fpga_version;
    reply["fw_version"] = fw_version;
    reply["freq"] = _freq;
    reply["sample_rate"] = _actual_sample_rate;
    reply["oversample"] = _oversample;
    if (_set_bandwidth)
        reply["bandwidth"] = _bandwidth;
    reply["gain_mode"] = _gain_mode_name;
    bladerf_gain gain;
    if (bladerf_get_gain(dev, RX_CH, &gain) == 0)
        reply["gain"] = gain;
    reply["bias_tee"] = _bias_tee;
    reply["ref_in_freq"] = _ref_in_freq;
    if (_ref_in_freq > 0) {
        bool locked = false;
        if (bladerf_get_pll_lock_state(dev, &locked) == 0)
            reply["pll_locked"] = locked;
    }
    float temp;
    if (bladerf_get_rfic_temperature(dev, &temp) == 0)
        reply["rfic_temperature"] = temp;
    reply["first_timestamp"] = first_timestamp.load();
    conn.send_json_reply(reply);
}

void bladeRFInput::adcstat_callback(kotekan::connectionInstance& conn) {
    std::unique_lock<std::mutex> lock(adcstat_mutex);
    // Stats published for an earlier request that timed out are stale.
    adcstat_ready = false;
    dump_adcstat = true;
    // An unbounded wait here would block every REST endpoint, since they share one thread.
    if (!adcstat_cv.wait_for(lock, std::chrono::milliseconds(_adcstat_timeout_ms),
                             [this] { return adcstat_ready || stop_thread; })) {
        dump_adcstat = false;
        lock.unlock();
        conn.send_error("Timed out waiting for a frame from the bladeRF.",
                        kotekan::HTTP_RESPONSE::REQUEST_FAILED);
        return;
    }
    if (stop_thread) {
        lock.unlock();
        conn.send_error("Stage shutting down.", kotekan::HTTP_RESPONSE::INTERNAL_ERROR);
        return;
    }

    nlohmann::json reply;
    reply["mean_i"] = adc_mean_i;
    reply["mean_q"] = adc_mean_q;
    reply["rms_i"] = adc_rms_i;
    reply["rms_q"] = adc_rms_q;
    reply["railfrac"] = adc_railfrac;
    adcstat_ready = false;
    lock.unlock();

    conn.send_json_reply(reply);
}

void bladeRFInput::set_config_callback(kotekan::connectionInstance& conn,
                                       nlohmann::json& json_request) {
    // Parse and check the whole request before changing anything.
    const bool has_freq = json_request.contains("freq");
    const bool has_gain = json_request.contains("gain");
    const bool has_bias_tee = json_request.contains("bias_tee");
    uint64_t freq = 0;
    int gain = 0;
    bool bias_tee = false;
    try {
        if (has_freq)
            freq = json_request["freq"].get<double>() * 1e6;
        if (has_gain)
            gain = json_request["gain"].get<int>();
        if (has_bias_tee)
            bias_tee = json_request["bias_tee"].get<bool>();
    } catch (const std::exception& e) {
        conn.send_error(fmt::format("Couldn't parse bladeRF parameters: {:s}", e.what()),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    if (!has_freq && !has_gain && !has_bias_tee) {
        conn.send_error("Expected one or more of freq, gain and bias_tee.",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    if (has_gain && _gain_mode != BLADERF_GAIN_MGC) {
        conn.send_error("gain can only be set with gain_mode manual.",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }

    int status = 0;
    std::string failed;
    if (has_freq) {
        status = bladerf_set_frequency(dev, RX_CH, freq);
        if (status == 0) {
            _freq = freq;
            INFO("Retuned to {:d} Hz", _freq);
        } else {
            failed = "freq";
        }
    }
    if (status == 0 && has_gain) {
        status = bladerf_set_gain(dev, RX_CH, gain);
        if (status == 0) {
            _gain = gain;
            INFO("Gain set to {:d} dB", _gain);
        } else {
            failed = "gain";
        }
    }
    if (status == 0 && has_bias_tee) {
        status = bladerf_set_bias_tee(dev, RX_CH, bias_tee);
        if (status == 0) {
            _bias_tee = bias_tee;
            INFO("Bias tee {:s}", _bias_tee ? "on" : "off");
        } else {
            failed = "bias_tee";
        }
    }
    if (status != 0) {
        conn.send_error(fmt::format("Setting {:s} failed: {:s}", failed, bladerf_strerror(status)),
                        kotekan::HTTP_RESPONSE::INTERNAL_ERROR);
        return;
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void bladeRFInput::main_thread() {
    // Both FATAL_ERROR with the reason on failure.
    if (!open_device() || !configure_device())
        return;

    // The callbacks use the device, so register them only now.
    using namespace std::placeholders;
    kotekan::restServer& rest_server = kotekan::restServer::instance();
    rest_server.register_post_callback(unique_name + "/set_config",
                                       std::bind(&bladeRFInput::set_config_callback, this, _1, _2));
    rest_server.register_get_callback(unique_name + "/adcstat",
                                      std::bind(&bladeRFInput::adcstat_callback, this, _1));
    rest_server.register_get_callback(unique_name + "/get_config",
                                      std::bind(&bladeRFInput::get_config_callback, this, _1));

    stream();
}

void bladeRFInput::stream() {
    auto& dropped_metric =
        Metrics::instance().add_counter("kotekan_bladerfinput_dropped_samples_total", unique_name);
    auto& discont_metric =
        Metrics::instance().add_counter("kotekan_bladerfinput_discontinuities_total", unique_name);
    auto& restart_metric =
        Metrics::instance().add_counter("kotekan_bladerfinput_stream_restarts_total", unique_name);
    kotekan::prometheus::Gauge* pll_metric = nullptr;
    if (_ref_in_freq > 0) {
        pll_metric = &Metrics::instance().add_gauge("kotekan_bladerfinput_pll_locked", unique_name);
        pll_metric->set(1);
    }

    const auto report_interval = std::chrono::seconds(30);
    auto next_drop_report = std::chrono::steady_clock::now();
    auto next_pll_check = next_drop_report + std::chrono::seconds(1);
    uint64_t n_dropped = 0, reported_dropped = 0, n_discont = 0;
    bool pll_locked = true;

    std::vector<int8_t> chunk(BYTES_PER_SAMPLE * _buffer_size);
    bool have_timestamp = false;
    uint64_t next_timestamp = 0;
    // The first reads after enabling RX can hold stale samples from the previous session,
    // followed by a jump in the timestamps, so they are discarded.
    uint64_t to_discard = _startup_discard;
    bool any_samples = false;
    int consecutive_errors = 0;

    // RX is enabled only now, right before the first read. With seconds between the two the
    // stream sometimes never starts: 3 of 8 sessions with 3 s, none of 10 without, on the
    // first unit with FX3 2.6.0.
    int status = bladerf_enable_module(dev, RX_CH, true);
    if (status != 0) {
        FATAL_ERROR("bladerf_enable_module() failed: {:s}", bladerf_strerror(status));
        return;
    }

    while (!stop_thread) {
        struct bladerf_metadata meta;
        memset(&meta, 0, sizeof(meta));
        meta.flags = BLADERF_META_FLAG_RX_NOW;
        // Each read returns contiguous samples starting at meta.timestamp. On a dropout it
        // returns early, and the next read starts after the gap.
        status = bladerf_sync_rx(dev, chunk.data(), _buffer_size, &meta, _timeout_ms);
        if (status != 0) {
            // libbladeRF reports a read that timed out as an unexpected error.
            if (!any_samples) {
                FATAL_ERROR("The bladeRF delivered no samples ({:s}). It sometimes fails to start "
                            "streaming, and only re-opening it, by restarting kotekan, recovers.",
                            bladerf_strerror(status));
                return;
            }
            if (++consecutive_errors > _max_stream_restarts) {
                FATAL_ERROR("bladerf_sync_rx() failed {:d} times in a row: {:s}",
                            consecutive_errors, bladerf_strerror(status));
                return;
            }
            // The next read restarts the stream. Its first samples may be stale, as at startup.
            WARN("bladerf_sync_rx() failed: {:s}. Restarting the stream ({:d} of {:d}).",
                 bladerf_strerror(status), consecutive_errors, _max_stream_restarts);
            restart_metric.inc();
            to_discard = _startup_discard;
            continue;
        }
        consecutive_errors = 0;
        any_samples = true;

        if (to_discard > 0) {
            to_discard -= std::min<uint64_t>(to_discard, meta.actual_count);
            continue;
        }
        if (!have_timestamp) {
            have_timestamp = true;
            next_timestamp = meta.timestamp;
            first_timestamp = meta.timestamp;
            INFO("Streaming; the first sample has timestamp {:d}", meta.timestamp);
            if (_frame0_utc > 0) {
                // A read returns samples the device took a pipeline of transfers earlier
                // (~10-30 ms here), so its arrival is not their time. The device's sample
                // counter, read between two host clock readings, places them.
                const auto host_s = [] {
                    return std::chrono::duration<double>(
                               std::chrono::system_clock::now().time_since_epoch())
                        .count();
                };
                uint64_t ts_now = 0;
                const double t1 = host_s();
                status = bladerf_get_timestamp(dev, BLADERF_RX, &ts_now);
                const double t2 = host_s();
                if (status != 0 || ts_now < meta.timestamp) {
                    FATAL_ERROR("Reading the bladeRF's sample counter failed ({:s}, {:d} < {:d}).",
                                bladerf_strerror(status), ts_now, meta.timestamp);
                    return;
                }
                const double lag_s = (double)(ts_now - meta.timestamp) / _actual_sample_rate;
                const double t_first = 0.5 * (t1 + t2) - lag_s;
                INFO("The first sample was taken {:.3f} ms before the counter read, which took "
                     "{:.3f} ms",
                     lag_s * 1e3, (t2 - t1) * 1e3);
                const int64_t seq = std::llround((t_first - _frame0_utc) * _actual_sample_rate);
                if (seq < 0) {
                    FATAL_ERROR("frame0_utc {:.3f} is in the future.", _frame0_utc);
                    return;
                }
                frame_seq = seq - seq % (int64_t)frame_samples;
                INFO("The first sample is sample {:d} from frame0_utc {:.3f}", seq, _frame0_utc);
                if (!write_samples(nullptr, seq % (int64_t)frame_samples))
                    return;
            }
        }
        if (meta.timestamp != next_timestamp) {
            if (meta.timestamp < next_timestamp) {
                FATAL_ERROR("The bladeRF timestamp went back from {:d} to {:d}.", next_timestamp,
                            meta.timestamp);
                return;
            }
            const uint64_t gap = meta.timestamp - next_timestamp;
            if (gap > _max_gap_fill) {
                FATAL_ERROR("The bladeRF dropped {:d} samples at once, more than max_gap_fill_ms "
                            "allows.",
                            gap);
                return;
            }
            if (!write_samples(nullptr, gap))
                return;
            n_dropped += gap;
            n_discont++;
            dropped_metric.inc(gap);
            discont_metric.inc();
        }
        if (!write_samples(chunk.data(), meta.actual_count))
            return;
        next_timestamp = meta.timestamp + meta.actual_count;

        const auto now = std::chrono::steady_clock::now();
        if (n_dropped > reported_dropped && now >= next_drop_report) {
            WARN("The bladeRF dropped {:d} samples ({:d} in total, in {:d} gaps), usually because "
                 "USB or this stage fell behind; they are zeros in the stream.",
                 n_dropped - reported_dropped, n_dropped, n_discont);
            reported_dropped = n_dropped;
            next_drop_report = now + report_interval;
        }
        if (pll_metric != nullptr && now >= next_pll_check) {
            bool locked = false;
            bladerf_get_pll_lock_state(dev, &locked);
            pll_metric->set(locked ? 1 : 0);
            if (locked != pll_locked) {
                if (locked)
                    INFO("The VCTCXO is locked to the reference again.");
                else
                    ERROR("The VCTCXO lost lock to the {:.3f} MHz reference; the sample clock is "
                          "free running.",
                          _ref_in_freq / 1e6);
                pll_locked = locked;
            }
            next_pll_check = now + std::chrono::seconds(1);
        }
    }
}

bool bladeRFInput::write_samples(const int8_t* src, size_t n) {
    while (n > 0) {
        if (frame_ptr == nullptr) {
            frame_ptr = (int8_t*)buf->wait_for_empty_frame(unique_name, frame_id);
            if (frame_ptr == nullptr)
                return false;
            frame_loc = 0;
        }
        const size_t len = std::min(n, frame_samples - frame_loc);
        if (src != nullptr) {
            memcpy(frame_ptr + BYTES_PER_SAMPLE * frame_loc, src, BYTES_PER_SAMPLE * len);
            src += BYTES_PER_SAMPLE * len;
        } else {
            memset(frame_ptr + BYTES_PER_SAMPLE * frame_loc, 0, BYTES_PER_SAMPLE * len);
        }
        frame_loc += len;
        n -= len;

        if (frame_loc == frame_samples) {
            // Lock-free check on the hot path; the stats are published under adcstat_mutex.
            if (dump_adcstat)
                publish_adcstat(frame_ptr, frame_samples);
            if (metadata_is_gnss_chan(buf)) {
                buf->allocate_new_metadata_object(frame_id);
                get_gnss_chan_metadata(buf, frame_id)->sample_seq = frame_seq;
            }
            frame_seq += frame_samples;
            buf->mark_frame_full(unique_name, frame_id);
            frame_id = (frame_id + 1) % buf->num_frames;
            frame_ptr = nullptr;
        }
    }
    return true;
}

void bladeRFInput::publish_adcstat(const int8_t* frame, size_t n) {
    int64_t sum_i = 0, sum_q = 0, sum_ii = 0, sum_qq = 0;
    size_t rail = 0;
    for (size_t k = 0; k < n; k++) {
        const int i = frame[2 * k], q = frame[2 * k + 1];
        sum_i += i;
        sum_q += q;
        sum_ii += i * i;
        sum_qq += q * q;
        if (i == 127 || i == -128 || q == 127 || q == -128)
            rail++;
    }
    const double mean_i = (double)sum_i / n, mean_q = (double)sum_q / n;
    const double rms_i = std::sqrt((double)sum_ii / n - mean_i * mean_i);
    const double rms_q = std::sqrt((double)sum_qq / n - mean_q * mean_q);
    const double railfrac = (double)rail / n;
    {
        std::lock_guard<std::mutex> lock(adcstat_mutex);
        adc_mean_i = mean_i;
        adc_mean_q = mean_q;
        adc_rms_i = rms_i;
        adc_rms_q = rms_q;
        adc_railfrac = railfrac;
        adcstat_ready = true;
        dump_adcstat = false;
    }
    adcstat_cv.notify_one();
    INFO("bladeRF ADC mean ({:.2f}, {:.2f}), RMS ({:.2f}, {:.2f}), rail fraction {:g}", mean_i,
         mean_q, rms_i, rms_q, railfrac);
}
