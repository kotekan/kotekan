#include "airspyInput.hpp"

#include <fcntl.h>              // for open, O_RDWR
#include <stdint.h>             // for uint32_t, uint8_t
#include <stdlib.h>             // for free, abs, malloc
#include <string.h>             // for memcpy
#include <unistd.h>             // for size_t, close, usleep
#include <json.hpp>             // for basic_json, json
#include <condition_variable>   // for condition_variable
#include <functional>           // for bind, function, _1, _2
#include <mutex>                // for mutex, unique_lock, lock_guard
#include <algorithm>            // for min
#include <cmath>                // for sqrt
#include <memory>               // for shared_ptr

#include "Config.hpp"           // for Config
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp"  // for make_input_desc
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "kotekanLogging.hpp"   // for ERROR, INFO, DEBUG, FATAL_ERROR
#include "restServer.hpp"       // for connectionInstance, HTTP_RESPONSE, restServer
#include "fmt.hpp"              // for compile_string_to_view, format
#include "NDArray.hpp"          // for GenericNDArray

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

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

    // Buffer carries int16 1-D samples. ensure_frame_desc validates byte size
    // against buf->frame_size and either records the descriptor or compares
    // against an earlier assertion -- see airspyFrameDesc.hpp.
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
    adcstat_cv.wait(lock, [this] { return adcstat_ready || stop_thread; });
    if (stop_thread) {
        lock.unlock();
        conn.send_error("Stage shutting down.", kotekan::HTTP_RESPONSE::INTERNAL_ERROR);
        return;
    }

    nlohmann::json reply;
    reply["rms"] = adcrms;
    reply["mean"] = adcmean;
    reply["railfrac"] = adcrailfrac;
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
}

int airspyInput::airspy_callback(airspy_transfer_t* transfer) {
    airspyInput* proc = static_cast<airspyInput*>(transfer->ctx);
    proc->airspy_producer(transfer);
    return 0;
}

void airspyInput::airspy_producer(airspy_transfer_t* transfer) {
    // Serialise overlapping callbacks; libairspy can in principle deliver them concurrently.
    pthread_mutex_lock(&recv_busy);

    void* in = transfer->samples;
    size_t bt = transfer->sample_count * BYTES_PER_SAMPLE;
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

        if (frame_loc == 0) {
            DEBUG("Airspy waiting for frame_id {:d}", frame_id);
            frame_ptr = (unsigned char*)buf->wait_for_empty_frame(unique_name, frame_id);
            if (frame_ptr == nullptr)
                break;
        }

        size_t copy_length = std::min<size_t>(bt, buf->frame_size - frame_loc);
        DEBUG("Filling Buffer {:d} With {:d} Data Samples ({})", frame_id,
              copy_length / BYTES_PER_SAMPLE, transfer->sample_count);

        memcpy(frame_ptr + frame_loc, in, copy_length);
        bt -= copy_length;
        in = (void*)((char*)in + copy_length);
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

            buf->mark_frame_full(unique_name, frame_id);
            frame_id = (frame_id + 1) % buf->num_frames;
        }
    }
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
