#include "airspyInput.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(airspyInput);

airspyInput::airspyInput(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&airspyInput::main_thread, this)) {

    buf = get_buffer("out_buf");
    register_producer(buf, unique_name.c_str());

    freq = config.get_default<float>(unique_name, "freq", 1420) * 1000000;          // MHz
    sample_bw = config.get_default<float>(unique_name, "sample_bw", 2.5) * 1000000; // BW in Hz
    gain_lna = config.get_default<int>(unique_name, "gain_lna", 5);                 // MAX: 14
    gain_if = config.get_default<int>(unique_name, "gain_if", 5);                   // MAX: 15
    gain_mix = config.get_default<int>(unique_name, "gain_mix", 5);                 // MAX: 15
    biast_power = config.get_default<bool>(unique_name, "biast_power", false) ? 1 : 0;
}

airspyInput::~airspyInput() {
    if (a_device != nullptr) {
        airspy_stop_rx(a_device);
        airspy_close(a_device);
    }
    airspy_exit();
}

void airspyInput::main_thread() {
    frame_id = 0;
    frame_loc = 0;
    recv_busy = PTHREAD_MUTEX_INITIALIZER;

    airspy_init();
    a_device = init_device();
    if (a_device == nullptr) {
        FATAL_ERROR("Error in airspyInput. Cannot find device.");
        return;
    }
    airspy_start_rx(a_device, airspy_callback, static_cast<void*>(this));
}

int airspyInput::airspy_callback(airspy_transfer_t* transfer) {
    airspyInput* proc = static_cast<airspyInput*>(transfer->ctx);
    proc->airspy_producer(transfer);
    return 0;
}
void airspyInput::airspy_producer(airspy_transfer_t* transfer) {
    // make sure two callbacks don't run at once
    pthread_mutex_lock(&recv_busy);

    void* in = transfer->samples;
    size_t bt = transfer->sample_count * BYTES_PER_SAMPLE;
    while (bt > 0) {
        if (frame_loc == 0) {
            DEBUG("Airspy waiting for frame_id {:d}", frame_id);
            frame_ptr = (unsigned char*)wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
            if (frame_ptr == nullptr)
                break;
        }

        size_t copy_length = bt < buf->frame_size ? bt : buf->frame_size;
        DEBUG("Filling Buffer {:d} With {:d} Data Samples", frame_id, copy_length / 2 / 2);
        // FILL THE BUFFER
        memcpy(frame_ptr + frame_loc, in, copy_length);
        bt -= copy_length;
        frame_loc = (frame_loc + copy_length) % buf->frame_size;

        if (frame_loc == 0) {
            DEBUG("Airspy Buffer {:d} Full", frame_id);
            mark_frame_full(buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % buf->num_frames;
        }
    }
    pthread_mutex_unlock(&recv_busy);
}

struct airspy_device* airspyInput::init_device() {
    int result;
    uint8_t board_id = AIRSPY_BOARD_ID_INVALID;

    struct airspy_device* dev;
    result = airspy_open(&dev);
    if (result != AIRSPY_SUCCESS) {
        ERROR("airspy_open() failed: {:s} ({:d})", airspy_error_name((enum airspy_error)result),
              result);
        return nullptr;
    }

    { // get the viable sample rates, compare to the config, and set choose the appropriate one
        uint32_t supported_samplerate_count;
        result = airspy_get_samplerates(dev, &supported_samplerate_count, 0);
        if (result != AIRSPY_SUCCESS) {
            ERROR("airspy_set_samplerate() failed: {:s} ({:d})",
                  airspy_error_name((enum airspy_error)result), result);
            return nullptr;
        }
        uint32_t* supported_samplerates =
            (uint32_t*)malloc(supported_samplerate_count * sizeof(uint32_t));
        result = airspy_get_samplerates(dev, supported_samplerates, supported_samplerate_count);
        if (result != AIRSPY_SUCCESS) {
            ERROR("airspy_set_samplerate() failed: {:s} ({:d})",
                  airspy_error_name((enum airspy_error)result), result);
            return nullptr;
        }
        int samplerate_idx = -1;
        for (uint i = 0; i < supported_samplerate_count; i++) {
            INFO("Samplerate: idx {:d} = {:d} Hz", i, supported_samplerates[i]);
            if (supported_samplerates[i] == sample_bw)
                samplerate_idx = i;
        }
        if (samplerate_idx < 0) {
            ERROR("Unsupported sample rate: {:f} Hz", sample_bw);
            return nullptr;
        }
        INFO("Selected sample rate: {:d} Hz -> idx {:d}", sample_bw, samplerate_idx)
        result = airspy_set_samplerate(dev, samplerate_idx);
        if (result != AIRSPY_SUCCESS) {
            ERROR("airspy_set_samplerate() failed: {:s} ({:d})",
                  airspy_error_name((enum airspy_error)result), result);
            return nullptr;
        }
    }

    result = airspy_set_sample_type(dev, AIRSPY_SAMPLE_INT16_IQ);
    if (result != AIRSPY_SUCCESS) {
        ERROR("airspy_set_sample_type() failed: {:s} ({:d})",
              airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }

    result = airspy_set_freq(dev, freq);
    if (result != AIRSPY_SUCCESS) {
        ERROR("airspy_set_freq() failed: {:s} ({:d})", airspy_error_name((enum airspy_error)result),
              result);
        return nullptr;
    }

    result = airspy_set_vga_gain(dev, gain_if);
    if (result != AIRSPY_SUCCESS) {
        ERROR("airspy_set_vga_gain() failed: {:s} ({:d})",
              airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }

    result = airspy_set_mixer_gain(dev, gain_mix);
    if (result != AIRSPY_SUCCESS) {
        ERROR("airspy_set_mixer_gain() failed: {:s} ({:d})",
              airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }
    result = airspy_set_mixer_agc(dev, 0); // Auto gain control: 0/1
    if (result != AIRSPY_SUCCESS) {
        ERROR("airspy_set_mixer_agc() failed: {:s} ({:d})",
              airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }

    result = airspy_set_lna_gain(dev, gain_lna);
    if (result != AIRSPY_SUCCESS) {
        ERROR("airspy_set_lna_gain() failed: {:s} ({:d})",
              airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }


    result = airspy_set_rf_bias(dev, biast_power);
    if (result != AIRSPY_SUCCESS) {
        ERROR("airspy_set_rf_bias() failed: {:s} ({:d})",
              airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }

    result = airspy_board_id_read(dev, &board_id);
    if (result != AIRSPY_SUCCESS) {
        ERROR("airspy_board_id_read() failed: {:s} ({:d})",
              airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }
    INFO("Board ID Number: {:d} ({:s})", board_id,
         airspy_board_id_name((enum airspy_board_id)board_id));

    airspy_read_partid_serialno_t read_partid_serialno;
    result = airspy_board_partid_serialno_read(dev, &read_partid_serialno);
    if (result != AIRSPY_SUCCESS) {
        ERROR("airspy_board_partid_serialno_read() failed: {:s} ({:d})",
              airspy_error_name((enum airspy_error)result), result);
        return nullptr;
    }
    INFO("Part ID Number: {:#08X} {:#08X}", read_partid_serialno.part_id[0],
         read_partid_serialno.part_id[1]);
    INFO("Serial Number: {:#08X}{:08X}", read_partid_serialno.serial_no[2],
         read_partid_serialno.serial_no[3]);

    return dev;
}
