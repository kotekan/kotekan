#include "airspyInput.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG
#include "restServer.hpp"      // for connectionInstance
#include <fcntl.h>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

REGISTER_KOTEKAN_STAGE(airspyInput);

airspyInput::airspyInput(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&airspyInput::main_thread, this)) {

    buf = get_buffer("out_buf");
    register_producer(buf, unique_name.c_str());

    freq = config.get_default<float>(unique_name, "freq", 1420) * 1000000;          // MHz
    sample_rate = config.get_default<float>(unique_name, "sample_bw", 2.5) * 1e6;   // IQ MSPS = BW MHz
    gain_lna = config.get_default<int>(unique_name, "gain_lna", 5);                 // MAX: 14
    gain_if = config.get_default<int>(unique_name, "gain_if", 5);                   // MAX: 15
    gain_mix = config.get_default<int>(unique_name, "gain_mix", 5);                 // MAX: 15
 
    airspy_sn = config.get_default<long>(unique_name, "serial", 0);
    airspy_fn = config.get_default<std::string>(unique_name, "airspy_file", "");

    biast_power = config.get_default<bool>(unique_name, "biast_power", false) ? 1 : 0;
    autostart = config.get_default<bool>(unique_name, "autostart", true) ? 1 : 0;
}

airspyInput::~airspyInput() {
    if (a_device != nullptr) {
        airspy_stop_rx(a_device);
        airspy_close(a_device);
    }
    airspy_exit();
}

void airspyInput::adcstat_callback(kotekan::connectionInstance& conn) {
    dump_adcstat = true;
    while (!adcstat_ready) {usleep(1000);}

    nlohmann::json reply;
    reply["rms"] = adcrms;
    reply["mean"] = adcmean;
    reply["railfrac"] = adcrailfrac;

    conn.send_json_reply(reply);
    adcstat_ready = false;
}

void airspyInput::rest_callback(kotekan::connectionInstance& conn,
                                   nlohmann::json& json_request) {
    int err;
    bool success=false;
    // need a lock here?
    try {
        freq = ((float)json_request["freq"]) * 1000000;
        INFO("Updating airspy LO to frequency to {:d}", freq);

        err=airspy_set_freq(dev, freq);
        if (err != AIRSPY_SUCCESS) {
            ERROR("airspy_set_freq() failed: {:s} ({:d})", airspy_error_name((enum airspy_error)err),
                  err)
        }
        else success=true;
    } catch (...) {}
    try {
        gain_lna = json_request["gain_lna"];
        INFO("Updating airspy LNA gain to {:d}", gain_lna);

        err = airspy_set_lna_gain(dev, gain_lna);
        if (err != AIRSPY_SUCCESS) {
            ERROR("airspy_set_lna_gain() failed: {:s} ({:d})", airspy_error_name((enum airspy_error)err),
                  err)
        }
        else success=true;
    } catch (...) {}
    try {
        gain_mix = json_request["gain_mix"];
        INFO("Updating airspy mixer gain to {:d}", gain_mix);

        err = airspy_set_mixer_gain(dev, gain_mix);
        if (err != AIRSPY_SUCCESS) {
            ERROR("airspy_set_mixer_gain() failed: {:s} ({:d})", airspy_error_name((enum airspy_error)err),
                  err)
        }
        else success=true;
    } catch (...) {}
    try {
        gain_if = json_request["gain_if"];
        INFO("Updating airspy IF gain to {:d}", gain_if);

        err = airspy_set_vga_gain(dev, gain_if);
        if (err != AIRSPY_SUCCESS) {
            ERROR("airspy_set_vga_gain() failed: {:s} ({:d})", airspy_error_name((enum airspy_error)err),
                  err)
        }
        else success=true;
    } catch (...) {}
    try {
        gain_if = json_request["gain_if"];
        INFO("Updating airspy IF gain to {:d}", gain_if);

        err = airspy_start_rx(a_device, airspy_callback, static_cast<void*>(this));
        if (err != AIRSPY_SUCCESS) {
            ERROR("airspy_start_rx() failed: {:s} ({:d})", airspy_error_name((enum airspy_error)err),
                  err)
        }
        else success=true;
    } catch (...) {}
    try {
        int add_lag = json_request["add_lag"];
        INFO("Updating airspy lag by {:d}", add_lag);
        success=true;
        pthread_mutex_lock(&recv_busy);
        lag += add_lag*BYTES_PER_SAMPLE;
        pthread_mutex_unlock(&recv_busy);
    } catch (...) {}

    if (success) {
        usleep(10000);
//        adcstat_callback(conn);
        conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
//        dump_rms = true;
    }
    else {
        conn.send_error("Couldn't parse airspy rx parameters.\n", kotekan::HTTP_RESPONSE::BAD_REQUEST);
    }
}

void airspyInput::main_thread() {
    int err;
    std::string endpoint = unique_name+"/config";
    using namespace std::placeholders;
    kotekan::restServer& rest_server = kotekan::restServer::instance();
    rest_server.register_post_callback(endpoint,
                                      std::bind(&airspyInput::rest_callback, this, _1, _2));

    endpoint = unique_name+"/adcstat";
    rest_server.register_get_callback(endpoint,
                                      std::bind(&airspyInput::adcstat_callback, this, _1));


    frame_id = 0;
    frame_loc = 0;
    recv_busy = PTHREAD_MUTEX_INITIALIZER;

    airspy_init();
    a_device = init_device();
    if (a_device == nullptr) {
        FATAL_ERROR("Error in airspyInput. Cannot find device.");
        return;
    }
    if (autostart) {
        err = airspy_start_rx(a_device, airspy_callback, static_cast<void*>(this));
        if (err != AIRSPY_SUCCESS) {
            ERROR("airspy_start_rx() failed: {:s} ({:d})", airspy_error_name((enum airspy_error)err),
                  err)
        }
    }
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
        if (lag > 0) {
            if (lag > bt){
                lag -= bt;
                bt = 0;
                continue;
            }
            else {
                bt-= lag;
                in = (void*)((char*)in + lag);
                lag=0;
            }
        }
        if (frame_loc == 0) {
            DEBUG("Airspy waiting for frame_id {:d}", frame_id);
            frame_ptr = (unsigned char*)wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
            if (frame_ptr == nullptr)
                break;
        }

        size_t copy_length = bt < (buf->frame_size-frame_loc) ? bt : (buf->frame_size-frame_loc);
        DEBUG("Filling Buffer {:d} With {:d} Data Samples {}", frame_id, copy_length / BYTES_PER_SAMPLE, transfer->sample_count);

        // FILL THE BUFFER
        memcpy(frame_ptr + frame_loc, in, copy_length);
        bt -= copy_length;
        in = (void*)((char*)in + copy_length);
        frame_loc = (frame_loc + copy_length) % buf->frame_size;

        if (frame_loc == 0) {
            DEBUG("Airspy Buffer {:d} Full", frame_id);
            if (dump_adcstat){
                float mean = 0, rms = 0;
                float rail = 0;
                short *fr = (short*)frame_ptr;
                for (uint i=0; i<buf->frame_size/BYTES_PER_SAMPLE; i++) if (abs(fr[i]) >= (2<<10)) rail++;
                rail/=buf->frame_size/BYTES_PER_SAMPLE;
                for (uint i=0; i<buf->frame_size/BYTES_PER_SAMPLE; i++) mean+=(float)fr[i];
                mean/=buf->frame_size/BYTES_PER_SAMPLE;
                for (uint i=0; i<buf->frame_size/BYTES_PER_SAMPLE; i++) rms+=((float)fr[i]-mean)*((float)fr[i]-mean);
                rms=sqrt(rms/(buf->frame_size/BYTES_PER_SAMPLE));
                adcrailfrac=rail;
                adcrms=rms;
                adcmean=mean;
#ifndef IQ_SAMPLING
                adcmean-=2048;
#endif
                INFO("Airspy ADC mean: {:f}, RMS: {:f}, rail fraction {:f}",adcmean,adcrms,adcrailfrac);
                adcstat_ready=true;
                dump_adcstat=false;
            }
#ifndef IQ_SAMPLING
            short *fr = (short*)frame_ptr;
            for (uint i=0; i<buf->frame_size/BYTES_PER_SAMPLE; i++) fr[i]-=2048;
#endif
            mark_frame_full(buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % buf->num_frames;
        }
    }
    pthread_mutex_unlock(&recv_busy);
}

struct airspy_device* airspyInput::init_device() {
    int result;
    uint8_t board_id = AIRSPY_BOARD_ID_INVALID;

    if (airspy_sn) {
        result = airspy_open_sn(&dev,airspy_sn);
    }
    else if (not airspy_fn.empty()) {
        int airspy_fd = open(airspy_fn.c_str(), O_RDWR);
        if (airspy_fd == -1) {
            ERROR("Error opening file: {:s}\n",airspy_fn);
            return nullptr;
        }
        result = airspy_open_sn(&dev,airspy_fd);
        close(airspy_fd);
    }
    else {
        result = airspy_open(&dev);
    }

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
            if (supported_samplerates[i] == sample_rate)
                samplerate_idx = i;
        }
        if (samplerate_idx < 0) {
            ERROR("Unsupported sample rate: {:f} Hz", sample_rate);
            return nullptr;
        }
        INFO("Selected sample rate: {:d} Hz -> idx {:d}", sample_rate, samplerate_idx)
        result = airspy_set_samplerate(dev, samplerate_idx);
        if (result != AIRSPY_SUCCESS) {
            ERROR("airspy_set_samplerate() failed: {:s} ({:d})",
                  airspy_error_name((enum airspy_error)result), result);
            return nullptr;
        }
    }

#ifdef IQ_SAMPLING
    result = airspy_set_sample_type(dev, AIRSPY_SAMPLE_INT16_IQ);
#else
    result = airspy_set_sample_type(dev, AIRSPY_SAMPLE_RAW);
#endif
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
