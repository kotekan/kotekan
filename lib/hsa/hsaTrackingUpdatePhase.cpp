// curl localhost:12048/updatable_config/pulsar_pointing/0  -X POST -H 'Content-Type:
// application/json' -d '{"ra":100.3, "dec":34.23, "scaling":99.0}'

#include "hsaTrackingUpdatePhase.hpp"

#include "Config.hpp" // for Config
#include "Telescope.hpp"
#include "buffer.h"               // for mark_frame_empty, Buffer, register_consumer, wait_for...
#include "bufferContainer.hpp"    // for bufferContainer
#include "configUpdater.hpp"      // for configUpdater
#include "hsaBase.h"              // for hsa_host_free, hsa_host_malloc
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface
#include "kotekanLogging.hpp"     // for DEBUG, WARN, ERROR, INFO
#include "restServer.hpp"         // for restServer, HTTP_RESPONSE, connectionInstance
#include "visUtil.hpp"            // for double_to_ts

#include <algorithm>  // for clamp
#include <cmath>      // for cos, sin, fmod, acos, asin, sqrt, atan2, pow
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <functional> // for placeholders
#include <regex>      // for match_results<>::_Base_type
#include <string.h>   // for memcpy
#include <string>     // for string, allocator, operator+, to_string, char_traits
#include <time.h>     // for tm, timespec, localtime
#include <vector>     // for vector

#define PI 3.14159265
#define light 299792458.
#define one_over_c 0.0033356
#define R2D 180. / PI
#define D2R PI / 180.
#define TAU 2 * PI
#define inst_long -119.6175
#define inst_lat 49.3203

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_HSA_COMMAND(hsaTrackingUpdatePhase);

hsaTrackingUpdatePhase::hsaTrackingUpdatePhase(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "", "") {

    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_beams = config.get<int16_t>(unique_name, "num_beams");

    _feed_sep_NS = config.get<float>(unique_name, "feed_sep_NS");
    _feed_sep_EW = config.get<int32_t>(unique_name, "feed_sep_EW");

    // Just for metadata manipulation
    metadata_buf = host_buffers.get_buffer("network_buf");
    metadata_buffer_id = 0;
    metadata_buffer_precondition_id = 0;
    register_consumer(metadata_buf, unique_name.c_str());
    freq_idx = FREQ_ID_NOT_SET;
    freq_MHz = -1;

    first_pass = true;

    // Gain stuff here
    gain_len = 2 * 2048 * _num_beams * sizeof(float);
    host_gain = (float*)hsa_host_malloc(gain_len, device.get_gpu_numa_node());
    gain_buf = host_buffers.get_buffer("gain_psr_buf");
    register_consumer(gain_buf, unique_name.c_str());
    gain_buf_id = 0;

    // Phase here
    phase_frame_len = _num_elements * _num_beams * 2 * sizeof(float);
    // Two alternating banks
    host_phase_0 = (float*)hsa_host_malloc(phase_frame_len, device.get_gpu_numa_node());
    host_phase_1 = (float*)hsa_host_malloc(phase_frame_len, device.get_gpu_numa_node());
    int index = 0;
    for (uint b = 0; b < _num_beams * _num_elements; b++) {
        host_phase_0[index++] = 0;
        host_phase_0[index++] = 0;
    }

    bankID = (uint*)hsa_host_malloc(device.get_gpu_buffer_depth(), device.get_gpu_numa_node());
    bank_use_0 = 0;
    bank_use_1 = 0;
    second_last = 0;

    // Register function to listen for new pulsar, and update ra and dec
    using namespace std::placeholders;
    for (int beam_id = 0; beam_id < _num_beams; beam_id++) {
        configUpdater::instance().subscribe(
            config.get<std::string>(unique_name, "updatable_config/psr_pt") + "/"
                + std::to_string(beam_id),
            [beam_id, this](nlohmann::json& json_msg) -> bool {
                return pulsar_grab_callback(json_msg, beam_id);
            });
    }
}

hsaTrackingUpdatePhase::~hsaTrackingUpdatePhase() {
    restServer::instance().remove_json_callback(endpoint_psrcoord);
    hsa_host_free(host_phase_0);
    hsa_host_free(host_phase_1);
    hsa_host_free(bankID);
    hsa_host_free(host_gain);
}

int hsaTrackingUpdatePhase::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    uint8_t* frame =
        wait_for_full_frame(metadata_buf, unique_name.c_str(), metadata_buffer_precondition_id);
    if (frame == nullptr)
        return -1;
    metadata_buffer_precondition_id =
        (metadata_buffer_precondition_id + 1) % metadata_buf->num_frames;


    // Wait for new gain
    if (first_pass) {
        uint8_t* frame = wait_for_full_frame(gain_buf, unique_name.c_str(), gain_buf_id);
        if (frame == nullptr)
            return -1;
        DEBUG("Applying inital host gains from {:s}[{:d}]", gain_buf->buffer_name, gain_buf_id);
        std::lock_guard<std::mutex> lock(_pulsar_lock);
        memcpy(host_gain, (float*)gain_buf->frames[gain_buf_id], gain_len);
        update_phase = true;
        mark_frame_empty(gain_buf, unique_name.c_str(), gain_buf_id);
        gain_buf_id = (gain_buf_id + 1) % gain_buf->num_frames;
    } else {
        auto timeout = double_to_ts(0);
        int status =
            wait_for_full_frame_timeout(gain_buf, unique_name.c_str(), gain_buf_id, timeout);
        if (status == 0) {
            DEBUG("Applying new host gains from {:s}[{:d}]", gain_buf->buffer_name, gain_buf_id);
            std::lock_guard<std::mutex> lock(_pulsar_lock);
            memcpy(host_gain, (float*)gain_buf->frames[gain_buf_id], gain_len);
            update_phase = true;
            mark_frame_empty(gain_buf, unique_name.c_str(), gain_buf_id);
            gain_buf_id = (gain_buf_id + 1) % gain_buf->num_frames;
        }
        if (status == -1)
            return -1;
    }
    return 0;
}

void hsaTrackingUpdatePhase::calculate_phase(struct psrCoord psr_coord, timespec time_now,
                                             float freq_now, float* gains, float* output) {

    float FREQ = freq_now;
    struct tm* timeinfo;
    timeinfo = localtime(&time_now.tv_sec);
    uint year = timeinfo->tm_year + 1900;
    uint month = timeinfo->tm_mon + 1;
    if (month < 3) {
        month = month + 12;
        year = year - 1;
    }
    uint day = timeinfo->tm_mday;
    float JD = 2 - int(year / 100.) + int(int(year / 100.) / 4.) + int(365.25 * year)
               + int(30.6001 * (month + 1)) + day + 1720994.5;
    double T = (JD - 2451545.0)
               / 36525.0; // Works if time after year 2000, otherwise T is -ve and might break
    double T0 = fmod((6.697374558 + (2400.051336 * T) + (0.000025862 * T * T)), 24.);
    double UT = (timeinfo->tm_hour) + (timeinfo->tm_min / 60.)
                + (timeinfo->tm_sec + time_now.tv_nsec / 1.e9) / 3600.;
    double GST = fmod((T0 + UT * 1.002737909), 24.);
    double LST = GST + inst_long / 15.;
    while (LST < 0) {
        LST = LST + 24;
    }
    LST = fmod(LST, 24);
    for (int b = 0; b < _num_beams; b++) {
        if (psr_coord.scaling[b] == 1) {
            for (uint32_t i = 0; i < _num_elements * 2; ++i) {
                output[b * _num_elements * 2 + i] = gains[b * _num_elements * 2 + i];
            }
            continue;
        }
        double hour_angle = LST * 15. - psr_coord.ra[b];
        double alt = sin(psr_coord.dec[b] * D2R) * sin(inst_lat * D2R)
                     + cos(psr_coord.dec[b] * D2R) * cos(inst_lat * D2R) * cos(hour_angle * D2R);
        alt = asin(std::clamp(alt, -1.0, 1.0));
        double az = (sin(psr_coord.dec[b] * D2R) - sin(alt) * sin(inst_lat * D2R))
                    / (cos(alt) * cos(inst_lat * D2R));
        az = acos(std::clamp(az, -1.0, 1.0));
        if (sin(hour_angle * D2R) >= 0) {
            az = TAU - az;
        }
        double projection_angle, effective_angle, offset_distance;
        for (int i = 0; i < 4; i++) {       // loop 4 cylinders
            for (int j = 0; j < 256; j++) { // loop 256 feeds
                float dist_y = j * _feed_sep_NS;
                float dist_x = i * _feed_sep_EW;
                projection_angle = 90 * D2R - atan2(dist_y, dist_x);
                offset_distance = sqrt(pow(dist_y, 2) + pow(dist_x, 2));
                effective_angle = projection_angle - az;
                float delay_real = cos(TAU * cos(effective_angle) * cos(alt) * offset_distance
                                       * FREQ * one_over_c);
                float delay_imag = -sin(TAU * cos(effective_angle) * cos(-alt) * offset_distance
                                        * FREQ * one_over_c);
                for (int p = 0; p < 2; p++) { // loop 2 pol
                    uint elem_id = p * 1024 + i * 256 + j;
                    // Not scrembled, assume reordering kernel has been run
                    output[(b * _num_elements + elem_id) * 2] =
                        delay_real * gains[(b * _num_elements + elem_id) * 2]
                        - delay_imag * gains[(b * _num_elements + elem_id) * 2 + 1];
                    output[(b * _num_elements + elem_id) * 2 + 1] =
                        delay_real * gains[(b * _num_elements + elem_id) * 2 + 1]
                        + delay_imag * gains[(b * _num_elements + elem_id) * 2];
                }
            }
        }
    }
}

hsa_signal_t hsaTrackingUpdatePhase::execute(int gpu_frame_id, hsa_signal_t precede_signal) {
    // Update phase every one second
    const uint64_t phase_update_period = 390625;
    uint64_t current_seq = get_fpga_seq_num(metadata_buf, metadata_buffer_id);
    second_now = (current_seq / phase_update_period) % 2;
    if (second_now != second_last) {
        update_phase = true;
    }
    second_last = second_now;

    if (first_pass) {
        first_pass = false;
        // From the metadata, figure out the frequency
        auto& tel = Telescope::instance();
        freq_idx = tel.to_freq_id(metadata_buf, metadata_buffer_id);
        freq_MHz = tel.to_freq(freq_idx);
        update_phase = true;
    }

    if (update_phase) {
        // GPS time, need ch_master
        DEBUG("updating phase gain={:f} {:f}", host_gain[0], host_gain[1]);
        time_now_gps = get_gps_time(metadata_buf, metadata_buffer_id);
        if (time_now_gps.tv_sec == 0) {
            ERROR("GPS time appears to be zero, bad news for pulsar timing!");
        }
        // use whichever bank that has no lock
        if (bank_use_0 == 0) { // no more outstanding async copy using bank0
            std::lock_guard<std::mutex> lock(_pulsar_lock);
            psr_coord = psr_coord_latest_update;
            calculate_phase(psr_coord, time_now_gps, freq_MHz, host_gain, host_phase_0);
            bank_active = 0;
            update_phase = false;
        } else if (bank_use_1 == 0) { // no more outstanding async copy using bank1
            std::lock_guard<std::mutex> lock(_pulsar_lock);
            psr_coord = psr_coord_latest_update;
            calculate_phase(psr_coord, time_now_gps, freq_MHz, host_gain, host_phase_1);
            bank_active = 1;
            update_phase = false;
        }
    }

    bankID[gpu_frame_id] = bank_active; // update or not, read from the latest bank
    set_psr_coord(metadata_buf, metadata_buffer_id, psr_coord);
    mark_frame_empty(metadata_buf, unique_name.c_str(), metadata_buffer_id);
    metadata_buffer_id = (metadata_buffer_id + 1) % metadata_buf->num_frames;

    // Do the data copy. Now I am doing async everytime there is new data
    //(i.e., when main_thread is being called, in principle I just need to copy in
    // when there is an update, which is of slower cadence. Down the road optimization

    // Get the gpu memory pointer. i will need multiple frame through the use of get_gpu_mem_array,
    // because while it has been sent away for async copy, the next update might be happening.
    void* gpu_memory_frame =
        device.get_gpu_memory_array("beamform_phase", gpu_frame_id, phase_frame_len);

    if (bankID[gpu_frame_id] == 0) {
        device.async_copy_host_to_gpu(gpu_memory_frame, (void*)host_phase_0, phase_frame_len,
                                      precede_signal, signals[gpu_frame_id]);
        bank_use_0 = bank_use_0 + 1;
    }
    if (bankID[gpu_frame_id] == 1) {
        device.async_copy_host_to_gpu(gpu_memory_frame, (void*)host_phase_1, phase_frame_len,
                                      precede_signal, signals[gpu_frame_id]);
        bank_use_1 = bank_use_1 + 1;
    }
    return signals[gpu_frame_id];
}

void hsaTrackingUpdatePhase::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);
    if (bankID[frame_id] == 1) {
        bank_use_1 = bank_use_1 - 1;
    }
    if (bankID[frame_id] == 0) {
        bank_use_0 = bank_use_0 - 1;
    }
}

bool hsaTrackingUpdatePhase::pulsar_grab_callback(nlohmann::json& json, const uint8_t beam_id) {
    {
        std::lock_guard<std::mutex> lock(_pulsar_lock);
        try {
            psr_coord_latest_update.ra[beam_id] = json.at("ra").get<float>();
        } catch (std::exception const& e) {
            WARN("[PSR] Pointing update fail to read RA {:s}", e.what());
            return false;
        }
        try {
            psr_coord_latest_update.dec[beam_id] = json.at("dec").get<float>();
        } catch (std::exception const& e) {
            WARN("[PSR] Pointing update fail to read DEC {:s}", e.what());
            return false;
        }
        try {
            psr_coord_latest_update.scaling[beam_id] = json.at("scaling").get<int>();
        } catch (std::exception const& e) {
            WARN("[PSR] Pointing update fail to read scaling factor {:s}", e.what());
            return false;
        }
        INFO("[psr] Updated Beam={:d} RA={:.2f} Dec={:.2f} Scl={:d}", beam_id,
             psr_coord_latest_update.ra[beam_id], psr_coord_latest_update.dec[beam_id],
             psr_coord_latest_update.scaling[beam_id]);
        update_phase = true;
    }
    return true;
}
