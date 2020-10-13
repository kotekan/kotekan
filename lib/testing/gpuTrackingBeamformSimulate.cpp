#include "gpuTrackingBeamformSimulate.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"       // for Telescope, FREQ_ID_NOT_SET
#include "buffer.h"            // for Buffer, mark_frame_empty, mark_frame_full, pass_metadata
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.h"     // for beamCoord, get_fpga_seq_num, get_gps_time
#include "kotekanLogging.hpp"  // for INFO, ERROR

#include <algorithm>  // for copy
#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cmath>      // for cos, sin, fmod, pow, acos, asin, atan2, sqrt
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <memory>     // for allocator_traits<>::value_type
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdio.h>    // for fclose, fopen, fread, snprintf, FILE
#include <stdlib.h>   // for free, malloc
#include <string.h>   // for memcpy
#include <time.h>     // for tm, timespec, localtime


#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b)&0x0F)

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
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(gpuTrackingBeamformSimulate);

gpuTrackingBeamformSimulate::gpuTrackingBeamformSimulate(Config& config,
                                                         const std::string& unique_name,
                                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&gpuTrackingBeamformSimulate::main_thread, this)) {

    // Apply config.
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _num_pol = config.get<int32_t>(unique_name, "num_pol");
    _num_beams = config.get<int32_t>(unique_name, "num_beams");
    _feed_sep_NS = config.get<float>(unique_name, "feed_sep_NS");
    _feed_sep_EW = config.get<int32_t>(unique_name, "feed_sep_EW");
    _source_ra = config.get<std::vector<float>>(unique_name, "source_ra");
    _source_dec = config.get<std::vector<float>>(unique_name, "source_dec");
    _reorder_map = config.get<std::vector<int32_t>>(unique_name, "reorder_map");
    _gain_dir =
        config.get<std::vector<std::string>>(unique_name, "tracking_gain/tracking_gain_dir");
    INFO("[TRACKING CPU] start with gain {:s} {:s} {:s}", _gain_dir[0], _gain_dir[1], _gain_dir[2]);
    std::vector<float> dg = {0.0, 0.0}; // re,im
    default_gains = config.get_default<std::vector<float>>(unique_name, "frb_missing_gains", dg);


    input_buf = get_buffer("network_in_buf");
    register_consumer(input_buf, unique_name.c_str());
    output_buf = get_buffer("beam_out_buf");
    register_producer(output_buf, unique_name.c_str());

    input_len = _samples_per_data_set * _num_elements * 2;
    output_len = _samples_per_data_set * _num_beams * _num_pol * 2;

    input_unpacked = (double*)malloc(input_len * sizeof(double));
    phase = (double*)malloc(_num_elements * _num_beams * 2 * sizeof(double));
    cpu_output = (float*)malloc(output_len * sizeof(float));
    assert(phase != nullptr);

    cpu_gain = (float*)malloc(2 * _num_elements * _num_beams * sizeof(float));
    second_last = 0;
    metadata_buf = get_buffer("network_in_buf");
    metadata_buffer_id = 0;
    metadata_buffer_precondition_id = 0;
    freq_now = FREQ_ID_NOT_SET;
    freq_MHz = -1;

    // Backward compatibility, array in c
    reorder_map_c = (int*)malloc(512 * sizeof(int));
    for (uint i = 0; i < 512; ++i) {
        reorder_map_c[i] = _reorder_map[i];
    }

    for (int i = 0; i < _num_beams; i++) {
        beam_coord.ra[i] = _source_ra[i];
        beam_coord.dec[i] = _source_dec[i];
    }
}

gpuTrackingBeamformSimulate::~gpuTrackingBeamformSimulate() {
    free(input_unpacked);
    free(cpu_output);
    free(phase);
    free(cpu_gain);
    free(reorder_map_c);
}

void gpuTrackingBeamformSimulate::reorder(unsigned char* data, int* map) {
    int* tmp512;
    tmp512 = (int*)malloc(_num_elements * sizeof(int));
    for (int j = 0; j < _samples_per_data_set; j++) {
        for (int i = 0; i < 512; i++) {
            int id = map[i];
            tmp512[i * 4] = data[j * _num_elements + (id * 4)];
            tmp512[i * 4 + 1] = data[j * _num_elements + (id * 4 + 1)];
            tmp512[i * 4 + 2] = data[j * _num_elements + (id * 4 + 2)];
            tmp512[i * 4 + 3] = data[j * _num_elements + (id * 4 + 3)];
        }
        for (uint i = 0; i < _num_elements; i++) {
            data[j * _num_elements + i] = tmp512[i];
        }
    }
    free(tmp512);
}

void gpuTrackingBeamformSimulate::calculate_phase(struct beamCoord beam_coord, timespec time_now,
                                                  float freq_now, float* gains, double* output) {
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
        double hour_angle = LST * 15. - beam_coord.ra[b];
        double alt = sin(beam_coord.dec[b] * D2R) * sin(inst_lat * D2R)
                     + cos(beam_coord.dec[b] * D2R) * cos(inst_lat * D2R) * cos(hour_angle * D2R);
        alt = asin(alt);
        double az = (sin(beam_coord.dec[b] * D2R) - sin(alt) * sin(inst_lat * D2R))
                    / (cos(alt) * cos(inst_lat * D2R));
        az = acos(az);
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

void gpuTrackingBeamformSimulate::cpu_tracking_beamformer(double* input_unpacked, double* phase,
                                                          float* cpu_output,
                                                          int _samples_per_data_set,
                                                          int _num_elements, int _num_beams,
                                                          int _num_pol) {
    float sum_re, sum_im;
    for (int t = 0; t < _samples_per_data_set; t++) {
        for (int b = 0; b < _num_beams; b++) {
            for (int p = 0; p < _num_pol; p++) {
                sum_re = 0.0;
                sum_im = 0.0;
                for (int n = 0; n < 1024; n++) {
                    // Input and phase both have pol as second fastest
                    sum_re += input_unpacked[(t * _num_elements + p * 1024 + n) * 2]
                                  * phase[((b * _num_pol + p) * 1024 + n) * 2]
                              + input_unpacked[(t * _num_elements + p * 1024 + n) * 2 + 1]
                                    * phase[((b * _num_pol + p) * 1024 + n) * 2 + 1];
                    sum_im += (input_unpacked[(t * _num_elements + p * 1024 + n) * 2 + 1]
                                   * phase[((b * _num_pol + p) * 1024 + n) * 2]
                               - input_unpacked[(t * _num_elements + p * 1024 + n) * 2]
                                     * phase[((b * _num_pol + p) * 1024 + n) * 2 + 1]);
                }
                cpu_output[(t * _num_beams * _num_pol + b * _num_pol + p) * 2] = sum_re;
                cpu_output[(t * _num_beams * _num_pol + b * _num_pol + p) * 2 + 1] = sum_im;
                // Output has polarization as fastest varying
            }
        }
    }
}

void gpuTrackingBeamformSimulate::main_thread() {

    auto& tel = Telescope::instance();

    int input_buf_id = 0;
    int output_buf_id = 0;

    while (!stop_thread) {

        unsigned char* input =
            (unsigned char*)wait_for_full_frame(input_buf, unique_name.c_str(), input_buf_id);
        if (input == nullptr)
            break;
        float* output =
            (float*)wait_for_empty_frame(output_buf, unique_name.c_str(), output_buf_id);
        if (output == nullptr)
            break;

        for (int i = 0; i < input_len; i++) {
            input_unpacked[i] = 0.0; // Need this
        }
        for (int i = 0; i < output_len; i++) {
            cpu_output[i] = 0;
        }

        INFO("Simulating GPU tracking beamform processing for {:s}[{:d}] putting result in "
             "{:s}[{:d}]",
             input_buf->buffer_name, input_buf_id, output_buf->buffer_name, output_buf_id);

        // Figure out which freq
        freq_now = tel.to_freq_id(metadata_buf, metadata_buffer_id);
        freq_MHz = tel.to_freq(freq_now);
        INFO("[CPU] freq_now={:d} freq+MHz={:.2f} metadat_buf_id={:d} input_buf_id={:d}", freq_now,
             freq_MHz, metadata_buffer_id, input_buf_id);

        // Load in gains (without dynamic update capability like in GPU, just get set once in the
        // beginning)
        FILE* ptr_myfile = nullptr;
        char filename[512];
        for (int b = 0; b < _num_beams; b++) {
            snprintf(filename, sizeof(filename), "%s/quick_gains_%04d_reordered.bin",
                     _gain_dir[b].c_str(), freq_now);
            ptr_myfile = fopen(filename, "rb");
            if (ptr_myfile == nullptr) {
                ERROR("CPU verification code: Cannot open gain file {:s}", filename);
                for (uint i = 0; i < _num_elements; i++) {
                    cpu_gain[(b * _num_elements + i) * 2] = default_gains[0];
                    cpu_gain[(b * _num_elements + i) * 2 + 1] = default_gains[1];
                }
            } else {
                if (_num_elements
                    != fread(&cpu_gain[b * _num_elements * 2], sizeof(float) * 2, _num_elements,
                             ptr_myfile)) {
                    ERROR("Couldn't read gain file...");
                }
                fclose(ptr_myfile);
            }
        }
        INFO("[CPU] gain {:.2f} {:.2f} {:.2f} {:.2f}", cpu_gain[0], cpu_gain[1], cpu_gain[2],
             cpu_gain[3]);

        // Update phase every one second
        const uint64_t phase_update_period = 390625;
        uint64_t current_seq = get_fpga_seq_num(metadata_buf, metadata_buffer_id);
        second_now = (current_seq / phase_update_period) % 2;
        if (second_now != second_last) {
            update_phase = true;
        }
        second_last = second_now;

        if (update_phase) {
            // GPS time, need ch_master
            time_now_gps = get_gps_time(metadata_buf, metadata_buffer_id);
            if (time_now_gps.tv_sec == 0) {
                ERROR("GPS time appears to be zero, bad news for tracking timing!");
            }
            calculate_phase(beam_coord, time_now_gps, freq_MHz, cpu_gain, phase);
            update_phase = false;
        }

        // Reorder
        reorder(input, reorder_map_c);

        // Unpack input data
        int dest_idx = 0;
        for (int i = 0; i < input_buf->frame_size; ++i) {
            input_unpacked[dest_idx++] = HI_NIBBLE(input[i]) - 8;
            input_unpacked[dest_idx++] = LO_NIBBLE(input[i]) - 8;
        }

        // Beamform 10 trackings.
        cpu_tracking_beamformer(input_unpacked, phase, cpu_output, _samples_per_data_set,
                                _num_elements, _num_beams, _num_pol);
        memcpy(output, cpu_output, output_buf->frame_size);

        INFO(
            "Simulating GPU tracking beamform processing done for {:s}[{:d}] result is in {:s}[{:d}"
            "]",
            input_buf->buffer_name, input_buf_id, output_buf->buffer_name, output_buf_id);

        pass_metadata(input_buf, input_buf_id, output_buf, output_buf_id);
        mark_frame_empty(input_buf, unique_name.c_str(), input_buf_id);
        mark_frame_full(output_buf, unique_name.c_str(), output_buf_id);

        input_buf_id = (input_buf_id + 1) % input_buf->num_frames;
        metadata_buffer_id = (metadata_buffer_id + 1) % metadata_buf->num_frames;
        output_buf_id = (output_buf_id + 1) % output_buf->num_frames;
    }
}
