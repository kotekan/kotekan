#include "clBeamformPhaseData.hpp"

#include "buffer.hpp"
#include "chimeMetadata.hpp"
#include "errors.h"

#include <math.h>
#include <time.h>

#define D2R 0.01745329252 // pi/180
#define TAU 6.28318530718 // 2*pi

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CL_COMMAND(clBeamformPhaseData);

clBeamformPhaseData::clBeamformPhaseData(Config& config, const std::string& unique_name,
                                         bufferContainer& host_buffers, clDeviceInterface& device,
                                         int inst) :
    clCommand(config, unique_name, host_buffers, device, inst, no_cl_command_state, "", "") {
    command_type = gpuCommandType::NOT_SET;

    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    network_buf = host_buffers.get_buffer("network_buf");

    beamforming_do_not_track = config.get_default<bool>(unique_name, "do_not_track", false);
    inst_lat = config.get<double>(unique_name, "instrument_lat");
    inst_long = config.get<double>(unique_name, "instrument_long");
    fixed_time = config.get<int>(unique_name, "fixed_time");
    ra = config.get<double>(unique_name, "ra");
    dec = config.get<double>(unique_name, "dec");
    feed_positions = config.get<std::vector<float>>(unique_name, "element_positions");
}

void clBeamformPhaseData::build() {
    clCommand::build();

    last_bankID = -1;

    // Create two phase banks
    phases[0] = (float*)malloc(_num_elements * sizeof(float));
    phases[1] = (float*)malloc(_num_elements * sizeof(float));

    start_beamform_time = time(nullptr); // Current time.
}

cl_event clBeamformPhaseData::execute(cl_event pre_event) {
    gpuCommand::pre_execute();

    time_t local_beamform_time;
    uint64_t current_seq;
    // TODO Make this a config file option
    // 390625 == 1 second.
    const uint64_t phase_update_period = 390625;

    // Update the phases only every "phase_update_period"
    //    uint32_t input_frame_len =  _num_elements * _num_local_freq * _samples_per_data_set;
    //    cl_mem input_memory = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);

    current_seq = get_fpga_seq_num(network_buf, gpu_frame_id);
    int64_t bankID = (current_seq / phase_update_period) % 2;
    cl_mem phase_memory =
        device.get_gpu_memory_array("phases", bankID, _num_elements * sizeof(float));

    if (bankID != last_bankID) {

        if (beamforming_do_not_track == 1) {
            if (fixed_time != 0) {
                local_beamform_time = fixed_time;
            } else {
                local_beamform_time = start_beamform_time; // Current time.
            }
        } else {
            local_beamform_time = get_first_packet_recv_time(network_buf, gpu_frame_id).tv_sec;
        }

        get_delays(phases[bankID], local_beamform_time);

        CHECK_CL_ERROR(clEnqueueWriteBuffer(device.getQueue(0), phase_memory, CL_FALSE, 0,
                                            _num_elements * sizeof(float),
                                            (cl_float*)phases[bankID], 1, &pre_event, &post_event));

        last_bankID = bankID;
        return post_event;
    }
    return pre_event;
}

void clBeamformPhaseData::get_delays(float* phases, time_t beamform_time) {
    // inverse speed of light in ns/m
    const double one_over_c = 3.3356;
    // offset of initial lst phase in degrees <-------------- PROBABLY WRONG PLEASE CHECK
    const double phi_0 = 280.46;
    // const double phi_0 = 160.64;
    // rate of change of LST with time in s
    const double lst_rate = 360. / 86164.09054;
    // UNIX timestamp of J2000 epoch time
    const double j2000_unix = 946728000;

    // This accounts for LST difference between J2000 and UNIX_Time in beamform_time.
    // It was verified with Kiyo's python code ch_util.ephemeris.transit_RA(), which
    // should account for both precession and nutation. Needs to be tested here though.
    double precession_offset = (beamform_time - j2000_unix) * 0.012791 / (365 * 24 * 3600);

    // calculate and modulate local sidereal time
    double lst = phi_0 + inst_long + lst_rate * (beamform_time - j2000_unix) - precession_offset;
    lst = fmod(lst, 360.);

    // convert lst to hour angle
    double hour_angle = lst - ra;
    // if(hour_angle < 0){hour_angle += 360.;}

    // get the alt/az based on the above
    double alt = sin(dec * D2R) * sin(inst_lat * D2R)
                 + cos(dec * D2R) * cos(inst_lat * D2R) * cos(hour_angle * D2R);
    alt = asin(alt);
    double az =
        (sin(dec * D2R) - sin(alt) * sin(inst_lat * D2R)) / (cos(alt) * cos(inst_lat * D2R));
    az = acos(az);
    if (sin(hour_angle * D2R) >= 0) {
        az = TAU - az;
    }

    // project, determine phases for each element
    // return geometric phase that instrument sees, i.e. -phases will be applied in beamformer
    double projection_angle, effective_angle, offset_distance;
    for (int i = 0; i < _num_elements; ++i) {
        projection_angle = 90 * D2R - atan2(feed_positions[2 * i + 1], feed_positions[2 * i]);
        offset_distance = cos(alt)
                          * sqrt(feed_positions[2 * i] * feed_positions[2 * i]
                                 + feed_positions[2 * i + 1] * feed_positions[2 * i + 1]);
        effective_angle = projection_angle - az;

        // z = (sin(dec*D2R) - sin(alt)*sin(inst_lat*D2R))/(cos(alt)*cos(inst_lat*D2R));
        phases[i] = TAU * cos(effective_angle) * offset_distance * one_over_c;
    }
    // ikt - commented out to test performance without INFO calls.
    //    INFO("get_delays: Computed delays: tnow = {:d}, lat = {:f}, long = {:f}, RA = {:f}, DEC =
    //    {:f}, LST = {:f}, ALT = {:f}, AZ = {:f}",
    //            (int)beamform_time, inst_lat, inst_long, ra, dec, lst, alt/D2R, az/D2R);

    return;
}
