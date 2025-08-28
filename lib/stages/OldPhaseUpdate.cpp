#include "OldPhaseUpdate.hpp"

#include "StageFactory.hpp"

#define PI 3.14159265
#define light 299792458.
#define one_over_c 0.0033356
#define R2D 180. / PI
#define D2R PI / 180.
#define TAU 2 * PI

REGISTER_KOTEKAN_STAGE(OldPhaseUpdate);

OldPhaseUpdate::OldPhaseUpdate(kotekan::Config& config, const std::string& unique_name,
                                   kotekan::bufferContainer& buffer_container) :
    BeamformingPhaseUpdate(config, unique_name, buffer_container) {

    _feed_sep_NS = config.get<float>(unique_name, "feed_sep_NS");
    _feed_sep_EW = config.get<float>(unique_name, "feed_sep_EW");
}

void OldPhaseUpdate::compute_phases(uint8_t* out_frame, const timespec& gps_time,
                                      const std::vector<float>& frequencies_in_frame,
                                      uint32_t beam_offset, uint8_t* gains_frame) {
    
    float* output = (float*)out_frame;
    float* gains = (float*)gains_frame;

    // This code only works with one frequency per frame.
    assert(frequencies_in_frame.size() == 1);
    float FREQ = frequencies_in_frame.at(0);
    struct tm* timeinfo;
    timeinfo = localtime(&gps_time.tv_sec);
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
                + (timeinfo->tm_sec + gps_time.tv_nsec / 1.e9) / 3600.;
    double GST = fmod((T0 + UT * 1.002737909), 24.);
    double LST = GST + _inst_long / 15.;
    while (LST < 0) {
        LST = LST + 24;
    }
    LST = fmod(LST, 24);
    for (uint32_t b = beam_offset; b < (_num_local_beams + beam_offset); b++) {
        if (_beam_coord.scaling[b] == 1) {
            for (uint32_t i = 0; i < _num_elements * 2; ++i) {
                output[b * _num_elements * 2 + i] = gains[b * _num_elements * 2 + i];
            }
            continue;
        }
        double hour_angle = LST * 15. - _beam_coord.ra[b];
        double alt = sin(_beam_coord.dec[b] * D2R) * sin(_inst_lat * D2R)
                     + cos(_beam_coord.dec[b] * D2R) * cos(_inst_lat * D2R) * cos(hour_angle * D2R);
        alt = asin(std::clamp(alt, -1.0, 1.0));
        double az = (sin(_beam_coord.dec[b] * D2R) - sin(alt) * sin(_inst_lat * D2R))
                    / (cos(alt) * cos(_inst_lat * D2R));
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

