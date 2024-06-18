#include "FloatPhaseUpdate.hpp"

#include "StageFactory.hpp"

#define PI 3.14159265
#define one_over_c 0.0033356
#define D2R PI / 180.
#define TAU 2 * PI

REGISTER_KOTEKAN_STAGE(FloatPhaseUpdate);

FloatPhaseUpdate::FloatPhaseUpdate(kotekan::Config& config, const std::string& unique_name,
                                   kotekan::bufferContainer& buffer_container) :
    BeamformingPhaseUpdate(config, unique_name, buffer_container) {}

void FloatPhaseUpdate::compute_phases(uint8_t* out_frame_int, const timespec& gps_time,
                                      const std::vector<float>& frequencies_in_frame,
                                      uint32_t beam_offset, uint8_t* gains_frame_int) {
    
    float* out_frame = (float*)out_frame_int;
    float* gains_frame = (float*)gains_frame_int;

    // Getting UTC time in ISO format from the GPS time in Unix format
    struct tm* utc_time;
    utc_time = gmtime(&gps_time.tv_sec);                                                                      //DOUBT: Confirm if it should be &gps_time.tv_sec since gps_time is passed by reference

    //Calculating the UT1 Julian Day from the UTC time and date
    uint year = utc_time->tm_year + 1900;
    uint month = utc_time->tm_mon + 1;
    if (month < 3) {
        month = month + 12;
        year = year - 1;
    }
    uint day = utc_time->tm_mday;
    uint hour = utc_time->tm_hour;
    float JD = 2 - int(year / 100.) + int(int(year / 100.) / 4.) + int(365.25 * year) //See this conversion for Skyfield
               + int(30.6001 * (month + 1)) + day + 1720994.5; 
    double T = (JD - 2451545.0)
               / 36525.0; // Works if time after year 2000, otherwise T is -ve and might break
    double T0 = fmod((6.697374558 + (2400.051336 * T) + (0.000025862 * T * T)), 24.);
    double UT = (timeinfo->tm_hour) + (timeinfo->tm_min / 60.)
                + (timeinfo->tm_sec + time_now.tv_nsec / 1.e9) / 3600.;
    double GST = fmod((T0 + UT * 1.002737909), 24.);
    double LST = GST + _inst_long / 15.;
    while (LST < 0) {
        LST = LST + 24;
    }
    LST = fmod(LST, 24);
    //double UT = (hour) + (utc_time->tm_min / 60.)
    //           + (utc_time->tm_sec + _DUT1 + gps_time.tv_nsec / 1.e9) / 3600.;                //TO DO:UT1_UTC should be a double value in struct _beam_coord containing UT1-UTC in seconds
    //double JD_UT1 = JD + UT / 24.;


    //Calculating the Local Stellar Angle (LSA) 
    //double JD_frac = fmod(JD_UT1, 1);
    //double DU = JD_UT1 - 2451545.0;
    //double ERA = fmod((0.7790572732640 + 0.00273781191135448 * DU + JD_frac), 1) * 360;
    //double LSA = ERA + _inst_long;
    //if (LSA<0) {
    //    LSA = LSA + 360;
    //}
    //if (LSA>360) {
    //    LSA = fmod(LSA, 360);
    //}

    //INFO("GPS Time: {:.9f}, Frequency {}", ts_to_double(gps_time), frequencies_in_frame[0]);

    //Calculating the hour angle and phases for all frequencies in a frame for each pointing 
    for (uint32_t b = beam_offset; b < (_num_local_beams + beam_offset); b++) {
        // Special case for forming a beam at the telescope zenith, e.g. only apply gains, no phases.  
	     if (_beam_coord.scaling[b] == 1) {   
            for (uint32_t i = 0; i < _num_elements * _num_local_freq * 2; ++i) {                             
                out_frame[b * _num_elements * _num_local_freq * 2 + i] = gains_frame[b * _num_elements * _num_local_freq * 2 + i];
            }
            continue;
        }
        //double hour_angle  = LSA - _beam_coord.ra[b];
        double hour_angle = LST * 15. - beam_coord.ra[b];
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
        int zero_feeds;

	//INFO("Beam: {} (Ra: {}, Dec {}), LSA: {}, ERA: {}, JD_UT1: {}, hour_angle {}, alt: {}, az: {}", b, _beam_coord.ra[b], _beam_coord.dec[b], LSA, ERA, JD_UT1, hour_angle, alt, az);

        //Looping over all frequencies in the frame
        for(uint32_t i = 0; i < _num_local_freq; i++){
            //Looping over the feeds
            for(size_t j = 0; j < feed_locations.size(); j++){
                double dist_x = feed_locations[j].first;
                double dist_y = feed_locations[j].second;
                if (dist_x == 0 && dist_y == 0) {
                  projection_angle = 0;
                  zero_feeds = 1;
                }
                else {
                  projection_angle = 90 * D2R - atan2(dist_y, dist_x);
                  zero_feeds = 0;
                }
                offset_distance = sqrt(pow(dist_y, 2) + pow(dist_x, 2));
                effective_angle = projection_angle - az;
                double delay_real = cos(TAU * cos(effective_angle) * cos(alt) * offset_distance
                                       * frequencies_in_frame[i] * one_over_c);
                double delay_imag = -sin(TAU * cos(effective_angle) * cos(-alt) * offset_distance            //DOUBT: Why is imag delay -sin(phase)? Has it something to do with the way gains are applied?
                                        * frequencies_in_frame[i] * one_over_c);
                //if (j == 0) {
                //   INFO("feed: 0, dist_x: {}, dist_y: {}, projection_angle: {}, delay_real: {}, delay_imag: {}", dist_x, dist_y, projection_angle, delay_real, delay_imag);
		//}
		//Looping over 2 polarisations
                for (int p = 0; p < 2; p++) {                                                                //TO DO: Make sure that the gains are in accordance with the feed locations and frequencies
                    uint elem_id = p * _num_elements / 2 + j;
                    uint offset = b * _num_local_freq * _num_elements + i * _num_elements;
                    // Not scrembled, assume reordering kernel has been run
                    if (zero_feeds == 1) {
                      out_frame[(offset + elem_id) * 2] = 0;
                      out_frame[(offset + elem_id) * 2 + 1] = 0;
                      }
                    else {
                      out_frame[(offset + elem_id) * 2] =                                        
                          delay_real * gains_frame[(offset + elem_id) * 2]
                          - delay_imag * gains_frame[(offset + elem_id) * 2 + 1];
                      out_frame[(offset + elem_id) * 2 + 1] =
                          delay_real * gains_frame[(offset + elem_id) * 2 + 1]
                          + delay_imag * gains_frame[(offset + elem_id) * 2]; 
                    }
                  }
            }
        }
        //INFO("LST: {}, ERA: {}, JD_UT1: {}, hour_angle {}", LST, ERA, JD_UT1, hour_angle);
    }
}

