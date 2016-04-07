#include <math.h>
#include <time.h>

#include "beamforming.h"
#include "config.h"
#include "errors.h"

#define D2R 0.01745329252 // pi/180
#define TAU 6.28318530718 // 2*pi

void get_delays(time_t unix_time, double ra, double dec, const struct Config * config, float * feed_positions, float * phases)
{
    //inverse speed of light in ns/m
    const double one_over_c = 3.3356;
    //offset of initial lst phase in degrees <-------------- PROBABLY WRONG PLEASE CHECK
    const double phi_0 = 280.46;
    //const double phi_0 = 160.64;
    //rate of change of LST with time in s
    const double lst_rate = 360./86164.09054;
    //UNIX timestamp of J2000 epoch time
    const double j2000_unix = 946728000;

    const double inst_lat = config->beamforming.instrument_lat;
    const double inst_long = config->beamforming.instrument_long;

    // This accounts for LST difference between J2000 and unix_time. 
    // It was verified with Kiyo's python code ch_util.ephemeris.transit_RA(), which 
    // should account for both precession and nutation. Needs to be tested here though. 
    double precession_offset = (unix_time - j2000_unix) * 0.012791 / (365 * 24 * 3600);

    //calculate and modulate local sidereal time
    double lst = phi_0 + inst_long + lst_rate*(unix_time - j2000_unix) - precession_offset;
    lst = fmod(lst, 360.);

    //convert lst to hour angle
    double hour_angle = lst - ra;
    //if(hour_angle < 0){hour_angle += 360.;}

    //get the alt/az based on the above
    double alt = sin(dec*D2R)*sin(inst_lat*D2R)+cos(dec*D2R)*cos(inst_lat*D2R)*cos(hour_angle*D2R);
    alt = asin(alt);
    double az = (sin(dec*D2R) - sin(alt)*sin(inst_lat*D2R))/(cos(alt)*cos(inst_lat*D2R));
    az = acos(az);
    if(sin(hour_angle*D2R) >= 0){az = TAU - az;}

    //project, determine phases for each element 
    //return geometric phase that instrument sees, i.e. -phases will be applied in beamformer
    double projection_angle, effective_angle, offset_distance;
    for(int i = 0; i < config->processing.num_elements; ++i)
    {
        projection_angle = 90*D2R - atan2(feed_positions[2*i+1],feed_positions[2*i]);
        offset_distance  = cos(alt)*sqrt(feed_positions[2*i]*feed_positions[2*i] + feed_positions[2*i+1]*feed_positions[2*i+1]);
        effective_angle  = projection_angle - az;

        //z = (sin(dec*D2R) - sin(alt)*sin(inst_lat*D2R))/(cos(alt)*cos(inst_lat*D2R));
        phases[i] = TAU*cos(effective_angle)*offset_distance*one_over_c;
    }

    INFO("get_delays: Computed delays: tnow = %d, lat = %f, long = %f, RA = %f, DEC = %f, LST = %f, ALT = %f, AZ = %f", (int)time(NULL), inst_lat, inst_long, ra, dec, lst, alt/D2R, az/D2R);

    return;
}
