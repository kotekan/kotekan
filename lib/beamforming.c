
#include <math.h>
#include <time.h>

#include "beamforming.h"
#include "config.h"

#define CHI_LAT 49.3203
#define CHI_LONG -119.6175
#define D2R 0.01745329252 // pi/180
#define TAU 6.28318530718 // 2*pi

void get_delays(double ra, double dec, const struct Config * config, float * feed_positions, float * phases)
{
    //inverse speed of light in ns/m
    const double one_over_c = 3.3356;
    //offset of initial lst phase in degrees <-------------- PROBABLY WRONG PLEASE CHECK
    const double phi_0 = 280.46;
    //rate of change of LST with time in s
    const double lst_rate = 360./86164.09054;
    //UNIX timestamp of J2000 epoch time
    const double j2000_unix = 946728000;

    const double inst_lat = config->beamforming.instrument_lat;
    const double inst_long = config->beamforming.instrument_long;

    //calculate and modulate local sidereal time
    double lst = phi_0 + inst_long + lst_rate*(time(NULL) - j2000_unix);
    lst = fmod(lst, 360.);

    //convert lst to hour angle
    double hour_angle = lst - ra;
    if(hour_angle < 0){hour_angle += 360.;}

    //get the alt/az based on the above
    double alt = sin(dec*D2R)*sin(inst_lat*D2R)+cos(dec*D2R)*cos(inst_lat*D2R)*cos(hour_angle*D2R);
    alt = asin(alt);
    double az = (sin(dec*D2R) - sin(alt)*sin(inst_lat*D2R))/(cos(alt)*cos(inst_lat*D2R));
    az = acos(az);
    if(sin(hour_angle*D2R) >= 0){az = TAU - az;}

    //project, determine phases for each element
    double projection_angle, effective_angle, offset_distance;
    for(int i = 0; i < config->processing.num_elements; ++i)
    {
        projection_angle = atan2(feed_positions[2*i+1],feed_positions[2*i]);
        offset_distance  = cos(alt)*sqrt(feed_positions[2*i]*feed_positions[2*i] + feed_positions[2*i+1]*feed_positions[2*i+1]);
        effective_angle  = projection_angle - az;

        phases[i] = TAU*sin(effective_angle)*offset_distance*one_over_c;
    }

    return;
}