#ifndef BEAMFORMING
#define BEAMFORMING

#include "config.h"

void get_delays(time_t unix_time, double ra, double dec, const struct Config * config, float * feed_positions, float * phases);

#endif