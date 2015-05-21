#ifndef BEAMFORMING
#define BEAMFORMING

#include "config.h"

void get_delays(double ra, double dec, const struct Config * config, float * feed_positions, float * phases);

#endif