#ifndef BEAMFORMING
#define BEAMFORMING

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

void get_delays(double ra, double dec, const struct Config * config, float * feed_positions, float * phases);

#ifdef __cplusplus
}
#endif

#endif