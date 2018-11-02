
#ifndef PULSAR_TIMING_HPP
#define PULSAR_TIMING_HPP

#include <vector>
#include "time.h"

class Polyco {

public:
    Polyco(double tmid, float dm, double phase_ref, double rot_freq, std::vector<float> coeff);

    double mjd2phase(double t);

    double unix2phase(timespec t);

    double next_toa(timespec t, float freq);

private:
    double tmid;
    float dm;
    double phase_ref;
    double rot_freq;
    std::vector<float> coeff;

};

timespec add_nsec(timespec t, long nsec);

#endif