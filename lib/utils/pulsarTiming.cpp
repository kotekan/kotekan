
#include "pulsarTiming.hpp"
#include <math.h>

Polyco::Polyco(double t, float d, double p, double f0, std::vector<float> c):
        tmid(t), dm(d), phase_ref(p), rot_freq(f0), coeff(c) {}

double Polyco::mjd2phase(float t) {

    double dt = (t - tmid) * 1440;
    double phase = phase_ref + dt*60*rot_freq;
    for (size_t i = 0; i < coeff.size(); i++) {
        phase += coeff[i] * pow(dt, i);
    }

    return phase;
}

double Polyco::unix2phase(timespec t) {

    // number of days between UNIX epoch and MJD epoch is 40587
    double t_mjd = ((double) t.tv_sec/ 86400.) + ((double) t.tv_nsec/ 86400e9) + 40587;

    return mjd2phase(t_mjd);
}

double Polyco::next_toa(timespec t, float freq) {

    // Adjust time for dispersion delay
    double dm_delay = 4140. * dm * pow(freq, -2);
    timespec psr_t = add_nsec(t, long(dm_delay * 1e9));

    double phase = unix2phase(psr_t);

    // time until next pulse in s
    return (1. - (phase - floor(phase))) / rot_freq;
}

timespec add_nsec(timespec temp, long nsec) {

    timespec new_time = temp;
    new_time.tv_sec += nsec / (int) 1e9;
    new_time.tv_nsec += nsec % (int) 1e9;
    if (new_time.tv_nsec < 0) {
        new_time.tv_sec -= 1;
        new_time.tv_nsec += (int) 1e9;
    } else if (new_time.tv_nsec > 1e9) {
        new_time.tv_sec += 1;
        new_time.tv_nsec -= (int) 1e9;
    }
    return new_time;
}