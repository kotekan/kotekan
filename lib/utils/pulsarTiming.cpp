#include "pulsarTiming.hpp"

#include "visUtil.hpp" // for add_nsec

#include "fmt.hpp" // for format, fmt

#include <cmath>       // for floor, pow
#include <memory>      // for allocator_traits<>::value_type
#include <stdexcept>   // for runtime_error
#include <sys/types.h> // for uint

Polyco::Polyco(double t, float d, double p, double f0, std::vector<float> c) :
    tmid(t), dm(d), phase_ref(p), rot_freq(f0), coeff(c) {}

Polyco::Polyco() {}

double Polyco::mjd2phase(double t) const {

    double dt = (t - tmid) * 1440;
    double phase = phase_ref + dt * 60 * rot_freq;
    for (size_t i = 0; i < coeff.size(); i++) {
        phase += coeff[i] * pow(dt, i);
    }

    return phase;
}

double Polyco::unix2phase(timespec t) const {

    return mjd2phase(ts2mjd(t));
}

double Polyco::next_toa(timespec t, float freq) const {

    // Adjust time for dispersion delay
    double dm_delay = -4140. * dm * pow(freq, -2);
    timespec psr_t = add_nsec(t, long(dm_delay * 1e9));

    double phase = unix2phase(psr_t);

    // time until next pulse in s
    return (1. - (phase - floor(phase))) / rot_freq;
}

SegmentedPolyco::SegmentedPolyco(double rot_freq, float dm, float seg, std::vector<double> tmid,
                                 std::vector<double> phase_ref,
                                 std::vector<std::vector<float>> coeff) {

    if (phase_ref.size() != tmid.size() || tmid.size() != coeff.size()) {
        throw std::runtime_error(
            fmt::format(fmt("Number of segments is inconsistent: phase_ref({:d}), rot_freq({:d}), "
                            "coeff({:d})."),
                        tmid.size(), phase_ref.size(), coeff.size()));
    }
    for (uint i = 0; i < phase_ref.size(); i++) {
        polycos.push_back(Polyco(tmid[i], dm, phase_ref[i], rot_freq, coeff[i]));
        // calculate bounds of segment (width seg is in s)
        segments.push_back({tmid[i] - seg / 172800, tmid[i] + seg / 172800});
    }
}

SegmentedPolyco::SegmentedPolyco() {}

const Polyco& SegmentedPolyco::get_polyco(timespec t) const {

    double tmid = ts2mjd(t);
    // Find segment that includes this time
    for (uint i = 0; i < polycos.size(); i++) {
        if (segments[i].first <= tmid && segments[i].second >= tmid)
            return polycos[i];
    }
    throw std::runtime_error("Could not find polyco for requested time.");
}

const Polyco& SegmentedPolyco::get_polyco(int i) const {

    if (polycos.size() == 0) {
        throw std::runtime_error("No polycos were initialized.");
    } else if (i < 0) {
        return polycos.at(polycos.size() + i);
    } else {
        return polycos.at(i);
    }
}
