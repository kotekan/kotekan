#include "pfbPrototype.hpp"

#include <cmath>     // for sin, cos, M_PI
#include <stdexcept> // for invalid_argument

namespace {

// Normalized sinc, sin(pi x) / (pi x).
double sinc(double x) {
    if (x == 0.0)
        return 1.0;
    return std::sin(M_PI * x) / (M_PI * x);
}

// Sample n of a length-L window.
double window_value(PfbWindow w, int n, int L) {
    const double t = static_cast<double>(n) / (L - 1);
    switch (w) {
        case PfbWindow::Rectangular:
            return 1.0;
        case PfbWindow::Hann:
            return 0.5 - 0.5 * std::cos(2.0 * M_PI * t);
        case PfbWindow::Hamming:
            return 0.54 - 0.46 * std::cos(2.0 * M_PI * t);
        case PfbWindow::Blackman:
            return 0.42 - 0.5 * std::cos(2.0 * M_PI * t) + 0.08 * std::cos(4.0 * M_PI * t);
    }
    return 1.0;
}

} // namespace

std::vector<float> pfb_prototype(int num_chan, int num_taps, PfbWindow window) {
    const int L = num_chan * num_taps;
    if (num_chan < 1 || num_taps < 1 || L < 2)
        throw std::invalid_argument("a PFB prototype needs at least two taps");
    std::vector<double> h(L);
    double sum = 0.0;
    for (int r = 0; r < L; ++r) {
        h[r] = sinc((r - (L - 1) / 2.0) / num_chan) * window_value(window, r, L);
        sum += h[r];
    }
    std::vector<float> proto(L);
    for (int r = 0; r < L; ++r)
        proto[r] = static_cast<float>(h[r] * num_chan / sum);
    return proto;
}

PfbWindow pfb_window_from_string(const std::string& name) {
    if (name == "rect")
        return PfbWindow::Rectangular;
    if (name == "hann")
        return PfbWindow::Hann;
    if (name == "hamming")
        return PfbWindow::Hamming;
    if (name == "blackman")
        return PfbWindow::Blackman;
    throw std::invalid_argument("unknown PFB window '" + name + "'");
}
