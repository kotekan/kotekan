#include "pfbPrototype.hpp"

#include <cmath>     // for sin, cos, M_PI, sqrt
#include <stdexcept> // for invalid_argument

namespace dsp {

namespace {

double sinc(double x) { // normalized sinc, sin(pi x)/(pi x)
    if (x == 0.0)
        return 1.0;
    const double a = M_PI * x;
    return std::sin(a) / a;
}

// Zeroth-order modified Bessel I0, series sum (ample for window beta <= ~20).
double bessel_i0(double x) {
    double term = 1.0, sum = 1.0;
    const double hx2 = 0.25 * x * x;
    for (int k = 1; k < 64; ++k) {
        term *= hx2 / (k * k);
        sum += term;
        if (term < 1e-12 * sum)
            break;
    }
    return sum;
}

// Window sample at index n of a length-L window.
double window_value(Window w, int n, int L, double beta) {
    const double t = static_cast<double>(n) / (L - 1); // 0..1
    switch (w) {
        case Window::Rectangular:
            return 1.0;
        case Window::Hann:
            return 0.5 - 0.5 * std::cos(2.0 * M_PI * t);
        case Window::Hamming:
            return 0.54 - 0.46 * std::cos(2.0 * M_PI * t);
        case Window::Blackman:
            return 0.42 - 0.5 * std::cos(2.0 * M_PI * t) + 0.08 * std::cos(4.0 * M_PI * t);
        case Window::Kaiser: {
            const double r = 2.0 * t - 1.0; // -1..1
            return bessel_i0(beta * std::sqrt(1.0 - r * r)) / bessel_i0(beta);
        }
    }
    return 1.0;
}

} // namespace

std::vector<float> pfb_prototype(int num_chan, int num_taps, Window window, double kaiser_beta) {
    const int L = num_chan * num_taps;
    std::vector<float> h(L);
    double sum = 0.0;
    for (int r = 0; r < L; ++r) {
        const double x = (r - (L - 1) / 2.0) / num_chan;           // sinc arg, cutoff pi/N
        const double v = sinc(x) * window_value(window, r, L, kaiser_beta);
        h[r] = static_cast<float>(v);
        sum += v;
    }
    // Normalize so the sum of taps == num_chan, i.e. each polyphase branch has
    // unit gain. This matches a straight FFT (a rectangular window with every
    // sample weighted 1), so the channelizer's passband gain -- and the output
    // power level -- is the same whether num_taps is 1 (straight FFT) or >1
    // (PFB). Normalizing to unit *total* sum instead would suppress the output
    // by a factor of num_chan in amplitude (num_chan^2 in power).
    const double scale = (sum != 0.0) ? num_chan / sum : 1.0;
    for (int r = 0; r < L; ++r)
        h[r] = static_cast<float>(h[r] * scale);
    return h;
}

Window window_from_string(const std::string& name) {
    if (name == "rect" || name == "rectangular")
        return Window::Rectangular;
    if (name == "hann")
        return Window::Hann;
    if (name == "hamming")
        return Window::Hamming;
    if (name == "blackman")
        return Window::Blackman;
    if (name == "kaiser")
        return Window::Kaiser;
    throw std::invalid_argument("pfb window '" + name + "' unknown");
}

} // namespace dsp
