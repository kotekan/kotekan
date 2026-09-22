#include "upchannelizeReference.hpp"

#include <algorithm> // for max, min
#include <cassert>   // for assert
#include <cmath>     // for cos, sin, nearbyint, M_PI
#include <complex>   // for complex
#include <cstdint>   // for int8_t, uint8_t
#include <vector>    // for vector

namespace kotekan {

namespace {

/// The normalized sinc, sin(pi x) / (pi x).
float sinc_normalized(const float x) {
    if (x == 0.0f)
        return 1.0f;
    const float pix = float(M_PI) * x;
    return std::sin(pix) / pix;
}

/// Round half to even, then clamp to +-max_magnitude.
int quantize(const float x, const int max_magnitude) {
    // `nearbyint` uses the current rounding mode, which is round-half-to-even by default.
    // That matches the `cvt.rni` in the generated PTX; `floor(x + 0.5)` would not, on ties.
    const float rounded = std::nearbyint(x);
    if (!(rounded > float(-max_magnitude)))
        return -max_magnitude;
    if (!(rounded < float(max_magnitude)))
        return max_magnitude;
    return int(rounded);
}

} // namespace

std::vector<float> upchan_window(const int num_taps, const int upchannelization_factor) {
    const int M = num_taps;
    const int U = upchannelization_factor;
    assert(M > 0);
    assert(U > 0);

    std::vector<float> window(M * U);
    for (int s = 0; s < M * U; ++s) {
        // Normalized to (-1/2, +1/2); see `Wkernel` in `julia/kernels/upchan.jl:76`.
        const float sp = (s - (M * U - 1) / float(2)) / float(M * U);
        const float cosine = std::cos(float(M_PI) * sp);
        window.at(s) = cosine * cosine * sinc_normalized(float(M) * sp) / float(U);
    }
    return window;
}

void upchannelize_reference(const std::complex<float>* const E, const float* const window,
                            const float* const gain, std::complex<float>* const Ebar,
                            const int num_taps, const int upchannelization_factor,
                            const int num_polarizations, const int num_dishes,
                            const int num_times_out) {
    const int M = num_taps;
    const int U = upchannelization_factor;
    const int P = num_polarizations;
    const int D = num_dishes;
    assert(M > 0);
    assert(U > 0);
    assert(P > 0);
    assert(D > 0);
    assert(num_times_out >= 0);

    // Precompute the phase factor exp(-2 pi i (u - (U-1)/2) s / U). It depends only on
    // (u, s), not on time, polarization or dish, so it is shared across the whole call.
    // See the derivation in `upchannelizeReference.hpp`.
    std::vector<std::complex<float>> phases(U * M * U);
    for (int u = 0; u < U; ++u) {
        for (int s = 0; s < M * U; ++s) {
            const float angle =
                -2.0f * float(M_PI) * (float(u) - float(U - 1) / 2.0f) * float(s) / float(U);
            phases.at(u * (M * U) + s) = std::complex<float>(std::cos(angle), std::sin(angle));
        }
    }

    const long E_time_stride = long(P) * D;
    const long Ebar_time_stride = long(U) * P * D;

    // The dish axis is the fastest-varying one in both arrays, so accumulate a whole row of
    // dishes at a time. Keeping `d` in the innermost loop walks `E` contiguously; putting it
    // outermost instead would stride by P*D complex values on every tap.
    std::vector<std::complex<float>> acc(D);

    for (int tbar = 0; tbar < num_times_out; ++tbar) {
        for (int u = 0; u < U; ++u) {
            const std::complex<float>* const phase = phases.data() + u * (M * U);
            const float G = gain[u];
            for (int p = 0; p < P; ++p) {
                std::fill(acc.begin(), acc.end(), std::complex<float>(0.0f, 0.0f));
                for (int s = 0; s < M * U; ++s) {
                    // Eqn. (83): the output at `tbar` reads M*U input samples starting at
                    // U*tbar, i.e. it looks (M-1)*U samples past its own block.
                    const long t = long(U) * tbar + s;
                    const std::complex<float> coeff = window[s] * phase[s];
                    const std::complex<float>* const Erow = E + t * E_time_stride + long(p) * D;
                    for (int d = 0; d < D; ++d)
                        acc[d] += coeff * Erow[d];
                }
                std::complex<float>* const Ebar_row =
                    Ebar + long(tbar) * Ebar_time_stride + long(u) * P * D + long(p) * D;
                for (int d = 0; d < D; ++d)
                    Ebar_row[d] = G * acc[d];
            }
        }
    }
}

std::uint8_t upchan_encode_int4(const std::complex<float> value) {
    const int real = quantize(value.real(), upchan_int4_max);
    const int imag = quantize(value.imag(), upchan_int4_max);
    // Real in the high nibble, imaginary in the low one, each offset-encoded by +8.
    return std::uint8_t((((unsigned)(real + 8) & 0xf) << 4) | ((unsigned)(imag + 8) & 0xf));
}

std::complex<std::int8_t> upchan_encode_int8(const std::complex<float> value) {
    return std::complex<std::int8_t>(std::int8_t(quantize(value.real(), upchan_int8_max)),
                                     std::int8_t(quantize(value.imag(), upchan_int8_max)));
}

} // namespace kotekan
