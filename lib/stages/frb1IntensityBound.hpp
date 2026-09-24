/**
 * @file
 * @brief Worst-case intensity of the FRB1 beamforming kernels for a given set of weights
 */

#ifndef FRB1_INTENSITY_BOUND_HPP
#define FRB1_INTENSITY_BOUND_HPP

#include "DataType.hpp" // for float16_t

#include <cmath>   // for hypot
#include <cstddef> // for ptrdiff_t

namespace kotekan {

/// The largest intensity the FRB1 kernels can produce without overflowing: the Float16 maximum
/// 65504, less 1% for the rounding errors of the kernels' Float16 arithmetic
constexpr double frb1_intensity_limit = 0.99 * 65504;

/**
 * @brief Upper bound on the FRB1 intensity for the weights of one frequency
 *
 * The FRB1 kernels (`julia/kernels/chimefrb.jl`, `julia/kernels/frb.jl`) multiply the weights W
 * by `input_gain = 1/(2√(M·N))`, beamform, and then square in Float16, before applying their
 * output gain. The upchannelizer's int4 output lies within ±7, so |E| ≤ 7√2, and the beam
 * voltage of polarization p is bounded by |Ẽ_p| ≤ input_gain · 7√2 · Σ_dish |W_p,dish|. Hence
 * the intermediate Σ_p |Ẽ_p|², and the intensity I, which averages it over time, are at most
 *
 *     98 / (4·M·N) · Σ_p (Σ_dish |W_p,dish|)²
 *
 * for any input. The kernels can overflow to Inf only if this bound exceeds 65504. For unit
 * weights on a fully populated grid the bound is 49·M·N, e.g. 50176 for CHIME.
 *
 * @param W                 Weights of one frequency, laid out as [P][dishN][dishM][re/im]
 * @param num_polarizations Number of polarizations P
 * @param num_dishes_M      Dish grid size M
 * @param num_dishes_N      Dish grid size N
 * @return The bound, or NaN if a weight is NaN
 */
inline double frb1_intensity_bound(const float16_t* const W, const int num_polarizations,
                                   const int num_dishes_M, const int num_dishes_N) {
    const std::ptrdiff_t num_dishes = std::ptrdiff_t(num_dishes_M) * num_dishes_N;
    double sum_polr = 0;
    for (int polr = 0; polr < num_polarizations; ++polr) {
        const float16_t* const W_polr = W + 2 * num_dishes * polr;
        double sum_dish = 0;
        for (std::ptrdiff_t dish = 0; dish < num_dishes; ++dish)
            sum_dish += std::hypot(double(W_polr[2 * dish + 0]), double(W_polr[2 * dish + 1]));
        sum_polr += sum_dish * sum_dish;
    }
    return 98 / (4 * double(num_dishes)) * sum_polr;
}

} // namespace kotekan

#endif // #ifndef FRB1_INTENSITY_BOUND_HPP
