#ifndef UPCHANNELIZE_REFERENCE_HPP
#define UPCHANNELIZE_REFERENCE_HPP

#include <complex> // for complex
#include <cstdint> // for int8_t, uint8_t
#include <vector>  // for vector

/**
 * @file
 * @brief Serial float32 reference implementation of the CHORD upchannelization.
 *
 * This is the "obviously correct but slow" counterpart to the generated
 * @c cudaUpchannelizer_<setup>_U<N> kernels. It implements eqn. (83) of Kendrick Smith's
 * notes, @c julia/docs/CHORD_GPU_upchannelization.pdf, directly:
 *
 * @f[
 *   \bar{E}[\bar\tau, f, u, \pi, d] = \mathrm{Quantize}\left(
 *       G[u + U f] \sum_{s=0}^{MU-1} W(s)\,
 *       e^{i\pi s (U-1)/U} e^{-2\pi i u s/U}\, E[U\bar\tau + s, f, \pi, d] \right)
 * @f]
 *
 * The two phase factors collapse into the single factor
 * @f$ \exp(-2\pi i (u - (U-1)/2) s / U) @f$, which is what the code below evaluates.
 * There is no FFT: the transform is a literal
 * @f$O(MU)@f$-per-output sum, so that the code can be read against the notes line by line.
 *
 * This header deliberately has no CUDA dependency, so that it can be unit tested on a
 * machine without a GPU.
 */

namespace kotekan {

/// Number of PFB taps used by the CHORD upchannelizer (`julia/kernels/upchan.jl:92`).
constexpr int upchan_default_num_taps = 4;

/**
 * @brief The sinc-Hanning PFB window @f$W(s)@f$, eqn. (11) of the notes with @f$N \to U@f$.
 *
 * @f[
 *   s' = \frac{s - (MU-1)/2}{MU}, \qquad
 *   W(s) = \frac{\cos^2(\pi s')\, \mathrm{sinc}(M s')}{U}
 * @f]
 *
 * where @f$\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)@f$ is the *normalized* sinc.
 *
 * Two details differ from a literal reading of eqn. (11), and both are needed to match
 * `Wkernel` at `julia/kernels/upchan.jl:76`:
 *
 * - The window is centred on @f$(MU-1)/2@f$, not @f$MU/2@f$, i.e. it is symmetric about
 *   the centre of the @f$MU@f$ samples. Footnote 4 of the notes flags this off-by-one as
 *   unresolved there.
 * - The @f$1/U@f$ factor undoes the "natural" upchannelization gain of @f$U@f$, so that
 *   amplitudes stay roughly constant (`upchan.jl:575-579`). The notes instead fold this
 *   into the gains @f$G@f$, but `setUpchanGain` supplies @f$G = 1@f$, so it has to live here.
 *
 * @param num_taps              Number of PFB taps @f$M@f$.
 * @param upchannelization_factor  Upchannelization factor @f$U@f$.
 * @returns The @f$MU@f$ window values, in order of @f$s@f$.
 */
std::vector<float> upchan_window(int num_taps, int upchannelization_factor);

/**
 * @brief Upchannelize one coarse frequency channel. Eqn. (83) of the notes.
 *
 * Purely serial, all arithmetic in float32. The caller supplies and interprets the
 * complex values, so this function is independent of the 4-bit or 8-bit sample encoding.
 *
 * @param E         Input, shape `[num_times][num_polarizations][num_dishes]`, where
 *                  `num_times >= upchannelization_factor * num_times_out + num_taps *
 *                  upchannelization_factor`. Time is the slowest-varying index.
 * @param window    The @f$MU@f$ window values from @c upchan_window.
 * @param gain      Per fine-frequency gain, `upchannelization_factor` entries.
 * @param Ebar      Output, shape
 *                  `[num_times_out][upchannelization_factor][num_polarizations][num_dishes]`.
 * @param num_taps  Number of PFB taps @f$M@f$.
 * @param upchannelization_factor  @f$U@f$.
 * @param num_polarizations        @f$P@f$.
 * @param num_dishes               @f$D@f$.
 * @param num_times_out            Number of output time samples @f$\bar{T}@f$ to produce.
 */
void upchannelize_reference(const std::complex<float>* E, const float* window, const float* gain,
                            std::complex<float>* Ebar, int num_taps, int upchannelization_factor,
                            int num_polarizations, int num_dishes, int num_times_out);

////////////////////////////////////////////////////////////////////////////////
// Sample encodings
//
// CHIME/CHORD pack a complex sample with the REAL part in the HIGH bits and the
// IMAGINARY part in the LOW bits; this is what "swapped" names in the kotekan type
// `int4x2_swapped_withoffset`. See `external/n2k/include/n2k/Correlator.hpp`
// (`real_part_in_low_bits = false`, `offset_encoded = true`).
//
// Decoding the other way round yields the complex conjugate of eqn. (83) -- consistently
// on input and output, so it is invisible unless compared against the GPU.
////////////////////////////////////////////////////////////////////////////////

/// Largest magnitude a 4-bit component may take. The kernel clamps to +-7 rather than
/// [-8, +7]: `-8` is reserved as the poison / missing-data marker.
constexpr int upchan_int4_max = 7;
/// Largest magnitude an 8-bit component may take.
constexpr int upchan_int8_max = 127;

/// Decode one `int4x2_swapped_withoffset` byte: real = high nibble - 8, imag = low nibble - 8.
inline std::complex<float> upchan_decode_int4(const std::uint8_t byte) {
    const int imag = int(byte & 0x0f) - 8;
    const int real = int(byte >> 4) - 8;
    return std::complex<float>(float(real), float(imag));
}

/// Encode one `int4x2_swapped_withoffset` byte, clamping to +-7 and rounding half to even.
std::uint8_t upchan_encode_int4(std::complex<float> value);

/// Decode one `cint8` sample: plain two's complement, without swapping real and imaginary
/// component.
inline std::complex<float> upchan_decode_int8(const std::complex<std::int8_t> value) {
    return std::complex<float>(float(value.real()), float(value.imag()));
}

/// Encode one `cint8` sample, clamping to +-127 and rounding half to even.
std::complex<std::int8_t> upchan_encode_int8(std::complex<float> value);

} // namespace kotekan

#endif // UPCHANNELIZE_REFERENCE_HPP
