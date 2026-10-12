#ifndef GNSS_44_HPP
#define GNSS_44_HPP

/**
 * @file
 * @brief The CHORD 4+4b post-PFB voltage codec, one definition for host and device.
 *
 * One byte per complex sample: real in the high nibble, imaginary in the low, each a 4-bit
 * offset-encoded integer (stored value - 8, so -8..+7). Zero is 0x88; 0x00 decodes to (-8,-8),
 * a full-scale DC spike, so zero-fill with @ref ZERO, not memset.
 *
 * In CHORD the PFB and the 4-bit quantizer run in the RFSoC F-engine, outside kotekan, so the
 * format is fixed by its consumer: the decode in kotekan's correlator reference model
 * lib/testing/gpuSimulate.cpp, which the X-engine is validated against:
 *     int xi = (input[ix] & 0x0f) - 8;         // imag = low nibble
 *     int xr = ((input[ix] & 0xf0) >> 4) - 8;  // real = high nibble
 * The encoder here must match that decode byte for byte. lib/cuda/cudaQuantizeKernel4.cu is a
 * different format for a different product (the FRB beam quantizer: eight two's-complement
 * values per uint32 with a per-chunk scale and offset).
 *
 * Scaling is the caller's job and is what makes 4 bits enough. The bandpass is not flat, so each
 * channel gets its own scale, chosen to put that bin's noise rms at ~2-3 lsb, as the RFSoC does
 * at startup. GNSS signals sit far below the noise, so the noise dithers them and the
 * quantization error, uncorrelated with a pseudorandom replica, averages down like thermal noise
 * instead of biasing the correlation. On sky data the C/N0 loss is < 0.2 dB at 2 lsb and
 * < 0.14 dB at 3 lsb; 3 lsb leaves more headroom for RFI.
 *
 * The header needs no CUDA headers (plain floats, no float2), so it builds in any translation
 * unit. GPU code uses it through the float2 wrappers in cudaGnssDespreadKernel.hpp and host code
 * (scripts/gnss/e2e.cpp) calls it directly; keep it the only codec.
 * lib/cuda/cudaGnssDespreadTest.cpp checks the encoder against an independent copy of the
 * reference decode.
 */

#ifdef __CUDACC__
#define GNSS44_HD __host__ __device__ inline
#else
#define GNSS44_HD inline
#endif

namespace gnss44 {

/// The offset-encoded zero. Use for gap fill; memset(0) would write (-8,-8).
constexpr unsigned char ZERO = 0x88;

/// Encode one complex sample. @c inv_scale maps volts to lsb. Rounds to nearest (half away from
/// zero) and clamps to [-8,7], never wraps; @c railed (may be null) is set to 1 on a clamp.
GNSS44_HD unsigned char pack(float re, float im, float inv_scale, int* railed) {
    // Explicit rounding rather than rintf(), so host and device use the same rounding rule.
    int r = (int)(re * inv_scale + (re * inv_scale >= 0.f ? 0.5f : -0.5f));
    int i = (int)(im * inv_scale + (im * inv_scale >= 0.f ? 0.5f : -0.5f));
    if (r < -8 || r > 7 || i < -8 || i > 7) {
        if (railed)
            *railed = 1;
        r = r < -8 ? -8 : (r > 7 ? 7 : r);
        i = i < -8 ? -8 : (i > 7 ? 7 : i);
    }
    return (unsigned char)((((r + 8) & 0x0f) << 4) | ((i + 8) & 0x0f));
}

/// Decode one 4+4b byte to volts. @c scale maps lsb to volts (1/inv_scale). This must stay
/// gpuSimulate.cpp's decode.
GNSS44_HD void unpack(unsigned char b, float scale, float* re, float* im) {
    *re = scale * ((float)(int)((b >> 4) & 0x0f) - 8.0f); // real = high nibble, offset -8
    *im = scale * ((float)(int)(b & 0x0f) - 8.0f);        // imag = low nibble, offset -8
}

} // namespace gnss44

#endif // GNSS_44_HPP
