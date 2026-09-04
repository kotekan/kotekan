#ifndef GNSS_44_HPP
#define GNSS_44_HPP

/**
 * @file
 * @brief The CHORD 4+4b post-PFB voltage codec -- ONE definition, host and device.
 *
 * FORMAT: one byte per complex sample, REAL in the HIGH nibble, IMAG in the LOW, each a 4-bit
 * OFFSET-ENCODED integer (stored value - 8 -> -8..+7). Zero is 0x88, NOT 0x00 (0x00 decodes to
 * (-8,-8), a full-scale DC spike -- which is why a zero-fill needs this constant and not memset).
 *
 * WHERE THE CONTRACT COMES FROM. In CHORD the PFB and the 4-bit quantizer are NOT in kotekan at
 * all: they live upstream in the RFSoC F-engine, and kotekan only ever CONSUMES 4+4b voltages off
 * the wire. So there is no encoder in this tree to copy, and the authoritative spec for what we
 * must PRODUCE is the CONSUMER -- kotekan's own correlator reference model,
 * lib/testing/gpuSimulate.cpp:
 *     int xi = (input[ix] & 0x0f) - 8;         // imag = LOW nibble
 *     int xr = ((input[ix] & 0xf0) >> 4) - 8;  // real = HIGH nibble
 * -- the decode the CHIME/CHORD X-engine is validated against. Match it byte-for-byte and our
 * fftwEngine stands in for an RFSoC: the stream becomes something a real F-engine could emit and a
 * real X-engine could eat. (Do NOT take lib/cuda/cudaQuantizeKernel4.cu as the model: that is the
 * FRB *beam* quantizer -- 8 TWO'S-COMPLEMENT reals per uint32 with a per-chunk scale+offset. It
 * looks authoritative and is a different format for a different product.)
 *
 * SCALING is the caller's job and is what makes 4 bits enough. The bandpass is not flat, so each
 * channel carries its own scale, chosen to put THAT BIN'S NOISE rms at ~2-3 lsb -- the job the
 * RFSoC does at startup. GNSS lives far below the noise, so the noise dithers the buried signal
 * and the quantization error, being uncorrelated with a pseudorandom replica, averages down like
 * thermal noise instead of biasing the correlation. Measured on real sky: < 0.2 dB C/N0 loss at
 * 2 lsb, < 0.14 at 3 lsb (scratchpad/quant_test.py, GPS L5 / Gal E5a / BeiDou B2a). Prefer 3 lsb
 * for RFI headroom, and watch railfrac.
 *
 * This header is deliberately CUDA-FREE (plain floats, no float2) so the CPU quantize stage --
 * which must build in the non-CUDA tree -- and the GPU despread share ONE codec. Duplicating it
 * would be the exact drift lib/cuda/cudaGnssDespreadTest.cpp exists to catch.
 */

#ifdef __CUDACC__
#define GNSS44_HD __host__ __device__ inline
#else
#define GNSS44_HD inline
#endif

namespace gnss44 {

/// The offset-encoded zero. Use for gap fill; memset(0) would write (-8,-8).
constexpr unsigned char ZERO = 0x88;

/// Encode one complex sample. @c inv_scale maps volts -> lsb. Round-to-nearest, CLAMPED to
/// [-8,7]; @c railed (may be null) is set to 1 on a clamp -- a rail is the RFI/CW watchdog's
/// numerator, never silently folded.
GNSS44_HD unsigned char pack(float re, float im, float inv_scale, int* railed) {
    // Deliberately not rintf(): this header is built by both nvcc and gcc, and the +0.5/floor
    // form is identical in both. Values are small (|v| <= 8 after scaling), so no precision risk.
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

/// Decode one 4+4b byte back to volts. @c scale maps lsb -> volts (1/inv_scale).
/// This IS gpuSimulate.cpp's decode; keep it that way.
GNSS44_HD void unpack(unsigned char b, float scale, float* re, float* im) {
    *re = scale * ((float)(int)((b >> 4) & 0x0f) - 8.0f); // real = HIGH nibble, offset -8
    *im = scale * ((float)(int)(b & 0x0f) - 8.0f);        // imag = LOW  nibble, offset -8
}

} // namespace gnss44

#endif // GNSS_44_HPP
