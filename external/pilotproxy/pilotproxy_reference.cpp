#include "pilotproxy_reference.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Frozen fxfft256 v1 reference transform, compiled into this translation
 * unit without its standalone file harness and with internal linkage: the
 * anonymous namespace keeps fxfft256(in[128][2], out[256][2]) private to
 * this file, so a host harness that links it (kotekan's
 * testPilotProxyDetector.cpp links the vendored static library) may include
 * fxfft256_ref.c at file scope for its own use without a duplicate
 * definition. The four C headers the file includes are included above so
 * that, inside the namespace, their include guards leave them empty. The
 * FX_N and FX_NIN macros remain visible. */
#define FXFFT256_REF_NO_MAIN
namespace {
#include "fxfft256_ref.c"
}  // namespace

static_assert(
    FX_NIN == PILOTPROXY_FINE_WINDOWS_PER_STREAM,
    "fxfft256_ref.c input length must equal the frozen window count.");
static_assert(
    FX_N == PILOTPROXY_FINE_NUM_BINS,
    "fxfft256_ref.c transform length must equal the frozen fine bin count.");

namespace {

constexpr int INT4_COMPONENT_BITS = 4;
int sign_extend_nbits(int x, int bits)
{
    const int mask = (1 << bits) - 1;
    const int sign = 1 << (bits - 1);

    x &= mask;
    return (x ^ sign) - sign;
}

}  // namespace

int pilotproxy_ref_sign_extend_i4(int x)
{
    return sign_extend_nbits(x, INT4_COMPONENT_BITS);
}

void pilotproxy_ref_unpack_complex_i4(
    InputType packed,
    int* real,
    int* imag)
{
    const int byte = static_cast<int>(static_cast<unsigned char>(packed));

    if (real != nullptr) {
        *real = pilotproxy_ref_sign_extend_i4(byte >> INT4_COMPONENT_BITS);
    }
    if (imag != nullptr) {
        *imag = pilotproxy_ref_sign_extend_i4(byte);
    }
}

void pilotproxy_ref_complex_mul_conj(
    int xr,
    int xi,
    int wr,
    int wi,
    int* yr,
    int* yi)
{
    if (yr != nullptr) {
        *yr = xr * wr + xi * wi;
    }
    if (yi != nullptr) {
        *yi = xi * wr - xr * wi;
    }
}

void pilotproxy_ref_row_sums_i32(
    const InputType* x,
    const InputType* w,
    int detector_rows,
    int* row_sums_out)
{
    if (row_sums_out == nullptr) {
        return;
    }
    if (x == nullptr || w == nullptr || detector_rows <= 0) {
        return;
    }

    for (int m = 0; m < detector_rows; ++m) {
        for (int n = 0; n < PILOTPROXY_NUM_WEIGHT_TERMS; ++n) {
            int dot_r = 0;
            int dot_i = 0;

            for (int k = 0; k < PILOTPROXY_DETECTOR_WINDOW_SAMPLES; ++k) {
                int xr;
                int xi;
                int wr;
                int wi;
                int yr;
                int yi;

                pilotproxy_ref_unpack_complex_i4(
                    x[m * PILOTPROXY_DETECTOR_WINDOW_SAMPLES + k],
                    &xr,
                    &xi);
                pilotproxy_ref_unpack_complex_i4(
                    w[n * PILOTPROXY_DETECTOR_WINDOW_SAMPLES + k],
                    &wr,
                    &wi);
                pilotproxy_ref_complex_mul_conj(xr, xi, wr, wi, &yr, &yi);

                dot_r += yr;
                dot_i += yi;
            }

            const long long out_index =
                (static_cast<long long>(n) * detector_rows + m) * 2;
            row_sums_out[out_index + 0] = dot_r;
            row_sums_out[out_index + 1] = dot_i;
        }
    }
}

void pilotproxy_ref_fine_powers_u64(
    const int* row_sums,
    int num_streams,
    unsigned long long out[PILOTPROXY_NUM_WEIGHT_TERMS][PILOTPROXY_FINE_NUM_BINS])
{
    if (out == nullptr) {
        return;
    }
    for (int n = 0; n < PILOTPROXY_NUM_WEIGHT_TERMS; ++n) {
        for (int f = 0; f < PILOTPROXY_FINE_NUM_BINS; ++f) {
            out[n][f] = 0ULL;
        }
    }
    if (row_sums == nullptr || num_streams <= 0) {
        return;
    }

    const long long rows =
        static_cast<long long>(num_streams) * PILOTPROXY_FINE_WINDOWS_PER_STREAM;
    int32_t in[FX_NIN][2];
    int32_t spectrum[FX_N][2];
    for (int n = 0; n < PILOTPROXY_NUM_WEIGHT_TERMS; ++n) {
        for (int s = 0; s < num_streams; ++s) {
            const int* src = row_sums
                + (static_cast<long long>(n) * rows
                   + static_cast<long long>(s) * PILOTPROXY_FINE_WINDOWS_PER_STREAM)
                      * 2;
            for (int m = 0; m < FX_NIN; ++m) {
                in[m][0] = src[2 * m];
                in[m][1] = src[2 * m + 1];
            }
            fxfft256(in, spectrum);
            for (int f = 0; f < FX_N; ++f) {
                const long long re = spectrum[f][0];
                const long long im = spectrum[f][1];
                out[n][f] += static_cast<unsigned long long>(re * re + im * im);
            }
        }
    }
}

void pilotproxy_ref_powers_from_fine_u64(
    const unsigned long long fine[PILOTPROXY_NUM_WEIGHT_TERMS][PILOTPROXY_FINE_NUM_BINS],
    unsigned long long P[PILOTPROXY_NUM_WEIGHT_TERMS])
{
    if (P == nullptr) {
        return;
    }
    for (int n = 0; n < PILOTPROXY_NUM_WEIGHT_TERMS; ++n) {
        P[n] = 0ULL;
    }
    if (fine == nullptr) {
        return;
    }
    for (int n = 0; n < PILOTPROXY_NUM_WEIGHT_TERMS; ++n) {
        for (int f = 0; f < PILOTPROXY_FINE_NUM_BINS; ++f) {
            P[n] += fine[n][f];
        }
    }
}
