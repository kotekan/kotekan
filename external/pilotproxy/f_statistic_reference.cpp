#include "f_statistic_reference.h"

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

int fstat_ref_sign_extend_i4(int x)
{
    return sign_extend_nbits(x, INT4_COMPONENT_BITS);
}

void fstat_ref_unpack_complex_i4(
    InputType packed,
    int* real,
    int* imag)
{
    const int byte = static_cast<int>(static_cast<unsigned char>(packed));

    if (real != nullptr) {
        *real = fstat_ref_sign_extend_i4(byte >> INT4_COMPONENT_BITS);
    }
    if (imag != nullptr) {
        *imag = fstat_ref_sign_extend_i4(byte);
    }
}

void fstat_ref_complex_mul_conj(
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

void fstat_ref_powers_u64(
    const InputType* x,
    const InputType* w,
    int detector_rows,
    unsigned long long P[FSTAT_REFERENCE_POWER_TERMS])
{
    if (P == nullptr) {
        return;
    }

    for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
        P[n] = 0ULL;
    }

    if (x == nullptr || w == nullptr || detector_rows <= 0) {
        return;
    }

    for (int m = 0; m < detector_rows; ++m) {
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            int dot_r = 0;
            int dot_i = 0;

            for (int k = 0; k < FSTAT_DETECTOR_WINDOW_SAMPLES; ++k) {
                int xr;
                int xi;
                int wr;
                int wi;
                int yr;
                int yi;

                fstat_ref_unpack_complex_i4(
                    x[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k],
                    &xr,
                    &xi);
                fstat_ref_unpack_complex_i4(
                    w[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k],
                    &wr,
                    &wi);
                fstat_ref_complex_mul_conj(xr, xi, wr, wi, &yr, &yi);

                dot_r += yr;
                dot_i += yi;
            }

            const long long zr = static_cast<long long>(dot_r);
            const long long zi = static_cast<long long>(dot_i);
            P[n] += static_cast<unsigned long long>(zr * zr + zi * zi);
        }
    }
}

void fstat_ref_row_sums_i32(
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
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            int dot_r = 0;
            int dot_i = 0;

            for (int k = 0; k < FSTAT_DETECTOR_WINDOW_SAMPLES; ++k) {
                int xr;
                int xi;
                int wr;
                int wi;
                int yr;
                int yi;

                fstat_ref_unpack_complex_i4(
                    x[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k],
                    &xr,
                    &xi);
                fstat_ref_unpack_complex_i4(
                    w[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k],
                    &wr,
                    &wi);
                fstat_ref_complex_mul_conj(xr, xi, wr, wi, &yr, &yi);

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
