/**
 * @file f_statistic_reference.h
 * @brief Bit-accurate CPU reference for the fixed-point detector kernel.
 *
 * This reference models packed complex int4 samples and natural matched-filter
 * weights. It computes the same x * conj(w) row sums as the CUDA kernel, the
 * frozen fxfft256 v1 fine power spectra over them, and the frame powers
 * P_r = sum_f S_r[f] that define the kernel's d_power_out (kernel core 3.0.0),
 * without modeling CUDA timing or launch behavior.
 */

#pragma once

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

int fstat_ref_sign_extend_i4(int x);

void fstat_ref_unpack_complex_i4(
    InputType packed,
    int* real,
    int* imag);

void fstat_ref_complex_mul_conj(
    int xr,
    int xi,
    int wr,
    int wi,
    int* yr,
    int* yi);

/**
 * Exact complex per-row dot products (v2 front-end reference).
 * Output layout matches FStat_Compute_RowSums_I32 for one block:
 * row_sums_out[(n * detector_rows + m) * 2 + {0,1}] = {Re, Im} z[n, m].
 */
void fstat_ref_row_sums_i32(
    const InputType* x,
    const InputType* w,
    int detector_rows,
    int* row_sums_out);

/**
 * Exact fine power spectra over a row-sum buffer in the
 * FStat_Compute_RowSums_I32 layout for one block (term-major, stream-major
 * rows, interleaved re/im; detector_rows = num_streams *
 * FSTAT_FINE_WINDOWS_PER_STREAM):
 *
 *   out[n][f] = sum over streams s of |fxfft256(z[n, s, 0..127])[f]|^2
 *
 * with the frozen fxfft256 v1 reference transform (cuda/fxfft256_ref.c,
 * compiled into the reference translation unit with internal linkage, so a
 * harness may include that file itself). Bit-identical to
 * FStat_Compute_FinePowers_U64 and to the fused kernel's d_fine_power_out.
 * out is zeroed first; a null row_sums or a non-positive stream count leaves
 * it zero.
 */
void fstat_ref_fine_powers_u64(
    const int* row_sums,
    int num_streams,
    unsigned long long out[FSTAT_NUM_WEIGHT_TERMS][FSTAT_FINE_NUM_BINS]);

/**
 * Frame powers from fine power spectra: P[n] = sum_f fine[n][f], the exact
 * uint64 all-bin sum. This is the kernel core 3.0.0 definition of the
 * d_power_out of the fused entries; F = 2 P[0] / (P[1] + P[2]).
 */
void fstat_ref_powers_from_fine_u64(
    const unsigned long long fine[FSTAT_NUM_WEIGHT_TERMS][FSTAT_FINE_NUM_BINS],
    unsigned long long P[FSTAT_NUM_WEIGHT_TERMS]);

#ifdef __cplusplus
}
#endif
