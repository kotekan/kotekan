/**
 * @file f_statistic_reference.h
 * @brief Bit-accurate CPU reference for the fixed-point F-statistic kernel.
 *
 * This reference models packed complex int4 samples and natural matched-filter
 * weights. It computes the same x * conj(w), |z|^2, and uint64 power sums as
 * the CUDA kernel, without modeling CUDA timing or launch behavior.
 */

#pragma once

#include "config.h"

#define FSTAT_REFERENCE_POWER_TERMS FSTAT_NUM_WEIGHT_TERMS

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

void fstat_ref_powers_u64(
    const InputType* x,
    const InputType* w,
    int detector_rows,
    unsigned long long P[FSTAT_REFERENCE_POWER_TERMS]);

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

#ifdef __cplusplus
}
#endif
