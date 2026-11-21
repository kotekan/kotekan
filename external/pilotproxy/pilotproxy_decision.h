/*
 * pilotproxy_decision.h - the fine decision v3 comparison, shared by every
 * consumer of the detector's [n_peak, d_rho] statistic.
 *
 * Header only, plain C11 or C++, and usable in CUDA device code. A block is
 * kept when
 *
 *     n_peak / d_rho <= word / 2^16,  evaluated as  n_peak * 2^16 <= word * d_rho,
 *
 * with both products formed exactly in 128 bits. This is the test of fine
 * decision v2 (pilotproxy_frac_less in the kernel epilogue) applied to the
 * largest target bin. As in Kotekan's RFI mask, 1 keeps and 0 flags.
 * pilot_proxy.fine_decision.fine_keep_v3 is the Python counterpart.
 */
#ifndef PILOTPROXY_DECISION_H
#define PILOTPROXY_DECISION_H

#include <stdint.h>

#if defined(__CUDACC__)
#define PILOTPROXY_DECISION_FN static inline __host__ __device__
#else
#define PILOTPROXY_DECISION_FN static inline
#endif

/* Exact 64 x 64 -> 128-bit product, from 32-bit halves. */
PILOTPROXY_DECISION_FN void pilotproxy_mul_64x64(uint64_t a, uint64_t b, uint64_t* hi, uint64_t* lo)
{
    const uint64_t a0 = a & 0xffffffffu, a1 = a >> 32;
    const uint64_t b0 = b & 0xffffffffu, b1 = b >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
    *lo = (mid << 32) | (p00 & 0xffffffffu);
    *hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

/*
 * Fine decision v3 of one channel and block, from the detector's pair and a Q16
 * threshold word in [1, 2^64 - 1]. Returns 1 to keep, 0 to flag, and -1 when
 * d_rho is zero and the block has no decision.
 */
PILOTPROXY_DECISION_FN int pilotproxy_v3_keep(uint64_t n_peak, uint64_t d_rho, uint64_t word)
{
    uint64_t lhs_hi, lhs_lo;
    if (d_rho == 0) {
        return -1;
    }
    pilotproxy_mul_64x64(word, d_rho, &lhs_hi, &lhs_lo);
    /* n_peak * 2^16 as hi:lo */
    {
        const uint64_t rhs_hi = n_peak >> 48;
        const uint64_t rhs_lo = n_peak << 16;
        /* keep unless word * d_rho < n_peak * 2^16 */
        return !((lhs_hi < rhs_hi) || (lhs_hi == rhs_hi && lhs_lo < rhs_lo));
    }
}

#endif /* PILOTPROXY_DECISION_H */
