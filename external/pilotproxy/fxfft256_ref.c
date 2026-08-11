/* fxfft256 v1 --- frozen deterministic fixed-point FFT reference.
 *
 * This file is the CUDA port template for the deployed fine-reduction
 * FFT: the per-stream code below is what one device thread (or one
 * thread-group stage schedule) must reproduce BIT-FOR-BIT. The Python
 * reference (src/pilot_proxy/fxfft.py) and the golden vectors
 * (tests/data/fxfft256_golden_v1.npz) define the contract; the test
 * suite compiles and bit-compares this file against both.
 *
 * Spec (fxfft256 v1) --- see fxfft.py docstring for the full statement:
 *   N=256 forward DFT (e^{-2 pi i b m / N}), radix-2 DIT, bit-reversed
 *   load of the zero-padded 128-sample input, natural-order output.
 *   Q15 twiddles frozen below (nearest-integer, no ties; +-32768 stored
 *   in int32, exact under the rounding rule). Butterfly t = round15(W*b);
 *   round15(v) = floor((v + 16384) / 32768) via arithmetic shift.
 *   No stage scaling; inputs |re|,|im| <= 2^20 provably cannot overflow
 *   int32 (working values stay below 2^30.7).
 *
 * Build (plain C, no CUDA needed):  cc -O2 -o fxfft256_ref fxfft256_ref.c
 * Harness protocol: ./fxfft256_ref in.bin out.bin
 *   in.bin  = uint32 LE count n, then n * 128 * 2 int32 LE (re, im)
 *   out.bin = n * 256 * 2 int32 LE (re, im)
 *
 * Determinism notes for the CUDA port:
 *   - use only int32/int64 arithmetic; no float, no FMA concerns;
 *   - (v >> 15) on negative int64 must be an arithmetic shift; this is
 *     the behavior of gcc, clang, and nvcc on all supported targets, and
 *     the self-check in main() aborts if the toolchain disagrees;
 *   - the twiddle table must be these literal constants, never libm;
 *   - any butterfly-order optimization is allowed only if the produced
 *     bits are identical (t is exactly 0 when b is 0, so skipping
 *     known-zero legs of stage 1 is safe).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FX_N 256
#define FX_NIN 128
#define FX_SHIFT 15
#define FX_ROUND 16384
#define FX_INPUT_ABS_MAX (1 << 20)

static const int32_t FX_TW[128][2] = {
    {32768, 0}, {32758, -804}, {32729, -1608}, {32679, -2411},
    {32610, -3212}, {32522, -4011}, {32413, -4808}, {32286, -5602},
    {32138, -6393}, {31972, -7180}, {31786, -7962}, {31581, -8740},
    {31357, -9512}, {31114, -10279}, {30853, -11039}, {30572, -11793},
    {30274, -12540}, {29957, -13279}, {29622, -14010}, {29269, -14733},
    {28899, -15447}, {28511, -16151}, {28106, -16846}, {27684, -17531},
    {27246, -18205}, {26791, -18868}, {26320, -19520}, {25833, -20160},
    {25330, -20788}, {24812, -21403}, {24279, -22006}, {23732, -22595},
    {23170, -23170}, {22595, -23732}, {22006, -24279}, {21403, -24812},
    {20788, -25330}, {20160, -25833}, {19520, -26320}, {18868, -26791},
    {18205, -27246}, {17531, -27684}, {16846, -28106}, {16151, -28511},
    {15447, -28899}, {14733, -29269}, {14010, -29622}, {13279, -29957},
    {12540, -30274}, {11793, -30572}, {11039, -30853}, {10279, -31114},
    {9512, -31357}, {8740, -31581}, {7962, -31786}, {7180, -31972},
    {6393, -32138}, {5602, -32286}, {4808, -32413}, {4011, -32522},
    {3212, -32610}, {2411, -32679}, {1608, -32729}, {804, -32758},
    {0, -32768}, {-804, -32758}, {-1608, -32729}, {-2411, -32679},
    {-3212, -32610}, {-4011, -32522}, {-4808, -32413}, {-5602, -32286},
    {-6393, -32138}, {-7180, -31972}, {-7962, -31786}, {-8740, -31581},
    {-9512, -31357}, {-10279, -31114}, {-11039, -30853}, {-11793, -30572},
    {-12540, -30274}, {-13279, -29957}, {-14010, -29622}, {-14733, -29269},
    {-15447, -28899}, {-16151, -28511}, {-16846, -28106}, {-17531, -27684},
    {-18205, -27246}, {-18868, -26791}, {-19520, -26320}, {-20160, -25833},
    {-20788, -25330}, {-21403, -24812}, {-22006, -24279}, {-22595, -23732},
    {-23170, -23170}, {-23732, -22595}, {-24279, -22006}, {-24812, -21403},
    {-25330, -20788}, {-25833, -20160}, {-26320, -19520}, {-26791, -18868},
    {-27246, -18205}, {-27684, -17531}, {-28106, -16846}, {-28511, -16151},
    {-28899, -15447}, {-29269, -14733}, {-29622, -14010}, {-29957, -13279},
    {-30274, -12540}, {-30572, -11793}, {-30853, -11039}, {-31114, -10279},
    {-31357, -9512}, {-31581, -8740}, {-31786, -7962}, {-31972, -7180},
    {-32138, -6393}, {-32286, -5602}, {-32413, -4808}, {-32522, -4011},
    {-32610, -3212}, {-32679, -2411}, {-32729, -1608}, {-32758, -804},
};

static int32_t round15(int64_t v) { return (int32_t)((v + FX_ROUND) >> FX_SHIFT); }

static unsigned bitrev8(unsigned i)
{
    unsigned r = 0, k;
    for (k = 0; k < 8; ++k) r |= ((i >> k) & 1u) << (7u - k);
    return r;
}

/* One stream: 128 (re,im) int32 in, 256 (re,im) int32 out. */
void fxfft256(const int32_t in[FX_NIN][2], int32_t out[FX_N][2])
{
    unsigned i, s;
    for (i = 0; i < FX_N; ++i) {
        unsigned src = bitrev8(i);
        if (src < FX_NIN) { out[i][0] = in[src][0]; out[i][1] = in[src][1]; }
        else { out[i][0] = 0; out[i][1] = 0; }
    }
    for (s = 1; s <= 8; ++s) {
        unsigned m = 1u << s, half = m >> 1, j0, t;
        for (j0 = 0; j0 < FX_N; j0 += m) {
            for (t = 0; t < half; ++t) {
                const int32_t *w = FX_TW[t << (8u - s)];
                int64_t c = w[0], sn = w[1];
                int64_t br = out[j0 + t + half][0], bi = out[j0 + t + half][1];
                int32_t tr = round15(br * c - bi * sn);
                int32_t ti = round15(bi * c + br * sn);
                int32_t ar = out[j0 + t][0], ai = out[j0 + t][1];
                out[j0 + t][0] = ar + tr;
                out[j0 + t][1] = ai + ti;
                out[j0 + t + half][0] = ar - tr;
                out[j0 + t + half][1] = ai - ti;
            }
        }
    }
}

/* Define FXFFT256_REF_NO_MAIN to compile only the fxfft256() transform,
 * e.g. when this file is included/linked into a host test harness that
 * has its own entry point (such as the Kotekan testPilotProxyDetector
 * stage). The default standalone-harness behavior is unchanged. */
#ifndef FXFFT256_REF_NO_MAIN
int main(int argc, char **argv)
{
    FILE *fi, *fo;
    uint32_t n, v;
    int32_t (*in)[2], (*out)[2];
    /* toolchain self-checks: arithmetic shift + two's complement */
    if ((((int64_t)-3) >> 1) != -2 || (int32_t)0x80000000u >= 0) {
        fprintf(stderr, "toolchain lacks arithmetic shift semantics\n");
        return 2;
    }
    if (argc != 3) { fprintf(stderr, "usage: %s in.bin out.bin\n", argv[0]); return 2; }
    fi = fopen(argv[1], "rb");
    if (!fi || fread(&n, 4, 1, fi) != 1) { fprintf(stderr, "bad input\n"); return 2; }
    in = malloc((size_t)FX_NIN * 2 * 4);
    out = malloc((size_t)FX_N * 2 * 4);
    fo = fopen(argv[2], "wb");
    if (!in || !out || !fo) { fprintf(stderr, "alloc/open failed\n"); return 2; }
    for (v = 0; v < n; ++v) {
        unsigned i;
        if (fread(in, 4, FX_NIN * 2, fi) != FX_NIN * 2) { fprintf(stderr, "short read\n"); return 2; }
        for (i = 0; i < FX_NIN; ++i)
            if (in[i][0] > FX_INPUT_ABS_MAX || in[i][0] < -FX_INPUT_ABS_MAX ||
                in[i][1] > FX_INPUT_ABS_MAX || in[i][1] < -FX_INPUT_ABS_MAX) {
                fprintf(stderr, "input exceeds contract\n"); return 2;
            }
        fxfft256((const int32_t (*)[2])in, out);
        if (fwrite(out, 4, FX_N * 2, fo) != FX_N * 2) { fprintf(stderr, "short write\n"); return 2; }
    }
    fclose(fi); fclose(fo); free(in); free(out);
    return 0;
}
#endif /* FXFFT256_REF_NO_MAIN */
