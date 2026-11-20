/**
 * @file config.h
 * @brief Compile-time configuration for the PilotProxy detector.
 *
 * Host code reads these values from the compiled library as
 * detector_window_samples, num_weight_terms, and reference_offset_bins
 * (PilotProxy_GetSpecs).
 */

#pragma once

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

/* ===========================================================================
 * CORE VERSION
 * ===========================================================================*/

/*
 * Core version reported by PilotProxy_GetVersion. The major number changes
 * when an entry point is removed or an output changes meaning. The minor
 * number changes when an entry point is added.
 */
#define PILOTPROXY_CORE_VERSION_MAJOR 4
#define PILOTPROXY_CORE_VERSION_MINOR 0
#define PILOTPROXY_CORE_VERSION_PATCH 0

/* ===========================================================================
 * ALGORITHM PARAMETERS
 * ===========================================================================*/

/**
 * Detector-window samples (K) in one matched-filter vector.
 *
 * K is chosen to keep the matched-filter cell width near 3.05 kHz. CHIME's
 * 390.625 kHz coarse channels use K = 128, and CHORD's 195.3125 kHz channels
 * use K = 64. Each stream still contributes 128 windows to the fine
 * transform (PILOTPROXY_FINE_WINDOWS_PER_STREAM).
 *
 * The build must set K to 64 or 128, and there is no default. The integer
 * bounds for that choice are checked by the guard-bit calculation and the
 * static assertions below.
 */
#ifndef PILOTPROXY_DETECTOR_WINDOW_SAMPLES
#error "Define PILOTPROXY_DETECTOR_WINDOW_SAMPLES (64 or 128) when building the core."
#endif

/**
 * Number of detector weight terms: target, lower reference, upper reference.
 */
#define PILOTPROXY_NUM_WEIGHT_TERMS 3

/**
 * Indices of the three terms in the power vector.
 */
#define PILOTPROXY_TARGET_WEIGHT_INDEX 0
#define PILOTPROXY_LOWER_REFERENCE_WEIGHT_INDEX 1
#define PILOTPROXY_UPPER_REFERENCE_WEIGHT_INDEX 2

/*
 * The two-reference power ratio is F = 2*P_target/(P_ref1+P_ref2), where
 * each P_r is the frame power, the sum over all bins of that term's fine
 * spectrum. The library does not form the ratio or any float. The host forms
 * F from the exact uint64 frame powers.
 */

/**
 * fxfft256 v1 fine-reduction geometry (frozen). The transform length along
 * the window axis and the padding are part of the frozen FFT specification
 * (src/pilot_proxy/fxfft.py) and are not tunable.
 */
#define PILOTPROXY_FINE_WINDOWS_PER_STREAM 128
#define PILOTPROXY_FINE_PAD_FACTOR 2
#define PILOTPROXY_FINE_NUM_BINS \
    (PILOTPROXY_FINE_PAD_FACTOR * PILOTPROXY_FINE_WINDOWS_PER_STREAM)

/**
 * Radix-2 stage count of the fine transform, log2(PILOTPROXY_FINE_NUM_BINS).
 *
 * It is set with PILOTPROXY_FINE_WINDOWS_PER_STREAM, and the relation is
 * asserted below. The transform reads its twiddles from the shared master
 * table, and each stage indexes the table as `t << (FX_MASTER_LOG2 - stage)`.
 * The decimation stride is folded into the shift, and at
 * PILOTPROXY_FINE_NUM_BINS = 256 this selects exactly the entries the frozen
 * fxfft256 v1 table held.
 */
#define PILOTPROXY_FINE_LOG2 8

/**
 * Reference-bin offset used to construct the three detector weight terms.
 *
 * The weight terms are:
 *   t = PILOTPROXY_TARGET_WEIGHT_INDEX: target bin
 *   t = PILOTPROXY_LOWER_REFERENCE_WEIGHT_INDEX: lower reference bin
 *   t = PILOTPROXY_UPPER_REFERENCE_WEIGHT_INDEX: upper reference bin
 *
 * An offset of 2 leaves one detector bin between the target bin and each
 * reference bin. The guard-bin dilation of the final mask is a different
 * parameter.
 */
#define PILOTPROXY_REFERENCE_BIN_OFFSET 2

/* ===========================================================================
 * DATA TYPE CONFIGURATION
 * ===========================================================================*/

/**
 * Use uint64 integer accumulation for the fine and frame power sums.
 *
 * For the int4xint4 detector at any supported K, every fine bin S_r[f] and
 * every frame power P_r = sum_f S_r[f] fits uint64 up to the 65536-stream
 * capacity (closures asserted below). The kernel is fixed point from end to
 * end. The dot products, the frozen transform, the fine sums and their
 * reduction to frame powers are all exact integers, and the library produces
 * no float.
 */
#define PILOTPROXY_USE_UINT64_POWER_ACCUMULATION 1

/**
 * Use the DP4A dot-product path (1) or the scalar integer path (0).
 *
 * DP4A is the default production path for the detector. In this path the
 * packed samples stay in global memory, the fixed weights are packed in
 * advance into int8 dot-product lanes, and two packed complex samples at a
 * time are unpacked in registers. The scalar path is built with make DP4A=0
 * for validation and debugging.
 */
#ifndef PILOTPROXY_USE_DP4A
#define PILOTPROXY_USE_DP4A 1
#endif

/**
 * Store DP4A weight lanes in CUDA constant memory.
 *
 * The detector has only 3 x (K/2) int32 DP4A weight lanes, or 768 bytes at
 * K = 128. Constant memory is kept as an experimental option. In the
 * production DP4A access pattern, different warp lanes read different
 * weight-lane addresses, which can serialize constant-memory access on Ada.
 * The constant table is also module-global, so it assumes one active weight
 * set per process call sequence.
 */
#ifndef PILOTPROXY_USE_CONSTANT_WEIGHT_LANES
#define PILOTPROXY_USE_CONSTANT_WEIGHT_LANES 0
#endif

/**
 * Preload DP4A weight lanes into per-block shared memory.
 *
 * The weight-lane table is small, read-only during a kernel launch, and reused
 * for every detector row of the block. Preloading it into shared memory
 * avoids a global load for each row and keeps the weights stored per handle.
 * It also avoids the divergent-read penalty that was found with constant
 * memory.
 */
#ifndef PILOTPROXY_USE_SHARED_WEIGHT_LANES
#define PILOTPROXY_USE_SHARED_WEIGHT_LANES 1
#endif

#if PILOTPROXY_USE_CONSTANT_WEIGHT_LANES && PILOTPROXY_USE_SHARED_WEIGHT_LANES
#error "Choose exactly one weight-lane placement mode: constant or shared."
#endif

/**
 * Packed complex sample data type.
 *
 * int8_t stores signed packed 4+4-bit complex samples.
 */
typedef int8_t InputType;

/**
 * Bits per real/imaginary component, derived from InputType size.
 */
#define PILOTPROXY_SAMPLE_BITS_PER_COMPONENT (sizeof(InputType) * 4)

/**
 * Fixed-point bit-growth model for the int4 detector datapath.
 *
 * One complex product component is the sum or difference of two int4xint4
 * products and needs 4 + 4 + 1 = 9 bits. Accumulating K taps adds log2(K)
 * guard bits, so completed real and imaginary dot-product components need
 * 9 + log2(K) bits. At K = 128 that is 16 bits, which is why the row sums
 * fit an int16 range even though they are stored as int32. At K = 64 the
 * bound is 15 bits. Both configurations store the row sums as int32.
 *
 * PILOTPROXY_DOT_ACCUM_GUARD_BITS must equal log2(PILOTPROXY_DETECTOR_WINDOW_SAMPLES).
 * It is derived here for each supported window length and asserted below.
 */
#define PILOTPROXY_INT4_COMPONENT_BITS 4
#define PILOTPROXY_COMPLEX_PRODUCT_COMPONENT_BITS \
    (PILOTPROXY_INT4_COMPONENT_BITS + PILOTPROXY_INT4_COMPONENT_BITS + 1)
#if PILOTPROXY_DETECTOR_WINDOW_SAMPLES == 128
#define PILOTPROXY_DOT_ACCUM_GUARD_BITS 7
#elif PILOTPROXY_DETECTOR_WINDOW_SAMPLES == 64
#define PILOTPROXY_DOT_ACCUM_GUARD_BITS 6
#else
#error "Unsupported PILOTPROXY_DETECTOR_WINDOW_SAMPLES; add its guard-bit count."
#endif
#define PILOTPROXY_COMPLEX_DOT_COMPONENT_BITS \
    (PILOTPROXY_COMPLEX_PRODUCT_COMPONENT_BITS + PILOTPROXY_DOT_ACCUM_GUARD_BITS)

/**
 * Worst-case magnitude bounds used by the static assertions below.
 *
 * Samples lie in [-8, 7] and quantized weight components in [-7, 7], since
 * the receiver's 15-level mid-tread quantizer emits only [-7, 7]. The bounds
 * here admit the full two's-complement range on both, so one complex-product
 * component is bounded by 8*8 + 8*8 = 128 and a completed K-tap dot-product
 * component by 128*K.
 *
 * The bound is conservative, and the true weight bound of 7 gives 112*K. The
 * admitted window lengths are set by the guard-bit list above (the #error
 * for an unsupported PILOTPROXY_DETECTOR_WINDOW_SAMPLES) and by the fxfft and
 * frame-power closures asserted below.
 */
#define PILOTPROXY_MAX_SAMPLE_COMPONENT 8
#define PILOTPROXY_MAX_PRODUCT_COMPONENT \
    (2 * PILOTPROXY_MAX_SAMPLE_COMPONENT * PILOTPROXY_MAX_SAMPLE_COMPONENT)
#define PILOTPROXY_MAX_ROW_SUM_COMPONENT \
    ((long long)PILOTPROXY_MAX_PRODUCT_COMPONENT * PILOTPROXY_DETECTOR_WINDOW_SAMPLES)

/** DP4A consumes two packed complex samples per tap pair. */
#define PILOTPROXY_DP4A_TAP_PAIRS (PILOTPROXY_DETECTOR_WINDOW_SAMPLES / 2)

#ifdef __cplusplus
namespace pilotproxy_geometry {

/** Compile-time integer log2 for exact powers of two. */
constexpr int log2_exact(long long v) {
    return v <= 1 ? 0 : 1 + log2_exact(v / 2);
}
constexpr bool is_pow2(long long v) {
    return v > 0 && (v & (v - 1)) == 0;
}

/**
 * Worst-case infinity-norm growth of one fxfft radix-2 butterfly. Since
 * |a +- W*b| <= |a| + |W||b| with |W| <= 1 + 2^-15 after Q15 rounding, a
 * stage maps the working norm M to (1 + sqrt(2) + 2^-15) M + 1. The value of
 * sqrt(2) is bounded from above by 1.41422 to keep the result conservative.
 * growth_after() bounds M / M_0 for an input norm M_0 and adds one M_0 per
 * stage, which covers the + 1 for any M_0 >= 1. At eight stages from 2^20 it
 * gives 2^30.94, which is looser than the 2^30.2 of the recursion itself.
 */
constexpr double kButterflyGrowth = 1.0 + 1.41422 + (1.0 / 32768.0);
constexpr double growth_after(int stages) {
    return stages == 0 ? 1.0 : kButterflyGrowth * growth_after(stages - 1) + 1.0;
}

constexpr int kFineStages = log2_exact(PILOTPROXY_FINE_NUM_BINS);
/** Post-transform bin magnitude, bounded by the triangle inequality. */
constexpr double kMaxBinMagnitude =
    static_cast<double>(PILOTPROXY_MAX_ROW_SUM_COMPONENT) * PILOTPROXY_FINE_WINDOWS_PER_STREAM;

/**
 * Frame-power closure. The frame power of one term is the sum over all bins
 * of its fine spectrum, P_r = sum_f S_r[f], held in uint64. For the exact
 * DFT X of one stream, Parseval over the zero-padded transform gives
 * sum_f |X_f|^2 = ||X||^2 = L_F * sum_m |z_m|^2. The frozen fixed-point
 * transform X~ obeys the 2-norm bound
 *
 *   ||X~|| <= (1 + d) ||X|| + c,   d <= 2.6e-4,   c <= 192,
 *
 * where d is the Q15 twiddle rounding and c is the round15 term. The
 * relative twiddle error is at most sqrt(2) * 2^-15 per stage, and only six
 * stages round, since stages 1 and 2 use the exact twiddles 1 and -i. The
 * round15 term is at most 0.5 per component per butterfly, or a 2-norm of
 * sqrt(128) per stage, and the error added at one stage grows by sqrt(2)
 * through each remaining stage.
 *
 * Because of the additive term, the ratio P_r / (L_F * sum |z|^2) is not
 * pinned near 1 at small amplitude, and unit-amplitude rows depart from
 * Parseval by tens of percent. The bound on P_r itself is monotone in ||X||,
 * so it is set by the maximum input norm, where the additive term is small.
 * At ||X|| = 2^22 per stream (K = 128) the factor on the power is
 * (1 + d + c / 2^22)^2 <= 1.00061, and at ||X|| = 2^21 (K = 64) it is
 * <= 1.00070. Both fit in the 1.001 slack applied below. Summed over
 * streams,
 *
 *   P_r <= L_F * 2 * R^2 * W * N_streams * 1.001
 *
 * with R = PILOTPROXY_MAX_ROW_SUM_COMPONENT, W = PILOTPROXY_FINE_WINDOWS_PER_STREAM
 * and L_F = PILOTPROXY_FINE_NUM_BINS. At K = 128 this is 2^60 * 1.001 for the
 * 65536-stream capacity (about 4 bits of margin to 2^64) and 2^55 * 1.001
 * at the 2048-stream CHIME frame (9 bits). The product is formed in
 * saturating integer arithmetic so the assertion itself cannot overflow at
 * compile time, and the 1.001 slack is applied as B + B/1000 without
 * widening.
 */
constexpr unsigned long long kU64Max = 18446744073709551615ULL;
constexpr unsigned long long mul_sat_u64(unsigned long long a, unsigned long long b) {
    return (a != 0ULL && b > kU64Max / a) ? kU64Max : a * b;
}
constexpr unsigned long long kClosureStreams = 65536ULL;
constexpr unsigned long long kRowSumComponent =
    static_cast<unsigned long long>(PILOTPROXY_MAX_ROW_SUM_COMPONENT);
constexpr unsigned long long kFramePowerBound = mul_sat_u64(
    mul_sat_u64(
        mul_sat_u64(
            static_cast<unsigned long long>(PILOTPROXY_FINE_NUM_BINS),
            mul_sat_u64(2ULL, mul_sat_u64(kRowSumComponent, kRowSumComponent))),
        static_cast<unsigned long long>(PILOTPROXY_FINE_WINDOWS_PER_STREAM)),
    kClosureStreams);

} // namespace pilotproxy_geometry

/* --- geometry consistency ------------------------------------------------ */
static_assert(
    pilotproxy_geometry::is_pow2(PILOTPROXY_DETECTOR_WINDOW_SAMPLES)
        && PILOTPROXY_DETECTOR_WINDOW_SAMPLES >= 2,
    "detector_window_samples must be a power of two >= 2.");
static_assert(
    PILOTPROXY_SAMPLE_BITS_PER_COMPONENT == PILOTPROXY_INT4_COMPONENT_BITS,
    "Production detector uses packed complex int4 samples and weights.");
static_assert(
    PILOTPROXY_DP4A_TAP_PAIRS * 2 == PILOTPROXY_DETECTOR_WINDOW_SAMPLES,
    "DP4A tap pairs must be exactly detector_window_samples/2.");
static_assert(
    (1 << PILOTPROXY_DOT_ACCUM_GUARD_BITS) == PILOTPROXY_DETECTOR_WINDOW_SAMPLES,
    "PILOTPROXY_DOT_ACCUM_GUARD_BITS must equal log2(detector_window_samples); "
    "update it whenever the window length changes.");
static_assert(
    pilotproxy_geometry::is_pow2(PILOTPROXY_FINE_NUM_BINS),
    "Fine transform length must be a power of two.");
static_assert(
    PILOTPROXY_FINE_NUM_BINS == PILOTPROXY_FINE_PAD_FACTOR * PILOTPROXY_FINE_WINDOWS_PER_STREAM,
    "Fine bin count must be pad_factor * windows_per_stream.");
static_assert(
    (1 << PILOTPROXY_FINE_LOG2) == PILOTPROXY_FINE_NUM_BINS,
    "PILOTPROXY_FINE_LOG2 must equal log2(PILOTPROXY_FINE_NUM_BINS); update it whenever "
    "the fine transform length changes.");

/* --- no-overflow closure, derived per geometry ---------------------------- */
static_assert(
    PILOTPROXY_MAX_ROW_SUM_COMPONENT <= 2147483647LL,
    "Completed dot-product components must fit the int32 row-sum storage.");
static_assert(
    pilotproxy_geometry::kFramePowerBound < pilotproxy_geometry::kU64Max
        && pilotproxy_geometry::kFramePowerBound
               <= pilotproxy_geometry::kU64Max - pilotproxy_geometry::kFramePowerBound / 1000ULL,
    "Frame power sum_f S_r[f] must fit uint64, with the 1.001 fixed-point "
    "Parseval slack, for stream counts up to 65536; reduce "
    "detector_window_samples or the stream capacity.");
static_assert(
    static_cast<double>(PILOTPROXY_MAX_ROW_SUM_COMPONENT)
            * pilotproxy_geometry::growth_after(pilotproxy_geometry::kFineStages)
        <= 2147483647.0,
    "fxfft working values must stay inside int32 across all butterfly stages; "
    "the admissible input norm falls as 2^31 / growth^stages, so this bound "
    "tightens every time the transform length doubles.");
static_assert(
    2.0 * pilotproxy_geometry::kMaxBinMagnitude * pilotproxy_geometry::kMaxBinMagnitude
            * 65536.0
        <= 18446744073709551615.0,
    "Feed-summed fine power must fit uint64 for stream counts up to 65536.");
#endif

/* ===========================================================================
 * GPU EXECUTION PARAMETERS
 * ===========================================================================*/

/**
 * Threads per CUDA block.
 *
 * The DP4A path consumes two taps per lane, so 64 threads cover the 64 tap
 * pairs of a K = 128 window with no idle lanes. At K = 64 the same block
 * strides over the 32 tap pairs with half the lanes active in that loop.
 * This is the default production block size. The scalar reference path
 * stays correct because it strides over taps.
 */
#ifndef PILOTPROXY_BLOCK_THREADS
#define PILOTPROXY_BLOCK_THREADS 64
#endif

/** CUDA warp size. */
#define PILOTPROXY_WARP_SIZE 32

/** Warps per CUDA block. */
#define PILOTPROXY_WARPS_PER_BLOCK (PILOTPROXY_BLOCK_THREADS / PILOTPROXY_WARP_SIZE)

/** Full warp participation mask for shuffle operations. */
#define PILOTPROXY_WARP_MASK 0xFFFFFFFF

/**
 * Maximum grid dimension of the staged row-sum launches. With a 128-tap
 * detector window and 64-thread DP4A blocks, 4096 blocks give one block per
 * 64 detector rows for a reference-sized block of 262144 rows. It also sets
 * the stream capacity of the handle-less fine stage
 * (PILOTPROXY_BLOCK_THREADS * PILOTPROXY_GRID_MAX_BLOCKS streams). The fused
 * kernel launches one block per stream and has no such cap.
 */
#ifndef PILOTPROXY_GRID_MAX_BLOCKS
#define PILOTPROXY_GRID_MAX_BLOCKS 4096
#endif
