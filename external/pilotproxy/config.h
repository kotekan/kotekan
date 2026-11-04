/**
 * @file config.h
 * @brief Compile-time configuration for the PilotProxy detector.
 *
 * Public host code should use detector_window_samples,
 * num_weight_terms, and reference_offset_bins.
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
 * Core version reported by PilotProxy_GetVersion. The major number changes when an
 * entry point is removed or an output changes meaning; the minor number when
 * an entry point is added.
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
 * K is a per-receiver build parameter. It scales with the coarse-channel
 * width to hold the fine-bin width near 3.05 kHz: CHIME channels are
 * 390.625 kHz wide and use K = 128; CHORD channels are 195.3125 kHz wide
 * and use K = 64. The window count per detector block
 * (PILOTPROXY_FINE_WINDOWS_PER_STREAM) is frozen at 128 for every receiver;
 * only the window length varies. Supported values are 64 and 128, and
 * the build must define it; there is no default. The
 * guard-bit chain and the static assertions below enforce the fixed-point
 * closure for each.
 */
#ifndef PILOTPROXY_DETECTOR_WINDOW_SAMPLES
#error "Define PILOTPROXY_DETECTOR_WINDOW_SAMPLES (64 or 128) when building the core."
#endif

/**
 * Number of detector weight terms: target, lower reference, upper reference.
 */
#define PILOTPROXY_NUM_WEIGHT_TERMS 3

/**
 * Indices inside the three-term power vector.
 */
#define PILOTPROXY_TARGET_WEIGHT_INDEX 0
#define PILOTPROXY_LOWER_REFERENCE_WEIGHT_INDEX 1
#define PILOTPROXY_UPPER_REFERENCE_WEIGHT_INDEX 2

/*
 * Two-reference raw power ratio contract: F = 2*P_target/(P_ref1+P_ref2),
 * where each P_r is the frame power, the all-bin sum of that term's fine
 * spectrum. The library forms neither the ratio nor any
 * float: hosts form F from the exact uint64 frame powers.
 */

/**
 * fxfft256 v1 fine-reduction geometry (frozen). The window-axis transform
 * length and padding are part of the frozen FFT specification
 * (src/pilot_proxy/fxfft.py); they are not tunable parameters.
 */
#define PILOTPROXY_FINE_WINDOWS_PER_STREAM 128
#define PILOTPROXY_FINE_PAD_FACTOR 2
#define PILOTPROXY_FINE_NUM_BINS \
    (PILOTPROXY_FINE_PAD_FACTOR * PILOTPROXY_FINE_WINDOWS_PER_STREAM)

/**
 * Radix-2 stage count of the fine transform, log2(PILOTPROXY_FINE_NUM_BINS).
 *
 * Set alongside PILOTPROXY_FINE_WINDOWS_PER_STREAM; the relation is asserted below.
 * The transform reads twiddles from the shared master table, so a stage indexes
 * it as `t << (FX_MASTER_LOG2 - stage)`: the decimation stride folds into the
 * shift, and at PILOTPROXY_FINE_NUM_BINS = 256 that selects exactly the entries the
 * frozen fxfft256 v1 table held.
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
 * This is not the final mask guard-bin dilation parameter.
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
 * capacity (closures asserted below). The kernel is fixed point end to end:
 * dot products, the frozen transform, the fine sums and their reduction to
 * frame powers are all exact integers; the library produces no float.
 */
#define PILOTPROXY_USE_UINT64_POWER_ACCUMULATION 1

/**
 * Use the DP4A dot-product path (1) or the scalar integer path (0).
 *
 * DP4A is the default production path for the detector. It keeps
 * packed samples in global memory, pre-packs the fixed weights into int8
 * dot-product lanes, and logically unpacks two packed complex samples at a time
 * in registers. The scalar path remains available via make DP4A=0 for direct
 * validation and debugging.
 */
#ifndef PILOTPROXY_USE_DP4A
#define PILOTPROXY_USE_DP4A 1
#endif

/**
 * Store DP4A weight lanes in CUDA constant memory.
 *
 * The detector has only 3 x (K/2) int32 DP4A weight lanes, or 768 bytes at
 * K = 128.
 * Constant memory is left as an experimental option because the production
 * DP4A access pattern has different warp lanes reading different weight-lane
 * addresses, which can serialize constant-memory access on Ada.  It is also a
 * module-global table, so it assumes one active weight set per process call
 * sequence.
 */
#ifndef PILOTPROXY_USE_CONSTANT_WEIGHT_LANES
#define PILOTPROXY_USE_CONSTANT_WEIGHT_LANES 0
#endif

/**
 * Preload DP4A weight lanes into per-block shared memory.
 *
 * The weight-lane table is tiny, read-only during a kernel launch, and reused
 * for every detector row handled by the block.  Shared-memory preload avoids
 * repeated per-row global loads while preserving per-handle weight storage and
 * avoiding the divergent-read penalty observed with constant memory.
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
 * One complex product component is the sum/difference of two int4xint4
 * products: 4 + 4 + 1 = 9 bits. Accumulating K taps adds log2(K) guard bits,
 * so completed real/imaginary dot-product components require 9 + log2(K)
 * bits. At K = 128 that is 16 bits, which is why the row sums fit an int16
 * range even though they are stored as int32; at K = 64 it is 15 bits with
 * the same storage. Shared-memory footprint scales with the window count,
 * and this margin is the lever that halves it.
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
 * Samples lie in [-8, 7] and quantized weight components in [-7, 7] (the
 * receiver's 15-level mid-tread quantizer emits only [-7, 7]); these bounds
 * conservatively admit the full two's-complement range on both, so one
 * complex-product component is bounded by 8*8 + 8*8 = 128 and a completed
 * K-tap dot-product component by 128*K.
 *
 * The bound is conservative (the true weight bound 7 gives 112*K). Which
 * window lengths are admitted is decided by the explicit guard-bit list
 * above (the #error for an unsupported PILOTPROXY_DETECTOR_WINDOW_SAMPLES) and
 * by the fxfft and frame-power closures asserted below, not by this margin.
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
 * Worst-case infinity-norm growth of one fxfft radix-2 butterfly:
 * |a +- W*b| <= |a| + |W||b| with |W| <= 1 + 2^-15 after Q15 rounding, so a
 * stage maps the working norm M -> (1 + sqrt(2) + 2^-15) M + 1. sqrt(2) is
 * bounded from above by 1.41422 to keep the result conservative.
 * growth_after() bounds M / M_0 for input norm M_0 and adds one M_0 per
 * stage, which covers the + 1 for any M_0 >= 1; at eight stages from 2^20 it
 * gives 2^30.94, looser than the 2^30.2 of the recursion itself.
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
 * Frame-power closure. The frame power of one term is
 * the all-bin sum of its fine spectrum, P_r = sum_f S_r[f], held in uint64.
 * For the exact DFT X of one stream, Parseval over the zero-padded
 * transform gives sum_f |X_f|^2 = ||X||^2 = L_F * sum_m |z_m|^2. The frozen
 * fixed-point transform X~ obeys the 2-norm bound
 *
 *   ||X~|| <= (1 + d) ||X|| + c,   d <= 2.6e-4,   c <= 192,
 *
 * where d is the Q15 twiddle rounding (relative error at most
 * sqrt(2) * 2^-15 per stage; stages 1 and 2 use the exact twiddles 1 and
 * -i, so six stages round) and c is the round15 term (at most 0.5 per
 * component per butterfly, 2-norm sqrt(128) per stage, propagated through
 * the remaining stages' sqrt(2) growth). The additive term is why the
 * RATIO P_r / (L_F * sum |z|^2) is not pinned near 1 at small amplitude
 * (unit-amplitude rows depart from Parseval by tens of per cent); the
 * bound on P_r itself is monotone in ||X||, so it is set by the maximum
 * input norm, where the additive term is small: at ||X|| = 2^22 per stream
 * (K = 128) the factor on the power is (1 + d + c / 2^22)^2 <= 1.00061, and
 * at ||X|| = 2^21 (K = 64) it is <= 1.00070, both inside the 1.001 slack
 * applied below. Summed over streams,
 *
 *   P_r <= L_F * 2 * R^2 * W * N_streams * 1.001
 *
 * with R = PILOTPROXY_MAX_ROW_SUM_COMPONENT, W = PILOTPROXY_FINE_WINDOWS_PER_STREAM
 * and L_F = PILOTPROXY_FINE_NUM_BINS. At K = 128 that is 2^60 * 1.001 for the
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
 * The DP4A path consumes two taps per lane, so 64 threads cover the 64
 * tap-pairs of a K = 128 window with no inactive tap-pair workers.  At
 * K = 64 the same block strides the 32 tap-pairs with half the lanes active
 * in that loop.  This is the default production block size.  The scalar
 * reference path remains correct because it strides over taps.
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
 * Maximum grid dimension of the staged row-sum launches.  With a 128-tap
 * detector window and 64-thread DP4A blocks, 4096 blocks gives one block per
 * 64 detector rows for a reference-sized 262144-row block.  It also sets the
 * stream capacity of the handle-less fine stage
 * (PILOTPROXY_BLOCK_THREADS * PILOTPROXY_GRID_MAX_BLOCKS streams).  The fused kernel
 * launches one block per stream and is not capped by it.
 */
#ifndef PILOTPROXY_GRID_MAX_BLOCKS
#define PILOTPROXY_GRID_MAX_BLOCKS 4096
#endif
