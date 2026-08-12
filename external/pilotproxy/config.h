/**
 * @file config.h
 * @brief Compile-time configuration for the F-statistic CUDA kernel.
 *
 * Public host code should use detector_window_samples,
 * num_weight_terms, and reference_offset_bins.
 */

#pragma once

#ifdef __cplusplus
#include <cfloat>
#include <cstdint>
#else
#include <float.h>
#include <stdint.h>
#endif

/* ===========================================================================
 * CORE VERSION
 * ===========================================================================*/

/*
 * 2.0.0: adds the v2 time-coherent front end (exact complex row-sum
 * emission via FStat_Compute_RowSums_I32). Existing v1 entry points are
 * unchanged; the ABI extension is additive. The major bump marks the
 * detector-statistic family change (fine-bin reduction downstream).
 *
 * 2.1.0: adds the on-device fine-reduction power stage
 * (FStat_Compute_FinePowers_U64): the frozen fxfft256 v1 fixed-point FFT
 * over the row-sum buffer plus exact uint64 feed sums. Existing entry
 * points are unchanged; the extension is additive and policy-free (no
 * CFAR or designated-set decision in the library).
 *
 * 2.2.0: adds the fused fine kernel (FStat_Compute_FusedFine_U64): one
 * launch from packed samples to exact fine and coarse power sums,
 * block-per-stream with shared-memory row sums (materialized to global
 * only through the optional debug tap) and the marginal identity
 * internal to the launch. Bit-identical to the composed
 * RowSums -> FinePowers path plus Powers_U64. Existing entry points are
 * unchanged; still policy-free.
 *
 * 2.3.0: adds the decision epilogue (FStat_Compute_FusedFineMask_U64):
 * the fused datapath plus a last-block epilogue forming the frozen fine
 * decision v1 (rank-based null-bulk CFAR, designated-set compare, exact
 * 128/192-bit integer comparisons; bit-identical to
 * src/pilot_proxy/fine_decision.py) and emitting one mask bit per
 * aligned frame. Calibration inputs (anchor, width, bulk mask, rank,
 * Q16 multiplier) are runtime-bundle data passed as arguments --- the
 * library remains policy-free in the sense that it compiles in no
 * channel constants. Existing entry points are unchanged;
 * FStat_Compute_FusedFine_U64 output is bit-identical to 2.2.0.
 */
#define FSTAT_CORE_VERSION_MAJOR 2
#define FSTAT_CORE_VERSION_MINOR 3
#define FSTAT_CORE_VERSION_PATCH 0

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
 * (FSTAT_FINE_WINDOWS_PER_STREAM) is frozen at 128 for every receiver;
 * only the window length varies. Supported values are 64 and 128. The
 * guard-bit chain and the static assertions below enforce the fixed-point
 * closure for each.
 */
#ifndef FSTAT_DETECTOR_WINDOW_SAMPLES
#define FSTAT_DETECTOR_WINDOW_SAMPLES 128
#endif

/**
 * Number of detector weight terms: target, lower reference, upper reference.
 */
#define FSTAT_NUM_WEIGHT_TERMS 3

/**
 * Indices inside the three-term power vector.
 */
#define FSTAT_TARGET_WEIGHT_INDEX 0
#define FSTAT_LOWER_REFERENCE_WEIGHT_INDEX 1
#define FSTAT_UPPER_REFERENCE_WEIGHT_INDEX 2

/**
 * Two-reference raw F-statistic contract: F = 2*P_target/(P_ref1+P_ref2).
 */
#define FSTAT_RAW_NUMDEN_SCALE 2.0
#define FSTAT_NO_PILOT_EXCESS_RAW_F 1.0

/**
 * fxfft256 v1 fine-reduction geometry (frozen). The window-axis transform
 * length and padding are part of the frozen FFT specification
 * (src/pilot_proxy/fxfft.py); they are not tunable parameters.
 */
#define FSTAT_FINE_WINDOWS_PER_STREAM 128
#define FSTAT_FINE_PAD_FACTOR 2
#define FSTAT_FINE_NUM_BINS \
    (FSTAT_FINE_PAD_FACTOR * FSTAT_FINE_WINDOWS_PER_STREAM)

/**
 * Radix-2 stage count of the fine transform, log2(FSTAT_FINE_NUM_BINS).
 *
 * Set alongside FSTAT_FINE_WINDOWS_PER_STREAM; the relation is asserted below.
 * The transform reads twiddles from the shared master table, so a stage indexes
 * it as `t << (FX_MASTER_LOG2 - stage)`: the decimation stride folds into the
 * shift, and at FSTAT_FINE_NUM_BINS = 256 that selects exactly the entries the
 * frozen fxfft256 v1 table held.
 */
#define FSTAT_FINE_LOG2 8

/**
 * Reference-bin offset used to construct the three detector weight terms.
 *
 * The weight terms are:
 *   t = FSTAT_TARGET_WEIGHT_INDEX: target bin
 *   t = FSTAT_LOWER_REFERENCE_WEIGHT_INDEX: lower reference bin
 *   t = FSTAT_UPPER_REFERENCE_WEIGHT_INDEX: upper reference bin
 *
 * This is not the final mask guard-bin dilation parameter.
 */
#define FSTAT_REFERENCE_BIN_OFFSET 2

/* ===========================================================================
 * DATA TYPE CONFIGURATION
 * ===========================================================================*/

/**
 * Use uint64 integer accumulation for per-weight power sums.
 *
 * For the int4xint4 detector at any supported K, the worst-case full-block
 * power sum fits well below 64 bits.  The kernel stays fixed point through dot
 * product and power accumulation; floating point is introduced only when the
 * accumulated integer powers are divided for diagnostic float outputs.
 */
#define FSTAT_USE_UINT64_POWER_ACCUMULATION 1

/**
 * Use the DP4A dot-product path (1) or the scalar integer path (0).
 *
 * DP4A is the default production path for the detector. It keeps
 * packed samples in global memory, pre-packs the fixed weights into int8
 * dot-product lanes, and logically unpacks two packed complex samples at a time
 * in registers. The scalar path remains available via make DP4A=0 for direct
 * validation and debugging.
 */
#ifndef FSTAT_USE_DP4A
#define FSTAT_USE_DP4A 1
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
#ifndef FSTAT_USE_CONSTANT_WEIGHT_LANES
#define FSTAT_USE_CONSTANT_WEIGHT_LANES 0
#endif

/**
 * Preload DP4A weight lanes into per-block shared memory.
 *
 * The weight-lane table is tiny, read-only during a kernel launch, and reused
 * for every detector row handled by the block.  Shared-memory preload avoids
 * repeated per-row global loads while preserving per-handle weight storage and
 * avoiding the divergent-read penalty observed with constant memory.
 */
#ifndef FSTAT_USE_SHARED_WEIGHT_LANES
#define FSTAT_USE_SHARED_WEIGHT_LANES 1
#endif

#if FSTAT_USE_CONSTANT_WEIGHT_LANES && FSTAT_USE_SHARED_WEIGHT_LANES
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
#define FSTAT_SAMPLE_BITS_PER_COMPONENT (sizeof(InputType) * 4)

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
 * FSTAT_DOT_ACCUM_GUARD_BITS must equal log2(FSTAT_DETECTOR_WINDOW_SAMPLES).
 * It is derived here for each supported window length and asserted below.
 */
#define FSTAT_INT4_COMPONENT_BITS 4
#define FSTAT_COMPLEX_PRODUCT_COMPONENT_BITS \
    (FSTAT_INT4_COMPONENT_BITS + FSTAT_INT4_COMPONENT_BITS + 1)
#if FSTAT_DETECTOR_WINDOW_SAMPLES == 128
#define FSTAT_DOT_ACCUM_GUARD_BITS 7
#elif FSTAT_DETECTOR_WINDOW_SAMPLES == 64
#define FSTAT_DOT_ACCUM_GUARD_BITS 6
#else
#error "Unsupported FSTAT_DETECTOR_WINDOW_SAMPLES; add its guard-bit count."
#endif
#define FSTAT_COMPLEX_DOT_COMPONENT_BITS \
    (FSTAT_COMPLEX_PRODUCT_COMPONENT_BITS + FSTAT_DOT_ACCUM_GUARD_BITS)

/**
 * Worst-case magnitude bounds used by the static assertions below.
 *
 * Samples lie in [-8, 7] and quantized weight components in [-7, 7] (the
 * receiver's 15-level mid-tread quantizer emits only [-7, 7]); these bounds
 * conservatively admit the full two's-complement range on both, so one
 * complex-product component is bounded by 8*8 + 8*8 = 128 and a completed
 * K-tap dot-product component by 128*K.
 *
 * Note the conservatism is load-bearing at the boundary: the true weight
 * bound (7) gives 112*K and admits K = 256, while this bound gives 128*K
 * and rules it out by a single unit at the coarse |z|^2 assertion.
 */
#define FSTAT_MAX_SAMPLE_COMPONENT 8
#define FSTAT_MAX_PRODUCT_COMPONENT \
    (2 * FSTAT_MAX_SAMPLE_COMPONENT * FSTAT_MAX_SAMPLE_COMPONENT)
#define FSTAT_MAX_ROW_SUM_COMPONENT \
    ((long long)FSTAT_MAX_PRODUCT_COMPONENT * FSTAT_DETECTOR_WINDOW_SAMPLES)

/** DP4A consumes two packed complex samples per tap pair. */
#define FSTAT_DP4A_TAP_PAIRS (FSTAT_DETECTOR_WINDOW_SAMPLES / 2)

#ifdef __cplusplus
namespace fstat_geometry {

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
 */
constexpr double kButterflyGrowth = 1.0 + 1.41422 + (1.0 / 32768.0);
constexpr double growth_after(int stages) {
    return stages == 0 ? 1.0 : kButterflyGrowth * growth_after(stages - 1) + 1.0;
}

constexpr int kFineStages = log2_exact(FSTAT_FINE_NUM_BINS);
/** Post-transform bin magnitude, bounded by the triangle inequality. */
constexpr double kMaxBinMagnitude =
    static_cast<double>(FSTAT_MAX_ROW_SUM_COMPONENT) * FSTAT_FINE_WINDOWS_PER_STREAM;

} // namespace fstat_geometry

/* --- geometry consistency ------------------------------------------------ */
static_assert(
    fstat_geometry::is_pow2(FSTAT_DETECTOR_WINDOW_SAMPLES)
        && FSTAT_DETECTOR_WINDOW_SAMPLES >= 2,
    "detector_window_samples must be a power of two >= 2.");
static_assert(
    FSTAT_SAMPLE_BITS_PER_COMPONENT == FSTAT_INT4_COMPONENT_BITS,
    "Production detector uses packed complex int4 samples and weights.");
static_assert(
    FSTAT_DP4A_TAP_PAIRS * 2 == FSTAT_DETECTOR_WINDOW_SAMPLES,
    "DP4A tap pairs must be exactly detector_window_samples/2.");
static_assert(
    (1 << FSTAT_DOT_ACCUM_GUARD_BITS) == FSTAT_DETECTOR_WINDOW_SAMPLES,
    "FSTAT_DOT_ACCUM_GUARD_BITS must equal log2(detector_window_samples); "
    "update it whenever the window length changes.");
static_assert(
    fstat_geometry::is_pow2(FSTAT_FINE_NUM_BINS),
    "Fine transform length must be a power of two.");
static_assert(
    FSTAT_FINE_NUM_BINS == FSTAT_FINE_PAD_FACTOR * FSTAT_FINE_WINDOWS_PER_STREAM,
    "Fine bin count must be pad_factor * windows_per_stream.");
static_assert(
    (1 << FSTAT_FINE_LOG2) == FSTAT_FINE_NUM_BINS,
    "FSTAT_FINE_LOG2 must equal log2(FSTAT_FINE_NUM_BINS); update it whenever "
    "the fine transform length changes.");

/* --- no-overflow closure, derived per geometry ---------------------------- */
static_assert(
    FSTAT_MAX_ROW_SUM_COMPONENT <= 2147483647LL,
    "Completed dot-product components must fit the int32 row-sum storage.");
static_assert(
    2.0 * static_cast<double>(FSTAT_MAX_ROW_SUM_COMPONENT)
            * static_cast<double>(FSTAT_MAX_ROW_SUM_COMPONENT)
        <= 2147483647.0,
    "Coarse |z|^2 must fit int32; reduce detector_window_samples or widen the "
    "squared-magnitude accumulator.");
static_assert(
    static_cast<double>(FSTAT_MAX_ROW_SUM_COMPONENT)
            * fstat_geometry::growth_after(fstat_geometry::kFineStages)
        <= 2147483647.0,
    "fxfft working values must stay inside int32 across all butterfly stages; "
    "the admissible input norm falls as 2^31 / growth^stages, so this bound "
    "tightens every time the transform length doubles.");
static_assert(
    2.0 * fstat_geometry::kMaxBinMagnitude * fstat_geometry::kMaxBinMagnitude
            * 65536.0
        <= 18446744073709551615.0,
    "Feed-summed fine power must fit uint64 for stream counts up to 65536.");
#endif

/* ===========================================================================
 * NUMERICAL STABILITY
 * ===========================================================================*/

/**
 * Minimum denominator for F-statistic division.
 */
#define FSTAT_DIVISION_EPSILON FLT_MIN

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
#ifndef FSTAT_BLOCK_THREADS
#define FSTAT_BLOCK_THREADS 64
#endif

/** CUDA warp size. */
#define FSTAT_WARP_SIZE 32

/** Warps per CUDA block. */
#define FSTAT_WARPS_PER_BLOCK (FSTAT_BLOCK_THREADS / FSTAT_WARP_SIZE)

/** Full warp participation mask for shuffle operations. */
#define FSTAT_WARP_MASK 0xFFFFFFFF

/**
 * Maximum grid dimension.  With a 128-tap detector window and 64-thread
 * DP4A blocks, 4096 blocks gives one block per 64 detector rows for a
 * reference-sized 262144-row block.  This improves occupancy over the older
 * 1024-block cap while keeping the per-block atomic output cost small.
 */
#ifndef FSTAT_GRID_MAX_BLOCKS
#define FSTAT_GRID_MAX_BLOCKS 4096
#endif
