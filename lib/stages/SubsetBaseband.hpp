#ifndef SUBSET_BASEBAND_HPP
#define SUBSET_BASEBAND_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <array>
#include <cstdint>
#include <stddef.h>
#include <string>
#include <vector>

/**
 * @class SubsetBaseband
 * @brief Extracts a subset of frequencies and dishes from a baseband electric field array.
 *
 * Takes electric field data in the time-major layout `E[time][frequency_local][P][D]`, one
 * byte per (polarization, dish), and produces an output buffer containing the intersection
 * of a configured set of frequencies with the input frame's frequencies, plus a configured
 * subset of dish blocks (each block is 8 contiguous dishes / 8 bytes).
 *
 * The polarization axis is the outer of the two element axes and is always 2 wide, so the
 * byte for (p, d) sits at element offset `p * num_dishes + d`. Both polarizations of every
 * selected dish are copied; the polarization axis is never subset, so the output P extent is
 * also always 2.
 *
 * The output contains the intersection of `coarse_freq_subset` and the input frame's
 * coarse-freq list, in `coarse_freq_subset` order. The input frame's coarse-freq list is
 * assumed to be the same for every frame; an error is raised if it changes.
 *
 * The output buffer must be sized for exactly the intersection size (validated on the
 * first frame).
 *
 * The selected dish indices are published in the output metadata as `dish_index` (the
 * position of each output dish in the telescope's `dish_inputs` table), so the file writers
 * can record which dishes the subset actually holds.
 *
 * The dish subset is specified as a list of 8-dish block indices into the input's dish axis.
 * Each block index `b` selects dishes `[b*8, b*8+8)` for both polarizations, i.e. input bytes
 * `[p*num_dishes + b*8, p*num_dishes + b*8 + 8)` for `p` in `{0, 1}`.
 *
 * Input format:  `E[time][frequency_local][P][D]`         (uint8, P = 2)
 * Output format: `E'[time][frequency_subset][P][D_subset]` (uint8, P = 2)
 *
 * @par Copy strategy
 * The per-(time, frequency) copy is planned once at startup rather than per frame. Runs of
 * dish blocks that are contiguous in both input and output are coalesced, so a request for
 * adjacent blocks becomes one wide copy per polarization instead of one copy per block.
 * Where the output row is a whole number of 64-byte chunks, an AVX512 path assembles each
 * chunk in a register -- a plain load when the chunk's source is contiguous, two 32-byte
 * loads when it is two halves, an 8-lane gather otherwise -- and writes it with a
 * non-temporal store, keeping the (large) output frame out of the cache. Otherwise a
 * portable `memcpy` path over the coalesced runs is used.
 *
 * @par Buffers
 * @buffer in_buf  Input electric-field buffer.
 *     @buffer_format uint8 E[time][frequency_local][P][D]
 *     @buffer_metadata chordMetadata
 * @buffer out_buf Output subset electric-field buffer.
 *     @buffer_format uint8 E[time][frequency_subset][P][D_subset]
 *     @buffer_metadata chordMetadata
 *
 * @conf timesamples_per_frame Int.   Number of time samples per frame.
 * @conf num_local_freq        Int.   Number of input frequency channels.
 * @conf num_elements          Int.   Number of input elements, i.e. 2 * number of input
 *                                    dishes. Must be even, and the implied dish count must
 *                                    be a multiple of 8.
 * @conf coarse_freq_subset    [Int]. Optional. Ordered list of coarse-frequency IDs to
 *                                    extract. If omitted, all input frequencies are
 *                                    passed through in input order.
 * @conf dish_blocks_subset    [Int]. Optional. Ordered list of 8-dish block indices to
 *                                    extract. Both polarizations of each selected dish are
 *                                    copied. If omitted, all dish blocks are passed through
 *                                    in input order.
 *
 */
class SubsetBaseband : public kotekan::Stage {
public:
    SubsetBaseband(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~SubsetBaseband() override;
    void main_thread() override;

private:
    /// Bytes per dish block on both the input and output dish axes.
    static constexpr uint32_t DISH_BLOCK = 8;

    /// Size of the polarization axis, on both input and output. Never subset.
    static constexpr uint32_t NUM_POL = 2;

    /// Bytes per non-temporal store, and hence per AVX512 output chunk.
    static constexpr uint32_t CHUNK_BYTES = 64;

    /// One maximal contiguous copy within a single (time, frequency) row: `len` bytes from
    /// input element offset `src_off` to output element offset `dst_off`.
    struct CopyRun {
        uint32_t src_off;
        uint32_t dst_off;
        uint32_t len;
    };

    /// How one 64-byte output chunk is assembled from the input row.
    enum class ChunkKind : uint8_t {
        Contiguous, ///< All 64 bytes are contiguous in the input.
        TwoHalves,  ///< Two contiguous 32-byte halves.
        Gather,     ///< Eight 8-byte dish blocks from arbitrary offsets.
    };

    /// Plan for one 64-byte output chunk. Only the fields its `kind` needs are set.
    struct ChunkPlan {
        ChunkKind kind;
        /// Contiguous: source offset of the whole chunk. TwoHalves: offset of the low half.
        uint32_t src_off;
        /// TwoHalves: source offset of the high 32 bytes.
        uint32_t src_off_hi;
        /// Gather: source offset of each of the eight 8-byte blocks.
        std::array<int64_t, DISH_BLOCK> gather_offsets;
    };

    /// Input buffer (E[T][F][P][D])
    Buffer* in_buf;

    /// Output buffer (E[T][F_sub][P][D_sub])
    Buffer* out_buf;

    /// Configuration
    uint32_t timesamples_per_frame;
    uint32_t num_local_freq;
    uint32_t num_elements;

    /// Number of input dishes (derived: num_elements / NUM_POL).
    uint32_t num_dishes_in;

    /// Requested output coarse-frequency IDs (in output order).
    /// If the config key is absent, this is left empty and all input frequencies are
    /// passed through (in input order).
    std::vector<int> coarse_freq_subset;

    /// True if `coarse_freq_subset` was supplied by config (filter), false to pass
    /// through all input frequencies.
    bool freq_subset_configured;

    /// Requested output dish-block indices (in output order); each block is 8 dishes.
    /// If the config key is absent, this defaults to all dish blocks (0..num_dishes_in/8).
    std::vector<int> dish_blocks_subset;

    /// Number of output dishes (derived: dish_blocks_subset.size() * DISH_BLOCK).
    uint32_t num_dishes_out;

    /// Number of output element bytes per (time, frequency) row (derived:
    /// NUM_POL * num_dishes_out).
    uint32_t num_elements_out;

    /// Output coarse-freq list (intersection, in `coarse_freq_subset` order).
    /// Built from the first frame; subsequent frames must match.
    std::vector<int> out_coarse_freq_;

    /// Mapping: input freq index for each output freq in `out_coarse_freq_`.
    std::vector<int> freq_in_index_;

    /// Cached input coarse-freq list. Built from the first frame; subsequent frames must match.
    std::vector<int> cached_in_coarse_freq_;

    /// Index into the telescope's `dish_inputs` table of each output dish, in output dish
    /// order. Written to the output metadata so downstream consumers know which dishes the
    /// data describes.
    std::vector<int> dish_index_out_;

    /// Source element offset of each 8-byte block of the output row, in output order.
    /// Indexed by output block, i.e. `p * dish_blocks_subset.size() + block`.
    std::vector<uint32_t> src_off_for_out_block_;

    /// `src_off_for_out_block_` coalesced into maximal contiguous copies. Used by the
    /// portable path, and by the AVX512 path's chunk classification.
    std::vector<CopyRun> copy_runs_;

    /// Per-64-byte-output-chunk assembly plan. Empty unless the AVX512 path is usable.
    std::vector<ChunkPlan> chunk_plans_;

    /// True if the output row is a whole number of 64-byte chunks and this build has
    /// AVX512, so `chunk_plans_` is populated and the fast path may be used. Frames whose
    /// base address is not 64-byte aligned still fall back to the portable path.
    bool use_avx512_fast_path;

    /// True after the first frame has initialized the freq map and validated buffer sizes.
    bool initialized_;

    /// Build `src_off_for_out_block_`, `copy_runs_` and (when usable) `chunk_plans_`.
    void plan_copy();

    /// Portable copy of one frame's worth of (time, frequency) rows.
    void copy_frame_scalar(const uint8_t* in_frame, uint8_t* out_frame, uint32_t num_freq_out,
                           size_t in_time_stride, size_t out_time_stride) const;

#ifdef __AVX512F__
    /// AVX512 + non-temporal-store copy of one frame's worth of (time, frequency) rows.
    /// Requires `out_frame` to be 64-byte aligned and `chunk_plans_` to be populated.
    void copy_frame_avx512(const uint8_t* in_frame, uint8_t* out_frame, uint32_t num_freq_out,
                           size_t in_time_stride, size_t out_time_stride) const;
#endif
};

#endif // SUBSET_BASEBAND_HPP
