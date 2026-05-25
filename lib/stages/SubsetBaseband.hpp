#ifndef SUBSET_BASEBAND_HPP
#define SUBSET_BASEBAND_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <cstdint>
#include <string>
#include <vector>

/**
 * @class SubsetBaseband
 * @brief Extracts a subset of frequencies and elements from a baseband electric field array.
 *
 * Takes electric field data in the time-major layout
 * (`E[time][frequency_local][element]`, one byte per element) and produces an output buffer
 * containing the intersection of a configured set of frequencies with the input frame's
 * frequencies, plus a configured subset of element blocks (each block is 8 contiguous
 * elements / 8 bytes).
 *
 * The output contains the intersection of `coarse_freq_subset` and the input frame's
 * coarse-freq list, in `coarse_freq_subset` order. The input frame's coarse-freq list is
 * assumed to be the same for every frame; an error is raised if it changes.
 *
 * The output buffer must be sized for exactly the intersection size (validated on the
 * first frame).
 *
 * The element subset is specified as a list of 8-element block indices into the input's
 * element axis. Each block index `b` selects bytes `[b*8, b*8+8)` of the input element axis.
 *
 * Input format:  `E[time][frequency_local][element]`           (uint8)
 * Output format: `E'[time][frequency_subset][element_subset]`  (uint8)
 *
 * @par Buffers
 * @buffer in_buf  Input electric-field buffer.
 *     @buffer_format uint8 E[time][frequency_local][element]
 *     @buffer_metadata chordMetadata
 * @buffer out_buf Output subset electric-field buffer.
 *     @buffer_format uint8 E[time][frequency_subset][element_subset]
 *     @buffer_metadata chordMetadata
 *
 * @conf timesamples_per_frame Int.   Number of time samples per frame.
 * @conf num_local_freq        Int.   Number of input frequency channels.
 * @conf num_elements          Int.   Number of input elements (must be a multiple of 8).
 * @conf coarse_freq_subset    [Int]. Optional. Ordered list of coarse-frequency IDs to
 *                                    extract. If omitted, all input frequencies are
 *                                    passed through in input order.
 * @conf element_blocks_subset [Int]. Optional. Ordered list of 8-element block indices to
 *                                    extract. If omitted, all element blocks are passed
 *                                    through in input order.
 *
 */
class SubsetBaseband : public kotekan::Stage {
public:
    SubsetBaseband(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~SubsetBaseband() override;
    void main_thread() override;

private:
    /// Input buffer (E[T][F][E])
    Buffer* in_buf;

    /// Output buffer (E[T][F_sub][E_sub])
    Buffer* out_buf;

    /// Configuration
    uint32_t timesamples_per_frame;
    uint32_t num_local_freq;
    uint32_t num_elements;

    /// Requested output coarse-frequency IDs (in output order).
    /// If the config key is absent, this is left empty and all input frequencies are
    /// passed through (in input order).
    std::vector<int> coarse_freq_subset;

    /// True if `coarse_freq_subset` was supplied by config (filter), false to pass
    /// through all input frequencies.
    bool freq_subset_configured;

    /// Requested output element-block indices (in output order); each block is 8 elements.
    /// If the config key is absent, this defaults to all element blocks (0..num_elements/8).
    std::vector<int> element_blocks_subset;

    /// Number of output element bytes (derived).
    uint32_t num_elements_out;

    /// Output coarse-freq list (intersection, in `coarse_freq_subset` order).
    /// Built from the first frame; subsequent frames must match.
    std::vector<int> out_coarse_freq_;

    /// Mapping: input freq index for each output freq in `out_coarse_freq_`.
    std::vector<int> freq_in_index_;

    /// Cached input coarse-freq list. Built from the first frame; subsequent frames must match.
    std::vector<int> cached_in_coarse_freq_;

    /// Per-output-element byte offsets into the input element row (block_index * 8).
    std::vector<int32_t> element_byte_offsets_;

    /// True after the first frame has initialized the freq map and validated buffer sizes.
    bool initialized_;
};

#endif // SUBSET_BASEBAND_HPP
