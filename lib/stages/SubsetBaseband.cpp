#include "SubsetBaseband.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "kotekanLogging.hpp"

#include "fmt.hpp"

#include <cstring>
#include <stdexcept>
#include <visUtil.hpp>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(SubsetBaseband);

// Element block size: copies always operate on 8-element (8-byte) blocks.
static constexpr uint32_t ELEMENT_BLOCK = 8;

STAGE_CONSTRUCTOR(SubsetBaseband) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Input dimensions
    timesamples_per_frame = config.get<uint32_t>(unique_name, "timesamples_per_frame");
    num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    num_elements = config.get<uint32_t>(unique_name, "num_elements");

    if (num_elements % ELEMENT_BLOCK != 0) {
        throw std::runtime_error(
            fmt::format(fmt("SubsetBaseband: num_elements ({:d}) must be a multiple of {:d}"),
                        num_elements, ELEMENT_BLOCK));
    }

    // Subset configuration (both lists are optional)
    freq_subset_configured = config.exists(unique_name, "coarse_freq_subset");
    if (freq_subset_configured) {
        coarse_freq_subset = config.get<std::vector<int>>(unique_name, "coarse_freq_subset");
        if (coarse_freq_subset.empty()) {
            throw std::runtime_error(
                "SubsetBaseband: coarse_freq_subset, if provided, must be non-empty");
        }
    }

    const uint32_t num_blocks_in = num_elements / ELEMENT_BLOCK;
    if (config.exists(unique_name, "element_blocks_subset")) {
        element_blocks_subset =
            config.get<std::vector<int>>(unique_name, "element_blocks_subset");
        if (element_blocks_subset.empty()) {
            throw std::runtime_error(
                "SubsetBaseband: element_blocks_subset, if provided, must be non-empty");
        }
        for (int b : element_blocks_subset) {
            if (b < 0 || (uint32_t)b >= num_blocks_in) {
                throw std::runtime_error(fmt::format(
                    fmt("SubsetBaseband: element_blocks_subset entry {:d} out of range "
                        "[0, {:d})"),
                    b, num_blocks_in));
            }
        }
    } else {
        // Default: pass through all element blocks in input order.
        element_blocks_subset.resize(num_blocks_in);
        for (uint32_t i = 0; i < num_blocks_in; i++) {
            element_blocks_subset[i] = (int)i;
        }
    }

    // Validate input buffer size: T * F * E bytes
    const size_t expected_in_size =
        (size_t)timesamples_per_frame * num_local_freq * num_elements;
    if (in_buf->frame_size != expected_in_size) {
        throw std::runtime_error(fmt::format(
            fmt("SubsetBaseband: in_buf frame size ({:d}) does not match expected "
                "size ({:d}) for shape [T={:d}][F={:d}][E={:d}]"),
            in_buf->frame_size, expected_in_size, timesamples_per_frame, num_local_freq,
            num_elements));
    }

    num_elements_out = (uint32_t)element_blocks_subset.size() * ELEMENT_BLOCK;
    if (num_elements_out % 2 != 0) {
        throw std::runtime_error(fmt::format(
            fmt("SubsetBaseband: num_elements_out ({:d}) must be even"), num_elements_out));
    }

    // Pre-compute per-output-element source byte offsets (block_index * 8).
    element_byte_offsets_.resize(element_blocks_subset.size());
    for (size_t i = 0; i < element_blocks_subset.size(); i++) {
        element_byte_offsets_[i] = element_blocks_subset[i] * (int32_t)ELEMENT_BLOCK;
    }

    initialized_ = false;

    INFO("SubsetBaseband: Input shape  [T={:d}][F={:d}][E={:d}]", timesamples_per_frame,
         num_local_freq, num_elements);
    INFO("SubsetBaseband: Output shape [T={:d}][F_sub=<from intersection>][E_sub={:d}]",
         timesamples_per_frame, num_elements_out);
}

SubsetBaseband::~SubsetBaseband() {}

void SubsetBaseband::main_thread() {
    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    // Strides (bytes) for the input array [T][F][E]
    const size_t in_freq_stride = (size_t)num_elements;
    const size_t in_time_stride = (size_t)num_local_freq * in_freq_stride;

    // Output frequency stride (bytes). Time stride is set after the freq map is built.
    const size_t out_freq_stride = (size_t)num_elements_out;
    size_t out_time_stride = 0;

    const size_t n_blocks_out = element_blocks_subset.size();

    while (!stop_thread) {
        uint8_t* in_frame = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (in_frame == nullptr)
            break;

        uint8_t* out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame == nullptr)
            break;

        auto in_meta = get_chord_metadata(in_buf, in_frame_id);
        std::vector<int> in_coarse_freq = in_meta->get_coarse_freq();

        if (!initialized_) {
            // Build the per-frame freq mapping. If a subset was configured, use the
            // intersection (in coarse_freq_subset order); otherwise pass through all
            // input frequencies in input order.
            if (freq_subset_configured) {
                for (int target : coarse_freq_subset) {
                    for (size_t j = 0; j < in_coarse_freq.size(); j++) {
                        if (in_coarse_freq[j] == target) {
                            out_coarse_freq_.push_back(target);
                            freq_in_index_.push_back((int)j);
                            break;
                        }
                    }
                }
            } else {
                out_coarse_freq_ = in_coarse_freq;
                freq_in_index_.resize(in_coarse_freq.size());
                for (size_t j = 0; j < in_coarse_freq.size(); j++) {
                    freq_in_index_[j] = (int)j;
                }
            }
            cached_in_coarse_freq_ = in_coarse_freq;

            const uint32_t num_freq_out = (uint32_t)out_coarse_freq_.size();

            // Validate output buffer size against the actual intersection.
            const size_t expected_out_size =
                (size_t)timesamples_per_frame * num_freq_out * num_elements_out;
            if (out_buf->frame_size != expected_out_size) {
                throw std::runtime_error(fmt::format(
                    fmt("SubsetBaseband: out_buf frame size ({:d}) does not match expected "
                        "intersection size ({:d}) for shape "
                        "[T={:d}][F_sub={:d}][E_sub={:d}] (intersection of "
                        "coarse_freq_subset and input metadata yielded {:d} frequencies)"),
                    out_buf->frame_size, expected_out_size, timesamples_per_frame, num_freq_out,
                    num_elements_out, num_freq_out));
            }

            out_buf->allocate_ndarray_frame_desc(
                kotekan::int4x2_swapped_withoffset, "E",
                {timesamples_per_frame, num_freq_out, 2, num_elements_out / 2},
                {"T", "F", "P", "D"});

            out_time_stride = (size_t)num_freq_out * out_freq_stride;
            initialized_ = true;

            INFO("SubsetBaseband: initialized with F_sub = {:d}{}", num_freq_out,
                 freq_subset_configured
                     ? fmt::format(" (out of {:d} requested)", coarse_freq_subset.size())
                     : std::string(" (pass-through, no coarse_freq_subset configured)"));
        } else {
            // Verify the input freq list hasn't changed.
            if (in_coarse_freq != cached_in_coarse_freq_) {
                throw std::runtime_error(
                    "SubsetBaseband: input frame coarse_freq list changed between frames; "
                    "this stage assumes a fixed input frequency list.");
            }
        }

        const uint32_t num_freq_out = (uint32_t)out_coarse_freq_.size();

        // Per-output-frequency upchan factor lookup.
        std::vector<int> in_upchan;
        if (in_meta->has_freq_upchan_factor()) {
            in_upchan = in_meta->get_freq_upchan_factor();
        }
        std::vector<int> out_upchan(num_freq_out, 1);
        for (uint32_t i = 0; i < num_freq_out; i++) {
            const int idx = freq_in_index_[i];
            if ((size_t)idx < in_upchan.size()) {
                out_upchan[i] = in_upchan[idx];
            }
        }

        // Copy data: for each (t, f_out), copy the configured 8-byte element blocks.
        for (uint32_t t = 0; t < timesamples_per_frame; t++) {
            for (uint32_t f_out = 0; f_out < num_freq_out; f_out++) {
                uint8_t* out_row =
                    out_frame + t * out_time_stride + f_out * out_freq_stride;
                const int f_in = freq_in_index_[f_out];
                const uint8_t* in_row =
                    in_frame + t * in_time_stride + (size_t)f_in * in_freq_stride;

                for (size_t i = 0; i < n_blocks_out; i++) {
                    std::memcpy(out_row + i * ELEMENT_BLOCK,
                                in_row + element_byte_offsets_[i], ELEMENT_BLOCK);
                }
            }
        }

        // Build output metadata.
        out_buf->allocate_new_metadata_object(out_frame_id);
        auto out_meta = get_chord_metadata(out_buf, out_frame_id);

        out_meta->type = kotekan::int4x2_swapped_withoffset;
        out_meta->dims = 4;
        out_meta->set_name("E");
        out_meta->set_array_dimension(0, timesamples_per_frame, "T");
        out_meta->set_array_dimension(1, num_freq_out, "F");
        out_meta->set_array_dimension(2, 2, "P");
        out_meta->set_array_dimension(3, num_elements_out / 2, "D");
        out_meta->set_strides_simple();

        out_meta->set_coarse_freq(out_coarse_freq_);
        out_meta->set_freq_upchan_factor(out_upchan);

        out_meta->set_fpga_seq_num(in_meta->get_fpga_seq_num());
        out_meta->set_time_downsampling_fpga(in_meta->get_time_downsampling_fpga());

        DEBUG("SubsetBaseband: produced subset frame {:d} -> {:d}", (int)in_frame_id,
              (int)out_frame_id);

        in_buf->mark_frame_empty(unique_name, in_frame_id++);
        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }
}
