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

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h> // for _mm_sfence
#endif

#ifdef __AVX512F__
#include <immintrin.h> // for __m512i, _mm512_stream_si512, _mm512_i64gather_epi64
#endif

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(SubsetBaseband);

STAGE_CONSTRUCTOR(SubsetBaseband) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Input dimensions
    timesamples_per_frame = config.get<uint32_t>(unique_name, "timesamples_per_frame");
    num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    num_elements = config.get<uint32_t>(unique_name, "num_elements");

    // The element axis is [P][D] with P outer and always NUM_POL wide.
    if (num_elements % NUM_POL != 0) {
        throw std::runtime_error(fmt::format(
            fmt("SubsetBaseband: num_elements ({:d}) must be a multiple of the polarization "
                "count ({:d}); the element axis is [P][D]"),
            num_elements, NUM_POL));
    }
    num_dishes_in = num_elements / NUM_POL;

    if (num_dishes_in % DISH_BLOCK != 0) {
        throw std::runtime_error(fmt::format(
            fmt("SubsetBaseband: number of input dishes ({:d}, from num_elements {:d} / P {:d}) "
                "must be a multiple of {:d}"),
            num_dishes_in, num_elements, NUM_POL, DISH_BLOCK));
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

    // `element_blocks_subset` selected 8-element blocks of the flat element axis, which is
    // not the same set of bytes as any dish block. Fail loudly rather than silently
    // defaulting to a full pass-through.
    if (config.exists(unique_name, "element_blocks_subset")) {
        throw std::runtime_error(
            "SubsetBaseband: element_blocks_subset is no longer supported; use "
            "dish_blocks_subset, which selects 8-dish blocks and copies both polarizations "
            "of each selected dish");
    }

    const uint32_t num_blocks_in = num_dishes_in / DISH_BLOCK;

    // Resolved by value, not by presence: `Config::exists` checks only this stage's own
    // block, while `get_default` walks up the config tree. `dish_blocks_subset` is shared
    // with the DPDK capture workers and hdf5FileWrite (which needs it to put the FPGA gains
    // into dish order), so it normally sits at global scope rather than on this stage. An
    // empty list therefore means "all blocks", exactly as omitting the key does.
    dish_blocks_subset =
        config.get_default<std::vector<int>>(unique_name, "dish_blocks_subset", {});
    if (!dish_blocks_subset.empty()) {
        for (int b : dish_blocks_subset) {
            if (b < 0 || (uint32_t)b >= num_blocks_in) {
                throw std::runtime_error(
                    fmt::format(fmt("SubsetBaseband: dish_blocks_subset entry {:d} out of range "
                                    "[0, {:d})"),
                                b, num_blocks_in));
            }
        }
    } else {
        // Default: pass through all dish blocks in input order.
        dish_blocks_subset.resize(num_blocks_in);
        for (uint32_t i = 0; i < num_blocks_in; i++) {
            dish_blocks_subset[i] = (int)i;
        }
    }

    // Validate input buffer size: T * F * P * D bytes
    const size_t expected_in_size = (size_t)timesamples_per_frame * num_local_freq * num_elements;
    if (in_buf->frame_size != expected_in_size) {
        throw std::runtime_error(
            fmt::format(fmt("SubsetBaseband: in_buf frame size ({:d}) does not match expected "
                            "size ({:d}) for shape [T={:d}][F={:d}][P={:d}][D={:d}]"),
                        in_buf->frame_size, expected_in_size, timesamples_per_frame, num_local_freq,
                        NUM_POL, num_dishes_in));
    }

    num_dishes_out = (uint32_t)dish_blocks_subset.size() * DISH_BLOCK;
    num_elements_out = NUM_POL * num_dishes_out;

    plan_copy();

    initialized_ = false;

    INFO("SubsetBaseband: Input shape  [T={:d}][F={:d}][P={:d}][D={:d}]", timesamples_per_frame,
         num_local_freq, NUM_POL, num_dishes_in);
    INFO("SubsetBaseband: Output shape [T={:d}][F_sub=<from intersection>][P={:d}][D_sub={:d}]",
         timesamples_per_frame, NUM_POL, num_dishes_out);
    INFO("SubsetBaseband: {:d} dish block(s) coalesced into {:d} copy run(s) per (time, freq) "
         "row of {:d} bytes",
         dish_blocks_subset.size(), copy_runs_.size(), num_elements_out);
    if (use_avx512_fast_path) {
        INFO("SubsetBaseband: AVX512 fast path enabled ({:d} chunk(s) of {:d} bytes per row, "
             "non-temporal stores)",
             chunk_plans_.size(), CHUNK_BYTES);
    } else {
#ifdef __AVX512F__
        INFO("SubsetBaseband: output row of {:d} bytes is not a whole number of {:d}-byte "
             "chunks, using the portable copy path",
             num_elements_out, CHUNK_BYTES);
#else
        INFO("SubsetBaseband: AVX512 not available, using the portable copy path");
#endif
    }
}

SubsetBaseband::~SubsetBaseband() {}

void SubsetBaseband::plan_copy() {
    const size_t n_blocks_out = dish_blocks_subset.size();

    // Source element offset of every 8-byte block of the output row. The output row is
    // [P][D_sub], so output block `p * n_blocks_out + i` holds dish block
    // `dish_blocks_subset[i]` of polarization `p`.
    src_off_for_out_block_.resize(NUM_POL * n_blocks_out);
    for (uint32_t p = 0; p < NUM_POL; p++) {
        for (size_t i = 0; i < n_blocks_out; i++) {
            src_off_for_out_block_[p * n_blocks_out + i] =
                p * num_dishes_in + (uint32_t)dish_blocks_subset[i] * DISH_BLOCK;
        }
    }

    // Which input dish each output dish position holds, in output dish order. The dish axis
    // is shared by both polarizations, so this is one entry per output dish, not per element.
    dish_index_out_.resize(num_dishes_out);
    for (size_t i = 0; i < n_blocks_out; i++) {
        for (uint32_t k = 0; k < DISH_BLOCK; k++) {
            dish_index_out_[i * DISH_BLOCK + k] = dish_blocks_subset[i] * (int)DISH_BLOCK + (int)k;
        }
    }

    // Coalesce into maximal runs that are contiguous in both source and destination. The
    // destination is walked in order, so a run extends whenever the source does too. For a
    // request of adjacent blocks this collapses to one run per polarization, and for a full
    // pass-through the two polarizations merge into a single run.
    copy_runs_.clear();
    for (size_t blk = 0; blk < src_off_for_out_block_.size(); blk++) {
        const uint32_t src_off = src_off_for_out_block_[blk];
        const uint32_t dst_off = (uint32_t)blk * DISH_BLOCK;
        if (!copy_runs_.empty()) {
            CopyRun& prev = copy_runs_.back();
            if (prev.src_off + prev.len == src_off && prev.dst_off + prev.len == dst_off) {
                prev.len += DISH_BLOCK;
                continue;
            }
        }
        copy_runs_.push_back({src_off, dst_off, DISH_BLOCK});
    }

    // The AVX512 path writes whole 64-byte chunks, so it needs the output row to be a whole
    // number of them. num_elements_out is 16 * n_blocks_out, so this holds whenever the
    // number of requested blocks is a multiple of 4.
    chunk_plans_.clear();
    use_avx512_fast_path = false;
#ifdef __AVX512F__
    if (num_elements_out % CHUNK_BYTES == 0) {
        const uint32_t blocks_per_chunk = CHUNK_BYTES / DISH_BLOCK;
        const uint32_t n_chunks = num_elements_out / CHUNK_BYTES;

        for (uint32_t c = 0; c < n_chunks; c++) {
            ChunkPlan plan{};
            for (uint32_t j = 0; j < blocks_per_chunk; j++) {
                plan.gather_offsets[j] = (int64_t)src_off_for_out_block_[c * blocks_per_chunk + j];
            }

            // Classify so the common shapes avoid a gather: a fully contiguous chunk is one
            // load, and two contiguous 32-byte halves are two loads and an insert.
            const int64_t base = plan.gather_offsets[0];
            bool contiguous = true;
            for (uint32_t j = 1; j < blocks_per_chunk; j++) {
                if (plan.gather_offsets[j] != base + (int64_t)j * DISH_BLOCK) {
                    contiguous = false;
                    break;
                }
            }

            const uint32_t half = blocks_per_chunk / 2;
            bool two_halves = true;
            for (uint32_t j = 1; j < half; j++) {
                if (plan.gather_offsets[j] != base + (int64_t)j * DISH_BLOCK) {
                    two_halves = false;
                    break;
                }
            }
            if (two_halves) {
                const int64_t base_hi = plan.gather_offsets[half];
                for (uint32_t j = half + 1; j < blocks_per_chunk; j++) {
                    if (plan.gather_offsets[j] != base_hi + (int64_t)(j - half) * DISH_BLOCK) {
                        two_halves = false;
                        break;
                    }
                }
            }

            if (contiguous) {
                plan.kind = ChunkKind::Contiguous;
                plan.src_off = (uint32_t)base;
            } else if (two_halves) {
                plan.kind = ChunkKind::TwoHalves;
                plan.src_off = (uint32_t)base;
                plan.src_off_hi = (uint32_t)plan.gather_offsets[half];
            } else {
                plan.kind = ChunkKind::Gather;
            }
            chunk_plans_.push_back(plan);
        }
        use_avx512_fast_path = true;
    }
#endif
}

void SubsetBaseband::copy_frame_scalar(const uint8_t* in_frame, uint8_t* out_frame,
                                       uint32_t num_freq_out, size_t in_time_stride,
                                       size_t out_time_stride) const {
    const size_t in_freq_stride = num_elements;
    const size_t out_freq_stride = num_elements_out;

    for (uint32_t t = 0; t < timesamples_per_frame; t++) {
        for (uint32_t f_out = 0; f_out < num_freq_out; f_out++) {
            uint8_t* out_row = out_frame + t * out_time_stride + f_out * out_freq_stride;
            const uint8_t* in_row =
                in_frame + t * in_time_stride + (size_t)freq_in_index_[f_out] * in_freq_stride;

            for (const CopyRun& run : copy_runs_) {
                std::memcpy(out_row + run.dst_off, in_row + run.src_off, run.len);
            }
        }
    }
}

#ifdef __AVX512F__
void SubsetBaseband::copy_frame_avx512(const uint8_t* in_frame, uint8_t* out_frame,
                                       uint32_t num_freq_out, size_t in_time_stride,
                                       size_t out_time_stride) const {
    const size_t in_freq_stride = num_elements;
    const size_t out_freq_stride = num_elements_out;
    const size_t n_chunks = chunk_plans_.size();

    for (uint32_t t = 0; t < timesamples_per_frame; t++) {
        for (uint32_t f_out = 0; f_out < num_freq_out; f_out++) {
            uint8_t* out_row = out_frame + t * out_time_stride + f_out * out_freq_stride;
            const uint8_t* in_row =
                in_frame + t * in_time_stride + (size_t)freq_in_index_[f_out] * in_freq_stride;

            for (size_t c = 0; c < n_chunks; c++) {
                const ChunkPlan& plan = chunk_plans_[c];
                __m512i data;

                switch (plan.kind) {
                    case ChunkKind::Contiguous:
                        data = _mm512_loadu_si512((const void*)(in_row + plan.src_off));
                        break;
                    case ChunkKind::TwoHalves: {
                        const __m256i lo =
                            _mm256_loadu_si256((const __m256i*)(in_row + plan.src_off));
                        const __m256i hi =
                            _mm256_loadu_si256((const __m256i*)(in_row + plan.src_off_hi));
                        data = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                        break;
                    }
                    case ChunkKind::Gather:
                    default: {
                        const __m512i idx =
                            _mm512_loadu_si512((const void*)plan.gather_offsets.data());
                        data = _mm512_i64gather_epi64(idx, (const long long*)in_row, 1);
                        break;
                    }
                }

                // The output frame is far larger than the cache and is not read back here,
                // so keep it out of the cache entirely.
                _mm512_stream_si512((__m512i*)(out_row + c * CHUNK_BYTES), data);
            }
        }
    }

    // Make the non-temporal stores visible before the frame is marked full.
    _mm_sfence();
}
#endif

void SubsetBaseband::main_thread() {
    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    // Strides (bytes) for the input array [T][F][P][D]
    const size_t in_freq_stride = (size_t)num_elements;
    const size_t in_time_stride = (size_t)num_local_freq * in_freq_stride;

    // Output frequency stride (bytes). Time stride is set after the freq map is built.
    const size_t out_freq_stride = (size_t)num_elements_out;
    size_t out_time_stride = 0;

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
                        "[T={:d}][F_sub={:d}][P={:d}][D_sub={:d}] (intersection of "
                        "coarse_freq_subset and input metadata yielded {:d} frequencies)"),
                    out_buf->frame_size, expected_out_size, timesamples_per_frame, num_freq_out,
                    NUM_POL, num_dishes_out, num_freq_out));
            }

            // The output shape depends on the freq intersection, which is only known
            // once the first input frame arrives, so this stage originates the
            // descriptor rather than requiring one from the config.
            out_buf->ensure_frame_desc(kotekan::GenericNDArray::describe(
                kotekan::int4x2_swapped_withoffset, "E",
                {timesamples_per_frame, num_freq_out, NUM_POL, num_dishes_out},
                {"T", "F", "P", "D"}, {1, 1, 1, 1}));

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

        // Copy data: for each (t, f_out), gather the configured dish blocks of both
        // polarizations into the output row. Non-temporal stores need a 64-byte aligned
        // destination; frames are page aligned in practice, but fall back if one is not.
        bool used_fast_path = false;
#ifdef __AVX512F__
        if (use_avx512_fast_path && ((uintptr_t)out_frame % CHUNK_BYTES) == 0) {
            copy_frame_avx512(in_frame, out_frame, num_freq_out, in_time_stride, out_time_stride);
            used_fast_path = true;
        }
#endif
        if (!used_fast_path) {
            copy_frame_scalar(in_frame, out_frame, num_freq_out, in_time_stride, out_time_stride);
        }

        // Build output metadata.
        out_buf->allocate_new_metadata_object(out_frame_id);
        auto out_meta = get_chord_metadata(out_buf, out_frame_id);

        out_meta->type = kotekan::int4x2_swapped_withoffset;
        out_meta->dims = 4;
        out_meta->set_name("E");
        out_meta->set_array_dimension(0, timesamples_per_frame, "T", 1);
        out_meta->set_array_dimension(1, num_freq_out, "F", 1);
        out_meta->set_array_dimension(2, NUM_POL, "P", 1);
        out_meta->set_array_dimension(3, num_dishes_out, "D", 1);
        out_meta->set_strides_simple();

        out_meta->set_coarse_freq(out_coarse_freq_);
        out_meta->set_freq_upchan_factor(out_upchan);
        out_meta->set_dish_index(dish_index_out_);

        out_meta->set_fpga_seq_num(in_meta->get_fpga_seq_num());
        out_meta->set_time_downsampling_fpga(in_meta->get_time_downsampling_fpga());

        DEBUG("SubsetBaseband: produced subset frame {:d} -> {:d}", (int)in_frame_id,
              (int)out_frame_id);

        in_buf->mark_frame_empty(unique_name, in_frame_id++);
        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }
}
