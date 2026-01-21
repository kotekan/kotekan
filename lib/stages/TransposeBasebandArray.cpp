#include "TransposeBasebandArray.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "kotekanLogging.hpp"
#include "chordMetadata.hpp"

#include "fmt.hpp"

#include <cstring>
#include <stdexcept>
#include <visUtil.hpp>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(TransposeBasebandArray);

// Compile-time constants for optimized code paths
static constexpr uint32_t NUM_LOCAL_FREQ = 384;
static constexpr uint32_t NUM_ELEMENTS = 128;
static constexpr uint32_t TIME_SHORT = 16;
static constexpr uint32_t ELEMENT_SHORT = 8;
static constexpr uint32_t ELEMENT_LONG = NUM_ELEMENTS / ELEMENT_SHORT;  // 16

STAGE_CONSTRUCTOR(TransposeBasebandArray) {
    // Register buffers
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    pl_mask_buf = get_buffer("pl_mask_buf");
    pl_mask_buf->register_consumer(unique_name);

    // Get configuration parameters
    timesamples_per_frame = config.get<uint32_t>(unique_name, "timesamples_per_frame");
    // TODO Remove this option.
    process_even = config.get<bool>(unique_name, "process_even");

    // Validate that config matches our constants
    uint32_t cfg_num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    uint32_t cfg_num_elements = config.get<uint32_t>(unique_name, "num_elements");
    uint32_t cfg_time_short = config.get_default<uint32_t>(unique_name, "time_short", 16);
    uint32_t cfg_element_short = config.get_default<uint32_t>(unique_name, "element_short", 8);

    if (cfg_num_local_freq != NUM_LOCAL_FREQ) {
        throw std::runtime_error(fmt::format(
            fmt("TransposeBasebandArray: num_local_freq ({:d}) must be {:d}"),
            cfg_num_local_freq, NUM_LOCAL_FREQ));
    }
    if (cfg_num_elements != NUM_ELEMENTS) {
        throw std::runtime_error(fmt::format(
            fmt("TransposeBasebandArray: num_elements ({:d}) must be {:d}"),
            cfg_num_elements, NUM_ELEMENTS));
    }
    if (cfg_time_short != TIME_SHORT) {
        throw std::runtime_error(fmt::format(
            fmt("TransposeBasebandArray: time_short ({:d}) must be {:d}"),
            cfg_time_short, TIME_SHORT));
    }
    if (cfg_element_short != ELEMENT_SHORT) {
        throw std::runtime_error(fmt::format(
            fmt("TransposeBasebandArray: element_short ({:d}) must be {:d}"),
            cfg_element_short, ELEMENT_SHORT));
    }

    // Calculate derived dimensions
    time_long = timesamples_per_frame / TIME_SHORT;

    // Validate that dimensions divide evenly
    if (timesamples_per_frame % TIME_SHORT != 0) {
        throw std::runtime_error(fmt::format(
            fmt("TransposeBasebandArray: timesamples_per_frame ({:d}) must be divisible by "
                "time_short ({:d})"),
            timesamples_per_frame, TIME_SHORT));
    }

    // Validate input buffer size
    // Input format: E[time_long][frequency_local][element_long][time_short][element_short]
    size_t expected_input_size =
        (size_t)time_long * NUM_LOCAL_FREQ * ELEMENT_LONG * TIME_SHORT * ELEMENT_SHORT;
    if (in_buf->frame_size != expected_input_size) {
        throw std::runtime_error(fmt::format(
            fmt("TransposeBasebandArray: in_buf frame size ({:d}) does not match expected "
                "size ({:d}) for shape [time_long={:d}][num_local_freq={:d}][element_long={:d}]"
                "[time_short={:d}][element_short={:d}]"),
            in_buf->frame_size, expected_input_size, time_long, NUM_LOCAL_FREQ, ELEMENT_LONG,
            TIME_SHORT, ELEMENT_SHORT));
    }

    // Validate output buffer size
    // Output format: E[time][frequency_local][element]
    size_t expected_output_size = (size_t)timesamples_per_frame * NUM_LOCAL_FREQ * NUM_ELEMENTS;
    if (out_buf->frame_size != expected_output_size) {
        throw std::runtime_error(fmt::format(
            fmt("TransposeBasebandArray: out_buf frame size ({:d}) does not match expected "
                "size ({:d}) for shape [time={:d}][num_local_freq={:d}][num_elements={:d}]"),
            out_buf->frame_size, expected_output_size, timesamples_per_frame, NUM_LOCAL_FREQ,
            NUM_ELEMENTS));
    }

    // Validate pl_mask buffer size
    // pl_mask format: uint64_t[T/64][F][E/8]
    size_t T = timesamples_per_frame;
    size_t expected_pl_mask_size = (T / 64) * NUM_LOCAL_FREQ * (NUM_ELEMENTS / 8) * sizeof(uint64_t);
    if (pl_mask_buf->frame_size != expected_pl_mask_size) {
        throw std::runtime_error(fmt::format(
            fmt("TransposeBasebandArray: pl_mask_buf frame size ({:d}) does not match expected "
                "size ({:d}) for shape [T/64={:d}][F={:d}][E/8={:d}] * sizeof(uint64_t)"),
            pl_mask_buf->frame_size, expected_pl_mask_size, T / 64, NUM_LOCAL_FREQ, NUM_ELEMENTS / 8));
    }

    // AVX512 fast path is always enabled with these constants
    use_avx512_fast_path = false;
#ifdef __AVX512F__
    // With our constants: time_short=16, element_short=8, num_elements=128, element_long=16
    // All requirements are met
    use_avx512_fast_path = true;
    INFO("TransposeBasebandArray: AVX512 fast path enabled");
#else
    INFO("TransposeBasebandArray: AVX512 not available, using scalar path");
#endif

    // Set the output buffer frame ndarray shape
    
    // Confusingly the array name is "E" for electic field
    out_buf->allocate_ndarray_frame_desc(
            kotekan::int4x2_swapped_withoffset, "E", {timesamples_per_frame, NUM_LOCAL_FREQ, 2, NUM_ELEMENTS/2},
            {"T", "F", "P", "D"});

    INFO("TransposeBasebandArray: Initialized (process_even={:s})", process_even ? "true" : "false");
    INFO("TransposeBasebandArray: Input shape: [{:d}][{:d}][{:d}][{:d}][{:d}]", time_long,
         NUM_LOCAL_FREQ, ELEMENT_LONG, TIME_SHORT, ELEMENT_SHORT);
    INFO("TransposeBasebandArray: Output shape: [{:d}][{:d}][{:d}]", timesamples_per_frame,
         NUM_LOCAL_FREQ, NUM_ELEMENTS);
}

TransposeBasebandArray::~TransposeBasebandArray() {}

#ifdef __AVX512F__
// Prefetch a 2048-byte block into L1 cache
// Block layout: [element_long=16][time_short=16][element_short=8]
// Each cache line is 64 bytes, so we need 32 prefetches
static inline void prefetch_block(const uint8_t* block) {
    // Prefetch all 32 cache lines (2048 bytes / 64 bytes per line)
    for (int i = 0; i < 32; i++) {
        _mm_prefetch((const char*)(block + i * 64), _MM_HINT_T0);
    }
}

void TransposeBasebandArray::transpose_block_avx512(const uint8_t* in, uint8_t* out,
                                                    size_t out_stride) {
    // Input block layout: [element_long=16][time_short=16][element_short=8] = 2048 bytes
    // For a fixed t_short, we need to gather 8 bytes from each of the 16 e_long positions
    // Each e_long's data is 128 bytes apart (time_short * element_short = 16 * 8 = 128)
    //
    // Output: 16 rows (one per t_short) of 128 bytes each (num_elements)
    // Written with stride out_stride between rows

    // Gather indices for 64-bit elements (8 bytes each)
    // For e_long 0-7: offsets are 0, 128, 256, 384, 512, 640, 768, 896 bytes
    // For e_long 8-15: offsets are 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920 bytes
    const __m512i indices_lo =
        _mm512_set_epi64(7 * 128, 6 * 128, 5 * 128, 4 * 128, 3 * 128, 2 * 128, 1 * 128, 0 * 128);
    const __m512i indices_hi = _mm512_set_epi64(15 * 128, 14 * 128, 13 * 128, 12 * 128, 11 * 128,
                                                10 * 128, 9 * 128, 8 * 128);

    for (uint32_t t_short = 0; t_short < TIME_SHORT; t_short++) {
        // Base address for this t_short within the input block
        const uint8_t* base = in + t_short * ELEMENT_SHORT;

        // Gather first 64 bytes (elements 0-63 from e_long 0-7)
        __m512i data_lo = _mm512_i64gather_epi64(indices_lo, (const long long*)base, 1);

        // Gather next 64 bytes (elements 64-127 from e_long 8-15)
        __m512i data_hi = _mm512_i64gather_epi64(indices_hi, (const long long*)base, 1);

        // Write 128 bytes using non-temporal stores (bypasses cache)
        uint8_t* out_row = out + t_short * out_stride;
        _mm512_stream_si512((__m512i*)out_row, data_lo);
        _mm512_stream_si512((__m512i*)(out_row + 64), data_hi);
    }
}

void TransposeBasebandArray::fill_block_0x88_avx512(uint8_t* out, size_t out_stride) {
    // Fill 16 rows of 128 bytes each with 0x88, using non-temporal stores
    // This matches the output pattern of transpose_block_avx512
    const __m512i fill_val = _mm512_set1_epi8(0x88);

    for (uint32_t t_short = 0; t_short < TIME_SHORT; t_short++) {
        uint8_t* out_row = out + t_short * out_stride;
        _mm512_stream_si512((__m512i*)out_row, fill_val);
        _mm512_stream_si512((__m512i*)(out_row + 64), fill_val);
    }
}
#endif

void TransposeBasebandArray::main_thread() {
    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);
    frameID pl_mask_frame_id(pl_mask_buf);

    // Pre-calculate strides for the loops (using constants where possible)
    constexpr size_t in_freq_stride = (size_t)ELEMENT_LONG * TIME_SHORT * ELEMENT_SHORT; // 2048
    const size_t in_tlong_stride = (size_t)NUM_LOCAL_FREQ * in_freq_stride;
    constexpr size_t out_freq_stride = (size_t)NUM_ELEMENTS; // 128
    const size_t out_time_stride = (size_t)NUM_LOCAL_FREQ * out_freq_stride;

    // pl_mask strides: shape is uint64_t[T/64][F][E/8]
    constexpr size_t pl_mask_e_dim = NUM_ELEMENTS / 8;  // E/8 = 16
    constexpr size_t pl_mask_freq_stride = pl_mask_e_dim;  // stride to next frequency
    constexpr size_t pl_mask_t64_stride = (size_t)NUM_LOCAL_FREQ * pl_mask_freq_stride;  // stride to next T/64

    while (!stop_thread) {
        // Wait for input frame
        uint8_t* in_frame = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (in_frame == nullptr)
            break;

        // Wait for output frame
        uint8_t* out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame == nullptr)
            break;

        // Wait for pl_mask frame
        uint8_t* pl_mask_frame = pl_mask_buf->wait_for_full_frame(unique_name, pl_mask_frame_id);
        if (pl_mask_frame == nullptr)
            break;

        const uint64_t* pl_mask_ptr = (const uint64_t*)pl_mask_frame;

        // Check if we should process this frame based on even/odd filtering
        /*bool is_even_frame = ((int)in_frame_id % 2) == 0;
        if (is_even_frame != process_even) {
            // Skip this frame - mark as done without processing
            DEBUG("TransposeBasebandArray: Skipping frame {:d} (process_even={:s}, frame is {:s})",
                  (int)in_frame_id, process_even ? "true" : "false", is_even_frame ? "even" : "odd");
            in_buf->mark_frame_empty(unique_name, in_frame_id++);
            pl_mask_buf->mark_frame_empty(unique_name, pl_mask_frame_id++);
            // Return the output frame without filling it
            out_buf->mark_frame_empty(unique_name, out_frame_id++);
            continue;
        }*/

        // Transpose the data
        // Input:  E[time_long][frequency_local][element_long][time_short][element_short]
        // Output: E'[time][frequency_local][element]

#ifdef __AVX512F__
        if (use_avx512_fast_path) {
            // AVX512 fast path: process one (t_long, freq) block at a time
            // Each block is 2048 bytes and produces 16 rows of 128 bytes
            size_t lost_blocks = 0;

            for (uint32_t t_long = 0; t_long < time_long; t_long++) {
                const uint32_t base_time = t_long * TIME_SHORT;

                // Calculate pl_mask index for this t_long
                // T = time_long * time_short, so time sample = t_long * time_short
                // t64_idx = (t_long * time_short) / 64
                // bit_start = (t_long * time_short) % 64
                const size_t t64_idx = (t_long * TIME_SHORT) / 64;
                const size_t bit_start = (t_long * TIME_SHORT) % 64;
                // Mask for the 16 bits corresponding to this t_long's time_short samples
                const uint64_t check_mask = 0xffffull << bit_start;

                // Prefetch the first block of this t_long iteration
                const uint8_t* first_block = in_frame + t_long * in_tlong_stride;
                prefetch_block(first_block);

                for (uint32_t freq = 0; freq < NUM_LOCAL_FREQ; freq++) {
                    // Prefetch next block while processing current one
                    if (freq + 1 < NUM_LOCAL_FREQ) {
                        const uint8_t* next_block =
                            in_frame + t_long * in_tlong_stride + (freq + 1) * in_freq_stride;
                        prefetch_block(next_block);
                    } else if (t_long + 1 < time_long) {
                        // Prefetch first block of next t_long
                        const uint8_t* next_block = in_frame + (t_long + 1) * in_tlong_stride;
                        prefetch_block(next_block);
                    }

                    // Check pl_mask for this (t_long, freq)
                    // Since zeros always impact all E/8 entries for a given frequency,
                    // we only need to check element index 0
                    size_t pl_mask_idx = t64_idx * pl_mask_t64_stride + freq * pl_mask_freq_stride;
                    uint64_t mask_val = pl_mask_ptr[pl_mask_idx];
                    bool has_packet_loss = (mask_val & check_mask) != check_mask;

                    // Output start for (base_time, freq)
                    uint8_t* out_block =
                        out_frame + base_time * out_time_stride + freq * out_freq_stride;

                    if (has_packet_loss) {
                        // Fill with 0x88 instead of transposing
                        fill_block_0x88_avx512(out_block, out_time_stride);
                        lost_blocks++;
                    } else {
                        // Input block start
                        const uint8_t* in_block =
                            in_frame + t_long * in_tlong_stride + freq * in_freq_stride;
                        transpose_block_avx512(in_block, out_block, out_time_stride);
                    }
                }
            }
            // Memory fence to ensure all non-temporal stores are completed
            _mm_sfence();

            // Log packet loss percentage
            double loss_percentage = 100.0 * double(lost_blocks) / double(time_long * NUM_LOCAL_FREQ);
            INFO("TransposeBasebandArray: Frame {:d} data loss = {:.4f}%",
                 (int)in_frame_id, loss_percentage);
        } else
#endif
        {
            // Scalar fallback path
            for (uint32_t t_long = 0; t_long < time_long; t_long++) {
                // Calculate pl_mask index for this t_long
                const size_t t64_idx = (t_long * TIME_SHORT) / 64;
                const size_t bit_start = (t_long * TIME_SHORT) % 64;
                const uint64_t check_mask = 0xffffull << bit_start;

                for (uint32_t freq = 0; freq < NUM_LOCAL_FREQ; freq++) {
                    // Check pl_mask for this (t_long, freq)
                    size_t pl_mask_idx = t64_idx * pl_mask_t64_stride + freq * pl_mask_freq_stride;
                    uint64_t mask_val = pl_mask_ptr[pl_mask_idx];
                    bool has_packet_loss = (mask_val & check_mask) != check_mask;

                    for (uint32_t e_long = 0; e_long < ELEMENT_LONG; e_long++) {
                        for (uint32_t t_short = 0; t_short < TIME_SHORT; t_short++) {
                            // Calculate output time index
                            uint32_t time = t_long * TIME_SHORT + t_short;

                            // Calculate output base index
                            size_t out_base = (size_t)time * out_time_stride
                                              + freq * out_freq_stride + e_long * ELEMENT_SHORT;

                            if (has_packet_loss) {
                                // Fill with 0x88 instead of copying
                                std::memset(&out_frame[out_base], 0x88, ELEMENT_SHORT);
                            } else {
                                // Calculate input base index
                                size_t in_base =
                                    (size_t)t_long * in_tlong_stride + freq * in_freq_stride
                                    + e_long * (TIME_SHORT * ELEMENT_SHORT) + t_short * ELEMENT_SHORT;

                                // Copy element_short contiguous bytes
                                std::memcpy(&out_frame[out_base], &in_frame[in_base], ELEMENT_SHORT);
                            }
                        }
                    }
                }
            }
        }

        DEBUG("TransposeBasebandArray: Transposed frame {:d}", in_frame_id);

        // get a copy of the input metadata
        auto in_meta = get_chord_metadata(in_buf, in_frame_id);
        // allocate memory for the output metadata
        out_buf->allocate_new_metadata_object(out_frame_id);
        auto out_meta = get_chord_metadata(out_buf, out_frame_id);

        // Set the metadata of the output buffer
        out_meta->type = kotekan::int4x2_swapped_withoffset;
        out_meta->dims = 4;
        out_meta->set_name("E");
        out_meta->set_array_dimension(0, timesamples_per_frame, "T");
        out_meta->set_array_dimension(1, NUM_LOCAL_FREQ, "F");
        out_meta->set_array_dimension(2, 2, "P");
        out_meta->set_array_dimension(3, NUM_ELEMENTS / 2, "D");
        
        out_meta->set_strides_simple();
        out_meta->set_coarse_freq(in_meta->get_coarse_freq());
        out_meta->set_freq_upchan_factor(in_meta->get_freq_upchan_factor());

        out_meta->set_fpga_seq_num(in_meta->get_fpga_seq_num());
        out_meta->set_time_downsampling_fpga(in_meta->get_time_downsampling_fpga());
        
        // Mark frames as done
        in_buf->mark_frame_empty(unique_name, in_frame_id++);
        out_buf->mark_frame_full(unique_name, out_frame_id++);
        pl_mask_buf->mark_frame_empty(unique_name, pl_mask_frame_id++);
    }
}
