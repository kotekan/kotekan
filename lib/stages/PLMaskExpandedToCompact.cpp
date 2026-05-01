#include "PLMaskExpandedToCompact.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "kotekanLogging.hpp"

#include "fmt.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <visUtil.hpp>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(PLMaskExpandedToCompact);

STAGE_CONSTRUCTOR(PLMaskExpandedToCompact) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    num_elements = config.get<uint32_t>(unique_name, "num_elements");
    num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");

    if (samples_per_data_set % 128 != 0) {
        throw std::runtime_error(fmt::format(
            fmt("PLMaskExpandedToCompact: samples_per_data_set ({:d}) must be a multiple of 128"),
            samples_per_data_set));
    }
    if (num_elements % 8 != 0) {
        throw std::runtime_error(
            fmt::format(fmt("PLMaskExpandedToCompact: num_elements ({:d}) must be a multiple of 8"),
                        num_elements));
    }

    const size_t T = samples_per_data_set;
    const size_t F = num_local_freq;
    const size_t E_div_8 = num_elements / 8;
    const size_t F_compact = (F + 3) / 4;

    // Validate expanded input buffer size: [T/64][F][E/8] uint64_t
    size_t expected_in_size = (T / 64) * F * E_div_8 * sizeof(uint64_t);
    if (in_buf->frame_size != expected_in_size) {
        throw std::runtime_error(fmt::format(
            fmt("PLMaskExpandedToCompact: in_buf frame size ({:d}) does not match expected "
                "expanded size ({:d}) for [T/64={:d}][F={:d}][E/8={:d}] * 8"),
            in_buf->frame_size, expected_in_size, T / 64, F, E_div_8));
    }

    // Validate compact output buffer size: [T/128][F_compact][E/8] uint64_t
    size_t expected_out_size = (T / 128) * F_compact * E_div_8 * sizeof(uint64_t);
    if (out_buf->frame_size != expected_out_size) {
        throw std::runtime_error(fmt::format(
            fmt("PLMaskExpandedToCompact: out_buf frame size ({:d}) does not match expected "
                "compact size ({:d}) for [T/128={:d}][F_compact={:d}][E/8={:d}] * 8"),
            out_buf->frame_size, expected_out_size, T / 128, F_compact, E_div_8));
    }

    // Set up ndarray frame descriptor for the output
    out_buf->allocate_ndarray_frame_desc(
        kotekan::uint1x8, "pl_mask",
        {(ptrdiff_t)(T / 128), (ptrdiff_t)F_compact, 2, (ptrdiff_t)(E_div_8 / 2), 8},
        {"T2hi64", "F4", "P", "D8", "T2lo64"});

    INFO("PLMaskExpandedToCompact: Expanded [T/64={:d}][F={:d}][E/8={:d}] -> "
         "Compact [T/128={:d}][F4={:d}][E/8={:d}]",
         T / 64, F, E_div_8, T / 128, F_compact, E_div_8);
}

PLMaskExpandedToCompact::~PLMaskExpandedToCompact() {}

// Halve the bits of a uint64: extract every other bit (even positions) and pack into 32 bits.
// Input:  bits [b0 b1 b2 b3 b4 b5 ... b62 b63]
// Output: bits [b0 b2 b4 ... b62] packed into the low 32 bits.
//
// Since packet loss is at 16-sample granularity, adjacent bit pairs are always identical,
// so extracting even bits is lossless. This is the inverse of the double_bits() operation
// in pl_mask_expander.cu.
static inline uint32_t halve_bits(uint64_t x) {
    // Use BMI2 pext if available, otherwise bit-serial fallback
#if defined(__BMI2__) && defined(__x86_64__)
    return (uint32_t)__builtin_ia32_pext_di(x, 0x5555555555555555ULL);
#else
    uint32_t result = 0;
    for (int i = 0; i < 32; i++) {
        result |= (uint32_t)((x >> (2 * i)) & 1ULL) << i;
    }
    return result;
#endif
}

void PLMaskExpandedToCompact::main_thread() {
    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    const size_t T = samples_per_data_set;
    const size_t F = num_local_freq;
    const size_t E_div_8 = num_elements / 8;
    const size_t F_compact = (F + 3) / 4;
    const size_t T_div_128 = T / 128;

    // Strides for the expanded array [T/64][F][E/8] of uint64_t
    const size_t exp_e_stride = 1;
    const size_t exp_f_stride = E_div_8;
    const size_t exp_t_stride = F * E_div_8;

    // Strides for the compact array [T/128][F_compact][E/8] of uint64_t
    const size_t comp_e_stride = 1;
    const size_t comp_f_stride = E_div_8;
    const size_t comp_t_stride = F_compact * E_div_8;

    while (!stop_thread) {
        uint8_t* in_frame = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (in_frame == nullptr)
            break;

        uint8_t* out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame == nullptr)
            break;

        const uint64_t* exp_ptr = (const uint64_t*)in_frame;
        uint64_t* comp_ptr = (uint64_t*)out_frame;

        for (size_t t128 = 0; t128 < T_div_128; t128++) {
            for (size_t f4 = 0; f4 < F_compact; f4++) {
                for (size_t e = 0; e < E_div_8; e++) {

                    // Start with all bits set (all valid)
                    uint64_t compact_val = ~0x0ULL;

                    // AND together up to 4 frequencies in this group
                    size_t f_base = f4 * 4;
                    size_t nf = (f_base + 4 <= F) ? 4 : (F - f_base);
                    for (size_t fi = 0; fi < nf; fi++) {
                        size_t f = f_base + fi;

                        // Two expanded uint64s at t64 = 2*t128 and 2*t128+1
                        uint64_t e0 = exp_ptr[(2 * t128) * exp_t_stride + f * exp_f_stride
                                              + e * exp_e_stride];
                        uint64_t e1 = exp_ptr[(2 * t128 + 1) * exp_t_stride + f * exp_f_stride
                                              + e * exp_e_stride];

                        // Halve bits: 64 expanded bits -> 32 compact bits each
                        uint32_t c0 = halve_bits(e0);
                        uint32_t c1 = halve_bits(e1);

                        // Combine into one compact uint64 and AND into the accumulator
                        compact_val &= ((uint64_t)c1 << 32) | (uint64_t)c0;
                    }

                    comp_ptr[t128 * comp_t_stride + f4 * comp_f_stride + e * comp_e_stride] =
                        compact_val;
                }
            }
        }

        // Copy and adapt metadata from input
        out_buf->allocate_new_metadata_object(out_frame_id);
        auto out_meta = get_chord_metadata(out_buf, out_frame_id);
        auto in_meta = get_chord_metadata(in_buf, in_frame_id);

        out_meta->set_fpga_seq_num(in_meta->get_fpga_seq_num());
        out_meta->type = kotekan::uint1x8;
        out_meta->dims = 5;
        out_meta->set_array_dimension(0, T_div_128, "T2hi64");
        out_meta->set_array_dimension(1, F_compact, "F4");
        out_meta->set_array_dimension(2, 2, "P");
        out_meta->set_array_dimension(3, E_div_8 / 2, "D8");
        out_meta->set_array_dimension(4, 8, "T2lo64");
        out_meta->set_name("pl_mask");
        out_meta->set_strides_simple();

        out_meta->set_coarse_freq(in_meta->get_coarse_freq());
        out_meta->set_freq_upchan_factor(in_meta->get_freq_upchan_factor());
        // Compact mask covers 128 baseband samples per uint64 (vs 64 for expanded)
        out_meta->set_time_downsampling_fpga(in_meta->get_time_downsampling_fpga() * 2);

        in_buf->mark_frame_empty(unique_name, in_frame_id);
        in_frame_id++;

        out_buf->mark_frame_full(unique_name, out_frame_id);
        out_frame_id++;
    }
}
