#include "gpuSimulateN2kPLExpand.hpp"

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType, GetType
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, metadata_is_chord, get_chord_metadata, CHO...
#include "kotekanLogging.hpp"  // for FATAL_ERROR, INFO
#include "metadata.hpp"        // for metadataObject

#include "fmt.hpp" // for compile_string_to_view

#include <assert.h>   // for assert
#include <cstdint>    // for uint64_t, int32_t, int64_t
#include <functional> // for bind, function
#include <memory>     // for shared_ptr, __shared_ptr_access
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(gpuSimulateN2kPLExpand);

gpuSimulateN2kPLExpand::gpuSimulateN2kPLExpand(Config& config, const std::string& unique_name,
                                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&gpuSimulateN2kPLExpand::main_thread, this)) {

    // Apply config.
    _num_elements = config.get<int32_t>(unique_name, "num_elements"); // = "2*D"
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");

    input_buf = get_buffer("in_buf");
    input_buf->register_consumer(unique_name);
    output_buf = get_buffer("out_buf");
    output_buf->register_producer(unique_name);

    /* new style array description */
    // number of elements = number of dishes * polarizations
    int nt = _samples_per_data_set / 64;
    int nf = _num_local_freq;
    int ne = _num_elements / 8;
    output_buf->allocate_new_frame_desc<kotekan::GetType<kotekan::uint1x8>::type, 5>(
        "pl_mask", {nt, nf, 2, ne / 2, 8}, {"Thi64", "F", "P", "D8", "Tlo64"});
}

gpuSimulateN2kPLExpand::~gpuSimulateN2kPLExpand() {}

void gpuSimulateN2kPLExpand::main_thread() {

    int input_frame_id = 0;
    int output_frame_id = 0;

    while (!stop_thread) {
        uint64_t* pl_mask = (uint64_t*)input_buf->wait_for_full_frame(unique_name, input_frame_id);
        if (pl_mask == nullptr)
            break;
        uint64_t* pl_mask_out =
            (uint64_t*)output_buf->wait_for_empty_frame(unique_name, output_frame_id);
        if (pl_mask_out == nullptr)
            break;

        INFO("Simulating GPU PL expansion for {:s}[{:d}] putting result in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, output_buf->buffer_name, output_frame_id);

        // number of elements = number of dishes * polarizations
        int nt = _samples_per_data_set / 64;
        int nf = _num_local_freq;
        int nf_hi = (_num_local_freq + 3) / 4;
        int ne = _num_elements / 8;

        // array access strides in raw (downsampled) PL mask
        int fstride_hi = ne;
        int tstride_hi = ne * nf_hi;

        // array access strides in expanded PL mask
        int fstride = ne;
        int tstride = ne * nf;

        INFO("Running stage expanding nt={:d}, nf={:d} 64-bit masks to nt={:d}, nf={:d} with "
             "downsampled num_elements = {:d} = {:d} / 8",
             nt / 2, nf_hi, nt, nf, ne, _num_elements);

        // Looping over entries in the _expanded_ PL mask.
        for (int t = 0; t < nt; t++) {
            for (int f_hi = 0; f_hi < nf_hi; f_hi++) {
                for (int e = 0; e < ne; e++) {

                    // Get the indices into the downsampled array
                    int t_hi = t >> 1;

                    // Grab the value of the PL mask
                    uint64_t pl = pl_mask[t_hi * tstride_hi + f_hi * fstride_hi + e];

                    // Each 64 bit value in the unexpanded mask represents 128
                    // time samples.
                    // To "expand" the mask we need to double each bit:
                    // interleave the bits of the 64 bit value to get a 128
                    // bit value:
                    //
                    // [3210] --> [33221100]

                    // We'll do this in chunks, if t is odd, we want the higher
                    // 32 bits to expand into a 64 bit value.  if t is even, we
                    // want the lower 32 bits.
                    if (t & 1)
                        pl >>= 32;

                    pl &= 0xFFFFFFFF; // Just in case mask out all but the
                                      // lower 32 bits.

                    // Now, run through the bits, grab the values, and put them
                    // in their places by hand.
                    //
                    // This is slow, but obvious.
                    uint64_t pl_out = 0;
                    // Loop over source bits
                    for (uint64_t b = 0; b < 32; b++) {
                        uint64_t b_out = b * 2;       // first destination bit
                        uint64_t bit = (pl >> b) & 1; // value of source bit
                        // add source bit to dest and dest+1
                        pl_out |= (bit << b_out) | (bit << (b_out + 1));
                    }

                    // Copy over frequency, and put value in place.
                    for (int f_lo = 0; f_lo < 4; f_lo++) {
                        int f = f_lo + (f_hi << 2);
                        pl_mask_out[t * tstride + f * fstride + e] = pl_out;
                    }
                } // e
            } // f_hi
        } // t


        // Fetch input metadata
        const std::shared_ptr<const metadataObject> mc_in = input_buf->get_metadata(input_frame_id);
        if (!mc_in) {
            FATAL_ERROR("Buffer {:s} frame {:d} had no metadata", input_buf->buffer_name,
                        input_frame_id);
        }
        assert(mc_in);
        if (!metadata_is_chord(mc_in)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        input_buf->buffer_name, input_frame_id);
        }
        assert(metadata_is_chord(mc_in));

        const std::shared_ptr<const chordMetadata> meta_in = get_chord_metadata(mc_in);

        // Create output metadata
        output_buf->allocate_new_metadata_object(output_frame_id);
        const std::shared_ptr<metadataObject> mc = output_buf->get_metadata(output_frame_id);
        if (!mc) {
            FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata", output_buf->buffer_name,
                        output_frame_id);
        }
        assert(mc);
        if (!metadata_is_chord(mc)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        output_buf->buffer_name, output_frame_id);
        }
        assert(metadata_is_chord(mc));
        const std::shared_ptr<chordMetadata> meta_out = get_chord_metadata(mc);
        assert(meta_out);

        // Start with a copy
        meta_out->deepCopy(meta_in);

        // Update changes
        meta_out->set_name("pl_mask");
        meta_out->type = kotekan::uint1x8;
        meta_out->dims = 5;
        assert(meta_out->dims <= CHORD_META_MAX_DIM);
        meta_out->set_array_dimension(0, nt, "Thi64");
        meta_out->set_array_dimension(1, nf, "F");
        meta_out->set_array_dimension(2, 2, "P");
        meta_out->set_array_dimension(3, ne / 2, "D8");
        meta_out->set_array_dimension(4, 8, "Tlo64");
        meta_out->set_strides_simple();
        // frame_desc set in constructor
        /* test that things are consistent */
        meta_out->check_frame_desc(output_buf->get_frame_desc());

        // This looks inconsistent
        meta_out->set_fpga_seq_num(meta_in->get_fpga_seq_num());
        meta_out->set_sample0_offset(2 * meta_in->get_sample0_offset());
        meta_out->set_offset_downsampling(meta_in->get_offset_downsampling());

        const std::vector<int> coarse_freq_in = meta_in->get_coarse_freq();
        const std::vector<int> freq_upchan_factor_in = meta_in->get_freq_upchan_factor();
        const std::vector<int64_t> half_fpga_sample0_in = meta_in->get_half_fpga_sample0();
        const std::vector<int> time_downsampling_fpga_in = meta_in->get_time_downsampling_fpga();
        std::vector<int> coarse_freq(_num_local_freq);
        std::vector<int> freq_upchan_factor(coarse_freq.size());
        std::vector<int64_t> half_fpga_sample0(coarse_freq.size());
        std::vector<int> time_downsampling_fpga(coarse_freq.size());

        for (int f = 0; f < static_cast<int>(coarse_freq.size()); f++) {
            coarse_freq[f] = coarse_freq_in[f];
            freq_upchan_factor[f] = freq_upchan_factor_in[f];
            time_downsampling_fpga[f] = time_downsampling_fpga_in[f] / 2;
            half_fpga_sample0[f] =
                half_fpga_sample0_in[f] + time_downsampling_fpga[f] - time_downsampling_fpga_in[f];
        }

        meta_out->set_coarse_freq(coarse_freq);
        meta_out->set_freq_upchan_factor(freq_upchan_factor);
        meta_out->set_half_fpga_sample0(half_fpga_sample0);
        meta_out->set_time_downsampling_fpga(time_downsampling_fpga);
        assert(meta_out->get_nfreq() <= CHORD_META_MAX_FREQ);

        input_buf->mark_frame_empty(unique_name, input_frame_id);
        output_buf->mark_frame_full(unique_name, output_frame_id);

        INFO("Simulating GPU PL expansion done for {:s}[{:d}] result is in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, output_buf->buffer_name, output_frame_id);

        input_frame_id = (input_frame_id + 1) % input_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buf->num_frames;
    }
}
