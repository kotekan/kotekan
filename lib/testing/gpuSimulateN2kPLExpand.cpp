#include "gpuSimulateN2kPLExpand.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_empty, mark_frame_full, pass_metadata
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata
#include "kotekanLogging.hpp"  // for INFO, DEBUG

#include <cstdint>
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function

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

                    pl &= 0xFFFFFFFF;  // Just in case mask out all but the
                                       // lower 32 bits.

                    // Now, run through the bits, grab the values, and put them
                    // in their places by hand.
                    //
                    // This is slow, but obvious.
                    uint64_t pl_out = 0;
                    // Loop over source bits
                    for (uint64_t b = 0; b < 32; b++) {
                        uint64_t b_out = b * 2;  // first destination bit
                        uint64_t bit = (pl >> b) & 1;  // value of source bit
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

        meta_out->set_name("cpusim_pl_mask");
        meta_out->type = kotekan::uint64;
        meta_out->dims = 4;
        assert(meta_out->dims <= CHORD_META_MAX_DIM);
        meta_out->set_array_dimension(0, nt, "Thi64");
        meta_out->set_array_dimension(1, nf, "F");
        meta_out->set_array_dimension(2, ne, "P");
        meta_out->set_array_dimension(3, ne, "D8");
        meta_out->set_strides_simple();
        meta_out->nfreq = _num_local_freq;
        assert(meta_out->nfreq <= CHORD_META_MAX_FREQ);

        meta_out->fpga_seq_num = meta_in->fpga_seq_num;
        meta_out->sample0_offset = meta_in->sample0_offset;
        meta_out->offset_downsampling = meta_in->offset_downsampling;
        for (int f = 0; f < meta_out->nfreq; f++) {
            meta_out->coarse_freq[f] = meta_in->coarse_freq[f];
            meta_out->freq_upchan_factor[f] = meta_in->freq_upchan_factor[f];
            meta_out->half_fpga_sample0[f] = meta_out->half_fpga_sample0[f];
            meta_out->time_downsampling_fpga[f] = meta_in->time_downsampling_fpga[f] / 2;
        }

        input_buf->mark_frame_empty(unique_name, input_frame_id);

        output_buf->mark_frame_full(unique_name, output_frame_id);

        INFO("Simulating GPU PL expansion done for {:s}[{:d}] result is in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, output_buf->buffer_name, output_frame_id);

        input_frame_id = (input_frame_id + 1) % input_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buf->num_frames;
    }
}
