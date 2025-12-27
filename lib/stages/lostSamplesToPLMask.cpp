#include "lostSamplesToPLMask.hpp"

#include "Config.hpp"          // for Config
#include "Metadata.hpp"        // for GenericNDArray
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for get_chord_metadata, chordMetadata
#include "nt_memset.h"         // for nt_memset

#include "json.hpp" // for basic_json, json, iter_impl

#include <algorithm>  // for max
#include <assert.h>   // for assert
#include <functional> // for bind, function
#include <memory>     // for __shared_ptr_access, shared_ptr
#include <string.h>   // for strncpy, memcpy
#include <string.h>   // for memcpy, size_t
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(lostSamplesToPLMask);

// various constants defining the array layout of the package loss mask
#define PL_MASK_DOWNSAMPLING_FACTOR 2
#define PL_MASK_HILO_SPLIT 64
#define PL_MASK_DISHES_PER_BIN 8
#define PL_MASK_FREQS_PER_BIN 4

#define BITS_PER_BYTE 8

// CHIME parameters, actually
#define NUM_DISHES 1024
#define NUM_POLARIZATIONS 2

lostSamplesToPLMask::lostSamplesToPLMask(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&lostSamplesToPLMask::main_thread, this)) {

    pl_mask_buf = get_buffer("pl_mask_buf");
    pl_mask_buf->register_producer(unique_name);

    // Register as producer for all lost samples buffers
    json in_bufs = config.get_value(unique_name, "lost_samples_buffers");
    for (json::iterator it = in_bufs.begin(); it != in_bufs.end(); ++it) {
        Buffer* buf = buffer_container.get_buffer(it.value());
        lost_samples_bufs.push_back(buf);
        buf->register_consumer(unique_name);
        if (buf->frame_size != lost_samples_bufs.at(0)->frame_size)
            FATAL_ERROR("Input buffers have different frame sizes: {:d} != {:d}", buf->frame_size,
                        lost_samples_bufs.at(0)->frame_size);
        if (buf->num_frames != lost_samples_bufs.at(0)->num_frames)
            FATAL_ERROR("Input buffers have different number of frames: {:d} != {:d}",
                        buf->num_frames, lost_samples_bufs.at(0)->num_frames);
    }
    const int num_freq_bins = int(lost_samples_bufs.size());

    if (pl_mask_buf->frame_size
        != lost_samples_bufs.at(0)->frame_size / PL_MASK_DOWNSAMPLING_FACTOR * NUM_DISHES
               / PL_MASK_DISHES_PER_BIN * NUM_POLARIZATIONS / BITS_PER_BYTE * num_freq_bins)
        FATAL_ERROR("Unexpected frames sizes for pl_mask {:d} and lost_samples {:d}",
                    pl_mask_buf->frame_size, lost_samples_bufs.at(0)->frame_size);
}

lostSamplesToPLMask::~lostSamplesToPLMask() {}

void lostSamplesToPLMask::main_thread() {

    const size_t num_freq_bins = lost_samples_bufs.size();

    while (!stop_thread) {

        uint8_t* pl_mask_frame =
            pl_mask_buf->wait_for_empty_frame(unique_name, pl_mask_buf_frame_id);
        if (pl_mask_frame == nullptr)
            break;
        pl_mask_buf->allocate_new_metadata_object(pl_mask_buf_frame_id);
        auto pl_mask_meta = get_chord_metadata(pl_mask_buf, pl_mask_buf_frame_id);

        for (size_t f = 0; f < lost_samples_bufs.size(); ++f) {
            auto lost_samples_buf = lost_samples_bufs.at(f);
            uint8_t* flag_frame =
                lost_samples_buf->wait_for_full_frame(unique_name, lost_samples_buf_frame_id);
            if (flag_frame == nullptr)
                break;

            // constant for all iterations but only set by the producer before
            // it marks the frame as full, so cannot be checked before the first
            // wait_for_full_frame()
            auto const expected_lost_samples_frame_desc = kotekan::GenericNDArray::create(
                kotekan::DataType::uint8, "lost_samples", {ptrdiff_t(lost_samples_buf->frame_size)},
                {"T"}, nullptr);
            assert(*lost_samples_buf->get_frame_desc() == *expected_lost_samples_frame_desc);

            // pl_mask buffer_format [time / 2 % 64][dish / 8][polr][freq / 4][time / 2 / 64]
            for (size_t ihi = 0; ihi < lost_samples_buf->frame_size;
                 ihi += PL_MASK_DOWNSAMPLING_FACTOR * PL_MASK_HILO_SPLIT) {
                uint8_t buf[PL_MASK_HILO_SPLIT / BITS_PER_BYTE];
                for (size_t ilo = 0; ilo < PL_MASK_DOWNSAMPLING_FACTOR * PL_MASK_HILO_SPLIT;
                     ilo += PL_MASK_DOWNSAMPLING_FACTOR * BITS_PER_BYTE) {

                    uint8_t out_byte = 0;
                    for (size_t b = 0; b < BITS_PER_BYTE; ++b) {
                        uint8_t flagged = 0;
                        for (size_t j = 0; j < PL_MASK_DOWNSAMPLING_FACTOR; ++j)
                            flagged |= flag_frame[ihi + ilo + PL_MASK_DOWNSAMPLING_FACTOR * b + j];
                        out_byte |= (!flagged) << b; // mask == 0 -> lost, lost_samples == 1 -> lost
                    }

                    buf[ilo / PL_MASK_DOWNSAMPLING_FACTOR / BITS_PER_BYTE] = out_byte;
                }

                // each flagged sample indicates all dishes, all polarization, 4
                // frequencies
                size_t idx =
                    (ihi / PL_MASK_DOWNSAMPLING_FACTOR / PL_MASK_HILO_SPLIT * num_freq_bins + f)
                    * NUM_POLARIZATIONS * NUM_DISHES / PL_MASK_DISHES_PER_BIN * PL_MASK_HILO_SPLIT
                    / BITS_PER_BYTE;
                for (int dbin = 0; dbin < NUM_DISHES / PL_MASK_DISHES_PER_BIN; ++dbin) {
                    for (int polr = 0; polr < NUM_POLARIZATIONS; ++polr) {
                        std::memcpy(&pl_mask_frame[idx], buf, sizeof(buf));
                        idx += sizeof(buf);
                    }
                }
            }

            // merge metadata from lost_samples buffers into pl_mask buffer
            const auto lost_samples_meta =
                get_chord_metadata(lost_samples_buf, lost_samples_buf_frame_id);
            assert(lost_samples_meta->get_coarse_freq().size() == PL_MASK_FREQS_PER_BIN);
            if (f == 0) { // first time
                pl_mask_meta->deepCopy(lost_samples_meta);

                // update array description
                std::strncpy(pl_mask_meta->name, "pl_mask", sizeof pl_mask_meta->name);
                pl_mask_meta->type = kotekan::uint1x8;
                pl_mask_meta->dims = 5;
                assert(pl_mask_meta->dims <= CHORD_META_MAX_DIM);
                std::strncpy(pl_mask_meta->dim_name[0], "T2hi64", sizeof pl_mask_meta->dim_name[0]);
                std::strncpy(pl_mask_meta->dim_name[1], "F4", sizeof pl_mask_meta->dim_name[1]);
                std::strncpy(pl_mask_meta->dim_name[2], "P", sizeof pl_mask_meta->dim_name[2]);
                std::strncpy(pl_mask_meta->dim_name[3], "D8", sizeof pl_mask_meta->dim_name[3]);
                std::strncpy(pl_mask_meta->dim_name[4], "T2lo64", sizeof pl_mask_meta->dim_name[4]);
                pl_mask_meta->dim[0] = lost_samples_bufs.at(0)->frame_size
                                       / PL_MASK_DOWNSAMPLING_FACTOR / PL_MASK_HILO_SPLIT;
                pl_mask_meta->dim[1] = lost_samples_bufs.size();
                pl_mask_meta->dim[2] = NUM_POLARIZATIONS;
                pl_mask_meta->dim[3] = NUM_DISHES / PL_MASK_DISHES_PER_BIN;
                pl_mask_meta->dim[4] =
                    PL_MASK_HILO_SPLIT / BITS_PER_BYTE; // because we count uint1x8, not uint1
                for (int d = pl_mask_meta->dims - 1; d >= 0; --d)
                    if (d == pl_mask_meta->dims - 1)
                        pl_mask_meta->stride[d] = 1;
                    else
                        pl_mask_meta->stride[d] =
                            pl_mask_meta->stride[d + 1] * pl_mask_meta->dim[d + 1];

                // add metadata that DPDK does not add

                pl_mask_meta->set_offset_downsampling(PL_MASK_DOWNSAMPLING_FACTOR
                                                      * PL_MASK_HILO_SPLIT);

                const std::vector<int> freq_upchan_factor(PL_MASK_FREQS_PER_BIN,
                                                          1); // we want 1/4 but we cannot
                pl_mask_meta->set_freq_upchan_factor(freq_upchan_factor);

                const std::vector<int64_t> half_fpga_sample0(
                    PL_MASK_FREQS_PER_BIN, PL_MASK_DOWNSAMPLING_FACTOR * PL_MASK_HILO_SPLIT / 2);
                pl_mask_meta->set_half_fpga_sample0(half_fpga_sample0);

                const std::vector<int> time_downsampling_fpga(
                    PL_MASK_FREQS_PER_BIN, PL_MASK_DOWNSAMPLING_FACTOR * PL_MASK_HILO_SPLIT);
                pl_mask_meta->set_time_downsampling_fpga(time_downsampling_fpga);

                // TODO: do I need to set frame_counter? The FEngine does.
                // pl_mask_meta->set_frame_counter(E_frame_index);
                // one the other hand, FEngine does not set fpga_seq_num, but
                // DPDK does
            } else {
                const auto lost_samples_coarse_freq = lost_samples_meta->get_coarse_freq();
                auto pl_mask_coarse_freq = pl_mask_meta->get_coarse_freq();
                pl_mask_coarse_freq.insert(pl_mask_coarse_freq.end(),
                                           lost_samples_coarse_freq.begin(),
                                           lost_samples_coarse_freq.end());
                pl_mask_meta->set_coarse_freq(pl_mask_coarse_freq);

                const std::vector<int> freq_upchan_factor(PL_MASK_FREQS_PER_BIN,
                                                          1); // we want 1/4 but we cannot
                auto pl_mask_freq_upchan_factor = pl_mask_meta->get_freq_upchan_factor();
                pl_mask_freq_upchan_factor.insert(pl_mask_freq_upchan_factor.end(),
                                                  freq_upchan_factor.begin(),
                                                  freq_upchan_factor.end());
                pl_mask_meta->set_freq_upchan_factor(pl_mask_freq_upchan_factor);

                const std::vector<int64_t> half_fpga_sample0(
                    PL_MASK_FREQS_PER_BIN, PL_MASK_DOWNSAMPLING_FACTOR * PL_MASK_HILO_SPLIT / 2);
                auto pl_mask_half_fpga_sample0 = pl_mask_meta->get_half_fpga_sample0();
                pl_mask_half_fpga_sample0.insert(pl_mask_half_fpga_sample0.end(),
                                                 half_fpga_sample0.begin(),
                                                 half_fpga_sample0.end());
                pl_mask_meta->set_half_fpga_sample0(pl_mask_half_fpga_sample0);

                const std::vector<int> time_downsampling_fpga(
                    PL_MASK_FREQS_PER_BIN, PL_MASK_DOWNSAMPLING_FACTOR * PL_MASK_HILO_SPLIT);
                auto pl_mask_time_downsampling_fpga = pl_mask_meta->get_time_downsampling_fpga();
                pl_mask_time_downsampling_fpga.insert(pl_mask_time_downsampling_fpga.end(),
                                                      time_downsampling_fpga.begin(),
                                                      time_downsampling_fpga.end());
                pl_mask_meta->set_time_downsampling_fpga(pl_mask_time_downsampling_fpga);
            }

            lost_samples_buf->mark_frame_empty(unique_name, lost_samples_buf_frame_id);
        }
        lost_samples_buf_frame_id =
            (lost_samples_buf_frame_id + 1) % lost_samples_bufs.at(0)->num_frames;

        pl_mask_buf->allocate_ndarray_frame_desc<kotekan::GetType_t<kotekan::uint1x8>, 5>(
            "pl_mask",
            {ptrdiff_t(lost_samples_bufs.at(0)->frame_size / PL_MASK_DOWNSAMPLING_FACTOR
                       / PL_MASK_HILO_SPLIT),
             ptrdiff_t(lost_samples_bufs.size()), NUM_POLARIZATIONS,
             NUM_DISHES / PL_MASK_DISHES_PER_BIN,
             PL_MASK_HILO_SPLIT / BITS_PER_BYTE /* because we count uint1x8, not uint1 */},
            {"T2hi64", "F4", "P", "D8", "T2lo64"});
        pl_mask_meta->check_frame_desc(pl_mask_buf->get_frame_desc());
        pl_mask_buf->mark_frame_full(unique_name, pl_mask_buf_frame_id);
        pl_mask_buf_frame_id = (pl_mask_buf_frame_id + 1) % pl_mask_buf->num_frames;
    }
}
