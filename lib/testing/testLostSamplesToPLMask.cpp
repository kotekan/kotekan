#include "testLostSamplesToPLMask.hpp"

#include "Config.hpp" // for Config
#include "Hash.hpp"
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
#include <string.h>   // for memcpy, size_t
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(testLostSamplesToPLMask);

// various constants defining the array layout of the package loss mask
#define PL_MASK_DOWNSAMPLING_FACTOR 2
#define PL_MASK_HILO_SPLIT 64
#define PL_MASK_DISHES_PER_BIN 8
#define PL_MASK_FREQS_PER_BIN 4

#define BITS_PER_BYTE 8

// CHIME parameters, actually
#define NUM_DISHES 1024
#define NUM_POLARIZATIONS 2

testLostSamplesToPLMask::testLostSamplesToPLMask(Config& config, const std::string& unique_name,
                                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&testLostSamplesToPLMask::main_thread, this)) {

    pl_mask_buf = get_buffer("pl_mask_buf");
    pl_mask_buf->register_producer(unique_name);

    // Register as producer for all lost samples buffers
    json ls_bufs = config.get_value(unique_name, "lost_samples_buffers");
    for (json::iterator it = ls_bufs.begin(); it != ls_bufs.end(); ++it) {
        Buffer* buf = buffer_container.get_buffer(it.value());
        lost_samples_bufs.push_back(buf);
        buf->register_producer(unique_name);
        if (buf->frame_size != lost_samples_bufs.at(0)->frame_size)
            FATAL_ERROR("Lost samples buffers have different frame sizes: {:d} != {:d}",
                        buf->frame_size, lost_samples_bufs.at(0)->frame_size);
        if (buf->num_frames != lost_samples_bufs.at(0)->num_frames)
            FATAL_ERROR("Lost samples buffers have different number of frames: {:d} != {:d}",
                        buf->num_frames, lost_samples_bufs.at(0)->num_frames);
    }
    const int num_freq_bins = int(lost_samples_bufs.size());

    if (pl_mask_buf->frame_size
        != lost_samples_bufs.at(0)->frame_size / PL_MASK_DOWNSAMPLING_FACTOR * NUM_DISHES
               / PL_MASK_DISHES_PER_BIN * NUM_POLARIZATIONS / BITS_PER_BYTE * num_freq_bins)
        FATAL_ERROR("Unexpected frames sizes for pl_mask {:d} and lost_samples {:d}",
                    pl_mask_buf->frame_size, lost_samples_bufs.at(0)->frame_size);
}

testLostSamplesToPLMask::~testLostSamplesToPLMask() {}

// produce some data somewhat randomly
static bool is_lost(int time, int fbin) {
    std::string buffer(2 * sizeof(int), '\0');
    // lost_samples do not depend on dish or polarization
    std::memcpy(buffer.data() + 0 * sizeof(int), &time, sizeof(int));
    std::memcpy(buffer.data() + 1 * sizeof(int), &fbin, sizeof(int));
    Hash hashval = hash(buffer);
    return hashval.l & 1;
}

void testLostSamplesToPLMask::main_thread() {

    const int num_freq_bins = int(lost_samples_bufs.size());
    const int samples_in_dataset = int(lost_samples_bufs.at(0)->frame_size);

    int frame_id = 0;
    int64_t seq_num = 0;
    while (!stop_thread) {

        uint8_t* pl_mask_frame = pl_mask_buf->wait_for_empty_frame(unique_name, frame_id);
        if (pl_mask_frame == nullptr)
            return;

        // buffer_format [time / 2 % 64][dish / 8][polr][freq / 4][time / 2 / 64]
        std::memset(pl_mask_frame, 0x0, pl_mask_buf->frame_size);
        int pl_idx = 0;
        for (int thi = 0;
             thi < samples_in_dataset / PL_MASK_DOWNSAMPLING_FACTOR / PL_MASK_HILO_SPLIT; ++thi)
            for (int fbin = 0; fbin < num_freq_bins; ++fbin)
                for (int polr = 0; polr < NUM_POLARIZATIONS; ++polr)
                    for (int dbin = 0; dbin < NUM_DISHES / PL_MASK_DISHES_PER_BIN; ++dbin)
                        for (int tlo = 0; tlo < PL_MASK_HILO_SPLIT; ++tlo) {
                            bool lost = false;
                            for (int ds = 0; ds < PL_MASK_DOWNSAMPLING_FACTOR; ++ds)
                                lost |= is_lost((thi * PL_MASK_HILO_SPLIT + tlo)
                                                        * PL_MASK_DOWNSAMPLING_FACTOR
                                                    + ds,
                                                fbin);
                            assert(size_t(pl_idx / BITS_PER_BYTE) < pl_mask_buf->frame_size);
                            // indexing is a bit annoying due to bits and downsampling
                            assert(pl_idx
                                   == (((thi * num_freq_bins + fbin) * NUM_POLARIZATIONS + polr)
                                           * NUM_DISHES / PL_MASK_DISHES_PER_BIN
                                       + dbin) * PL_MASK_HILO_SPLIT
                                          + tlo);
                            pl_mask_frame[pl_idx / BITS_PER_BYTE] |= (!lost)
                                                                     << (pl_idx % BITS_PER_BYTE);
                            pl_idx += 1;
                        }

        pl_mask_buf->allocate_new_metadata_object(frame_id);
        auto pl_mask_meta = get_chord_metadata(pl_mask_buf, frame_id);

        // physics metadata
        // TODO: add more that dpdk adds
#warning "THIS IS INCONSISTENT"
        pl_mask_meta->set_fpga_seq_num(seq_num);
        pl_mask_meta->set_sample0_offset(seq_num);
        pl_mask_meta->set_offset_downsampling(PL_MASK_DOWNSAMPLING_FACTOR * PL_MASK_HILO_SPLIT);

        std::vector<int> coarse_freq(num_freq_bins * PL_MASK_FREQS_PER_BIN);
        for (size_t f = 0; f < coarse_freq.size(); ++f)
            coarse_freq.at(f) = f;
        pl_mask_meta->set_coarse_freq(coarse_freq);

        const std::vector<int> freq_upchan_factor(num_freq_bins * PL_MASK_FREQS_PER_BIN,
                                                  1); // we want 1/4 but we cannot
        pl_mask_meta->set_freq_upchan_factor(freq_upchan_factor);

        const std::vector<int64_t> half_fpga_sample0(num_freq_bins * PL_MASK_FREQS_PER_BIN,
                                                     PL_MASK_DOWNSAMPLING_FACTOR
                                                         * PL_MASK_HILO_SPLIT / 2);
        pl_mask_meta->set_half_fpga_sample0(half_fpga_sample0);

        const std::vector<int> time_downsampling_fpga(num_freq_bins * PL_MASK_FREQS_PER_BIN,
                                                      PL_MASK_DOWNSAMPLING_FACTOR
                                                          * PL_MASK_HILO_SPLIT);
        pl_mask_meta->set_time_downsampling_fpga(time_downsampling_fpga);

        // array description
        std::strncpy(pl_mask_meta->name, "pl_mask", sizeof pl_mask_meta->name);
        pl_mask_meta->type = kotekan::uint1x8;
        pl_mask_meta->dims = 5;
        assert(pl_mask_meta->dims <= CHORD_META_MAX_DIM);
        std::strncpy(pl_mask_meta->dim_name[0], "T2hi64", sizeof pl_mask_meta->dim_name[0]);
        std::strncpy(pl_mask_meta->dim_name[1], "F4", sizeof pl_mask_meta->dim_name[1]);
        std::strncpy(pl_mask_meta->dim_name[2], "P", sizeof pl_mask_meta->dim_name[2]);
        std::strncpy(pl_mask_meta->dim_name[3], "D8", sizeof pl_mask_meta->dim_name[3]);
        std::strncpy(pl_mask_meta->dim_name[4], "T2lo64", sizeof pl_mask_meta->dim_name[4]);
        pl_mask_meta->dim[0] =
            lost_samples_bufs.at(0)->frame_size / PL_MASK_DOWNSAMPLING_FACTOR / PL_MASK_HILO_SPLIT;
        pl_mask_meta->dim[1] = lost_samples_bufs.size();
        pl_mask_meta->dim[2] = NUM_POLARIZATIONS;
        pl_mask_meta->dim[3] = NUM_DISHES / PL_MASK_DISHES_PER_BIN;
        pl_mask_meta->dim[4] =
            PL_MASK_HILO_SPLIT / BITS_PER_BYTE; // because we count uint1x8, not uint1
        for (int d = pl_mask_meta->dims - 1; d >= 0; --d)
            if (d == pl_mask_meta->dims - 1)
                pl_mask_meta->stride[d] = 1;
            else
                pl_mask_meta->stride[d] = pl_mask_meta->stride[d + 1] * pl_mask_meta->dim[d + 1];

        pl_mask_buf->allocate_new_frame_desc<kotekan::GetType_t<kotekan::uint1x8>, 5>(
            "pl_mask",
            {ptrdiff_t(lost_samples_bufs.at(0)->frame_size / PL_MASK_DOWNSAMPLING_FACTOR
                       / PL_MASK_HILO_SPLIT),
             ptrdiff_t(lost_samples_bufs.size()), NUM_POLARIZATIONS,
             NUM_DISHES / PL_MASK_DISHES_PER_BIN,
             PL_MASK_HILO_SPLIT / BITS_PER_BYTE /* because we count uint1x8, not uint1 */},
            {"T2hi64", "F4", "P", "D8", "T2lo64"});
        pl_mask_meta->check_frame_desc(pl_mask_buf->get_frame_desc());

        // done
        pl_mask_buf->mark_frame_full(unique_name, frame_id);

        // lost_samples_buf
        for (int fbin = 0; fbin < num_freq_bins; ++fbin) {
            auto lost_samples_buf = lost_samples_bufs.at(fbin);
            uint8_t* flag_frame = lost_samples_buf->wait_for_empty_frame(unique_name, frame_id);
            if (flag_frame == nullptr)
                return;

            // this one has easy indexing, everyting is just a linear array
            for (int t = 0; t < samples_in_dataset; ++t)
                flag_frame[t] = is_lost(t, fbin);

            lost_samples_buf->allocate_new_metadata_object(frame_id);
            auto lost_samples_meta = get_chord_metadata(lost_samples_buf, frame_id);

            // physics metadata
            // TODO: add more that dpdk adds
#warning "THIS IS INCONSISTENT"
            lost_samples_meta->set_fpga_seq_num(seq_num);
            lost_samples_meta->set_sample0_offset(seq_num);

            lost_samples_meta->set_coarse_freq(
                std::vector<int>(&coarse_freq[fbin * PL_MASK_FREQS_PER_BIN],
                                 &coarse_freq[(fbin + 1) * PL_MASK_FREQS_PER_BIN]));

            // array description
            std::strncpy(lost_samples_meta->name, "lost_samples", sizeof lost_samples_meta->name);
            lost_samples_meta->type = kotekan::uint8;
            lost_samples_meta->dims = 1;
            assert(lost_samples_meta->dims <= CHORD_META_MAX_DIM);
            std::strncpy(lost_samples_meta->dim_name[0], "T",
                         sizeof lost_samples_meta->dim_name[0]);
            lost_samples_meta->dim[0] = lost_samples_bufs.at(0)->frame_size;
            lost_samples_meta->stride[0] = 1;

            lost_samples_buf->allocate_new_frame_desc<kotekan::GetType_t<kotekan::uint8>, 1>(
                "lost_samples", {ptrdiff_t(lost_samples_bufs.at(0)->frame_size)}, {"T"});
            lost_samples_meta->check_frame_desc(lost_samples_buf->get_frame_desc());

            // done
            lost_samples_buf->mark_frame_full(unique_name, frame_id);
        }
        frame_id = (frame_id + 1) % pl_mask_buf->num_frames;
    }
}
