#include "testLostSamplesToPLMask.hpp"

#include "Config.hpp" // for Config
#include "Hash.hpp"
#include "Metadata.hpp"        // for GenericNDArray
#include "NDArray.hpp"         // for GenericNDArray, NDArray
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

testLostSamplesToPLMask::testLostSamplesToPLMask(Config& config, const std::string& unique_name,
                                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&testLostSamplesToPLMask::main_thread, this)),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")) {

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
        != lost_samples_bufs.at(0)->frame_size / PL_MASK_DOWNSAMPLING_FACTOR * num_dishes
               / PL_MASK_DISHES_PER_BIN * num_polarizations / BITS_PER_BYTE * num_freq_bins)
        FATAL_ERROR("Unexpected frames sizes for pl_mask {:d} and lost_samples {:d}",
                    pl_mask_buf->frame_size, lost_samples_bufs.at(0)->frame_size);

    pl_mask_buf->set_frame_desc(kotekan::NDArray<kotekan::GetType_t<kotekan::uint1x8>, 5>::describe(
        "pl_mask",
        {ptrdiff_t(lost_samples_bufs.at(0)->frame_size / PL_MASK_DOWNSAMPLING_FACTOR
                   / PL_MASK_HILO_SPLIT),
         ptrdiff_t(lost_samples_bufs.size()), num_polarizations,
         num_dishes / PL_MASK_DISHES_PER_BIN,
         PL_MASK_HILO_SPLIT / BITS_PER_BYTE /* because we count uint1x8, not uint1 */},
        {"T2hi64", "F4", "P", "D8", "T2lo64"}));

    for (int fbin = 0; fbin < num_freq_bins; ++fbin) {
        auto lost_samples_buf = lost_samples_bufs.at(fbin);
        lost_samples_buf->set_frame_desc(
            kotekan::NDArray<kotekan::GetType_t<kotekan::uint8>, 1>::describe(
                "lost_samples", {ptrdiff_t(lost_samples_bufs.at(0)->frame_size)}, {"T"}));
    }
}

testLostSamplesToPLMask::~testLostSamplesToPLMask() {}

// produce some data somewhat randomly
static bool is_lost(int time, int fbin) {
    std::string buffer(2 * sizeof(int), '\0');
    // ensure result is consistent across PL mask downsampling.
    int time_ds = (time / PL_MASK_DOWNSAMPLING_FACTOR) * PL_MASK_DOWNSAMPLING_FACTOR;
    // lost_samples do not depend on dish or polarization
    std::memcpy(buffer.data() + 0 * sizeof(int), &time_ds, sizeof(int));
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
                for (int polr = 0; polr < num_polarizations; ++polr)
                    for (int dbin = 0; dbin < num_dishes / PL_MASK_DISHES_PER_BIN; ++dbin)
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
                                   == (((thi * num_freq_bins + fbin) * num_polarizations + polr)
                                           * num_dishes / PL_MASK_DISHES_PER_BIN
                                       + dbin) * PL_MASK_HILO_SPLIT
                                          + tlo);
                            pl_mask_frame[pl_idx / BITS_PER_BYTE] |= (!lost)
                                                                     << (pl_idx % BITS_PER_BYTE);
                            pl_idx += 1;
                        }

        pl_mask_buf->allocate_new_metadata_object(frame_id);
        auto pl_mask_meta = get_chord_metadata(pl_mask_buf, frame_id);

        pl_mask_meta->set_from_frame_desc(pl_mask_buf->get_frame_desc<kotekan::GenericNDArray>());

        // physics metadata
        // TODO: add more that dpdk adds
        pl_mask_meta->set_fpga_seq_num(seq_num);
        pl_mask_meta->set_time_downsampling_fpga(PL_MASK_DOWNSAMPLING_FACTOR * PL_MASK_HILO_SPLIT);

        std::vector<int> coarse_freq(num_freq_bins * PL_MASK_FREQS_PER_BIN);
        for (size_t f = 0; f < coarse_freq.size(); ++f)
            coarse_freq.at(f) = f;
        pl_mask_meta->set_coarse_freq(coarse_freq);

        const std::vector<int> freq_upchan_factor(num_freq_bins * PL_MASK_FREQS_PER_BIN,
                                                  1); // we want 1/4 but we cannot
        const std::vector<int> freq_upchan_index(num_freq_bins * PL_MASK_FREQS_PER_BIN, 0);
        pl_mask_meta->set_freq_upchan_factor(freq_upchan_factor);
        pl_mask_meta->set_freq_upchan_index(freq_upchan_index);

        pl_mask_meta->check_frame_desc(pl_mask_buf->get_frame_desc<kotekan::GenericNDArray>());

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
            lost_samples_meta->set_from_frame_desc(
                lost_samples_buf->get_frame_desc<kotekan::GenericNDArray>());

            // physics metadata
            // TODO: add more that dpdk adds
            lost_samples_meta->set_fpga_seq_num(seq_num);
            lost_samples_meta->set_freq_upchan_factor(std::vector<int>(PL_MASK_FREQS_PER_BIN, 1));
            lost_samples_meta->set_freq_upchan_index(std::vector<int>(PL_MASK_FREQS_PER_BIN, 0));
            lost_samples_meta->set_time_downsampling_fpga(1);

            lost_samples_meta->set_coarse_freq(
                std::vector<int>(&coarse_freq[fbin * PL_MASK_FREQS_PER_BIN],
                                 &coarse_freq[(fbin + 1) * PL_MASK_FREQS_PER_BIN]));

            lost_samples_meta->check_frame_desc(
                lost_samples_buf->get_frame_desc<kotekan::GenericNDArray>());

            // done
            lost_samples_buf->mark_frame_full(unique_name, frame_id);
        }
        seq_num += samples_in_dataset;
        frame_id = (frame_id + 1) % pl_mask_buf->num_frames;
    }
}
