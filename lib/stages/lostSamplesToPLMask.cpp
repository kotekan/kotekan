#include "lostSamplesToPLMask.hpp"

#include <assert.h>             // for assert
#include <fmt/core.h>           // for format
#include <stddef.h>             // for ptrdiff_t
#include <algorithm>            // for copy, max
#include <functional>           // for bind, function
#include <memory>               // for shared_ptr, __shared_ptr_access
#include <vector>               // for vector
#include <cstring>              // for size_t, memcpy

#include "Config.hpp"           // for Config
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "chordMetadata.hpp"    // for chordMetadata, get_chord_metadata
#include "json.hpp"             // for basic_json, json, iter_impl
#include "DataType.hpp"         // for DataType, GetType_t
#include "NDArray.hpp"          // for GenericNDArray
#include "fmt.hpp"              // for compile_string_to_view
#include "kotekanLogging.hpp"   // for FATAL_ERROR

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

lostSamplesToPLMask::lostSamplesToPLMask(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&lostSamplesToPLMask::main_thread, this)),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")) {

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
        != lost_samples_bufs.at(0)->frame_size / PL_MASK_DOWNSAMPLING_FACTOR * num_dishes
               / PL_MASK_DISHES_PER_BIN * num_polarizations / BITS_PER_BYTE * num_freq_bins)
        FATAL_ERROR("Unexpected frames sizes for pl_mask {:d} and lost_samples {:d}",
                    pl_mask_buf->frame_size, lost_samples_bufs.at(0)->frame_size);

    pl_mask_buf->allocate_ndarray_frame_desc<kotekan::GetType_t<kotekan::uint1x8>, 5>(
        "pl_mask",
        {ptrdiff_t(lost_samples_bufs.at(0)->frame_size / PL_MASK_DOWNSAMPLING_FACTOR
                   / PL_MASK_HILO_SPLIT),
         ptrdiff_t(lost_samples_bufs.size()), num_polarizations,
         num_dishes / PL_MASK_DISHES_PER_BIN,
         PL_MASK_HILO_SPLIT / BITS_PER_BYTE /* because we count uint1x8, not uint1 */},
        {"T2hi64", "F4", "P", "D8", "T2lo64"});
}

lostSamplesToPLMask::~lostSamplesToPLMask() {}

void lostSamplesToPLMask::main_thread() {

    const size_t num_freq_bins = lost_samples_bufs.size();

    while (!stop_thread) {

        uint8_t* pl_mask_frame =
            pl_mask_buf->wait_for_empty_frame(unique_name, pl_mask_buf_frame_id);
        if (pl_mask_frame == nullptr)
            return;
        pl_mask_buf->allocate_new_metadata_object(pl_mask_buf_frame_id);
        auto pl_mask_meta = get_chord_metadata(pl_mask_buf, pl_mask_buf_frame_id);

        for (size_t f = 0; f < lost_samples_bufs.size(); ++f) {
            auto lost_samples_buf = lost_samples_bufs.at(f);
            uint8_t* flag_frame =
                lost_samples_buf->wait_for_full_frame(unique_name, lost_samples_buf_frame_id);
            if (flag_frame == nullptr)
                return;

            // constant for all iterations but only set by the producer before
            // it marks the frame as full, so cannot be checked before the first
            // wait_for_full_frame()
            auto const expected_lost_samples_frame_desc = kotekan::GenericNDArray::create(
                kotekan::DataType::uint8, "lost_samples", {ptrdiff_t(lost_samples_buf->frame_size)},
                {"T"}, nullptr);
            assert(*lost_samples_buf->get_ndarray_frame_desc()
                   == *expected_lost_samples_frame_desc);

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
                    * num_polarizations * num_dishes / PL_MASK_DISHES_PER_BIN * PL_MASK_HILO_SPLIT
                    / BITS_PER_BYTE;
                for (int dbin = 0; dbin < num_dishes / PL_MASK_DISHES_PER_BIN; ++dbin) {
                    for (int polr = 0; polr < num_polarizations; ++polr) {
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
                pl_mask_meta->set_from_frame_desc(pl_mask_buf->get_ndarray_frame_desc());

                // update metadata
                pl_mask_meta->set_time_downsampling_fpga(pl_mask_meta->get_time_downsampling_fpga()
                                                         * PL_MASK_DOWNSAMPLING_FACTOR
                                                         * PL_MASK_HILO_SPLIT);

                // TODO: do I need to set frame_counter? The FEngine does.
                // pl_mask_meta->set_frame_counter(E_frame_index);
            } else {
                const auto lost_samples_coarse_freq = lost_samples_meta->get_coarse_freq();
                auto pl_mask_coarse_freq = pl_mask_meta->get_coarse_freq();
                pl_mask_coarse_freq.insert(pl_mask_coarse_freq.end(),
                                           lost_samples_coarse_freq.begin(),
                                           lost_samples_coarse_freq.end());
                pl_mask_meta->set_coarse_freq(pl_mask_coarse_freq);

                const auto lost_samples_freq_upchan_factor =
                    lost_samples_meta->get_freq_upchan_factor();
                auto pl_mask_freq_upchan_factor = pl_mask_meta->get_freq_upchan_factor();
                pl_mask_freq_upchan_factor.insert(pl_mask_freq_upchan_factor.end(),
                                                  lost_samples_freq_upchan_factor.begin(),
                                                  lost_samples_freq_upchan_factor.end());
                pl_mask_meta->set_freq_upchan_factor(pl_mask_freq_upchan_factor);

                const auto lost_samples_freq_upchan_index =
                    lost_samples_meta->get_freq_upchan_index();
                auto pl_mask_freq_upchan_index = pl_mask_meta->get_freq_upchan_index();
                pl_mask_freq_upchan_index.insert(pl_mask_freq_upchan_index.end(),
                                                 lost_samples_freq_upchan_index.begin(),
                                                 lost_samples_freq_upchan_index.end());
                pl_mask_meta->set_freq_upchan_index(pl_mask_freq_upchan_index);
            }

            lost_samples_buf->mark_frame_empty(unique_name, lost_samples_buf_frame_id);
        }
        lost_samples_buf_frame_id =
            (lost_samples_buf_frame_id + 1) % lost_samples_bufs.at(0)->num_frames;

        pl_mask_meta->check_frame_desc(pl_mask_buf->get_ndarray_frame_desc());
        pl_mask_buf->mark_frame_full(unique_name, pl_mask_buf_frame_id);
        pl_mask_buf_frame_id = (pl_mask_buf_frame_id + 1) % pl_mask_buf->num_frames;
    }
}
