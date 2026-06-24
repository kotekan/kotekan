#include "testXpose.hpp"

#include "Config.hpp"          // for Config
#include "Metadata.hpp"        // for GenericNDArray
#include "NDArray.hpp"         // for GenericNDArray, Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for get_chord_metadata, chordMetadata
#include "visUtil.hpp"         // for frameID

#include "json.hpp" // for basic_json, json, iter_impl

#include <algorithm>  // for iota
#include <assert.h>   // for assert
#include <cstdint>    // for int32_t
#include <functional> // for bind, function
#include <memory>     // for __shared_ptr_access, shared_ptr
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(testXpose);

// various constants defining the array layout of the package loss mask
#define PL_MASK_DOWNSAMPLING_FACTOR 2
#define PL_MASK_HILO_SPLIT 64
#define PL_MASK_DISHES_PER_BIN 8
#define PL_MASK_FREQS_PER_BIN 4

#define BITS_PER_BYTE 8

// CHIME parameters, actually
#define NUM_DISHES 1024
#define NUM_POLARIZATIONS 2

testXpose::testXpose(Config& config, const std::string& unique_name,
                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testXpose::main_thread, this)),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    num_times(config.get<int>(unique_name, "num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_xposed_frequencies(config.get<int>(unique_name, "num_xposed_frequencies")) {

    out_xposed_buf = get_buffer("out_xposed_buf");
    out_xposed_buf->register_producer(unique_name);
    if (out_xposed_buf->frame_size
        != num_times * num_xposed_frequencies * num_polarizations * num_dishes * sizeof(uint8_t))
        FATAL_ERROR("Unexpected frames sizes for out_xposed_buf, expected {:d} received {:d}",
                    out_xposed_buf->frame_size,
                    num_times * num_xposed_frequencies * num_polarizations * num_dishes
                        * sizeof(uint8_t));
    out_xposed_buf->require_frame_desc(kotekan::GenericNDArray::describe(
        kotekan::int4x2_swapped_withoffset, "E",
        {num_times, num_xposed_frequencies, num_polarizations, num_dishes}, {"T", "F", "P", "D"},
        {1, 1, 1, 1}));

    scatter_indices_buf = get_buffer("scatter_indices_buf");
    scatter_indices_buf->register_producer(unique_name);
    if (scatter_indices_buf->frame_size != num_polarizations * num_dishes * sizeof(int32_t))
        FATAL_ERROR("Unexpected frames sizes for scatter_indeces, expected {:d} received {:d}",
                    scatter_indices_buf->frame_size,
                    num_polarizations * num_dishes * sizeof(int32_t));
    // TODO: this is not quite correct. Really if going to cylinder order
    // the array is {4,2,256} {"C", "P", "D"}
    scatter_indices_buf->require_frame_desc(kotekan::GenericNDArray::describe(
        kotekan::int32, "scatter_indices", {num_polarizations, num_dishes}, {"P", "D"}, {1, 1}));

    // Register as producer for all xpose2048 input buffers
    json bufs = config.get_value(unique_name, "out_bufs");
    for (json::iterator it = bufs.begin(); it != bufs.end(); ++it) {
        Buffer* buf = buffer_container.get_buffer(it.value().get<std::string>());
        out_bufs.push_back(buf);
        buf->register_producer(unique_name);
        buf->require_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::int4x2_swapped_withoffset, "E",
            {num_frequencies, num_times, num_polarizations * num_dishes}, {"F", "T", "E"},
            {1, 1, 1}));
        if (buf->frame_size
            != num_polarizations * num_dishes * num_times * num_frequencies * sizeof(uint8_t))
            FATAL_ERROR("Input samples bufferer {:s} has unexpected size {:d} instead of {:d}",
                        it.value().get<std::string>(), buf->frame_size,
                        num_polarizations * num_dishes * num_times * num_frequencies
                            * sizeof(uint8_t));
        if (buf->num_frames != out_bufs.at(0)->num_frames)
            FATAL_ERROR("Input samples buffers have different number of frames: {:d} != {:d}",
                        buf->num_frames, out_bufs.at(0)->num_frames);
    }

    const int num_freq_bins = int(out_bufs.size());
    if (num_freq_bins * num_frequencies != num_xposed_frequencies)
        FATAL_ERROR("Number of frequencies in input buffers and expected output buffer do not "
                    "match: {:d} != {:d}",
                    num_freq_bins * num_frequencies, num_xposed_frequencies);

    assert(num_frequencies == 1
           && "More than 1 frequency in CPU buffer is strange and may not be supported");
}

testXpose::~testXpose() {}

void testXpose::main_thread() {
    const int num_elements = num_polarizations * num_dishes;

    // some made up scattering that shuffles everyting one element forward (and
    // wraps around)
    std::vector<int32_t> scatter_indices(num_elements);
    std::iota(scatter_indices.begin(), scatter_indices.end(), 1);
    scatter_indices.back() = 0;

    // this is only done once
    const frameID scatter_indices_id(scatter_indices_buf);
    int32_t* scatter_indices_frame =
        (int32_t*)scatter_indices_buf->wait_for_empty_frame(unique_name, scatter_indices_id);
    if (scatter_indices_frame == nullptr)
        return;
    std::copy(scatter_indices.begin(), scatter_indices.end(), scatter_indices_frame);
    scatter_indices_buf->allocate_new_metadata_object(scatter_indices_id);
    auto scatter_indices_meta = get_chord_metadata(scatter_indices_buf, scatter_indices_id);
    scatter_indices_meta->set_fpga_seq_num(0);
    scatter_indices_meta->set_from_frame_desc(
        scatter_indices_buf->get_frame_desc<kotekan::GenericNDArray>());
    scatter_indices_meta->check_frame_desc(
        scatter_indices_buf->get_frame_desc<kotekan::GenericNDArray>());
    scatter_indices_buf->mark_frame_full(unique_name, scatter_indices_id);

    // for each time
    int seq_num = 0;
    frameID out_xposed_id(out_xposed_buf);
    frameID out_id(out_bufs.at(0));
    while (!stop_thread) {
        // we can only store 15^2-1 values excluding the poison value of a 0
        // nibble and some "neutral" value (we pick 0xff), so we pick a couple of
        // locations then shuffle them for each time, such that for time t,
        // frequency f and value v we put it into element
        // e = v * 17 + 2*f + t (mod num_elements)
        // at each time step at each frequency.
        // Note that this may overwrite values that were already set, so the
        // order of the loops below matters.
        uint8_t* out_xposed_frame =
            out_xposed_buf->wait_for_empty_frame(unique_name, out_xposed_id);
        if (out_xposed_frame == nullptr)
            return;

        // all unset elements remain at 0xff
        std::memset(out_xposed_frame, 0xff, out_xposed_buf->frame_size);

        for (int buf_num = 0; buf_num < (int)out_bufs.size(); ++buf_num) {
            uint8_t* out_frame = out_bufs.at(buf_num)->wait_for_empty_frame(unique_name, out_id);
            if (out_frame == nullptr)
                return;

            // all unset elements remain at 0xff
            std::memset(out_frame, 0xff, out_bufs.at(buf_num)->frame_size);

            for (int f = 0; f < num_frequencies; ++f) {
                const int f_xposed = buf_num * num_frequencies + f;
                assert(f_xposed < num_xposed_frequencies);
                for (int t = 0; t < num_times; ++t) {
                    const int block_idx = f * num_times * num_elements + t * num_elements;
                    const int xposed_block_idx =
                        t * num_xposed_frequencies * num_elements + f_xposed * num_elements;
                    for (int v = 0; v < 15 * 15 - 1; ++v) {
                        const int el_idx =
                            (v * 17 + 2 * f_xposed + t + seq_num * num_times) % num_elements;
                        const int idx = block_idx + el_idx;
                        assert((unsigned int)idx < out_bufs.at(buf_num)->frame_size);

                        // build valid nibbles
                        uint8_t val = (((v / 15) + 1) << 4) | (((v % 15) + 1) << 0);

                        // untransposed data
                        out_frame[idx] = val;

                        // write transposed result, note that this replicates the
                        // scattering_indices knowledge
                        const int xposed_idx = xposed_block_idx + (el_idx + 1) % num_elements;
                        assert((unsigned int)xposed_idx < out_xposed_buf->frame_size);
                        out_xposed_frame[xposed_idx] = val;
                    }
                }
            }
            out_bufs.at(buf_num)->allocate_new_metadata_object(out_id);
            auto out_meta = get_chord_metadata(out_bufs.at(buf_num), out_id);
            out_meta->set_fpga_seq_num(seq_num * num_times);
            out_meta->set_freq_upchan_factor(std::vector<int>(num_frequencies, 1));
            out_meta->set_freq_upchan_index(std::vector<int>(num_frequencies, 0));
            out_meta->set_time_downsampling_fpga(1);
            std::vector<int> coarse_freq(num_frequencies);
            std::iota(coarse_freq.begin(), coarse_freq.end(), buf_num * num_frequencies);
            out_meta->set_coarse_freq(coarse_freq);
            out_meta->set_from_frame_desc(
                out_bufs.at(buf_num)->get_frame_desc<kotekan::GenericNDArray>());
            out_meta->check_frame_desc(
                out_bufs.at(buf_num)->get_frame_desc<kotekan::GenericNDArray>());
            out_bufs.at(buf_num)->mark_frame_full(unique_name, out_id);
        }

        out_xposed_buf->allocate_new_metadata_object(out_xposed_id);
        auto out_xposed_meta = get_chord_metadata(out_xposed_buf, out_xposed_id);
        out_xposed_meta->set_fpga_seq_num(seq_num * num_times);
        out_xposed_meta->set_freq_upchan_factor(std::vector<int>(num_xposed_frequencies, 1));
        out_xposed_meta->set_freq_upchan_index(std::vector<int>(num_xposed_frequencies, 0));
        out_xposed_meta->set_time_downsampling_fpga(1);
        std::vector<int> coarse_freq(num_xposed_frequencies);
        std::iota(coarse_freq.begin(), coarse_freq.end(), 0);
        out_xposed_meta->set_coarse_freq(coarse_freq);
        out_xposed_meta->set_from_frame_desc(
            out_xposed_buf->get_frame_desc<kotekan::GenericNDArray>());
        out_xposed_meta->check_frame_desc(
            out_xposed_buf->get_frame_desc<kotekan::GenericNDArray>());
        out_xposed_buf->mark_frame_full(unique_name, out_xposed_id);

        out_xposed_id += 1;
        out_id += 1;

        seq_num += 1;
    }
}
