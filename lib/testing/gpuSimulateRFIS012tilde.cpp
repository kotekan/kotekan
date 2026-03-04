#include "gpuSimulateRFIS012tilde.hpp"

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType, GetType
#include "N2Util.hpp"          // for frameID
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, metadata_is_chord, get_chord_metadata, CHO...
#include "div.hpp"             // for div_noremainder
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
using kotekan::div_noremainder;
using kotekan::Stage;
using N2::frameID;

REGISTER_KOTEKAN_STAGE(gpuSimulateRFIS012tilde);

gpuSimulateRFIS012tilde::gpuSimulateRFIS012tilde(Config& config, const std::string& unique_name,
                                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&gpuSimulateRFIS012tilde::main_thread, this)),
    _num_polarizations(config.get<int64_t>(unique_name, "num_polarizations")),
    _num_dishes(config.get<int64_t>(unique_name, "num_dishes")),
    _num_elements(_num_polarizations * _num_dishes),
    _num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    _samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    _rfi_downsampling_factor(config.get<int64_t>(unique_name, "rfi_downsampling_factor")) {

    // Grab buffers.
    in_rfi_s012_buf = get_buffer("in_rfi_s012_buf");
    in_rfi_s012_buf->register_consumer(unique_name);
    in_bf_mask_buf = get_buffer("in_bf_mask_buf");
    in_bf_mask_buf->register_consumer(unique_name);

    out_rfi_s012tilde_buf = get_buffer("out_rfi_s012tilde_buf");
    out_rfi_s012tilde_buf->register_producer(unique_name);

    if (_samples_per_data_set % _rfi_downsampling_factor != 0) {
        FATAL_ERROR("samples_per_data_set must be a multiple of rfi_downsampling_factor");
    }
    assert(_samples_per_data_set % _rfi_downsampling_factor == 0);

    size_t bf_mask_size = _num_elements;
    size_t rfi_s012_size = (_samples_per_data_set / _rfi_downsampling_factor) * _num_local_freq * 3 * _num_elements * sizeof(uint64_t);
    
    if (in_rfi_s012_buf->frame_size != rfi_s012_size) {
        FATAL_ERROR("in_rfi_s012_buf ({:s}) has frame size: {:d}, expected {:d}",
                in_rfi_s012_buf->buffer_name, in_rfi_s012_buf->frame_size, rfi_s012_size);
    }
    assert(in_rfi_s012_buf->frame_size == rfi_s012_size);

    if (in_bf_mask_buf->frame_size != bf_mask_size) {
        FATAL_ERROR("in_bf_mask_buf ({:s}) has frame size: {:d}, expected {:d}",
                in_bf_mask_buf->buffer_name, in_bf_mask_buf->frame_size, bf_mask_size);
    }
    assert(in_bf_mask_buf->frame_size == bf_mask_size);

    int64_t nt = _samples_per_data_set / _rfi_downsampling_factor;
    int64_t nf = _num_local_freq;
    out_rfi_s012tilde_buf->allocate_ndarray_frame_desc<uint64_t, 3>("S012tilde", {nt, nf, 3},
                                                                    {"Trfi", "F", "S"});
}

gpuSimulateRFIS012tilde::~gpuSimulateRFIS012tilde() {}

void gpuSimulateRFIS012tilde::main_thread() {

    frameID in_bf_mask_frame_id(in_bf_mask_buf);
    frameID in_rfi_s012_frame_id(in_rfi_s012_buf);
    frameID out_rfi_s012tilde_frame_id(out_rfi_s012tilde_buf);

    // BF mask (for now) is not updated in time. Only read once.
    uint8_t* bf_mask =
        (uint8_t*)in_bf_mask_buf->wait_for_full_frame(unique_name, in_bf_mask_frame_id);
    if (bf_mask == nullptr)
        return;

    for (int64_t e = 0; e < _num_elements; e++) {
        INFO("BF[{:03d}]: {:08b}", e, bf_mask[e]);
    }

    while (!stop_thread) {
        // Grab in/out buffers
        uint64_t* rfi_s012 =
            (uint64_t*)in_rfi_s012_buf->wait_for_full_frame(unique_name, in_rfi_s012_frame_id);
        if (rfi_s012 == nullptr)
            break;
        uint64_t* rfi_s012tilde = (uint64_t*)out_rfi_s012tilde_buf->wait_for_empty_frame(
            unique_name, out_rfi_s012tilde_frame_id);
        if (rfi_s012tilde == nullptr)
            break;

        INFO("Simulating GPU RFI S012 for {:s}[{:d}], {:s}[{:d}] and putting result in {:s}[{:d}]",
             in_bf_mask_buf->buffer_name, in_bf_mask_frame_id, in_rfi_s012_buf->buffer_name,
             in_rfi_s012_frame_id, out_rfi_s012tilde_buf->buffer_name, out_rfi_s012tilde_frame_id);

        // Dimension sizes
        uint64_t nt = _samples_per_data_set / _rfi_downsampling_factor;
        uint64_t nf = _num_local_freq;
        uint64_t ne = _num_elements;

        // Looping over time, freq, and S012 index. (spectator indices)
        for (uint64_t tfs = 0; tfs < 3 * nf * nt; tfs++) {

            rfi_s012tilde[tfs] = 0;

            // Now accumulate.
            for (uint64_t e = 0; e < ne; e++) {
                if (bf_mask[e])
                    rfi_s012tilde[tfs] += rfi_s012[tfs * ne + e];
            } // e
        } // tfs

        // Fetch input metadata
        const std::shared_ptr<const metadataObject> mc_in =
            in_rfi_s012_buf->get_metadata(in_rfi_s012_frame_id);
        if (!mc_in) {
            FATAL_ERROR("Buffer {:s} frame {:d} had no metadata", in_rfi_s012_buf->buffer_name,
                        in_rfi_s012_frame_id);
        }
        assert(mc_in);
        if (!metadata_is_chord(mc_in)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        in_rfi_s012_buf->buffer_name, in_rfi_s012_frame_id);
        }
        assert(metadata_is_chord(mc_in));

        const std::shared_ptr<const chordMetadata> meta_in = get_chord_metadata(mc_in);

        // Create output metadata
        out_rfi_s012tilde_buf->allocate_new_metadata_object(out_rfi_s012tilde_frame_id);
        const std::shared_ptr<metadataObject> mc =
            out_rfi_s012tilde_buf->get_metadata(out_rfi_s012tilde_frame_id);
        if (!mc) {
            FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata",
                        out_rfi_s012tilde_buf->buffer_name, out_rfi_s012tilde_frame_id);
        }
        assert(mc);
        if (!metadata_is_chord(mc)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        out_rfi_s012tilde_buf->buffer_name, out_rfi_s012tilde_frame_id);
        }
        assert(metadata_is_chord(mc));
        const std::shared_ptr<chordMetadata> meta_out = get_chord_metadata(mc);
        assert(meta_out);

        // Start with a copy
        meta_out->deepCopy(meta_in);

        meta_out->set_from_frame_desc(out_rfi_s012tilde_buf->get_ndarray_frame_desc());

        // test that things are consistent
        meta_out->check_frame_desc(out_rfi_s012tilde_buf->get_ndarray_frame_desc());

        // Set non-NDArray things.
        meta_out->set_fpga_seq_num(meta_in->get_fpga_seq_num());

        INFO("Simulating GPU RFI S012tilde done for {:s}[{:d}]+{:s}[{:d}] result is in {:s}[{:d}]",
             in_bf_mask_buf->buffer_name, in_bf_mask_frame_id, in_rfi_s012_buf->buffer_name,
             in_rfi_s012_frame_id, out_rfi_s012tilde_buf->buffer_name, out_rfi_s012tilde_frame_id);

        in_rfi_s012_buf->mark_frame_empty(unique_name, in_rfi_s012_frame_id++);
        out_rfi_s012tilde_buf->mark_frame_full(unique_name, out_rfi_s012tilde_frame_id++);
    }

    // Only release the bf_mask when we're done.
    in_bf_mask_buf->mark_frame_empty(unique_name, in_bf_mask_frame_id);
}
