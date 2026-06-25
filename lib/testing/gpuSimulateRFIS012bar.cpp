#include "gpuSimulateRFIS012bar.hpp"

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType, GetType
#include "N2Util.hpp"          // for frameID
#include "NDArray.hpp"         // for NDArray, GenericNDArray, Config
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

REGISTER_KOTEKAN_STAGE(gpuSimulateRFIS012bar);

gpuSimulateRFIS012bar::gpuSimulateRFIS012bar(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&gpuSimulateRFIS012bar::main_thread, this)),
    _num_polarizations(config.get<int64_t>(unique_name, "num_polarizations")),
    _num_dishes(config.get<int64_t>(unique_name, "num_dishes")),
    _num_elements(_num_polarizations * _num_dishes),
    _num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    _samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    _rfi_downsampling_factor(config.get<int64_t>(unique_name, "rfi_downsampling_factor")),
    _rfi_second_downsampling_factor(
        config.get<int64_t>(unique_name, "rfi_second_downsampling_factor")) {

    // Grab buffers
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // input validation
    if (_samples_per_data_set % _rfi_downsampling_factor != 0) {
        FATAL_ERROR("samples_per_data_set must be a multiple of rfi_downsampling_factor");
    }
    assert(_samples_per_data_set % _rfi_downsampling_factor == 0);

    if ((_samples_per_data_set / _rfi_downsampling_factor) % _rfi_second_downsampling_factor != 0) {
        FATAL_ERROR("samples_per_data_set / rfi_downsampling_factor must be a multiple of "
                    "rfi_second_downsampling_factor");
    }
    assert((_samples_per_data_set / _rfi_downsampling_factor) % _rfi_second_downsampling_factor
           == 0);

    in_buf->require_frame_desc(kotekan::NDArray<uint64_t, 5>::describe(
        "S012",
        {_samples_per_data_set / _rfi_downsampling_factor, _num_local_freq, 3, _num_polarizations,
         _num_dishes},
        {"Trfi", "F", "S", "P", "D"}, {_rfi_downsampling_factor, 1, 1, 1, 1}));

    int64_t nt = _samples_per_data_set / _rfi_downsampling_factor / _rfi_second_downsampling_factor;
    out_buf->require_frame_desc(kotekan::NDArray<uint64_t, 5>::describe(
        "S012bar", {nt, _num_local_freq, 3, _num_polarizations, _num_dishes},
        {"Trfibar", "F", "S", "P", "D"},
        {_rfi_downsampling_factor * _rfi_second_downsampling_factor, 1, 1, 1, 1}));
}

gpuSimulateRFIS012bar::~gpuSimulateRFIS012bar() {}

void gpuSimulateRFIS012bar::main_thread() {

    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    while (!stop_thread) {
        uint64_t* rfi_in = (uint64_t*)in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (rfi_in == nullptr)
            break;
        uint64_t* rfi_out = (uint64_t*)out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (rfi_out == nullptr)
            break;

        INFO("Simulating GPU RFI S012 for {:s}[{:d}] and putting result in {:s}[{:d}]",
             in_buf->buffer_name, in_frame_id, out_buf->buffer_name, out_frame_id);

        // Fetch input metadata
        const std::shared_ptr<const metadataObject> mc_in = in_buf->get_metadata(in_frame_id);
        if (!mc_in) {
            FATAL_ERROR("Buffer {:s} frame {:d} had no metadata", in_buf->buffer_name, in_frame_id);
        }
        assert(mc_in);
        if (!metadata_is_chord(mc_in)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata", in_buf->buffer_name,
                        in_frame_id);
        }
        assert(metadata_is_chord(mc_in));

        const std::shared_ptr<const chordMetadata> meta_in = get_chord_metadata(mc_in);

        // Create output metadata
        out_buf->allocate_new_metadata_object(out_frame_id);
        const std::shared_ptr<metadataObject> mc = out_buf->get_metadata(out_frame_id);
        if (!mc) {
            FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata", out_buf->buffer_name,
                        out_frame_id);
        }
        assert(mc);
        if (!metadata_is_chord(mc)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata", out_buf->buffer_name,
                        out_frame_id);
        }
        assert(metadata_is_chord(mc));
        const std::shared_ptr<chordMetadata> meta_out = get_chord_metadata(mc);
        assert(meta_out);

        // Start with a copy
        meta_out->deepCopy(meta_in);
        meta_out->set_from_frame_desc(out_buf->get_frame_desc<kotekan::GenericNDArray>());

        // test that things are consistent
        meta_out->check_frame_desc(out_buf->get_frame_desc<kotekan::GenericNDArray>());

        // Set non-NDArray things.
        meta_out->set_fpga_seq_num(meta_in->get_fpga_seq_num());
        meta_out->set_time_downsampling_fpga(meta_in->get_time_downsampling_fpga()
                                             * _rfi_second_downsampling_factor);

        // number of elements = number of dishes * polarizations
        uint64_t nt_in = _samples_per_data_set / _rfi_downsampling_factor;
        uint64_t nf = _num_local_freq;
        uint64_t ne = _num_elements;

        uint64_t nt_out = nt_in / _rfi_second_downsampling_factor;

        // array access strides in rfi_s012 buffers
        uint64_t tstride = 3 * nf * ne;

        // Set to 0 to start.
        for (uint64_t tfse = 0; tfse < tstride * nt_out; tfse++)
            rfi_out[tfse] = 0;

        // Looping over entries in the input buffer.
        for (uint64_t t_in = 0; t_in < nt_in; t_in++) {

            // downsampled time in output buffer
            uint64_t t_out = t_in / _rfi_second_downsampling_factor;

            // accumulate!
            for (uint64_t fse = 0; fse < tstride; fse++)
                rfi_out[t_out * tstride + fse] += rfi_in[t_in * tstride + fse];

        } // t_in


        INFO("Simulating GPU RFI S012 done for {:s}[{:d}] result is in {:s}[{:d}]",
             in_buf->buffer_name, in_frame_id, out_buf->buffer_name, out_frame_id);

        in_buf->mark_frame_empty(unique_name, in_frame_id++);
        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }
}
