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

/**
 * @brief Perform on CPU the equivalent of the cudaPLMaskAccumulator stage.
 *
 * An example of this stage being used can be found in
 * `config/tests/verify_cuda_pl_accum.yaml`.
 *
 * This stage computes sums over the PL mask,
 * for sub_integration_ntime time samples, and places the results in the
 * PL counts buffer.
 *
 * The raw packet loss (PL) mask is downsampled by 2 in time, 4 in frequency,
 * and 8 in element (polarization x dish).
 *
 * In memory the PL mask has its time axis split into fast (length 64 bits)
 * and coarse (length samples_per_data_set / 2 / 64 bits) axes.  Each bit in
 * the mask corresponds to 2 times (and 4 frequencies), so the length 64 bit
 * fast time axis represents 128 time samples.
 *
 * @par Buffers
 * @buffer in_buf The input packet loss mask (not expanded).
 *      @buffer_format bitmask: uint64_t, equivalently uint8_t or uint1x8_t
 *      @buffer_shape [samples_per_data_set / 128, num_local_freq / 4,
 *          num_elements / 8] or equivalently [samples_per_data_set / 128,
 *          num_local_freq / 4, num_elements / 8, 8] if the datatype is uint8_t
 *          or uint1x8_t. If the elements axis is constructed as polarization,
 *          dish pairs, its shape is taken to be [num_polarizations,
 *          num_dishes / 8]. Size of a frame is samples_per_data_set
 *          * num_local_freq * num_elements / 512 bytes.
 *      @buffer_metadata chordMetadata. time_downsampling_fpga = 128
 *
 * @buffer out_buf The output PL counts buffer
 *      @buffer_format uint64_t
 *      @buffer_shape [samples_per_data_set / sub_integration_ntime, num_local_freq,
 *          num_polarizations, num_dishes] or equivalently
 *          [samples_per_data_set / sub_integration_ntime, num_local_freq * num_elements]
 *      @buffer_metadata chordMetadata. time_downsampling_fpga = sub_integration_ntime
 *
 * @conf  num_elements         Int.  Number of feeds or (antennas x
 *      polarizations).
 * @conf num_local_freq        Int.  Number of frequencies.
 * @conf samples_per_data_set  Int.  Number of samples per frame.
 * @conf sub_integration_ntime  Int.  Time samples to integration the PL mask for.
 */
class gpuSimulatePLMaskAccumulator : public kotekan::Stage {
public:
    gpuSimulatePLMaskAccumulator(kotekan::Config& config, const std::string& unique_name,
                                 kotekan::bufferContainer& buffer_container);
    ~gpuSimulatePLMaskAccumulator();
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;

    // Config options
    const int64_t _num_polarizations;
    const int64_t _num_dishes;
    const int64_t _num_elements;
    const int64_t _num_local_freq;
    const int64_t _samples_per_data_set;
    const int64_t _sub_integration_ntime;
    const int64_t _num_integrations;
};

REGISTER_KOTEKAN_STAGE(gpuSimulatePLMaskAccumulator);

gpuSimulatePLMaskAccumulator::gpuSimulatePLMaskAccumulator(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&gpuSimulatePLMaskAccumulator::main_thread, this)),
    _num_polarizations(config.get<int64_t>(unique_name, "num_polarizations")),
    _num_dishes(config.get<int64_t>(unique_name, "num_dishes")),
    _num_elements(_num_polarizations * _num_dishes),
    _num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    _samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    _sub_integration_ntime(config.get<int64_t>(unique_name, "sub_integration_ntime")),
    _num_integrations(div_noremainder(_samples_per_data_set, _sub_integration_ntime)) {

    // Grab Buffers
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Check input sizes and buffer compatibility
    if (_samples_per_data_set % 128 != 0) {
        FATAL_ERROR("samples_per_data_set must be a multiple of 128");
    }
    if (_sub_integration_ntime % 2 != 0) {
        FATAL_ERROR("sub_integration_ntime must be even");
    }
    if (_samples_per_data_set % _sub_integration_ntime != 0) {
        FATAL_ERROR("samples_per_data_set must be a multiple of sub_integration_ntime");
    }
    if (_num_local_freq % 4 != 0) {
        FATAL_ERROR("num_local_freq must be a multiple of 4");
    }
    if (_num_dishes % 8 != 0) {
        FATAL_ERROR("num_dishes must be a multiple of 8");
    }
    if (_num_elements % 128 != 0) {
        FATAL_ERROR("num_elements (dish x pol) must be a multiple of 128");
    }

    // Make frame desc for produced buffer (this also checks the size)
    in_buf->allocate_ndarray_frame_desc<kotekan::uint1x8_t, 5>(
        "pl_mask",
        {div_noremainder(_samples_per_data_set, 128), div_noremainder(_num_local_freq, 4),
         _num_polarizations, div_noremainder(_num_dishes, 8), 64 / 8},
        {"T2hi64", "F4", "P", "D8", "T2lo64"});
    out_buf->allocate_ndarray_frame_desc<uint64_t, 4>(
        "pl_counts", {_num_integrations, _num_local_freq, _num_polarizations, _num_dishes},
        {"Tc", "F", "P", "D"});
}

gpuSimulatePLMaskAccumulator::~gpuSimulatePLMaskAccumulator() {}

void gpuSimulatePLMaskAccumulator::main_thread() {

    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    while (!stop_thread) {
        const uint64_t* pl_mask = (uint64_t*)in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (pl_mask == nullptr)
            break;
        uint64_t* pl_counts = (uint64_t*)out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (pl_counts == nullptr)
            break;

        INFO("Simulating GPU RFI S0 for {:s}[{:d}] and putting result in {:s}[{:d}]",
             in_buf->buffer_name, in_frame_id, out_buf->buffer_name, out_frame_id);

        // number of elements = number of dishes * polarizations
        const uint64_t nf = _num_local_freq;
        const uint64_t ne = _num_elements;

        const uint64_t nsub = _sub_integration_ntime;
        const uint64_t nt_int = _num_integrations;

        const uint64_t nf_pl = (_num_local_freq + 3) / 4;
        uint64_t ne_pl = _num_elements / 8;

        // array access strides in raw (downsampled) PL mask
        const uint64_t fstride_pl = ne_pl;
        const uint64_t tstride_pl = ne_pl * nf_pl;

        // array access strides in S012
        const uint64_t fstride_pl_counts = ne;
        const uint64_t tstride_pl_counts = nf * ne;

        // Set to 0 to start.
        for (uint64_t tfe = 0; tfe < tstride_pl_counts * nt_int; tfe++)
            pl_counts[tfe] = 0;

        // Looping over entries in the PL counts buffer.
        for (uint64_t t_int = 0; t_int < nt_int; t_int++) {

            // Now accumulate.
            for (uint64_t tsub = 0; tsub < nsub; tsub++) {

                // Global t index
                uint64_t t = tsub + nsub * t_int;

                for (uint64_t f = 0; f < nf; f++) {
                    for (uint64_t e = 0; e < ne; e++) {

                        uint64_t idx_pl_counts =
                            e + fstride_pl_counts * f + tstride_pl_counts * t_int;

                        uint64_t e_pl = e / 8;
                        uint64_t f_pl = f / 4;
                        uint64_t t_pl = t / 2;
                        uint64_t t_pl_hi = t_pl / 64;
                        uint64_t t_pl_lo = t_pl % 64;
                        uint64_t idx_pl = e_pl + fstride_pl * f_pl + tstride_pl * t_pl_hi;

                        // Grab the value of the PL mask
                        uint64_t pl = (pl_mask[idx_pl] >> t_pl_lo) & 1u;

                        pl_counts[idx_pl_counts] += pl;
                    } // e
                } // f
            } // tsub
        } // t_rfi

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

        meta_out->set_from_frame_desc(out_buf->get_ndarray_frame_desc());

        // Set non-NDArray things.
        meta_out->set_fpga_seq_num(meta_in->get_fpga_seq_num());
        meta_out->set_time_downsampling_fpga(
            div_noremainder(meta_in->get_time_downsampling_fpga(), 128) * _sub_integration_ntime);

        // test that things are consistent
        meta_out->check_frame_desc(out_buf->get_ndarray_frame_desc());

        INFO("Simulating GPU RFI S012 done for {:s}[{:d}] result is in {:s}[{:d}]",
             in_buf->buffer_name, in_frame_id, out_buf->buffer_name, out_frame_id);

        in_buf->mark_frame_empty(unique_name, in_frame_id++);
        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }
}
