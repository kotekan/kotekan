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
#include <bitset>     // for bitset
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
 * @brief Count lost samples from the Packet Loss (PL) Mask, assuming the mask is scalar (all
 * elements have the same packet loss).
 *
 * The raw packet loss mask is downsampled by 2 in time, 4 in frequency,
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
 *      @buffer_format int32_t
 *      @buffer_shape [samples_per_data_set / sub_integration_ntime, num_local_freq] or equivalently
 *      @buffer_metadata chordMetadata. time_downsampling_fpga = sub_integration_ntime
 *
 * @conf  num_polarizations     Int.  Number of polarizations
 * @conf  num_dishes            Int.  Number of dishes
 * @conf  num_local_freq        Int.  Number of frequencies.
 * @conf  samples_per_data_set  Int.  Number of samples per frame.
 * @conf  sub_integration_ntime Int.  Time samples to integrate the PL mask over.
 */
class CountLostPLSamplesScalar : public kotekan::Stage {
public:
    CountLostPLSamplesScalar(kotekan::Config& config, const std::string& unique_name,
                             kotekan::bufferContainer& buffer_container);
    ~CountLostPLSamplesScalar();
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

REGISTER_KOTEKAN_STAGE(CountLostPLSamplesScalar);

CountLostPLSamplesScalar::CountLostPLSamplesScalar(Config& config, const std::string& unique_name,
                                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&CountLostPLSamplesScalar::main_thread, this)),
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
    if (_sub_integration_ntime % 128 != 0) {
        FATAL_ERROR("sub_integration_ntime must be a multiple of 128");
    }
    if (_samples_per_data_set % _sub_integration_ntime != 0) {
        FATAL_ERROR("samples_per_data_set must be a multiple of sub_integration_ntime");
    }
    if (_num_dishes % 8 != 0) {
        FATAL_ERROR("num_dishes must be a multiple of 8");
    }

    // Make frame desc for produced buffer (this also checks the size)
    in_buf->allocate_ndarray_frame_desc<kotekan::uint1x8_t, 5>(
        "pl_mask",
        {div_noremainder(_samples_per_data_set, 128), (_num_local_freq + 3) / 4,
         _num_polarizations, div_noremainder(_num_dishes, 8), 64 / 8},
        {"T2hi64", "F4", "P", "D8", "T2lo64"});
    out_buf->allocate_ndarray_frame_desc<int32_t, 2>(
        "pl_lost_counts_scalar", {_num_integrations, _num_local_freq}, {"Tc", "F"});
}

CountLostPLSamplesScalar::~CountLostPLSamplesScalar() {}

void CountLostPLSamplesScalar::main_thread() {

    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    while (!stop_thread) {
        uint64_t* pl_mask = (uint64_t*)in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (pl_mask == nullptr)
            break;
        int32_t* pl_counts = (int32_t*)out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (pl_counts == nullptr)
            break;

        DEBUG("Counting lost samples in PL Mask {:s}[{:d}] and putting result in {:s}[{:d}]",
              in_buf->buffer_name, in_frame_id, out_buf->buffer_name, out_frame_id);

        // number of elements = number of dishes * polarizations
        uint64_t nf = _num_local_freq;

        uint64_t nsub_pl = div_noremainder(_sub_integration_ntime, 128);
        uint64_t nt_int = _num_integrations;

        uint64_t nf_pl = (_num_local_freq + 3) / 4;
        uint64_t ne_pl = _num_elements / 8;

        // array access strides in raw (downsampled) PL mask
        uint64_t fstride_pl = ne_pl;
        uint64_t tstride_pl = ne_pl * nf_pl;

        // Looping over entries in the pl_counts buffer.
        for (uint64_t t_int = 0; t_int < nt_int; t_int++) {

            // Initialize to 0
            for (uint64_t f = 0; f < nf; f++)
                pl_counts[f + t_int * nf] = 0;

            // Accumulate over outer (T/128) PL time axis
            for (uint64_t t_sub_pl = 0; t_sub_pl < nsub_pl; t_sub_pl++) {
                for (uint64_t f = 0; f < nf; f++) {

                    // We're ignoring the element axis of the PL Mask here, assuming its a scalar
                    // (values identical for all elements).

                    // PL Mask outer time axis index.
                    uint64_t t_pl_hi = t_sub_pl + nsub_pl * t_int;

                    // PL Mask frequency axis index
                    uint64_t f_pl = f / 4;

                    // PL mask access index
                    uint64_t pl_idx = f_pl * fstride_pl + t_pl_hi * tstride_pl;

                    // PL counts access index
                    uint64_t pl_counts_idx = f + t_int * nf;

                    // pl_mask here is a uint64, covering 128 time samples.
                    // Sum over inner time axis (the whole uint64 value) with popcount.
                    // Want to count bad samples, not good, so take complement (64 - popcount())
                    // Multiply by 2 to account for x2 downsampling. Each bit covers 2 time samples.
                    pl_counts[pl_counts_idx] += 2 * (64 - std::bitset<64>(pl_mask[pl_idx]).count());
                } // f
            } // t_sub_pl
        } // t_int

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
        meta_out->set_fpga_seq_num(meta_in->get_fpga_seq_num()); // just in case
        meta_out->set_time_downsampling_fpga(
            div_noremainder(meta_in->get_time_downsampling_fpga(), 128) * _sub_integration_ntime);

        // test that things are consistent
        meta_out->check_frame_desc(out_buf->get_ndarray_frame_desc());

        in_buf->mark_frame_empty(unique_name, in_frame_id++);
        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }
}
