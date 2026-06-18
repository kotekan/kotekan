#include <stddef.h>             // for size_t
#include <stdint.h>             // for int64_t, uint64_t, uint8_t, int32_t
#include <functional>           // for bind, function, placeholders
#include <memory>               // for allocator, shared_ptr, __shared_ptr_access
#include <vector>               // for vector
#include <string>               // for basic_string, char_traits, operator!=, string

#include "Config.hpp"           // for Config
#include "N2Util.hpp"           // for frameID, modulo
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "chordMetadata.hpp"    // for chordMetadata, get_chord_metadata
#include "div.hpp"              // for div_noremainder
#include "kotekanLogging.hpp"   // for FATAL_ERROR, DEBUG, INFO
#include "fmt.hpp"              // for compile_string_to_view
#include "DataType.hpp"         // for DataType
#include "Stage.hpp"            // for Stage


using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::div_noremainder;
using kotekan::Stage;
using N2::frameID;

/**
 * @class RfiMaskSum
 * @brief Accumulate the raw rate RFI mask into counts per correlator frame. This counts the bad
 * samples, e.g. samples for which RFIMask = 0.  The count will be in [0, sub_integration_ntime]
 *
 * num_integrations := samples_per_data_set / sub_integration_ntime
 * num_freq := num_local_freq
 *
 * @par Buffers
 * @buffer  in_buf      RFImask buffer from n2k.
 *         @buffer_format   NDArray uint1x8 [samples_per_dataset / 128 / 8, num_freq, 128]
 *         @buffer_metadata chordMetadata
 * @buffer  out_buf     RFI count of bad samples.
 *      @buffer_format      NDArray int32 [num_integrations, num_freq]
 *      @buffer_metadata    chordMetadata
 *
 * @conf    num_local_freq              int64_t Number of frequencies in
 *                                          buffers, required.
 * @conf    samples_per_data_set        int64_t Total number of time samples covered by each
 *                                          input frame. nt_outer in n2k.
 * @conf    sub_integration_ntime       int64_t Number of time samples integrated in each
 *                                          entry in correlation and counts buffers.  n2_inner
 *                                          in n2k.
 * @conf    rfi_downsampling_factor     int64_t Number of time samples used to compute
 *                                          RFImask.  The values in the RFIMask buffer are
 *                                          repeated this many times.
 *
 * @author  Geoff Ryan
 */
class RfiMaskSum : public kotekan::Stage {
public:
    RfiMaskSum(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    ~RfiMaskSum() = default;

    /**
     * @brief The main thread function for RfiMaskSum.
     *
     * This function is responsible for the main logic of the RfiMaskSum class.
     */
    void main_thread() override;

private:
    // Buffers to read/write
    Buffer* in_buf;  /// Buffer containing input rfimask
    Buffer* out_buf; /// Output for the RFI counts

    // Parameters saved from the config files
    const int64_t _num_local_freq;

    const int64_t _samples_per_data_set;
    const int64_t _sub_integration_ntime;
    const int64_t _rfi_downsampling_factor; ///< Downsampling factor for RFI mask
    const int64_t _num_integrations;

    void accumulate_rfimask_in_sample(const uint8_t* rfimask, int64_t t_vis,
                                      int32_t* rfimask_count);
};

REGISTER_KOTEKAN_STAGE(RfiMaskSum);


RfiMaskSum::RfiMaskSum(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&RfiMaskSum::main_thread, this)),
    _num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    _samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    _sub_integration_ntime(config.get<int64_t>(unique_name, "sub_integration_ntime")),
    _rfi_downsampling_factor(config.get<int64_t>(unique_name, "rfi_downsampling_factor")),
    _num_integrations(div_noremainder(_samples_per_data_set, _sub_integration_ntime)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Ensure outgoing buffer is of type N2
    if (out_buf->buffer_type != "standard")
        FATAL_ERROR("RfiMaskSum out_buf ({:s}) is not of type standard.", out_buf->buffer_name);

    // Sanity checks on initialization
    {
        // number of frequencies in incoming frames from n2k
        if (!(_num_local_freq > 0))
            FATAL_ERROR("num_local_freq is not positive: {:d}", _num_local_freq);

        // sampling information
        if (!(_samples_per_data_set > 0))
            FATAL_ERROR("samples_per_data_set is not positve: {:d}", _samples_per_data_set);
        if (!(_sub_integration_ntime > 0))
            FATAL_ERROR("sub_integration_ntime is not positve: {:d}", _sub_integration_ntime);
        if (!(_samples_per_data_set % _sub_integration_ntime == 0))
            FATAL_ERROR(
                "samples_per_data_set ({:d}) is not a multiple of sub_integration_ntime ({:d})",
                _samples_per_data_set, _sub_integration_ntime);
        if (!(_samples_per_data_set % 1024 == 0))
            FATAL_ERROR("samples_per_data_set ({:d}) is not a multiple of 1024",
                        _samples_per_data_set);

        // RFI downsampling factor checks
        if (!(_rfi_downsampling_factor > 0))
            FATAL_ERROR("rfi_downsampling_factor is not positive: {:d}", _rfi_downsampling_factor);

        if (!(_sub_integration_ntime % _rfi_downsampling_factor == 0)) {
            FATAL_ERROR("sub_integration_ntime {} is not a multiple of rfi_downsampling_factor {}",
                        _sub_integration_ntime, _rfi_downsampling_factor);
        }
    }

    // Ensure incoming buffer frame sizes are correct
    size_t in_rfimask_frame_size = _num_local_freq * _samples_per_data_set / 8;

    if (in_buf->frame_size != in_rfimask_frame_size)
        FATAL_ERROR("RfiMaskSum in_buf ({:s}) has frame size {:d}. Expected {:d}.",
                    in_buf->buffer_name, in_buf->frame_size, in_rfimask_frame_size);

    in_buf->allocate_ndarray_frame_desc(
        kotekan::uint1x8, "RFImask",
        {div_noremainder(_samples_per_data_set, 1024), _num_local_freq, 1024 / 8},
        {"T8hi128", "F", "T8lo128"});
    out_buf->allocate_ndarray_frame_desc(kotekan::int32, "RFImask_counts",
                                         {_num_integrations, _num_local_freq}, {"Tc", "F"});
}

void RfiMaskSum::main_thread() {

    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    INFO("Accumulating RFIMask from {:s}[{:d}] putting result in {:s}[{:d}]", in_buf->buffer_name,
         in_frame_id, out_buf->buffer_name, out_frame_id);

    while (!stop_thread) {

        // Fetch a new frame and get its sequence id
        DEBUG("Waiting for new RFImask frame {:s}[{:d}].", in_buf->buffer_name, in_frame_id);
        const uint8_t* rfimask = (uint8_t*)in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (rfimask == nullptr)
            break;
        DEBUG("Waiting for new RFImask_count frame {:s}[{:d}].", out_buf->buffer_name,
              out_frame_id);
        int32_t* rfimask_count = (int32_t*)out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (rfimask_count == nullptr)
            break;

        DEBUG("Accumulating RFI mask");
        // Accumulate each visibility sample in the in_frame
        // t_outer
        for (int64_t t_int = 0; t_int < _num_integrations; t_int++) {

            accumulate_rfimask_in_sample(rfimask, t_int, rfimask_count + t_int * _num_local_freq);
        } // t_int

        DEBUG("Setting Metadata");
        // Get metadata for all incoming frames.
        std::shared_ptr<chordMetadata> in_meta = get_chord_metadata(in_buf, in_frame_id);

        out_buf->allocate_new_metadata_object(out_frame_id);
        const std::shared_ptr<chordMetadata> out_meta = get_chord_metadata(out_buf, out_frame_id);

        out_meta->deepCopy(in_meta);
        out_meta->set_from_frame_desc(out_buf->get_ndarray_frame_desc());
        out_meta->set_time_downsampling_fpga(
            div_noremainder(in_meta->get_time_downsampling_fpga(), 1024) * _sub_integration_ntime);

        out_meta->check_frame_desc(out_buf->get_ndarray_frame_desc());

        // Advance to the next frame
        in_buf->mark_frame_empty(unique_name, in_frame_id++);
        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }
}

void RfiMaskSum::accumulate_rfimask_in_sample(const uint8_t* rfimask, int64_t t_vis,
                                              int32_t* rfimask_count) {
    // Sum the RFI mask
    //
    // Each RFI mask frame has structure:
    //
    //      rfimask[Thi1024, F, Tlo1024/sizeof(unit-size)]
    //
    // In particular:
    //
    //      int1 rfimask[samples_per_data_set / 1024, _num_local_freq, 1024]
    //      int8 rfimask[samples_per_data_set / 1024, _num_local_freq, 128]
    //
    // Raw time sample index at start of vis sample (n2k integration):
    //
    //      t = vis_samp_n * n_fpga_samples_per_n2k_correlation
    //
    // For an int8 rfimask, for a raw index t:
    //
    //      thi = t / 1024
    //      tlo = (t % 1024) / 8
    //      tbit = t % 8
    //
    //      t = tbit + 8*tlo + 1024*thi
    //
    // Furthermore, the RFI mask is only computed every rfi_downsampling_factor
    // samples, so t can be incremented by rfi_downsampling_factor.

    const uint64_t rfi_stride_f = 128; // = rfimask_fast_time_len / bits_per_entry = 1024 / 8;
    const uint64_t rfi_stride_thi = rfi_stride_f * _num_local_freq;

    for (int64_t f = 0; f < _num_local_freq; ++f) {
        rfimask_count[f] = 0;
        for (int64_t t = t_vis * _sub_integration_ntime; t < (t_vis + 1) * _sub_integration_ntime;
             t += _rfi_downsampling_factor) {

            // Casting to a uint64_t here is a micro-optimization,
            // the assembly is slighty simpler if the compliler knows
            // the numerator is non-negative.
            const int64_t thi = ((uint64_t)t) / 1024;
            const int64_t tlo = (((uint64_t)t) % 1024) / 8;
            const int64_t tbit = ((uint64_t)t) % 8;

            const int64_t idx = thi * rfi_stride_thi + f * rfi_stride_f + tlo;

            const uint8_t rfimask_val = (rfimask[idx] >> tbit) & 0x1;

            rfimask_count[f] += (1 - rfimask_val) * _rfi_downsampling_factor;
        } // t

    } // f
}
