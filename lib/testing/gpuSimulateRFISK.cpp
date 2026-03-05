#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType, GetType
#include "N2Util.hpp"          // for frameID
#include "Stage.hpp"           // for Stage
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
#include <string>     // for string
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::div_noremainder;
using kotekan::Stage;
using N2::frameID;

class gpuSimulateRFISK : public kotekan::Stage {
public:
    gpuSimulateRFISK(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container);
    ~gpuSimulateRFISK();
    void main_thread() override;

private:
    const int64_t _num_polarizations;
    const int64_t _num_dishes;
    const int64_t _num_elements;
    const int64_t _num_local_freq;
    const int64_t _samples_per_data_set;
    const int64_t _rfi_downsampling_factor;
    const bool _bar_mode;

    Buffer* in_bf_mask_buf;
    Buffer* in_rfi_s012_buf;
    Buffer* out_rfi_sk_buf;
    Buffer* out_rfi_sktilde_buf;
    Buffer* out_rfi_mask_buf;
};

REGISTER_KOTEKAN_STAGE(gpuSimulateRFISK);

gpuSimulateRFISK::gpuSimulateRFISK(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&gpuSimulateRFISK::main_thread, this)),
    _num_polarizations(config.get<int64_t>(unique_name, "num_polarizations")),
    _num_dishes(config.get<int64_t>(unique_name, "num_dishes")),
    _num_elements(_num_polarizations * _num_dishes),
    _num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    _samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    _rfi_downsampling_factor(config.get<int64_t>(unique_name, "rfi_total_downsampling_factor")),
    _bar_mode(config.get<bool>(unique_name, "bar_mode")) {

    // Grab Buffers
    in_rfi_s012_buf = get_buffer("in_rfi_s012_buf");
    in_rfi_s012_buf->register_consumer(unique_name);

    in_bf_mask_buf = get_buffer("in_bf_mask_buf");
    in_bf_mask_buf->register_consumer(unique_name);

    out_rfi_sk_buf = get_buffer("out_rfi_sk_buf");
    out_rfi_sk_buf->register_producer(unique_name);

    out_rfi_sktilde_buf = get_buffer("out_rfi_sktilde_buf");
    out_rfi_sktilde_buf->register_producer(unique_name);

    out_rfi_mask_buf = get_buffer("out_rfi_mask_buf");
    out_rfi_mask_buf->register_producer(unique_name);

    int64_t nt = _samples_per_data_set / _rfi_downsampling_factor;
    size_t bf_mask_size = _num_elements;
    size_t rfi_s012_size = nt * _num_local_freq * 3 * _num_elements * sizeof(uint64_t);

    // Check input sizes and buffer compatibility
    if (_samples_per_data_set % _rfi_downsampling_factor != 0) {
        FATAL_ERROR("samples_per_data_set must be a multiple of rfi_downsampling_factor");
    }
    assert(_samples_per_data_set % _rfi_downsampling_factor == 0);

    if (in_rfi_s012_buf->frame_size != rfi_s012_size) {
        FATAL_ERROR("in_rfi_s012_buf ({:s}) has frame size: {:d}, expected: {:d}",
                    in_rfi_s012_buf->buffer_name, in_rfi_s012_buf->frame_size, rfi_s012_size);
    }
    assert(in_rfi_s012_buf->frame_size == rfi_s012_size);

    if (in_bf_mask_buf->frame_size != bf_mask_size) {
        FATAL_ERROR("in_bf_mask_buf ({:s}) has frame size: {:d}, expected: {:d}",
                    in_bf_mask_buf->buffer_name, in_bf_mask_buf->frame_size, bf_mask_size);
    }
    assert(in_bf_mask_buf->frame_size == bf_mask_size);

    // Make frame desc for produced buffers
    if(_bar_mode) {
        out_rfi_sk_buf->allocate_ndarray_frame_desc<float, 5>(
            "SKbar", {nt, _num_local_freq, 3, _num_polarizations, _num_dishes},
            {"Trfibar", "F", "SK", "P", "D"});
        out_rfi_sktilde_buf->allocate_ndarray_frame_desc<float, 3>(
            "SKbartilde", {nt, _num_local_freq, 3},
            {"Trfibar", "F", "SK"});
        out_rfi_mask_buf->allocate_ndarray_frame_desc<kotekan::uint1x8_t, 3>(
            "RFImask", {_samples_per_data_set / 1024, _num_local_freq, 128},
            {"T8hi128", "F", "T8lo128"});
    } else {
        out_rfi_sk_buf->allocate_ndarray_frame_desc<float, 5>(
            "SK", {nt, _num_local_freq, 3, _num_polarizations, _num_dishes},
            {"Trfi", "F", "SK", "P", "D"});
        out_rfi_sktilde_buf->allocate_ndarray_frame_desc<float, 3>(
            "SKtilde", {nt, _num_local_freq, 3},
            {"Trfi", "F", "SK"});
        out_rfi_mask_buf->allocate_ndarray_frame_desc<kotekan::uint1x8_t, 3>(
            "RFImask", {_samples_per_data_set / 1024, _num_local_freq, 128},
            {"T8hi128", "F", "T8lo128"});
    }
}

gpuSimulateRFISK::~gpuSimulateRFISK() {}

void gpuSimulateRFISK::main_thread() {

    frameID in_bf_mask_frame_id(in_bf_mask_buf);
    frameID in_rfi_s012_frame_id(in_rfi_s012_buf);
    frameID out_rfi_sktilde_frame_id(out_rfi_sktilde_buf);
    frameID out_rfi_mask_frame_id(out_rfi_mask_buf);

    while (!stop_thread) {
        uint8_t* bf_mask =
            (uint8_t*)in_bf_mask_buf->wait_for_full_frame(unique_name, in_bf_mask_frame_id);
        if (bf_mask == nullptr)
            break;
        uint64_t* rfi_s012 =
            (uint64_t*)in_rfi_s012_buf->wait_for_full_frame(unique_name, in_rfi_s012_frame_id);
        if (rfi_s012 == nullptr)
            break;
        float* rfi_sktilde =
            (float*)out_rfi_sktilde_buf->wait_for_empty_frame(unique_name, out_rfi_sktilde_frame_id);
        if (rfi_sktilde == nullptr)
            break;
        uint8_t* rfi_mask =
            (uint8_t*)out_rfi_mask_buf->wait_for_empty_frame(unique_name, out_rfi_mask_frame_id);
        if (rfi_mask == nullptr)
            break;

        INFO("Simulating GPU RFI S012 for {:s}[{:d}], {:s}[{:d}] and putting result in {:s}[{:d}], {:s}[{:d}]",
             in_bf_mask_buf->buffer_name, in_bf_mask_frame_id,
             in_rfi_s012_buf->buffer_name, in_rfi_s012_frame_id,
             out_rfi_sktilde_buf->buffer_name, out_rfi_sktilde_frame_id,
             out_rfi_mask_buf->buffer_name, out_rfi_mask_frame_id);

        // number of elements = number of dishes * polarizations
        uint64_t nt = _samples_per_data_set;
        uint64_t nf = _num_local_freq;
        uint64_t ne = _num_elements;

        uint64_t nt_rfi = nt / _rfi_downsampling_factor;
        uint64_t nsub = _rfi_downsampling_factor;

        // array access strides in S012
        uint64_t sstride_s012 = ne;
        uint64_t fstride_s012 = 3 * ne;
        uint64_t tstride_s012 = nf * 3 * ne;

        // array access strides in SK
        uint64_t fstride_sk = 3;
        uint64_t tstride_sk = nf * 3;

        // Set to 0 to start.
        for (uint64_t tfs = 0; tfs < tstride_sk * nt_rfi; tfs++)
            rfi_sktilde[tfs] = 0;
        for (uint64_t tf = 0; tf < nt * nf / 8; tf++)
            rfi_mask[tf] = 0;

        // Looping over entries in the rfi buffer.
        for (uint64_t t_rfi = 0; t_rfi < nt_rfi; t_rfi++) {

            for (uint64_t f = 0; f < nf; f++) {

                uint64_t sk_idx = f * fstride_sk + t_rfi * tstride_sk;
                uint64_t s012_idx = f * fstride_s012 + t_rfi * tstride_s012;

                uint64_t n = 0;

                for (uint64_t e = 0; e < ne; e++) {
                    if (bf_mask[e])
                        n += rfi_s012[s012_idx + e];


                        uint64_t idx_v = e + fstride * f + tstride * t;
                        uint64_t idx_rfi = e + fstride_rfi * f + tstride_rfi * t_rfi;

                        uint64_t e_pl = e / 8;
                        uint64_t f_pl = f / 4;
                        uint64_t t_pl = t / 2;
                        uint64_t t_pl_hi = t_pl / 64;
                        uint64_t t_pl_lo = t_pl % 64;
                        uint64_t idx_pl = e_pl + fstride_pl * f_pl + tstride_pl * t_pl_hi;

                        // Grab the value of the PL mask
                        uint64_t pl = (pl_mask[idx_pl] >> t_pl_lo) & 1u;

                        // Decode the input voltage in the CHIME format:
                        // offset encoded by 8, imaginary part in lo 4 bits,
                        // real part in hi 4 bits.
                        int32_t ei = static_cast<int32_t>(voltage[idx_v] & 0x0f) - 8;
                        int32_t er = static_cast<int32_t>((voltage[idx_v] & 0xf0) >> 4) - 8;

                        uint64_t e2 = er * er + ei * ei;
                        uint64_t e4 = e2 * e2;

                        rfi_s012[idx_rfi + 0 * sstride_rfi] += pl;
                        rfi_s012[idx_rfi + 1 * sstride_rfi] += pl * e2;
                        rfi_s012[idx_rfi + 2 * sstride_rfi] += pl * e4;
                    } // e
                } // f
            } // tsub
        } // t_rfi

        // Fetch input metadata
        const std::shared_ptr<const metadataObject> mc_in =
            in_voltage_buf->get_metadata(in_voltage_frame_id);
        if (!mc_in) {
            FATAL_ERROR("Buffer {:s} frame {:d} had no metadata", in_voltage_buf->buffer_name,
                        in_voltage_frame_id);
        }
        assert(mc_in);
        if (!metadata_is_chord(mc_in)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        in_voltage_buf->buffer_name, in_voltage_frame_id);
        }
        assert(metadata_is_chord(mc_in));

        const std::shared_ptr<const chordMetadata> meta_in = get_chord_metadata(mc_in);

        // Create output metadata
        out_rfis012_buf->allocate_new_metadata_object(out_rfis012_frame_id);
        const std::shared_ptr<metadataObject> mc =
            out_rfis012_buf->get_metadata(out_rfis012_frame_id);
        if (!mc) {
            FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata",
                        out_rfis012_buf->buffer_name, out_rfis012_frame_id);
        }
        assert(mc);
        if (!metadata_is_chord(mc)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        out_rfis012_buf->buffer_name, out_rfis012_frame_id);
        }
        assert(metadata_is_chord(mc));
        const std::shared_ptr<chordMetadata> meta_out = get_chord_metadata(mc);
        assert(meta_out);

        // Start with a copy
        meta_out->deepCopy(meta_in);

        meta_out->set_from_frame_desc(out_rfis012_buf->get_ndarray_frame_desc());

        // test that things are consistent
        meta_out->check_frame_desc(out_rfis012_buf->get_ndarray_frame_desc());

        // Set non-NDArray things.
        meta_out->set_fpga_seq_num(meta_in->get_fpga_seq_num());
        meta_out->set_time_downsampling_fpga(meta_in->get_time_downsampling_fpga()
                                             * _rfi_downsampling_factor);

        INFO("Simulating GPU RFI S012 done for {:s}[{:d}]+{:s}[{:d}] result is in {:s}[{:d}]",
             in_plmask_buf->buffer_name, in_plmask_frame_id, in_voltage_buf->buffer_name,
             in_voltage_frame_id, out_rfis012_buf->buffer_name, out_rfis012_frame_id);

        in_plmask_buf->mark_frame_empty(unique_name, in_plmask_frame_id++);
        in_voltage_buf->mark_frame_empty(unique_name, in_voltage_frame_id++);
        out_rfis012_buf->mark_frame_full(unique_name, out_rfis012_frame_id++);
    }
}
