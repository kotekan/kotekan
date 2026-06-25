#include "gpuSimulateRFIS012.hpp"

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

REGISTER_KOTEKAN_STAGE(gpuSimulateRFIS012);

gpuSimulateRFIS012::gpuSimulateRFIS012(Config& config, const std::string& unique_name,
                                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&gpuSimulateRFIS012::main_thread, this)),
    _num_polarizations(config.get<int64_t>(unique_name, "num_polarizations")),
    _num_dishes(config.get<int64_t>(unique_name, "num_dishes")),
    _num_elements(_num_polarizations * _num_dishes),
    _num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    _samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    _rfi_downsampling_factor(config.get<int64_t>(unique_name, "rfi_downsampling_factor")) {

    // Grab Buffers
    in_voltage_buf = get_buffer("in_voltage_buf");
    in_voltage_buf->register_consumer(unique_name);
    in_plmask_buf = get_buffer("in_plmask_buf");
    in_plmask_buf->register_consumer(unique_name);

    out_rfis012_buf = get_buffer("out_buf");
    out_rfis012_buf->register_producer(unique_name);


    // Check input sizes and buffer compatibility
    if (_samples_per_data_set % _rfi_downsampling_factor != 0) {
        FATAL_ERROR("samples_per_data_set must be a multiple of rfi_downsampling_factor");
    }
    assert(_samples_per_data_set % _rfi_downsampling_factor == 0);

    if (_num_elements % 8 != 0) {
        FATAL_ERROR("num_elements (num_polarizations x num_dishes) must be a multiple of 8");
    }

    in_voltage_buf->require_frame_desc(
        kotekan::NDArray<kotekan::int4x2_swapped_withoffset_t, 4>::describe(
            "E", {_samples_per_data_set, _num_local_freq, _num_polarizations, _num_dishes},
            {"T", "F", "P", "D"}, {1, 1, 1, 1}));
    in_plmask_buf->require_frame_desc(kotekan::NDArray<kotekan::uint1x8_t, 5>::describe(
        "pl_mask",
        {_samples_per_data_set / 128, _num_local_freq / 4, _num_polarizations, _num_dishes / 8, 8},
        {"T2hi64", "F4", "P", "D8", "T2lo64"}, {128, 4, 1, 8, 16}));

    // Make frame desc for produced buffer
    int64_t nt = _samples_per_data_set / _rfi_downsampling_factor;
    out_rfis012_buf->require_frame_desc(kotekan::NDArray<uint64_t, 5>::describe(
        "S012", {nt, _num_local_freq, 3, _num_polarizations, _num_dishes},
        {"Trfi", "F", "S", "P", "D"}, {_rfi_downsampling_factor, 1, 1, 1, 1}));
}

gpuSimulateRFIS012::~gpuSimulateRFIS012() {}

void gpuSimulateRFIS012::main_thread() {

    frameID in_plmask_frame_id(in_plmask_buf);
    frameID in_voltage_frame_id(in_voltage_buf);
    frameID out_rfis012_frame_id(out_rfis012_buf);

    while (!stop_thread) {
        uint64_t* pl_mask =
            (uint64_t*)in_plmask_buf->wait_for_full_frame(unique_name, in_plmask_frame_id);
        if (pl_mask == nullptr)
            break;
        uint8_t* voltage =
            (uint8_t*)in_voltage_buf->wait_for_full_frame(unique_name, in_voltage_frame_id);
        if (voltage == nullptr)
            break;
        uint64_t* rfi_s012 =
            (uint64_t*)out_rfis012_buf->wait_for_empty_frame(unique_name, out_rfis012_frame_id);
        if (rfi_s012 == nullptr)
            break;

        INFO("Simulating GPU RFI S012 for {:s}[{:d}], {:s}[{:d}] and putting result in {:s}[{:d}]",
             in_plmask_buf->buffer_name, in_plmask_frame_id, in_voltage_buf->buffer_name,
             in_voltage_frame_id, out_rfis012_buf->buffer_name, out_rfis012_frame_id);

        // number of elements = number of dishes * polarizations
        uint64_t nt = _samples_per_data_set;
        uint64_t nf = _num_local_freq;
        uint64_t ne = _num_elements;

        uint64_t nt_rfi = nt / _rfi_downsampling_factor;
        uint64_t nsub = _rfi_downsampling_factor;

        uint64_t nf_pl = (_num_local_freq + 3) / 4;
        uint64_t ne_pl = _num_elements / 8;

        // array access strides in raw (downsampled) PL mask
        uint64_t fstride_pl = ne_pl;
        uint64_t tstride_pl = ne_pl * nf_pl;

        // array access strides in voltage
        uint64_t fstride = ne;
        uint64_t tstride = ne * nf;

        // array access strides in S012
        uint64_t sstride_rfi = ne;
        uint64_t fstride_rfi = 3 * ne;
        uint64_t tstride_rfi = nf * 3 * ne;

        // Set to 0 to start.
        for (uint64_t tfse = 0; tfse < tstride_rfi * nt_rfi; tfse++)
            rfi_s012[tfse] = 0;

        // Looping over entries in the rfi buffer.
        for (uint64_t t_rfi = 0; t_rfi < nt_rfi; t_rfi++) {

            // Now accumulate.
            for (uint64_t tsub = 0; tsub < nsub; tsub++) {

                // Global t index
                uint64_t t = tsub + nsub * t_rfi;

                for (uint64_t f = 0; f < nf; f++) {
                    for (uint64_t e = 0; e < ne; e++) {

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

        meta_out->set_from_frame_desc(out_rfis012_buf->get_frame_desc<kotekan::GenericNDArray>());

        // test that things are consistent
        meta_out->check_frame_desc(out_rfis012_buf->get_frame_desc<kotekan::GenericNDArray>());

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
