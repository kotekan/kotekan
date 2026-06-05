#include "gpuSimulateN2kCorr.hpp"

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType, int4x2_swapped_withoffset_t, GetType, uint1x8_t
#include "NDArray.hpp"         // for NDArray, GenericNDArray, Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, metadata_is_chord, get_chord_metadata, CHO...
#include "kotekanLogging.hpp"  // for FATAL_ERROR, INFO, DEBUG
#include "metadata.hpp"        // for metadataObject

#include "fmt.hpp" // for compile_string_to_view

#include <assert.h>   // for assert
#include <cstdint>    // for int32_t, uint8_t
#include <cstdlib>    // for abort
#include <fmt/core.h> // for format
#include <functional> // for bind, function
#include <memory>     // for shared_ptr, __shared_ptr_access
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(gpuSimulateN2kCorr);

gpuSimulateN2kCorr::gpuSimulateN2kCorr(Config& config, const std::string& unique_name,
                                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&gpuSimulateN2kCorr::main_thread, this)) {

    // Apply config.
    _num_elements = config.get<int32_t>(unique_name, "num_elements"); // = "2*D"
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _sub_integration_ntime = config.get<int>(unique_name, "sub_integration_ntime");

    // Check parameter compatibility
    if (_num_elements <= 0) {
        FATAL_ERROR("num_elements ({:d}) is not positive.", _num_elements);
        std::abort();
    }
    if (_num_local_freq <= 0) {
        FATAL_ERROR("num_local_freq ({:d}) is not positive.", _num_local_freq);
        std::abort();
    }
    if (_samples_per_data_set <= 0) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not positive.", _samples_per_data_set);
        std::abort();
    }
    if (_sub_integration_ntime <= 0) {
        FATAL_ERROR("sub_integration_ntime ({:d}) is not positive.", _sub_integration_ntime);
        std::abort();
    }
    if (_samples_per_data_set % _sub_integration_ntime != 0) {
        FATAL_ERROR(
            "samples_per_data_set ({:d}) is not a multiple of sub_integration_ntime ({:d}).",
            _samples_per_data_set, _sub_integration_ntime);
        std::abort();
    }
    if (_samples_per_data_set % 1024 != 0) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not a multiple of 1024.",
                    _samples_per_data_set);
        std::abort();
    }
    if (_num_elements % _corr_blocksize != 0) {
        FATAL_ERROR("num_elements ({:d}) is not a multiple of _corr_blocksize ({:d}).",
                    _num_elements, _corr_blocksize);
        std::abort();
    }

    input_buf = get_buffer("network_in_buf");
    input_buf->register_consumer(unique_name);
    rfimask_buf = get_buffer("rfimask_in_buf");
    rfimask_buf->register_consumer(unique_name);
    output_buf = get_buffer("corr_out_buf");
    output_buf->register_producer(unique_name);

    /* new style array description */
    // number of elements = number of dishes * polarizations
    int nt_inner = _sub_integration_ntime;
    int nt_outer = _samples_per_data_set / nt_inner;
    input_buf->require_frame_desc(
        kotekan::NDArray<kotekan::int4x2_swapped_withoffset_t, 4>::describe(
            "E", {_samples_per_data_set, _num_local_freq, 2, _num_elements / 2},
            {"T", "F", "P", "D"}));
    rfimask_buf->require_frame_desc(
        kotekan::NDArray<kotekan::GetType<kotekan::uint1x8>::type, 3>::describe(
            "RFImask", {_samples_per_data_set / 1024, _num_local_freq, 128},
            {"T8hi128", "F", "T8lo128"}));
    output_buf->require_frame_desc(
        kotekan::NDArray<kotekan::GetType<kotekan::int32>::type, 6>::describe(
            "n2k_correlation",
            {nt_outer, _num_local_freq, (_num_elements / 16) * (_num_elements / 16 + 1) / 2, 16, 16,
             2},
            {"Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"}));
}

gpuSimulateN2kCorr::~gpuSimulateN2kCorr() {}

void gpuSimulateN2kCorr::main_thread() {

    int input_frame_id = 0;
    int rfimask_frame_id = 0;
    int output_frame_id = 0;

    while (!stop_thread) {
        char* input = (char*)input_buf->wait_for_full_frame(unique_name, input_frame_id);
        if (input == nullptr)
            break;
        uint8_t* rfimask =
            (uint8_t*)rfimask_buf->wait_for_full_frame(unique_name, rfimask_frame_id);
        if (rfimask == nullptr)
            break;
        int* output = (int*)output_buf->wait_for_empty_frame(unique_name, output_frame_id);
        if (output == nullptr)
            break;

        INFO("Simulating GPU processing for {:s}[{:d}] and {:s}[{:d}] putting result in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, rfimask_buf->buffer_name, rfimask_frame_id,
             output_buf->buffer_name, output_frame_id);

        // number of elements = number of dishes * polarizations
        int nt_inner = _sub_integration_ntime;
        int nt_outer = _samples_per_data_set / nt_inner;

        int num_blocks_lin = _num_elements / _corr_blocksize;
        int num_blocks = (num_blocks_lin * (num_blocks_lin + 1)) / 2;

        int fstride = _corr_blocksize * _corr_blocksize * num_blocks;
        int tstride = _num_local_freq * fstride;

        INFO("Running stage with nt_outer={:d}, nt_inner={:d}, _num_local_freq={:d}, "
             "_num_elements={:d}",
             nt_outer, nt_inner, _num_local_freq, _num_elements);

        for (int tout = 0; tout < nt_outer; ++tout) {
            for (int f = 0; f < _num_local_freq; ++f) {
                // loop through blocks
                for (int jhi = 0; jhi < _num_elements / _corr_blocksize; jhi++) {
                    for (int ihi = jhi; ihi < _num_elements / _corr_blocksize; ihi++) {
                        for (int jlo = 0; jlo < _corr_blocksize; jlo++) {
                            for (int ilo = 0; ilo < _corr_blocksize; ilo++) {
                                int real = 0;
                                int imag = 0;

                                for (int tin = 0; tin < nt_inner; ++tin) {
                                    int t = tout * nt_inner + tin;
                                    int ix = (t * _num_local_freq + f) * _num_elements
                                             + (_corr_blocksize * ihi + ilo);
                                    int iy = (t * _num_local_freq + f) * _num_elements
                                             + (_corr_blocksize * jhi + jlo);

                                    // Extract the indices for the RFI mask
                                    // 0x3FF extracts the low 10 bits for the
                                    // fast (1024 = 2^10) index.
                                    int t_rfi_in = t & 0x3FF;
                                    // Shift down by 10 bits to get the slow
                                    // (divided by 1024) index.
                                    int t_rfi_out = t >> 10;
                                    // The fast index refers to a particular
                                    // bit in a byte.  The bit index is
                                    // in the low 3 bits (7 = 0b0111)
                                    int t_rfi_in_bit = t_rfi_in & 7;
                                    // The byte address is in the remaining bits
                                    int t_rfi_in_byte = t_rfi_in >> 3;
                                    // Assemble the index into the rfi_mask
                                    // array.
                                    int rfi_idx =
                                        t_rfi_in_byte + 128 * (f + _num_local_freq * t_rfi_out);

                                    // Decode input with the CHIME convention:
                                    // offset encoded by 8, imaginary part in
                                    // lo 4 bits, real part in high 4 bits.
                                    int xi = (input[ix] & 0x0f) - 8;
                                    int xr = ((input[ix] & 0xf0) >> 4) - 8;
                                    int yi = (input[iy] & 0x0f) - 8;
                                    int yr = ((input[iy] & 0xf0) >> 4) - 8;

                                    // get the RFI value. 1 for clean,
                                    // 0 for dirty.
                                    int rfi = (rfimask[rfi_idx] >> t_rfi_in_bit) & 1;

                                    // accumulate!
                                    real += rfi * (xr * yr + xi * yi);
                                    imag += rfi * (xi * yr - yi * xr);
                                }

                                // clang-format off
                                int o = 2*( tout * tstride + f * fstride + 256*(ihi*(ihi+1)/2 + jhi)
                                        + _corr_blocksize*ilo + jlo );
                                output[o + 0] = +real;
                                output[o + 1] = +imag;
                                // clang-format on

                            } // ilo
                        } // jlo
                    } // iji
                } // jhi

                DEBUG("Done t_outer {:d} of {:d} (freq {:d} of {:d}, nt_inner={:d})...", tout,
                      nt_outer, f, _num_local_freq, nt_inner);

                if (stop_thread)
                    break;
            } // f
            if (stop_thread)
                break;
        } // tout

        // Fetch input metadata
        const std::shared_ptr<const metadataObject> mc_in = input_buf->get_metadata(input_frame_id);
        if (!mc_in) {
            FATAL_ERROR("Buffer {:s} frame {:d} had no metadata", input_buf->buffer_name,
                        input_frame_id);
        }
        assert(mc_in);
        if (!metadata_is_chord(mc_in)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        input_buf->buffer_name, input_frame_id);
        }
        assert(metadata_is_chord(mc_in));

        const std::shared_ptr<const chordMetadata> meta_in = get_chord_metadata(mc_in);

        // Create output metadata
        output_buf->allocate_new_metadata_object(output_frame_id);
        const std::shared_ptr<metadataObject> mc = output_buf->get_metadata(output_frame_id);
        if (!mc) {
            FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata", output_buf->buffer_name,
                        output_frame_id);
        }
        assert(mc);
        if (!metadata_is_chord(mc)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        output_buf->buffer_name, output_frame_id);
        }
        assert(metadata_is_chord(mc));
        const std::shared_ptr<chordMetadata> meta_out = get_chord_metadata(mc);
        assert(meta_out);

        // Check input metadata describes expected size.
        if (meta_in->get_nfreq() != _num_local_freq) {
            FATAL_ERROR("Buffer {:s} metadata has nfreq {:d}, expected num_local_freq ({:d}).",
                        input_buf->buffer_name, meta_in->get_nfreq(), _num_local_freq);
        }
        assert(meta_in->get_nfreq() == _num_local_freq);

        // Start out with a copy
        meta_out->deepCopy(meta_in);

        // Assign changes
        meta_out->set_name("n2k_correlation");
        meta_out->type = kotekan::int32;
        meta_out->dims = 6;
        assert(meta_out->dims <= CHORD_META_MAX_DIM);
        meta_out->set_array_dimension(0, nt_outer, "Tc");
        meta_out->set_array_dimension(1, _num_local_freq, "F");
        meta_out->set_array_dimension(2, (_num_elements / 16) * (_num_elements / 16 + 1) / 2,
                                      "DPhi");
        meta_out->set_array_dimension(3, 16, "DPlo1");
        meta_out->set_array_dimension(4, 16, "DPlo2");
        meta_out->set_array_dimension(5, 2, "C");
        meta_out->set_strides_simple();

        // frame_desc set in constructor
        /* test that things are consistent */
        meta_out->check_frame_desc(output_buf->get_frame_desc<kotekan::GenericNDArray>());

        meta_out->set_fpga_seq_num(meta_in->get_fpga_seq_num());
        meta_out->set_time_downsampling_fpga(meta_in->get_time_downsampling_fpga()
                                             * _sub_integration_ntime);

        std::vector<int> coarse_freq(_num_local_freq);
        const std::vector<int> coarse_freq_in = meta_in->get_coarse_freq();

        for (int f = 0; f < _num_local_freq; f++) {
            coarse_freq[f] = coarse_freq_in[f];
        }

        meta_out->set_coarse_freq(coarse_freq);
        assert(meta_out->get_nfreq() <= CHORD_META_MAX_FREQ);

        input_buf->mark_frame_empty(unique_name, input_frame_id);
        rfimask_buf->mark_frame_empty(unique_name, rfimask_frame_id);

        // Pretend some samples were lost
        // chordMetadata* chord_metadata = (chordMetadata*)
        // output_buf->get_metadata(output_frame_id); chord_metadata->lost_timesamples[0] = 1;

        output_buf->mark_frame_full(unique_name, output_frame_id);

        INFO("Simulating GPU processing done for {:s}[{:d}] and {:s}[{:d}] result is in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, rfimask_buf->buffer_name, rfimask_frame_id,
             output_buf->buffer_name, output_frame_id);

        input_frame_id = (input_frame_id + 1) % input_buf->num_frames;
        rfimask_frame_id = (rfimask_frame_id + 1) % rfimask_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buf->num_frames;
    }
}
