#include "gpuSimulateN2kPL1bitCorr.hpp"

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType, GetType
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, metadata_is_chord, get_chord_metadata, CHO...
#include "div.hpp"             // for div_ceil, div_noremainder, num_triangle_blocks
#include "kotekanLogging.hpp"  // for FATAL_ERROR, INFO, DEBUG
#include "metadata.hpp"        // for metadataObject

#include "fmt.hpp" // for compile_string_to_view

#include <assert.h>   // for assert
#include <bitset>     // for bitset
#include <cstdint>    // for uint64_t, int32_t, int64_t
#include <cstdlib>    // for abort
#include <functional> // for bind, function
#include <memory>     // for shared_ptr, __shared_ptr_access
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::div_ceil;
using kotekan::div_noremainder;
using kotekan::num_triangle_blocks;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(gpuSimulateN2kPL1bitCorr);

gpuSimulateN2kPL1bitCorr::gpuSimulateN2kPL1bitCorr(Config& config, const std::string& unique_name,
                                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&gpuSimulateN2kPL1bitCorr::main_thread, this)) {

    // Apply config.
    _num_elements = config.get<int32_t>(unique_name, "num_elements"); // = "2*D"
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _sub_integration_ntime = config.get<int32_t>(unique_name, "sub_integration_ntime");

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
        // Also covers the multiple of 64 required by PL mask.
        FATAL_ERROR("samples_per_data_set ({:d}) is not a multiple of 1024.",
                    _samples_per_data_set);
        std::abort();
    }
    if (_num_elements % (8 * _blocksize) != 0) {
        FATAL_ERROR("num_elements ({:d}) / 8 is not a multiple of _blocksize ({:d}).",
                    _num_elements, _blocksize);
        std::abort();
    }

    input_plmask_buf = get_buffer("in_plmask_buf");
    input_plmask_buf->register_consumer(unique_name);
    input_rfimask_buf = get_buffer("in_rfimask_buf");
    input_rfimask_buf->register_consumer(unique_name);
    output_buf = get_buffer("out_buf");
    output_buf->register_producer(unique_name);

    // TODO: Check input_plmask; does it have the right time scaling?
    const auto input_plmask_frame_desc = input_plmask_buf->get_ndarray_frame_desc();
    assert(std::string(input_plmask_frame_desc->get_dimname(0)) == "T");
    const auto input_plmask_time_scaling = input_plmask_frame_desc->get_dimscaling(0);

    /* new style array description */
    int n_integrations = _samples_per_data_set / _sub_integration_ntime;
    int nf = _num_local_freq;
    int ne = _num_elements / 8;
    int n_blocks = num_triangle_blocks(ne, _blocksize);
    output_buf->allocate_ndarray_frame_desc<kotekan::GetType<kotekan::int32>::type, 5>(
        "n2k_counts", {n_integrations, nf, n_blocks, _blocksize, _blocksize},
        {"Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"}, {input_plmask_time_scaling, 1, 8, 1, 1});
}

gpuSimulateN2kPL1bitCorr::~gpuSimulateN2kPL1bitCorr() {}

void gpuSimulateN2kPL1bitCorr::main_thread() {

    int input_pl_frame_id = 0;
    int input_rfi_frame_id = 0;
    int output_frame_id = 0;

    while (!stop_thread) {
        uint64_t* pl_mask =
            (uint64_t*)input_plmask_buf->wait_for_full_frame(unique_name, input_pl_frame_id);
        if (pl_mask == nullptr)
            break;
        uint64_t* rfi_mask =
            (uint64_t*)input_rfimask_buf->wait_for_full_frame(unique_name, input_rfi_frame_id);
        if (rfi_mask == nullptr)
            break;
        int32_t* counts = (int32_t*)output_buf->wait_for_empty_frame(unique_name, output_frame_id);
        if (counts == nullptr)
            break;

        INFO("Simulating GPU PL Correlator for {:s}[{:d}] and {:s}[{:d}] putting result in "
             "{:s}[{:d}]",
             input_plmask_buf->buffer_name, input_pl_frame_id, input_rfimask_buf->buffer_name,
             input_rfi_frame_id, output_buf->buffer_name, output_frame_id);

        // num_times = samples_per_data_set
        // num_freq = num_local_freq
        // num_el = num_elements
        //
        // PL:  bool1 [num_times/64][num_freq][num_el/8][64]
        // RFI: bool1 [num_times/1024][num_freq][1024]
        //
        // PL:  u64 [num_times/64][num_freq][num_el/8]
        // RFI: u64 [num_times/1024][num_freq][16]

        int n_integrations = _samples_per_data_set / _sub_integration_ntime;
        int nt64_inner = _sub_integration_ntime / 64;
        int nf = _num_local_freq;
        int ne = _num_elements / 8;

        int n_block_lin = div_ceil(ne, _blocksize);
        int n_blocks = num_triangle_blocks(ne, _blocksize);

        // Strides into the (expanded) PL mask (as uint64_t's)
        int df_pl = ne;      // freq stride
        int dt_pl = ne * nf; // slow time stride

        // Strides into the RFI mask (as uint64_t's)
        int df_rfi = 16;      // freq stride
        int dt_rfi = 16 * nf; // slow time stride

        // strides into the count matrix
        int diin_c = _blocksize;                            // inner block i stride
        int db_c = _blocksize * _blocksize;                 // block stride
        int df_c = n_blocks * _blocksize * _blocksize;      // freq stride
        int dt_c = nf * n_blocks * _blocksize * _blocksize; // t_out stride

        assert(_num_elements % (8 * _blocksize) == 0);

        INFO("Running PL corr with nt_outer={:d}, nt_inner={:d}, num_freq={:d}, "
             "num_downsampled_elements={:d}",
             n_integrations, _sub_integration_ntime, nf, ne);

        // Loop over outer (unsummed) time and frequency
        for (int t_out = 0; t_out < n_integrations; t_out++) {
            for (int f = 0; f < nf; f++) {
                // Loop over the blocks
                int block_idx = 0;
                for (int i_out = 0; i_out < n_block_lin; i_out++) {
                    // Only taking the LOWER triangular blocks!  j <= i.
                    for (int j_out = 0; j_out <= i_out; j_out++) {
                        // Loop inside the blocks
                        for (int i_in = 0; i_in < _blocksize; i_in++) {
                            for (int j_in = 0; j_in < _blocksize; j_in++) {
                                int i = i_out * _blocksize + i_in;
                                int j = j_out * _blocksize + j_in;

                                // Now we can do the sum!
                                int32_t c = 0;
                                for (int t64_in = 0; t64_in < nt64_inner; t64_in++) {
                                    int t64 = t_out * nt64_inner + t64_in;
                                    int t_pl = t64;
                                    int t_rfi_lo = t64 & 0xF;
                                    int t_rfi_hi = t64 >> 4;

                                    uint64_t pl_i = pl_mask[t_pl * dt_pl + f * df_pl + i];
                                    uint64_t pl_j = pl_mask[t_pl * dt_pl + f * df_pl + j];
                                    uint64_t rfi =
                                        rfi_mask[t_rfi_hi * dt_rfi + f * df_rfi + t_rfi_lo];
                                    uint64_t v = pl_i & pl_j & rfi;
                                    std::bitset<64> v_bits(v);
                                    c += v_bits.count();
                                } // t64_in

                                counts[t_out * dt_c + f * df_c + block_idx * db_c + i_in * diin_c
                                       + j_in] = c;
                            } // j_in
                        } // i_in

                        block_idx++;
                    } // j_out
                } // i_out

                DEBUG("Done t_outer {:d} of {:d} (freq {:d} of {:d}, nt_inner={:d})...", t_out,
                      n_integrations, f, nf, _sub_integration_ntime);
            } // f
        } // t_out

        // Fetch input metadata
        const std::shared_ptr<const metadataObject> mc_in =
            input_plmask_buf->get_metadata(input_pl_frame_id);
        if (!mc_in) {
            FATAL_ERROR("Buffer {:s} frame {:d} had no metadata", input_plmask_buf->buffer_name,
                        input_pl_frame_id);
        }
        assert(mc_in);
        if (!metadata_is_chord(mc_in)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        input_plmask_buf->buffer_name, input_pl_frame_id);
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

        // Checking incoming metadata describes data as expected.
        if (meta_in->get_nfreq() != nf)
            FATAL_ERROR("in_plmask_buf ({:s}) has nfreq {:d}, expected {:d}",
                        input_plmask_buf->buffer_name, meta_in->get_nfreq(), nf);

        // Copy to start
        meta_out->deepCopy(meta_in);

        // Update the changes
        meta_out->set_name("n2k_counts");
        meta_out->type = kotekan::int32;
        meta_out->dims = 5;
        assert(meta_out->dims <= CHORD_META_MAX_DIM);
        meta_out->set_array_dimension(0, n_integrations, "Tc",
                                      div_noremainder(meta_in->get_time_downsampling_fpga(), 64)
                                          * _sub_integration_ntime);
        meta_out->set_array_dimension(1, nf, "F", 1);
        meta_out->set_array_dimension(2, n_blocks, "D8Phi", 8);
        meta_out->set_array_dimension(3, _blocksize, "D8Plo1", 1);
        meta_out->set_array_dimension(4, _blocksize, "D8Plo2", 1);
        meta_out->set_strides_simple();
        // frame_desc set in constructor
        /* test that things are consistent */
        meta_out->check_frame_desc(output_buf->get_ndarray_frame_desc());

        meta_out->set_fpga_seq_num(meta_in->get_fpga_seq_num());
        meta_out->set_time_downsampling_fpga(
            div_noremainder(meta_in->get_time_downsampling_fpga(), 64) * _sub_integration_ntime);

        const std::vector<int> coarse_freq_in = meta_in->get_coarse_freq();
        const std::vector<int> freq_upchan_factor_in = meta_in->get_freq_upchan_factor();
        const std::vector<int> freq_upchan_index_in = meta_in->get_freq_upchan_index();
        std::vector<int> coarse_freq(_num_local_freq);
        std::vector<int> freq_upchan_factor(coarse_freq.size());
        std::vector<int> freq_upchan_index(coarse_freq.size());

        for (int f = 0; f < static_cast<int>(coarse_freq.size()); f++) {
            coarse_freq[f] = coarse_freq_in[f];
            freq_upchan_factor[f] = freq_upchan_factor_in[f];
            freq_upchan_index[f] = freq_upchan_index_in[f];
        }

        meta_out->set_coarse_freq(coarse_freq);
        meta_out->set_freq_upchan_factor(freq_upchan_factor);
        meta_out->set_freq_upchan_index(freq_upchan_index);
        assert(meta_out->get_nfreq() <= CHORD_META_MAX_FREQ);

        input_plmask_buf->mark_frame_empty(unique_name, input_pl_frame_id);
        input_rfimask_buf->mark_frame_empty(unique_name, input_rfi_frame_id);

        output_buf->mark_frame_full(unique_name, output_frame_id);

        INFO("Simulating GPU PL correlation done for {:s}[{:d}] and {:s}[{:d}] result is in "
             "{:s}[{:d}]",
             input_plmask_buf->buffer_name, input_pl_frame_id, input_rfimask_buf->buffer_name,
             input_rfi_frame_id, output_buf->buffer_name, output_frame_id);

        input_pl_frame_id = (input_pl_frame_id + 1) % input_plmask_buf->num_frames;
        input_rfi_frame_id = (input_rfi_frame_id + 1) % input_rfimask_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buf->num_frames;
    }
}
