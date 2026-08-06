#include "cudaCorrelatorDual.hpp"

#include "DataType.hpp"            // for int4x2_swapped_withoffset_t, uint1x8_t
#include "NDArray.hpp"             // for NDArray
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer, extent_t, read_descriptor_t
#include "cudaCommand.hpp"         // for cudaCommand, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "div.hpp"                 // for div_noremainder, num_triangle_blocks
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG, INFO

#include "fmt.hpp" // for compile_string_to_view

#include <array>              // for array
#include <cassert>            // for assert
#include <chordMetadata.hpp>  // for chordMetadata
#include <cstddef>            // for ptrdiff_t
#include <cstdint>            // for int32_t, int8_t, uint32_t
#include <cuda_runtime_api.h> // for cudaGetLastError
#include <stdexcept>          // for runtime_error
#include <string>             // for string
#include <tuple>              // for tuple, make_tuple

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::div_noremainder;
using kotekan::num_triangle_blocks;

REGISTER_CUDA_COMMAND(cudaCorrelatorDual);

// The gathered-tile count per channel: mixed (every synthetic row x live-antenna cols) then
// the synthetic triangle. Must match the consumer's recomputation -- documented in the hpp.
static std::vector<int2> build_tile_selection(const std::vector<std::int32_t>& chans,
                                              int num_elements, int num_synth,
                                              int num_live_elements) {
    std::vector<int2> sel;
    const int na16 = num_elements / 16;
    const int nt16 = (num_elements + num_synth) / 16;
    const int nlive16 = (num_live_elements + 15) / 16;
    for (std::int32_t f : chans) {
        for (int ihi = na16; ihi < nt16; ihi++)
            for (int jhi = 0; jhi < nlive16; jhi++)
                sel.push_back({f, 512 * (ihi * (ihi + 1) / 2 + jhi)});
        for (int ihi = na16; ihi < nt16; ihi++)
            for (int jhi = na16; jhi <= ihi; jhi++)
                sel.push_back({f, 512 * (ihi * (ihi + 1) / 2 + jhi)});
    }
    return sel;
}

cudaCorrelatorDual::cudaCorrelatorDual(Config& config, const std::string& unique_name,
                                       bufferContainer& host_buffers, cudaDeviceInterface& device,
                                       const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst),
    _buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    _num_times(config.get<int>(unique_name, "num_times")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _num_synth(config.get_default<int>(unique_name, "num_synth", 128)),
    _num_live_elements(
        config.get_default<int>(unique_name, "num_live_elements", _num_elements)),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _sub_integration_ntime(config.get<int>(unique_name, "sub_integration_ntime")),
    _voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    _rfi_RFImask_name(config.get<std::string>(unique_name, "rfi_RFImask_name")),
    _n2k_correlation_name(config.get<std::string>(unique_name, "n2k_correlation_name")),
    _gnss_tiles_name(config.get<std::string>(unique_name, "gnss_tiles_name")),
    _gnss_synth_name(
        config.get_default<std::string>(unique_name, "gnss_synth_name", "gnss_synth")),
    _gnss_local_channels(config.get<std::vector<std::int32_t>>(unique_name,
                                                               "gnss_local_channels")),
    _tile_sel(build_tile_selection(_gnss_local_channels, _num_elements, _num_synth,
                                   _num_live_elements)),
    voltage(_voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{_buffer_depth * _num_times, _num_local_freq, 2,
                                          _num_elements / 2},
            std::array<std::string, 4>{"T", "F", "P", "D"},
            std::array<std::ptrdiff_t, 4>{1, 1, 1, 1}, *this),
    rfi_RFImask(_rfi_RFImask_name, "RFImask",
                std::array<std::ptrdiff_t, 3>{_buffer_depth * div_noremainder(_num_times, 8 * 128),
                                              _num_local_freq, 128},
                std::array<std::string, 3>{"T8hi128", "F", "T8lo128"},
                std::array<std::ptrdiff_t, 3>{1024, 1, 8}, *this),
    n2k_correlation([&]() {
        // Standard N^2 output -- IDENTICAL declaration to cudaCorrelator's, so downstream
        // (cudaOutputData -> host_correlation_buffer -> N2Accumulate) is unchanged.
        const int num_subintegrations = div_noremainder(_num_times, _sub_integration_ntime);
        const int blocksize = 16;
        const int triangle_num_blocks = num_triangle_blocks(_num_elements, blocksize);
        const std::array<std::ptrdiff_t, 6> n2k_lengths{
            num_subintegrations, _num_local_freq, triangle_num_blocks, blocksize, blocksize, 2};
        const std::array<std::string, 6> n2k_dimnames{"Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"};
        const std::array<std::ptrdiff_t, 6> n2k_dimscalings{_sub_integration_ntime, 1, 16, 1, 1, 1};
        return NDArrayBuffer<std::int32_t, 6>(_n2k_correlation_name, "n2k_correlation", n2k_lengths,
                                              n2k_dimnames, n2k_dimscalings, *this);
    }()),
    gnss_tiles([&]() {
        const int num_subintegrations = div_noremainder(_num_times, _sub_integration_ntime);
        const int n_chan = (int)_gnss_local_channels.size();
        const int tiles_per_chan = n_chan > 0 ? (int)_tile_sel.size() / n_chan : 0;
        const std::array<std::ptrdiff_t, 6> lengths{num_subintegrations, n_chan, tiles_per_chan,
                                                    16, 16, 2};
        const std::array<std::string, 6> dimnames{"Tc", "Fg", "Tile", "DPlo1", "DPlo2", "C"};
        const std::array<std::ptrdiff_t, 6> dimscalings{_sub_integration_ntime, 1, 1, 1, 1, 1};
        return NDArrayBuffer<std::int32_t, 6>(_gnss_tiles_name, "gnss_n2_tiles", lengths, dimnames,
                                              dimscalings, *this);
    }()),
    dual_correlator(_num_elements, _num_synth, _num_local_freq) {
    if (_num_times % _sub_integration_ntime)
        throw std::runtime_error(
            "The sub_integration_ntime parameter must evenly divide samples_per_data_set");
    for (std::int32_t f : _gnss_local_channels)
        if (f < 0 || f >= _num_local_freq)
            throw std::runtime_error(
                "cudaCorrelatorDual: gnss_local_channels entry outside [0, num_local_freq) -- "
                "these are LOCAL frame indices, not global freq_ids");

    voltage.register_consumer();
    rfi_RFImask.register_consumer();

    gpu_buffers_used.push_back(std::make_tuple(_n2k_correlation_name, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gnss_tiles_name, true, false, true));

    // Upload the tile-selection list once (constant for the run).
    int2* d_sel = (int2*)device.get_gpu_memory(unique_name + "_tile_sel",
                                               _tile_sel.size() * sizeof(int2));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sel, _tile_sel.data(), _tile_sel.size() * sizeof(int2),
                                cudaMemcpyHostToDevice));

    // Zero-fill (0x88 == 0+0j offset-encoded) every synth frame slot ONCE. The injector,
    // when present, overwrites only its lanes/channels each frame; without one the synthetic
    // input stays all-zero and the N^2 prefix must be bit-identical to cudaCorrelator's.
    const size_t synth_len = (size_t)_num_times * _num_local_freq * _num_synth;
    for (int s = 0; s < _buffer_depth; s++) {
        void* d_synth =
            device.get_gpu_memory_array(_gnss_synth_name, s, _buffer_depth, synth_len);
        CHECK_CUDA_ERROR(cudaMemset(d_synth, 0x88, synth_len));
    }

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaCorrelatorDual");
    INFO("cudaCorrelatorDual: {:d}+{:d} stations, {:d} freqs, {:d} gnss channels x {:d} tiles "
         "gathered ({:.2f} MB/frame vs {:.1f} MB full triangle)",
         _num_elements, _num_synth, _num_local_freq, (int)_gnss_local_channels.size(),
         (int)(_gnss_local_channels.empty()
                   ? 0
                   : _tile_sel.size() / _gnss_local_channels.size()),
         _tile_sel.size() * 512 * 4 * div_noremainder(_num_times, _sub_integration_ntime) / 1.0e6,
         (double)div_noremainder(_num_times, _sub_integration_ntime) * _num_local_freq
             * dual_correlator.params.vmat_fstride * 4 / 1.0e6);
}

cudaCorrelatorDual::~cudaCorrelatorDual() {}

int cudaCorrelatorDual::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers
    DEBUG("Waiting for voltage input ringbuffer data for frame {:d}...", gpu_frame_id);
    const int voltage_errcode =
        voltage.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            if (available_elements < _num_times)
                return read_descriptor_t{.claimed = 0, .read = 0};
            else
                return read_descriptor_t{.claimed = _num_times, .read = _num_times};
        });
    if (voltage_errcode < 0)
        return voltage_errcode;
    DEBUG("Finished waiting for voltage input for data frame {:d}.", gpu_frame_id);

    DEBUG("Waiting for rfi_RFImask input ringbuffer data for frame {:d}...", gpu_frame_id);
    const int rfi_RFImask_errcode =
        rfi_RFImask.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            const int rfi_needed_samples = div_noremainder(_num_times, 8 * 128);
            if (available_elements < rfi_needed_samples)
                return read_descriptor_t{.claimed = 0, .read = 0};
            else
                return read_descriptor_t{.claimed = rfi_needed_samples, .read = rfi_needed_samples};
        });
    if (rfi_RFImask_errcode < 0)
        return rfi_RFImask_errcode;
    DEBUG("Finished waiting for rfi_RFImask input for data frame {:d}.", gpu_frame_id);

    return 0;
}

cudaEvent_t cudaCorrelatorDual::execute(cudaPipelineState& pipestate,
                                        const std::vector<cudaEvent_t>&) {
    pre_execute();

    voltage.check_metadata();
    rfi_RFImask.check_metadata();
    n2k_correlation.set_metadata(voltage.get_metadata());
    gnss_tiles.set_metadata(voltage.get_metadata());

    const std::shared_ptr<const chordMetadata> voltage_meta = voltage.get_metadata();
    const std::shared_ptr<const chordMetadata> rfi_meta = rfi_RFImask.get_metadata();
    const std::shared_ptr<chordMetadata> n2k_corr_meta = n2k_correlation.get_metadata();
    const std::shared_ptr<chordMetadata> tiles_meta = gnss_tiles.get_metadata();

    // Ensure consistency:
    assert(voltage_meta->get_fpga_seq_num()
               + voltage.get_read_valid().begin() * voltage_meta->get_time_downsampling_fpga()
           == rfi_meta->get_fpga_seq_num()
                  + rfi_RFImask.get_read_valid().begin() * rfi_meta->get_time_downsampling_fpga());

    // The input ringbuffer metadata do not contain time-dependent metadata,
    // so we must reconstruct it here. (fpga_seq_num)
    const auto seq0 = voltage_meta->get_fpga_seq_num()
                      + voltage.get_read_valid().begin()
                            * voltage_meta->get_time_downsampling_fpga();
    n2k_corr_meta->set_fpga_seq_num(seq0);
    n2k_corr_meta->set_time_downsampling_fpga(_sub_integration_ntime
                                              * voltage_meta->get_time_downsampling_fpga());
    tiles_meta->set_fpga_seq_num(seq0);
    tiles_meta->set_time_downsampling_fpga(_sub_integration_ntime
                                           * voltage_meta->get_time_downsampling_fpga());

    // The ringbuffering here is fishy. We should fix the kernel instead. [inherited note]

    const std::ptrdiff_t time_offset =
        voltage.get_read_valid().begin() % voltage.get_ndarray().extent(0);
    // Ensure there is no ring-buffer wrap-around
    assert((voltage.get_read_valid().end() - 1) % voltage.get_ndarray().extent(0) >= time_offset);
    const kotekan::int4x2_swapped_withoffset_t* const input_memory =
        &voltage.get_ndarray()(time_offset, 0, 0, 0);

    const std::ptrdiff_t rfi_time_offset =
        rfi_RFImask.get_read_valid().begin() % rfi_RFImask.get_ndarray().extent(0);
    // Ensure there is no ring-buffer wrap-around
    assert((rfi_RFImask.get_read_valid().end() - 1) % rfi_RFImask.get_ndarray().extent(0)
           >= rfi_time_offset);
    const kotekan::uint1x8_t* const rfi_RFImask_memory =
        &rfi_RFImask.get_ndarray()(rfi_time_offset, 0, 0);

    // aka "nt_outer" in n2k.hpp
    const int num_subintegrations = div_noremainder(_num_times, _sub_integration_ntime);

    // The synthetic input: this frame slot's array (no ring -- the injector runs earlier in
    // THIS command list, same stream, so its pack is complete before the kernel reads).
    const size_t synth_len = (size_t)_num_times * _num_local_freq * _num_synth;
    const int8_t* const synth_memory = (const int8_t*)device.get_gpu_memory_array(
        _gnss_synth_name, pipestate.gpu_frame_id, _gpu_buffer_depth, synth_len);

    // The extended triangle, per frame slot; never leaves the GPU.
    const auto& dp = dual_correlator.params;
    const size_t vis_len = (size_t)num_subintegrations * dp.vmat_tstride * sizeof(std::int32_t);
    std::int32_t* const d_vis = (std::int32_t*)device.get_gpu_memory_array(
        unique_name + "_vis", pipestate.gpu_frame_id, _gpu_buffer_depth, vis_len);

    const int2* d_sel = (const int2*)device.get_gpu_memory(unique_name + "_tile_sel",
                                                           _tile_sel.size() * sizeof(int2));

    record_start_event();
    const cudaStream_t stream = device.getStream(cuda_stream_id);

    dual_correlator.launch(d_vis, (const int8_t*)input_memory, synth_memory,
                           (const uint32_t*)rfi_RFImask_memory, num_subintegrations,
                           _sub_integration_ntime, stream, false);

    // SPLIT 1: the pure-antenna block is a verbatim prefix of each (t,f) slice -- one strided
    // copy into the standard-layout output. Width = the antenna triangle, src pitch = the
    // extended one.
    const size_t width_bytes =
        (size_t)num_triangle_blocks(_num_elements, 16) * 512 * sizeof(std::int32_t);
    const size_t src_pitch = (size_t)dp.vmat_fstride * sizeof(std::int32_t);
    CHECK_CUDA_ERROR(cudaMemcpy2DAsync(n2k_correlation.get_ndarray().data(), width_bytes, d_vis,
                                       src_pitch, width_bytes,
                                       (size_t)num_subintegrations * _num_local_freq,
                                       cudaMemcpyDeviceToDevice, stream));

    // SPLIT 2: the GNSS tiles of the comb channels only.
    CHECK_CUDA_ERROR(gnss_cuda::launch_gather_gnss_tiles(
        d_vis, d_sel, (int)_tile_sel.size(), num_subintegrations, dp.vmat_fstride,
        dp.vmat_tstride, gnss_tiles.get_ndarray().data(), stream));

    CHECK_CUDA_ERROR(cudaGetLastError());

    return record_end_event();
}

void cudaCorrelatorDual::finalize_frame() {
    // Advance the input ringbuffers
    voltage.finish_read();
    rfi_RFImask.finish_read();
    cudaCommand::finalize_frame();
}
