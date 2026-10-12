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

#include <algorithm>          // for find (a gather's channels in the comb)
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

// The live-antenna tile columns to gather. `live_element_tiles` lists them; absent, the
// columns 0..ceil(n_live/16)-1. A count alone cannot describe live elements that are not a
// prefix (e.g. {0..15, 64..79}, columns {0, 4}). Entries must be below num_elements/16, and
// ascending when AA tiles are gathered; neither is checked.
static std::vector<int> live_tile_columns(Config& config, const std::string& unique_name,
                                          int num_live_elements) {
    std::vector<int> cols =
        config.get_default<std::vector<int>>(unique_name, "live_element_tiles", {});
    if (cols.empty())
        for (int j = 0; j < (num_live_elements + 15) / 16; j++)
            cols.push_back(j);
    return cols;
}

// One gather: the tiles of `chans` (a subset of the comb) for the synth rows holding lanes
// [lane_base, lane_base + n_lanes). Several chains can share one correlator pass: each gather
// holds only its own chain's rows, so its buffer has the layout a one-chain correlator with
// num_synth = n_lanes would produce. The order is the tile list in the hpp, and
// GnssN2RecordAssemble recomputes it.
static std::vector<int2> build_tile_selection(const std::vector<std::int32_t>& comb,
                                              const std::vector<std::int32_t>& chans,
                                              int num_elements, const std::vector<int>& live_cols,
                                              bool compacted, int lane_base, int n_lanes,
                                              bool gather_aa, bool gather_bb) {
    // compacted: with a freq map the kernel writes slice k for comb[k], so the gather must
    // index by k, not by the channel number.
    std::vector<int2> sel;
    const int na16 = num_elements / 16;
    const int row_lo = na16 + lane_base / 16;
    const int row_hi = na16 + (lane_base + n_lanes) / 16;
    for (size_t ci = 0; ci < chans.size(); ci++) {
        std::int32_t f = chans[ci];
        if (compacted) {
            const auto it = std::find(comb.begin(), comb.end(), chans[ci]);
            if (it == comb.end())
                throw std::runtime_error("cudaCorrelatorDual: a gather asks for local channel "
                                         + std::to_string(chans[ci])
                                         + ", which is not in gnss_local_channels");
            f = (std::int32_t)(it - comb.begin());
        }
        // Mixed (synth row x live antenna column) first. GnssN2RecordAssemble reads the tile
        // of row ihi and list position slot at mix_k = (ihi - row_lo)*n_live_cols + slot. The
        // slot is the position in live_cols, not the column, so the gather order defines the
        // record's element axis and the consumer needs no knowledge of which columns are live.
        // Blocks appended after this one do not move the mixed tiles.
        for (int ihi = row_lo; ihi < row_hi; ihi++)
            for (size_t k = 0; k < live_cols.size(); k++)
                sel.push_back({f, 512 * (ihi * (ihi + 1) / 2 + live_cols[k])});
        // The tile count is one contract stated in three places: here, GnssN2RecordAssemble's
        // _n_mixed/_n_aa/_n_bb, and config/gen_chord_gnss_config.py's n2_tiles_per_chan, which
        // sizes the tiles buffer. A mismatch is fatal: GnssN2RecordAssemble checks the frame
        // size at startup, and gpuDeviceInterface's size check catches this side when
        // cudaOutputData first copies the buffer out.
        //
        // AA (antenna x antenna, the live N^2) next, only on request: the lower triangle over
        // positions in live_cols, row-major with k2 <= k1, so the (k1, k2) tile sits at
        // k1*(k1+1)/2 + k2 in this block. These are the triangle's tiles only if live_cols is
        // ascending. With gnss_freq_map the kernel writes the AA block only when BLOCK_MASK_AA
        // is in its mask; the constructor adds it iff a gather wants AA.
        if (gather_aa)
            for (size_t k1 = 0; k1 < live_cols.size(); k1++)
                for (size_t k2 = 0; k2 <= k1; k2++)
                    sel.push_back(
                        {f, 512 * (live_cols[k1] * (live_cols[k1] + 1) / 2 + live_cols[k2])});
        // BB (synth x synth) last, only on request: the lower triangle over this gather's synth
        // tile rows, (ihi - row_lo, jhi - row_lo) at k1*(k1+1)/2 + k2 in this block. The kernel
        // computes it regardless; only the visibility capture reads it (36 tiles per channel
        // at 128 lanes).
        if (gather_bb)
            for (int ihi = row_lo; ihi < row_hi; ihi++)
                for (int jhi = row_lo; jhi <= ihi; jhi++)
                    sel.push_back({f, 512 * (ihi * (ihi + 1) / 2 + jhi)});
    }
    return sel;
}

// Does any gather want the AA block? Needed before the kernel wrapper is built, since it sets
// the block-class mask.
static bool any_gather_aa(Config& config, const std::string& unique_name) {
    const std::vector<std::string> names =
        config.get_default<std::vector<std::string>>(unique_name, "gnss_gathers", {});
    if (names.empty())
        return config.get_default<bool>(unique_name, "gnss_gather_aa", false);
    for (const std::string& g : names)
        if (config.get_default<bool>(unique_name, "gather_" + g + "_aa", false))
            return true;
    return false;
}

// The comb as a freq map, or empty when the map is off. Built from one config.get: begin()
// and end() from two separate calls would point into two different temporaries.
static std::vector<int> freq_map_for(Config& config, const std::string& unique_name) {
    if (!config.get_default<bool>(unique_name, "gnss_freq_map", false))
        return {};
    const std::vector<std::int32_t> chans =
        config.get<std::vector<std::int32_t>>(unique_name, "gnss_local_channels");
    return std::vector<int>(chans.begin(), chans.end());
}

cudaCorrelatorDual::cudaCorrelatorDual(Config& config, const std::string& unique_name,
                                       bufferContainer& host_buffers, cudaDeviceInterface& device,
                                       const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst),
    _buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    _num_times(config.get<int>(unique_name, "num_times")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _num_synth(config.get_default<int>(unique_name, "num_synth", 128)),
    _num_live_elements(config.get_default<int>(unique_name, "num_live_elements", _num_elements)),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _sub_integration_ntime(config.get<int>(unique_name, "sub_integration_ntime")),
    _voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    _rfi_RFImask_name(config.get<std::string>(unique_name, "rfi_RFImask_name")),
    _n2k_correlation_name(config.get<std::string>(unique_name, "n2k_correlation_name")),
    _gnss_tiles_name(config.get<std::string>(unique_name, "gnss_tiles_name")),
    _gnss_synth_name(config.get_default<std::string>(unique_name, "gnss_synth_name", "gnss_synth")),
    _gnss_local_channels(config.get<std::vector<std::int32_t>>(unique_name, "gnss_local_channels")),
    _live_tile_cols(live_tile_columns(config, unique_name, _num_live_elements)),
    _synth_compact(config.get_default<bool>(unique_name, "gnss_synth_compact", false)),
    _rfi_all_pass(config.get_default<bool>(unique_name, "rfi_all_pass", false)),
    voltage(_voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{_buffer_depth * _num_times, _num_local_freq, 2,
                                          _num_elements / 2},
            std::array<std::string, 4>{"T", "F", "P", "D"},
            std::array<std::ptrdiff_t, 4>{1, 1, 1, 1}, *this),
    n2k_correlation([&]() {
        // Standard N^2 output, declared exactly as cudaCorrelator's so the same downstream
        // stages consume it.
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
    dual_correlator(n2k_dual::DualCorrelatorParams(
        _num_elements, _num_synth, _num_local_freq,
        // Freq-map mode: compute the mixed and synthetic blocks over only the GNSS comb, plus
        // the AA block when a gather wants the live antennas' own N^2 (the visibility
        // capture). Without gnss_freq_map the launch is the full triangle over every channel
        // and the N^2 prefix is available for the standard pipeline.
        config.get_default<bool>(unique_name, "gnss_freq_map", false)
            ? (n2k_dual::BLOCK_MASK_MIXED | n2k_dual::BLOCK_MASK_BB
               | (any_gather_aa(config, unique_name) ? n2k_dual::BLOCK_MASK_AA : 0))
            : n2k_dual::BLOCK_MASK_ALL,
        freq_map_for(config, unique_name),
        config.get_default<bool>(unique_name, "gnss_synth_compact", false))),
    _freq_map_mode(config.get_default<bool>(unique_name, "gnss_freq_map", false)) {
    if (_num_times % _sub_integration_ntime)
        throw std::runtime_error(
            "The sub_integration_ntime parameter must evenly divide samples_per_data_set");
    for (std::int32_t f : _gnss_local_channels)
        if (f < 0 || f >= _num_local_freq)
            throw std::runtime_error(
                "cudaCorrelatorDual: gnss_local_channels entry outside [0, num_local_freq) -- "
                "these are LOCAL frame indices, not global freq_ids");

    voltage.register_consumer();
    if (!_rfi_all_pass) {
        rfi_RFImask.emplace(
            _rfi_RFImask_name, "RFImask",
            std::array<std::ptrdiff_t, 3>{_buffer_depth * div_noremainder(_num_times, 8 * 128),
                                          _num_local_freq, 128},
            std::array<std::string, 3>{"T8hi128", "F", "T8lo128"},
            std::array<std::ptrdiff_t, 3>{1024, 1, 8}, *this);
        rfi_RFImask->register_consumer();
    } else {
        // Constant all-ones mask, as cudaRFISKtilde writes with first-stage excision off. The
        // layout does not matter when every bit is set.
        const size_t mask_bytes = (size_t)_num_times / 8 * _num_local_freq;
        void* d_mask = device.get_gpu_memory(unique_name + "_allpass_mask", mask_bytes);
        CHECK_CUDA_ERROR(cudaMemset(d_mask, 0xFF, mask_bytes));
    }

    gpu_buffers_used.push_back(std::make_tuple(_n2k_correlation_name, true, false, true));

    if (_synth_compact && !_freq_map_mode)
        throw std::runtime_error("cudaCorrelatorDual: gnss_synth_compact needs gnss_freq_map -- "
                                 "the compact synth array has one slice per COMB channel");

    // The gathers, one per output buffer: a single gather over the comb and every synth row
    // (gnss_tiles_name, gnss_gather_aa/bb), or one per name in gnss_gathers, each with its own
    // lane rows and channels.
    const int num_subintegrations = div_noremainder(_num_times, _sub_integration_ntime);
    const std::vector<std::string> gnames =
        config.get_default<std::vector<std::string>>(unique_name, "gnss_gathers", {});
    auto add_gather = [&](const std::string& tiles_name, const std::vector<std::int32_t>& chans,
                          int lane_base, int n_lanes, bool aa, bool bb) {
        if (lane_base < 0 || lane_base % 16 || n_lanes <= 0 || n_lanes % 16
            || lane_base + n_lanes > _num_synth)
            throw std::runtime_error("cudaCorrelatorDual: gather '" + tiles_name + "' lanes ["
                                     + std::to_string(lane_base) + ", "
                                     + std::to_string(lane_base + n_lanes)
                                     + ") are not whole tile rows inside num_synth");
        Gather g;
        g.name = tiles_name;
        g.n_chan = (int)chans.size();
        g.sel = build_tile_selection(_gnss_local_channels, chans, _num_elements, _live_tile_cols,
                                     _freq_map_mode, lane_base, n_lanes, aa, bb);
        const int tiles_per_chan = g.n_chan > 0 ? (int)g.sel.size() / g.n_chan : 0;
        const std::array<std::ptrdiff_t, 6> lengths{
            num_subintegrations, g.n_chan, tiles_per_chan, 16, 16, 2};
        const std::array<std::string, 6> dimnames{"Tc", "Fg", "Tile", "DPlo1", "DPlo2", "C"};
        const std::array<std::ptrdiff_t, 6> dimscalings{_sub_integration_ntime, 1, 1, 1, 1, 1};
        g.buf = std::make_unique<NDArrayBuffer<std::int32_t, 6>>(
            tiles_name, "gnss_n2_tiles", lengths, dimnames, dimscalings, *this);
        gpu_buffers_used.push_back(std::make_tuple(tiles_name, true, false, true));
        // Upload the tile-selection list once (constant for the run).
        g.d_sel = (int2*)device.get_gpu_memory(unique_name + "_tile_sel_" + tiles_name,
                                               g.sel.size() * sizeof(int2));
        CHECK_CUDA_ERROR(
            cudaMemcpy(g.d_sel, g.sel.data(), g.sel.size() * sizeof(int2), cudaMemcpyHostToDevice));
        _gathers.push_back(std::move(g));
    };
    if (gnames.empty()) {
        add_gather(_gnss_tiles_name, _gnss_local_channels, 0, _num_synth,
                   config.get_default<bool>(unique_name, "gnss_gather_aa", false),
                   config.get_default<bool>(unique_name, "gnss_gather_bb", false));
    } else {
        for (const std::string& g : gnames) {
            const std::string pre = "gather_" + g + "_";
            add_gather(config.get<std::string>(unique_name, pre + "tiles_name"),
                       config.get_default<std::vector<std::int32_t>>(unique_name, pre + "channels",
                                                                     _gnss_local_channels),
                       config.get_default<int>(unique_name, pre + "lane_base", 0),
                       config.get_default<int>(unique_name, pre + "lanes", _num_synth),
                       config.get_default<bool>(unique_name, pre + "aa", false),
                       config.get_default<bool>(unique_name, pre + "bb", false));
        }
    }

    // Zero-fill (0x88 == 0+0j offset-encoded) every synth frame slot once. The injectors
    // overwrite only their lanes and channels each frame, so unused lanes stay zero. Compact:
    // one slice per comb channel (the injectors' gnss_synth_channels), not per local channel.
    _synth_len = (size_t)_num_times
                 * (_synth_compact ? _gnss_local_channels.size() : _num_local_freq) * _num_synth;
    for (int s = 0; s < _buffer_depth; s++) {
        void* d_synth = device.get_gpu_memory_array(_gnss_synth_name, s, _buffer_depth, _synth_len);
        CHECK_CUDA_ERROR(cudaMemset(d_synth, 0x88, _synth_len));
    }

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaCorrelatorDual");
    INFO("cudaCorrelatorDual: {:d}+{:d} stations, {:d} freqs, {:d} gnss channels{:s}, {:d} "
         "gather(s) vs {:.1f} MB full triangle",
         _num_elements, _num_synth, _num_local_freq, (int)_gnss_local_channels.size(),
         _synth_compact ? " (compact synth)" : "", (int)_gathers.size(),
         (double)num_subintegrations * _num_local_freq * dual_correlator.params.vmat_fstride * 4
             / 1.0e6);
    for (const Gather& g : _gathers)
        INFO("cudaCorrelatorDual: gather '{:s}': {:d} channels x {:d} tiles ({:.2f} MB/frame)",
             g.name, g.n_chan, g.n_chan > 0 ? (int)g.sel.size() / g.n_chan : 0,
             g.sel.size() * 512 * 4 * num_subintegrations / 1.0e6);
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

    if (rfi_RFImask) {
        DEBUG("Waiting for rfi_RFImask input ringbuffer data for frame {:d}...", gpu_frame_id);
        const int rfi_RFImask_errcode =
            rfi_RFImask->wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
                const int rfi_needed_samples = div_noremainder(_num_times, 8 * 128);
                if (available_elements < rfi_needed_samples)
                    return read_descriptor_t{.claimed = 0, .read = 0};
                else
                    return read_descriptor_t{.claimed = rfi_needed_samples,
                                             .read = rfi_needed_samples};
            });
        if (rfi_RFImask_errcode < 0)
            return rfi_RFImask_errcode;
        DEBUG("Finished waiting for rfi_RFImask input for data frame {:d}.", gpu_frame_id);
    }

    return 0;
}

cudaEvent_t cudaCorrelatorDual::execute(cudaPipelineState& pipestate,
                                        const std::vector<cudaEvent_t>&) {
    pre_execute();

    voltage.check_metadata();
    if (rfi_RFImask)
        rfi_RFImask->check_metadata();
    n2k_correlation.set_metadata(voltage.get_metadata());
    for (Gather& g : _gathers)
        g.buf->set_metadata(voltage.get_metadata());

    const std::shared_ptr<const chordMetadata> voltage_meta = voltage.get_metadata();
    const std::shared_ptr<chordMetadata> n2k_corr_meta = n2k_correlation.get_metadata();

    // Ensure consistency:
    if (rfi_RFImask) {
        const std::shared_ptr<const chordMetadata> rfi_meta = rfi_RFImask->get_metadata();
        assert(voltage_meta->get_fpga_seq_num()
                   + voltage.get_read_valid().begin() * voltage_meta->get_time_downsampling_fpga()
               == rfi_meta->get_fpga_seq_num()
                      + rfi_RFImask->get_read_valid().begin()
                            * rfi_meta->get_time_downsampling_fpga());
        (void)rfi_meta;
    }

    // The input ringbuffer metadata do not contain time-dependent metadata,
    // so we must reconstruct it here. (fpga_seq_num)
    const auto seq0 =
        voltage_meta->get_fpga_seq_num()
        + voltage.get_read_valid().begin() * voltage_meta->get_time_downsampling_fpga();
    n2k_corr_meta->set_fpga_seq_num(seq0);
    n2k_corr_meta->set_time_downsampling_fpga(_sub_integration_ntime
                                              * voltage_meta->get_time_downsampling_fpga());
    for (Gather& g : _gathers) {
        const std::shared_ptr<chordMetadata> tiles_meta = g.buf->get_metadata();
        tiles_meta->set_fpga_seq_num(seq0);
        tiles_meta->set_time_downsampling_fpga(_sub_integration_ntime
                                               * voltage_meta->get_time_downsampling_fpga());
    }

    // The ringbuffering here is fishy. We should fix the kernel instead.

    const std::ptrdiff_t time_offset =
        voltage.get_read_valid().begin() % voltage.get_ndarray().extent(0);
    // Ensure there is no ring-buffer wrap-around
    assert((voltage.get_read_valid().end() - 1) % voltage.get_ndarray().extent(0) >= time_offset);
    const kotekan::int4x2_swapped_withoffset_t* const input_memory =
        &voltage.get_ndarray()(time_offset, 0, 0, 0);

    const kotekan::uint1x8_t* rfi_RFImask_memory = nullptr;
    if (rfi_RFImask) {
        const std::ptrdiff_t rfi_time_offset =
            rfi_RFImask->get_read_valid().begin() % rfi_RFImask->get_ndarray().extent(0);
        // Ensure there is no ring-buffer wrap-around
        assert((rfi_RFImask->get_read_valid().end() - 1) % rfi_RFImask->get_ndarray().extent(0)
               >= rfi_time_offset);
        rfi_RFImask_memory = &rfi_RFImask->get_ndarray()(rfi_time_offset, 0, 0);
    } else {
        const size_t mask_bytes = (size_t)_num_times / 8 * _num_local_freq;
        rfi_RFImask_memory = (const kotekan::uint1x8_t*)device.get_gpu_memory(
            unique_name + "_allpass_mask", mask_bytes);
    }

    // aka "nt_outer" in n2k.hpp
    const int num_subintegrations = div_noremainder(_num_times, _sub_integration_ntime);

    // The synthetic input: this frame slot's array, not a ring. The injectors must run earlier
    // in this command list on the same stream, so that their packs are complete before the
    // kernel reads them.
    const int8_t* const synth_memory = (const int8_t*)device.get_gpu_memory_array(
        _gnss_synth_name, pipestate.gpu_frame_id, _gpu_buffer_depth, _synth_len);

    // The extended triangle, per frame slot; never leaves the GPU.
    const auto& dp = dual_correlator.params;
    // Per-subintegration stride of the output: n_freq_out slices (dp.vmat_tstride counts
    // nfreq).
    const size_t vis_tstride = (size_t)dp.n_freq_out * dp.vmat_fstride;
    const size_t vis_len = (size_t)num_subintegrations * vis_tstride * sizeof(std::int32_t);
    std::int32_t* const d_vis = (std::int32_t*)device.get_gpu_memory_array(
        unique_name + "_vis", pipestate.gpu_frame_id, _gpu_buffer_depth, vis_len);

    record_start_event();
    const cudaStream_t stream = device.getStream(cuda_stream_id);

    dual_correlator.launch(d_vis, (const int8_t*)input_memory, synth_memory,
                           (const uint32_t*)rfi_RFImask_memory, num_subintegrations,
                           _sub_integration_ntime, stream, false);

    // Split 1: the pure-antenna block is a verbatim prefix of each (t,f) slice, so one strided
    // copy fills the standard-layout output. Skipped in freq-map mode, where the triangle
    // covers only the comb channels (and holds AA only for a gather that asked).
    if (!_freq_map_mode) {
        const size_t width_bytes =
            (size_t)num_triangle_blocks(_num_elements, 16) * 512 * sizeof(std::int32_t);
        const size_t src_pitch = (size_t)dp.vmat_fstride * sizeof(std::int32_t);
        CHECK_CUDA_ERROR(cudaMemcpy2DAsync(
            n2k_correlation.get_ndarray().data(), width_bytes, d_vis, src_pitch, width_bytes,
            (size_t)num_subintegrations * _num_local_freq, cudaMemcpyDeviceToDevice, stream));
    }

    // Split 2: the GNSS tiles of the comb channels, one gather per output buffer.
    for (Gather& g : _gathers)
        CHECK_CUDA_ERROR(gnss_cuda::launch_gather_gnss_tiles(
            d_vis, g.d_sel, (int)g.sel.size(), num_subintegrations, dp.vmat_fstride,
            (int)vis_tstride, g.buf->get_ndarray().data(), stream));

    CHECK_CUDA_ERROR(cudaGetLastError());

    return record_end_event();
}

void cudaCorrelatorDual::finalize_frame() {
    // Advance the input ringbuffers
    voltage.finish_read();
    if (rfi_RFImask)
        rfi_RFImask->finish_read();
    cudaCommand::finalize_frame();
}
