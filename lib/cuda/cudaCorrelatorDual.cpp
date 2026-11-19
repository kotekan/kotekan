#include "cudaCorrelatorDual.hpp"

#include "DataType.hpp"            // for int4x2_swapped_withoffset_t, uint1x8_t
#include "NDArray.hpp"             // for NDArray
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer, extent_t, read_descriptor_t
#include "cudaCommand.hpp"         // for cudaCommand, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "div.hpp"                 // for div_noremainder
#include "gnssGpuRows.hpp"         // for GpuRow, gpu_rows
#include "gnssTileSelection.hpp"   // for build_tile_selection
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

REGISTER_CUDA_COMMAND(cudaCorrelatorDual);

// The live-antenna tile columns to gather, as `live_element_tiles` lists them: a count alone
// cannot describe live elements that are not a prefix (e.g. {0..15, 64..79}, columns {0, 4}).
// There must be one column per 16 of num_live_elements. Entries must be below
// num_elements/16, and ascending when AA tiles are gathered; neither is checked.
static std::vector<int> live_tile_columns(Config& config, const std::string& unique_name) {
    const std::vector<int> cols = config.get<std::vector<int>>(unique_name, "live_element_tiles");
    const int num_live_elements = config.get<int>(unique_name, "num_live_elements");
    if ((int)cols.size() != (num_live_elements + 15) / 16)
        throw std::runtime_error("cudaCorrelatorDual: live_element_tiles lists "
                                 + std::to_string(cols.size()) + " columns, but num_live_elements "
                                 + std::to_string(num_live_elements) + " needs "
                                 + std::to_string((num_live_elements + 15) / 16));
    return cols;
}

// Does any gather want the AA block? Needed before the kernel wrapper is built, since it sets
// the block-class mask.
static bool any_gather_aa(Config& config, const std::string& unique_name) {
    const std::vector<std::string> names =
        config.get_default<std::vector<std::string>>(unique_name, "gnss_gathers", {});
    for (const std::string& g : names)
        if (config.get_default<bool>(unique_name, "gather_" + g + "_aa", false))
            return true;
    return false;
}

// The comb as the kernel's freq map. Built from one config.get: begin() and end() from two
// separate calls would point into two different temporaries.
static std::vector<int> freq_map_for(Config& config, const std::string& unique_name) {
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
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _sub_integration_ntime(config.get<int>(unique_name, "sub_integration_ntime")),
    _voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    _gnss_synth_name(config.get_default<std::string>(unique_name, "gnss_synth_name", "gnss_synth")),
    _gnss_local_channels(config.get<std::vector<std::int32_t>>(unique_name, "gnss_local_channels")),
    _live_tile_cols(live_tile_columns(config, unique_name)),
    voltage(_voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{_buffer_depth * _num_times, _num_local_freq, 2,
                                          _num_elements / 2},
            std::array<std::string, 4>{"T", "F", "P", "D"},
            std::array<std::ptrdiff_t, 4>{1, 1, 1, 1}, *this),
    dual_correlator(n2k_dual::DualCorrelatorParams(
        _num_elements, _num_synth, _num_local_freq,
        // The mixed and synthetic blocks over only the GNSS comb, plus the AA block when a
        // gather wants the live antennas' own N^2 (the visibility capture).
        n2k_dual::BLOCK_MASK_MIXED | n2k_dual::BLOCK_MASK_BB
            | (any_gather_aa(config, unique_name) ? n2k_dual::BLOCK_MASK_AA : 0),
        freq_map_for(config, unique_name), true)) {
    // The live configs still carry these keys. The three flags name the only layout this
    // command has, so they are read to refuse false; the three names have no effect and are
    // read only so a usage report does not list them unused.
    for (const char* key : {"gnss_freq_map", "gnss_synth_compact", "rfi_all_pass"})
        if (!config.get_default<bool>(unique_name, key, true))
            throw std::runtime_error(std::string("cudaCorrelatorDual: ") + key
                                     + " false is not supported; it must be true");
    for (const char* key : {"n2k_correlation_name", "rfi_RFImask_name", "gnss_tiles_name"})
        config.get_default<std::string>(unique_name, key, "");

    if (_num_times % _sub_integration_ntime)
        throw std::runtime_error(
            "The sub_integration_ntime parameter must evenly divide samples_per_data_set");
    for (std::int32_t f : _gnss_local_channels)
        if (f < 0 || f >= _num_local_freq)
            throw std::runtime_error(
                "cudaCorrelatorDual: gnss_local_channels entry outside [0, num_local_freq) -- "
                "these are LOCAL frame indices, not global freq_ids");

    voltage.register_consumer();

    // Constant all-ones RFI mask, as cudaRFISKtilde writes with first-stage excision off. The
    // layout does not matter when every bit is set.
    const size_t mask_bytes = (size_t)_num_times / 8 * _num_local_freq;
    void* d_mask = device.get_gpu_memory(unique_name + "_allpass_mask", mask_bytes);
    CHECK_CUDA_ERROR(cudaMemset(d_mask, 0xFF, mask_bytes));

    // The gathers, one per name in gnss_gathers and per output buffer, each with its own lane
    // rows and channels.
    const int num_subintegrations = div_noremainder(_num_times, _sub_integration_ntime);
    // A GPU half's gnss_rows (gnssGpuRows.hpp) replaces gnss_gathers.
    const bool rows =
        !config.get_default<nlohmann::json>(unique_name, "gnss_rows", nlohmann::json()).is_null();
    if (rows && config.exists(unique_name, "gnss_gathers"))
        throw std::runtime_error("cudaCorrelatorDual: set gnss_gathers or gnss_rows, not both");
    const std::vector<std::string> gnames =
        rows ? std::vector<std::string>{}
             : config.get<std::vector<std::string>>(unique_name, "gnss_gathers");
    if (!rows && gnames.empty())
        throw std::runtime_error("cudaCorrelatorDual: gnss_gathers is empty; it must name at "
                                 "least one gather");
    // Each gather selects whole tile rows inside num_synth.
    auto select = [&](const std::string& tiles_name, const std::vector<std::int32_t>& chans,
                      int lane_base, int n_lanes, bool aa, bool bb) {
        if (lane_base < 0 || lane_base % 16 || n_lanes <= 0 || n_lanes % 16
            || lane_base + n_lanes > _num_synth)
            throw std::runtime_error("cudaCorrelatorDual: gather '" + tiles_name + "' lanes ["
                                     + std::to_string(lane_base) + ", "
                                     + std::to_string(lane_base + n_lanes)
                                     + ") are not whole tile rows inside num_synth");
        return build_tile_selection<int2>(_gnss_local_channels, chans, _num_elements,
                                          _live_tile_cols, lane_base, n_lanes, aa, bb);
    };
    auto add_gather = [&](const std::string& tiles_name, std::vector<int2> sel, int n_chan) {
        Gather g;
        g.name = tiles_name;
        g.n_chan = n_chan;
        g.sel = std::move(sel);
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
    // gnss_rows: one gather over every row, in list order on one channel axis, so a row's tiles
    // are channels [fg_offset, fg_offset + its channels) of every Tc slice. The rows must gather
    // the same tile count per channel.
    if (rows) {
        const std::string tiles_name = config.get<std::string>(unique_name, "gnss_rows_tiles_name");
        const std::vector<gnss_gpu::GpuRow> grows = gnss_gpu::gpu_rows(config, unique_name);
        std::vector<int2> sel;
        int n_chan = 0;
        for (const gnss_gpu::GpuRow& r : grows) {
            if (r.lanes != grows[0].lanes)
                throw std::runtime_error("cudaCorrelatorDual: gnss_rows " + r.name + " has "
                                         + std::to_string(r.lanes)
                                         + " lanes, unlike the first row: one gather needs one "
                                           "tile count per channel");
            const std::vector<int2> s =
                select(tiles_name, r.channels, r.lane_base, r.lanes, false, false);
            sel.insert(sel.end(), s.begin(), s.end());
            n_chan += (int)r.channels.size();
        }
        add_gather(tiles_name, std::move(sel), n_chan);
    }
    for (const std::string& g : gnames) {
        const std::string pre = "gather_" + g + "_";
        const std::string tiles_name = config.get<std::string>(unique_name, pre + "tiles_name");
        const std::vector<std::int32_t> chans = config.get_default<std::vector<std::int32_t>>(
            unique_name, pre + "channels", _gnss_local_channels);
        add_gather(tiles_name,
                   select(tiles_name, chans,
                          config.get_default<int>(unique_name, pre + "lane_base", 0),
                          config.get_default<int>(unique_name, pre + "lanes", _num_synth),
                          config.get_default<bool>(unique_name, pre + "aa", false),
                          config.get_default<bool>(unique_name, pre + "bb", false)),
                   (int)chans.size());
    }

    // Zero-fill (0x88 == 0+0j offset-encoded) every synth frame slot once. The injectors
    // overwrite only their lanes and channels each frame, so unused lanes stay zero. One
    // slice per comb channel (the injectors' gnss_synth_channels), not per local channel.
    _synth_len = (size_t)_num_times * _gnss_local_channels.size() * _num_synth;
    for (int s = 0; s < _buffer_depth; s++) {
        void* d_synth = device.get_gpu_memory_array(_gnss_synth_name, s, _buffer_depth, _synth_len);
        CHECK_CUDA_ERROR(cudaMemset(d_synth, 0x88, _synth_len));
    }

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaCorrelatorDual");
    INFO("cudaCorrelatorDual: {:d}+{:d} stations, {:d} freqs, {:d} gnss channels (compact "
         "synth), {:d} gather(s) vs {:.1f} MB full triangle",
         _num_elements, _num_synth, _num_local_freq, (int)_gnss_local_channels.size(),
         (int)_gathers.size(),
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

    return 0;
}

cudaEvent_t cudaCorrelatorDual::execute(cudaPipelineState& pipestate,
                                        const std::vector<cudaEvent_t>&) {
    pre_execute();

    voltage.check_metadata();
    for (Gather& g : _gathers)
        g.buf->set_metadata(voltage.get_metadata());

    const std::shared_ptr<const chordMetadata> voltage_meta = voltage.get_metadata();

    // The input ringbuffer metadata do not contain time-dependent metadata,
    // so we must reconstruct it here. (fpga_seq_num)
    const auto seq0 =
        voltage_meta->get_fpga_seq_num()
        + voltage.get_read_valid().begin() * voltage_meta->get_time_downsampling_fpga();
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

    const size_t mask_bytes = (size_t)_num_times / 8 * _num_local_freq;
    const kotekan::uint1x8_t* const rfi_RFImask_memory =
        (const kotekan::uint1x8_t*)device.get_gpu_memory(unique_name + "_allpass_mask", mask_bytes);

    // aka "nt_outer" in n2k.hpp
    const int num_subintegrations = div_noremainder(_num_times, _sub_integration_ntime);

    // The synthetic input: this frame slot's array, not a ring. The injectors must run earlier
    // in this command list on the same stream, so that their packs are complete before the
    // kernel reads them.
    const int8_t* const synth_memory = (const int8_t*)device.get_gpu_memory_array(
        _gnss_synth_name, pipestate.gpu_frame_id, _gpu_buffer_depth, _synth_len);

    // The extended triangle, per frame slot; never leaves the GPU.
    const auto& dp = dual_correlator.params;
    // Per-subintegration stride of the output: n_freq_out slices.
    const size_t vis_tstride = (size_t)dp.n_freq_out * dp.vmat_fstride;
    const size_t vis_len = (size_t)num_subintegrations * vis_tstride * sizeof(std::int32_t);
    std::int32_t* const d_vis = (std::int32_t*)device.get_gpu_memory_array(
        unique_name + "_vis", pipestate.gpu_frame_id, _gpu_buffer_depth, vis_len);

    record_start_event();
    const cudaStream_t stream = device.getStream(cuda_stream_id);

    dual_correlator.launch(d_vis, (const int8_t*)input_memory, synth_memory,
                           (const uint32_t*)rfi_RFImask_memory, num_subintegrations,
                           _sub_integration_ntime, stream, false);

    // The GNSS tiles of the comb channels, one gather per output buffer.
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
    cudaCommand::finalize_frame();
}
