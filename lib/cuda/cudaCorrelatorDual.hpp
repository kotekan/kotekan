#ifndef KOTEKAN_CUDA_CORRELATOR_DUAL_H
#define KOTEKAN_CUDA_CORRELATOR_DUAL_H

#include "Config.hpp"                  // for Config
#include "DataType.hpp"                // for int4x2_swapped_withoffset_t
#include "NDArrayBuffer.hpp"           // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"       // for NDArrayRingBuffer
#include "bufferContainer.hpp"         // for bufferContainer
#include "cudaCommand.hpp"             // for cudaCommand, cudaPipelineState
#include "cudaDeviceInterface.hpp"     // for cudaDeviceInterface
#include "driver_types.h"              // for cudaEvent_t
#include "n2k_dual/DualCorrelator.hpp" // for DualCorrelator

#include <cstdint>  // for int32_t, uint32_t
#include <memory>   // for unique_ptr (the gathers' buffers)
#include <optional> // for optional (rfi_all_pass)
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class cudaCorrelatorDual
 * @brief cudaCorrelator's N^2 plus synthetic GNSS replica stations, in one launch.
 *
 * Drives external/n2k_dual (see its README.md): the visibility triangle over the concatenated
 * station axis [antennas | synthetic]. The synthetic input is a separate per-frame-slot GPU
 * array (@c gnss_synth_name) that cudaGnssInject fills with quantized replica waveforms.
 *
 * The output is split on the GPU:
 *  - Without @c gnss_freq_map, the pure-antenna block is a verbatim prefix of each (t,f) slice
 *    of the extended triangle, so the standard @c n2k_correlation output is one strided
 *    cudaMemcpy2DAsync, shaped as cudaCorrelator's and bit-identical to it whatever the
 *    synthetic input holds (scripts/gnss/n2dualtest.cpp checks this). With @c gnss_freq_map
 *    the extended triangle covers only the comb channels and @c n2k_correlation is not
 *    written.
 *  - The GNSS tiles of the comb channels are gathered into small tile buffers; the extended
 *    triangle never leaves the GPU.
 *
 * Tile list (fixed at construction; GnssN2RecordAssemble recomputes it from its own config).
 * Per channel of a gather, up to three blocks in this order, each row-major. The gather's
 * synthetic tile rows are ihi in [na16 + lane_base/16, na16 + (lane_base + lanes)/16), with
 * na16 = num_elements/16.
 *  - mixed: each synthetic row against the @c live_element_tiles columns, in listed order;
 *  - if aa: the live antennas' lower triangle by list position, (k1, k2) at k1*(k1+1)/2 + k2;
 *  - if bb: the synthetic lower triangle over the gather's rows.
 * Each tile is [16][16][2] int32 at offset 512*(ihi*(ihi+1)/2 + jhi) within a (t,f) slice,
 * n2k's layout. V_ij = E_i * conj(E_j) with i the row, so a mixed tile is
 * replica * conj(antenna); cudaGnssInject's @c conjugate packs conj(replica), which makes the
 * mixed tile the despread sum itself.
 *
 * The synthetic array is [num_times][num_local_freq][num_synth] int8 (4+4 bit, offset-encoded)
 * per frame slot, or [num_times][comb][num_synth] with @c gnss_synth_compact. This command
 * zero-fills (0x88) every slot once at construction; the injectors overwrite only their lanes
 * and channels each frame. Samples must be in [-7,7]: the value -8 silently corrupts the
 * results (n2k's negate_4bit), so the injectors' quantizer clamps.
 *
 * @conf As cudaCorrelator (buffer_depth, num_times, num_elements, num_local_freq,
 *       sub_integration_ntime, voltage_name, rfi_RFImask_name, n2k_correlation_name), plus:
 * @conf  num_synth            Int. Synthetic stations (multiple of 128). Default 128.
 * @conf  num_live_elements    Int. Live antennas; sets the default live_element_tiles.
 *                             Default num_elements.
 * @conf  live_element_tiles   List of Int. The live-antenna tile columns (16 elements each)
 *                             to gather, in the order they occupy the record's element axis.
 *                             Default 0..ceil(num_live_elements/16)-1. Set it whenever the
 *                             live elements are not a prefix: live elements {0..15, 64..79}
 *                             are columns [0, 4]. Its length must equal
 *                             ceil(num_live_elements/16) or the consumer's mix_k indexing and
 *                             this gather disagree. It must be ascending when AA tiles are
 *                             gathered.
 * @conf  gnss_local_channels  List of Int. Local frame channel indices, in
 *                             [0, num_local_freq), of the GNSS comb; not global freq_ids.
 * @conf  gnss_freq_map        Bool, default false. Compute only the comb channels, and only
 *                             the mixed and BB blocks (plus AA if a gather asks for it).
 * @conf  gnss_gather_aa       Bool, default false. Also gather the live antennas' own N^2
 *                             tiles (lower triangle over live_element_tiles, after the mixed
 *                             block) and, with gnss_freq_map, compute the AA block on the comb
 *                             channels. For the visibility capture; the tracker never reads
 *                             these tiles.
 * @conf  gnss_gather_bb       Bool, default false. Also gather the BB (synth x synth) lower
 *                             triangle, after the AA block: the replica side of the
 *                             capture's (N+M)^2. The block is computed either way.
 * @conf  gnss_tiles_name      String. Name of the gathered-tiles output buffer when
 *                             gnss_gathers is empty; required, but unused, otherwise.
 * @conf  gnss_gathers         List of String, default empty. Several chains in one pass: the
 *                             synth axis holds every chain's lanes at its own offset and each
 *                             named gather copies out one buffer. Per name G:
 *                             gather_G_tiles_name (String), gather_G_channels (local, a
 *                             subset of gnss_local_channels; default all), gather_G_lane_base
 *                             and gather_G_lanes (default 0 and num_synth; whole tile rows),
 *                             gather_G_aa, gather_G_bb (Bool). Tile order inside a buffer is
 *                             the tile list above restricted to those rows. When the list is
 *                             empty, gnss_gather_aa, gnss_gather_bb and gnss_tiles_name
 *                             describe a single gather over every lane.
 * @conf  gnss_synth_compact   Bool, default false. The synth array is [T][comb][num_synth],
 *                             one slice per gnss_local_channels entry in that order, instead
 *                             of [T][num_local_freq][num_synth]. Needs gnss_freq_map; the
 *                             injectors must write the same axis (gnss_synth_channels).
 * @conf  gnss_synth_name      String. Name of the synthetic-input GPU array. Default
 *                             "gnss_synth".
 * @conf  rfi_all_pass         Bool, default false. True: consume no RFI-mask ring and use a
 *                             constant all-ones mask. This matches the real mask on channels
 *                             where first-stage excision is off or exempt (cudaRFISKtilde's
 *                             rfi_first_stage_excision_exempt_* keys).
 */
class cudaCorrelatorDual : public cudaCommand {
public:
    cudaCorrelatorDual(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                       int inst);
    ~cudaCorrelatorDual();
    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    // Common configuration values (which do not change in a run)
    const std::int32_t _buffer_depth;
    const std::int32_t _num_times;
    const std::int32_t _num_elements; // antennas (A input), CHORD pathfinder = 128
    const std::int32_t _num_synth;    // synthetic stations (B input)
    const std::int32_t _num_live_elements;
    const std::int32_t _num_local_freq;
    const std::int32_t _sub_integration_ntime;

    const std::string _voltage_name;
    const std::string _rfi_RFImask_name;
    const std::string _n2k_correlation_name;
    const std::string _gnss_tiles_name;
    const std::string _gnss_synth_name;

    std::vector<std::int32_t> _gnss_local_channels;

    /// The live-antenna tile columns gathered, in output order; not necessarily a prefix.
    /// Positions in this list define the record's element axis (see build_tile_selection).
    const std::vector<int> _live_tile_cols;

    /// gnss_synth_compact: the synth array has one slice per comb channel.
    const bool _synth_compact;

    /// One gathered-tile output: a selection over the extended triangle into its own buffer.
    struct Gather {
        std::string name;
        std::vector<int2> sel; ///< (slice, int32 offset within the slice) per tile, in order
        int n_chan = 0;
        int2* d_sel = nullptr;
        std::unique_ptr<NDArrayBuffer<std::int32_t, 6>> buf; // [Tc][chan][tile][16][16][2]
    };
    std::vector<Gather> _gathers;
    size_t _synth_len = 0; ///< bytes per synth frame slot (set in the constructor)

    const bool _rfi_all_pass;

    /// Signaling ring buffer for the input (voltage) data.
    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;
    /// Absent when rfi_all_pass (a constant all-ones device mask is used instead).
    std::optional<NDArrayRingBuffer<kotekan::uint1x8_t, 3>> rfi_RFImask;
    NDArrayBuffer<std::int32_t, 6> n2k_correlation; // standard N^2 out, shape as cudaCorrelator

    /// Cuda kernel wrapper object
    n2k_dual::DualCorrelator dual_correlator;

    /// gnss_freq_map: the extended triangle covers only the comb channels, so there is no
    /// standard N^2 to pass through.
    const bool _freq_map_mode;
};

namespace gnss_cuda {
/// Copy the selected (freq, tile) list out of the extended visibility triangle.
/// sel[k].x = output slice index, sel[k].y = int32 offset of the tile within that slice.
/// out layout: [nt_outer][n_sel][512] int32.
cudaError_t launch_gather_gnss_tiles(const std::int32_t* vis, const int2* sel, int n_sel,
                                     int nt_outer, int vmat_fstride, int vmat_tstride,
                                     std::int32_t* out, cudaStream_t stream);
} // namespace gnss_cuda

#endif // KOTEKAN_CUDA_CORRELATOR_DUAL_H
