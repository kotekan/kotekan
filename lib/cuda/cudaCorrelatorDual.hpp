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

#include <cstdint> // for int32_t, uint32_t
#include <memory>  // for unique_ptr (the gathers' buffers)
#include <string>  // for string
#include <vector>  // for vector

/**
 * @class cudaCorrelatorDual
 * @brief cudaCorrelator's N^2 plus synthetic GNSS replica stations, in one launch.
 *
 * Drives external/n2k_dual (see its README.md): the visibility triangle over the concatenated
 * station axis [antennas | synthetic]. The synthetic input is a separate per-frame-slot GPU
 * array (@c gnss_synth_name) that cudaGnssInject fills with quantized replica waveforms.
 *
 * The extended triangle covers only the GNSS comb channels (@c gnss_local_channels, as the
 * kernel's freq map) and only the mixed and BB blocks, plus AA when a gather asks for it. It
 * never leaves the GPU: each gather copies the tiles of its chain's comb channels into a small
 * tile buffer. The standard N^2 comes from cudaCorrelator, not from here.
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
 * The synthetic array is [num_times][comb][num_synth] int8 (4+4 bit, offset-encoded) per frame
 * slot, one slice per @c gnss_local_channels entry in that order; the injectors write the
 * same axis (their gnss_synth_channels). This command zero-fills (0x88) every slot once at
 * construction; the injectors overwrite only their lanes and channels each frame. Samples
 * must be in [-7,7]: the value -8 silently corrupts the results (n2k's negate_4bit), so the
 * injectors' quantizer clamps.
 *
 * No RFI-mask ring is consumed: the kernel gets a constant all-ones mask. This matches the
 * real mask on channels where first-stage excision is off or exempt (cudaRFISKtilde's
 * rfi_first_stage_excision_exempt_* keys).
 *
 * @conf As cudaCorrelator (buffer_depth, num_times, num_elements, num_local_freq,
 *       sub_integration_ntime, voltage_name), plus:
 * @conf  num_synth            Int. Synthetic stations (multiple of 128). Default 128.
 * @conf  num_live_elements    Int. Live antennas; live_element_tiles must list
 *                             ceil(num_live_elements/16) columns.
 * @conf  live_element_tiles   List of Int. The live-antenna tile columns (16 elements each)
 *                             to gather, in the order they occupy the record's element axis:
 *                             live elements {0..15, 64..79} are columns [0, 4]. Its length
 *                             must equal ceil(num_live_elements/16) or the consumer's mix_k
 *                             indexing and this gather disagree. It must be ascending when
 *                             AA tiles are gathered.
 * @conf  gnss_local_channels  List of Int. Local frame channel indices, in
 *                             [0, num_local_freq), of the GNSS comb; not global freq_ids.
 * @conf  gnss_gathers         List of String, not empty. Several chains in one pass: the
 *                             synth axis holds every chain's lanes at its own offset and each
 *                             named gather copies out one buffer. Per name G:
 *                             gather_G_tiles_name (String), gather_G_channels (local, a
 *                             subset of gnss_local_channels; default all), gather_G_lane_base
 *                             and gather_G_lanes (default 0 and num_synth; whole tile rows),
 *                             gather_G_aa, gather_G_bb (Bool, default false: the live
 *                             antennas' own N^2 tiles, and the BB lower triangle over the
 *                             gather's rows, for the visibility capture; the tracker never
 *                             reads them). Tile order inside a buffer is the tile list above
 *                             restricted to those rows. The BB block is computed either way.
 * @conf  gnss_synth_name      String. Name of the synthetic-input GPU array. Default
 *                             "gnss_synth".
 * @conf  gnss_freq_map, gnss_synth_compact, rfi_all_pass
 *                             Bool, default true; false is refused. They name the layout
 *                             above, the only one this command has.
 * @conf  n2k_correlation_name, rfi_RFImask_name, gnss_tiles_name
 *                             String, ignored.
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
    const std::int32_t _num_local_freq;
    const std::int32_t _sub_integration_ntime;

    const std::string _voltage_name;
    const std::string _gnss_synth_name;

    std::vector<std::int32_t> _gnss_local_channels;

    /// The live-antenna tile columns gathered, in output order; not necessarily a prefix.
    /// Positions in this list define the record's element axis (see build_tile_selection).
    const std::vector<int> _live_tile_cols;

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

    /// Signaling ring buffer for the input (voltage) data.
    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;

    /// Cuda kernel wrapper object
    n2k_dual::DualCorrelator dual_correlator;
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
