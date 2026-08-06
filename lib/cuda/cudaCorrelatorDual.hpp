#ifndef KOTEKAN_CUDA_CORRELATOR_DUAL_H
#define KOTEKAN_CUDA_CORRELATOR_DUAL_H

#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for int4x2_swapped_withoffset_t
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "driver_types.h"          // for cudaEvent_t
#include "n2k_dual/DualCorrelator.hpp" // for DualCorrelator

#include <cstdint>  // for int32_t, uint32_t
#include <optional> // for optional (rfi_all_pass)
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class cudaCorrelatorDual
 * @brief GNSS path-B N^2: cudaCorrelator's job PLUS synthetic replica stations, one launch.
 *
 * A deliberately separate command -- cudaCorrelator and external/n2k are NOT modified (others
 * develop them). This clone drives external/n2k_dual: the visibility triangle over the
 * CONCATENATED station axis [antennas | synthetic], where the synthetic input is a separate
 * per-frame-slot GPU array (named @c gnss_synth_name) filled by cudaGnssInject with quantized
 * GNSS replica waveforms. See docs/gnss_gpu_search.md section 11.
 *
 * The output is split on the GPU:
 *  - The pure-antenna block is a VERBATIM PREFIX of each (t,f) slice of the extended
 *    triangle (tile packing orders by row), so the standard `n2k_correlation` output is one
 *    strided cudaMemcpy2DAsync -- downstream (cudaOutputData -> N2Accumulate) is unchanged
 *    and shape-identical to cudaCorrelator's. With the synthetic input all-zero (0x88) the
 *    prefix is BIT-IDENTICAL to what cudaCorrelator produces (verified: n2dualtest gate [3]).
 *  - The GNSS tiles -- mixed (synthetic x live-antenna) and synthetic x synthetic -- are
 *    gathered for ONLY the configured GNSS comb channels into a small `gnss_tiles` output
 *    (see the tile-list note below), ~0.7 MB/frame against the 107 MB extended triangle,
 *    which never leaves the GPU.
 *
 * Tile list (fixed at construction, consumer recomputes it from the same config): for each
 * local channel in @c gnss_local_channels, mixed tiles (ihi in [NA/16, NT/16), jhi in
 * [0, ceil(num_live_elements/16))) in row-major (ihi, jhi) order, THEN the synthetic
 * triangle (ihi in [NA/16, NT/16), jhi in [NA/16, ihi]) in the same order. Each tile is
 * [16][16][2] int32, memory offset within a (t,f) slice = 512*(ihi*(ihi+1)/2 + jhi), i.e.
 * exactly n2k's documented layout. V_ij = E_i * conj(E_j) with i the row: mixed tiles carry
 * conj-of-what-the-despread-wants (path A == conj(V)/s -- measured, n2dualxval).
 *
 * The synthetic array is [num_times][num_local_freq][num_synth] int8 4+4b offset-encoded per
 * frame slot. This command zero-fills (0x88) every slot ONCE at construction; the injector
 * overwrites only its lanes/channels each frame. Samples must be in [-7,7]: the value -8
 * silently corrupts the whole launch (n2k's negate_4bit) -- the injector's quantizer clamps.
 *
 * @conf As cudaCorrelator (buffer_depth, num_times, num_elements, num_local_freq,
 *       sub_integration_ntime, voltage_name, rfi_RFImask_name, n2k_correlation_name), plus:
 * @conf  num_synth            Int. Synthetic stations (multiple of 128). Default 128.
 * @conf  num_live_elements    Int. Live antennas (bounds the mixed-tile gather). Default
 *                             num_elements.
 * @conf  gnss_local_channels  List of Int. LOCAL frame channel indices (0..num_local_freq)
 *                             of the GNSS comb. GLOBAL freq_ids do NOT go here -- the config
 *                             generator derives local from global (the DC-replica lesson).
 * @conf  gnss_tiles_name      String. Base name for the gathered-tiles output buffer.
 * @conf  gnss_synth_name      String. Name of the synthetic-input GPU array. Default
 *                             "gnss_synth".
 * @conf  rfi_all_pass         Bool, default false. True = do not consume an RFI-mask ring;
 *                             use a constant all-ones mask instead. For the GNSS dev
 *                             deployment, where the RFI chain is dropped: production runs
 *                             with rfi_first_stage_excision_enabled: false, i.e. its
 *                             correlator also sees an all-good mask, so this is
 *                             behavior-identical -- not an approximation.
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

    /// (freq, int32-offset-within-slice) of each gathered tile, in output order.
    std::vector<int2> _tile_sel;

    const bool _rfi_all_pass;

    /// Signaling ring buffer for the input (voltage) data.
    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;
    /// Absent when rfi_all_pass (a constant all-ones device mask is used instead).
    std::optional<NDArrayRingBuffer<kotekan::uint1x8_t, 3>> rfi_RFImask;
    NDArrayBuffer<std::int32_t, 6> n2k_correlation; // standard N^2 out, shape as cudaCorrelator
    NDArrayBuffer<std::int32_t, 6> gnss_tiles;      // [Tc][gnss chan][tile][16][16][2]

    /// Cuda kernel wrapper object (the two-input clone; n2k itself is untouched).
    n2k_dual::DualCorrelator dual_correlator;

    /// gnss_freq_map: compute MIXED|BB over ONLY the comb channels. vis_out is then compacted
    /// to those channels and the AA (antenna) block is NOT computed -- so the standard N^2
    /// pass-through is unavailable in this mode. 2.90x -> 1.21x stock N^2 (n2timing).
    /// Declared AFTER dual_correlator: it is initialized after it (-Werror=reorder).
    const bool _freq_map_mode;
};

namespace gnss_cuda {
/// Copy the selected (freq, tile) list out of the extended visibility triangle.
/// sel[k].x = freq index, sel[k].y = int32 offset of the tile within a (t,f) slice.
/// out layout: [nt_outer][n_sel][512] int32.
cudaError_t launch_gather_gnss_tiles(const std::int32_t* vis, const int2* sel, int n_sel,
                                     int nt_outer, int vmat_fstride, int vmat_tstride,
                                     std::int32_t* out, cudaStream_t stream);
} // namespace gnss_cuda

#endif // KOTEKAN_CUDA_CORRELATOR_DUAL_H
