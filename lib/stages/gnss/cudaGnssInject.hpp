/**
 * @file
 * @brief GNSS replica injection: synthesize and quantize replica waveforms into the N^2
 *        correlator's synthetic input.
 *  - cudaGnssInject : public cudaCommand
 */

#ifndef CUDA_GNSS_INJECT_HPP
#define CUDA_GNSS_INJECT_HPP

#include "DataType.hpp"          // for int4x2_swapped_withoffset_t
#include "NDArrayRingBuffer.hpp" // for NDArrayRingBuffer
#include "cudaCommand.hpp"
#include "cudaGnssChordTrack.hpp" // for cudaGnssChordTrackState

#include <string>
#include <vector>

/**
 * @class cudaGnssInject
 * @brief Fill cudaCorrelatorDual's synthetic-station array with quantized GNSS replicas.
 *
 * Runs in the same cudaProcess as @ref cudaCorrelatorDual, before it on the same stream, which
 * is all the synchronization there is: the pack completes before the N^2 kernel reads the synth
 * array. The array needs no ring semantics: it is a per-frame-slot named GPU array, filled with
 * 0x88 (0+0j) once by cudaCorrelatorDual's constructor, and each injector overwrites only its
 * own lanes and channels.
 *
 * Seeds, fleet trims and slot membership live in the command's @ref cudaGnssChordTrackState.
 * Several injectors (one per chain) can run in one kotekan instance, each with its own state
 * but one REST server between them, so each needs its own seed_endpoint, trim_endpoint,
 * set_trim_endpoint, get_prns_endpoint and set_prns_endpoint.
 *
 * Replicas are synthesized per record (hops_per_record; 4 records per 8192-hop frame at CHORD),
 * from gnss::propagate_seed at each record's window start with the fleet code trim from
 * cudaGnssChordTrackState::snapshot_trims, through GnssCudaDespread::build_jobs like every
 * other Spec-taking despread.
 *
 * Lanes: lane = synth_lane_base + 4*prn_slot + row (E/P/L/P_HEAD), fixed per slot as
 * satellites rise and set; slots without a seed are written as 0x88, so their tiles read
 * exactly 0. Requires 4*n_prn <= num_synth. The pack quantizes each lane and channel with its
 * own scale (see launch_pack44), which the consumer, GnssN2RecordAssemble, undoes using the
 * replica energies in the control block.
 *
 * The voltage ring is consumed for its time reference only (the frame's absolute hop); no
 * voltage bytes are read. That makes the injector a flow-control participant of the ring, which
 * is intended: it must produce a synth window for every window the correlator consumes.
 *
 * @conf As the ring consumers (buffer_depth, num_times, num_elements, num_local_freq,
 *       voltage_name), plus the @ref cudaGnssChordTrackState keys (set `n_elements: 1`; it is
 *       unused here), and:
 * @conf  num_synth            Int. Lanes this chain writes (4 per PRN slot).
 * @conf  synth_lane_base      Int. First lane of this chain in the synth array, a multiple of
 *                             16 (tile rows).
 * @conf  synth_lane_pitch     Int. Width of the synth array's lane axis (the correlator's
 *                             num_synth), a multiple of 16.
 * @conf  gnss_synth_name      String. Named GPU array shared with cudaCorrelatorDual.
 * @conf  gnss_local_channels  List of Int. Local frame channel index of each covering
 *                             channel, in local (bank) order. Must have n_channels entries.
 *                             Global freq_ids do not go here.
 * @conf  gnss_synth_channels  List of Int. The synth array's channel axis as local frame
 *                             channels (the correlator's comb, for gnss_synth_compact); must
 *                             contain every gnss_local_channels entry.
 * @conf  gnss_ctl_name        String. Named GPU array for the control block.
 */
class cudaGnssInject : public cudaCommand {
public:
    cudaGnssInject(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                   int instance_num, std::shared_ptr<cudaCommandState> state);
    ~cudaGnssInject() override = default;

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    cudaGnssChordTrackState* st();

    const std::int32_t _buffer_depth;
    const std::int32_t _num_times;
    const std::int32_t _num_elements; // voltage ring station axis (for the ring declaration)
    const std::int32_t _num_local_freq;
    const std::int32_t _num_synth; ///< lanes this chain writes (4 per PRN slot)
    const std::string _voltage_name;
    const std::string _gnss_synth_name;
    std::vector<std::int32_t> _gnss_local_channels;
    /// Synth array layout (see the @conf list).
    const std::int32_t _synth_lane_base;
    const std::int32_t _synth_lane_pitch;
    std::vector<std::int32_t> _gnss_synth_channels;
    std::int32_t _synth_n_chan;   ///< the synth array's channel axis
    std::vector<int> _h_chan_map; ///< covering channel -> index on that axis

    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;

    std::vector<int> _h_slot2spec; ///< staging, [n_rec][n_prn]
    bool _uploaded_static = false;
    uint64_t _frames = 0;

    /// The control block this command publishes for GnssN2RecordAssemble: the gnssGpuChain.hpp
    /// layout without corr, [FrameHdr][window_start x MAX_REC][PrnCtl x MAX_REC x n_prn]
    /// [energy x max_jobs x n_chan]. The consumer fills corr from the N^2 tiles and needs the
    /// replica energies to undo the pack's per-lane scale.
    std::string _mem_ctl;
    std::vector<char> _ctl_stage;
    size_t _ctl_bytes = 0;
    /// Byte offset of the energy array within the control block.
    size_t _ctl_off_energy = 0;
};

#endif // CUDA_GNSS_INJECT_HPP
