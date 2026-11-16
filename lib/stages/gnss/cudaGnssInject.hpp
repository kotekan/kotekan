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
#include "gnssGpuRows.hpp"        // for GpuRow

#include <memory>  // for shared_ptr
#include <string>  // for string
#include <utility> // for pair
#include <vector>  // for vector

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
 * Seeds, fleet trims and slot membership live in the command's gnss::TrackChain.
 * Several injectors (one per chain) can run in one kotekan instance, each with its own state
 * but one REST server between them, so each needs its own seed_endpoint, set_trim_endpoint,
 * get_prns_endpoint and set_prns_endpoint.
 *
 * One injector can also host every chain of a GPU half, as the rows of a @c chains block
 * (cudaGnssChordTrackState), each with its own gnss::TrackChain, endpoints and gnss_row. It
 * claims the voltage ring once, packs the rows in gnss_rows order, and uploads every row's
 * control block in one copy per frame. Each row has its own GPU arrays, named for the row, and
 * its own line in /gpu_profile.
 *
 * Replicas are synthesized per record (hops_per_record; 4 records per 8192-hop frame at CHORD),
 * from gnss::propagate_seed at each record's window start with the fleet code trim from
 * gnss::TrackChain::snapshot_trims, through GnssCudaDespread::build_jobs like every
 * other Spec-taking despread. Under the state's psi_synth the synthesis and the pack are one
 * launch (GnssCudaDespread::enqueue_psi_pack); the control block, energies and lanes are the
 * same. The Psi tables are built on the despread's builder thread, never here: a seeded slot
 * whose tables are not on the device yet is left out of the record (dark, as without a seed;
 * GnssCudaDespread::psi_ready) and joins when they land, so after the first seeds a chain's
 * slots light up one per record over a few frames.
 *
 * Lanes: lane = synth_lane_base + lanes_per_slot*prn_slot + k, k the lane's trial in slot
 * order (E/P/L/P_HEAD with four lanes per slot, P/P_HEAD with two), fixed per slot as
 * satellites rise and set; slots without a seed are written as 0x88, so their tiles read
 * exactly 0. Requires lanes_per_slot*n_prn <= num_synth. The pack quantizes each lane and channel
 * with its own scale (see launch_pack44), which the consumer, GnssN2RecordAssemble, undoes using
 * the replica energies in the control block.
 *
 * The voltage ring is consumed for its time reference only (the frame's absolute hop); no
 * voltage bytes are read. That makes the injector a flow-control participant of the ring, which
 * is intended: it must produce a synth window for every window the correlator consumes.
 *
 * @conf As the ring consumers (buffer_depth, num_times, num_elements, num_local_freq,
 *       voltage_name), plus the gnss::TrackChain keys (set `n_elements: 1`; it is
 *       unused here), and:
 * @conf  num_synth            Int. Lanes this chain writes (lanes_per_slot per PRN slot).
 * @conf  lanes_per_slot       Int, default 4. Synth lanes per PRN slot: 4 packs E, P, L and
 *                             P_HEAD; 2 packs P and P_HEAD and carries the kernel's per-channel
 *                             rotation in PrnCtl::rot, from which GnssN2RecordAssemble forms the
 *                             E and L rows. 2 needs epl_from_prompt and at most
 *                             gnss_gpu::MAX_ROT_CHAN covering channels.
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
 * @conf  gnss_row             String, instead of num_synth, synth_lane_base,
 *                             gnss_local_channels and gnss_ctl_name: this chain's row name in
 *                             the half's gnss_rows (gnssGpuRows.hpp), which gives its lanes and
 *                             channels. The control block goes at the row's ctl_offset in the
 *                             half's shared array @c gnss_rows_ctl_name, with the row tag in
 *                             FrameHdr::_pad1.
 * @conf  chains               Map, optional. The rows this command hosts, each with its chain's
 *                             keys (cudaGnssChordTrackState); gnss_row is required in each.
 */
class cudaGnssInject : public cudaCommand {
public:
    cudaGnssInject(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                   int instance_num, std::shared_ptr<cudaCommandState> state);
    ~cudaGnssInject() override;

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;
    /// With a chains block, one /gpu_profile row per chain.
    std::vector<std::pair<std::string, std::shared_ptr<StatTracker>>> profile_rows() override;

private:
    cudaGnssChordTrackState* st();

    /// One chain: its place in the synth and control arrays, and this instance's state for it.
    struct Row {
        gnss::TrackChain* chain;
        /// Its gnss_rows entry, or a row with no name.
        gnss_gpu::GpuRow gpu;
        /// Prefix of its GPU array names: the command's unique_name, or its row's name.
        std::string name;
        /// Prefix of its log lines: empty, or its row's key.
        std::string label;
        std::int32_t num_synth;      ///< lanes this chain writes (lanes_per_slot per slot)
        std::int32_t lanes_per_slot; ///< gnss_gpu::LANES_FULL or LANES_PROMPT
        std::int32_t lane_base;      ///< first lane of this chain in the synth array
        std::vector<std::int32_t> local_channels;
        std::vector<int> chan_map; ///< covering channel -> index on the synth channel axis
        /// The control block (see _stage), and the byte offset of its energy array.
        size_t ctl_offset, ctl_bytes, ctl_off_energy;
        int* slot2spec; ///< in _stage, [n_rec][n_prn]
        bool uploaded_static = false;
        uint64_t frames = 0;
        /// This frame's jobs and the device energy rows they wrote, [n_jobs * ROWS_PLAIN][n_chan].
        int n_jobs = 0;
        double* d_energy = nullptr;
        /// Hosting only: this frame's events around the row's work, and their trackers.
        cudaEvent_t start_event = nullptr, end_event = nullptr;
        std::shared_ptr<StatTracker> execute_time, utilization;
    };
    /// Enqueue @p row's frame: its slot swaps, synthesis and pack, and its control block in
    /// _stage.
    void enqueue_row(Row& row, const cudaPipelineState& pipestate, long long hop0_frame,
                     unsigned char* d_synth, cudaStream_t stream);

    std::vector<Row> _rows;
    const bool _hosted;
    const std::int32_t _buffer_depth;
    const std::int32_t _num_times;
    const std::int32_t _num_elements; // voltage ring station axis (for the ring declaration)
    const std::int32_t _num_local_freq;
    const std::string _voltage_name;
    const std::string _gnss_synth_name;
    /// Synth array layout (see the @conf list).
    const std::int32_t _synth_lane_pitch;
    std::vector<std::int32_t> _gnss_synth_channels;
    std::int32_t _synth_n_chan; ///< the synth array's channel axis

    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;

    /// The control blocks this command publishes for GnssN2RecordAssemble, one per row: the
    /// gnssGpuChain.hpp layout without corr, [FrameHdr][window_start x MAX_REC]
    /// [PrnCtl x MAX_REC x n_prn][energy x max_jobs x n_chan]. The consumer fills corr from the
    /// N^2 tiles and needs the replica energies to undo the pack's per-lane scale.
    std::string _mem_ctl;
    size_t _ctl_array_bytes = 0; ///< the _mem_ctl array's size
    /// Pinned staging, this instance's: the rows' control blocks as they sit in the _mem_ctl
    /// array, [_ctl_begin, _ctl_begin + _ctl_span), then each row's slot2spec.
    char* _stage = nullptr;
    size_t _ctl_begin = 0, _ctl_span = 0;
};

#endif // CUDA_GNSS_INJECT_HPP
