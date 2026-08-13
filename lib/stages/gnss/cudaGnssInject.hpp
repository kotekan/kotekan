/**
 * @file
 * @brief GNSS path B: synthesize + quantize replica waveforms into the N^2's synthetic input.
 */

#ifndef CUDA_GNSS_INJECT_HPP
#define CUDA_GNSS_INJECT_HPP

#include "DataType.hpp"          // for int4x2_swapped_withoffset_t
#include "NDArrayRingBuffer.hpp" // for NDArrayRingBuffer
#include "cudaCommand.hpp"
#include "cudaGnssChordTrack.hpp" // for cudaGnssChordTrackState (the shared seed machinery)

#include <string>
#include <vector>

/**
 * @class cudaGnssInject
 * @brief Fill cudaCorrelatorDual's synthetic-station array with quantized GNSS replicas.
 *
 * Runs in the SAME cudaProcess as @ref cudaCorrelatorDual, ORDERED BEFORE IT on the same
 * stream -- that ordering is the whole synchronization story: the pack is complete before the
 * N^2 kernel reads the synth array, and the array needs no ring semantics (it is a per-frame-
 * slot named GPU array, zero-filled 0x88 once by cudaCorrelatorDual's constructor).
 *
 * SEEDS: this command's state is a @ref cudaGnssChordTrackState -- the same class the tracker
 * uses, so the broker seed contract, TTL expiry (snapshot_seeds) and the seed->Spec
 * propagation (gnss::propagate_seed) are single implementations that cannot fork. The config
 * MUST override `seed_endpoint`/`trim_endpoint` (e.g. "/gnss_inject/set_seeds") so the REST
 * paths do not collide with the tracker's state on the same node. The broker POSTs the same
 * JSON to both endpoints.
 *
 * SYNTHESIS is per record (hops_per_record, 4 records per 8192-hop frame), with
 * propagate_seed at each record's window start -- REPLICAS ARE BIT-IDENTICAL to the ones the
 * tracker builds for the same seeds, which is what the live path-A/path-B comparison needs.
 * No DLL trim is applied (the injector has no trim loop); run the A/B with the tracker's
 * `code_trim: false`.
 *
 * LANES: lane = 4*prn_slot + trial (E/P/L/P_HEAD), STABLE across satellites rising and
 * setting; inactive slots carry 0x88 (their tiles read exactly 0). Requires
 * 4*n_prn <= num_synth. The consumer recovers amplitude as conj(V_mixed)/M^2_diag -- the
 * quantizer scale cancels (see launch_pack44).
 *
 * The voltage ring is consumed for its TIME REFERENCE only (the frame's absolute hop); no
 * voltage bytes are read. This makes the injector a flow-control participant of the ring,
 * which is intended: it must produce a synth window for every window the correlator consumes.
 *
 * @conf As the ring consumers (buffer_depth, num_times, num_elements, num_local_freq,
 *       voltage_name) plus the cudaGnssChordTrackState set (prns, n_channels, channel_ids,
 *       signal, hops_per_record, samples_per_data_set, sample_rate, f_offset_hz,
 *       seed_endpoint, trim_endpoint, seed_ttl_s, ...; set `n_elements: 1` -- unused here),
 *       and:
 * @conf  num_synth            Int. Width of the synth array's station axis. Default 128.
 * @conf  gnss_synth_name      String. Named GPU array shared with cudaCorrelatorDual.
 *                             Default "gnss_synth".
 * @conf  gnss_local_channels  List of Int. LOCAL frame channel index of each covering
 *                             channel, in local (bank) order -- the pack's chan_map. Must
 *                             have n_channels entries. GLOBAL freq_ids do NOT go here.
 */
/**
 * @class cudaGnssInjectState
 * @brief The injector's own instance of the tracker's state machinery.
 *
 * A thin subclass rather than cudaGnssChordTrackState itself only because the factory's
 * registrar variable is named from the state CLASS, so two commands cannot register the same
 * class. Semantics are identical; the config must still give this instance its own
 * `seed_endpoint`/`trim_endpoint`.
 */
class cudaGnssInjectState : public cudaGnssChordTrackState {
public:
    using cudaGnssChordTrackState::cudaGnssChordTrackState;
};

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
    const std::int32_t _num_synth;
    const std::string _voltage_name;
    const std::string _gnss_synth_name;
    std::vector<std::int32_t> _gnss_local_channels;

    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage;

    std::vector<int> _h_slot2spec; ///< staging, [n_rec][n_prn]
    bool _uploaded_static = false;
    uint64_t _frames = 0;

    /// RE-PIN PHASE STEP (task #52), per PRN slot, carried ACROSS frames and records.
    ///
    /// propagate_seed runs per record, so the replica's Doppler -- and therefore its
    /// ABSOLUTELY-anchored carrier phase, 2*pi*fcar*t_abs -- moves every record. The step is
    /// (dop - dop_prev)*t_abs, which at 3.37 days of uptime is 109-1127 CYCLES per record on
    /// sky. The assembler has always known how to fold that out (PrnCtl::reanchored == 2) and
    /// this producer never told it, so the step went into every correlation and read downstream
    /// as a per-record common phase that is "white in time". It is not white; it is this.
    ///
    /// Kept in the DOPPLER domain on purpose: the difference must be taken before f_offset
    /// (1.176 GHz) is added, or cancellation costs 0.4 rad. See PrnCtl::dcyc.
    std::vector<double> _dop_prev;  ///< previous record's propagated Doppler, Hz
    std::vector<double> _t_prev;    ///< and the absolute time it was pinned at, s
    std::vector<uint8_t> _dop_prev_ok; ///< 0 => no history: emit reanchored = 1 (break the arc)

    /// M5: the epl-format CONTROL BLOCK this command publishes for the path-B consumer --
    /// [FrameHdr][window_start x MAX_REC][PrnCtl x MAX_REC x n_prn][energy x jobs x n_chan],
    /// i.e. the shipped gnssGpuChain.hpp layout MINUS the corr array, which the consumer fills
    /// from the N^2 tiles. Everything here is already known to the injector (it builds the
    /// specs and gets `energy` back from launch_waveform), and the energy rows are what let the
    /// consumer undo the pack's per-lane scale -- see docs/gnss_gpu_search.md 11.11.
    std::string _mem_ctl;
    std::vector<char> _ctl_stage;
    size_t _ctl_bytes = 0;
    /// Byte offset of the energy array within the control block.
    size_t _ctl_off_energy = 0;
};

#endif // CUDA_GNSS_INJECT_HPP
