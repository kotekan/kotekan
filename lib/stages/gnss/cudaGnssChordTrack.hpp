/**
 * @file
 * @brief CHORD N-antenna GNSS despread command: waveform generation + N x M correlation.
 *  - cudaGnssChordTrackState : public cudaCommandState
 *  - cudaGnssChordTrack : public cudaCommand
 */

#ifndef CUDA_GNSS_CHORD_TRACK_HPP
#define CUDA_GNSS_CHORD_TRACK_HPP

#include "Config.hpp"
#include "GnssCudaDespread.hpp"
#include "gnssChannelizedReplica.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "restServer.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

/**
 * @class cudaGnssChordTrackState
 * @brief Broker seeds + per-PRN control, shared across this command's instances.
 */
class cudaGnssChordTrackState : public cudaCommandState {
public:
    cudaGnssChordTrackState(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaGnssChordTrackState() = default;

    /// Broker seed contract, identical to cudaGnssTrack's /set_seeds: a JSON array of
    /// {prn, doppler_hz, code_phase_chips, code_phase_rate, doppler_rate_hz_s,
    ///  carrier_trim_hz, ref_hop}. Deliberately the SAME contract so one broker can drive
    /// either chain and the airspy tooling works unchanged against a CHORD node.
    void set_seeds_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    // Geometry (immutable after construction).
    std::vector<int> prns;
    int n_prn = 0, n_chan = 0, n_elem = 0;
    int elem_stride = 0, frame_chan_stride = 0;
    int hops_per_record = 0, n_hops_frame = 0, fft_len = 16384;
    double sample_rate = 3.2e9, f_offset_hz = 0.0, dll_spacing = 0.5;
    double doppler_margin_hz = 5000.0;

    /// One PRN's live seed. Model-primary: the broker owns these and refreshes them every
    /// cycle, so there is no frozen-seed state to age or unfreeze here.
    struct Seed {
        bool have = false;
        double doppler_hz = 0.0;
        double cp_chips = 0.0;    ///< prompt code phase at ref_hop
        double cp_rate = 0.0;     ///< chips per hop (the broker's measured l-a residual)
        double dop_rate = 0.0;    ///< Hz/s
        double ctrim_hz = 0.0;    ///< broker carrier trim
        long long ref_hop = 0;    ///< absolute hop the cp is referenced to
    };
    std::mutex seed_mtx;
    std::vector<Seed> seeds;

    std::unique_ptr<gnss::ChannelizedReplicaBank> replica;
    std::unique_ptr<GnssCudaDespread> despread;
    std::vector<int> covering; ///< local channel indices this signal occupies (0..n_chan-1)
};

/**
 * @class cudaGnssChordTrack
 * @brief Despread N antennas against M references, per record window.
 *
 * ⚠️ RELATIONSHIP TO cudaGnssTrack -- READ BEFORE MERGING GNSS-SIDE WORK.
 *
 * This is a SEPARATE command, not a mode of @ref cudaGnssTrack, by explicit decision. It
 * duplicates part of that stage's role, so the two can drift, and a fix to one may need
 * mirroring in the other. What is shared and what is not:
 *
 *   SHARED (single implementation -- changes propagate for free):
 *     * replica synthesis          cudaGnssReplicaDevice.cuh
 *     * job construction           GnssCudaDespread::build_jobs
 *     * the record schema          gnssRecord.hpp
 *     * the output frame layout    gnssGpuChain.hpp
 *     * the broker seed contract   /set_seeds, same JSON as cudaGnssTrack
 *
 *   DUPLICATED (must be mirrored by hand if changed there):
 *     * per-record seed -> Spec construction and code-phase extrapolation
 *     * the PrnCtl / FrameHdr control block the assembler reads
 *
 *   DELIBERATELY ABSENT here, and why:
 *     * The FROZEN-SEED machinery (hold-on-lock, anchor ageing, the snap-to-model fence and
 *       the code-currency f_ref re-pin). The airspy node retired all of it in favour of
 *       model-primary seeding (`--dop-continuous`), where the seed follows the BRDC model
 *       every cycle and the re-pin is free. Starting there rather than reproducing the state
 *       machine it replaced is the point; if CHORD ever needs the fence, port it knowingly.
 *     * The VOLTAGE PEEL. Deferred until acq/track and beam mapping are proven -- a single-PRN
 *       replica is likely sub-quantization at 4+4b. rows_spec is therefore always 4.
 *     * The channel-major device RING. The CHORD correlator reads the tap's frame in its native
 *       [hop][chan][elem] order, so there is nothing to transpose into.
 *
 * RECORD LENGTH. hops_per_record defaults to 2048 (10.49 ms at CHORD's 5.12 us hop), which
 * divides the tap's 8192-hop frame exactly 4 ways and stays under the 20 ms NH20 secondary
 * period -- so a record straddles at most ONE overlay transition, which is exactly the case
 * P_HEAD/m_head exists to handle. A code period is 195.3125 hops, NOT an integer, which is
 * fine: the despread anchors to an absolute sample index with exact cp0/cps, so nothing
 * requires whole hops per period.
 *
 * @conf prns, n_channels, n_elements, elem_stride, frame_chan_stride
 * @conf hops_per_record   default 2048
 * @conf signal            gnssSignal.hpp name, e.g. GPS_L5_Q
 * @conf sample_rate       pre-channelization, 3.2e9 on CHORD
 * @conf seed_endpoint     default "/chord_track/set_seeds"
 */
class cudaGnssChordTrack : public cudaCommand {
public:
    cudaGnssChordTrack(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                       int instance_num, std::shared_ptr<cudaCommandState> state);
    ~cudaGnssChordTrack() override = default;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    cudaGnssChordTrackState* st();

    std::string _gpu_mem_input, _gpu_mem_output;
    std::string _mem_jobs, _mem_wave, _mem_scale, _mem_chanids;
    size_t _in_frame_len = 0, _out_frame_len = 0;
    std::vector<char> _ctl_stage; ///< host staging for the FrameHdr + PrnCtl control block
    bool _uploaded_static = false;
};

#endif // CUDA_GNSS_CHORD_TRACK_HPP
