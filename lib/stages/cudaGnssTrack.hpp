#ifndef CUDA_GNSS_TRACK_HPP
#define CUDA_GNSS_TRACK_HPP

#include "Config.hpp"
#include "GnssCudaDespread.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "gnssChannelizedReplica.hpp"
#include "restServer.hpp"

#include <memory>
#include <mutex>
#include <vector>

/**
 * Shared state for @ref cudaGnssTrack (one per stage, shared by the per-frame command
 * instances): the replica bank + GPU despread driver, the broker-facing seed set (REST
 * /set_seeds, same contract as GnssChannelizedTracker), the tracking control state (pinned
 * f_ref per PRN, fence anchors), and the device ring bookkeeping. All of pass-1's decisions
 * live HERE; the per-instance command objects are stateless dispatchers.
 */
class cudaGnssTrackState : public cudaCommandState {
public:
    cudaGnssTrackState(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaGnssTrackState();

    void set_seeds_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    // Geometry / config (immutable after construction).
    std::vector<int> prns;
    int n_prn = 0, n_chan = 0, chan_offset = 0, fft_len = 0, hops_per_record = 0;
    double sample_rate = 5e6, capture_utc0 = 0.0, doppler_margin_hz = 5000.0;
    double dll_spacing = 0.5, fll_reacq_hz = 200.0;
    long long ring_hops = 0; ///< multiple of hops_per_record (windows never straddle the wrap)

    std::unique_ptr<gnss::ChannelizedReplicaBank> replica;
    std::unique_ptr<GnssCudaDespread> despread;

    // Broker seeds (REST-updated; snapshotted per GPU frame under the mutex).
    std::mutex seed_mtx;
    std::vector<double> dop, cp, cp_rate, dop_rate, ctrim;
    std::vector<long long> ref_hop;
    std::vector<uint8_t> active;

    // Tracking control state (pass-1 owns it; command instances execute in frame order).
    std::vector<double> f_ref;         ///< pinned replica carrier per PRN (NaN = unset)
    std::vector<long long> reacq_hop;  ///< ff_hz ramp anchor per PRN

    // Device ring bookkeeping (the ring memory itself is a named GPU allocation).
    bool ring_init = false;
    long long hop0 = 0;      ///< absolute hop of ring position 0 (window tiling anchor)
    long long next_hop = 0;  ///< absolute hop the next ingest is expected to start at
    long long next_rec = 0;  ///< next unprocessed record index k (window = hop0 + k*hpr)
};

/**
 * @class cudaGnssTrack
 * @brief cudaProcess command: GNSS channelized tracking despread (phase F0).
 *
 * Per GPU frame: transpose-ingest the staged channelized frame into an internal channel-major
 * device ring (zero-filling valve-drop gaps so ring position == absolute hop - hop0), run
 * pass-1 control for every complete record window (seed extrapolation with the quadratic
 * code-Doppler feed-forward, code-currency translation into the pinned f_ref, fence), then
 * ONE batched E/P/L despread launch per record reading the window in place from the ring.
 * Emits the gnssGpuChain.hpp frame (control block + raw per-channel correlations) into the
 * @c gpu_mem_output array for cudaOutputData; GnssGpuRecordAssemble builds tracker records
 * downstream. Semantics match GnssChannelizedTracker with carrier_shared (fll_gain must be 0).
 *
 * @conf All GnssChannelizedTracker geometry/config keys (signal, sample_rate, f_offset,
 *       spectrum_length, n_channels, channel_offset, num_taps, pfb_window, prns,
 *       hops_per_record, code_doppler_sign, dll_spacing_chips, fll_reacq_hz, capture_utc0,
 *       doppler_margin_hz), plus:
 * @conf gpu_mem_input   staged channelized frame ([n_hops_f][n_chan] complex float)
 * @conf gpu_mem_output  gnssGpuChain frame (frame_bytes(n_prn, n_chan))
 * @conf ring_records    device ring length in records (default 50)
 * @conf seed_endpoint   REST path for broker seeds (default "/track/set_seeds")
 */
class cudaGnssTrack : public cudaCommand {
public:
    cudaGnssTrack(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                  int instance_num, std::shared_ptr<cudaCommandState> state);
    ~cudaGnssTrack();

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    cudaGnssTrackState* st();

    std::string _gpu_mem_input, _gpu_mem_output;
    size_t _in_frame_len = 0, _out_frame_len = 0;
    int _n_hops_frame = 0;

    // Host staging for the control block (pageable: the async H2D is host-synchronous on
    // return, so per-instance reuse is safe).
    std::vector<char> _ctl_stage;
};

#endif
