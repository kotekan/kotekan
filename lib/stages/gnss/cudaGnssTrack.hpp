/**
 * @file
 * @brief GNSS tracking of one channelized input: despread it against the replicas on the GPU.
 *  - cudaGnssTrack : public cudaCommand
 */

#ifndef CUDA_GNSS_TRACK_HPP
#define CUDA_GNSS_TRACK_HPP

#include "cudaCommand.hpp"
#include "cudaGnssChordTrack.hpp" // for cudaGnssChordTrackState
#include "gnssTrackChain.hpp"     // for TrackChain

#include <cuda_runtime_api.h> // for cudaStream_t
#include <memory>             // for shared_ptr, unique_ptr
#include <string>             // for string
#include <vector>             // for vector

/// cudaGnssTrack's state: cudaGnssInject's, registered under its own name, plus the chain of
/// its @c data block.
class cudaGnssTrackState : public cudaGnssChordTrackState {
public:
    cudaGnssTrackState(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);

    /// The data component, despread at the pilot chain's seeds; null without a @c data block.
    std::unique_ptr<gnss::TrackChain> data;
};

/**
 * @class cudaGnssTrack
 * @brief Despread one antenna's 4+4-bit channelized voltages against a chain's replicas,
 *        writing a gnssGpuChain.hpp frame for GnssGpuRecordAssemble.
 *
 * Where @ref cudaGnssInject hands the replicas to the N^2 correlator, this command correlates
 * them itself (GnssCudaDespread::enqueue_batch_nm), for a receiver with no correlator. The
 * chain is the same gnss::TrackChain: seeds, fleet trims, slot membership and their REST
 * endpoints, and the control block of each frame. A cudaInputData and a cudaSyncInput before it
 * put the frame on the device with its GnssChanMetadata, and a cudaSyncOutput and a
 * cudaOutputData after it copy its output to the host. Several commands, one per chain, can
 * read one input.
 *
 * The data is conjugated before the correlation, as the replicas assume
 * (gnss::Channelization).
 *
 * A @c data block adds a data component despread at this chain's own seeds, trims and PRN map,
 * for a signal whose data and pilot share code timing and carrier (GPS_L5_I_NH under
 * GPS_L5_Q_NH). It is a gnss::TrackChain configured at `<command>/data`, which sets its own
 * `signal` and `gpu_mem_output` and inherits every other key from the command. Its code length
 * must divide this chain's, so a seed places both, and it serves no endpoints. Its frame has
 * the same layout as this chain's and goes to its own GnssGpuRecordAssemble.
 *
 * @par GPU Memory
 * @gpu_mem gpu_mem_input   The channelized frame, [samples_per_data_set][num_local_freq].
 *     @gpu_mem_type        staging
 *     @gpu_mem_format      Array of @c uint8_t, 4+4 bits (gnss44.hpp)
 *     @gpu_mem_metadata    GnssChanMetadata
 * @gpu_mem gpu_mem_output  The gnssGpuChain.hpp frame, one element.
 *     @gpu_mem_type        staging
 *     @gpu_mem_format      gnss_gpu::frame_bytes(n_prn, n_channels, 4, 1) bytes
 * @gpu_mem data/gpu_mem_output  The data component's frame, as gpu_mem_output's.
 *     @gpu_mem_type        staging
 *     @gpu_mem_format      gnss_gpu::frame_bytes(n_prn, n_channels, 4, 1) bytes
 *
 * @conf The gnss::TrackChain keys, with `n_elements: 1`, and:
 * @conf  gpu_mem_input        String. The input frame's GPU memory name.
 * @conf  gpu_mem_output       String. The output frame's GPU memory name.
 * @conf  num_local_freq       Int. Channels in the input frame.
 * @conf  gnss_local_channels  List of Int. Index in the input frame of each covering channel,
 *                             in channel_ids order.
 * @conf  data                 Block (optional). The data component: `signal` and
 *                             `gpu_mem_output`.
 */
class cudaGnssTrack : public cudaCommand {
public:
    cudaGnssTrack(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                  int instance_num, std::shared_ptr<cudaCommandState> state);
    ~cudaGnssTrack() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    gnss::TrackChain& chain();

    /// One chain's frame: its control block staging, device output, jobs, seeds and replica
    /// scratch, and the jobs and active slots enqueue_frame counts into it.
    struct ChainFrame {
        gnss::TrackChain* C;
        char* stage;
        char* d_out;
        void* d_jobs;
        const std::vector<gnss::TrackChain::Seed>* seeds;
        void* d_wave;
        int n_jobs = 0;
        int n_active = 0;
    };

    /// Build the frame of pilot @p P, and of its data component @p D when non-null, from their
    /// seeds and @p trim_now: each control block in its stage, copied to its d_out after the
    /// despread of each record into it. A record whose two chains pair slot for slot
    /// (GnssCudaDespread::enqueue_batch_nm_pair) synthesises both replicas in one launch.
    void enqueue_frame(ChainFrame& P, ChainFrame* D, const std::vector<double>& trim_now,
                       long long hop0_frame, const unsigned char* d_frame,
                       const float* d_chan_scale, const int* d_chan_ids, cudaStream_t stream);

    const std::string _gpu_mem_input;
    const std::string _gpu_mem_output;
    const std::int32_t _num_local_freq;
    const std::vector<std::int32_t> _local_channels;
    /// The data block's output; empty without one.
    std::string _gpu_mem_output_data;

    /// Pinned staging for the frame's control block: the header, window starts and PrnCtl rows.
    char* _stage = nullptr;
    char* _stage_data = nullptr; ///< the data component's
    bool _uploaded_static = false;
    uint64_t _frames = 0;
};

#endif // CUDA_GNSS_TRACK_HPP
