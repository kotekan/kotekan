#ifndef KOTEKAN_CUDA_SYNC_STREAM_HPP
#define KOTEKAN_CUDA_SYNC_STREAM_HPP

#include "cudaCommand.hpp"

#include <vector>

/**
 * @class cudaSyncStream
 * @brief A synchronization point between one or more cuda streams.
 *
 * This command object adds a set of cudaStreamWaitEvent events to the queue given by @c cuda_stream
 * Which wait on the last event currently in each of the streams in @c source_cuda_streams
 *
 * @conf source_cuda_streams Array of cuda streams to synchronize on
 *
 * @author Andre Renard
 */
class cudaSyncStream : public cudaCommand {
public:
    cudaSyncStream(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaSyncStream();
    int wait_on_precondition(int gpu_frame_id) override;
    cudaEvent_t execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame(int frame_id) override;

    std::string get_performance_metric_string() override;

protected:
    /// A constructor meant to be called by subclassers who need to
    /// override the constructor behavior.  Specifically, this does not read the config value @c
    /// source_cuda_stream to set the streams to be waited on, nor does it read @c cuda_stream or
    /// set the default stream; the subclasser must do these things.
    cudaSyncStream(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                   bool called_by_subclasser);

    /// Sets the list of streams on which this object should synchronize.
    void set_source_cuda_streams(const std::vector<int32_t>& streams);

    /// List of stream_id's to sync against
    std::vector<int32_t> _source_cuda_streams;
};


#endif // KOTEKAN_CUDA_SYNC_STREAM_HPP
