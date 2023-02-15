#ifndef KOTEKAN_CUDA_SYNC_STREAM_HPP
#define KOTEKAN_CUDA_SYNC_STREAM_HPP

#include "cudaCommand.hpp"

#include <vector>

/**
 * @class cudaSyncStream
 * @brief cudaCommand for adding wait events to a stream based on events form other streams
 *
 * @author Andre Renard
 */
class cudaSyncStream : public cudaCommand {
public:
    cudaSyncStream(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaSyncStream();
    int wait_on_precondition(int gpu_frame_id) override;
    cudaEvent_t execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame(int frame_id) override;

protected:
    /// List of stream_id's to sync against
    std::vector<int32_t> _source_cuda_streams;
};


#endif // KOTEKAN_CUDA_SYNC_STREAM_HPP
