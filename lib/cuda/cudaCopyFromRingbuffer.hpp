#ifndef KOTEKAN_CUDA_COPYFROMRINGBUFFER_HPP
#define KOTEKAN_CUDA_COPYFROMRINGBUFFER_HPP

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "ringbuffer.hpp"

/**
 * @class cudaCopyFromRingbuffer
 * @brief cudaCommand for copying GPU frames from a RingBuffer into a (vanilla) Buffer (frame-based
 * buffer).
 *
 * @author Dustin Lang
 *
 * @par GPU Memory
 * @gpu_mem  gpu_mem_input  Input matrix of data (of unspecified type)
 *   @gpu_mem_type   staging
 *   @gpu_mem_format array of raw bytes
 * @gpu_mem  gpu_mem_output Output matrix of data (of unspecified type)
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of raw bytes
 *
 * @conf ring_buffer_size - Int - size of the ring buffer
 * @conf output_size - Int - size of the destination frames
 *
 */
class cudaCopyFromRingbuffer : public cudaCommand {
public:
    cudaCopyFromRingbuffer(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                           int instance_num);
    ~cudaCopyFromRingbuffer() override;

    int wait_on_precondition() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

    void finalize_frame() override;

protected:
private:
    size_t _output_size;
    size_t _ring_buffer_size;

    size_t input_cursor;

    /// GPU side memory name for the time-stream input
    std::string _gpu_mem_input;
    /// GPU side memory name for the time-stream output
    std::string _gpu_mem_output;

    /// Host side buffer for the frame-based output, if we're doing that
    Buffer* out_buffer;

    // Host side signalling buffer
    RingBuffer* signal_buffer;
};

#endif // KOTEKAN_CUDA_COPYFROMRINGBUFFER_HPP
