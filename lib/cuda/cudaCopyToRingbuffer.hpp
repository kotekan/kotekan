#ifndef KOTEKAN_CUDA_COPYTORINGBUFFER_HPP
#define KOTEKAN_CUDA_COPYTORINGBUFFER_HPP

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "ringbuffer.hpp"

/**
 * @class cudaCopyToRingbuffer
 * @brief cudaCommand for copying GPU frames into a ringbuffer.
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
 * @conf input_size - Int
 * @conf output_size - Int
 *
 * The input array is assumed to be of size @c input_size bytes.
 *
 * The output array is assumed to have total available size @c output_size bytes.
 */
class cudaCopyToRingbuffer : public cudaCommand {
public:
    cudaCopyToRingbuffer(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                         int instance_num,
                         const std::shared_ptr<cudaCommandState>& state);

    int wait_on_precondition() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

protected:
private:
    size_t _input_size;
    size_t _ring_buffer_size;

    size_t output_cursor;

    /// GPU side memory name for the time-stream input
    std::string _gpu_mem_input;
    /// GPU side memory name for the time-stream output
    std::string _gpu_mem_output;

    // Host side buffer
    RingBuffer* signal_buffer;

    /// Optional, name of the field in the pipeline state object that holds the number of input
    /// columns to copy.
    // std::string _input_columns_field;
};

#endif // KOTEKAN_CUDA_COPYTORINGBUFFER_HPP
