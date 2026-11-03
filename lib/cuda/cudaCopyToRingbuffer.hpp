#ifndef KOTEKAN_CUDA_COPYTORINGBUFFER_HPP
#define KOTEKAN_CUDA_COPYTORINGBUFFER_HPP

#include "Config.hpp"              // for Config
#include "NDArray.hpp"             // for GenericNDArray
#include "buffer.hpp"              // for Buffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "chordMetadata.hpp"       // for chordMetadata
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "driver_types.h"          // for cudaEvent_t
#include "ringbuffer.hpp"          // for RingBuffer

#include <stddef.h> // for size_t
#include <stdint.h> // for int64_t
#include <string>   // for string, basic_string
#include <vector>   // for vector

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
    /// @p name is the command's name in logs and metrics (a subclass passes its own).
    cudaCopyToRingbuffer(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                         int instance_num, const std::string& name = "cudaCopyToRingbuffer");
    ~cudaCopyToRingbuffer();

    int wait_on_precondition() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

protected:
    /// Enqueue the frame's copy into the ring and keep its metadata; returns that metadata
    /// (nullptr without any). execute() wraps this in the start and end events; a subclass that
    /// works on the copied frame before the ring's write is finished calls it from its own
    /// execute() and enqueues its work after it on the same stream.
    std::shared_ptr<chordMetadata> copy_frame(const std::vector<cudaEvent_t>& pre_events);

    /// Attach (or check) the ring's descriptor: the input's layout, with the ring's capacity in
    /// the slowest dimension.
    void set_ring_frame_desc(const kotekan::GenericNDArray& in_desc);

    size_t _input_size;
    size_t _ring_buffer_size;

    size_t output_cursor;
    int64_t initial_fpga_seq_num; // fpga_seq_num of first frame, used for consistency checking

    // One of these will be set:
    /// GPU side memory name for the frame-based input
    std::string _gpu_mem_input;
    /// Host side buffer for the frame-based input
    Buffer* in_buffer;

    /// GPU side memory name for the ring-buffer output
    std::string _gpu_mem_output;

    // RingBuffer object for signalling other GPU pipelines
    RingBuffer* signal_buffer;

    /// Optional, name of the field in the pipeline state object that holds the number of input
    /// columns to copy.
    // std::string _input_columns_field;
};

#endif // KOTEKAN_CUDA_COPYTORINGBUFFER_HPP
