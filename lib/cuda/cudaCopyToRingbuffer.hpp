#ifndef KOTEKAN_CUDA_COPYTORINGBUFFER_HPP
#define KOTEKAN_CUDA_COPYTORINGBUFFER_HPP

#include "Config.hpp"              // for Config
#include "NDArray.hpp"             // for GenericNDArray
#include "buffer.hpp"              // for Buffer
#include "bufferContainer.hpp"     // for bufferContainer
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
 *
 * Each frame's write offset in the ring (bytes since the start of the stream) is set in the
 * pipeline state under the ring's name, for later commands that work on the frame in place.
 *
 * @conf data_lag_s - Double, default 0. With in_buf, hold each frame until the wall clock is
 *       this many seconds past its last sample's time (its metadata's FPGA sequence number plus
 *       samples_per_data_set), so everything downstream sees data at least that old. A
 *       consumer that needs information from elsewhere about the signal it processes (another
 *       receiver's data symbols) then has that long to receive it. in_buf must hold the
 *       frames this adds.
 */
class cudaCopyToRingbuffer : public cudaCommand {
public:
    cudaCopyToRingbuffer(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                         int instance_num);
    ~cudaCopyToRingbuffer();

    int wait_on_precondition() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

protected:
private:
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

    /// data_lag_s, and the frame length it is timed from
    double _data_lag_s;
    int64_t _samples_per_data_set;
    /// Wait until in_buf's frame @p buf_index is _data_lag_s old.
    void wait_for_lag(int buf_index);

    /// GPU side memory name for the ring-buffer output
    std::string _gpu_mem_output;

    // RingBuffer object for signalling other GPU pipelines
    RingBuffer* signal_buffer;

    /// Optional, name of the field in the pipeline state object that holds the number of input
    /// columns to copy.
    // std::string _input_columns_field;
};

#endif // KOTEKAN_CUDA_COPYTORINGBUFFER_HPP
