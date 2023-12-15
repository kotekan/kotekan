#include "cudaCopyFromRingbuffer.hpp"

#include "cudaUtils.hpp"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCopyFromRingbuffer);

cudaCopyFromRingbuffer::cudaCopyFromRingbuffer(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               cudaDeviceInterface& device, int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaCopyFromRingbuffer", ""),
    input_cursor(0) {
    _output_size = config.get<int>(unique_name, "output_size");
    _ring_buffer_size = config.get<int>(unique_name, "ring_buffer_size");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    signal_buffer = dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "host_signal")));
    if (instance_num == 0)
        signal_buffer->register_consumer(unique_name);

    set_command_type(gpuCommandType::KERNEL);

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));
}

int cudaCopyFromRingbuffer::wait_on_precondition() {
    // Wait for there to be data available in the ringbuffer.
    DEBUG("Waiting for ringbuffer data for frame {:d}...", gpu_frame_id);
    std::optional<size_t> val = signal_buffer->wait_and_claim_readable(unique_name, _output_size);
    DEBUG("Finished waiting for data frame {:d}.", gpu_frame_id);
    if (!val.has_value())
        return -1;
    input_cursor = val.value();
    return 0;
}

cudaEvent_t cudaCopyFromRingbuffer::execute(cudaPipelineState& pipestate,
                                            const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    (void)pipestate;
    pre_execute();

    void* output_memory =
        device.get_gpu_memory_array(_gpu_mem_output, gpu_frame_id, _gpu_buffer_depth, _output_size);

    void* rb_memory = device.get_gpu_memory(_gpu_mem_input, _ring_buffer_size);

    size_t ncopy = _output_size;
    size_t nwrap = 0;
    if (input_cursor + _output_size > _ring_buffer_size) {
        ncopy = _ring_buffer_size - input_cursor;
        nwrap = _output_size - ncopy;
    }

    record_start_event();

    CHECK_CUDA_ERROR(cudaMemcpyAsync(output_memory, (char*)rb_memory + input_cursor, ncopy,
                                     cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));
    if (nwrap)
        CHECK_CUDA_ERROR(cudaMemcpyAsync((char*)output_memory + ncopy, rb_memory, nwrap,
                                         cudaMemcpyDeviceToDevice,
                                         device.getStream(cuda_stream_id)));

    return record_end_event();
}

void cudaCopyFromRingbuffer::finalize_frame() {
    cudaCommand::finalize_frame();
    signal_buffer->finish_read(unique_name, _output_size);
}
