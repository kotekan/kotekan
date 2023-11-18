#include "cudaCopyFromRingbuffer.hpp"

#include "cudaUtils.hpp"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCopyFromRingbuffer);

cudaCopyFromRingbuffer::cudaCopyFromRingbuffer(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "cudaCopyFromRingbuffer", "") {
    _output_size = config.get<int>(unique_name, "output_size");
    _ring_buffer_size = config.get<int>(unique_name, "ring_buffer_size");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    signal_buffer = host_buffers.get_buffer(config.get<std::string>(unique_name, "host_signal"));
    register_consumer(signal_buffer, unique_name.c_str());

    input_cursor = 0;
    set_command_type(gpuCommandType::KERNEL);

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));
}

int cudaCopyFromRingbuffer::wait_on_precondition(int frame_id) {
    // Wait for there to be data in the input (network) buffer.
    INFO("Waiting for data frame {:d}...", frame_id);
    //uint8_t* frame = wait_for_full_frame(signal_buffer, unique_name.c_str(), frame_id);
    uint8_t* frame = buffer_claim_next_full_frame(signal_buffer, unique_name.c_str(), frame_id);
    INFO("Finished waiting for data frame {:d}.", frame_id);
    if (frame == nullptr)
        return -1;
    return 0;
}

cudaEvent_t cudaCopyFromRingbuffer::execute(cudaPipelineState& pipestate,
                                            const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(pipestate.gpu_frame_id);

    void* output_memory =
        device.get_gpu_memory_array(_gpu_mem_output, pipestate.gpu_frame_id, _output_size);

    void* rb_memory = device.get_gpu_memory(_gpu_mem_input, _ring_buffer_size);

    size_t ncopy = _output_size;
    size_t nwrap = 0;
    if (input_cursor + _output_size > _ring_buffer_size) {
        ncopy = _ring_buffer_size - input_cursor;
        nwrap = _output_size - ncopy;
    }

    record_start_event(pipestate.gpu_frame_id);
    
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output_memory, (char*)rb_memory + input_cursor, ncopy,
                                     cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));
    if (nwrap)
        CHECK_CUDA_ERROR(cudaMemcpyAsync((char*)output_memory + ncopy, rb_memory, nwrap,
                                         cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));

    input_cursor = (input_cursor + _output_size) % _ring_buffer_size;
    return record_end_event(pipestate.gpu_frame_id);
}

void cudaCopyFromRingbuffer::finalize_frame(int frame_id) {
    cudaCommand::finalize_frame(frame_id);
    mark_frame_empty(signal_buffer, unique_name.c_str(), frame_id);
}
