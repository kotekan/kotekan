#include "cudaCopyToRingbuffer.hpp"

#include "cudaUtils.hpp"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCopyToRingbuffer);

cudaCopyToRingbuffer::cudaCopyToRingbuffer(Config& config, const std::string& unique_name,
                                           bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "cudaCopyToRingbuffer", "") {
    _input_size = config.get<int>(unique_name, "input_size");
    _output_size = config.get<int>(unique_name, "output_size");
    _ring_buffer_size = config.get<int>(unique_name, "ring_buffer_size");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    //_input_columns_field = config.get_default<std::string>(unique_name, "input_columns_field", "");
    signal_buffer = host_buffers.get_buffer(config.get<std::string>(unique_name, "host_signal"));
    register_producer(signal_buffer, unique_name.c_str());
    buffer_set_ring_buffer_size(signal_buffer, _ring_buffer_size);
    buffer_set_ring_buffer_read_size(signal_buffer, _output_size);

    output_cursor = 0;
    set_command_type(gpuCommandType::KERNEL);

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));
}

int cudaCopyToRingbuffer::wait_on_precondition(int gpu_frame_id) {
    return buffer_wait_for_ring_buffer_writable(signal_buffer, _input_size);
}

cudaEvent_t cudaCopyToRingbuffer::execute(cudaPipelineState& pipestate,
                                          const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(pipestate.gpu_frame_id);

    void* input_memory =
        device.get_gpu_memory_array(_gpu_mem_input, pipestate.gpu_frame_id, _input_size);

    void* rb_memory = device.get_gpu_memory(_gpu_mem_output, _ring_buffer_size);

    size_t ncopy = _input_size;
    size_t nwrap = 0;
    if (output_cursor + _input_size > _ring_buffer_size) {
        ncopy = _ring_buffer_size - output_cursor;
        nwrap = _input_size - ncopy;
    }

    record_start_event(pipestate.gpu_frame_id);
    
    CHECK_CUDA_ERROR(cudaMemcpyAsync((char*)rb_memory + output_cursor, input_memory, ncopy,
                                     cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));
    if (nwrap)
        CHECK_CUDA_ERROR(cudaMemcpyAsync(rb_memory, (char*)input_memory + ncopy, nwrap,
                                         cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));

    output_cursor = (output_cursor + _input_size) % _ring_buffer_size;
    // FIXME -- signal *now*, when we have *queued* the cuda work?  Or in finalize_frame, when it has finished??
    buffer_wrote_to_ring_buffer(signal_buffer, _input_size);

    return record_end_event(pipestate.gpu_frame_id);
}

//void cudaCopyToRingbuffer::finalize_frame(int gpu_frame_id) {
// At this point we know the Cuda copy completed, but do we *really* need that to be the case??
//return buffer_wrote_to_ringbuffer(signal_buffer, _input_size);
//}

