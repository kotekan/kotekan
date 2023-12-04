#include "cudaCopyToRingbuffer.hpp"

#include "cudaUtils.hpp"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

class cudaCopyToRingbufferState : public cudaCommandState {
public:
    cudaCopyToRingbufferState(kotekan::Config& config, const std::string& unique_name, kotekan::bufferContainer& buffers, cudaDeviceInterface& dev) :
        cudaCommandState(config, unique_name, buffers, dev),
        cursor(0) {}
    int cursor;
};

static cudaCopyToRingbufferState* get_state(std::shared_ptr<gpuCommandState> state) {
    return static_cast<cudaCopyToRingbufferState*>(state.get());
}

REGISTER_CUDA_COMMAND_WITH_STATE(cudaCopyToRingbuffer, cudaCopyToRingbufferState);

cudaCopyToRingbuffer::cudaCopyToRingbuffer(Config& config, const std::string& unique_name,
                                           bufferContainer& host_buffers,
                                           cudaDeviceInterface& device, int instance_num,
                                           const std::shared_ptr<cudaCommandState>& state) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, state,
                "cudaCopyToRingbuffer", "") {
    _input_size = config.get<int>(unique_name, "input_size");
    _ring_buffer_size = config.get<int>(unique_name, "ring_buffer_size");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    //_input_columns_field = config.get_default<std::string>(unique_name, "input_columns_field",
    //"");
    signal_buffer = dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "host_signal")));
    assert(signal_buffer);
    if (instance_num == 0)
        signal_buffer->register_producer(unique_name);

    set_command_type(gpuCommandType::KERNEL);

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input,  true,  true,  false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, false, false, true));
}

int cudaCopyToRingbuffer::wait_on_precondition() {
    return signal_buffer->wait_for_writable(unique_name, _input_size);
}

cudaEvent_t cudaCopyToRingbuffer::execute(cudaPipelineState& pipestate,
                                          const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    (void)pipestate;
    pre_execute();

    void* input_memory =
        device.get_gpu_memory_array(_gpu_mem_input, gpu_frame_id, _gpu_buffer_depth, _input_size);

    void* rb_memory = device.get_gpu_memory(_gpu_mem_output, _ring_buffer_size);

    size_t output_cursor = get_state(command_state)->cursor;

    size_t ncopy = _input_size;
    size_t nwrap = 0;
    if (output_cursor + _input_size > _ring_buffer_size) {
        ncopy = _ring_buffer_size - output_cursor;
        nwrap = _input_size - ncopy;
    }

    record_start_event();

    CHECK_CUDA_ERROR(cudaMemcpyAsync((char*)rb_memory + output_cursor, input_memory, ncopy,
                                     cudaMemcpyDeviceToDevice, device.getStream(cuda_stream_id)));
    if (nwrap)
        CHECK_CUDA_ERROR(cudaMemcpyAsync(rb_memory, (char*)input_memory + ncopy, nwrap,
                                         cudaMemcpyDeviceToDevice,
                                         device.getStream(cuda_stream_id)));

    output_cursor = (output_cursor + _input_size) % _ring_buffer_size;
    get_state(command_state)->cursor = output_cursor;
    // FIXME -- signal *now*, when we have *queued* the cuda work?  Or in finalize_frame, when it
    // has finished?? if we do it here, probably need a syncInput after the cudaInput that is
    // waiting on this buffer.
    // signal_buffer->wrote(unique_name, _input_size);

    return record_end_event();
}

void cudaCopyToRingbuffer::finalize_frame() {
    cudaCommand::finalize_frame();
    // At this point we know the Cuda copy completed, but do we *really* need that to be the case??
    signal_buffer->finish_write(unique_name, _input_size);
}
