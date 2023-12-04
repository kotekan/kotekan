#include "cudaCopyFromRingbuffer.hpp"

#include "cudaUtils.hpp"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

class cudaCopyFromRingbufferState : public cudaCommandState {
public:
    cudaCopyFromRingbufferState(kotekan::Config& config, const std::string& unique_name,
                                kotekan::bufferContainer& buffers, cudaDeviceInterface& dev) :
        cudaCommandState(config, unique_name, buffers, dev),
        cursor(0) {}
    int cursor;
};

static cudaCopyFromRingbufferState* get_state(std::shared_ptr<gpuCommandState> state) {
    return static_cast<cudaCopyFromRingbufferState*>(state.get());
}

REGISTER_CUDA_COMMAND_WITH_STATE(cudaCopyFromRingbuffer, cudaCopyFromRingbufferState);

cudaCopyFromRingbuffer::cudaCopyFromRingbuffer(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               cudaDeviceInterface& device, int instance_num,
                                               const std::shared_ptr<cudaCommandState>& state) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, state,
                "cudaCopyFromRingbuffer", "") {
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
    // Wait for there to be data in the input (network) buffer.
    DEBUG("Waiting for data frame {:d}...", gpu_frame_id);
    int offset = signal_buffer->wait_and_claim_readable(unique_name, _output_size);
    DEBUG("Finished waiting for data frame {:d}.", gpu_frame_id);
    if (offset == -1)
        return -1;
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

    size_t input_cursor = get_state(command_state)->cursor;

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

    input_cursor = (input_cursor + _output_size) % _ring_buffer_size;
    get_state(command_state)->cursor = input_cursor;
    return record_end_event();
}

void cudaCopyFromRingbuffer::finalize_frame() {
    cudaCommand::finalize_frame();
    signal_buffer->finish_read(unique_name, _output_size);
}
