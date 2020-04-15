#include "hsaHostToDeviceCopy.hpp"

#include "gpuCommand.hpp"     // for gpuCommandType, gpuCommandType::COPY_IN
#include "kotekanLogging.hpp" // for DEBUG2

#include <exception> // for exception
#include <stdint.h>  // for uint8_t

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaHostToDeviceCopy);

hsaHostToDeviceCopy::hsaHostToDeviceCopy(Config& config, const std::string& unique_name,
                                         bufferContainer& host_buffers,
                                         hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "hsaHostToDeviceCopy", ""),
    in_buf(host_buffers.get_buffer(config.get<std::string>(unique_name, "in_buf"))),
    in_buf_id(in_buf),
    in_buf_precondition_id(in_buf),
    in_buf_finalize_id(in_buf),
    _gpu_memory_name(config.get<std::string>(unique_name, "gpu_memory_name")) {
    command_type = gpuCommandType::COPY_IN;

    register_consumer(in_buf, unique_name.c_str());
}

hsaHostToDeviceCopy::~hsaHostToDeviceCopy() {}

int hsaHostToDeviceCopy::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Wait for there to be data in the input buffer.
    uint8_t* frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buf_precondition_id);
    if (frame == nullptr)
        return -1;
    in_buf_precondition_id++;

    return 0;
}

hsa_signal_t hsaHostToDeviceCopy::execute(int gpu_frame_id, hsa_signal_t precede_signal) {
    // Get the gpu and cpu memory pointers.
    void* gpu_memory_frame =
        device.get_gpu_memory_array(_gpu_memory_name, gpu_frame_id, in_buf->frame_size);
    void* host_memory_frame = (void*)in_buf->frames[(int)in_buf_id];

    DEBUG2("Copy data to GPU frame name: {} from buffer: {}", _gpu_memory_name,
           in_buf->buffer_name);

    // Do the input data copy.
    device.async_copy_host_to_gpu(gpu_memory_frame, host_memory_frame, in_buf->frame_size,
                                  precede_signal, signals[gpu_frame_id]);
    in_buf_id++;

    return signals[gpu_frame_id];
}

void hsaHostToDeviceCopy::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);
    mark_frame_empty(in_buf, unique_name.c_str(), in_buf_finalize_id);
    in_buf_finalize_id++;
}
