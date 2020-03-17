#include "hsaInputLostSamples.hpp"

#include "Config.hpp"             // for Config
#include "buffer.h"               // for Buffer, mark_frame_empty, wait_for_full_frame
#include "bufferContainer.hpp"    // for bufferContainer
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::COPY_IN
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config

#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaInputLostSamples);

hsaInputLostSamples::hsaInputLostSamples(Config& config, const std::string& unique_name,
                                         bufferContainer& host_buffers,
                                         hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "", "") {
    command_type = gpuCommandType::COPY_IN;

    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    input_frame_len = _samples_per_data_set * sizeof(uint32_t);

    lost_samples_buf = host_buffers.get_buffer("lost_samples_buf");
    lost_samples_buffer_id = 0;
    lost_samples_buffer_precondition_id = 0;
    lost_samples_buffer_finalize_id = 0;
}

hsaInputLostSamples::~hsaInputLostSamples() {}

int hsaInputLostSamples::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Wait for there to be data in the input buffer.
    uint8_t* frame = wait_for_full_frame(lost_samples_buf, unique_name.c_str(),
                                         lost_samples_buffer_precondition_id);
    if (frame == nullptr)
        return -1;
    lost_samples_buffer_precondition_id =
        (lost_samples_buffer_precondition_id + 1) % lost_samples_buf->num_frames;
    return 0;
}

hsa_signal_t hsaInputLostSamples::execute(int gpu_frame_id, hsa_signal_t precede_signal) {
    // Get the gpu and cpu memory pointers.
    void* gpu_memory_frame =
        device.get_gpu_memory_array("lost_samples", gpu_frame_id, input_frame_len);
    void* host_memory_frame = (void*)lost_samples_buf->frames[lost_samples_buffer_id];
    // Do the input data copy.
    device.async_copy_host_to_gpu(gpu_memory_frame, host_memory_frame, input_frame_len,
                                  precede_signal, signals[gpu_frame_id]);
    lost_samples_buffer_id = (lost_samples_buffer_id + 1) % lost_samples_buf->num_frames;
    return signals[gpu_frame_id];
}

void hsaInputLostSamples::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);
    mark_frame_empty(lost_samples_buf, unique_name.c_str(), lost_samples_buffer_finalize_id);
    lost_samples_buffer_finalize_id =
        (lost_samples_buffer_finalize_id + 1) % lost_samples_buf->num_frames;
}
