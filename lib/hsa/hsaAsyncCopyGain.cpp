#include "hsaAsyncCopyGain.hpp"

#include "buffer.hpp"             // for Buffer, mark_frame_empty, register_consumer, wait_for_...
#include "bufferContainer.hpp"    // for bufferContainer
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::COPY_IN
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface
#include "kotekanLogging.hpp"     // for DEBUG
#include "visUtil.hpp"            // for double_to_ts

#include <algorithm> // for max
#include <iterator>  // for begin

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaAsyncCopyGain);

hsaAsyncCopyGain::hsaAsyncCopyGain(Config& config, const std::string& unique_name,
                                   bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "hsaAsyncCopyGain", "") {
    command_type = gpuCommandType::COPY_IN;

    gain_len = 2 * 2048 * sizeof(float);
    gain_buf = host_buffers.get_buffer("gain_frb_buf");
    gain_buf->register_consumer(unique_name);
    gain_buf_id = 0;
    gain_buf_finalize_id = 0;
    gain_buf_precondition_id = 0;
    frames_to_update = 0;
    frame_copy_active.insert(std::begin(frame_copy_active), device.get_gpu_buffer_depth(), false);
    first_pass = true;
}

hsaAsyncCopyGain::~hsaAsyncCopyGain() {}

int hsaAsyncCopyGain::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    std::lock_guard<std::mutex> lock(update_mutex);

    // Check for new gains
    if (first_pass) {
        uint8_t* frame =
            wait_for_full_frame(gain_buf, unique_name.c_str(), gain_buf_precondition_id);
        if (frame == nullptr)
            return -1;
        gain_buf_precondition_id = (gain_buf_precondition_id + 1) % gain_buf->num_frames;
        first_pass = false;
        frames_to_update = device.get_gpu_buffer_depth();
    } else {
        // Check for new gains only if filled all gpu frames (not currently updating frames)
        bool current_update_active = false;
        for (bool in_use : frame_copy_active) {
            if (in_use) {
                current_update_active = true;
                break;
            }
        }
        if (frames_to_update == 0 && !current_update_active) {
            auto timeout = double_to_ts(0);
            int status = wait_for_full_frame_timeout(gain_buf, unique_name.c_str(),
                                                     gain_buf_precondition_id, timeout);
            DEBUG("status of gain_buf_precondition_id[{:d}]={:d} (0=ready 1=not)",
                  gain_buf_precondition_id, status);
            if (status == 0) {
                frames_to_update = device.get_gpu_buffer_depth();
                gain_buf_precondition_id = (gain_buf_precondition_id + 1) % gain_buf->num_frames;
            }
            if (status == -1)
                return -1;
        }
    }
    return 0;
}


hsa_signal_t hsaAsyncCopyGain::execute(int gpu_frame_id, hsa_signal_t precede_signal) {
    std::lock_guard<std::mutex> lock(update_mutex);

    if (frames_to_update > 0) {
        frame_copy_active.at(gpu_frame_id) = true;
        DEBUG("Going to async copy gain_buf_id={:d} gpu_frame_id={:d}", gain_buf_id, gpu_frame_id);
        void* device_gain = device.get_gpu_memory_array("beamform_gain", gpu_frame_id, gain_len);
        void* host_gain = (void*)gain_buf->frames[gain_buf_id];
        device.async_copy_host_to_gpu(device_gain, host_gain, gain_len, precede_signal,
                                      signals[gpu_frame_id]);

        frames_to_update--;
        if (frames_to_update == 0) {
            gain_buf_id = (gain_buf_id + 1) % gain_buf->num_frames;
        }
        return signals[gpu_frame_id];
    }
    return precede_signal;
}

void hsaAsyncCopyGain::finalize_frame(int frame_id) {
    std::lock_guard<std::mutex> lock(update_mutex);

    if (frame_copy_active.at(frame_id)) {
        frame_copy_active.at(frame_id) = false;
        hsaCommand::finalize_frame(frame_id);
        DEBUG("finalize_frame for gpu_frame_id={:d} using gain_buf_finalize_id={:d}", frame_id,
              gain_buf_finalize_id);

        bool current_update_active = false;
        for (bool in_use : frame_copy_active) {
            if (in_use) {
                current_update_active = true;
                break;
            }
        }
        // We've updated all required GPU frames and can release the host frame
        if (!current_update_active && frames_to_update == 0) {
            mark_frame_empty(gain_buf, unique_name.c_str(), gain_buf_finalize_id);
            gain_buf_finalize_id = (gain_buf_finalize_id + 1) % gain_buf->num_frames;
        }
    }
}
