#include "hsaRfiUpdateBadInputs.hpp"

#include "Config.hpp"             // for Config
#include "buffer.hpp"             // for mark_frame_empty, register_consumer, Buffer, wait_for_...
#include "bufferContainer.hpp"    // for bufferContainer
#include "chimeMetadata.hpp"      // for get_rfi_num_bad_inputs, set_rfi_num_bad_inputs
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::COPY_IN
#include "hsaBase.h"              // for hsa_host_free, hsa_host_malloc
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config
#include "kotekanLogging.hpp"     // for DEBUG, CHECK_MEM
#include "visUtil.hpp"            // for double_to_ts

#include <algorithm> // for max
#include <exception> // for exception
#include <iterator>  // for begin
#include <regex>     // for match_results<>::_Base_type
#include <string.h>  // for memcpy

using kotekan::Config;

REGISTER_HSA_COMMAND(hsaRfiUpdateBadInputs);

hsaRfiUpdateBadInputs::hsaRfiUpdateBadInputs(Config& config, const std::string& unique_name,
                                             kotekan::bufferContainer& host_buffers,
                                             hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "hsaRfiUpdateBadInputs", "") {
    command_type = gpuCommandType::COPY_IN;

    uint32_t num_elements = config.get<uint32_t>(unique_name, "num_elements");
    input_mask_len = sizeof(uint8_t) * num_elements;

    host_mask = (uint8_t*)hsa_host_malloc(input_mask_len, device.get_gpu_numa_node());
    CHECK_MEM(host_mask);

    // Get buffers (for metadata)
    _network_buf = host_buffers.get_buffer("network_buf");
    _network_buf->register_consumer(unique_name);

    _network_buf_precondition_id = 0;
    _network_buf_execute_id = 0;
    _network_buf_finalize_id = 0;

    _in_buf = host_buffers.get_buffer("bad_inputs_buf");
    _in_buf->register_consumer(unique_name);
    _in_buf_precondition_id = 0;
    first_pass = true;
    num_bad_inputs = 0;

    frames_to_update = 0;
    frame_copy_active.insert(std::begin(frame_copy_active), _gpu_buffer_depth, false);

    // Alloc memory on GPU
    device.get_gpu_memory_array("input_mask", 0, _gpu_buffer_depth, input_mask_len);
}

hsaRfiUpdateBadInputs::~hsaRfiUpdateBadInputs() {
    hsa_host_free(host_mask);
}

inline void hsaRfiUpdateBadInputs::copy_frame(int gpu_frame_id) {
    (void)gpu_frame_id;

    frames_to_update = _gpu_buffer_depth;
    memcpy(host_mask, _in_buf->frames[_in_buf_precondition_id], input_mask_len);
    num_bad_inputs = get_rfi_num_bad_inputs(_in_buf, _in_buf_precondition_id);
    DEBUG("gpu_frame_id={:d} using _in_buf_precondition_id={:d}", gpu_frame_id,
          _in_buf_precondition_id);
    _in_buf->mark_frame_empty(unique_name, _in_buf_precondition_id);
    _in_buf_precondition_id = (_in_buf_precondition_id + 1) % _in_buf->num_frames;
}

int hsaRfiUpdateBadInputs::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Check for bad input updates
    std::lock_guard<std::mutex> lock(update_mutex);
    if (first_pass) {
        uint8_t* frame = _in_buf->wait_for_full_frame(unique_name, _in_buf_precondition_id);
        if (frame == nullptr)
            return -1;
        first_pass = false;
        copy_frame(gpu_frame_id);
    } else {
        // Check for new bad inputs only if all gpu frames have been updated (not currently updating
        // frame)
        bool current_update_active = false;
        for (bool in_use : frame_copy_active) {
            if (in_use) {
                current_update_active = true;
                break;
            }
        }
        if (frames_to_update == 0 && !current_update_active) {
            auto timeout = double_to_ts(0);
            int status =
                _in_buf->wait_for_full_frame_timeout(unique_name, _in_buf_precondition_id, timeout);
            DEBUG("status of bad inputs _in_buf_precondition_id[{:d}]={:d} (0=ready 1=not)",
                  _in_buf_precondition_id, status);
            if (status == 0)
                copy_frame(gpu_frame_id);
            if (status == -1)
                return -1;
        }
    }
    return 0;
}

hsa_signal_t hsaRfiUpdateBadInputs::execute(int gpu_frame_id, hsa_signal_t precede_signal) {
    std::lock_guard<std::mutex> lock(update_mutex);

    // We need to set the number of bad inputs used in this frame
    set_rfi_num_bad_inputs(_network_buf, _network_buf_execute_id, num_bad_inputs);
    _network_buf_execute_id = (_network_buf_execute_id + 1) % _network_buf->num_frames;

    if (frames_to_update > 0) {
        frames_to_update--;
        frame_copy_active.at(gpu_frame_id) = true;

        // Copy memory to GPU
        DEBUG("Copying bad input list to GPU[{:d}], frames to update: {:d}", device.get_gpu_id(),
              frames_to_update);
        void* gpu_mem = device.get_gpu_memory_array("input_mask", gpu_frame_id, _gpu_buffer_depth,
                                                    input_mask_len);
        device.async_copy_host_to_gpu(gpu_mem, (void*)host_mask, input_mask_len, precede_signal,
                                      signals[gpu_frame_id]);

        return signals[gpu_frame_id];
    } else {
        return precede_signal;
    }
}

void hsaRfiUpdateBadInputs::finalize_frame(int frame_id) {
    std::lock_guard<std::mutex> lock(update_mutex);

    // Only mark input empty if filling frame and
    // no more frames to finalize.
    if (frame_copy_active.at(frame_id)) {
        frame_copy_active.at(frame_id) = false;
        hsaCommand::finalize_frame(frame_id);
    }

    _network_buf->mark_frame_empty(unique_name, _network_buf_finalize_id);
    _network_buf_finalize_id = (_network_buf_finalize_id + 1) % _network_buf->num_frames;
}
