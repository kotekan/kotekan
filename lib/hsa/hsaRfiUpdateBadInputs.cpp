#include "hsaRfiUpdateBadInputs.hpp"

#include "configUpdater.hpp"
#include "visUtil.hpp"

#include "fmt/ranges.h"

using kotekan::Config;

REGISTER_HSA_COMMAND(hsaRfiUpdateBadInputs);

hsaRfiUpdateBadInputs::hsaRfiUpdateBadInputs(Config& config, const string& unique_name,
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
    register_consumer(_network_buf, unique_name.c_str());

    _network_buf_precondition_id = 0;
    _network_buf_execute_id = 0;
    _network_buf_finalize_id = 0;

    _in_buf = host_buffers.get_buffer("bad_inputs_buf");
    register_consumer(_in_buf, unique_name.c_str());
    _in_buf_finalize_id = 0;
    _in_buf_precondition_id = 0;
    frame_to_fill_finalize = 0;
    first_pass = true;
    num_bad_inputs = 0;

    frames_to_update = 0;
    frame_copy_active.insert(std::begin(frame_copy_active), device.get_gpu_buffer_depth(), false);

    // Alloc memory on GPU
    device.get_gpu_memory_array("input_mask", 0, input_mask_len);
}

hsaRfiUpdateBadInputs::~hsaRfiUpdateBadInputs() {
    hsa_host_free(host_mask);
}

int hsaRfiUpdateBadInputs::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Check for bad input updates
    std::lock_guard<std::mutex> lock(update_mutex);
    if (first_pass) {
        DEBUG("hsaRfiUpdateBadInputs::wait_on_precondition: Waiting for full frame...");
        uint8_t* frame = wait_for_full_frame(_in_buf, unique_name.c_str(), _in_buf_precondition_id);
        DEBUG("hsaRfiUpdateBadInputs::wait_on_precondition: Bad inputs update.");
        if (frame == NULL)
            return -1;
        first_pass = false;
        frames_to_update = device.get_gpu_buffer_depth();
        frame_to_fill_finalize = frames_to_update;
        memcpy(host_mask, frame, input_mask_len);
        num_bad_inputs = get_rfi_num_bad_inputs(_in_buf, _in_buf_precondition_id);
        _in_buf_precondition_id = (_in_buf_precondition_id + 1) % _in_buf->num_frames;
    } else {
        // Check for new bad inputs only if all gpu frames have been updated (not currently updating frame)
        if (frames_to_update == 0 && !frame_copy_active.at(gpu_frame_id)) {
            auto timeout = double_to_ts(0);
            int status = wait_for_full_frame_timeout(_in_buf, unique_name.c_str(),
                                                     _in_buf_precondition_id, timeout);
            DEBUG("status of bad inputs _in_buf_precondition_id[{:d}]={:d} (0=ready 1=not)",
                  _in_buf_precondition_id, status);
            if (status == 0) {
                frames_to_update = device.get_gpu_buffer_depth();
                frame_to_fill_finalize = frames_to_update;
                memcpy(host_mask, _in_buf->frames[_in_buf_precondition_id], input_mask_len);
                num_bad_inputs = get_rfi_num_bad_inputs(_in_buf, _in_buf_precondition_id);
                _in_buf_precondition_id = (_in_buf_precondition_id + 1) % _in_buf->num_frames;
            }
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
        DEBUG("Copying bad input list to GPU[{:d}], frames to update: {:d}",
              device.get_gpu_id(), frames_to_update);
        void* gpu_mem = device.get_gpu_memory_array("input_mask", gpu_frame_id, input_mask_len);
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
    if (frame_to_fill_finalize > 0 && frame_copy_active.at(frame_id)) {
        frame_copy_active.at(frame_id) = false;
        hsaCommand::finalize_frame(frame_id);
        INFO("finalize_frame for gpu_frame_id={:d} using _in_buf_finalize_id={:d}", frame_id,
              _in_buf_finalize_id);
        frame_to_fill_finalize--;
        if (frame_to_fill_finalize == 0) {
            mark_frame_empty(_in_buf, unique_name.c_str(), _in_buf_finalize_id);
            _in_buf_finalize_id = (_in_buf_finalize_id + 1) % _in_buf->num_frames;
        }
    }

    mark_frame_empty(_network_buf, unique_name.c_str(), _network_buf_finalize_id);
    _network_buf_finalize_id = (_network_buf_finalize_id + 1) % _network_buf->num_frames;
}
