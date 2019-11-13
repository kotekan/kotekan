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

    auto input_reorder = parse_reorder_default(config, unique_name);
    input_remap = std::get<0>(input_reorder);

    host_mask = (uint8_t*)hsa_host_malloc(input_mask_len, device.get_gpu_numa_node());
    CHECK_MEM(host_mask);

    // Get buffers (for metadata)
    _network_buf = host_buffers.get_buffer("network_buf");
    register_consumer(_network_buf, unique_name.c_str());

    _network_buf_precondition_id = 0;
    _network_buf_execute_id = 0;
    _network_buf_finalize_id = 0;

    _in_buf_len = num_elements * sizeof(uint8_t);
    _in_buf = host_buffers.get_buffer("bad_inputs_buf");
    register_consumer(_in_buf, unique_name.c_str());
    _in_buf_id = 0;
    _in_buf_finalize_id = 0;
    _in_buf_precondition_id = 0;
    frame_to_fill = 0;
    frame_to_fill_finalize = 0;
    filling_frame = false;
    first_pass = true;

    update_bad_inputs = false;
    frames_to_update = 0;
    frames_to_update_finalize = 0;

    // Alloc memory on GPU
    device.get_gpu_memory_array("input_mask", 0, input_mask_len);
    
}

hsaRfiUpdateBadInputs::~hsaRfiUpdateBadInputs() {
    hsa_host_free(host_mask);
}

int hsaRfiUpdateBadInputs::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Check for bad input updates
    if (first_pass) {
        uint8_t* frame =
            wait_for_full_frame(_in_buf, unique_name.c_str(), _in_buf_precondition_id);
        if (frame == NULL)
            return -1;
        _in_buf_precondition_id = (_in_buf_precondition_id + 1) % _in_buf->num_frames;
        first_pass = false;
        frame_to_fill = device.get_gpu_buffer_depth();
        frame_to_fill_finalize = frame_to_fill;
        filling_frame = true;
    } else {
        // Check for new bad inputs only if filled all gpu frames (not currently filling frame)
        if (!filling_frame) {
            auto timeout = double_to_ts(0);
            int status = wait_for_full_frame_timeout(_in_buf, unique_name.c_str(),
                                                     _in_buf_precondition_id, timeout);
            DEBUG("status of bad inputs _in_buf_precondition_id[{:d}]={:d} (0=ready 1=not)",
                  _in_buf_precondition_id, status);
            if (status == 0) {
                filling_frame = true;
                frame_to_fill = device.get_gpu_buffer_depth();
                frame_to_fill_finalize = frame_to_fill;
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
    set_rfi_num_bad_inputs(_network_buf, _network_buf_execute_id, bad_inputs_correlator.size());
    _network_buf_execute_id = (_network_buf_execute_id + 1) % _network_buf->num_frames;

    if (update_bad_inputs && frames_to_update > 0) {
        frames_to_update--;

        // Copy memory to GPU
        DEBUG("Copying bad input list to GPU[{:d}], frames to update: {:d}, update: {}, cylinder "
              "order: {}, correlator order: {}",
              device.get_gpu_id(), frames_to_update, update_bad_inputs, bad_inputs_cylinder,
              bad_inputs_correlator);
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
    if (update_bad_inputs && frames_to_update_finalize > 0) {
        frames_to_update_finalize--;
        hsaCommand::finalize_frame(frame_id);
        // Check if we've finished loading the new bad input mask in all frames.
        if (frames_to_update_finalize == 0) {
            update_bad_inputs = false;
        }
    }

    mark_frame_empty(_network_buf, unique_name.c_str(), _network_buf_finalize_id);
    _network_buf_finalize_id = (_network_buf_finalize_id + 1) % _network_buf->num_frames;
}
