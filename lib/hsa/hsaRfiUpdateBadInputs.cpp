#include "hsaRfiUpdateBadInputs.hpp"

#include "configUpdater.hpp"
#include "visUtil.hpp"

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

    // Alloc memory on GPU
    device.get_gpu_memory_array("input_mask", 0, input_mask_len);

    kotekan::configUpdater::instance().subscribe(
        config.get<std::string>(unique_name, "updatable_config/gain_psr"),
        std::bind(&hsaRfiUpdateBadInputs::update_bad_inputs_callback, this, std::placeholders::_1));
}

hsaRfiUpdateBadInputs::~hsaRfiUpdateBadInputs() {
    hsa_host_free(host_mask);
}

int hsaRfiUpdateBadInputs::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    uint8_t* frame =
        wait_for_full_frame(_network_buf, unique_name.c_str(), _network_buf_precondition_id);
    if (frame == nullptr)
        return -1;

    _network_buf_precondition_id = (_network_buf_precondition_id + 1) % _network_buf->num_frames;
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

bool hsaRfiUpdateBadInputs::update_bad_inputs_callback(nlohmann::json& json) {

    // It might be possible to reduce the scope of this lock.
    // However all the operations in this command object are async, so it should be fast.
    std::lock_guard<std::mutex> lock(update_mutex);

    // TODO this isn't ideal, we could read off a buffer and then apply bad inputs coming from
    // that buffer fed by a stage.  However for the bad inputs list, we are very unlikely to hit
    // this condition. This will be fixed when we refactor the command objects and create
    // a generic command object for optional memory array copies.
    // Note for this to happen we'd need two bad input list updates within less than ~0.5 seconds.
    if (update_bad_inputs) {
        WARN("Got new bad inputs list before applying the last list, not applying new bad inputs!");
        return true;
    }

    try {
        bad_inputs_cylinder = json["bad_inputs"].get<std::vector<int>>();
    } catch (std::exception const& e) {
        ERROR("Failed to parse bad input list {:s}", e.what());
        return false;
    }

    // Reorder list
    bad_inputs_correlator.clear();
    for (auto element : bad_inputs_cylinder)
        bad_inputs_correlator.push_back(input_remap[element]);

    // Zero bad inputs mask
    for (uint32_t i = 0; i < input_mask_len; ++i) {
        host_mask[i] = 1;
    }

    // Add current bad input mask
    for (auto element : bad_inputs_correlator) {
        if (element < (int)input_mask_len && element >= 0) {
            host_mask[element] = 0;
        } else {
            ERROR("Got a bad input with invalid index");
            return false;
        }
    }

    update_bad_inputs = true;
    frames_to_update = device.get_gpu_buffer_depth();
    frames_to_update_finalize = frames_to_update;
    return true;
}