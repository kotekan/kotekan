#include "cudaInputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaInputData);

cudaInputData::cudaInputData(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device,
                             int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num),
    in_buf(host_buffers.get_buffer(config.get<std::string>(unique_name, "in_buf"))),
    do_once(config.get_default<bool>(unique_name, "do_once", false)), did_once(false) {
    if (do_once && instance_num > 0)
        return;

    if (instance_num == 0)
        in_buf->register_consumer(unique_name);

    if (in_buf->frame_size) {
        uint flags;
        // only register the memory if it isn't already...
        if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, in_buf->frames[instance_num])) {
            CHECK_CUDA_ERROR(cudaHostRegister(in_buf->frames[instance_num], in_buf->frame_size, 0));
        }
        _gpu_mem = config.get<std::string>(unique_name, "gpu_mem");
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem, true, false, true));
    } else {
        _gpu_mem = "";
    }

    set_command_type(gpuCommandType::COPY_IN);
    set_name("input: " + _gpu_mem);
}

cudaInputData::~cudaInputData() {
    if (do_once && instance_num > 0)
        return;

    if (in_buf->frame_size) {
        uint flags;
        // only unregister if it's actually been registered
        if (cudaSuccess == cudaHostGetFlags(&flags, in_buf->frames[instance_num])) {
            CHECK_CUDA_ERROR(cudaHostUnregister(in_buf->frames[instance_num]));
        }
    }
}

int cudaInputData::wait_on_precondition() {
    if (do_once && instance_num > 0)
        return in_buf->shutdown_signal ? -1 : 0;
    if (do_once && did_once)
        return in_buf->shutdown_signal ? -1 : 0;

    // Wait for there to be data in the input (network) buffer.
    uint8_t* const frame =
        in_buf->wait_for_full_frame(unique_name, gpu_frame_id % in_buf->num_frames);
    if (frame == nullptr)
        return -1;

    return 0;
}

cudaEvent_t cudaInputData::execute(cudaPipelineState&, const std::vector<cudaEvent_t>& pre_events) {
    pre_execute();

    if (do_once && instance_num > 0)
        return end_event;
    if (do_once && did_once)
        return end_event;

    const int buf_index = gpu_frame_id % in_buf->num_frames;
    const size_t input_frame_len = in_buf->frame_size;

    if (input_frame_len) {
        void* const gpu_memory_frame =
            do_once ? device.get_gpu_memory(_gpu_mem, input_frame_len)
                    : device.get_gpu_memory_array(_gpu_mem, gpu_frame_id, _gpu_buffer_depth,
                                                  input_frame_len);
        void* const host_memory_frame = (void*)in_buf->frames[buf_index];

        device.async_copy_host_to_gpu(gpu_memory_frame, host_memory_frame, input_frame_len,
                                      cuda_stream_id, pre_events[cuda_stream_id], &start_event,
                                      &end_event);

        // Copy (reference to) metadata also
        std::shared_ptr<metadataObject> meta = in_buf->metadata[buf_index];
        if (meta)
            device.claim_gpu_memory_array_metadata(_gpu_mem, gpu_frame_id, meta);
    }

    return end_event;
}

void cudaInputData::finalize_frame() {
    cudaCommand::finalize_frame();

    if (do_once && instance_num > 0)
        return;
    if (do_once && did_once)
        return;

    in_buf->mark_frame_empty(unique_name, gpu_frame_id % in_buf->num_frames);

    did_once = true;
}

std::string cudaInputData::get_performance_metric_string() {
    const double t = get_last_gpu_execution_time();
    const double transfer_speed = in_buf->frame_size / t;
    return fmt::format("Time: {:.3f} ms, Speed: {:.2f} GB/s ({:.2f} Gb/s)", t * 1.0e+3,
                       transfer_speed / 1.0e+9, 8 * transfer_speed / 1.0e+9);
}
