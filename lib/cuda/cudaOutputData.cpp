/**
 * @file
 * @brief CUDA command to copy data from the GPU to the host
 *  - cudaOutputData : public cudaCommand
 */

#include "cudaOutputData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaOutputData);

cudaOutputData::cudaOutputData(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device,
                               int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num) {
    std::string in_buf_name = config.get_default<std::string>(unique_name, "in_buf", "");
    if (in_buf_name.size()) {
        in_buffer = host_buffers.get_buffer(in_buf_name);
        if (instance_num == 0)
            register_consumer(in_buffer, unique_name.c_str());
    } else
        in_buffer = nullptr;

    output_buffer = host_buffers.get_buffer(config.get<std::string>(unique_name, "out_buf"));
    if (instance_num == 0) {
        register_producer(output_buffer, unique_name.c_str());
        for (int i = 0; i < output_buffer->num_frames; i++) {
            uint flags;
            // only register the memory if it isn't already...
            if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, output_buffer->frames[i])) {
                CHECK_CUDA_ERROR(
                    cudaHostRegister(output_buffer->frames[i], output_buffer->frame_size, 0));
            }
        }
    }

    _gpu_mem = config.get<std::string>(unique_name, "gpu_mem");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem, true, true, false));

    set_command_type(gpuCommandType::COPY_OUT);
    set_name("output: " + _gpu_mem);
}

cudaOutputData::~cudaOutputData() {
    if (instance_num == 0)
        for (int i = 0; i < output_buffer->num_frames; i++) {
            uint flags;
            // only register the memory if it isn't already...
            if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, output_buffer->frames[i])) {
                CHECK_CUDA_ERROR(cudaHostUnregister(output_buffer->frames[i]));
            }
        }
}

int cudaOutputData::wait_on_precondition() {
    // Wait for there to be space in the host output buffer
    uint8_t* frame = wait_for_empty_frame(output_buffer, unique_name.c_str(),
                                          gpu_frame_id % output_buffer->num_frames);
    if (frame == nullptr) {
        DEBUG("FAILED to wait_for_empty_frame on output_buffer {:s}[:d]", unique_name.c_str(),
              gpu_frame_id);
        return -1;
    }
    return 0;
}

cudaEvent_t cudaOutputData::execute_base(cudaPipelineState& pipestate,
                                         const std::vector<cudaEvent_t>& pre_events) {
    cudaEvent_t rtn = cudaCommand::execute_base(pipestate, pre_events);
    did_generate_output = (rtn != nullptr);
    return rtn;
}

cudaEvent_t cudaOutputData::execute(cudaPipelineState&,
                                    const std::vector<cudaEvent_t>& pre_events) {
    pre_execute();

    size_t output_len = output_buffer->frame_size;

    void* gpu_output_frame = device.get_gpu_memory_array(_gpu_mem, gpu_frame_id, output_len);
    int out_id = gpu_frame_id % output_buffer->num_frames;
    void* host_output_frame = (void*)output_buffer->frames[out_id];

    device.async_copy_gpu_to_host(host_output_frame, gpu_output_frame, output_len, cuda_stream_id,
                                  pre_events[cuda_stream_id], start_event, end_event);

    if (!in_buffer) {
        // Check for metadata attached to the GPU frame
        metadataContainer* meta = device.get_gpu_memory_array_metadata(_gpu_mem, gpu_frame_id);
        if (meta) {
            // Attach the metadata to the host buffer frame
            CHECK_ERROR_F(pthread_mutex_lock(&output_buffer->lock));
            if (output_buffer->metadata[out_id] == NULL) {
                DEBUG("Passing metadata from GPU array {:s}[{:d}] to output buffer {:s}[{:d}]",
                      _gpu_mem, gpu_frame_id, output_buffer->buffer_name, out_id);
                output_buffer->metadata[out_id] = meta;
                increment_metadata_ref_count(meta);
            }
            CHECK_ERROR_F(pthread_mutex_unlock(&output_buffer->lock));
        }
    }
    return end_event;
}

void cudaOutputData::finalize_frame() {
    cudaCommand::finalize_frame();

    int out_id = gpu_frame_id % output_buffer->num_frames;
    if (in_buffer) {
        int in_id = gpu_frame_id % in_buffer->num_frames;
        if (did_generate_output) {
            DEBUG("Passing metadata from input (host) buffer {:s}[{:d}] to output (host) buffer "
                  "{:s}[{:d}]",
                  in_buffer->buffer_name, in_id, output_buffer->buffer_name, out_id);
            pass_metadata(in_buffer, in_id, output_buffer, out_id);
        }
        mark_frame_empty(in_buffer, unique_name.c_str(), in_id);
    }

    if (!did_generate_output) {
        DEBUG("Did not generate output for input GPU frame {:d}", gpu_frame_id);
        return;
    }

    DEBUG("Generated CPU output {:s}[{:d}] for input GPU frame {:d}", output_buffer->buffer_name,
          out_id, gpu_frame_id);

    mark_frame_full(output_buffer, unique_name.c_str(), out_id);
}

std::string cudaOutputData::get_performance_metric_string() {
    double t = (double)get_last_gpu_execution_time();
    double transfer_speed = (double)output_buffer->frame_size / t * 1e-9;
    return fmt::format("Time: {:.3f} ms, Speed: {:.2f} GB/s ({:.2f} Gb/s)", t * 1e3, transfer_speed,
                       transfer_speed * 8);
}
