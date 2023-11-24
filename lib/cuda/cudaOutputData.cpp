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
                               bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "", "") {
    std::string in_buf_name = config.get_default<std::string>(unique_name, "in_buf", "");
    if (in_buf_name.size()) {
        in_buffer = host_buffers.get_buffer(in_buf_name);
        in_buffer->register_consumer(unique_name);
    } else
        in_buffer = nullptr;

    output_buffer = host_buffers.get_buffer(config.get<std::string>(unique_name, "out_buf"));
    output_buffer->register_producer(unique_name);

    if (output_buffer->frame_size)
        for (int i = 0; i < output_buffer->num_frames; i++) {
            uint flags;
            // only register the memory if it isn't already...
            if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, output_buffer->frames[i])) {
                CHECK_CUDA_ERROR(
                    cudaHostRegister(output_buffer->frames[i], output_buffer->frame_size, 0));
            }
        }

    if (output_buffer->frame_size == 0)
        _gpu_mem = "";
    else {
        _gpu_mem = config.get<std::string>(unique_name, "gpu_mem");
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem, true, true, false));
    }

    output_buffer_execute_id = 0;
    output_buffer_precondition_id = 0;
    output_buffer_id = 0;
    in_buffer_id = 0;

    did_generate_output.resize(_gpu_buffer_depth);

    set_command_type(gpuCommandType::COPY_OUT);

    kernel_command = "cudaOutputData: " + _gpu_mem;
    set_log_prefix(fmt::format("{:s} ({:30s})", unique_name, get_name()));
}

cudaOutputData::~cudaOutputData() {
    if (output_buffer->frame_size)
        for (int i = 0; i < output_buffer->num_frames; i++) {
            uint flags;
            // only register the memory if it isn't already...
            if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, output_buffer->frames[i])) {
                CHECK_CUDA_ERROR(cudaHostUnregister(output_buffer->frames[i]));
            }
        }
}

int cudaOutputData::wait_on_precondition(int) {
    // Wait for there to be space in the host output buffer
    uint8_t* frame =
        output_buffer->wait_for_empty_frame(unique_name, output_buffer_precondition_id);
    if (frame == nullptr) {
        DEBUG("FAILED to wait_for_empty_frame on output_buffer {:s}[:d]", unique_name.c_str(),
              output_buffer_precondition_id);
        return -1;
    }
    output_buffer_precondition_id = (output_buffer_precondition_id + 1) % output_buffer->num_frames;
    return 0;
}

cudaEvent_t cudaOutputData::execute_base(cudaPipelineState& pipestate,
                                         const std::vector<cudaEvent_t>& pre_events) {
    bool should = should_execute(pipestate, pre_events);
    did_generate_output[pipestate.gpu_frame_id] = should;
    if (!should)
        return nullptr;
    return execute(pipestate, pre_events);
}

cudaEvent_t cudaOutputData::execute(cudaPipelineState& pipestate,
                                    const std::vector<cudaEvent_t>& pre_events) {
    pre_execute(pipestate.gpu_frame_id);

    size_t output_len = output_buffer->frame_size;

    if (output_len) {
        void* gpu_output_frame = device.get_gpu_memory_array(_gpu_mem, pipestate.gpu_frame_id,
                                                             _gpu_buffer_depth, output_len);
        void* host_output_frame = (void*)output_buffer->frames[output_buffer_execute_id];

        device.async_copy_gpu_to_host(host_output_frame, gpu_output_frame, output_len,
                                      cuda_stream_id, pre_events[cuda_stream_id],
                                      start_events[pipestate.gpu_frame_id],
                                      end_events[pipestate.gpu_frame_id]);

        if (!in_buffer) {
            // Check for metadata attached to the GPU frame
            metadataContainer* meta =
                device.get_gpu_memory_array_metadata(_gpu_mem, pipestate.gpu_frame_id);
            if (meta) {
                // Attach the metadata to the host buffer frame
                bool passed = output_buffer->set_metadata(output_buffer_execute_id, meta);
                if (passed)
                    DEBUG("Passing metadata from GPU array {:s}[{:d}] to output buffer {:s}[{:d}]",
                          _gpu_mem, pipestate.gpu_frame_id, output_buffer->buffer_name,
                          output_buffer_execute_id);
            }
        }
    }

    output_buffer_execute_id = (output_buffer_execute_id + 1) % output_buffer->num_frames;
    return end_events[pipestate.gpu_frame_id];
}

void cudaOutputData::finalize_frame(int frame_id) {
    cudaCommand::finalize_frame(frame_id);

    if (in_buffer) {
        if (did_generate_output[frame_id]) {
            DEBUG("Passing metadata from input (host) buffer {:s}[{:d}] to {:s}[{:d}]",
                  in_buffer->buffer_name, in_buffer_id, output_buffer->buffer_name,
                  output_buffer_id);
            pass_metadata(in_buffer, in_buffer_id, output_buffer, output_buffer_id);
        }
        in_buffer->mark_frame_empty(unique_name, in_buffer_id);
        in_buffer_id = (in_buffer_id + 1) % in_buffer->num_frames;
    }

    if (!did_generate_output[frame_id]) {
        DEBUG("Did not generate output for input GPU frame {:d}", frame_id);
        return;
    }

    DEBUG("Generated CPU output {:s}[{:d}] for input GPU frame {:d}", output_buffer->buffer_name,
          output_buffer_id, frame_id);

    output_buffer->mark_frame_full(unique_name, output_buffer_id);
    output_buffer_id = (output_buffer_id + 1) % output_buffer->num_frames;
}

std::string cudaOutputData::get_performance_metric_string() {
    double t = (double)get_last_gpu_execution_time();
    double transfer_speed = (double)output_buffer->frame_size / t * 1e-9;
    return fmt::format("Time: {:.3f} ms, Speed: {:.2f} GB/s ({:.2f} Gb/s)", t * 1e3, transfer_speed,
                       transfer_speed * 8);
}
