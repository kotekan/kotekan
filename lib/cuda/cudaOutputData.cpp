/**
 * @file
 * @brief CUDA command to copy data from the GPU to the host
 *  - cudaOutputData : public cudaCommand
 */

#include "cudaOutputData.hpp"

#include "chordMetadata.hpp"  // for chordMetadata
#include "cudaUtils.hpp"      // for CHECK_CUDA_ERROR
#include "cuda_runtime_api.h" // for cudaHostGetFlags, cudaHostRegister, cudaHostUnregister
#include "gpuCommand.hpp"     // for gpuCommandType
#include "kotekanLogging.hpp" // for DEBUG
#include "metadata.hpp"       // for metadataObject

#include "fmt.hpp" // for compile_string_to_view, format, format_string

#include <algorithm>   // for max
#include <memory>      // for allocator, shared_ptr
#include <stdint.h>    // for uint8_t
#include <sys/types.h> // for uint, size_t
#include <tuple>       // for tuple, make_tuple

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
            in_buffer->register_consumer(unique_name);
    } else
        in_buffer = nullptr;

    output_buffer = host_buffers.get_buffer(config.get<std::string>(unique_name, "out_buf"));
    if (instance_num == 0)
        output_buffer->register_producer(unique_name);

    if (output_buffer->frame_size) {
        uint flags;
        // only register the memory if it isn't already...
        if (cudaErrorInvalidValue
            == cudaHostGetFlags(&flags, output_buffer->frames[instance_num])) {
            CHECK_CUDA_ERROR(cudaHostRegister(output_buffer->frames[instance_num],
                                              output_buffer->frame_size, 0));
        }
    }

    if (output_buffer->frame_size == 0)
        _gpu_mem = "";
    else {
        _gpu_mem = config.get<std::string>(unique_name, "gpu_mem");
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem, true, true, false));
    }

    set_command_type(gpuCommandType::COPY_OUT);
    set_name("output: " + _gpu_mem);
}

cudaOutputData::~cudaOutputData() {
    if (output_buffer->frame_size) {
        uint flags;
        // only register the memory if it isn't already...
        if (cudaErrorInvalidValue
            == cudaHostGetFlags(&flags, output_buffer->frames[instance_num])) {
            CHECK_CUDA_ERROR(cudaHostUnregister(output_buffer->frames[instance_num]));
        }
    }
}

int cudaOutputData::wait_on_precondition() {
    // Wait for there to be space in the host output buffer
    uint8_t* frame =
        output_buffer->wait_for_empty_frame(unique_name, gpu_frame_id % output_buffer->num_frames);
    if (frame == nullptr) {
        DEBUG("FAILED to wait_for_empty_frame on output_buffer {:s}[:d]", unique_name.c_str(),
              gpu_frame_id);
        return -1;
    }
    return 0;
}

cudaEvent_t cudaOutputData::execute_base(cudaPipelineState& pipestate,
                                         const std::vector<cudaEvent_t>& pre_events) {
    bool should = should_execute(pipestate, pre_events);
    did_generate_output = should;
    DEBUG("gpu_frame_id {:d}: should_execute? {:s}", gpu_frame_id, (should ? "True" : "False"));
    if (!should)
        return nullptr;
    return execute(pipestate, pre_events);
}

cudaEvent_t cudaOutputData::execute(cudaPipelineState&,
                                    const std::vector<cudaEvent_t>& pre_events) {
    pre_execute();

    size_t output_len = output_buffer->frame_size;
    DEBUG("execute [{:d}]: output_len {:d}", gpu_frame_id, output_len);
    if (output_len) {
        void* gpu_output_frame =
            device.get_gpu_memory_array(_gpu_mem, gpu_frame_id, _gpu_buffer_depth, output_len);
        int out_id = gpu_frame_id % output_buffer->num_frames;
        void* host_output_frame = (void*)output_buffer->frames[out_id];

        device.async_copy_gpu_to_host(host_output_frame, gpu_output_frame, output_len,
                                      cuda_stream_id, pre_events[cuda_stream_id], &start_event,
                                      &end_event);

        if (!in_buffer) {
            // Check for metadata attached to the GPU frame
            std::shared_ptr<metadataObject> meta =
                device.get_gpu_memory_array_metadata(_gpu_mem, gpu_frame_id);
            if (meta) {
                // Attach the metadata to the host buffer frame
                bool passed = output_buffer->set_metadata(out_id, meta);
                if (passed)
                    DEBUG("Passing metadata from GPU array {:s}[{:d}] to output buffer {:s}[{:d}]",
                          _gpu_mem, gpu_frame_id, output_buffer->buffer_name, out_id);

                // TODO: actuall get the NDArray information that was set up by
                // the CPU commands
                auto chord = std::dynamic_pointer_cast<chordMetadata>(meta);
                if (chord) {
                    /* new style array description */
                    std::vector<ptrdiff_t> dimensions(chord->dim, chord->dim + chord->dims);
                    std::vector<kotekan::Symbol> dimnames(chord->dims);
                    for (size_t d = 0; d < dimnames.size(); ++d) {
                        dimnames.at(d) =
                            std::string(chord->dim_name[d],
                                        strnlen(chord->dim_name[d], sizeof(chord->dim_name[d])));
                    }

                    output_buffer->allocate_new_frame_desc(out_id, chord->type, chord->get_name(),
                                                           dimensions, dimnames);
                    /* test that things are consistent */
                    chord->check_frame_desc(output_buffer->get_frame_desc(out_id));
                }
            }
        }
    }
    return end_event;
}

void cudaOutputData::finalize_frame() {
    cudaCommand::finalize_frame();
    DEBUG("finalize_frame for gpu_frame_id {:d}", gpu_frame_id);
    int out_id = gpu_frame_id % output_buffer->num_frames;
    if (in_buffer) {
        int in_id = gpu_frame_id % in_buffer->num_frames;
        if (did_generate_output) {
            DEBUG("Passing metadata from input (host) buffer {:s}[{:d}] to {:s}[{:d}]",
                  in_buffer->buffer_name, in_id, output_buffer->buffer_name, out_id);
            in_buffer->pass_metadata(in_id, output_buffer, out_id);
        }
        in_buffer->mark_frame_empty(unique_name, in_id);
    }

    if (!did_generate_output) {
        DEBUG("Did not generate output for input GPU frame {:d}", gpu_frame_id);
        return;
    }

    DEBUG("Generated CPU output {:s}[{:d}] for input GPU frame {:d}", output_buffer->buffer_name,
          out_id, gpu_frame_id);

    output_buffer->mark_frame_full(unique_name, out_id);
}

std::string cudaOutputData::get_performance_metric_string() {
    double t = (double)get_last_gpu_execution_time();
    double transfer_speed = (double)output_buffer->frame_size / t * 1e-9;
    return fmt::format("Time: {:.3f} ms, Speed: {:.2f} GB/s ({:.2f} Gb/s)", t * 1e3, transfer_speed,
                       transfer_speed * 8);
}
