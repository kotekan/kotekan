#include "cudaCopyToRingbuffer.hpp"

#include "cudaUtils.hpp"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCopyToRingbuffer);

cudaCopyToRingbuffer::cudaCopyToRingbuffer(Config& config, const std::string& unique_name,
                                           bufferContainer& host_buffers,
                                           cudaDeviceInterface& device, int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaCopyToRingbuffer", ""),
    output_cursor(0) {
    _input_size = config.get<size_t>(unique_name, "input_size");
    _ring_buffer_size = config.get<size_t>(unique_name, "ring_buffer_size");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    _gpu_mem_input = config.get_default<std::string>(unique_name, "gpu_mem_input", "");
    if (_gpu_mem_input.size() == 0) {
        std::string bufname = config.get<std::string>(unique_name, "in_buf");
        in_buffer = host_buffers.get_buffer(bufname);
        if (!in_buffer)
            throw std::runtime_error("In cudaCopyToRingbuffer " + unique_name
                                     + ", must set either gpu_mem_input or in_buf");
        DEBUG("Initializing cudaCopyToRingbuffer: from host buffer \"{:s}\" to GPU memory "
              "\"{:s}\", chunk size {:d}, ring buffer size {:d}",
              bufname, _gpu_mem_output, _input_size, _ring_buffer_size);
        if (instance_num == 0)
            in_buffer->register_consumer(unique_name);

        if (in_buffer->frame_size) {
            uint flags;
            // only register the memory if it isn't already...
            if (cudaErrorInvalidValue
                == cudaHostGetFlags(&flags, in_buffer->frames[instance_num])) {
                CHECK_CUDA_ERROR(
                    cudaHostRegister(in_buffer->frames[instance_num], in_buffer->frame_size, 0));
            }
        }
    } else {
        in_buffer = nullptr;
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
        DEBUG("Initializing cudaCopyToRingbuffer: from GPU memory \"{:s}\" to GPU memory \"{:s}\", "
              "chunk size {:d}, ring buffer size {:d}",
              _gpu_mem_input, _gpu_mem_output, _input_size, _ring_buffer_size);
    }
    //_input_columns_field = config.get_default<std::string>(unique_name, "input_columns_field",
    //"");
    signal_buffer = dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "signal_buf")));
    assert(signal_buffer);
    if (instance_num == 0)
        signal_buffer->register_producer(unique_name);

    set_command_type(gpuCommandType::COPY_IN);

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, false, false, true));
}

cudaCopyToRingbuffer::~cudaCopyToRingbuffer() {
    if (in_buffer && in_buffer->frame_size) {
        uint flags;
        // only unregister if it's already been registered
        if (cudaSuccess == cudaHostGetFlags(&flags, in_buffer->frames[instance_num])) {
            CHECK_CUDA_ERROR(cudaHostUnregister(in_buffer->frames[instance_num]));
        }
    }
}

int cudaCopyToRingbuffer::wait_on_precondition() {
    DEBUG("Waiting for ringbuffer space for writing to become available ({:d} bytes)", _input_size);
    std::optional<size_t> val = signal_buffer->wait_for_writable(unique_name, _input_size);
    DEBUG("Ringbuffer space for writing is now available");
    if (!val.has_value())
        return -1;
    output_cursor = val.value();

    if (in_buffer) {
        // Wait for there to be data in the input (host-side) buffer.
        DEBUG("Waiting for input data frame {:d}", gpu_frame_id);
        uint8_t* frame =
            in_buffer->wait_for_full_frame(unique_name, gpu_frame_id % in_buffer->num_frames);
        if (frame == nullptr)
            return -1;
        DEBUG("Input data frame {:d} is now available", gpu_frame_id);
    }
    return 0;
}

cudaEvent_t cudaCopyToRingbuffer::execute(cudaPipelineState& pipestate,
                                          const std::vector<cudaEvent_t>& pre_events) {
    (void)pipestate;
    pre_execute();

    void* rb_memory = device.get_gpu_memory(_gpu_mem_output, _ring_buffer_size);

    size_t ncopy = _input_size;
    size_t nwrap = 0;
    if (output_cursor + _input_size > _ring_buffer_size) {
        ncopy = _ring_buffer_size - output_cursor;
        nwrap = _input_size - ncopy;
    }

    record_start_event();
    std::shared_ptr<metadataObject> meta;
    if (!in_buffer) {
        void* input_memory = device.get_gpu_memory_array(_gpu_mem_input, gpu_frame_id,
                                                         _gpu_buffer_depth, _input_size);
        DEBUG("Copying from GPU frame array into ringbuffer: {:d} to copy + {:d} wrapping around",
              ncopy, nwrap);
        CHECK_CUDA_ERROR(cudaMemcpyAsync((char*)rb_memory + output_cursor, input_memory, ncopy,
                                         cudaMemcpyDeviceToDevice,
                                         device.getStream(cuda_stream_id)));
        if (nwrap)
            CHECK_CUDA_ERROR(cudaMemcpyAsync(rb_memory, (char*)input_memory + ncopy, nwrap,
                                             cudaMemcpyDeviceToDevice,
                                             device.getStream(cuda_stream_id)));
        // Copy (reference to) metadata also
        meta = device.get_gpu_memory_array_metadata(_gpu_mem_input, gpu_frame_id);
    } else {
        int buf_index = gpu_frame_id % in_buffer->num_frames;
        void* host_memory_frame = (void*)in_buffer->frames[buf_index];

        DEBUG("Copying from host memory into ringbuffer: {:d} to copy + {:d} wrapping around",
              ncopy, nwrap);
        device.async_copy_host_to_gpu((char*)rb_memory + output_cursor, host_memory_frame, ncopy,
                                      cuda_stream_id, pre_events[cuda_stream_id], nullptr, nullptr);
        if (nwrap)
            device.async_copy_host_to_gpu(rb_memory, (char*)host_memory_frame + ncopy, nwrap,
                                          cuda_stream_id, pre_events[cuda_stream_id], nullptr,
                                          nullptr);

        // Copy (reference to) metadata also
        meta = in_buffer->metadata[buf_index];
        DEBUG("Metadata from input buffer frame: {:p}", static_cast<void*>(meta.get()));
    }
    if (meta) {
        DEBUG("Copying metadata for frame {:d} to GPU array {:s}", gpu_frame_id, _gpu_mem_output);
        device.claim_gpu_memory_array_metadata(_gpu_mem_output, 0, meta);
    }

    // FIXME -- signal *now*, when we have *queued* the cuda work?  Or in finalize_frame, when it
    // has finished?? if we do it here, probably need a syncInput after the cudaInput that is
    // waiting on this buffer.
    // signal_buffer->wrote(unique_name, _input_size);

    return record_end_event();
}

void cudaCopyToRingbuffer::finalize_frame() {
    cudaCommand::finalize_frame();
    // Release reference to metadata, if we grabbed it
    DEBUG("finalize_frame() for frame {:d}, releasing metadata on GPU output buffer {:s}",
          gpu_frame_id, _gpu_mem_output);
    // device.release_gpu_memory_array_metadata(_gpu_mem_output, 0);
    if (in_buffer)
        in_buffer->mark_frame_empty(unique_name, gpu_frame_id % in_buffer->num_frames);
    // At this point we know the Cuda copy completed, but do we *really* need that to be the case??
    signal_buffer->finish_write(unique_name, _input_size);
}
