#include "cudaCopyFromRingbuffer.hpp"

#include <assert.h>            // for assert
#include <stdint.h>            // for uint8_t
#include <string.h>            // for strnlen
#include <sys/types.h>         // for uint
#include <cstddef>             // for ptrdiff_t
#include <memory>              // for shared_ptr, __shared_ptr_access, dynamic_pointer_cast, mak...
#include <optional>            // for optional
#include <stdexcept>           // for runtime_error
#include <tuple>               // for tuple, make_tuple

#include "NDArray.hpp"         // for GenericNDArray
#include "Symbol.hpp"          // for Symbol
#include "chordMetadata.hpp"   // for chordMetadata
#include "cudaUtils.hpp"       // for CHECK_CUDA_ERROR
#include "cuda_runtime_api.h"  // for cudaHostGetFlags, cudaMemcpyAsync, cudaHostRegister, cudaH...
#include "gpuCommand.hpp"      // for gpuCommandType
#include "kotekanLogging.hpp"  // for DEBUG
#include "fmt.hpp"             // for compile_string_to_view

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCopyFromRingbuffer);

cudaCopyFromRingbuffer::cudaCopyFromRingbuffer(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               cudaDeviceInterface& device, int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaCopyFromRingbuffer", ""),
    input_cursor(0), initial_fpga_seq_num(-1) {
    _output_size = config.get<size_t>(unique_name, "output_size");
    _ring_buffer_size = config.get<size_t>(unique_name, "ring_buffer_size");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get_default<std::string>(unique_name, "gpu_mem_output", "");
    if (_gpu_mem_output.size() == 0) {
        // We're reading from GPU ringbuffer to host memory
        std::string bufname = config.get<std::string>(unique_name, "out_buf");
        out_buffer = host_buffers.get_buffer(bufname);
        if (!out_buffer)
            throw std::runtime_error("In cudaCopyFromRingbuffer " + unique_name
                                     + ", must set either gpu_mem_output or out_buf");
        DEBUG("Initializing cudaCopyFromRingbuffer: from GPU memory \"{:s}\" to host buffer "
              "\"{:s}\", chunk size {:d}, ring buffer size {:d}",
              _gpu_mem_input, bufname, _output_size, _ring_buffer_size);
        if (instance_num == 0)
            out_buffer->register_producer(unique_name);

        if (out_buffer->frame_size) {
            uint flags;
            // only register the memory if it isn't already...
            if (cudaErrorInvalidValue
                == cudaHostGetFlags(&flags, out_buffer->frames[instance_num])) {
                CHECK_CUDA_ERROR(
                    cudaHostRegister(out_buffer->frames[instance_num], out_buffer->frame_size, 0));
            }
        }
    } else {
        out_buffer = nullptr;
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));
        DEBUG("Initializing cudaCopyFromRingbuffer: from GPU ringbuffer memory \"{:s}\" to GPU "
              "memory \"{:s}\", "
              "chunk size {:d}, ring buffer size {:d}",
              _gpu_mem_input, _gpu_mem_output, _output_size, _ring_buffer_size);
    }

    signal_buffer = dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "host_signal")));
    if (instance_num == 0)
        signal_buffer->register_consumer(unique_name);

    set_command_type(gpuCommandType::COPY_OUT);

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
}

cudaCopyFromRingbuffer::~cudaCopyFromRingbuffer() {
    if (out_buffer && out_buffer->frame_size) {
        uint flags;
        // only unregister if it's already been registered
        if (cudaSuccess == cudaHostGetFlags(&flags, out_buffer->frames[instance_num])) {
            CHECK_CUDA_ERROR(cudaHostUnregister(out_buffer->frames[instance_num]));
        }
    }
}

int cudaCopyFromRingbuffer::wait_on_precondition() {
    // Wait for there to be data available in the ringbuffer.
    DEBUG("Waiting for ringbuffer data for frame {:d}...", gpu_frame_id);
    // signal_buffer->print_full_status();
    std::optional<size_t> val =
        signal_buffer->wait_and_claim_readable(unique_name, instance_num, _output_size);
    DEBUG("Finished waiting for data frame {:d}.", gpu_frame_id);
    // signal_buffer->print_full_status();
    if (!val.has_value()) {
        DEBUG("Got no value when waiting for ringbuffer data; quitting");
        return -1;
    }
    input_cursor = val.value();

    if (out_buffer) {
        // Wait for there to be room in the output (host-side) buffer.
        uint8_t* frame =
            out_buffer->wait_for_empty_frame(unique_name, gpu_frame_id % out_buffer->num_frames);
        if (frame == nullptr) {
            DEBUG("FAILED to wait_for_empty_frame on output_buffer {:s}[:d]", unique_name.c_str(),
                  gpu_frame_id);
            return -1;
        }
    }
    return 0;
}

cudaEvent_t cudaCopyFromRingbuffer::execute(cudaPipelineState& pipestate,
                                            const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    (void)pipestate;
    pre_execute();

    void* rb_memory = device.get_gpu_memory(_gpu_mem_input, _ring_buffer_size);

    auto in_meta = std::dynamic_pointer_cast<chordMetadata>(signal_buffer->get_metadata(0));
    assert(in_meta);
    // Copy metadata (because we modify it)
    auto out_meta = std::make_shared<chordMetadata>();
    out_meta->deepCopy(in_meta);

    assert(input_cursor % in_meta->sample_bytes() == 0);
    if (initial_fpga_seq_num == -1) { // first time
        if (instance_num == 0) {      // we handle frame 0 of the buffer depth
            assert(input_cursor == 0);
            initial_fpga_seq_num = in_meta->get_fpga_seq_num();
        } else { // handle one of the later frames, frame 0 handler has set metadata
            initial_fpga_seq_num = in_meta->get_fpga_seq_num();
        }
    } else { // not first time
        assert(in_meta->get_fpga_seq_num() == initial_fpga_seq_num);
    }
    out_meta->set_fpga_seq_num(in_meta->get_fpga_seq_num()
                               + in_meta->get_time_downsampling_fpga()
                                     * (input_cursor / in_meta->sample_bytes()));
    assert(input_cursor % in_meta->sample_bytes() == 0);
    assert(out_meta->dims > 0);
    assert(out_buffer->frame_size % out_meta->sample_bytes() == 0);
    out_meta->dim[0] = out_buffer->frame_size / out_meta->sample_bytes();

    size_t start = input_cursor % _ring_buffer_size;
    size_t ncopy = _output_size;
    size_t nwrap = 0;
    if (start + _output_size > _ring_buffer_size) {
        ncopy = _ring_buffer_size - start;
        nwrap = _output_size - ncopy;
    }

    record_start_event();

    if (out_buffer) {
        int out_id = gpu_frame_id % out_buffer->num_frames;
        void* host_output_frame = (void*)out_buffer->frames[out_id];
        assert(out_buffer->is_frame_empty(out_id) && "waited for in precondition");

        device.async_copy_gpu_to_host(host_output_frame, (char*)rb_memory + start, ncopy,
                                      cuda_stream_id, pre_events[cuda_stream_id], nullptr, nullptr);
        if (nwrap)
            device.async_copy_gpu_to_host((char*)host_output_frame + ncopy, rb_memory, nwrap,
                                          cuda_stream_id, nullptr, nullptr, nullptr);

        out_buffer->set_metadata(out_id, out_meta);
        /* new style array description */
        // difficult to move to constructor since it depends on frame_desc in the
        // signal_buffer which may not be set at contructor time
        std::vector<std::ptrdiff_t> extents(out_meta->dim, out_meta->dim + out_meta->dims);
        std::vector<kotekan::Symbol> dimnames;
        for (int d = 0; d < out_meta->dims; ++d)
            dimnames.push_back(
                std::string(out_meta->dim_name[d],
                            strnlen(out_meta->dim_name[d], sizeof(out_meta->dim_name[d]))));
        out_buffer->ensure_frame_desc(kotekan::GenericNDArray::describe(
            out_meta->type, out_meta->get_name(), extents, dimnames));
        /* test that things are consistent */
        out_meta->check_frame_desc(out_buffer->get_frame_desc<kotekan::GenericNDArray>());

    } else {
        int out_id = gpu_frame_id % _gpu_buffer_depth;
        void* output_memory = device.get_gpu_memory_array(_gpu_mem_output, gpu_frame_id,
                                                          _gpu_buffer_depth, _output_size);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(output_memory, (char*)rb_memory + start, ncopy,
                                         cudaMemcpyDeviceToDevice,
                                         device.getStream(cuda_stream_id)));
        if (nwrap)
            CHECK_CUDA_ERROR(cudaMemcpyAsync((char*)output_memory + ncopy, rb_memory, nwrap,
                                             cudaMemcpyDeviceToDevice,
                                             device.getStream(cuda_stream_id)));

        device.claim_gpu_memory_array_metadata(_gpu_mem_output, out_id, out_meta);
    }
    return record_end_event();
}

void cudaCopyFromRingbuffer::finalize_frame() {
    cudaCommand::finalize_frame();
    DEBUG("About to finalize frame {:d}", gpu_frame_id);
    // signal_buffer->print_full_status();
    signal_buffer->finish_read(unique_name, instance_num, _output_size);
    DEBUG("After finalizing frame {:d}", gpu_frame_id);
    // signal_buffer->print_full_status();
    if (out_buffer) {
        int out_id = gpu_frame_id % out_buffer->num_frames;
        out_buffer->mark_frame_full(unique_name, out_id);
    }
}
