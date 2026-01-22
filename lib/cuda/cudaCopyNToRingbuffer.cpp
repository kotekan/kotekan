// Copyright (c) 2025 Kotekan Project
#include "cudaCopyNToRingbuffer.hpp"

#include "DataType.hpp"       // for DataType
#include "chordMetadata.hpp"  // for chordMetadata
#include "cudaUtils.hpp"      // for CHECK_CUDA_ERROR
#include "cuda_runtime_api.h" // for cudaHostGetFlags, cudaHostUnregister
#include "gpuCommand.hpp"     // for gpuCommandType
#include "kotekanLogging.hpp" // for DEBUG, ERROR, INFO

#include "fmt.hpp" // for compile_string_to_view, join

#include <algorithm>   // for max
#include <cassert>     // for assert
#include <memory>      // for shared_ptr, __shared_ptr_access, allocator, dynamic_pointe...
#include <optional>    // for optional
#include <stdexcept>   // for runtime_error
#include <stdint.h>    // for int64_t, uint8_t
#include <sys/types.h> // for size_t, uint
#include <tuple>       // for tuple, make_tuple
#include <vector>      // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCopyNToRingbuffer);

cudaCopyNToRingbuffer::cudaCopyNToRingbuffer(Config& config, const std::string& unique_name,
                                             bufferContainer& host_buffers,
                                             cudaDeviceInterface& device, int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaCopyNToRingbuffer", ""),
    output_cursor(0) {
    // Get the list of input buffer names (comma-separated)
    std::vector<std::string> in_buf_names =
        config.get<std::vector<std::string>>(unique_name, "in_bufs");
    if (in_buf_names.empty())
        throw std::runtime_error("cudaCopyNToRingbuffer: in_bufs must be set and non-empty");

    total_input_size = 0;
    first_time = true;
    for (const auto& bufname : in_buf_names) {
        Buffer* buf = host_buffers.get_buffer(bufname);
        if (!buf)
            throw std::runtime_error("cudaCopyNToRingbuffer: could not find buffer " + bufname);
        in_buffers.push_back(buf);
        total_input_size += buf->frame_size;
        if (instance_num == 0) {
            buf->register_consumer(unique_name);
        }
    }

    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");

    signal_buffer = dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "signal_buf")));
    assert(signal_buffer && "Buffer given by signal_buf is null or not a RingBuffer");
    if (instance_num == 0)
        signal_buffer->register_producer(unique_name);

    set_command_type(gpuCommandType::COPY_IN);
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, false, false, true));
}

cudaCopyNToRingbuffer::~cudaCopyNToRingbuffer() {
    for (auto buf : in_buffers) {
        assert(buf);
        if (buf->frame_size) {
            uint flags;
            if (cudaSuccess == cudaHostGetFlags(&flags, buf->frames[instance_num])) {
                CHECK_CUDA_ERROR(cudaHostUnregister(buf->frames[instance_num]));
            }
        }
    }
}

int cudaCopyNToRingbuffer::wait_on_precondition() {
    // Wait for all input buffers to have data
    for (auto buf : in_buffers) {
        assert(buf);
        DEBUG("Waiting for input data frame {} from buffer {}", gpu_frame_id, buf->buffer_name);
        uint8_t* frame = buf->wait_for_full_frame(unique_name, gpu_frame_id % buf->num_frames);
        if (!frame)
            return -1;
        DEBUG("Input data frame {} is now available from buffer {}", gpu_frame_id,
              buf->buffer_name);
    }

    // Wait for ringbuffer space
    DEBUG("Waiting for ringbuffer space for writing to become available ({} bytes)",
          total_input_size);
    std::optional<size_t> val =
        signal_buffer->wait_for_writable(unique_name, instance_num, total_input_size);
    DEBUG("Ringbuffer space for writing is now available");
    if (!val.has_value())
        return -1;
    output_cursor = val.value();
    return 0;
}

cudaEvent_t cudaCopyNToRingbuffer::execute(cudaPipelineState& /*pipestate*/,
                                           const std::vector<cudaEvent_t>& pre_events) {
    pre_execute();

    void* rb_memory = device.get_gpu_memory(_gpu_mem_output, signal_buffer->size);

    // Compute total size and wrapping
    size_t start = output_cursor % signal_buffer->size;

    record_start_event();

    // Merge all input buffers into the ringbuffer sequentially
    size_t offset = 0;
    for (auto buf : in_buffers) {
        assert(buf);
        int buf_index = gpu_frame_id % buf->num_frames;
        void* host_memory_frame = static_cast<void*>(buf->frames[buf_index]);
        size_t sz = buf->frame_size;

        // Copy to ringbuffer, handle wrapping if needed
        if (start + offset + sz <= (size_t)signal_buffer->size) {
            device.async_copy_host_to_gpu((char*)rb_memory + start + offset, host_memory_frame, sz,
                                          cuda_stream_id, pre_events[cuda_stream_id], nullptr,
                                          nullptr);
        } else {
            size_t first = signal_buffer->size - (start + offset);
            size_t second = sz - first;
            device.async_copy_host_to_gpu((char*)rb_memory + start + offset, host_memory_frame,
                                          first, cuda_stream_id, pre_events[cuda_stream_id],
                                          nullptr, nullptr);
            device.async_copy_host_to_gpu(rb_memory, (char*)host_memory_frame + first, second,
                                          cuda_stream_id, nullptr, nullptr, nullptr);
        }

        offset += sz;
    }

    // We only need to set the metadata on the ring buffer once.
    // TODO we should do a metadata validation for every input buffer to double check
    // we are not mixing incompatible frames.
    if (first_time && instance_num == 0) {
        // Get the first input metadata
        const auto meta_in0 = std::dynamic_pointer_cast<chordMetadata>(
            in_buffers.at(0)->get_metadata(gpu_frame_id % in_buffers.at(0)->num_frames));
        if (!meta_in0)
            throw std::runtime_error("cudaCopyNToRingbuffer: input buffer has no chordMetadata");

        // Copy metadata (because we modify it)
        auto meta_ring = std::make_shared<chordMetadata>();
        meta_ring->deepCopy(meta_in0);

        // Pull most of the metadata from the first input buffer.
        // TODO: Check metadata matches on all subsequent buffers.

        // Set the quantity
        meta_ring->set_name("E");

        // Set the shape of the array
        meta_ring->dims = 4;
        meta_ring->set_array_dimension(0, _gpu_buffer_depth, "Thi16384");
        meta_ring->set_array_dimension(1, in_buffers.size(), "F");
        assert(meta_in0->dim[0] == 16384);
        meta_ring->set_array_dimension(2, meta_in0->dim[0], "Tlo16384");
        meta_ring->set_array_dimension(3, meta_in0->dim[1], "E");
        meta_ring->set_strides_simple();

        // Set the data type
        meta_ring->type = kotekan::int4x2_swapped_withoffset;

        // Set the FPGA seq number of the first sample
        assert(output_cursor == 0);
        meta_ring->set_fpga_seq_num(meta_in0->get_fpga_seq_num());

        // Merge metadata from all input buffers
        // NB This is highly specific to CHIME.
        std::vector<int> coarse_freq(in_buffers.size());
        std::vector<int> freq_upchan_factor(in_buffers.size());
        std::vector<int> freq_upchan_index(in_buffers.size());
        for (size_t i = 0; i < in_buffers.size(); ++i) {
            auto meta_in = std::dynamic_pointer_cast<chordMetadata>(
                in_buffers.at(i)->get_metadata(gpu_frame_id % in_buffers.at(i)->num_frames));
            if (!meta_in)
                throw std::runtime_error(
                    "cudaCopyNToRingbuffer: input buffer has no chordMetadata");
            // Set the frequency for each of the input buffers
            coarse_freq.at(i) = meta_in->get_coarse_freq().at(0);
            freq_upchan_factor.at(i) = meta_in->get_freq_upchan_factor().at(0);
            freq_upchan_index.at(i) = meta_in->get_freq_upchan_index().at(0);
            // Check that the seq_num matches for all input buffers
            assert(meta_ring->get_fpga_seq_num() == meta_in->get_fpga_seq_num());
        }
        meta_ring->set_coarse_freq(coarse_freq);
        meta_ring->set_freq_upchan_factor(freq_upchan_factor);
        meta_ring->set_freq_upchan_index(freq_upchan_index);
        signal_buffer->set_metadata(0, meta_ring);

        // Debug log the merged metadata with the data set above
        INFO("cudaCopyNToRingbuffer: Merged metadata frequency list: {}",
             fmt::join(meta_ring->get_coarse_freq(), ", "));
        first_time = false;
    }

    // Check that metadata currently in the ring matches what we get from the frames
    auto meta_ring = std::dynamic_pointer_cast<chordMetadata>(signal_buffer->metadata[0]);
    assert(meta_ring); // By construction above, this should always exist.
    const std::vector<int> coarse_freq = meta_ring->get_coarse_freq();
    const std::vector<int> freq_upchan_factor = meta_ring->get_freq_upchan_factor();
    const std::vector<int> freq_upchan_index = meta_ring->get_freq_upchan_index();
    for (size_t i = 0; i < in_buffers.size(); ++i) {
        auto meta_in = std::dynamic_pointer_cast<chordMetadata>(
            in_buffers.at(i)->get_metadata(gpu_frame_id % in_buffers.at(i)->num_frames));
        if (!meta_in)
            throw std::runtime_error("cudaCopyNToRingbuffer: input buffer has no chordMetadata");
        // Check that the frequencies match
        if (coarse_freq.at(i) != meta_in->get_coarse_freq().at(0)) {
            ERROR("cudaCopyNToRingbuffer: Mismatch in frequency for input buffer {}: "
                  "metadata has {}, frame has {}",
                  in_buffers.at(i)->buffer_name, coarse_freq.at(i),
                  meta_in->get_coarse_freq().at(0));
            throw std::runtime_error("cudaCopyNToRingbuffer: metadata frequency mismatch");
        }
        if (coarse_freq.at(i) != meta_in->get_freq_upchan_factor().at(0)) {
            ERROR("cudaCopyNToRingbuffer: Mismatch in frequency for input buffer {}: "
                  "metadata has {}, frame has {}",
                  in_buffers.at(i)->buffer_name, freq_upchan_factor.at(i),
                  meta_in->get_freq_upchan_factor().at(0));
            throw std::runtime_error(
                "cudaCopyNToRingbuffer: metadata upchannelization factor mismatch");
        }
        if (freq_upchan_index.at(i) != meta_in->get_freq_upchan_index().at(0)) {
            ERROR("cudaCopyNToRingbuffer: Mismatch in frequency for input buffer {}: "
                  "metadata has {}, frame has {}",
                  in_buffers.at(i)->buffer_name, freq_upchan_index.at(i),
                  meta_in->get_freq_upchan_index().at(0));
            throw std::runtime_error(
                "cudaCopyNToRingbuffer: metadata upchannelization index mismatch");
        }
        // Check that the fpga_seq_num + the output_cursor matches the input frame fpga_seq_num
        // This ensures that the time in the ring buffer metadata matches the data we just copied
        if (meta_ring->get_fpga_seq_num()
                + (int64_t)output_cursor / (meta_ring->dim[1] * meta_ring->dim[3])
            != meta_in->get_fpga_seq_num()) {
            ERROR("cudaCopyNToRingbuffer: Mismatch in fpga_seq_num for input buffer {}: "
                  "metadata has {}, frame has {} (output_cursor {})",
                  in_buffers.at(i)->buffer_name, meta_ring->get_fpga_seq_num(),
                  meta_in->get_fpga_seq_num(), output_cursor);
            throw std::runtime_error("cudaCopyNToRingbuffer: metadata time code mismatch");
        }
    }

    return record_end_event();
}

void cudaCopyNToRingbuffer::finalize_frame() {
    DEBUG("finalize_frame() for frame {}, releasing metadata on GPU output buffer {}", gpu_frame_id,
          _gpu_mem_output);
    for (auto buf : in_buffers) {
        assert(buf);
        buf->mark_frame_empty(unique_name, gpu_frame_id % buf->num_frames);
    }
    signal_buffer->finish_write(unique_name, instance_num, total_input_size);
    cudaCommand::finalize_frame();
}
