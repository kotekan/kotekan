// Copyright (c) 2025 Kotekan Project
#include "cudaCopyNtoRingbuffer.hpp"
#include "chordMetadata.hpp"
#include "cudaUtils.hpp"
#include <cassert>
#include <stdexcept>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCopyNtoRingbuffer);

cudaCopyNtoRingbuffer::cudaCopyNtoRingbuffer(Config& config, const std::string& unique_name,
											 bufferContainer& host_buffers,
											 cudaDeviceInterface& device, int instance_num)
	: cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
				  "cudaCopyNtoRingbuffer", ""),
		output_cursor(0) {
	// Get the list of input buffer names (comma-separated)
	std::vector<std::string> in_buf_names = config.get<std::vector<std::string>>(unique_name, "in_bufs");
	if (in_buf_names.empty())
		throw std::runtime_error("cudaCopyNtoRingbuffer: in_bufs must be set and non-empty");

	total_input_size = 0;
	first_time = true;
	for (const auto& bufname : in_buf_names) {
		Buffer* buf = host_buffers.get_buffer(bufname);
		if (!buf)
			throw std::runtime_error("cudaCopyNtoRingbuffer: could not find buffer " + bufname);
		in_buffers.push_back(buf);
		total_input_size += buf->frame_size;
		if (instance_num == 0) {
			buf->register_consumer(unique_name);
			// Make sure all the host frames are registered with CUDA
			// TODO, check if this will always happen at the framework level.
			if (buf->frame_size) {
				for (int i = 0; i < buf->num_frames; ++i) {
					uint8_t* frame = buf->frames[i];
					uint flags;
					if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, frame)) {
						CHECK_CUDA_ERROR(cudaHostRegister(frame, buf->frame_size, 0));
					}
				}
			}
		}
	}

	_gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");

	signal_buffer = dynamic_cast<RingBuffer*>(
		host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "signal_buf")));
	assert(signal_buffer);
	if (instance_num == 0)
		signal_buffer->register_producer(unique_name);

	set_command_type(gpuCommandType::COPY_IN);
	gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, false, false, true));
}

cudaCopyNtoRingbuffer::~cudaCopyNtoRingbuffer() {
	for (size_t i = 0; i < in_buffers.size(); ++i) {
		Buffer* buf = in_buffers[i];
		if (buf && buf->frame_size) {
			uint flags;
			if (cudaSuccess == cudaHostGetFlags(&flags, buf->frames[instance_num])) {
				CHECK_CUDA_ERROR(cudaHostUnregister(buf->frames[instance_num]));
			}
		}
	}
}

int cudaCopyNtoRingbuffer::wait_on_precondition() {
	// Wait for all input buffers to have data
	for (size_t i = 0; i < in_buffers.size(); ++i) {
		Buffer* buf = in_buffers[i];
		DEBUG("Waiting for input data frame {} from buffer {}", gpu_frame_id, buf->buffer_name);
		uint8_t* frame = buf->wait_for_full_frame(unique_name, gpu_frame_id % buf->num_frames);
		if (!frame)
			return -1;
		DEBUG("Input data frame {} is now available from buffer {}", gpu_frame_id, buf->buffer_name);
	}

	// Wait for ringbuffer space
	DEBUG("Waiting for ringbuffer space for writing to become available ({} bytes)", total_input_size);
	std::optional<size_t> val = signal_buffer->wait_for_writable(unique_name, instance_num, total_input_size);
	DEBUG("Ringbuffer space for writing is now available");
	if (!val.has_value())
		return -1;
	output_cursor = val.value();
	return 0;
}

cudaEvent_t cudaCopyNtoRingbuffer::execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>& pre_events) {
	(void)pipestate;
	pre_execute();

	void* rb_memory = device.get_gpu_memory(_gpu_mem_output, signal_buffer->size);

	// Compute total size and wrapping
	size_t start = output_cursor % signal_buffer->size;

	record_start_event();

	// Merge all input buffers into the ringbuffer sequentially
	size_t offset = 0;
	std::vector<std::shared_ptr<chordMetadata>> metas;
	for (size_t i = 0; i < in_buffers.size(); ++i) {
		Buffer* buf = in_buffers[i];
		int buf_index = gpu_frame_id % buf->num_frames;
		void* host_memory_frame = (void*)buf->frames[buf_index];
		size_t sz = buf->frame_size;

		// Copy to ringbuffer, handle wrapping if needed
		if (start + offset + sz <= (size_t)signal_buffer->size) {
			device.async_copy_host_to_gpu((char*)rb_memory + start + offset, host_memory_frame, sz,
										  cuda_stream_id, pre_events[cuda_stream_id], nullptr, nullptr);
		} else {
			size_t first = signal_buffer->size - (start + offset);
			size_t second = sz - first;
			device.async_copy_host_to_gpu((char*)rb_memory + start + offset, host_memory_frame, first,
										  cuda_stream_id, pre_events[cuda_stream_id], nullptr, nullptr);
			device.async_copy_host_to_gpu(rb_memory, (char*)host_memory_frame + first, second,
										  cuda_stream_id, nullptr, nullptr, nullptr);
		}

		offset += sz;
	}

	// We only need to set the metadata on the ring buffer once. 
	if (first_time && instance_num == 0) {
		// Allocate a new metadata object on the signal buffer
		signal_buffer->allocate_new_metadata_object(0);
		auto meta_ring = std::dynamic_pointer_cast<chordMetadata>(signal_buffer->metadata[0]);
		if (!meta_ring)
			throw std::runtime_error("cudaCopyNtoRingbuffer: could not get chordMetadata from signal buffer");

		// Merge metadata from all input buffers
		// NB This is highly specific to CHIME.
		for (uint i = 0; i < in_buffers.size(); ++i) {
			auto meta_in = std::dynamic_pointer_cast<chordMetadata>(in_buffers[i]->get_metadata(gpu_frame_id % in_buffers[i]->num_frames));
			if (!meta_in)
				throw std::runtime_error(
					"cudaCopyNtoRingbuffer: input buffer has no chordMetadata");
			if (i == 0) {
				// Set the shape of the array
				meta_ring->dims = 4;
				meta_ring->set_array_dimension(0, _gpu_buffer_depth, "T_frame");
				meta_ring->set_array_dimension(1, in_buffers.size(), "F");
				meta_ring->set_array_dimension(2, meta_in->dim[0], "T_sample");
				meta_ring->set_array_dimension(3, meta_in->dim[1], "E");

				// Set the data type
				meta_ring->type = kotekan::int4x2_swapped_withoffset;

				// Set the FPGA seq number of the first sample
				// NB that interpretation of time is a little confused with the array layout
				// above.  So the transpose kernel will need specal attention to calculating 
				// the actualy time.  It does not follow the usual T_actual pattern.
				meta_ring->sample0_offset = meta_in->sample0_offset;
			}
			meta_ring->coarse_freq[i] = meta_in->coarse_freq[0];
			assert(meta_ring->sample0_offset == meta_in->sample0_offset);
		}
		// Debug log the merged metadata with the data set above
		INFO("cudaCopyNtoRingbuffer: Merged metadata frequency list: {}",
			 fmt::join(std::vector<int>(meta_ring->coarse_freq,
										meta_ring->coarse_freq + in_buffers.size()),
					   ", "));
		first_time = false;	
	}

	return record_end_event();
}

void cudaCopyNtoRingbuffer::finalize_frame() {
	DEBUG("finalize_frame() for frame {}, releasing metadata on GPU output buffer {}",
		  gpu_frame_id, _gpu_mem_output);
	for (size_t i = 0; i < in_buffers.size(); ++i) {
		Buffer* buf = in_buffers[i];
		buf->mark_frame_empty(unique_name, gpu_frame_id % buf->num_frames);
	}
	signal_buffer->finish_write(unique_name, instance_num, total_input_size);
	cudaCommand::finalize_frame();
}
