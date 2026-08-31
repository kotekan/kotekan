#include "cudaCopyFromRingbuffer.hpp"

#include "NDArray.hpp"        // for GenericNDArray
#include "Symbol.hpp"         // for Symbol
#include "chordMetadata.hpp"  // for chordMetadata
#include "cudaUtils.hpp"      // for CHECK_CUDA_ERROR
#include "cuda_runtime_api.h" // for cudaHostGetFlags, cudaMemcpyAsync, cudaHostRegister, cudaH...
#include "gpuCommand.hpp"     // for gpuCommandType
#include "kotekanLogging.hpp" // for DEBUG

#include "fmt.hpp" // for compile_string_to_view

#include <assert.h>    // for assert
#include <cstddef>     // for ptrdiff_t
#include <memory>      // for shared_ptr, __shared_ptr_access, dynamic_pointer_cast, mak...
#include <optional>    // for optional
#include <stdexcept>   // for runtime_error
#include <stdint.h>    // for uint8_t
#include <string.h>    // for strnlen
#include <sys/types.h> // for uint
#include <tuple>       // for tuple, make_tuple

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
    _expect_quantity = config.get_default<std::string>(unique_name, "expect_quantity_name", "");
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

        register_host_buffer(out_buffer);
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
    if (out_buffer)
        unregister_host_buffer(out_buffer);
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
        std::vector<std::ptrdiff_t> dimscalings(out_meta->dim_scaling,
                                                out_meta->dim_scaling + out_meta->dims);
        // ⚠️ THE RING'S SLOT-0 METADATA IS NOT VALID UNTIL ITS PRODUCER HAS STAMPED IT
        // (2026-08-31). We publish the output buffer's ndarray descriptor from
        // signal_buffer->get_metadata(0), and on a freshly started pipeline that object can
        // still be (a) untouched -- dims 0, so byte size 0 -- or (b) a metadata_pool object
        // recycled from ANOTHER quantity, still carrying its dim names. Both reach
        // ensure_frame_desc(), which FATALs, and the controlled shutdown that follows takes
        // the DPDK workers and the whole node with it. One restart on 2026-08-31 cost cx19
        // ("dimname mismatch: SK != S", case b) and cx44 ("frame description size (0)",
        // case a). The `assert(out_meta->dims > 0)` above is compiled out in Release, so
        // nothing caught either one.
        //
        // ⚠️ A SIZE CHECK ALONE CANNOT CATCH (b): dim[0] is forced from frame_size a few
        // lines up, so a foreign descriptor always has the RIGHT byte size and differs only
        // in its labels. Hence the quantity-name gate.
        //
        // So publish only what we can show is ours. Deferring costs nothing -- the
        // descriptor is pure metadata and the frame DATA is copied either way -- and a later
        // frame publishes it once the producer has stamped slot 0. It stays loud: a
        // descriptor that NEVER becomes publishable is a real bug and must not hide.
        bool desc_ok = out_meta->dims > 0;
        std::string why;
        if (!desc_ok)
            why = "metadata has no dimensions yet";
        const std::string qname = out_meta->get_name();
        if (desc_ok && qname.empty()) {
            desc_ok = false;
            why = "metadata carries no quantity name yet";
        }
        for (size_t d = 0; desc_ok && d < dimnames.size(); ++d)
            if (!dimnames[d].valid()) {
                desc_ok = false;
                why = fmt::format("dimension {:d} has no name yet", d);
            }
        if (desc_ok && !_expect_quantity.empty() && qname != _expect_quantity) {
            desc_ok = false;
            why = fmt::format("ring metadata is '{:s}', not the '{:s}' this copy produces "
                              "-- a recycled pool object, not our producer's",
                              qname, _expect_quantity);
        }

        // ⚠️ AND CHECK THE DESCRIPTOR WE ACTUALLY BUILT, not just the metadata fields it came
        // from. `dims > 0` is NOT enough: an uninitialised pool object can carry dims == 3
        // with ZERO extents, whose byte size is 0, and that walks straight into
        // ensure_frame_desc()'s first FATAL ("frame description size (0) does not match
        // frame_size"). That is precisely how cx19 died AGAIN on the 17:01 restart, with the
        // first version of this guard already in the binary. Build it, measure it, then
        // decide.
        std::shared_ptr<const kotekan::FrameDesc> cand;
        if (desc_ok) {
            cand = kotekan::GenericNDArray::describe(out_meta->type, qname, extents, dimnames,
                                                     dimscalings);
            if (!cand || cand->get_byte_size() != (size_t)out_buffer->frame_size) {
                desc_ok = false;
                why = fmt::format("descriptor byte size {:d} != frame_size {:d}",
                                  cand ? cand->get_byte_size() : 0,
                                  (size_t)out_buffer->frame_size);
            }
        }

        if (desc_ok) {
            out_buffer->ensure_frame_desc(cand);
            /* test that things are consistent */
            out_meta->check_frame_desc(out_buffer->get_frame_desc<kotekan::GenericNDArray>());
            if (_desc_deferred) {
                INFO("cudaCopyFromRingbuffer[{:s}]: published the '{:s}' descriptor for {:s} "
                     "after deferring {:d} frame(s)",
                     unique_name, qname, out_buffer->buffer_name, _desc_deferred);
                _desc_deferred = 0;
            }
        } else if (++_desc_deferred, (_desc_deferred & (_desc_deferred - 1)) == 0) {
            // Powers of two only: loud at first, then quiet, but never silent.
            WARN("cudaCopyFromRingbuffer[{:s}]: NOT publishing an ndarray descriptor for "
                 "{:s} ({:s}); deferred {:d} frame(s) so far. The data still copies.",
                 unique_name, out_buffer->buffer_name, why, _desc_deferred);
        }

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
