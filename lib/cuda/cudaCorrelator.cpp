#include "cudaCorrelator.hpp"

#include "chordMetadata.hpp"
#include "div.hpp"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::div_noremainder;
using kotekan::mod;

REGISTER_CUDA_COMMAND(cudaCorrelator);

cudaCorrelator::cudaCorrelator(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device,
                               int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _samples_per_data_set(config.get<int>(unique_name, "samples_per_data_set")),
    _sub_integration_ntime(config.get<int>(unique_name, "sub_integration_ntime")),
    n2correlator(_num_elements, _num_local_freq) {
    _gpu_mem_voltage = config.get<std::string>(unique_name, "gpu_mem_voltage");
    _gpu_mem_correlation_triangle =
        config.get<std::string>(unique_name, "gpu_mem_correlation_triangle");
    if (_samples_per_data_set % _sub_integration_ntime)
        throw std::runtime_error(
            "The sub_integration_ntime parameter must evenly divide samples_per_data_set");
    // Find input buffer used for signalling ring-buffer state
    input_ringbuf_signal = dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")));
    if (inst == 0)
        input_ringbuf_signal->register_consumer(unique_name);

    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_voltage, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_correlation_triangle, true, false, true));

    // TODO: code for rfi mask. Just using a placeholder zero mask for now.
    void* device_rfimask = device.get_gpu_memory("rfimask", _num_local_freq * _samples_per_data_set
                                                                * sizeof(uint) / 32);
    cudaMemset(device_rfimask, 0xFF, _num_local_freq * _samples_per_data_set * sizeof(uint) / 32);
    rfimask = reinterpret_cast<uint*>(device_rfimask);

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaCorrelator");
}

cudaCorrelator::~cudaCorrelator() {}

int cudaCorrelator::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    const std::ptrdiff_t input_bytes = _num_elements * _num_local_freq * _samples_per_data_set;
    DEBUG("Input ring-buffer byte count: {:d}", input_bytes);
    DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::ptrdiff_t> val_in =
        input_ringbuf_signal->wait_and_claim_readable(unique_name, instance_num, input_bytes);
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    if (!val_in.has_value())
        return -1;
    unmodded_input_cursor = val_in.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    // Mod input cursor by the ringbuffer size
    input_cursor = mod(unmodded_input_cursor, input_ringbuf_signal->size);
    // Assert that we don't wrap around!
    assert(input_cursor + input_bytes <= input_ringbuf_signal->size);
    DEBUG("Modded input ring-buffer byte offset: {:d}", input_cursor);
    return 0;
}

cudaEvent_t cudaCorrelator::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();

    // Get the base ringbuffer address
    void* all_input_memory = device.get_gpu_memory(_gpu_mem_voltage, input_ringbuf_signal->size);
    // Add our ringbuffer offset (where we should read from)
    int8_t* input_memory = ((int8_t*)all_input_memory) + input_cursor;

    // aka "nt_outer" in n2k.hpp
    int num_subintegrations = _samples_per_data_set / _sub_integration_ntime;
    int vis_blocks_tr_root = (_num_elements + 1) / 16;
    int num_vis_blocks = vis_blocks_tr_root * (vis_blocks_tr_root + 1) / 2;
    int output_array_len =
        num_subintegrations * _num_local_freq * 16 * 16 * num_vis_blocks * 2 * sizeof(int32_t);
    void* output_memory = device.get_gpu_memory_array(_gpu_mem_correlation_triangle, gpu_frame_id,
                                                      _gpu_buffer_depth, output_array_len);

    record_start_event();

    n2correlator.launch((int*)output_memory, (int8_t*)input_memory, rfimask, num_subintegrations,
                        _sub_integration_ntime, device.getStream(cuda_stream_id), true);

    CHECK_CUDA_ERROR(cudaGetLastError());

    const std::shared_ptr<metadataObject> in_mc = input_ringbuf_signal->get_metadata(0);
    if (metadata_is_chord(in_mc)) {
        const std::shared_ptr<chordMetadata> in_meta = get_chord_metadata(in_mc);
        // Assert that input metadata array shape is as expected.
        DEBUG("Input metadata: array shape {:s}, array type {:s}", in_meta->get_dimensions_string(),
              in_meta->get_type_string());
        // Assert T x F x P x D
        assert(in_meta->dims == 4);
        // in_meta->dim[0] is in the ringbuffer
        assert(in_meta->dim[1] == _num_local_freq);
        assert(in_meta->dim[2] == 2);
        assert(in_meta->dim[3] == _num_elements / 2);
        // Set metadata on output buffer (correlation matrix)
        std::shared_ptr<metadataObject> const out_mc = device.create_gpu_memory_array_metadata(
            _gpu_mem_correlation_triangle, gpu_frame_id, in_mc->parent_pool);
        std::shared_ptr<chordMetadata> const out_meta = get_chord_metadata(out_mc);
        *out_meta = *in_meta;
        // Output shape is
        // (nt_outer = samples_per_data_set / sub_integration_ntimes x
        //  num_freq x
        //  num_elements (= num_polarizations x num_dishes) x
        //  num_elements (= num_polarizations x num_dishes) x
        //  2 complex components)
        // In int32 format.
        out_meta->set_name("N2");
        out_meta->type = chordDataType::int32;
        out_meta->dims = 7;
        out_meta->set_array_dimension(0, num_subintegrations, "Tc");
        out_meta->set_array_dimension(1, _num_local_freq, "F");
        out_meta->set_array_dimension(2, 2, "P2");
        out_meta->set_array_dimension(3, _num_elements / 2, "D2");
        out_meta->set_array_dimension(4, 2, "P1");
        out_meta->set_array_dimension(5, _num_elements / 2, "D1");
        out_meta->set_array_dimension(6, 2, "C");
        for (int d = out_meta->dims - 1; d >= 0; --d)
            if (d == out_meta->dims - 1)
                out_meta->stride[d] = 1;
            else
                out_meta->stride[d] = out_meta->stride[d + 1] * out_meta->dim[d + 1];

        // Since we do not use a ring buffer we need to set `meta->sample0_offset`
        assert(input_cursor % in_meta->sample_bytes() == 0);
        out_meta->sample0_offset = div_noremainder(unmodded_input_cursor, in_meta->sample_bytes());

        for (int freq = 0; freq < out_meta->nfreq; ++freq) {
            out_meta->time_downsampling_fpga[freq] =
                _sub_integration_ntime * in_meta->time_downsampling_fpga[freq];
            out_meta->half_fpga_sample0[freq] =
                in_meta->half_fpga_sample0[freq] + out_meta->time_downsampling_fpga[freq];
        }

        DEBUG("Set output metadata: array shape {:s}, array type {:s}",
              out_meta->get_dimensions_string(), out_meta->get_type_string());
    }

    return record_end_event();
}

void cudaCorrelator::finalize_frame() {
    // Advance the input ringbuffer
    const std::ptrdiff_t input_bytes = _num_elements * _num_local_freq * _samples_per_data_set;
    DEBUG("Advancing input ringbuffer by {:d} bytes", input_bytes);
    input_ringbuf_signal->finish_read(unique_name, instance_num, input_bytes);
    cudaCommand::finalize_frame();
}
