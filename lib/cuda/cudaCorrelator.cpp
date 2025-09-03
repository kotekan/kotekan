#include "cudaCorrelator.hpp"

#include <array>
#include <chordMetadata.hpp>
#include <cmath>
#include <cstdint>
#include <div.hpp>
#include <string>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::div_noremainder;
using kotekan::mod;

REGISTER_CUDA_COMMAND(cudaCorrelator);

cudaCorrelator::cudaCorrelator(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device,
                               const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst),
    _buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    _num_times(config.get<int>(unique_name, "num_times")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _samples_per_data_set(config.get<int>(unique_name, "samples_per_data_set")),
    _sub_integration_ntime(config.get<int>(unique_name, "sub_integration_ntime")),
    _voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    _n2k_correlation_name(config.get<std::string>(unique_name, "n2k_correlation_name")),
    voltage(_voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{_buffer_depth * _num_times, _num_local_freq, 2,
                                          _num_elements / 2},
            std::array<std::string, 4>{"T", "F", "P", "D"}, *this),
    n2k_correlation([&]() {
        // aka "nt_outer" in n2k.hpp
        const int num_subintegrations = _samples_per_data_set / _sub_integration_ntime;
        const int blocksize = 16;
        const int linear_num_blocks = (_num_elements + 1) / blocksize;
        const int triangle_num_blocks = linear_num_blocks * (linear_num_blocks + 1) / 2;
        const std::array<std::ptrdiff_t, 6> n2k_lengths{
            num_subintegrations, _num_local_freq, triangle_num_blocks, blocksize, blocksize, 2};
        const std::array<std::string, 6> n2k_dimnames{"Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"};
        return NDArrayBuffer<std::int32_t, 6>(
            _n2k_correlation_name, "n2k_correlation", n2k_lengths,
            std::array<std::string, 6>{"Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"}, *this);
    }()),
    n2correlator(_num_elements, _num_local_freq) {
    if (_samples_per_data_set % _sub_integration_ntime)
        throw std::runtime_error(
            "The sub_integration_ntime parameter must evenly divide samples_per_data_set");

    voltage.register_consumer();

    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(_n2k_correlation_name, true, false, true));

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
    DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
    const int code = voltage.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
        if (available_elements < _samples_per_data_set)
            return read_descriptor_t{.claimed = 0, .read = 0};
        else
            return read_descriptor_t{.claimed = _samples_per_data_set,
                                     .read = _samples_per_data_set};
    });
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    return code;
}

cudaEvent_t cudaCorrelator::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();

    const std::ptrdiff_t time_offset =
        voltage.get_read_valid().begin() % voltage.get_ndarray().extent(0);
    const kotekan::int4x2_swapped_withoffset_t* const input_memory =
        &voltage.get_ndarray()(time_offset, 0, 0, 0);

    // aka "nt_outer" in n2k.hpp
    const int num_subintegrations = _samples_per_data_set / _sub_integration_ntime;

    record_start_event();

    n2correlator.launch(n2k_correlation.get_ndarray().data(), (int8_t*)input_memory, rfimask,
                        num_subintegrations, _sub_integration_ntime,
                        device.getStream(cuda_stream_id), true);

    CHECK_CUDA_ERROR(cudaGetLastError());

    voltage.check_metadata();
    n2k_correlation.set_metadata(voltage.get_metadata());

    // Since we do not use a ring buffer we need to set `meta->sample0_offset`
    // TODO: do this automatically in `NDArrayRingBuffer`
    const std::shared_ptr<const chordMetadata> in_meta = voltage.get_metadata();
    const std::shared_ptr<chordMetadata> out_meta = n2k_correlation.get_metadata();
    out_meta->sample0_offset = voltage.get_read_valid().begin();
    for (int freq = 0; freq < out_meta->nfreq; ++freq) {
        out_meta->time_downsampling_fpga[freq] =
            _sub_integration_ntime * in_meta->time_downsampling_fpga[freq];
        out_meta->half_fpga_sample0[freq] =
            in_meta->half_fpga_sample0[freq] + out_meta->time_downsampling_fpga[freq];
    }

    return record_end_event();
}

void cudaCorrelator::finalize_frame() {
    // Advance the input ringbuffer
    DEBUG("Advancing input ringbuffer by {:d} samples", _samples_per_data_set);
    voltage.finish_read();
    cudaCommand::finalize_frame();
}
