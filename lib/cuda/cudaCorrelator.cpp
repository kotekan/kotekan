#include "cudaCorrelator.hpp"

#include "DataType.hpp"            // for int4x2_swapped_withoffset_t, uint1x8_t
#include "NDArray.hpp"             // for NDArray
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer, extent_t, read_descriptor_t
#include "cudaCommand.hpp"         // for cudaCommand, REGISTER_CUDA_COMMAND, _factory_aliascud...
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "div.hpp"                 // for div_noremainder, mod, num_triangle_blocks
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG
#include "n2k/Correlator.hpp"      // for Correlator

#include "fmt.hpp" // for compile_string_to_view

#include <array>              // for array
#include <cassert>            // for assert
#include <chordMetadata.hpp>  // for chordMetadata
#include <cstddef>            // for ptrdiff_t
#include <cstdint>            // for int32_t, int8_t, uint32_t
#include <cuda_runtime_api.h> // for cudaGetLastError
#include <functional>         // for function
#include <memory>             // for shared_ptr, __shared_ptr_access
#include <stdexcept>          // for runtime_error
#include <string>             // for allocator, basic_string, string
#include <tuple>              // for tuple, make_tuple

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::div_noremainder;
using kotekan::mod;
using kotekan::num_triangle_blocks;

REGISTER_CUDA_COMMAND(cudaCorrelator);

cudaCorrelator::cudaCorrelator(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device,
                               const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst),
    _buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    _num_times(config.get<int>(unique_name, "num_times")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _sub_integration_ntime(config.get<int>(unique_name, "sub_integration_ntime")),
    _voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    _rfi_RFImask_name(config.get<std::string>(unique_name, "rfi_RFImask_name")),
    _n2k_correlation_name(config.get<std::string>(unique_name, "n2k_correlation_name")),
    _poison_buffers(config.get_default<bool>(unique_name, "poison_buffers", false)),
    voltage(_voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{_buffer_depth * _num_times, _num_local_freq, 2,
                                          _num_elements / 2},
            std::array<std::string, 4>{"T", "F", "P", "D"},
            std::array<std::ptrdiff_t, 4>{1, 1, 1, 1}, *this),
    rfi_RFImask(_rfi_RFImask_name, "RFImask",
                std::array<std::ptrdiff_t, 3>{_buffer_depth * div_noremainder(_num_times, 8 * 128),
                                              _num_local_freq, 128},
                std::array<std::string, 3>{"T8hi128", "F", "T8lo128"},
                std::array<std::ptrdiff_t, 3>{1024, 1, 8}, *this),
    n2k_correlation([&]() {
        // aka "nt_outer" in n2k.hpp
        const int num_subintegrations = div_noremainder(_num_times, _sub_integration_ntime);
        const int blocksize = 16;
        const int triangle_num_blocks = num_triangle_blocks(_num_elements, blocksize);
        const std::array<std::ptrdiff_t, 6> n2k_lengths{
            num_subintegrations, _num_local_freq, triangle_num_blocks, blocksize, blocksize, 2};
        const std::array<std::string, 6> n2k_dimnames{"Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"};
        const std::array<std::ptrdiff_t, 6> n2k_dimscalings{_sub_integration_ntime, 1, 16, 1, 1, 1};
        return NDArrayBuffer<std::int32_t, 6>(_n2k_correlation_name, "n2k_correlation", n2k_lengths,
                                              n2k_dimnames, n2k_dimscalings, *this);
    }()),
    n2correlator(_num_elements, _num_local_freq) {
    if (_num_times % _sub_integration_ntime)
        throw std::runtime_error(
            "The sub_integration_ntime parameter must evenly divide samples_per_data_set");

    voltage.register_consumer();
    rfi_RFImask.register_consumer();

    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(_n2k_correlation_name, true, false, true));

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaCorrelator");
}

cudaCorrelator::~cudaCorrelator() {}

int cudaCorrelator::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers
    DEBUG("Waiting for voltage input ringbuffer data for frame {:d}...", gpu_frame_id);
    const int voltage_errcode =
        voltage.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            if (available_elements < _num_times)
                return read_descriptor_t{.claimed = 0, .read = 0};
            else
                return read_descriptor_t{.claimed = _num_times, .read = _num_times};
        });
    if (voltage_errcode < 0)
        return voltage_errcode;
    DEBUG("Finished waiting for voltage input for data frame {:d}.", gpu_frame_id);

    DEBUG("Waiting for rfi_RFImask input ringbuffer data for frame {:d}...", gpu_frame_id);
    const int rfi_RFImask_errcode =
        rfi_RFImask.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            const int rfi_needed_samples = div_noremainder(_num_times, 8 * 128);
            if (available_elements < rfi_needed_samples)
                return read_descriptor_t{.claimed = 0, .read = 0};
            else
                return read_descriptor_t{.claimed = rfi_needed_samples, .read = rfi_needed_samples};
        });
    if (rfi_RFImask_errcode < 0)
        return rfi_RFImask_errcode;
    DEBUG("Finished waiting for rfi_RFImask input for data frame {:d}.", gpu_frame_id);

    return 0;
}

cudaEvent_t cudaCorrelator::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();

    voltage.check_metadata();
    rfi_RFImask.check_metadata();
    n2k_correlation.set_metadata(voltage.get_metadata());

    const std::shared_ptr<const chordMetadata> voltage_meta = voltage.get_metadata();
    const std::shared_ptr<const chordMetadata> rfi_meta = rfi_RFImask.get_metadata();
    const std::shared_ptr<chordMetadata> n2k_corr_meta = n2k_correlation.get_metadata();

    // Ensure consistency:
    assert(voltage_meta->get_fpga_seq_num()
               + voltage.get_read_valid().begin() * voltage_meta->get_time_downsampling_fpga()
           == rfi_meta->get_fpga_seq_num()
                  + rfi_RFImask.get_read_valid().begin() * rfi_meta->get_time_downsampling_fpga());

    // The input ringbuffer metadata do not contain time-dependent metadata,
    // so we must reconstruct it here. (fpga_seq_num)
    n2k_corr_meta->set_fpga_seq_num(voltage_meta->get_fpga_seq_num()
                                    + voltage.get_read_valid().begin()
                                          * voltage_meta->get_time_downsampling_fpga());
    n2k_corr_meta->set_time_downsampling_fpga(_sub_integration_ntime
                                              * voltage_meta->get_time_downsampling_fpga());

    // Set poison for debug checks.
    if (_poison_buffers)
        n2k_correlation.set_to_poison(0x80);

    // The ringbuffering here is fishy. We should fix the kernel instead.

    const std::ptrdiff_t time_offset =
        voltage.get_read_valid().begin() % voltage.get_ndarray().extent(0);
    // Ensure there is no ring-buffer wrap-around
    assert((voltage.get_read_valid().end() - 1) % voltage.get_ndarray().extent(0) >= time_offset);
    const kotekan::int4x2_swapped_withoffset_t* const input_memory =
        &voltage.get_ndarray()(time_offset, 0, 0, 0);

    const std::ptrdiff_t rfi_time_offset =
        rfi_RFImask.get_read_valid().begin() % rfi_RFImask.get_ndarray().extent(0);
    // Ensure there is no ring-buffer wrap-around
    assert((rfi_RFImask.get_read_valid().end() - 1) % rfi_RFImask.get_ndarray().extent(0)
           >= rfi_time_offset);
    const kotekan::uint1x8_t* const rfi_RFImask_memory =
        &rfi_RFImask.get_ndarray()(rfi_time_offset, 0, 0);

    // aka "nt_outer" in n2k.hpp
    const int num_subintegrations = div_noremainder(_num_times, _sub_integration_ntime);

    record_start_event();

    n2correlator.launch(n2k_correlation.get_ndarray().data(), (const int8_t*)input_memory,
                        (const uint32_t*)rfi_RFImask_memory, num_subintegrations,
                        _sub_integration_ntime, device.getStream(cuda_stream_id), true);

    CHECK_CUDA_ERROR(cudaGetLastError());

    // Check if poison value made it through
    if (_poison_buffers)
        n2k_correlation.check_for_poison(0x80);

    return record_end_event();
}

void cudaCorrelator::finalize_frame() {
    // Advance the input ringbuffers
    voltage.finish_read();
    rfi_RFImask.finish_read();
    cudaCommand::finalize_frame();
}
