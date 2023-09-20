#include "cudaCorrelator.hpp"

#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

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

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_voltage, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_correlation_triangle, true, false, true));

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaCorrelator");
}

cudaCorrelator::~cudaCorrelator() {}

cudaEvent_t cudaCorrelator::execute(cudaPipelineState& pipestate,
                                    const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(pipestate.gpu_frame_id);

    uint32_t input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    void* input_memory =
        device.get_gpu_memory_array(_gpu_mem_voltage, pipestate.gpu_frame_id, input_frame_len);
    // aka "nt_outer" in n2k.hpp
    uint32_t num_subintegrations = _samples_per_data_set / _sub_integration_ntime;
    uint32_t output_array_len =
        num_subintegrations * _num_local_freq * _num_elements * _num_elements * 2 * sizeof(int32_t);
    void* output_memory = device.get_gpu_memory_array(_gpu_mem_correlation_triangle,
                                                      pipestate.gpu_frame_id, output_array_len);

    record_start_event(pipestate.gpu_frame_id);

    n2correlator.launch((int*)output_memory, (int8_t*)input_memory, num_subintegrations,
                        _sub_integration_ntime, device.getStream(cuda_stream_id));

    CHECK_CUDA_ERROR(cudaGetLastError());

    return record_end_event(pipestate.gpu_frame_id);
}
