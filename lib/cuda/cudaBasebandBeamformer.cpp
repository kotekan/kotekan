#include "cudaBasebandBeamformer.hpp"

#include "math.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaBasebandBeamformer);

cudaBasebandBeamformer::cudaBasebandBeamformer(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "baseband_beamformer", "bb.ptx") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_beams = config.get<int>(unique_name, "num_beams");
    _gpu_mem_voltage = config.get<std::string>(unique_name, "gpu_mem_voltage");
    _gpu_mem_phase = config.get<std::string>(unique_name, "gpu_mem_phase");
    _gpu_mem_output_scaling = config.get<std::string>(unique_name, "gpu_mem_output_scaling");
    _gpu_mem_formed_beams = config.get<std::string>(unique_name, "gpu_mem_formed_beams");
    _gpu_mem_info = unique_name + "/info";

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_voltage, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_phase, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output_scaling, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_formed_beams, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_info", false, true, true));

    if (_num_elements != cuda_nelements)
        throw std::runtime_error("The num_elements config setting must be "
                                 + std::to_string(cuda_nelements)
                                 + " for the CUDA baseband beamformer");
    if (_num_local_freq != cuda_nfreq)
        throw std::runtime_error("The num_local_freq config setting must be "
                                 + std::to_string(cuda_nfreq)
                                 + " for the CUDA baseband beamformer");
    if (_samples_per_data_set != cuda_nsamples)
        throw std::runtime_error("The samples_per_data_set config setting must be "
                                 + std::to_string(cuda_nsamples)
                                 + " for the CUDA baseband beamformer");
    if (_num_beams != cuda_nbeams)
        throw std::runtime_error("The num_beams config setting must be "
                                 + std::to_string(cuda_nbeams)
                                 + " for the CUDA baseband beamformer");

    voltage_len = (size_t)_num_elements * _num_local_freq * _samples_per_data_set;
    phase_len = (size_t)_num_elements * _num_local_freq * _num_beams * 2;
    shift_len = (size_t)_num_local_freq * _num_beams * 2 * sizeof(int32_t);
    output_len = (size_t)_num_local_freq * _num_beams * _samples_per_data_set * 2;
    info_len = (size_t)(threads_x * threads_y * blocks_x * sizeof(int32_t));

    command_type = gpuCommandType::KERNEL;

    std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_name}, opts);
}

cudaBasebandBeamformer::~cudaBasebandBeamformer() {}


// This struct is Erik's interpretation of what Julia is expecting for its "CuDevArray" type.
template<typename T, int64_t N>
struct CuDeviceArray {
    T* ptr;
    int64_t maxsize;
    int64_t dims[N];
    int64_t len;
};
typedef CuDeviceArray<int32_t, 1> kernel_arg;

cudaEvent_t cudaBasebandBeamformer::execute(cudaPipelineState& pipestate,
                                            const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(pipestate.gpu_frame_id);

    void* voltage_memory =
        device.get_gpu_memory_array(_gpu_mem_voltage, pipestate.gpu_frame_id, voltage_len);
    int8_t* phase_memory =
        (int8_t*)device.get_gpu_memory_array(_gpu_mem_phase, pipestate.gpu_frame_id, phase_len);
    int32_t* shift_memory = (int32_t*)device.get_gpu_memory_array(
        _gpu_mem_output_scaling, pipestate.gpu_frame_id, shift_len);
    void* output_memory =
        device.get_gpu_memory_array(_gpu_mem_formed_beams, pipestate.gpu_frame_id, output_len);
    int32_t* info_memory = (int32_t*)device.get_gpu_memory(_gpu_mem_info, info_len);

    host_info.resize(_gpu_buffer_depth);
    for (int i = 0; i < _gpu_buffer_depth; i++)
        host_info[i].resize(info_len / sizeof(int32_t));

    record_start_event(pipestate.gpu_frame_id);

    // Initialize info_memory return codes
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_len, device.getStream(cuda_stream_id)));

    // A, E, s, J
    const char* exc = "exception";
    kernel_arg arr[5];

    arr[0].ptr = (int32_t*)phase_memory;
    arr[0].maxsize = phase_len;
    arr[0].dims[0] = phase_len;
    arr[0].len = phase_len;

    arr[1].ptr = (int32_t*)voltage_memory;
    arr[1].maxsize = voltage_len;
    arr[1].dims[0] = voltage_len;
    arr[1].len = voltage_len;

    arr[2].ptr = shift_memory;
    arr[2].maxsize = shift_len;
    arr[2].dims[0] = shift_len / sizeof(int32_t);
    arr[2].len = shift_len / sizeof(int32_t);

    arr[3].ptr = (int32_t*)output_memory;
    arr[3].maxsize = output_len;
    arr[3].dims[0] = output_len;
    arr[3].len = output_len;

    arr[4].ptr = (int32_t*)info_memory;
    arr[4].maxsize = info_len;
    arr[4].dims[0] = info_len / sizeof(int32_t);
    arr[4].len = info_len / sizeof(int32_t);

    void* parameters[] = {
        &(exc), &(arr[0]), &(arr[1]), &(arr[2]), &(arr[3]), &(arr[4]),
    };

    DEBUG("Kernel_name: {}", kernel_name);
    DEBUG("runtime_kernels[kernel_name]: {}", (void*)runtime_kernels[kernel_name]);
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_name],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shared_mem_bytes));

    DEBUG("Running CUDA Baseband Beamformer on GPU frame {:d}", pipestate.gpu_frame_id);
    CHECK_CU_ERROR(cuLaunchKernel(runtime_kernels[kernel_name], blocks_x, blocks_y, 1, threads_x,
                                  threads_y, 1, shared_mem_bytes, device.getStream(cuda_stream_id),
                                  parameters, NULL));

    // Copy "info" result code back to host memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_info[pipestate.gpu_frame_id].data(), info_memory,
                                     info_len, cudaMemcpyDeviceToHost,
                                     device.getStream(cuda_stream_id)));

    return record_end_event(pipestate.gpu_frame_id);
}

void cudaBasebandBeamformer::finalize_frame(int gpu_frame_id) {
    cudaCommand::finalize_frame(gpu_frame_id);
    for (size_t i = 0; i < host_info[gpu_frame_id].size(); i++)
        if (host_info[gpu_frame_id][i] != 0)
            ERROR(
                "cudaBasebandBeamformer returned 'info' value {:d} at index {:d} (zero indicates no"
                "error)",
                host_info[gpu_frame_id][i], i);
}
