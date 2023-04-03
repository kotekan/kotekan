#include "cudaFRBBeamformer.hpp"

#include "math.h"

#include "fmt.hpp"

#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaFRBBeamformer);

static const size_t sizeof_float16_t = 2;

cudaFRBBeamformer::cudaFRBBeamformer(Config& config, const std::string& unique_name,
                                     bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "FRB_beamformer", "frb.ptx") {
    _num_dishes = config.get<int>(unique_name, "num_dishes");
    _dish_grid_size = config.get<int>(unique_name, "dish_grid_size");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _time_downsampling = config.get<int>(unique_name, "time_downsampling");
    _gpu_mem_dishlayout = config.get<std::string>(unique_name, "gpu_mem_dishlayout");
    _gpu_mem_voltage = config.get<std::string>(unique_name, "gpu_mem_voltage");
    _gpu_mem_phase = config.get<std::string>(unique_name, "gpu_mem_phase");
    _gpu_mem_beamgrid = config.get<std::string>(unique_name, "gpu_mem_beamgrid");

    set_command_type(gpuCommandType::KERNEL);

    std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_name}, opts);

    if (_samples_per_data_set != cuda_samples_per_data_set)
        throw std::runtime_error(fmt::format(
            "The samples_per_data_set config setting must be {:d} for the CUDA FRB beamformer",
            cuda_samples_per_data_set));
    if (_time_downsampling != cuda_time_downsampling)
        throw std::runtime_error(fmt::format(
            "The time_downsampling config setting must be {:d} for the CUDA FRB beamformer",
            cuda_time_downsampling));
    if (_num_dishes != cuda_num_dishes)
        throw std::runtime_error(
            fmt::format("The num_dishes config setting must be {:d} for the CUDA FRB beamformer",
                        cuda_num_dishes));
    if (_dish_grid_size != cuda_dish_grid_size)
        throw std::runtime_error(fmt::format(
            "The dish_grid_size config setting must be {:d} for the CUDA FRB beamformer",
            cuda_dish_grid_size));
    if (_num_local_freq != cuda_num_local_freq)
        throw std::runtime_error(fmt::format(
            "The num_local_freq config setting must be {:d} for the CUDA FRB beamformer",
            cuda_num_local_freq));

    // Assumption:
    int32_t dish_m = _dish_grid_size;
    int32_t dish_n = _dish_grid_size;
    int32_t beam_p = _dish_grid_size * 2;
    int32_t beam_q = _dish_grid_size * 2;
    if (dish_m * dish_n < _num_dishes)
        throw std::runtime_error("Config parameter dish_grid_size^2 must be >= num_dishes");
    int32_t Td = (_samples_per_data_set + _time_downsampling - 1) / _time_downsampling;

    dishlayout_len = (size_t)dish_m * dish_n * 2 * sizeof(int16_t);
    // 2 polarizations x 2 for complex
    phase_len = (size_t)dish_m * dish_n * _num_local_freq * 2 * 2 * sizeof_float16_t;
    // 2 polarizations
    voltage_len = (size_t)_num_dishes * _num_local_freq * _samples_per_data_set * 2;
    beamgrid_len = (size_t)beam_p * beam_q * _num_local_freq * Td * sizeof_float16_t;
    info_len = (size_t)(threads_x * threads_y * blocks_x * sizeof(int32_t));

    host_info.resize(info_len / sizeof(int32_t));

    // Allocate GPU memory for dish-layout array, and fill from the config file entry!
    int16_t* dishlayout_memory =
        (int16_t*)device.get_gpu_memory(_gpu_mem_dishlayout, dishlayout_len);
    std::vector<int> dishlayout_config =
        config.get<std::vector<int>>(unique_name, "frb_beamformer_dish_layout");
    if (dishlayout_config.size() != (size_t)(dish_m * dish_n * 2))
        throw std::runtime_error(fmt::format("Config parameter frb_beamformer_dish_layout (length "
                                             "{}) must have length = 2 * dish_grid_size^2 = {}",
                                             dishlayout_config.size(), 2 * dish_m * dish_n));
    int16_t* dishlayout_cpu_memory = (int16_t*)malloc(dishlayout_len);
    for (size_t i = 0; i < dishlayout_len / sizeof(int16_t); i++)
        dishlayout_cpu_memory[i] = dishlayout_config[i];
    CHECK_CUDA_ERROR(cudaMemcpy(dishlayout_memory, dishlayout_cpu_memory, dishlayout_len,
                                cudaMemcpyHostToDevice));
    free(dishlayout_cpu_memory);
}

cudaFRBBeamformer::~cudaFRBBeamformer() {}

// This struct is Erik's interpretation of what Julia is expecting for its "CuDevArray" type.
template<typename T, int64_t N>
struct CuDeviceArray {
    T* ptr;
    int64_t maxsize;
    int64_t dims[N];
    int64_t len;
};
typedef CuDeviceArray<int32_t, 1> kernel_arg;

cudaEvent_t cudaFRBBeamformer::execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events,
                                       bool* quit) {
    (void)pre_events;
    (void)quit;
    pre_execute(gpu_frame_id);

    void* dishlayout_memory = device.get_gpu_memory(_gpu_mem_dishlayout, dishlayout_len);
    void* phase_memory = device.get_gpu_memory_array(_gpu_mem_phase, gpu_frame_id, phase_len);
    void* voltage_memory = device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id, voltage_len);
    void* beamgrid_memory =
        device.get_gpu_memory_array(_gpu_mem_beamgrid, gpu_frame_id, beamgrid_len);
    int32_t* info_memory = (int32_t*)device.get_gpu_memory(_gpu_mem_info, info_len);

    record_start_event(gpu_frame_id);

    // Initialize info_memory return codes
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_len, device.getStream(cuda_stream_id)));

    // dishlayout (S), phase (W), voltage (E), beamgrid (I), info
    const char* exc = "exception";
    // kernel_arg arr[5];
    kernel_arg arr[8];

    arr[0].ptr = (int32_t*)dishlayout_memory;
    arr[0].maxsize = dishlayout_len;
    arr[0].dims[0] = dishlayout_len / sizeof(int16_t);
    arr[0].len = dishlayout_len / sizeof(int16_t);

    arr[1].ptr = (int32_t*)phase_memory;
    arr[1].maxsize = phase_len;
    arr[1].dims[0] = phase_len / sizeof_float16_t;
    arr[1].len = phase_len / sizeof_float16_t;

    arr[2].ptr = (int32_t*)voltage_memory;
    arr[2].maxsize = voltage_len;
    arr[2].dims[0] = voltage_len;
    arr[2].len = voltage_len;

    arr[3].ptr = (int32_t*)beamgrid_memory;
    arr[3].maxsize = beamgrid_len;
    arr[3].dims[0] = beamgrid_len / sizeof_float16_t;
    arr[3].len = beamgrid_len / sizeof_float16_t;

    arr[4].ptr = (int32_t*)info_memory;
    arr[4].maxsize = info_len;
    arr[4].dims[0] = info_len / sizeof(int32_t);
    arr[4].len = info_len / sizeof(int32_t);

    arr[5].ptr = (int32_t*)nullptr;
    arr[5].maxsize = 0;
    arr[5].dims[0] = 0;
    arr[5].len = 0;

    arr[6].ptr = (int32_t*)nullptr;
    arr[6].maxsize = 0;
    arr[6].dims[0] = 0;
    arr[6].len = 0;

    arr[7].ptr = (int32_t*)nullptr;
    arr[7].maxsize = 0;
    arr[7].dims[0] = 0;
    arr[7].len = 0;

    void* parameters[] = {
        &(exc),    &(arr[0]), &(arr[1]), &(arr[2]), &(arr[3]),
        &(arr[4]), &(arr[5]), &(arr[6]), &(arr[7]),
    };

    DEBUG("Kernel_name: {}", kernel_name);
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_name],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shared_mem_bytes));

    DEBUG("Running CUDA FRB Beamformer on GPU frame {:d}", gpu_frame_id);
    CHECK_CU_ERROR(cuLaunchKernel(runtime_kernels[kernel_name], blocks_x, blocks_y, 1, threads_x,
                                  threads_y, 1, shared_mem_bytes, device.getStream(cuda_stream_id),
                                  parameters, NULL));

    // Copy "info" result code back to host memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_info.data(), info_memory, info_len,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

    return record_end_event(gpu_frame_id);
}

void cudaFRBBeamformer::finalize_frame(int gpu_frame_id) {
    cudaCommand::finalize_frame(gpu_frame_id);
    for (size_t i = 0; i < host_info.size(); i++)
        if (host_info[i] != 0)
            ERROR("cudaFRBBeamformer returned 'info' value {:d} at index {:d} (zero indicates no "
                  "error)",
                  host_info[i], i);
}
