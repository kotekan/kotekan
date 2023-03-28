#include "cudaUpchannelize.hpp"

#include "cudaUtils.hpp"
#include "math.h"

#include "fmt.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

const int sizeof_float16_t = 2;

REGISTER_CUDA_COMMAND(cudaUpchannelize);

cudaUpchannelize::cudaUpchannelize(Config& config, const std::string& unique_name,
                                   bufferContainer& host_buffers, cudaDeviceInterface& device,
                                   std::string name, std::string kernel_fn, int nsamples,
                                   std::string kernel_symbol) :
    // cudaCommand(config, unique_name, host_buffers, device, "upchannelize", "upchan.ptx") {
    cudaCommand(config, unique_name, host_buffers, device, name, kernel_fn),
    kernel_name(kernel_symbol) {
    _num_dishes = config.get<int>(unique_name, "num_dishes");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _upchan_factor = config.get<int>(unique_name, "upchan_factor");
    _gpu_mem_input_voltage = config.get<std::string>(unique_name, "gpu_mem_input_voltage");
    _gpu_mem_output_voltage = config.get<std::string>(unique_name, "gpu_mem_output_voltage");
    _gpu_mem_gain = config.get<std::string>(unique_name, "gpu_mem_gain");
    _gpu_mem_info = config.get<std::string>(unique_name, "gpu_mem_info");
    // Try config value "freq_gains" as either a scalar float or a vector of floats
    float gain0 = config.get_default<float>(unique_name, "freq_gains", 1.);
    std::vector<float> gains =
        config.get_default<std::vector<float>>(unique_name, "freq_gains", std::vector<float>());

    if (_num_dishes != cuda_ndishes)
        throw std::runtime_error(fmt::format(
            "The num_dishes config setting must be {:d} for the CUDA Upchannelizer", cuda_ndishes));
    if (_num_local_freq != cuda_nfreq)
        throw std::runtime_error(
            fmt::format("The num_local_freq config setting must be {:d} for the CUDA Upchannelizer",
                        cuda_nfreq));
    // if (_samples_per_data_set != cuda_nsamples)
    if (_samples_per_data_set != nsamples)
        throw std::runtime_error(fmt::format(
            "The samples_per_data_set config setting must be {:d} for the CUDA Upchannelizer",
            // cuda_nsamples));
            nsamples));

    size_t ngains = _num_local_freq * _upchan_factor;
    if (gains.size() == 0) {
        for (size_t i = 0; i < ngains; i++)
            gains.push_back(gain0);
    }
    if (gains.size() != ngains)
        throw std::runtime_error(
            fmt::format("The number of elements in the 'freq_gains' config setting array must be "
                        "{:d} for the CUDA Upchannelizer",
                        ngains));
    // number of polarizations
    const int P = 2;
    gain_len = ngains * sizeof(float16_t);
    // 2 complex terms, int4+4
    voltage_input_len = (size_t)_samples_per_data_set * P * _num_local_freq * _num_dishes;
    voltage_output_len = voltage_input_len;
    info_len = (size_t)(threads_x * threads_y * blocks_x * sizeof(int32_t));

    set_command_type(gpuCommandType::KERNEL);

    std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    // INFO("Looking for PTX symbol {:s} in {:s}", get_kernel_function_name(), kernel);
    // build_ptx({get_kernel_function_name()}, opts);
    build_ptx({kernel_symbol}, opts);

    std::vector<float16_t> gains16;
    gains16.resize(gains.size());
    for (size_t i = 0; i < gains.size(); i++)
        gains16[i] = gains[i];

    const float16_t* gain_host = gains16.data();
    float16_t* gain_gpu = (float16_t*)device.get_gpu_memory(_gpu_mem_gain, gain_len);
    CHECK_CUDA_ERROR(cudaMemcpy(gain_gpu, gain_host, gain_len, cudaMemcpyHostToDevice));
}

cudaUpchannelize::~cudaUpchannelize() {}


// This struct is Erik's interpretation of what Julia is expecting for its "CuDevArray" type.
template<typename T, int64_t N>
struct CuDeviceArray {
    T* ptr;
    int64_t maxsize;
    int64_t dims[N];
    int64_t len;
};
typedef CuDeviceArray<int32_t, 1> kernel_arg;

cudaEvent_t cudaUpchannelize::execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events,
                                      bool* quit) {
    (void)pre_events;
    (void)quit;
    pre_execute(gpu_frame_id);

    void* voltage_input_memory =
        device.get_gpu_memory_array(_gpu_mem_input_voltage, gpu_frame_id, voltage_input_len);
    void* voltage_output_memory =
        device.get_gpu_memory_array(_gpu_mem_output_voltage, gpu_frame_id, voltage_output_len);
    float16_t* gain_memory = (float16_t*)device.get_gpu_memory(_gpu_mem_gain, gain_len);
    int32_t* info_memory =
        (int32_t*)device.get_gpu_memory_array(_gpu_mem_info, gpu_frame_id, info_len);

    record_start_event(gpu_frame_id);

    CUresult err;
    // A, E, s, J
    const char* exc = "exception";
    kernel_arg arr[4];

    arr[0].ptr = (int32_t*)gain_memory;
    arr[0].maxsize = gain_len;
    arr[0].dims[0] = gain_len / sizeof(float16_t);
    arr[0].len = gain_len / sizeof(float16_t);

    arr[1].ptr = (int32_t*)voltage_input_memory;
    arr[1].maxsize = voltage_input_len;
    arr[1].dims[0] = voltage_input_len;
    arr[1].len = voltage_input_len;

    arr[2].ptr = (int32_t*)voltage_output_memory;
    arr[2].maxsize = voltage_output_len;
    arr[2].dims[0] = voltage_output_len;
    arr[2].len = voltage_output_len;

    arr[3].ptr = (int32_t*)info_memory;
    arr[3].maxsize = info_len;
    arr[3].dims[0] = info_len / sizeof(int32_t);
    arr[3].len = info_len / sizeof(int32_t);

    void* parameters[] = {
        &(exc), &(arr[0]), &(arr[1]), &(arr[2]), &(arr[3]),
    };

    DEBUG("Kernel_name: {}", kernel_name);
    DEBUG("runtime_kernels[kernel_name]: {}", (void*)runtime_kernels[kernel_name]);
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_name],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shared_mem_bytes));

    DEBUG("Running CUDA Upchannelizer on GPU frame {:d}", gpu_frame_id);
    err = cuLaunchKernel(runtime_kernels[kernel_name], blocks_x, blocks_y, 1, threads_x, threads_y,
                         1, shared_mem_bytes, device.getStream(cuda_stream_id), parameters, NULL);

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        INFO("Error number: {}", err);
        ERROR("ERROR IN cuLaunchKernel: {}", errStr);
    }

    return record_end_event(gpu_frame_id);
}
