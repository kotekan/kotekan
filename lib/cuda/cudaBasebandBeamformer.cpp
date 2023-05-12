/**
 * @file
 * @brief CUDA BasebandBeamformer kernel
 *
 * This file has been generated automatically.
 * Do not modify this C++ file, your changes will be lost.
 */

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

#include <array>
#include <fmt.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

/**
 * @class cudaBasebandBeamformer
 * @brief cudaCommand for BasebandBeamformer
 */
class cudaBasebandBeamformer : public cudaCommand {
public:
    cudaBasebandBeamformer(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaBasebandBeamformer();
    cudaEvent_t execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events,
                        bool* quit) override;

private:
    // Julia's `CuDevArray` type
    template<typename T, int64_t N>
    struct CuDeviceArray {
        T* ptr;
        int64_t maxsize; // bytes
        int64_t dims[N]; // elements
        int64_t len;     // elements
        CuDeviceArray(void* const ptr, const std::size_t bytes) :
            ptr(static_cast<T*>(ptr)), maxsize(bytes), dims{int64_t(maxsize / sizeof(T))},
            len(maxsize / sizeof(T)) {}
    };
    using kernel_arg = CuDeviceArray<int32_t, 1>;

    // Kernel design parameters:
    static constexpr int cuda_number_of_beams = 96;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 512;
    static constexpr int cuda_number_of_frequencies = 16;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 32768;
    static constexpr int cuda_shift_parameter_sigma = 3;

    // Kernel compile parameters:
    static constexpr int minthreads = 768;
    static constexpr int blocks_per_sm = 1;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 24;
    static constexpr int blocks = 512;
    static constexpr int shmem_bytes = 67712;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z13julia_bb_362113CuDeviceArrayI6Int8x4Li1ELi1EES_I6Int4x8Li1ELi1EES_I5Int32Li1ELi1EES_"
        "IS1_Li1ELi1EES_IS2_Li1ELi1EE";

    // Kernel arguments:
    static constexpr std::size_t A_length = 3145728UL;
    static constexpr std::size_t E_length = 536870912UL;
    static constexpr std::size_t s_length = 12288UL;
    static constexpr std::size_t J_length = 100663296UL;
    static constexpr std::size_t info_length = 1572864UL;

    // Runtime parameters:

    // GPU memory:
    const std::string A_memname;
    const std::string E_memname;
    const std::string s_memname;
    const std::string J_memname;
    const std::string info_memname;
};

REGISTER_CUDA_COMMAND(cudaBasebandBeamformer);

cudaBasebandBeamformer::cudaBasebandBeamformer(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "BasebandBeamformer",
                "BasebandBeamformer.ptx"),
    A_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    s_memname(config.get<std::string>(unique_name, "gpu_mem_output_scaling")),
    J_memname(config.get<std::string>(unique_name, "gpu_mem_formed_beams")),
    info_memname(config.get<std::string>(unique_name, "gpu_mem_info")) {
    const int num_elements = config.get<int>(unique_name, "num_elements");
    if (num_elements != (cuda_number_of_dishes * cuda_number_of_polarizations))
        throw std::runtime_error(
            "The num_elements config setting must be "
            + std::to_string(cuda_number_of_dishes * cuda_number_of_polarizations)
            + " for the CUDA Baseband Beamformer");
    const int num_local_freq = config.get<int>(unique_name, "num_local_freq");
    if (num_local_freq != (cuda_number_of_frequencies))
        throw std::runtime_error("The num_local_freq config setting must be "
                                 + std::to_string(cuda_number_of_frequencies)
                                 + " for the CUDA Baseband Beamformer");
    const int samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    if (samples_per_data_set != (cuda_number_of_timesamples))
        throw std::runtime_error("The samples_per_data_set config setting must be "
                                 + std::to_string(cuda_number_of_timesamples)
                                 + " for the CUDA Baseband Beamformer");
    const int num_beams = config.get<int>(unique_name, "num_beams");
    if (num_beams != (cuda_number_of_beams))
        throw std::runtime_error("The num_beams config setting must be "
                                 + std::to_string(cuda_number_of_beams)
                                 + " for the CUDA Baseband Beamformer");


    set_command_type(gpuCommandType::KERNEL);
    const std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_symbol}, opts);
}

cudaBasebandBeamformer::~cudaBasebandBeamformer() {}

cudaEvent_t cudaBasebandBeamformer::execute(const int gpu_frame_id,
                                            const std::vector<cudaEvent_t>& /*pre_events*/,
                                            bool* const /*quit*/) {
    pre_execute(gpu_frame_id);

    void* const A_memory = device.get_gpu_memory_array(A_memname, gpu_frame_id, A_length);
    void* const E_memory = device.get_gpu_memory_array(E_memname, gpu_frame_id, E_length);
    void* const s_memory = device.get_gpu_memory_array(s_memname, gpu_frame_id, s_length);
    void* const J_memory = device.get_gpu_memory_array(J_memname, gpu_frame_id, J_length);
    void* const info_memory = device.get_gpu_memory_array(info_memname, gpu_frame_id, info_length);

    record_start_event(gpu_frame_id);

    const char* exc_arg = "exception";
    kernel_arg A_arg(A_memory, A_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg s_arg(s_memory, s_length);
    kernel_arg J_arg(J_memory, J_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {&exc_arg, &A_arg, &E_arg, &s_arg, &J_arg, &info_arg};

    DEBUG("kernel_symbol: {}", kernel_symbol);
    DEBUG("runtime_kernels[kernel_symbol]: {}", static_cast<void*>(runtime_kernels[kernel_symbol]));
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_symbol],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA  on GPU frame {:d}", gpu_frame_id);
    const CUresult err =
        cuLaunchKernel(runtime_kernels[kernel_symbol], blocks, 1, 1, threads_x, threads_y, 1,
                       shmem_bytes, device.getStream(cuda_stream_id), args, NULL);

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        INFO("Error number: {}", err);
        ERROR("cuLaunchKernel: {}", errStr);
    }

    return record_end_event(gpu_frame_id);
}
