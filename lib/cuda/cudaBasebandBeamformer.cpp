/**
 * @file
 * @brief CUDA baseband beamforming kernel
 *
 * This file has been generated automatically from the source file `bb.jl`.
 * Do not modify this C++ file, your changes will be lost.
 */

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

#include <fmt.hpp>
//
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

// Julia's `CuDevArray` type
template<typename T, int64_t N>
struct CuDeviceArray {
    T* ptr;
    int64_t maxsize;
    int64_t dims[N];
    int64_t len;
};
using kernel_arg = CuDeviceArray<int32_t, 1>;

/**
 * @class cudaBasebandBeamformer
 * @brief cudaCommand for baseband beamforming.
 *
 * Kernel by Kendrick Smith and Erik Schnetter.
 * https://github.com/eschnett/GPUIndexSpaces.jl/blob/main/output/bb.ptx
 */
class cudaBasebandBeamformer : public cudaCommand {
public:
    cudaBasebandBeamformer(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaBasebandBeamformer();
    cudaEvent_t execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events,
                        bool* quit) override;
    // virtual void finalize_frame(int gpu_frame_id) override;

private:
    // Kernel design parameters:
    static constexpr int cuda_number_of_beams = 96;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 512;
    static constexpr int cuda_number_of_frequencies = 16;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 32768;
    static constexpr double cuda_sampling_time_usec = 1.7066666666666668;
    static constexpr int cuda_shift_parameter_sigma = 3;

    // Kernel compile parameters:
    static constexpr int minthreads = 768;
    static constexpr int blocks_per_sm = 1;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 24;
    static constexpr int blocks = 512;
    static constexpr int shmem_bytes = 67712;

    // Kernel name
    const char* const kernel_name = "_Z13julia_bb_363213CuDeviceArrayI6Int8x4Li1ELi1EES_I6Int4x8Li1ELi1EES_I5Int32Li1ELi1EES_IS1_Li1ELi1EES_IS2_Li1ELi1EE";

    // Kernel arguments
    static constexpr std::size_t A_length = 3145728L;
    static constexpr std::size_t E_length = 536870912L;
    static constexpr std::size_t s_length = 12288L;
    static constexpr std::size_t J_length = 100663296L;
    static constexpr std::size_t info_length = 1572864L;

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
    cudaCommand(config, unique_name, host_buffers, device, "baseband_beamformer", "bb.ptx"),
    A_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    s_memname(config.get<std::string>(unique_name, "gpu_mem_output_scaling")),
    J_memname(config.get<std::string>(unique_name, "gpu_mem_formed_beams")),
    info_memname(config.get<std::string>(unique_name, "gpu_mem_info"))
//
{
    const int num_elements = config.get<int>(unique_name, "num_elements");
    const int num_local_freq = config.get<int>(unique_name, "num_local_freq");
    const int samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    const int num_beams = config.get<int>(unique_name, "num_beams");

    if (num_elements != cuda_number_of_dishes * cuda_number_of_polarizations)
        throw std::runtime_error(
            "The num_elements config setting must be "
            + std::to_string(cuda_number_of_dishes * cuda_number_of_polarizations)
            + " for the CUDA baseband beamformer");
    if (num_local_freq != cuda_number_of_frequencies)
        throw std::runtime_error("The num_local_freq config setting must be "
                                 + std::to_string(cuda_number_of_frequencies)
                                 + " for the CUDA baseband beamformer");
    if (samples_per_data_set != cuda_number_of_timesamples)
        throw std::runtime_error("The samples_per_data_set config setting must be "
                                 + std::to_string(cuda_number_of_timesamples)
                                 + " for the CUDA baseband beamformer");
    if (num_beams != cuda_number_of_beams)
        throw std::runtime_error("The num_beams config setting must be "
                                 + std::to_string(cuda_number_of_beams)
                                 + " for the CUDA baseband beamformer");

    set_command_type(gpuCommandType::KERNEL);
    const std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_name}, opts);
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

    // A, E, s, J
    const char* exc = "exception";
    // Divide lengths by sizeof(int32_t)?
    kernel_arg A_arg = {static_cast<int32_t*>(A_memory), A_length, {A_length}, A_length};
    kernel_arg E_arg = {static_cast<int32_t*>(E_memory), E_length, {E_length}, E_length};
    kernel_arg s_arg = {static_cast<int32_t*>(s_memory), s_length, {s_length}, s_length};
    kernel_arg J_arg = {static_cast<int32_t*>(J_memory), J_length, {J_length}, J_length};
    kernel_arg info_arg = {
        static_cast<int32_t*>(info_memory), info_length, {info_length}, info_length};

    // arr[1].ptr = (int32_t*)voltage_memory;
    // arr[1].maxsize = voltage_len;
    // arr[1].dims[0] = voltage_len;
    // arr[1].len = voltage_len;
    //
    // arr[2].ptr = shift_memory;
    // arr[2].maxsize = shift_len;
    // arr[2].dims[0] = shift_len / sizeof(int32_t);
    // arr[2].len = shift_len / sizeof(int32_t);
    //
    // arr[3].ptr = (int32_t*)output_memory;
    // arr[3].maxsize = output_len;
    // arr[3].dims[0] = output_len;
    // arr[3].len = output_len;
    //
    // arr[4].ptr = (int32_t*)info_memory;
    // arr[4].maxsize = info_len;
    // arr[4].dims[0] = info_len / sizeof(int32_t);
    // arr[4].len = info_len / sizeof(int32_t);

    void* parameters[] = {&exc, &A_arg, &E_arg, &s_arg, &J_arg, &info_arg};

    DEBUG("Kernel_name: {}", kernel_name);
    DEBUG("runtime_kernels[kernel_name]: {}", static_cast<void*>(runtime_kernels[kernel_name]));
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_name],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA Baseband Beamformer on GPU frame {:d}", gpu_frame_id);
    const CUresult err =
        cuLaunchKernel(runtime_kernels[kernel_name], blocks, 1, 1, threads_x, threads_y, 1,
                       shmem_bytes, device.getStream(cuda_stream_id), parameters, NULL);

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        INFO("Error number: {}", err);
        ERROR("cuLaunchKernel: {}", errStr);
    }

    return record_end_event(gpu_frame_id);
}
