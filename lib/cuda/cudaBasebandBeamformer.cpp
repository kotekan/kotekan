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
#include <bufferContainer.hpp>
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
    virtual ~cudaBasebandBeamformer();

    // int wait_on_precondition(int gpu_frame_id) override;
    cudaEvent_t execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events,
                        bool* quit) override;
    void finalize_frame(int gpu_frame_id) override;

private:
    // Julia's `CuDevArray` type
    template<typename T, std::int64_t N>
    struct CuDeviceArray {
        T* ptr;
        std::int64_t maxsize; // bytes
        std::int64_t dims[N]; // elements
        std::int64_t len;     // elements
        CuDeviceArray(void* const ptr, const std::size_t bytes) :
            ptr(static_cast<T*>(ptr)), maxsize(bytes), dims{std::int64_t(maxsize / sizeof(T))},
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
        "_Z13julia_bb_357913CuDeviceArrayI6Int8x4Li1ELi1EES_I6Int4x8Li1ELi1EES_I5Int32Li1ELi1EES_"
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

    // Host-side buffer arrays
    std::vector<std::vector<std::int32_t>> host_info;
};

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

    //
    // const std::string A_buffer_name = "host_" + A_memname;
    // Buffer* const A_buffer = host_buffers.get_buffer(A_buffer_name.c_str());
    // assert(A_buffer);
    //
    // register_consumer(A_buffer, unique_name.c_str());
    //
    //
    //
    // const std::string E_buffer_name = "host_" + E_memname;
    // Buffer* const E_buffer = host_buffers.get_buffer(E_buffer_name.c_str());
    // assert(E_buffer);
    //
    // register_consumer(E_buffer, unique_name.c_str());
    //
    //
    //
    // const std::string s_buffer_name = "host_" + s_memname;
    // Buffer* const s_buffer = host_buffers.get_buffer(s_buffer_name.c_str());
    // assert(s_buffer);
    //
    // register_consumer(s_buffer, unique_name.c_str());
    //
    //
    //
    // const std::string J_buffer_name = "host_" + J_memname;
    // Buffer* const J_buffer = host_buffers.get_buffer(J_buffer_name.c_str());
    // assert(J_buffer);
    //
    //
    // register_producer(J_buffer, unique_name.c_str());
    //
    //
    // const std::string info_buffer_name = "host_" + info_memname;
    // Buffer* const info_buffer = host_buffers.get_buffer(info_buffer_name.c_str());
    // assert(info_buffer);
    //
    //
    // register_producer(info_buffer, unique_name.c_str());
    //
    //
}

cudaBasebandBeamformer::~cudaBasebandBeamformer() {}

// int cudaBasebandBeamformer::wait_on_precondition(const int gpu_frame_id) {
//
//
//     const std::string A_buffer_name = "host_" + A_memname;
//     Buffer* const A_buffer = host_buffers.get_buffer(A_buffer_name.c_str());
//     assert(A_buffer);
//     uint8_t* const A_frame = wait_for_full_frame(A_buffer, unique_name.c_str(), gpu_frame_id);
//     if (!A_frame)
//         return -1;
//
//
//
//     const std::string E_buffer_name = "host_" + E_memname;
//     Buffer* const E_buffer = host_buffers.get_buffer(E_buffer_name.c_str());
//     assert(E_buffer);
//     uint8_t* const E_frame = wait_for_full_frame(E_buffer, unique_name.c_str(), gpu_frame_id);
//     if (!E_frame)
//         return -1;
//
//
//
//     const std::string s_buffer_name = "host_" + s_memname;
//     Buffer* const s_buffer = host_buffers.get_buffer(s_buffer_name.c_str());
//     assert(s_buffer);
//     uint8_t* const s_frame = wait_for_full_frame(s_buffer, unique_name.c_str(), gpu_frame_id);
//     if (!s_frame)
//         return -1;
//
//
//
//
//
//
//
//     return 0;
// }

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

    // Initialize host-side buffer arrays
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));

    const char* exc_arg = "exception";
    kernel_arg A_arg(A_memory, A_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg s_arg(s_memory, s_length);
    kernel_arg J_arg(J_memory, J_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &A_arg, &E_arg, &s_arg, &J_arg, &info_arg,
    };

    DEBUG("kernel_symbol: {}", kernel_symbol);
    DEBUG("runtime_kernels[kernel_symbol]: {}", static_cast<void*>(runtime_kernels[kernel_symbol]));
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_symbol],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

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

void cudaBasebandBeamformer::finalize_frame(const int gpu_frame_id) {
    cudaCommand::finalize_frame(gpu_frame_id);

    //
    //
    //
    //
    // const std::string J_buffer_name = "host_" + J_memname;
    // Buffer* const J_buffer = host_buffers.get_buffer(J_buffer_name.c_str());
    // assert(J_buffer);
    // mark_frame_full(J_buffer, unique_name.c_str(), gpu_frame_id);
    //
    //
    // const std::string info_buffer_name = "host_" + info_memname;
    // Buffer* const info_buffer = host_buffers.get_buffer(info_buffer_name.c_str());
    // assert(info_buffer);
    // mark_frame_full(info_buffer, unique_name.c_str(), gpu_frame_id);
    //
    for (std::size_t i = 0; i < host_info[gpu_frame_id].size(); ++i)
        if (host_info[gpu_frame_id][i] != 0)
            ERROR("cudaBasebandBeamformer returned 'info' value {:d} at index {:d} (zero indicates "
                  "noerror)",
                  host_info[gpu_frame_id][i], int(i));
}
