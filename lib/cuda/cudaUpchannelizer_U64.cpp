/**
 * @file
 * @brief CUDA Upchannelizer_U64 kernel
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
 * @class cudaUpchannelizer_U64
 * @brief cudaCommand for Upchannelizer_U64
 */
class cudaUpchannelizer_U64 : public cudaCommand {
public:
    cudaUpchannelizer_U64(Config& config, const std::string& unique_name,
                          bufferContainer& host_buffers, cudaDeviceInterface& device);
    virtual ~cudaUpchannelizer_U64();

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
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 512;
    static constexpr int cuda_number_of_frequencies = 16;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_taps = 4;
    static constexpr int cuda_number_of_timesamples = 32768;
    static constexpr int cuda_upchannelization_factor = 64;

    // Kernel compile parameters:
    static constexpr int minthreads = 512;
    static constexpr int blocks_per_sm = 2;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 16;
    static constexpr int blocks = 128;
    static constexpr int shmem_bytes = 66816;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z17julia_upchan_407113CuDeviceArrayI5Int32Li1ELi1EES_I9Float16x2Li1ELi1EES_"
        "I6Int4x8Li1ELi1EES_IS2_Li1ELi1EES_IS0_Li1ELi1EE";

    // Kernel arguments:
    static constexpr std::size_t Tactual_length = 1UL;
    static constexpr std::size_t G_length = 2048UL;
    static constexpr std::size_t E_length = 536870912UL;
    static constexpr std::size_t Ebar_length = 536870912UL;
    static constexpr std::size_t info_length = 262144UL;

    // Runtime parameters:
    std::vector<float> freq_gains;

    // GPU memory:
    const std::string Tactual_memname;
    const std::string G_memname;
    const std::string E_memname;
    const std::string Ebar_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::vector<std::int32_t>> host_info;
};

REGISTER_CUDA_COMMAND(cudaUpchannelizer_U64);

cudaUpchannelizer_U64::cudaUpchannelizer_U64(Config& config, const std::string& unique_name,
                                             bufferContainer& host_buffers,
                                             cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "Upchannelizer_U64",
                "Upchannelizer_U64.ptx"),
    Tactual_memname(config.get<std::string>(unique_name, "Tactual")),
    G_memname(config.get<std::string>(unique_name, "gpu_mem_gain")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_input_voltage")),
    Ebar_memname(config.get<std::string>(unique_name, "gpu_mem_output_voltage")),
    info_memname(unique_name + "/info") {
    // // Add Graphviz entries for the GPU buffers used by this kernel.
    //
    //
    // gpu_buffers_used.push_back(std::make_tuple(Tactual_memname, true, true, false));
    //
    //
    //
    //
    // gpu_buffers_used.push_back(std::make_tuple(G_memname, true, true, false));
    //
    //
    //
    //
    // gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    //
    //
    //
    //
    // gpu_buffers_used.push_back(std::make_tuple(Ebar_memname, true, true, false));
    //
    //
    //
    //
    //
    // gpu_buffers_used.push_back(std::make_tuple(get_name() + "_info", false, true, true));
    //
    //

    const int num_dishes = config.get<int>(unique_name, "num_dishes");
    if (num_dishes != (cuda_number_of_dishes))
        throw std::runtime_error("The num_dishes config setting must be "
                                 + std::to_string(cuda_number_of_dishes)
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
    const int upchan_factor = config.get<int>(unique_name, "upchan_factor");
    if (upchan_factor != (cuda_upchannelization_factor))
        throw std::runtime_error("The upchan_factor config setting must be "
                                 + std::to_string(cuda_upchannelization_factor)
                                 + " for the CUDA Baseband Beamformer");

    const std::vector<float> freq_gains = config.get<std::vector<float>>(unique_name, "freq_gains");
    std::vector<float16_t> freq_gains16(freq_gains.size());
    for (std::size_t i = 0; i < freq_gains16.size(); i++)
        freq_gains16[i] = freq_gains[i];
    const void* const G_host = freq_gains16.data();
    void* const G_memory = device.get_gpu_memory(G_memname, G_length);
    CHECK_CUDA_ERROR(cudaMemcpy(G_memory, G_host, G_length, cudaMemcpyHostToDevice));

    set_command_type(gpuCommandType::KERNEL);
    const std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_symbol}, opts);

    //
    // const std::string Tactual_buffer_name = "host_" + Tactual_memname;
    // Buffer* const Tactual_buffer = host_buffers.get_buffer(Tactual_buffer_name.c_str());
    // assert(Tactual_buffer);
    //
    // register_consumer(Tactual_buffer, unique_name.c_str());
    //
    //
    //
    // const std::string G_buffer_name = "host_" + G_memname;
    // Buffer* const G_buffer = host_buffers.get_buffer(G_buffer_name.c_str());
    // assert(G_buffer);
    //
    // register_consumer(G_buffer, unique_name.c_str());
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
    // const std::string Ebar_buffer_name = "host_" + Ebar_memname;
    // Buffer* const Ebar_buffer = host_buffers.get_buffer(Ebar_buffer_name.c_str());
    // assert(Ebar_buffer);
    //
    //
    // register_producer(Ebar_buffer, unique_name.c_str());
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

cudaUpchannelizer_U64::~cudaUpchannelizer_U64() {}

// int cudaUpchannelizer_U64::wait_on_precondition(const int gpu_frame_id) {
//
//
//     const std::string Tactual_buffer_name = "host_" + Tactual_memname;
//     Buffer* const Tactual_buffer = host_buffers.get_buffer(Tactual_buffer_name.c_str());
//     assert(Tactual_buffer);
//     uint8_t* const Tactual_frame = wait_for_full_frame(Tactual_buffer, unique_name.c_str(),
//     gpu_frame_id); if (!Tactual_frame)
//         return -1;
//
//
//
//     const std::string G_buffer_name = "host_" + G_memname;
//     Buffer* const G_buffer = host_buffers.get_buffer(G_buffer_name.c_str());
//     assert(G_buffer);
//     uint8_t* const G_frame = wait_for_full_frame(G_buffer, unique_name.c_str(), gpu_frame_id);
//     if (!G_frame)
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
//
//
//
//
//     return 0;
// }

cudaEvent_t cudaUpchannelizer_U64::execute(const int gpu_frame_id,
                                           const std::vector<cudaEvent_t>& /*pre_events*/,
                                           bool* const /*quit*/) {
    pre_execute(gpu_frame_id);

    void* const Tactual_memory =
        device.get_gpu_memory_array(Tactual_memname, gpu_frame_id, Tactual_length);
    void* const G_memory = device.get_gpu_memory_array(G_memname, gpu_frame_id, G_length);
    void* const E_memory = device.get_gpu_memory_array(E_memname, gpu_frame_id, E_length);
    void* const Ebar_memory = device.get_gpu_memory_array(Ebar_memname, gpu_frame_id, Ebar_length);
    std::int32_t* const info_memory =
        static_cast<std::int32_t*>(device.get_gpu_memory(info_memname, info_length));
    host_info.resize(_gpu_buffer_depth);
    for (int i = 0; i < _gpu_buffer_depth; ++i)
        host_info[i].resize(info_length / sizeof(std::int32_t));

    record_start_event(gpu_frame_id);

    // Initialize host-side buffer arrays
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));

    const char* exc_arg = "exception";
    kernel_arg Tactual_arg(Tactual_memory, Tactual_length);
    kernel_arg G_arg(G_memory, G_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg Ebar_arg(Ebar_memory, Ebar_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &Tactual_arg, &G_arg, &E_arg, &Ebar_arg, &info_arg,
    };

    DEBUG("kernel_symbol: {}", kernel_symbol);
    DEBUG("runtime_kernels[kernel_symbol]: {}", static_cast<void*>(runtime_kernels[kernel_symbol]));
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_symbol],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA Upchannelizer_U64 on GPU frame {:d}", gpu_frame_id);
    const CUresult err =
        cuLaunchKernel(runtime_kernels[kernel_symbol], blocks, 1, 1, threads_x, threads_y, 1,
                       shmem_bytes, device.getStream(cuda_stream_id), args, NULL);

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        INFO("Error number: {}", err);
        ERROR("cuLaunchKernel: {}", errStr);
    }

    // Copy results back to host memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_info[gpu_frame_id].data(), info_memory, info_length,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

    return record_end_event(gpu_frame_id);
}

void cudaUpchannelizer_U64::finalize_frame(const int gpu_frame_id) {
    cudaCommand::finalize_frame(gpu_frame_id);

    //
    //
    //
    //
    // const std::string Ebar_buffer_name = "host_" + Ebar_memname;
    // Buffer* const Ebar_buffer = host_buffers.get_buffer(Ebar_buffer_name.c_str());
    // assert(Ebar_buffer);
    // mark_frame_full(Ebar_buffer, unique_name.c_str(), gpu_frame_id);
    //
    //
    // const std::string info_buffer_name = "host_" + info_memname;
    // Buffer* const info_buffer = host_buffers.get_buffer(info_buffer_name.c_str());
    // assert(info_buffer);
    // mark_frame_full(info_buffer, unique_name.c_str(), gpu_frame_id);
    //
    for (std::size_t i = 0; i < host_info[gpu_frame_id].size(); ++i)
        if (host_info[gpu_frame_id][i] != 0)
            ERROR("cudaUpchannelizer_U64 returned 'info' value {:d} at index {:d} (zero indicates "
                  "noerror)",
                  host_info[gpu_frame_id][i], int(i));
}
