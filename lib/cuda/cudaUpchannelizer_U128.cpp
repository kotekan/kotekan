/**
 * @file
 * @brief CUDA Upchannelizer_U128 kernel
 *
 * This file has been generated automatically.
 * Do not modify this C++ file, your changes will be lost.
 */

#include <algorithm>
#include <array>
#include <bufferContainer.hpp>
#include <cassert>
#include <chordMetadata.hpp>
#include <cstring>
#include <cudaCommand.hpp>
#include <cudaDeviceInterface.hpp>
#include <fmt.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

/**
 * @class cudaUpchannelizer_U128
 * @brief cudaCommand for Upchannelizer_U128
 */
class cudaUpchannelizer_U128 : public cudaCommand {
public:
    cudaUpchannelizer_U128(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, cudaDeviceInterface& device);
    virtual ~cudaUpchannelizer_U128();

    // int wait_on_precondition(int gpu_frame_id) override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
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
    static constexpr int cuda_upchannelization_factor = 128;

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
        "_Z6upchan13CuDeviceArrayI5Int32Li1ELi1EES_I9Float16x2Li1ELi1EES_I6Int4x8Li1ELi1EES_IS2_"
        "Li1ELi1EES_IS0_Li1ELi1EE";

    // Kernel arguments:
    static constexpr std::size_t Tactual_length = 4UL;
    static constexpr std::size_t G_length = 4096UL;
    static constexpr std::size_t E_length = 536870912UL;
    static constexpr std::size_t Ebar_length = 536870912UL;
    static constexpr std::size_t info_length = 262144UL;

    // Runtime parameters:

    // GPU memory:
    const std::string Tactual_memname;
    const std::string G_memname;
    const std::string E_memname;
    const std::string Ebar_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::vector<std::uint8_t>> host_Tactual;
    std::vector<std::vector<std::uint8_t>> host_info;

    // Declare extra variables (if any)
};

REGISTER_CUDA_COMMAND(cudaUpchannelizer_U128);

cudaUpchannelizer_U128::cudaUpchannelizer_U128(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "Upchannelizer_U128",
                "Upchannelizer_U128.ptx"),
    Tactual_memname(unique_name + "/Tactual"),
    G_memname(config.get<std::string>(unique_name, "gpu_mem_gain")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_input_voltage")),
    Ebar_memname(config.get<std::string>(unique_name, "gpu_mem_output_voltage")),
    info_memname(unique_name + "/gpu_mem_info")

    ,
    host_Tactual(_gpu_buffer_depth), host_info(_gpu_buffer_depth) {
    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_Tactual", false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(G_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(Ebar_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));


    set_command_type(gpuCommandType::KERNEL);
    const std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_symbol}, opts);

    // Initialize extra variables (if necessary)
}

cudaUpchannelizer_U128::~cudaUpchannelizer_U128() {}

cudaEvent_t cudaUpchannelizer_U128::execute(cudaPipelineState& pipestate,
                                            const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute(pipestate.gpu_frame_id);

    host_Tactual[pipestate.gpu_frame_id].resize(Tactual_length);
    void* const Tactual_memory = device.get_gpu_memory(Tactual_memname, Tactual_length);
    void* const G_memory =
        device.get_gpu_memory_array(G_memname, pipestate.gpu_frame_id, _gpu_buffer_depth, G_length);
    void* const E_memory =
        device.get_gpu_memory_array(E_memname, pipestate.gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const Ebar_memory = device.get_gpu_memory_array(Ebar_memname, pipestate.gpu_frame_id,
                                                          _gpu_buffer_depth, Ebar_length);
    host_info[pipestate.gpu_frame_id].resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    const char* const axislabels_G[] = {"Fbar"};
    const std::size_t axislengths_G[] = {2048};
    const std::size_t ndims_G = sizeof axislabels_G / sizeof *axislabels_G;
    const metadataContainer* const mc_G =
        device.get_gpu_memory_array_metadata(G_memname, pipestate.gpu_frame_id);
    assert(mc_G && metadata_container_is_chord(mc_G));
    const chordMetadata* const meta_G = get_chord_metadata(mc_G);
    INFO("input G array: {:s} {:s}", meta_G->get_type_string(), meta_G->get_dimensions_string());
    assert(meta_G->type == float16);
    assert(meta_G->dims == ndims_G);
    for (std::size_t dim = 0; dim < ndims_G; ++dim) {
        assert(std::strncmp(meta_G->dim_name[dim], axislabels_G[ndims_G - 1 - dim],
                            sizeof meta_G->dim_name[dim])
               == 0);
        assert(meta_G->dim[dim] == int(axislengths_G[ndims_G - 1 - dim]));
    }
    const char* const axislabels_E[] = {"D", "P", "F", "T"};
    const std::size_t axislengths_E[] = {512, 2, 16, 32768};
    const std::size_t ndims_E = sizeof axislabels_E / sizeof *axislabels_E;
    const metadataContainer* const mc_E =
        device.get_gpu_memory_array_metadata(E_memname, pipestate.gpu_frame_id);
    assert(mc_E && metadata_container_is_chord(mc_E));
    const chordMetadata* const meta_E = get_chord_metadata(mc_E);
    INFO("input E array: {:s} {:s}", meta_E->get_type_string(), meta_E->get_dimensions_string());
    assert(meta_E->type == int4p4);
    assert(meta_E->dims == ndims_E);
    for (std::size_t dim = 0; dim < ndims_E; ++dim) {
        assert(std::strncmp(meta_E->dim_name[dim], axislabels_E[ndims_E - 1 - dim],
                            sizeof meta_E->dim_name[dim])
               == 0);
        assert(meta_E->dim[dim] == int(axislengths_E[ndims_E - 1 - dim]));
    }
    const char* const axislabels_Ebar[] = {"D", "P", "Fbar", "Tbar"};
    const std::size_t axislengths_Ebar[] = {512, 2, 2048, 256};
    const std::size_t ndims_Ebar = sizeof axislabels_Ebar / sizeof *axislabels_Ebar;
    metadataContainer* const mc_Ebar = device.create_gpu_memory_array_metadata(
        Ebar_memname, pipestate.gpu_frame_id, mc_E->parent_pool);
    chordMetadata* const meta_Ebar = get_chord_metadata(mc_Ebar);
    chord_metadata_copy(meta_Ebar, meta_E);
    meta_Ebar->type = int4p4;
    meta_Ebar->dims = ndims_Ebar;
    for (std::size_t dim = 0; dim < ndims_Ebar; ++dim) {
        std::strncpy(meta_Ebar->dim_name[dim], axislabels_Ebar[ndims_Ebar - 1 - dim],
                     sizeof meta_Ebar->dim_name[dim]);
        meta_Ebar->dim[dim] = axislengths_Ebar[ndims_Ebar - 1 - dim];
    }
    INFO("output Ebar array: {:s} {:s}", meta_Ebar->get_type_string(),
         meta_Ebar->get_dimensions_string());

    record_start_event(pipestate.gpu_frame_id);

    const char* exc_arg = "exception";
    kernel_arg Tactual_arg(Tactual_memory, Tactual_length);
    kernel_arg G_arg(G_memory, G_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg Ebar_arg(Ebar_memory, Ebar_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &Tactual_arg, &G_arg, &E_arg, &Ebar_arg, &info_arg,
    };

    // Modify kernel arguments (if necessary)
    *(std::int32_t*)host_Tactual[pipestate.gpu_frame_id].data() = cuda_number_of_timesamples;


    // Copy inputs to device memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tactual_memory, host_Tactual[pipestate.gpu_frame_id].data(),
                                     Tactual_length, cudaMemcpyHostToDevice,
                                     device.getStream(cuda_stream_id)));

    // Initialize host-side buffer arrays
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));

    DEBUG("kernel_symbol: {}", kernel_symbol);
    DEBUG("runtime_kernels[kernel_symbol]: {}", static_cast<void*>(runtime_kernels[kernel_symbol]));
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_symbol],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA Upchannelizer_U128 on GPU frame {:d}", pipestate.gpu_frame_id);
    const CUresult err =
        cuLaunchKernel(runtime_kernels[kernel_symbol], blocks, 1, 1, threads_x, threads_y, 1,
                       shmem_bytes, device.getStream(cuda_stream_id), args, NULL);

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        ERROR("cuLaunchKernel: Error number: {}: {}", err, errStr);
    }

    // Copy results back to host memory
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_info[pipestate.gpu_frame_id].data(), info_memory,
                                     info_length, cudaMemcpyDeviceToHost,
                                     device.getStream(cuda_stream_id)));

    // Check error codes
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
    const std::int32_t error_code =
        *std::max_element((const std::int32_t*)&*host_info[pipestate.gpu_frame_id].begin(),
                          (const std::int32_t*)&*host_info[pipestate.gpu_frame_id].end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    return record_end_event(pipestate.gpu_frame_id);
}

void cudaUpchannelizer_U128::finalize_frame(const int gpu_frame_id) {
    cudaCommand::finalize_frame(gpu_frame_id);

    for (std::size_t i = 0; i < host_info[gpu_frame_id].size(); ++i)
        if (host_info[gpu_frame_id][i] != 0)
            ERROR("cudaUpchannelizer_U128 returned 'info' value {:d} at index {:d} (zero indicates "
                  "noerror)",
                  host_info[gpu_frame_id][i], int(i));
}
