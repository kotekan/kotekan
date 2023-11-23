/**
 * @file
 * @brief CUDA BasebandBeamformer_chord kernel
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
 * @class cudaBasebandBeamformer_chord
 * @brief cudaCommand for BasebandBeamformer_chord
 */
class cudaBasebandBeamformer_chord : public cudaCommand {
public:
    cudaBasebandBeamformer_chord(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, cudaDeviceInterface& device);
    virtual ~cudaBasebandBeamformer_chord();

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
    const char* const kernel_symbol = "_Z2bb13CuDeviceArrayI6Int8x4Li1ELi1EES_I6Int4x8Li1ELi1EES_"
                                      "I5Int32Li1ELi1EES_IS1_Li1ELi1EES_IS2_Li1ELi1EE";

    // Kernel arguments:
    static constexpr std::size_t A_length = 3145728ULL;
    static constexpr std::size_t E_length = 536870912ULL;
    static constexpr std::size_t s_length = 12288ULL;
    static constexpr std::size_t J_length = 100663296ULL;
    static constexpr std::size_t info_length = 1572864ULL;

    // Runtime parameters:

    // GPU memory:
    const std::string A_memname;
    const std::string E_memname;
    const std::string s_memname;
    const std::string J_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::vector<std::uint8_t>> host_info;

    // Declare extra variables (if any)
};

REGISTER_CUDA_COMMAND(cudaBasebandBeamformer_chord);

cudaBasebandBeamformer_chord::cudaBasebandBeamformer_chord(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& host_buffers,
                                                           cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "BasebandBeamformer_chord",
                "BasebandBeamformer_chord.ptx"),
    A_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    s_memname(config.get<std::string>(unique_name, "gpu_mem_output_scaling")),
    J_memname(config.get<std::string>(unique_name, "gpu_mem_formed_beams")),
    info_memname(unique_name + "/gpu_mem_info")

    ,
    host_info(_gpu_buffer_depth) {
    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(A_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(s_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(J_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));


    set_command_type(gpuCommandType::KERNEL);
    const std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_symbol}, opts);

    // Initialize extra variables (if necessary)
}

cudaBasebandBeamformer_chord::~cudaBasebandBeamformer_chord() {}

cudaEvent_t cudaBasebandBeamformer_chord::execute(cudaPipelineState& pipestate,
                                                  const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute(pipestate.gpu_frame_id);

    void* const A_memory =
        device.get_gpu_memory_array(A_memname, pipestate.gpu_frame_id, _gpu_buffer_depth, A_length);
    void* const E_memory =
        device.get_gpu_memory_array(E_memname, pipestate.gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const s_memory =
        device.get_gpu_memory_array(s_memname, pipestate.gpu_frame_id, _gpu_buffer_depth, s_length);
    void* const J_memory =
        device.get_gpu_memory_array(J_memname, pipestate.gpu_frame_id, _gpu_buffer_depth, J_length);
    host_info[pipestate.gpu_frame_id].resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    const char* const axislabels_A[] = {"C", "D", "B", "P", "F"};
    const std::size_t axislengths_A[] = {2, 512, 96, 2, 16};
    const std::size_t ndims_A = sizeof axislabels_A / sizeof *axislabels_A;
    const metadataContainer* const mc_A =
        device.get_gpu_memory_array_metadata(A_memname, pipestate.gpu_frame_id);
    assert(mc_A && metadata_container_is_chord(mc_A));
    const chordMetadata* const meta_A = get_chord_metadata(mc_A);
    INFO("input A array: {:s} {:s}", meta_A->get_type_string(), meta_A->get_dimensions_string());
    assert(meta_A->type == int8);
    assert(meta_A->dims == ndims_A);
    for (std::size_t dim = 0; dim < ndims_A; ++dim) {
        assert(std::strncmp(meta_A->dim_name[dim], axislabels_A[ndims_A - 1 - dim],
                            sizeof meta_A->dim_name[dim])
               == 0);
        assert(meta_A->dim[dim] == int(axislengths_A[ndims_A - 1 - dim]));
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
    const char* const axislabels_s[] = {"B", "P", "F"};
    const std::size_t axislengths_s[] = {96, 2, 16};
    const std::size_t ndims_s = sizeof axislabels_s / sizeof *axislabels_s;
    const metadataContainer* const mc_s =
        device.get_gpu_memory_array_metadata(s_memname, pipestate.gpu_frame_id);
    assert(mc_s && metadata_container_is_chord(mc_s));
    const chordMetadata* const meta_s = get_chord_metadata(mc_s);
    INFO("input s array: {:s} {:s}", meta_s->get_type_string(), meta_s->get_dimensions_string());
    assert(meta_s->type == int32);
    assert(meta_s->dims == ndims_s);
    for (std::size_t dim = 0; dim < ndims_s; ++dim) {
        assert(std::strncmp(meta_s->dim_name[dim], axislabels_s[ndims_s - 1 - dim],
                            sizeof meta_s->dim_name[dim])
               == 0);
        assert(meta_s->dim[dim] == int(axislengths_s[ndims_s - 1 - dim]));
    }
    const char* const axislabels_J[] = {"T", "P", "F", "B"};
    const std::size_t axislengths_J[] = {32768, 2, 16, 96};
    const std::size_t ndims_J = sizeof axislabels_J / sizeof *axislabels_J;
    metadataContainer* const mc_J = device.create_gpu_memory_array_metadata(
        J_memname, pipestate.gpu_frame_id, mc_E->parent_pool);
    chordMetadata* const meta_J = get_chord_metadata(mc_J);
    chord_metadata_copy(meta_J, meta_E);
    meta_J->type = int4p4;
    meta_J->dims = ndims_J;
    for (std::size_t dim = 0; dim < ndims_J; ++dim) {
        std::strncpy(meta_J->dim_name[dim], axislabels_J[ndims_J - 1 - dim],
                     sizeof meta_J->dim_name[dim]);
        meta_J->dim[dim] = axislengths_J[ndims_J - 1 - dim];
    }
    INFO("output J array: {:s} {:s}", meta_J->get_type_string(), meta_J->get_dimensions_string());

    record_start_event(pipestate.gpu_frame_id);

    const char* exc_arg = "exception";
    kernel_arg A_arg(A_memory, A_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg s_arg(s_memory, s_length);
    kernel_arg J_arg(J_memory, J_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &A_arg, &E_arg, &s_arg, &J_arg, &info_arg,
    };

    // Modify kernel arguments (if necessary)


    // Copy inputs to device memory

    // Initialize host-side buffer arrays
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));

    DEBUG("kernel_symbol: {}", kernel_symbol);
    DEBUG("runtime_kernels[kernel_symbol]: {}", static_cast<void*>(runtime_kernels[kernel_symbol]));
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_symbol],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA BasebandBeamformer_chord on GPU frame {:d}", pipestate.gpu_frame_id);
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

void cudaBasebandBeamformer_chord::finalize_frame(const int gpu_frame_id) {
    cudaCommand::finalize_frame(gpu_frame_id);

    for (std::size_t i = 0; i < host_info[gpu_frame_id].size(); ++i)
        if (host_info[gpu_frame_id][i] != 0)
            ERROR("cudaBasebandBeamformer_chord returned 'info' value {:d} at index {:d} (zero "
                  "indicates noerror)",
                  host_info[gpu_frame_id][i], int(i));
}
