/**
 * @file
 * @brief CUDA BasebandBeamformer_pathfinder kernel
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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

/**
 * @class cudaBasebandBeamformer_pathfinder
 * @brief cudaCommand for BasebandBeamformer_pathfinder
 */
class cudaBasebandBeamformer_pathfinder : public cudaCommand {
public:
    cudaBasebandBeamformer_pathfinder(Config& config, const std::string& unique_name,
                                      bufferContainer& host_buffers, cudaDeviceInterface& device,
                                      int inst);
    virtual ~cudaBasebandBeamformer_pathfinder();

    // int wait_on_precondition(int gpu_frame_id) override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

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
    static constexpr int cuda_number_of_beams = 16;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 64;
    static constexpr int cuda_number_of_frequencies = 128;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 32768;
    static constexpr int cuda_shift_parameter_sigma = 2;

    // Kernel compile parameters:
    static constexpr int minthreads = 128;
    static constexpr int blocks_per_sm = 8;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 4;
    static constexpr int blocks = 2048;
    static constexpr int shmem_bytes = 9472;

    // Kernel name:
    const char* const kernel_symbol = "_Z2bb13CuDeviceArrayI6Int8x4Li1ELi1EES_I6Int4x8Li1ELi1EES_"
                                      "I5Int32Li1ELi1EES_IS1_Li1ELi1EES_IS2_Li1ELi1EE";

    // Kernel arguments:
    // A: gpu_mem_phase
    static constexpr chordDataType A_type = int8;
    static constexpr std::size_t A_rank = 0 + 1 + 1 + 1 + 1 + 1;
    static constexpr std::array<const char*, A_rank> A_labels = {
        "C", "D", "B", "P", "F",
    };
    static constexpr std::array<std::size_t, A_rank> A_lengths = {
        2, 64, 16, 2, 128,
    };
    static constexpr std::size_t A_length = chord_datatype_bytes(A_type) * 2 * 64 * 16 * 2 * 128;
    static_assert(A_length <= std::size_t(std::numeric_limits<int>::max()));
    //
    // E: gpu_mem_voltage
    static constexpr chordDataType E_type = int4p4;
    static constexpr std::size_t E_rank = 0 + 1 + 1 + 1 + 1;
    static constexpr std::array<const char*, E_rank> E_labels = {
        "D",
        "P",
        "F",
        "T",
    };
    static constexpr std::array<std::size_t, E_rank> E_lengths = {
        64,
        2,
        128,
        32768,
    };
    static constexpr std::size_t E_length = chord_datatype_bytes(E_type) * 64 * 2 * 128 * 32768;
    static_assert(E_length <= std::size_t(std::numeric_limits<int>::max()));
    //
    // s: gpu_mem_output_scaling
    static constexpr chordDataType s_type = int32;
    static constexpr std::size_t s_rank = 0 + 1 + 1 + 1;
    static constexpr std::array<const char*, s_rank> s_labels = {
        "B",
        "P",
        "F",
    };
    static constexpr std::array<std::size_t, s_rank> s_lengths = {
        16,
        2,
        128,
    };
    static constexpr std::size_t s_length = chord_datatype_bytes(s_type) * 16 * 2 * 128;
    static_assert(s_length <= std::size_t(std::numeric_limits<int>::max()));
    //
    // J: gpu_mem_formed_beams
    static constexpr chordDataType J_type = int4p4;
    static constexpr std::size_t J_rank = 0 + 1 + 1 + 1 + 1;
    static constexpr std::array<const char*, J_rank> J_labels = {
        "T",
        "P",
        "F",
        "B",
    };
    static constexpr std::array<std::size_t, J_rank> J_lengths = {
        32768,
        2,
        128,
        16,
    };
    static constexpr std::size_t J_length = chord_datatype_bytes(J_type) * 32768 * 2 * 128 * 16;
    static_assert(J_length <= std::size_t(std::numeric_limits<int>::max()));
    //
    // info: gpu_mem_info
    static constexpr chordDataType info_type = int32;
    static constexpr std::size_t info_rank = 0 + 1 + 1 + 1;
    static constexpr std::array<const char*, info_rank> info_labels = {
        "thread",
        "warp",
        "block",
    };
    static constexpr std::array<std::size_t, info_rank> info_lengths = {
        32,
        4,
        2048,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 4 * 2048;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()));
    //

    // Kotekan buffer names
    const std::string A_memname;
    const std::string E_memname;
    const std::string s_memname;
    const std::string J_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::vector<std::uint8_t>> host_info;
};

REGISTER_CUDA_COMMAND(cudaBasebandBeamformer_pathfinder);

cudaBasebandBeamformer_pathfinder::cudaBasebandBeamformer_pathfinder(Config& config,
                                                                     const std::string& unique_name,
                                                                     bufferContainer& host_buffers,
                                                                     cudaDeviceInterface& device,
                                                                     int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "BasebandBeamformer_pathfinder", "BasebandBeamformer_pathfinder.ptx"),
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

    // Only one of the instances of this pipeline stage need to build the kernel
    if (inst == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx("BasebandBeamformer_pathfinder.ptx", {kernel_symbol}, opts);
    }

    // Initialize extra variables (if necessary)
}

cudaBasebandBeamformer_pathfinder::~cudaBasebandBeamformer_pathfinder() {}

cudaEvent_t
cudaBasebandBeamformer_pathfinder::execute(cudaPipelineState& pipestate,
                                           const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    void* const A_memory = device.get_gpu_memory_array(A_memname, pipestate.gpu_frame_id, A_length);
    void* const E_memory = device.get_gpu_memory_array(E_memname, pipestate.gpu_frame_id, E_length);
    void* const s_memory = device.get_gpu_memory_array(s_memname, pipestate.gpu_frame_id, s_length);
    void* const J_memory = device.get_gpu_memory_array(J_memname, pipestate.gpu_frame_id, J_length);
    host_info[pipestate.gpu_frame_id].resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    /// A is an input buffer: check metadata
    const metadataContainer* const mc_A =
        device.get_gpu_memory_array_metadata(A_memname, pipestate.gpu_frame_id);
    assert(mc_A && metadata_container_is_chord(mc_A));
    const chordMetadata* const meta_A = get_chord_metadata(mc_A);
    INFO("input A array: {:s} {:s}", meta_A->get_type_string(), meta_A->get_dimensions_string());
    assert(meta_A->type == A_type);
    assert(meta_A->dims == A_rank);
    for (std::size_t dim = 0; dim < A_rank; ++dim) {
        assert(std::strncmp(meta_A->dim_name[dim], A_labels[A_rank - 1 - dim],
                            sizeof meta_A->dim_name[dim])
               == 0);
        assert(meta_A->dim[dim] == int(A_lengths[A_rank - 1 - dim]));
    }
    //
    /// E is an input buffer: check metadata
    const metadataContainer* const mc_E =
        device.get_gpu_memory_array_metadata(E_memname, pipestate.gpu_frame_id);
    assert(mc_E && metadata_container_is_chord(mc_E));
    const chordMetadata* const meta_E = get_chord_metadata(mc_E);
    INFO("input E array: {:s} {:s}", meta_E->get_type_string(), meta_E->get_dimensions_string());
    assert(meta_E->type == E_type);
    assert(meta_E->dims == E_rank);
    for (std::size_t dim = 0; dim < E_rank; ++dim) {
        assert(std::strncmp(meta_E->dim_name[dim], E_labels[E_rank - 1 - dim],
                            sizeof meta_E->dim_name[dim])
               == 0);
        assert(meta_E->dim[dim] == int(E_lengths[E_rank - 1 - dim]));
    }
    //
    /// s is an input buffer: check metadata
    const metadataContainer* const mc_s =
        device.get_gpu_memory_array_metadata(s_memname, pipestate.gpu_frame_id);
    assert(mc_s && metadata_container_is_chord(mc_s));
    const chordMetadata* const meta_s = get_chord_metadata(mc_s);
    INFO("input s array: {:s} {:s}", meta_s->get_type_string(), meta_s->get_dimensions_string());
    assert(meta_s->type == s_type);
    assert(meta_s->dims == s_rank);
    for (std::size_t dim = 0; dim < s_rank; ++dim) {
        assert(std::strncmp(meta_s->dim_name[dim], s_labels[s_rank - 1 - dim],
                            sizeof meta_s->dim_name[dim])
               == 0);
        assert(meta_s->dim[dim] == int(s_lengths[s_rank - 1 - dim]));
    }
    //
    /// J is an output buffer: set metadata
    metadataContainer* const mc_J = device.create_gpu_memory_array_metadata(
        J_memname, pipestate.gpu_frame_id, mc_E->parent_pool);
    chordMetadata* const meta_J = get_chord_metadata(mc_J);
    chord_metadata_copy(meta_J, meta_E);
    meta_J->type = J_type;
    meta_J->dims = J_rank;
    for (std::size_t dim = 0; dim < J_rank; ++dim) {
        std::strncpy(meta_J->dim_name[dim], J_labels[J_rank - 1 - dim],
                     sizeof meta_J->dim_name[dim]);
        meta_J->dim[dim] = J_lengths[J_rank - 1 - dim];
    }
    INFO("output J array: {:s} {:s}", meta_J->get_type_string(), meta_J->get_dimensions_string());
    //

    record_start_event();

    const char* exc_arg = "exception";
    kernel_arg A_arg(A_memory, A_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg s_arg(s_memory, s_length);
    kernel_arg J_arg(J_memory, J_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &A_arg, &E_arg, &s_arg, &J_arg, &info_arg,
    };

    // Copy inputs to device memory

    // Initialize host-side buffer arrays
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));

    DEBUG("kernel_symbol: {}", kernel_symbol);
    DEBUG("runtime_kernels[kernel_symbol]: {}",
          static_cast<void*>(device.runtime_kernels[kernel_symbol]));
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[kernel_symbol],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA BasebandBeamformer_pathfinder on GPU frame {:d}", pipestate.gpu_frame_id);
    const CUresult err =
        cuLaunchKernel(device.runtime_kernels[kernel_symbol], blocks, 1, 1, threads_x, threads_y, 1,
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

    return record_end_event();
}

void cudaBasebandBeamformer_pathfinder::finalize_frame() {
    cudaCommand::finalize_frame();

    for (std::size_t i = 0; i < host_info[gpu_frame_id].size(); ++i)
        if (host_info[gpu_frame_id][i] != 0)
            ERROR("cudaBasebandBeamformer_pathfinder returned 'info' value {:d} at index {:d} "
                  "(zero indicates noerror)",
                  host_info[gpu_frame_id][i], int(i));
}
