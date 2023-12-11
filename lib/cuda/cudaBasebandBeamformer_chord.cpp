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
#include <limits>
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
                                 bufferContainer& host_buffers, cudaDeviceInterface& device,
                                 const int inst);
    virtual ~cudaBasebandBeamformer_chord();

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
    // A: gpu_mem_phase
    static constexpr chordDataType A_type = int8;
    static constexpr std::size_t A_rank = 0 + 1 + 1 + 1 + 1 + 1;
    static constexpr std::array<const char*, A_rank> A_labels = {
        "C", "D", "B", "P", "F",
    };
    static constexpr std::array<std::size_t, A_rank> A_lengths = {
        2, 512, 96, 2, 16,
    };
    static constexpr std::size_t A_length = chord_datatype_bytes(A_type) * 2 * 512 * 96 * 2 * 16;
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
        512,
        2,
        16,
        32768,
    };
    static constexpr std::size_t E_length = chord_datatype_bytes(E_type) * 512 * 2 * 16 * 32768;
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
        96,
        2,
        16,
    };
    static constexpr std::size_t s_length = chord_datatype_bytes(s_type) * 96 * 2 * 16;
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
        16,
        96,
    };
    static constexpr std::size_t J_length = chord_datatype_bytes(J_type) * 32768 * 2 * 16 * 96;
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
        24,
        512,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 24 * 512;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()));
    //

    // Kotekan buffer names

    const std::string A_memname;

    const std::string E_memname;

    const std::string s_memname;

    const std::string J_memname;

    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::vector<std::uint8_t>> info_host;
};

REGISTER_CUDA_COMMAND(cudaBasebandBeamformer_chord);

cudaBasebandBeamformer_chord::cudaBasebandBeamformer_chord(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& host_buffers,
                                                           cudaDeviceInterface& device,
                                                           const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "BasebandBeamformer_chord", "BasebandBeamformer_chord.ptx"),
    A_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    s_memname(config.get<std::string>(unique_name, "gpu_mem_output_scaling")),
    J_memname(config.get<std::string>(unique_name, "gpu_mem_formed_beams")),
    info_memname(unique_name + "/gpu_mem_info")

    ,
    info_host(_gpu_buffer_depth) {
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
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts);
    }
}

cudaBasebandBeamformer_chord::~cudaBasebandBeamformer_chord() {}

cudaEvent_t cudaBasebandBeamformer_chord::execute(cudaPipelineState& /*pipestate*/,
                                                  const std::vector<cudaEvent_t>& /*pre_events*/) {
    const int gpu_frame_index = gpu_frame_id % _gpu_buffer_depth;

    pre_execute();

    void* const A_memory =
        device.get_gpu_memory_array(A_memname, gpu_frame_id, _gpu_buffer_depth, A_length);
    void* const E_memory =
        device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const s_memory =
        device.get_gpu_memory_array(s_memname, gpu_frame_id, _gpu_buffer_depth, s_length);
    void* const J_memory =
        device.get_gpu_memory_array(J_memname, gpu_frame_id, _gpu_buffer_depth, J_length);
    info_host.at(gpu_frame_index).resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    /// A is an input buffer: check metadata
    const metadataContainer* const A_mc =
        device.get_gpu_memory_array_metadata(A_memname, gpu_frame_id);
    assert(A_mc && metadata_container_is_chord(A_mc));
    const chordMetadata* const A_meta = get_chord_metadata(A_mc);
    INFO("input A array: {:s} {:s}", A_meta->get_type_string(), A_meta->get_dimensions_string());
    assert(A_meta->type == A_type);
    assert(A_meta->dims == A_rank);
    for (std::size_t dim = 0; dim < A_rank; ++dim) {
        assert(std::strncmp(A_meta->dim_name[dim], A_labels[A_rank - 1 - dim],
                            sizeof A_meta->dim_name[dim])
               == 0);
        assert(A_meta->dim[dim] == int(A_lengths[A_rank - 1 - dim]));
    }
    //
    /// E is an input buffer: check metadata
    const metadataContainer* const E_mc =
        device.get_gpu_memory_array_metadata(E_memname, gpu_frame_id);
    assert(E_mc && metadata_container_is_chord(E_mc));
    const chordMetadata* const E_meta = get_chord_metadata(E_mc);
    INFO("input E array: {:s} {:s}", E_meta->get_type_string(), E_meta->get_dimensions_string());
    assert(E_meta->type == E_type);
    assert(E_meta->dims == E_rank);
    for (std::size_t dim = 0; dim < E_rank; ++dim) {
        assert(std::strncmp(E_meta->dim_name[dim], E_labels[E_rank - 1 - dim],
                            sizeof E_meta->dim_name[dim])
               == 0);
        assert(E_meta->dim[dim] == int(E_lengths[E_rank - 1 - dim]));
    }
    //
    /// s is an input buffer: check metadata
    const metadataContainer* const s_mc =
        device.get_gpu_memory_array_metadata(s_memname, gpu_frame_id);
    assert(s_mc && metadata_container_is_chord(s_mc));
    const chordMetadata* const s_meta = get_chord_metadata(s_mc);
    INFO("input s array: {:s} {:s}", s_meta->get_type_string(), s_meta->get_dimensions_string());
    assert(s_meta->type == s_type);
    assert(s_meta->dims == s_rank);
    for (std::size_t dim = 0; dim < s_rank; ++dim) {
        assert(std::strncmp(s_meta->dim_name[dim], s_labels[s_rank - 1 - dim],
                            sizeof s_meta->dim_name[dim])
               == 0);
        assert(s_meta->dim[dim] == int(s_lengths[s_rank - 1 - dim]));
    }
    //
    /// J is an output buffer: set metadata
    metadataContainer* const J_mc =
        device.create_gpu_memory_array_metadata(J_memname, gpu_frame_id, E_mc->parent_pool);
    chordMetadata* const J_meta = get_chord_metadata(J_mc);
    chord_metadata_copy(J_meta, E_meta);
    J_meta->type = J_type;
    J_meta->dims = J_rank;
    for (std::size_t dim = 0; dim < J_rank; ++dim) {
        std::strncpy(J_meta->dim_name[dim], J_labels[J_rank - 1 - dim],
                     sizeof J_meta->dim_name[dim]);
        J_meta->dim[dim] = J_lengths[J_rank - 1 - dim];
    }
    INFO("output J array: {:s} {:s}", J_meta->get_type_string(), J_meta->get_dimensions_string());
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

    DEBUG("Running CUDA BasebandBeamformer_chord on GPU frame {:d}", gpu_frame_id);
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
    CHECK_CUDA_ERROR(cudaMemcpyAsync(info_host.at(gpu_frame_index).data(), info_memory, info_length,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

    // Check error codes
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
    const std::int32_t error_code =
        *std::max_element((const std::int32_t*)&*info_host.at(gpu_frame_index).begin(),
                          (const std::int32_t*)&*info_host.at(gpu_frame_index).end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    for (std::size_t i = 0; i < info_host.at(gpu_frame_index).size(); ++i)
        if (info_host.at(gpu_frame_index)[i] != 0)
            ERROR("cudaBasebandBeamformer_chord returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host.at(gpu_frame_index)[i], i);

    return record_end_event();
}

void cudaBasebandBeamformer_chord::finalize_frame() {
    device.release_gpu_memory_array_metadata(A_memname, gpu_frame_id);
    device.release_gpu_memory_array_metadata(E_memname, gpu_frame_id);
    device.release_gpu_memory_array_metadata(s_memname, gpu_frame_id);
    device.release_gpu_memory_array_metadata(J_memname, gpu_frame_id);

    cudaCommand::finalize_frame();
}
