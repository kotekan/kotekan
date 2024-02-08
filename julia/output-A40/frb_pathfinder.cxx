/**
 * @file
 * @brief CUDA FRBBeamformer_pathfinder kernel
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
 * @class cudaFRBBeamformer_pathfinder
 * @brief cudaCommand for FRBBeamformer_pathfinder
 */
class cudaFRBBeamformer_pathfinder : public cudaCommand {
public:
    cudaFRBBeamformer_pathfinder(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, cudaDeviceInterface& device,
                                 const int inst);
    virtual ~cudaFRBBeamformer_pathfinder();

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
    static constexpr int cuda_beam_layout_M = 16;
    static constexpr int cuda_beam_layout_N = 24;
    static constexpr int cuda_dish_layout_M = 8;
    static constexpr int cuda_dish_layout_N = 12;
    static constexpr int cuda_downsampling_factor = 40;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 64;
    static constexpr int cuda_number_of_frequencies = 2048;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 144;

    // Kernel compile parameters:
    static constexpr int minthreads = 192;
    static constexpr int blocks_per_sm = 4;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 6;
    static constexpr int blocks = 2048;
    static constexpr int shmem_bytes = 13920;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z3frb13CuDeviceArrayI7Int16x2Li1ELi1EES_I9Float16x2Li1ELi1EES_I6Int4x8Li1ELi1EES_IS1_"
        "Li1ELi1EES_I5Int32Li1ELi1EE";

    // Kernel arguments:
    // S: gpu_mem_dishlayout
    static constexpr chordDataType S_type = int16;
    static constexpr std::size_t S_rank = 0 + 1 + 1;
    static constexpr std::array<const char*, S_rank> S_labels = {
        "MN",
        "D",
    };
    static constexpr std::array<std::size_t, S_rank> S_lengths = {
        2,
        96,
    };
    static constexpr std::size_t S_length = chord_datatype_bytes(S_type) * 2 * 96;
    static_assert(S_length <= std::size_t(std::numeric_limits<int>::max()));
    //
    // W: gpu_mem_phase
    static constexpr chordDataType W_type = float16;
    static constexpr std::size_t W_rank = 0 + 1 + 1 + 1 + 1 + 1;
    static constexpr std::array<const char*, W_rank> W_labels = {
        "C", "dishM", "dishN", "P", "F",
    };
    static constexpr std::array<std::size_t, W_rank> W_lengths = {
        2, 8, 12, 2, 2048,
    };
    static constexpr std::size_t W_length = chord_datatype_bytes(W_type) * 2 * 8 * 12 * 2 * 2048;
    static_assert(W_length <= std::size_t(std::numeric_limits<int>::max()));
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
        2048,
        144,
    };
    static constexpr std::size_t E_length = chord_datatype_bytes(E_type) * 64 * 2 * 2048 * 144;
    static_assert(E_length <= std::size_t(std::numeric_limits<int>::max()));
    //
    // I: gpu_mem_beamgrid
    static constexpr chordDataType I_type = float16;
    static constexpr std::size_t I_rank = 0 + 1 + 1 + 1 + 1;
    static constexpr std::array<const char*, I_rank> I_labels = {
        "beamP",
        "beamQ",
        "Tbar",
        "F",
    };
    static constexpr std::array<std::size_t, I_rank> I_lengths = {
        16,
        24,
        3,
        2048,
    };
    static constexpr std::size_t I_length = chord_datatype_bytes(I_type) * 16 * 24 * 3 * 2048;
    static_assert(I_length <= std::size_t(std::numeric_limits<int>::max()));
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
        6,
        2048,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 6 * 2048;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()));
    //

    // Kotekan buffer names
    const std::string S_memname;
    const std::string W_memname;
    const std::string E_memname;
    const std::string I_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::vector<std::uint8_t>> info_host;
};

REGISTER_CUDA_COMMAND(cudaFRBBeamformer_pathfinder);

cudaFRBBeamformer_pathfinder::cudaFRBBeamformer_pathfinder(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& host_buffers,
                                                           cudaDeviceInterface& device,
                                                           const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "FRBBeamformer_pathfinder", "FRBBeamformer_pathfinder.ptx"),
    S_memname(config.get<std::string>(unique_name, "gpu_mem_dishlayout")),
    W_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    I_memname(config.get<std::string>(unique_name, "gpu_mem_beamgrid")),
    info_memname(unique_name + "/gpu_mem_info")

    ,
    info_host(_gpu_buffer_depth) {
    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(S_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(W_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(I_memname, true, true, false));
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

cudaFRBBeamformer_pathfinder::~cudaFRBBeamformer_pathfinder() {}

cudaEvent_t cudaFRBBeamformer_pathfinder::execute(cudaPipelineState& /*pipestate*/,
                                                  const std::vector<cudaEvent_t>& /*pre_events*/) {
    const int gpu_frame_index = gpu_frame_id % _gpu_buffer_depth;

    pre_execute();

    void* const S_memory =
        device.get_gpu_memory_array(S_memname, gpu_frame_id, _gpu_buffer_depth, S_length);
    void* const W_memory =
        device.get_gpu_memory_array(W_memname, gpu_frame_id, _gpu_buffer_depth, W_length);
    void* const E_memory =
        device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const I_memory =
        device.get_gpu_memory_array(I_memname, gpu_frame_id, _gpu_buffer_depth, I_length);
    info_host.at(gpu_frame_index).resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    /// S is an input buffer: check metadata
    const metadataContainer* const S_mc =
        device.get_gpu_memory_array_metadata(S_memname, gpu_frame_id);
    assert(S_mc && metadata_container_is_chord(S_mc));
    const chordMetadata* const S_meta = get_chord_metadata(S_mc);
    INFO("input S array: {:s} {:s}", S_meta->get_type_string(), S_meta->get_dimensions_string());
    assert(S_meta->type == S_type);
    assert(S_meta->dims == S_rank);
    for (std::size_t dim = 0; dim < S_rank; ++dim) {
        assert(std::strncmp(S_meta->dim_name[dim], S_labels[S_rank - 1 - dim],
                            sizeof S_meta->dim_name[dim])
               == 0);
        assert(S_meta->dim[dim] == int(S_lengths[S_rank - 1 - dim]));
    }
    //
    /// W is an input buffer: check metadata
    const metadataContainer* const W_mc =
        device.get_gpu_memory_array_metadata(W_memname, gpu_frame_id);
    assert(W_mc && metadata_container_is_chord(W_mc));
    const chordMetadata* const W_meta = get_chord_metadata(W_mc);
    INFO("input W array: {:s} {:s}", W_meta->get_type_string(), W_meta->get_dimensions_string());
    assert(W_meta->type == W_type);
    assert(W_meta->dims == W_rank);
    for (std::size_t dim = 0; dim < W_rank; ++dim) {
        assert(std::strncmp(W_meta->dim_name[dim], W_labels[W_rank - 1 - dim],
                            sizeof W_meta->dim_name[dim])
               == 0);
        assert(W_meta->dim[dim] == int(W_lengths[W_rank - 1 - dim]));
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
    /// I is an output buffer: set metadata
    metadataContainer* const I_mc =
        device.create_gpu_memory_array_metadata(I_memname, gpu_frame_id, E_mc->parent_pool);
    chordMetadata* const I_meta = get_chord_metadata(I_mc);
    chord_metadata_copy(I_meta, E_meta);
    I_meta->type = I_type;
    I_meta->dims = I_rank;
    for (std::size_t dim = 0; dim < I_rank; ++dim) {
        std::strncpy(I_meta->dim_name[dim], I_labels[I_rank - 1 - dim],
                     sizeof I_meta->dim_name[dim]);
        I_meta->dim[dim] = I_lengths[I_rank - 1 - dim];
    }
    INFO("output I array: {:s} {:s}", I_meta->get_type_string(), I_meta->get_dimensions_string());
    //

    record_start_event();

    const char* exc_arg = "exception";
    kernel_arg S_arg(S_memory, S_length);
    kernel_arg W_arg(W_memory, W_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg I_arg(I_memory, I_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &S_arg, &W_arg, &E_arg, &I_arg, &info_arg,
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

    DEBUG("Running CUDA FRBBeamformer_pathfinder on GPU frame {:d}", gpu_frame_id);
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
            ERROR("cudaFRBBeamformer_pathfinder returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host.at(gpu_frame_index)[i], i);

    return record_end_event();
}

void cudaFRBBeamformer_pathfinder::finalize_frame() {
    device.release_gpu_memory_array_metadata(S_memname, gpu_frame_id);
    device.release_gpu_memory_array_metadata(W_memname, gpu_frame_id);
    device.release_gpu_memory_array_metadata(E_memname, gpu_frame_id);
    device.release_gpu_memory_array_metadata(I_memname, gpu_frame_id);

    cudaCommand::finalize_frame();
}
