/**
 * @file
 * @brief CUDA FRBBeamformer kernel
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
 * @class cudaFRBBeamformer
 * @brief cudaCommand for FRBBeamformer
 */
class cudaFRBBeamformer : public cudaCommand {
public:
    cudaFRBBeamformer(Config& config, const std::string& unique_name, bufferContainer& host_buffers,
                      cudaDeviceInterface& device, int inst);
    virtual ~cudaFRBBeamformer();

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
    static constexpr int cuda_beam_layout_M = 48;
    static constexpr int cuda_beam_layout_N = 48;
    static constexpr int cuda_dish_layout_M = 24;
    static constexpr int cuda_dish_layout_N = 24;
    static constexpr int cuda_downsampling_factor = 40;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 512;
    static constexpr int cuda_number_of_frequencies = 256;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 2064;

    // Kernel compile parameters:
    static constexpr int minthreads = 768;
    static constexpr int blocks_per_sm = 1;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 24;
    static constexpr int blocks = 256;
    static constexpr int shmem_bytes = 76896;

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
        576,
    };
    static constexpr std::size_t S_length = chord_datatype_bytes(S_type) * 2 * 576;
    static_assert(S_length <= std::size_t(std::numeric_limits<int>::max()));
    //
    // W: gpu_mem_phase
    static constexpr chordDataType W_type = float16;
    static constexpr std::size_t W_rank = 0 + 1 + 1 + 1 + 1 + 1;
    static constexpr std::array<const char*, W_rank> W_labels = {
        "C", "dishM", "dishN", "P", "F",
    };
    static constexpr std::array<std::size_t, W_rank> W_lengths = {
        2, 24, 24, 2, 256,
    };
    static constexpr std::size_t W_length = chord_datatype_bytes(W_type) * 2 * 24 * 24 * 2 * 256;
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
        512,
        2,
        256,
        2064,
    };
    static constexpr std::size_t E_length = chord_datatype_bytes(E_type) * 512 * 2 * 256 * 2064;
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
        48,
        48,
        51,
        256,
    };
    static constexpr std::size_t I_length = chord_datatype_bytes(I_type) * 48 * 48 * 51 * 256;
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
        24,
        256,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 24 * 256;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()));
    //

    // Kotekan buffer names
    const std::string S_memname;
    const std::string W_memname;
    const std::string E_memname;
    const std::string I_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::vector<std::uint8_t>> host_info;
};

REGISTER_CUDA_COMMAND(cudaFRBBeamformer);

cudaFRBBeamformer::cudaFRBBeamformer(Config& config, const std::string& unique_name,
                                     bufferContainer& host_buffers, cudaDeviceInterface& device,
                                     int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "FRBBeamformer", "FRBBeamformer.ptx"),
    S_memname(config.get<std::string>(unique_name, "gpu_mem_dishlayout")),
    W_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    I_memname(config.get<std::string>(unique_name, "gpu_mem_beamgrid")),
    info_memname(unique_name + "/gpu_mem_info")

    ,
    host_info(_gpu_buffer_depth) {
    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(S_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(W_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(I_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));

    set_command_type(gpuCommandType::KERNEL);
    const std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    device.build_ptx("FRBBeamformer.ptx", {kernel_symbol}, opts);

    // Initialize extra variables (if necessary)
}

cudaFRBBeamformer::~cudaFRBBeamformer() {}

cudaEvent_t cudaFRBBeamformer::execute(cudaPipelineState& pipestate,
                                       const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    void* const S_memory = device.get_gpu_memory_array(S_memname, pipestate.gpu_frame_id, S_length);
    void* const W_memory = device.get_gpu_memory_array(W_memname, pipestate.gpu_frame_id, W_length);
    void* const E_memory = device.get_gpu_memory_array(E_memname, pipestate.gpu_frame_id, E_length);
    void* const I_memory = device.get_gpu_memory_array(I_memname, pipestate.gpu_frame_id, I_length);
    host_info[pipestate.gpu_frame_id].resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    /// S is an input buffer: check metadata
    const metadataContainer* const mc_S =
        device.get_gpu_memory_array_metadata(S_memname, pipestate.gpu_frame_id);
    assert(mc_S && metadata_container_is_chord(mc_S));
    const chordMetadata* const meta_S = get_chord_metadata(mc_S);
    INFO("input S array: {:s} {:s}", meta_S->get_type_string(), meta_S->get_dimensions_string());
    assert(meta_S->type == S_type);
    assert(meta_S->dims == S_rank);
    for (std::size_t dim = 0; dim < S_rank; ++dim) {
        assert(std::strncmp(meta_S->dim_name[dim], S_labels[S_rank - 1 - dim],
                            sizeof meta_S->dim_name[dim])
               == 0);
        assert(meta_S->dim[dim] == int(S_lengths[S_rank - 1 - dim]));
    }
    //
    /// W is an input buffer: check metadata
    const metadataContainer* const mc_W =
        device.get_gpu_memory_array_metadata(W_memname, pipestate.gpu_frame_id);
    assert(mc_W && metadata_container_is_chord(mc_W));
    const chordMetadata* const meta_W = get_chord_metadata(mc_W);
    INFO("input W array: {:s} {:s}", meta_W->get_type_string(), meta_W->get_dimensions_string());
    assert(meta_W->type == W_type);
    assert(meta_W->dims == W_rank);
    for (std::size_t dim = 0; dim < W_rank; ++dim) {
        assert(std::strncmp(meta_W->dim_name[dim], W_labels[W_rank - 1 - dim],
                            sizeof meta_W->dim_name[dim])
               == 0);
        assert(meta_W->dim[dim] == int(W_lengths[W_rank - 1 - dim]));
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
    /// I is an output buffer: set metadata
    metadataContainer* const mc_I = device.create_gpu_memory_array_metadata(
        I_memname, pipestate.gpu_frame_id, mc_E->parent_pool);
    chordMetadata* const meta_I = get_chord_metadata(mc_I);
    chord_metadata_copy(meta_I, meta_E);
    meta_I->type = I_type;
    meta_I->dims = I_rank;
    for (std::size_t dim = 0; dim < I_rank; ++dim) {
        std::strncpy(meta_I->dim_name[dim], I_labels[I_rank - 1 - dim],
                     sizeof meta_I->dim_name[dim]);
        meta_I->dim[dim] = I_lengths[I_rank - 1 - dim];
    }
    INFO("output I array: {:s} {:s}", meta_I->get_type_string(), meta_I->get_dimensions_string());
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

    DEBUG("Running CUDA FRBBeamformer on GPU frame {:d}", pipestate.gpu_frame_id);
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

void cudaFRBBeamformer::finalize_frame() {
    cudaCommand::finalize_frame();

    for (std::size_t i = 0; i < host_info[gpu_frame_id].size(); ++i)
        if (host_info[gpu_frame_id][i] != 0)
            ERROR("cudaFRBBeamformer returned 'info' value {:d} at index {:d} (zero indicates "
                  "noerror)",
                  host_info[gpu_frame_id][i], int(i));
}
