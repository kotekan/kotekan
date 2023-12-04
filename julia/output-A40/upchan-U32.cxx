/**
 * @file
 * @brief CUDA Upchannelizer_U32 kernel
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
 * @class cudaUpchannelizer_U32
 * @brief cudaCommand for Upchannelizer_U32
 */
class cudaUpchannelizer_U32 : public cudaCommand {
public:
    cudaUpchannelizer_U32(Config& config, const std::string& unique_name,
                          bufferContainer& host_buffers, cudaDeviceInterface& device, int inst);
    virtual ~cudaUpchannelizer_U32();

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
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 512;
    static constexpr int cuda_number_of_frequencies = 16;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_taps = 4;
    static constexpr int cuda_number_of_timesamples = 32768;
    static constexpr int cuda_upchannelization_factor = 32;

    // Kernel compile parameters:
    static constexpr int minthreads = 512;
    static constexpr int blocks_per_sm = 2;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 16;
    static constexpr int blocks = 128;
    static constexpr int shmem_bytes = 67840;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z6upchan13CuDeviceArrayI5Int32Li1ELi1EES_I9Float16x2Li1ELi1EES_I6Int4x8Li1ELi1EES_IS2_"
        "Li1ELi1EES_IS0_Li1ELi1EE";

    // Kernel arguments:
    // Tactual: Tactual
    static constexpr chordDataType Tactual_type = int32;
    static constexpr std::size_t Tactual_rank = 0;
    static constexpr std::array<const char*, Tactual_rank> Tactual_labels = {};
    static constexpr std::array<std::size_t, Tactual_rank> Tactual_lengths = {};
    static constexpr std::size_t Tactual_length = chord_datatype_bytes(Tactual_type);
    static_assert(Tactual_length <= std::size_t(std::numeric_limits<int>::max()));
    //
    // G: gpu_mem_gain
    static constexpr chordDataType G_type = float16;
    static constexpr std::size_t G_rank = 0 + 1;
    static constexpr std::array<const char*, G_rank> G_labels = {
        "Fbar",
    };
    static constexpr std::array<std::size_t, G_rank> G_lengths = {
        512,
    };
    static constexpr std::size_t G_length = chord_datatype_bytes(G_type) * 512;
    static_assert(G_length <= std::size_t(std::numeric_limits<int>::max()));
    //
    // E: gpu_mem_input_voltage
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
    // Ebar: gpu_mem_output_voltage
    static constexpr chordDataType Ebar_type = int4p4;
    static constexpr std::size_t Ebar_rank = 0 + 1 + 1 + 1 + 1;
    static constexpr std::array<const char*, Ebar_rank> Ebar_labels = {
        "D",
        "P",
        "Fbar",
        "Tbar",
    };
    static constexpr std::array<std::size_t, Ebar_rank> Ebar_lengths = {
        512,
        2,
        512,
        1024,
    };
    static constexpr std::size_t Ebar_length =
        chord_datatype_bytes(Ebar_type) * 512 * 2 * 512 * 1024;
    static_assert(Ebar_length <= std::size_t(std::numeric_limits<int>::max()));
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
        16,
        128,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 16 * 128;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()));
    //

    // Kotekan buffer names
    const std::string Tactual_memname;
    const std::string G_memname;
    const std::string E_memname;
    const std::string Ebar_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::vector<std::uint8_t>> host_Tactual;
    std::vector<std::vector<std::uint8_t>> host_info;
};

REGISTER_CUDA_COMMAND(cudaUpchannelizer_U32);

cudaUpchannelizer_U32::cudaUpchannelizer_U32(Config& config, const std::string& unique_name,
                                             bufferContainer& host_buffers,
                                             cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "Upchannelizer_U32", "Upchannelizer_U32.ptx"),
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

    if (inst == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx("Upchannelizer_U32.ptx", {kernel_symbol}, opts);
    }

    // Initialize extra variables (if necessary)
}

cudaUpchannelizer_U32::~cudaUpchannelizer_U32() {}

cudaEvent_t cudaUpchannelizer_U32::execute(cudaPipelineState& pipestate,
                                           const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    host_Tactual[pipestate.gpu_frame_id].resize(Tactual_length);
    void* const Tactual_memory = device.get_gpu_memory(Tactual_memname, Tactual_length);
    void* const G_memory = device.get_gpu_memory_array(G_memname, pipestate.gpu_frame_id, G_length);
    void* const E_memory = device.get_gpu_memory_array(E_memname, pipestate.gpu_frame_id, E_length);
    void* const Ebar_memory =
        device.get_gpu_memory_array(Ebar_memname, pipestate.gpu_frame_id, Ebar_length);
    host_info[pipestate.gpu_frame_id].resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    /// G is an input buffer: check metadata
    const metadataContainer* const mc_G =
        device.get_gpu_memory_array_metadata(G_memname, pipestate.gpu_frame_id);
    assert(mc_G && metadata_container_is_chord(mc_G));
    const chordMetadata* const meta_G = get_chord_metadata(mc_G);
    INFO("input G array: {:s} {:s}", meta_G->get_type_string(), meta_G->get_dimensions_string());
    assert(meta_G->type == G_type);
    assert(meta_G->dims == G_rank);
    for (std::size_t dim = 0; dim < G_rank; ++dim) {
        assert(std::strncmp(meta_G->dim_name[dim], G_labels[G_rank - 1 - dim],
                            sizeof meta_G->dim_name[dim])
               == 0);
        assert(meta_G->dim[dim] == int(G_lengths[G_rank - 1 - dim]));
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
    /// Ebar is an output buffer: set metadata
    metadataContainer* const mc_Ebar = device.create_gpu_memory_array_metadata(
        Ebar_memname, pipestate.gpu_frame_id, mc_E->parent_pool);
    chordMetadata* const meta_Ebar = get_chord_metadata(mc_Ebar);
    chord_metadata_copy(meta_Ebar, meta_E);
    meta_Ebar->type = Ebar_type;
    meta_Ebar->dims = Ebar_rank;
    for (std::size_t dim = 0; dim < Ebar_rank; ++dim) {
        std::strncpy(meta_Ebar->dim_name[dim], Ebar_labels[Ebar_rank - 1 - dim],
                     sizeof meta_Ebar->dim_name[dim]);
        meta_Ebar->dim[dim] = Ebar_lengths[Ebar_rank - 1 - dim];
    }
    INFO("output Ebar array: {:s} {:s}", meta_Ebar->get_type_string(),
         meta_Ebar->get_dimensions_string());
    //

    record_start_event();

    const char* exc_arg = "exception";
    kernel_arg Tactual_arg(Tactual_memory, Tactual_length);
    kernel_arg G_arg(G_memory, G_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg Ebar_arg(Ebar_memory, Ebar_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &Tactual_arg, &G_arg, &E_arg, &Ebar_arg, &info_arg,
    };

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
    DEBUG("runtime_kernels[kernel_symbol]: {}",
          static_cast<void*>(device.runtime_kernels[kernel_symbol]));
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[kernel_symbol],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA Upchannelizer_U32 on GPU frame {:d}", pipestate.gpu_frame_id);
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

void cudaUpchannelizer_U32::finalize_frame() {
    cudaCommand::finalize_frame();

    for (std::size_t i = 0; i < host_info[gpu_frame_id].size(); ++i)
        if (host_info[gpu_frame_id][i] != 0)
            ERROR("cudaUpchannelizer_U32 returned 'info' value {:d} at index {:d} (zero indicates "
                  "noerror)",
                  host_info[gpu_frame_id][i], int(i));
}
