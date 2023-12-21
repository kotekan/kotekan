/**
 * @file
 * @brief CUDA TransposeKernel_chord kernel
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
 * @class cudaTransposeKernel_chord
 * @brief cudaCommand for TransposeKernel_chord
 */
class cudaTransposeKernel_chord : public cudaCommand {
public:
    cudaTransposeKernel_chord(Config& config, const std::string& unique_name,
                              bufferContainer& host_buffers, cudaDeviceInterface& device,
                              const int inst);
    virtual ~cudaTransposeKernel_chord();

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
    static constexpr int cuda_number_of_timesamples = 32768;
    static constexpr int cuda_inner_number_of_dishes = 8;
    static constexpr int cuda_inner_number_of_timesamples = 16;

    // Kernel compile parameters:
    static constexpr int minthreads = 512;
    static constexpr int blocks_per_sm = 1;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 16;
    static constexpr int blocks = 16;
    static constexpr int shmem_bytes = 0;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z12xpose_kernel13CuDeviceArrayI6Int4x8Li1ELi1EES_IS0_Li1ELi1EES_I5Int32Li1ELi1EE";

    // Kernel arguments:
    // Ein: gpu_mem_voltage
    static constexpr chordDataType Ein_type = int4p4;
    static constexpr std::size_t Ein_rank = 0 + 1 + 1 + 1 + 1 + 1 + 1;
    static constexpr std::array<const char*, Ein_rank> Ein_labels = {
        "Dshort", "Tshort", "D", "P", "F", "T",
    };
    static constexpr std::array<std::size_t, Ein_rank> Ein_lengths = {
        8, 16, 64, 2, 16, 2048,
    };
    static constexpr std::size_t Ein_length =
        chord_datatype_bytes(Ein_type) * 8 * 16 * 64 * 2 * 16 * 2048;
    static_assert(Ein_length <= std::size_t(std::numeric_limits<int>::max()));
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
        16,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 16 * 16;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()));
    //

    // Kotekan buffer names
    const std::string Ein_memname;
    const std::string E_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::uint8_t> info_host;
};

REGISTER_CUDA_COMMAND(cudaTransposeKernel_chord);

cudaTransposeKernel_chord::cudaTransposeKernel_chord(Config& config, const std::string& unique_name,
                                                     bufferContainer& host_buffers,
                                                     cudaDeviceInterface& device, const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "TransposeKernel_chord", "TransposeKernel_chord.ptx"),
    Ein_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    info_memname(unique_name + "/gpu_mem_info")

    ,
    info_host(info_length) {
    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(Ein_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
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

cudaTransposeKernel_chord::~cudaTransposeKernel_chord() {}

cudaEvent_t cudaTransposeKernel_chord::execute(cudaPipelineState& /*pipestate*/,
                                               const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    void* const Ein_memory =
        device.get_gpu_memory_array(Ein_memname, gpu_frame_id, _gpu_buffer_depth, Ein_length);
    void* const E_memory =
        device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    /// Ein is an input buffer: check metadata
    std::shared_ptr<metadataObject> const Ein_mc =
        device.get_gpu_memory_array_metadata(Ein_memname, gpu_frame_id);
    assert(Ein_mc && metadata_is_chord(Ein_mc));
    const std::shared_ptr<chordMetadata> Ein_meta = get_chord_metadata(Ein_mc);
    INFO("input Ein array: {:s} {:s}", Ein_meta->get_type_string(),
         Ein_meta->get_dimensions_string());
    assert(Ein_meta->type == Ein_type);
    assert(Ein_meta->dims == Ein_rank);
    for (std::size_t dim = 0; dim < Ein_rank; ++dim) {
        assert(std::strncmp(Ein_meta->dim_name[dim], Ein_labels[Ein_rank - 1 - dim],
                            sizeof Ein_meta->dim_name[dim])
               == 0);
        assert(Ein_meta->dim[dim] == int(Ein_lengths[Ein_rank - 1 - dim]));
    }
    //
    /// E is an output buffer: set metadata
    std::shared_ptr<metadataObject> const E_mc =
        device.create_gpu_memory_array_metadata(E_memname, gpu_frame_id, Ein_mc->parent_pool);
    std::shared_ptr<chordMetadata> const E_meta = get_chord_metadata(E_mc);
    chord_metadata_copy(E_meta, E_meta);
    E_meta->type = E_type;
    E_meta->dims = E_rank;
    for (std::size_t dim = 0; dim < E_rank; ++dim) {
        std::strncpy(E_meta->dim_name[dim], E_labels[E_rank - 1 - dim],
                     sizeof E_meta->dim_name[dim]);
        E_meta->dim[dim] = E_lengths[E_rank - 1 - dim];
    }
    INFO("output E array: {:s} {:s}", E_meta->get_type_string(), E_meta->get_dimensions_string());
    //

    record_start_event();

    const char* exc_arg = "exception";
    kernel_arg Ein_arg(Ein_memory, Ein_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg,
        &Ein_arg,
        &E_arg,
        &info_arg,
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

    DEBUG("Running CUDA TransposeKernel_chord on GPU frame {:d}", gpu_frame_id);
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
    CHECK_CUDA_ERROR(cudaMemcpyAsync(info_host.data(), info_memory, info_length,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

    // Check error codes
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
    const std::int32_t error_code = *std::max_element((const std::int32_t*)&*info_host.begin(),
                                                      (const std::int32_t*)&*info_host.end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    for (std::size_t i = 0; i < info_host.size(); ++i)
        if (info_host[i] != 0)
            ERROR("cudaTransposeKernel_chord returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host[i], i);

    return record_end_event();
}

void cudaTransposeKernel_chord::finalize_frame() {
    // device.release_gpu_memory_array_metadata(Ein_memname, gpu_frame_id);
    // device.release_gpu_memory_array_metadata(E_memname, gpu_frame_id);

    cudaCommand::finalize_frame();
}
