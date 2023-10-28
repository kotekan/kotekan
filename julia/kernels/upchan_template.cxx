/**
 * @file
 * @brief CUDA {{{kernel_name}}} kernel
 *
 * This file has been generated automatically.
 * Do not modify this C++ file, your changes will be lost.
 */

#include <bufferContainer.hpp>
#include <chordMetadata.hpp>
#include <cudaCommand.hpp>
#include <cudaDeviceInterface.hpp>

#include <fmt.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

/**
 * @class cuda{{{kernel_name}}}
 * @brief cudaCommand for {{{kernel_name}}}
 */
class cuda{{{kernel_name}}} : public cudaCommand {
public:
    cuda{{{kernel_name}}}(Config & config, const std::string& unique_name,
                          bufferContainer& host_buffers, cudaDeviceInterface& device);
    virtual ~cuda{{{kernel_name}}}();
    
    // int wait_on_precondition(int gpu_frame_id) override;
    cudaEvent_t execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>& pre_events) override;
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
            ptr(static_cast<T*>(ptr)),
            maxsize(bytes),
            dims{std::int64_t(maxsize / sizeof(T))},
            len(maxsize / sizeof(T)) {}
    };
    using kernel_arg = CuDeviceArray<int32_t, 1>;

    // Kernel design parameters:
    {{#kernel_design_parameters}}
    static constexpr {{{type}}} {{{name}}} = {{{value}}};
    {{/kernel_design_parameters}}

    // Kernel compile parameters:
    static constexpr int minthreads = {{{minthreads}}};
    static constexpr int blocks_per_sm = {{{num_blocks_per_sm}}};

    // Kernel call parameters:
    static constexpr int threads_x = {{{num_threads}}};
    static constexpr int threads_y = {{{num_warps}}};
    static constexpr int blocks = {{{num_blocks}}};
    static constexpr int shmem_bytes = {{{shmem_bytes}}};

    // Kernel name:
    const char* const kernel_symbol = "{{{kernel_symbol}}}";

    // Kernel arguments:
    {{#kernel_arguments}}
    static constexpr std::size_t {{{name}}}_length = {{{value}}};
    {{/kernel_arguments}}

    // Runtime parameters:
    {{#runtime_parameters}}
    {{{type}}} {{{name}}};
    {{/runtime_parameters}}

    // GPU memory:
    {{#memnames}}
    const std::string {{{name}}}_memname;
    {{/memnames}}

    // Host-side buffer arrays
    {{#memnames}}
    {{^hasbuffer}}
    std::vector<std::vector<std::uint8_t>> host_{{{name}}};
    {{/hasbuffer}}
    {{/memnames}}

    // Declare extra variables (if any)
    {{{declare_extra_variables}}}
};

REGISTER_CUDA_COMMAND(cuda{{{kernel_name}}});

cuda{{{kernel_name}}}::cuda{{{kernel_name}}}(Config& config,
                                             const std::string& unique_name,
                                             bufferContainer& host_buffers,
                                             cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "{{{kernel_name}}}", "{{{kernel_name}}}.ptx")
    {{#memnames}}
    {{#hasbuffer}}
    , {{{name}}}_memname(config.get<std::string>(unique_name, "{{{kotekan_name}}}"))
    {{/hasbuffer}}
    {{^hasbuffer}}
    , {{{name}}}_memname(unique_name + "/{{{kotekan_name}}}")
    {{/hasbuffer}}
    {{/memnames}}

    {{#memnames}}
    {{^hasbuffer}}
    , host_{{{name}}}(_gpu_buffer_depth)
    {{/hasbuffer}}
    {{/memnames}}
{
    // Add Graphviz entries for the GPU buffers used by this kernel
    {{#memnames}}
    {{#hasbuffer}}
    gpu_buffers_used.push_back(std::make_tuple({{{name}}}_memname, true, true, false));
    {{/hasbuffer}}
    {{^hasbuffer}}
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_{{{kotekan_name}}}", false, true, true));
    {{/hasbuffer}}
    {{/memnames}}

    {{#runtime_parameters}}
    const {{{type}}} {{{name}}} = config.get<{{{type}}}>(unique_name, "{{{name}}}");
    std::vector<float16_t> freq_gains16(freq_gains.size());
    for (std::size_t i=0; i<freq_gains16.size(); i++)
      freq_gains16[i] = freq_gains[i];
    const void* const G_host = freq_gains16.data();
    void* const G_memory = device.get_gpu_memory(G_memname, G_length);
    CHECK_CUDA_ERROR(cudaMemcpy(G_memory, G_host, G_length, cudaMemcpyHostToDevice));
    {{/runtime_parameters}}

    set_command_type(gpuCommandType::KERNEL);
    const std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_symbol}, opts);

    // Initialize extra variables (if necessary)
    {{{init_extra_variables}}}
}

cuda{{{kernel_name}}}::~cuda{{{kernel_name}}}() {}

cudaEvent_t cuda{{{kernel_name}}}::execute(cudaPipelineState& pipestate,
                                           const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute(pipestate.gpu_frame_id);

    {{#memnames}}
    {{#hasbuffer}}
    void* const {{{name}}}_memory = device.get_gpu_memory_array({{{name}}}_memname, pipestate.gpu_frame_id, {{{name}}}_length);
    {{/hasbuffer}}
    {{^hasbuffer}}
    host_{{{name}}}[pipestate.gpu_frame_id].resize({{{name}}}_length);
    void* const {{{name}}}_memory = device.get_gpu_memory({{{name}}}_memname, {{{name}}}_length);
    {{/hasbuffer}}
    {{/memnames}}

    {{#memnames}}
    {{#hasbuffer}}
    const char* const axislabels_{{{name}}}[] = { {{{axislabels}}} };
    const std::size_t axislengths_{{{name}}}[] = { {{{axislengths}}} };
    const std::size_t ndims_{{{name}}} = sizeof axislabels_{{{name}}} / sizeof *axislabels_{{{name}}};
    {{^isoutput}}
    const metadataContainer* const mc_{{{name}}} =
        device.get_gpu_memory_array_metadata({{{name}}}_memname, pipestate.gpu_frame_id);
    assert(mc_{{{name}}} && metadata_container_is_chord(mc_{{{name}}}));
    const chordMetadata* const meta_{{{name}}} = get_chord_metadata(mc_{{{name}}});
    INFO("input {{{name}}} array: {:s} {:s}",
        meta_{{{name}}}->get_type_string(),
        meta_{{{name}}}->get_dimensions_string());
    assert(meta_{{{name}}}->type == {{{type}}});
    assert(meta_{{{name}}}->dims == ndims_{{{name}}});
    for (std::size_t dim = 0; dim < ndims_{{{name}}}; ++dim) {
        assert(std::strncmp(meta_{{{name}}}->dim_name[dim],
                            axislabels_{{{name}}}[ndims_{{{name}}} - 1 - dim],
                            sizeof meta_{{{name}}}->dim_name[dim]) == 0);
        assert(meta_{{{name}}}->dim[dim] == int(axislengths_{{{name}}}[ndims_{{{name}}} - 1 - dim]));
    }
    {{/isoutput}}
    {{#isoutput}}
    metadataContainer* const mc_{{{name}}} =
        device.create_gpu_memory_array_metadata({{{name}}}_memname, pipestate.gpu_frame_id, mc_E->parent_pool);
    chordMetadata* const meta_{{{name}}} = get_chord_metadata(mc_{{{name}}});
    chord_metadata_copy(meta_{{{name}}}, meta_E);
    meta_{{{name}}}->type = {{{type}}};
    meta_{{{name}}}->dims = ndims_{{{name}}};
    for (std::size_t dim = 0; dim < ndims_{{{name}}}; ++dim) {
        std::strncpy(meta_{{{name}}}->dim_name[dim],
                     axislabels_{{{name}}}[ndims_{{{name}}} - 1 - dim],
                     sizeof meta_{{{name}}}->dim_name[dim]);
        meta_{{{name}}}->dim[dim] = axislengths_{{{name}}}[ndims_{{{name}}} - 1 - dim];
    }
    INFO("output {{{name}}} array: {:s} {:s}",
        meta_{{{name}}}->get_type_string(),
        meta_{{{name}}}->get_dimensions_string());
    {{/isoutput}}
    {{/hasbuffer}}
    {{/memnames}}

    record_start_event(pipestate.gpu_frame_id);

    const char* exc_arg = "exception";
    {{#kernel_arguments}}
    kernel_arg {{{name}}}_arg({{{name}}}_memory, {{{name}}}_length);
    {{/kernel_arguments}}
    void* args[] = {
        &exc_arg,
        {{#kernel_arguments}}
        &{{{name}}}_arg,
        {{/kernel_arguments}}
    };

    // Modify kernel arguments (if necessary)
    {{{modify_kernel_arguments}}}

    // Copy inputs to device memory
    {{#memnames}}
    {{^hasbuffer}}
    {{^isoutput}}
    CHECK_CUDA_ERROR(cudaMemcpyAsync({{{name}}}_memory,
                                     host_{{{name}}}[pipestate.gpu_frame_id].data(),
                                     {{{name}}}_length,
                                     cudaMemcpyHostToDevice,
                                     device.getStream(cuda_stream_id)));
    {{/isoutput}}
    {{/hasbuffer}}
    {{/memnames}}

    // Initialize host-side buffer arrays
    // TODO: Skip this for performance
    {{#memnames}}
    {{^hasbuffer}}
    {{#isoutput}}
    CHECK_CUDA_ERROR(cudaMemsetAsync({{{name}}}_memory, 0xff, {{{name}}}_length, device.getStream(cuda_stream_id)));
    {{/isoutput}}
    {{/hasbuffer}}
    {{/memnames}}

    DEBUG("kernel_symbol: {}", kernel_symbol);
    DEBUG("runtime_kernels[kernel_symbol]: {}", static_cast<void*>(runtime_kernels[kernel_symbol]));
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_symbol],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA {{{kernel_name}}} on GPU frame {:d}", pipestate.gpu_frame_id);
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
    {{#memnames}}
    {{^hasbuffer}}
    {{#isoutput}}
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_{{{name}}}[pipestate.gpu_frame_id].data(),
                                     {{{name}}}_memory,
                                     {{{name}}}_length,
                                     cudaMemcpyDeviceToHost,
                                     device.getStream(cuda_stream_id)));
    {{/isoutput}}
    {{/hasbuffer}}
    {{/memnames}}

    // Check error codes
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
    const std::int32_t error_code = *std::max_element((const std::int32_t*)&*host_info[pipestate.gpu_frame_id].begin(),
                                                      (const std::int32_t*)&*host_info[pipestate.gpu_frame_id].end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    return record_end_event(pipestate.gpu_frame_id);
}

void cuda{{{kernel_name}}}::finalize_frame(const int gpu_frame_id) {
    cudaCommand::finalize_frame(gpu_frame_id);

    {{#memnames}}
    {{^hasbuffer}}
    {{#isoutput}}
    for (std::size_t i = 0; i < host_{{{name}}}[gpu_frame_id].size(); ++i)
        if (host_{{{name}}}[gpu_frame_id][i] != 0)
            ERROR("cuda{{{kernel_name}}} returned '{{{name}}}' value {:d} at index {:d} (zero indicates noerror)",
                  host_{{{name}}}[gpu_frame_id][i], int(i));
    {{/isoutput}}
    {{/hasbuffer}}
    {{/memnames}}
}
