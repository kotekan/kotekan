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
#include <limits>
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
    enum class args {
        {{#kernel_arguments}}
            {{{name}}},
        {{/kernel_arguments}}
        count
    };

    {{#kernel_arguments}}
    // {{{name}}}: {{{kotekan_name}}}
    static constexpr chordDataType {{{name}}}_type = {{{type}}};
    enum {{{name}}}_indices {
        {{#axes}}
            {{{name}}}_index_{{{label}}},
        {{/axes}}
        {{{name}}}_rank,
    };
    // static constexpr std::size_t {{{name}}}_rank = 0
    //     {{#axes}}
    //         +1
    //     {{/axes}}
    // ;
    static constexpr std::array<const char*, {{{name}}}_rank> {{{name}}}_labels = {
        {{#axes}}
            "{{{label}}}",
        {{/axes}}
    };
    static constexpr std::array<std::size_t, {{{name}}}_rank> {{{name}}}_lengths = {
        {{#axes}}
            {{{length}}},
        {{/axes}}
    };
    static constexpr std::size_t {{{name}}}_length = chord_datatype_bytes({{{name}}}_type)
        {{#axes}}
            * {{{length}}}
        {{/axes}}
        ;
    static_assert({{{name}}}_length <= std::size_t(std::numeric_limits<int>::max()));
    //
    {{/kernel_arguments}}

    // Kotekan buffer names
    {{#kernel_arguments}}
    const std::string {{{name}}}_memname;
    {{/kernel_arguments}}

    // Host-side buffer arrays
    {{#kernel_arguments}}
        {{^hasbuffer}}
            std::vector<std::vector<std::uint8_t>> {{{name}}}_host;
        {{/hasbuffer}}
    {{/kernel_arguments}}
};

REGISTER_CUDA_COMMAND(cuda{{{kernel_name}}});

cuda{{{kernel_name}}}::cuda{{{kernel_name}}}(Config& config,
                                             const std::string& unique_name,
                                             bufferContainer& host_buffers,
                                             cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "{{{kernel_name}}}", "{{{kernel_name}}}.ptx")
    {{#kernel_arguments}}
        {{#hasbuffer}}
            , {{{name}}}_memname(config.get<std::string>(unique_name, "{{{kotekan_name}}}"))
        {{/hasbuffer}}
        {{^hasbuffer}}
            , {{{name}}}_memname(unique_name + "/{{{kotekan_name}}}")
        {{/hasbuffer}}
    {{/kernel_arguments}}

    {{#kernel_arguments}}
        {{^hasbuffer}}
            , {{{name}}}_host(_gpu_buffer_depth)
        {{/hasbuffer}}
    {{/kernel_arguments}}
{
    // Add Graphviz entries for the GPU buffers used by this kernel
    {{#kernel_arguments}}
        {{#hasbuffer}}
            gpu_buffers_used.push_back(std::make_tuple({{{name}}}_memname, true, true, false));
        {{/hasbuffer}}
        {{^hasbuffer}}
            gpu_buffers_used.push_back(std::make_tuple(get_name() + "_{{{kotekan_name}}}", false, true, true));
        {{/hasbuffer}}
    {{/kernel_arguments}}

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

    {{#kernel_arguments}}
        {{#hasbuffer}}
            void* const {{{name}}}_memory =
                args::{{{name}}} == args::E || args::{{{name}}} == args::Ebar
                ? device.get_gpu_memory_array({{{name}}}_memname, pipestate.gpu_frame_id, {{{name}}}_length / 2)
                : device.get_gpu_memory_array({{{name}}}_memname, pipestate.gpu_frame_id, {{{name}}}_length);
        {{/hasbuffer}}
        {{^hasbuffer}}
        {{{name}}}_host[pipestate.gpu_frame_id].resize({{{name}}}_length);
            void* const {{{name}}}_memory = device.get_gpu_memory({{{name}}}_memname, {{{name}}}_length);
        {{/hasbuffer}}
    {{/kernel_arguments}}

    {{#kernel_arguments}}
        {{#hasbuffer}}
            {{^isoutput}}
                /// {{{name}}} is an input buffer: check metadata
                const metadataContainer* const {{{name}}}_mc =
                    device.get_gpu_memory_array_metadata({{{name}}}_memname, pipestate.gpu_frame_id);
                assert({{{name}}}_mc && metadata_container_is_chord({{{name}}}_mc));
                const chordMetadata* const {{{name}}}_meta = get_chord_metadata({{{name}}}_mc);
                INFO("input {{{name}}} array: {:s} {:s}",
                    {{{name}}}_meta->get_type_string(),
                    {{{name}}}_meta->get_dimensions_string());
                assert({{{name}}}_meta->type == {{{name}}}_type);
                assert({{{name}}}_meta->dims == {{{name}}}_rank);
                for (std::size_t dim = 0; dim < {{{name}}}_rank; ++dim) {
                    assert(std::strncmp({{{name}}}_meta->dim_name[dim],
                                        {{{name}}}_labels[{{{name}}}_rank - 1 - dim],
                                        sizeof {{{name}}}_meta->dim_name[dim]) == 0);
                    if (args::{{{name}}} == args::E)
                        assert({{{name}}}_meta->dim[dim] <= int({{{name}}}_lengths[{{{name}}}_rank - 1 - dim]));
                    else
                        assert({{{name}}}_meta->dim[dim] == int({{{name}}}_lengths[{{{name}}}_rank - 1 - dim]));
                }
                //
            {{/isoutput}}
            {{#isoutput}}
                /// {{{name}}} is an output buffer: set metadata
                metadataContainer* const {{{name}}}_mc =
                    device.create_gpu_memory_array_metadata({{{name}}}_memname, pipestate.gpu_frame_id, E_mc->parent_pool);
                chordMetadata* const {{{name}}}_meta = get_chord_metadata({{{name}}}_mc);
                chord_metadata_copy({{{name}}}_meta, E_meta);
                {{{name}}}_meta->type = {{{name}}}_type;
                {{{name}}}_meta->dims = {{{name}}}_rank;
                for (std::size_t dim = 0; dim < {{{name}}}_rank; ++dim) {
                    std::strncpy({{{name}}}_meta->dim_name[dim],
                                 {{{name}}}_labels[{{{name}}}_rank - 1 - dim],
                                 sizeof {{{name}}}_meta->dim_name[dim]);
                    {{{name}}}_meta->dim[dim] = {{{name}}}_lengths[{{{name}}}_rank - 1 - dim];
                }
                INFO("output {{{name}}} array: {:s} {:s}",
                    {{{name}}}_meta->get_type_string(),
                    {{{name}}}_meta->get_dimensions_string());
                //
            {{/isoutput}}
        {{/hasbuffer}}
    {{/kernel_arguments}}

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

    *(std::int32_t*)Tactual_host[pipestate.gpu_frame_id].data() = E_meta->dim[0];

    // Update input voltage data pointer:
    //
    // We need to re-process a small number of time samples, so we
    // move the pointer backwards in time by a bit, wrapping around if
    // necessary.
    //
    // Although we told Kotekan in the last iteration that we wouldn't
    // need these inputs again, it's fine to look at them again
    // because they won't be overwritten yet, if the GPU buffer depth
    // is large enough.

    const bool is_first_iteration = pipestate.get_int("gpu_frame_counter") == 0;
    INFO("is_first_iteration: {}", is_first_iteration);

    // We need an overlap of this many time samples
    const int needed_overlap = (cuda_number_of_taps - 1) * cuda_upchannelization_factor;
    INFO("needed_overlap: {}", needed_overlap);

    // The overlap for this iteration, because there is no overlap available in the first iteration
    const int overlap = is_first_iteration ? 0 : needed_overlap;
    INFO("overlap: {}", overlap);
    // The respective offset in bytes
    const std::ptrdiff_t offset = std::ptrdiff_t(1) * E_lengths[0] * E_lengths[1] * E_lengths[2] * overlap;
    INFO("offset: {}", offset);

    // Calculate the total ringbuffer size
    const int gpu_buffer_depth = config.get<int>(unique_name, "buffer_depth");
    INFO("gpu_buffer_depth: {}", gpu_buffer_depth);
    const std::ptrdiff_t ringbuffer_size = gpu_buffer_depth * E_length;
    INFO("ringbuffer_size: {}", ringbuffer_size);

    // Beginning of the ringbuffer
    void* const E_memory0 = device.get_gpu_memory_array(E_memname, 0, E_length / 2);
    INFO("E_memory0: {}", E_memory0);

    // New pointer
    INFO("E_memory: {}", E_memory);
    void* const new_E_memory =
        (char *)E_memory0 + ((char *)E_memory - (char *)E_memory0 - offset + ringbuffer_size) % ringbuffer_size;
    INFO("new_E_memory: {}", new_E_memory);
    assert(new_E_memory >= E_memory0 && (char *)new_E_memory < (char *)E_memory0 + ringbuffer_size);

    // New number of input time samples
    const int cuda_num_timesamples = E_meta->dim[0] + overlap;
    INFO("cuda_num_timesamples: {}", cuda_num_timesamples);
    assert(cuda_num_timesamples <= cuda_max_number_of_timesamples);

    // Number of output time samples
    assert((cuda_num_timesamples - needed_overlap) % cuda_upchannelization_factor == 0);
    const int cuda_num_output_timesamples = (cuda_num_timesamples - needed_overlap) / cuda_upchannelization_factor;
    INFO("cuda_num_output_timesamples: {}", cuda_num_output_timesamples);

    // Update metadata
    Ebar_meta->dim[0] = cuda_num_output_timesamples;
    assert(Ebar_meta->dim[0] <= int(Ebar_lengths[3]));

    // Update kernel arguments: new `E_memory` pointer and new total number of input time samples
    E_arg = kernel_arg(new_E_memory, E_length);
    *(std::int32_t*)Tactual_host[pipestate.gpu_frame_id].data() = cuda_num_timesamples;

    // Copy inputs to device memory
    // TODO: Pass scalar kernel arguments more efficiently, i.e. without a separate `cudaMemcpy`
    {{#kernel_arguments}}
    {{^hasbuffer}}
        {{^isoutput}}
            CHECK_CUDA_ERROR(cudaMemcpyAsync({{{name}}}_memory,
                                             {{{name}}}_host[pipestate.gpu_frame_id].data(),
                                             {{{name}}}_length,
                                             cudaMemcpyHostToDevice,
                                             device.getStream(cuda_stream_id)));
            {{/isoutput}}
        {{/hasbuffer}}
    {{/kernel_arguments}}

    // Initialize host-side buffer arrays
    // TODO: Skip this for performance
    {{#kernel_arguments}}
        {{^hasbuffer}}
            {{#isoutput}}
                CHECK_CUDA_ERROR(cudaMemsetAsync({{{name}}}_memory, 0xff, {{{name}}}_length, device.getStream(cuda_stream_id)));
            {{/isoutput}}
        {{/hasbuffer}}
    {{/kernel_arguments}}

    DEBUG("kernel_symbol: {}", kernel_symbol);
    DEBUG("runtime_kernels[kernel_symbol]: {}", static_cast<void*>(runtime_kernels[kernel_symbol]));
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_symbol],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA {{{kernel_name}}} on GPU frame {:d}", pipestate.gpu_frame_id);
    const CUresult err =
        cuLaunchKernel(runtime_kernels[kernel_symbol],
                       blocks, 1, 1, threads_x, threads_y, 1,
                       shmem_bytes,
                       device.getStream(cuda_stream_id),
                       args, NULL);

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        ERROR("cuLaunchKernel: Error number: {}: {}", err, errStr);
    }

    // Copy results back to host memory
    // TODO: Skip this for performance
    {{#kernel_arguments}}
        {{^hasbuffer}}
            {{#isoutput}}
                CHECK_CUDA_ERROR(cudaMemcpyAsync({{{name}}}_host[pipestate.gpu_frame_id].data(),
                                                 {{{name}}}_memory,
                                                 {{{name}}}_length,
                                                 cudaMemcpyDeviceToHost,
                                                 device.getStream(cuda_stream_id)));
            {{/isoutput}}
        {{/hasbuffer}}
    {{/kernel_arguments}}

    // Check error codes
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
    const std::int32_t error_code = *std::max_element((const std::int32_t*)&*info_host[pipestate.gpu_frame_id].begin(),
                                                      (const std::int32_t*)&*info_host[pipestate.gpu_frame_id].end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    return record_end_event(pipestate.gpu_frame_id);
}

void cuda{{{kernel_name}}}::finalize_frame(const int gpu_frame_id) {
    cudaCommand::finalize_frame(gpu_frame_id);

    for (std::size_t i = 0; i < info_host[gpu_frame_id].size(); ++i)
        if (info_host[gpu_frame_id][i] != 0)
            ERROR("cuda{{{kernel_name}}} returned 'info' value {:d} at index {:d} (zero indicates no error)",
                info_host[gpu_frame_id][i], i);
}
