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
#include <mutex>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

namespace {
    // Round down `x` to the next lower multiple of `y`
    template<typename T, typename U>
    auto round_down(T x, U y) {
        assert(x >= 0);
        assert(y > 0);
        auto r = x / y * y;
        assert(r % y == 0);
        assert(0 <= r && r <= x && r + y > x);
        return r;
    }

    // Calculate `x mod y`, returning `x` with `0 <= x < y`
    template<typename T, typename U>
    auto mod(T x, U y) {
        assert(y > 0);
        while (x < 0)
            x += y;
        auto r = x % y;
        assert(0 <= r && r < y);
        return r;
    }
}

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
    static_assert({{{name}}}_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
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

    // Loop-carried information

    // How many time samples from the previous iteration still need to be processed (or processed again)?
    std::int64_t unprocessed;
    // How many time samples were not provided in the previous iteration and need to be provided in this iteration?
    std::int64_t unprovided;
};

REGISTER_CUDA_COMMAND(cuda{{{kernel_name}}});

cuda{{{kernel_name}}}::cuda{{{kernel_name}}}(Config& config,
                                             const std::string& unique_name,
                                             bufferContainer& host_buffers,
                                             cudaDeviceInterface& device):
    cudaCommand(config, unique_name, host_buffers, device, "{{{kernel_name}}}", "{{{kernel_name}}}.ptx"),
    {{#kernel_arguments}}
        {{#hasbuffer}}
            {{{name}}}_memname(config.get<std::string>(unique_name, "{{{kotekan_name}}}")),
        {{/hasbuffer}}
        {{^hasbuffer}}
            {{{name}}}_memname(unique_name + "/{{{kotekan_name}}}"),
        {{/hasbuffer}}
    {{/kernel_arguments}}

    {{#kernel_arguments}}
        {{^hasbuffer}}
            {{{name}}}_host(_gpu_buffer_depth),
        {{/hasbuffer}}
    {{/kernel_arguments}}
    unprocessed(0),
    unprovided(0)
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

    // // Create a ring buffer. Create it only once.
    // assert(E_length % _gpu_buffer_depth == 0);
    // const std::ptrdiff_t E_buffer_size = E_length / _gpu_buffer_depth;
    // const std::ptrdiff_t E_ringbuffer_size = E_length;
    // // if (cuda_upchannelization_factor == 128)
    //     device.create_gpu_memory_ringbuffer(E_memname + "_ringbuffer", E_ringbuffer_size,
    //                                         E_memname, 0, E_buffer_size);
}

cuda{{{kernel_name}}}::~cuda{{{kernel_name}}}() {}

cudaEvent_t cuda{{{kernel_name}}}::execute(cudaPipelineState& pipestate,
                                           const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute(pipestate.gpu_frame_id);

    {{#kernel_arguments}}
        {{#hasbuffer}}
            void* const {{{name}}}_memory =
                args::{{{name}}} == args::E || args::{{{name}}} == args::Ebar
                ? device.get_gpu_memory_array({{{name}}}_memname, pipestate.gpu_frame_id, {{{name}}}_length / _gpu_buffer_depth)
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

    // We need to (re-)process some time samples, from the previous
    // iteration (`unprocessed`), and we still to provide some time
    // samples that were not provided in the previous iteration
    // (`unprovided`).

    // Although we told Kotekan in the last iteration that we wouldn't
    // need these inputs again, it's fine to look at them again
    // because they won't be overwritten yet, if the GPU buffer depth
    // is large enough.

    // We need an overlap of this many time samples, i.e. this many
    // time samples will need to be re-processed in the next
    // iteration. This also defines the number of time samples that
    // will be missing from the output.
    INFO("cuda_algorithm_overlap: {}", cuda_algorithm_overlap);
    // The input is processed in batches of this size. This means that
    // the input we provide must be a multiple of this number.
    INFO("cuda_granularity_number_of_timesamples: {}", cuda_granularity_number_of_timesamples);

    INFO("gpu_frame_id: {}", pipestate.gpu_frame_id);

    // Beginning of the input ringbuffer
    void* const E_memory0 = device.get_gpu_memory_array(E_memname, 0, E_length / _gpu_buffer_depth);
    INFO("E_memory0: {}", E_memory0);

    // Beginning of the output ringbuffer
    void* const Ebar_memory0 = device.get_gpu_memory_array(Ebar_memname, 0, Ebar_length / _gpu_buffer_depth);
    INFO("Ebar_memory0: {}", Ebar_memory0);

    // Set E_memory to beginning of input ring buffer
    E_arg = kernel_arg(E_memory0, E_length);

    // Set Ebar_memory to beginning of output ring buffer
    Ebar_arg = kernel_arg(Ebar_memory0, Ebar_length);

    // Current nominal index into input ringuffer
    const std::int64_t nominal_Tmin = pipestate.gpu_frame_id * cuda_max_number_of_timesamples / _gpu_buffer_depth;
    INFO("nominal Tmin: {}", nominal_Tmin);

    // Current nominal index into output ringuffer
    const std::int64_t nominal_Tbarmin =
        pipestate.gpu_frame_id * cuda_max_number_of_timesamples / cuda_upchannelization_factor / _gpu_buffer_depth;
    INFO("nominal Tbarmin: {}", nominal_Tbarmin);

    // Current unprocessed time samples
    INFO("unprocessed: {}", unprocessed);

    // Current unprovided time samples
    INFO("unprovided: {}", unprovided);

    // Actual index into input ringbuffer
    const std::int64_t Tmin = nominal_Tmin - unprocessed;
    INFO("Tmin: {}", Tmin);

    // Actual index into output ringbuffer
    const std::int64_t Tbarmin = nominal_Tbarmin - unprovided;
    INFO("Tbarmin: {}", Tbarmin);

    // Nominal end of input time span (given by available data)
    const std::int64_t nominal_Tmax = nominal_Tmin + E_meta->dim[0];
    INFO("nominal Tmax: {}", nominal_Tmax);

    // We cannot process all time samples because the input size needs
    // to be a multiple of `cuda_granularity_number_of_timesamples`.
    const std::int64_t nominal_Tlength = nominal_Tmax - Tmin;
    INFO("nominal Tlength: {}", nominal_Tlength);
    const std::int64_t Tlength = round_down(nominal_Tlength, cuda_granularity_number_of_timesamples);
    INFO("Tlength: {}", Tlength);

    // End of input time span
    const std::int64_t Tmax = Tmin + Tlength;
    INFO("Tmax: {}", Tmax);

    // Output time span (defined by input time span length)
    assert(Tlength % cuda_upchannelization_factor == 0);
    assert(cuda_algorithm_overlap % cuda_upchannelization_factor == 0);
    assert(Tlength >= cuda_algorithm_overlap);
    const std::int64_t Tbarlength = (Tlength - cuda_algorithm_overlap) / cuda_upchannelization_factor;
    INFO("Tbarlength: {}", Tbarlength);

    // End of output time span
    const std::int64_t Tbarmax = Tbarmin + Tbarlength;
    INFO("Tbarmax: {}", Tbarmax);

    // Wrap lower bounds into ringbuffer
    const std::int64_t Tmin_wrapped = mod(Tmin, cuda_max_number_of_timesamples);
    const std::int64_t Tmax_wrapped = Tmin_wrapped + Tmax - Tmin;
    const std::int64_t Tbarmin_wrapped = mod(Tbarmin, cuda_max_number_of_timesamples / cuda_upchannelization_factor);
    const std::int64_t Tbarmax_wrapped = Tbarmin_wrapped + Tbarmax - Tbarmin;
    INFO("Tmin_wrapped: {}", Tmin_wrapped);
    INFO("Tmax_wrapped: {}", Tmax_wrapped);
    INFO("Tbarmin_wrapped: {}", Tbarmin_wrapped);
    INFO("Tbarmax_wrapped: {}", Tbarmax_wrapped);

    assert(Tmin_wrapped >= 0 && Tmin_wrapped <= Tmax_wrapped && Tmax_wrapped <= std::numeric_limits<int32_t>::max());
    assert(Tbarmin_wrapped >= 0 && Tbarmin_wrapped <= Tbarmax_wrapped && Tbarmax_wrapped <= std::numeric_limits<int32_t>::max());

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    *(std::int32_t*)Tmin_host[pipestate.gpu_frame_id].data() = Tmin_wrapped;
    *(std::int32_t*)Tmax_host[pipestate.gpu_frame_id].data() = Tmax_wrapped;
    *(std::int32_t*)Tbarmin_host[pipestate.gpu_frame_id].data() = Tbarmin_wrapped;
    *(std::int32_t*)Tbarmax_host[pipestate.gpu_frame_id].data() = Tbarmax_wrapped;

    // Update metadata
    Ebar_meta->dim[0] = Tbarlength - unprovided;
    assert(Ebar_meta->dim[0] <= int(Ebar_lengths[3]));

    // Calculate the number of  unprocessed time samples for the next iteration
    unprocessed -= Tlength - cuda_algorithm_overlap - cuda_max_number_of_timesamples / _gpu_buffer_depth;
    unprovided -= Tbarlength - cuda_max_number_of_timesamples / cuda_upchannelization_factor / _gpu_buffer_depth;
    INFO("new unprocessed: {}", unprocessed);
    INFO("new unprovided: {}", unprovided);
    assert(unprocessed >= 0 && unprocessed < cuda_max_number_of_timesamples / _gpu_buffer_depth);
    assert(unprovided >= 0 && unprovided < cuda_max_number_of_timesamples / cuda_upchannelization_factor / _gpu_buffer_depth);

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

    // Initialize outputs
    //     0x88 = (-8,-8), an unused value to detect uninitialized output
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(cudaMemsetAsync(Ebar_memory, 0x88ff, Ebar_length, device.getStream(cuda_stream_id)));

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
    {{#kernel_arguments}}
        {{#hasbuffer}}
            device.release_gpu_memory_array_metadata({{{name}}}_memname, gpu_frame_id);
        {{/hasbuffer}}
    {{/kernel_arguments}}

    for (std::size_t i = 0; i < info_host[gpu_frame_id].size(); ++i)
        if (info_host[gpu_frame_id][i] != 0)
            ERROR("cuda{{{kernel_name}}} returned 'info' value {:d} at index {:d} (zero indicates no error)",
                info_host[gpu_frame_id][i], i);

    cudaCommand::finalize_frame(gpu_frame_id);
}
