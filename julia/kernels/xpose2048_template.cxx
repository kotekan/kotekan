/**
 * @file
 * @brief CUDA {{{kernel_name}}} kernel
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
#include <div.hpp>
#include <fmt.hpp>
#include <limits>
#include <ringbuffer.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::round_down, kotekan::div_noremainder, kotekan::div, kotekan::mod;

/**
 * @class cuda{{{kernel_name}}}
 * @brief cudaCommand for {{{kernel_name}}}
 */
class cuda{{{kernel_name}}} : public cudaCommand {
public:
    cuda{{{kernel_name}}}(Config & config, const std::string& unique_name,
                          bufferContainer& host_buffers, cudaDeviceInterface& device, const int instance_num);
    virtual ~cuda{{{kernel_name}}}();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>& pre_events) override;
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
            ptr(static_cast<T*>(ptr)),
            maxsize(bytes),
            dims{std::int64_t(maxsize / sizeof(T))},
            len(maxsize / sizeof(T)) {}
    };
    using array_desc = CuDeviceArray<int32_t, 1>;

    // Kernel design parameters:
    {{#kernel_design_parameters}}
        static constexpr {{{type}}} {{{name}}} = {{{value}}};
    {{/kernel_design_parameters}}

    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

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
        static constexpr const char *{{{name}}}_name = "{{{name}}}";
        static constexpr chordDataType {{{name}}}_type = {{{type}}};
        {{^isscalar}}
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
            static constexpr std::array<std::ptrdiff_t, {{{name}}}_rank> {{{name}}}_lengths = {
                {{#axes}}
                    {{{length}}},
                {{/axes}}
            };
            static constexpr std::ptrdiff_t {{{name}}}_length = chord_datatype_bytes({{{name}}}_type)
                {{#axes}}
                    * {{{length}}}
                {{/axes}}
                ;
            static_assert({{{name}}}_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
            static constexpr auto {{{name}}}_calc_stride = [](int dim) {
                std::ptrdiff_t str = 1;
                for (int d = 0; d < dim; ++d)
                    str *= {{{name}}}_lengths[d];
                return str;
            };
            static constexpr std::array<std::ptrdiff_t, {{{name}}}_rank + 1> {{{name}}}_strides = {
                {{#axes}}
                    {{{name}}}_calc_stride({{{name}}}_index_{{{label}}}),
                {{/axes}}
                {{{name}}}_calc_stride({{{name}}}_rank),
            };
            // static_assert({{{name}}}_length == chord_datatype_bytes({{{name}}}_type) * {{{name}}}_strides[{{{name}}}_rank]);
        {{/isscalar}}
        //
    {{/kernel_arguments}}

    // Kotekan buffer names
    {{#kernel_arguments}}
        {{^isscalar}}
            const std::string {{{name}}}_memname;
        {{/isscalar}}
    {{/kernel_arguments}}

    // Host-side buffer arrays
    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                std::vector<std::uint8_t> {{{name}}}_host;
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    static constexpr std::ptrdiff_t Ein_T_sample_bytes =
        chord_datatype_bytes(Ein_type) * Ein_lengths[Ein_index_D] * Ein_lengths[Ein_index_P] * Ein_lengths[Ein_index_F];
    static constexpr std::ptrdiff_t E_T_sample_bytes =
        chord_datatype_bytes(E_type) * E_lengths[E_index_D] * E_lengths[E_index_P] * E_lengths[E_index_F];

    RingBuffer* input_ringbuf_signal;
    RingBuffer* output_ringbuf_signal;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tinmin, Tinmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tmin, Tmax;
};

REGISTER_CUDA_COMMAND(cuda{{{kernel_name}}});

cuda{{{kernel_name}}}::cuda{{{kernel_name}}}(Config& config,
                                             const std::string& unique_name,
                                             bufferContainer& host_buffers,
                                             cudaDeviceInterface& device,
                                             const int instance_num):
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
        "{{{kernel_name}}}", "{{{kernel_name}}}.ptx"),
    {{#kernel_arguments}}
        {{^isscalar}}
            {{#hasbuffer}}
                {{{name}}}_memname(config.get<std::string>(unique_name, "{{{kotekan_name}}}")),
            {{/hasbuffer}}
            {{^hasbuffer}}
                {{{name}}}_memname(unique_name + "/{{{kotekan_name}}}"),
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                {{{name}}}_host({{{name}}}_length),
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}
    // Find input and output buffers used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")))),
    output_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "out_signal"))))
{
    // Check ringbuffer sizes
    assert(input_ringbuf_signal->size == Ein_length);
    assert(output_ringbuf_signal->size == E_length);

    // Register host memory
    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                {
                    const cudaError_t ierr = cudaHostRegister({{{name}}}_host.data(), {{{name}}}_host.size(), 0);
                    assert(ierr == cudaSuccess);
                }
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    // Add Graphviz entries for the GPU buffers used by this kernel
    {{#kernel_arguments}}
        {{^isscalar}}
            {{#hasbuffer}}
                gpu_buffers_used.push_back(std::make_tuple({{{name}}}_memname, true, true, false));
            {{/hasbuffer}}
            {{^hasbuffer}}
                gpu_buffers_used.push_back(std::make_tuple(get_name() + "_{{{kotekan_name}}}", false, true, true));
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage need to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "{{{kernel_name}}}_");
    }

    if (instance_num == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
        output_ringbuf_signal->register_producer(unique_name);
        output_ringbuf_signal->allocate_new_metadata_object(0);
    }
}

cuda{{{kernel_name}}}::~cuda{{{kernel_name}}}() {}

std::int64_t cuda{{{kernel_name}}}::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements);
}
std::int64_t cuda{{{kernel_name}}}::num_produced_elements(std::int64_t num_available_elements) const {
    return num_consumed_elements(num_available_elements);
}

std::int64_t cuda{{{kernel_name}}}::num_processed_elements(std::int64_t num_available_elements) const {
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cuda{{{kernel_name}}}::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::ptrdiff_t> val_in1 = input_ringbuf_signal->wait_without_claiming(unique_name, instance_num);
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    if (!val_in1.has_value())
        return -1;
    const std::ptrdiff_t input_bytes = val_in1.value();
    DEBUG("Input ring-buffer byte count: {:d}", input_bytes);

    // How many inputs samples are available?
    const std::ptrdiff_t Tin_available = div_noremainder(input_bytes, Ein_T_sample_bytes);
    DEBUG("Available samples:      Tin_available: {:d}", Tin_available);

    // How many outputs will we process and consume?
    const std::ptrdiff_t Tin_processed = num_processed_elements(Tin_available);
    const std::ptrdiff_t Tin_consumed = num_consumed_elements(Tin_available);
    DEBUG("Will process (samples): Tin_processed: {:d}", Tin_processed);
    DEBUG("Will consume (samples): Tin_consumed:  {:d}", Tin_consumed);
    assert(Tin_processed > 0);
    assert(Tin_consumed <= Tin_processed);
    const std::ptrdiff_t Tin_consumed2 = num_consumed_elements(Tin_processed);
    assert(Tin_consumed2 == Tin_consumed);

    const std::optional<std::ptrdiff_t> val_in2 =
        input_ringbuf_signal->wait_and_claim_readable(unique_name, instance_num, Tin_consumed * Ein_T_sample_bytes);
    if (!val_in2.has_value())
        return -1;
    const std::ptrdiff_t input_cursor = val_in2.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    Tinmin = div_noremainder(input_cursor, Ein_T_sample_bytes);
    Tinmax = Tinmin + Tin_processed;
    const std::ptrdiff_t Tinlength = Tinmax - Tinmin;
    DEBUG("Input samples:");
    DEBUG("    Tinmin:    {:d}", Tinmin);
    DEBUG("    Tinmax:    {:d}", Tinmax);
    DEBUG("    Tinlength: {:d}", Tinlength);

    // How many outputs will we produce?
    const std::ptrdiff_t T_produced = num_produced_elements(Tin_available);
    DEBUG("Will produce (samples): T_produced: {:d}", T_produced);
    const std::ptrdiff_t Tlength = T_produced;

    // to bytes
    const std::ptrdiff_t output_bytes =
        Tlength * E_T_sample_bytes;
    DEBUG("Will produce {:d} output bytes", output_bytes);

    // Wait for space to be available in our output ringbuffer...
    DEBUG("Waiting for output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::optional<std::ptrdiff_t> val_out =
        output_ringbuf_signal->wait_for_writable(unique_name, instance_num, output_bytes);
    DEBUG("Finished waiting for output for data frame {:d}.", gpu_frame_id);
    if (!val_out.has_value())
        return -1;
    const std::ptrdiff_t output_cursor = val_out.value();
    DEBUG("Output ring-buffer byte offset {:d}", output_cursor);

    assert(mod(output_cursor, E_T_sample_bytes) == 0);
    Tmin = output_cursor / E_T_sample_bytes;
    Tmax = Tmin + Tlength;
    DEBUG("Output samples:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);
    DEBUG("    Tlength: {:d}", Tlength);

    return 0;
}

cudaEvent_t cuda{{{kernel_name}}}::execute(cudaPipelineState& /*pipestate*/, const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    {{#kernel_arguments}}
        {{^isscalar}}
            {{#hasbuffer}}
                void* const {{{name}}}_memory =
                    args::{{{name}}} == args::Ein ?
                        device.get_gpu_memory({{{name}}}_memname, input_ringbuf_signal->size) :
                    args::{{{name}}} == args::E ?
                        device.get_gpu_memory({{{name}}}_memname, output_ringbuf_signal->size) :
                    args::{{{name}}} == args::scatter_indices ?
                        device.get_gpu_memory({{{name}}}_memname, {{{name}}}_length) :
                        device.get_gpu_memory_array({{{name}}}_memname, gpu_frame_id, _gpu_buffer_depth, {{{name}}}_length);
            {{/hasbuffer}}
            {{^hasbuffer}}
                void* const {{{name}}}_memory = device.get_gpu_memory({{{name}}}_memname, {{{name}}}_length);
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    {{#kernel_arguments}}
        {{^isscalar}}
            {{#hasbuffer}}
                {{^isoutput}}
                    // {{{name}}} is an input buffer: check metadata
                    const std::shared_ptr<metadataObject> {{{name}}}_mc =
                        args::{{{name}}} == args::Ein ?
                            input_ringbuf_signal->get_metadata(0) :
                            device.get_gpu_memory_array_metadata({{{name}}}_memname, gpu_frame_id);
                    assert({{{name}}}_mc);
                    assert(metadata_is_chord({{{name}}}_mc));
                    const std::shared_ptr<chordMetadata> {{{name}}}_meta = get_chord_metadata({{{name}}}_mc);
                    DEBUG("input {{{name}}} array: {:s} {:s}",
                          {{{name}}}_meta->get_type_string(),
                          {{{name}}}_meta->get_dimensions_string());
                    if (args::{{{name}}} == args::Ein)
                        assert(std::strncmp({{{name}}}_meta->name, "E", sizeof {{{name}}}_meta->name) == 0);
                    else
                        assert(std::strncmp({{{name}}}_meta->name, {{{name}}}_name, sizeof {{{name}}}_meta->name) == 0);
                    assert({{{name}}}_meta->type == {{{name}}}_type);
                    assert({{{name}}}_meta->dims == {{{name}}}_rank);
                    for (std::ptrdiff_t dim = 0; dim < {{{name}}}_rank; ++dim) {
                        assert(std::strncmp({{{name}}}_meta->dim_name[{{{name}}}_rank - 1 - dim],
                                            {{{name}}}_labels[dim],
                                            sizeof {{{name}}}_meta->dim_name[{{{name}}}_rank - 1 - dim]) == 0);
                        if (args::{{{name}}} == args::Ein && dim == Ein_index_T) {
                            assert({{{name}}}_meta->dim[{{{name}}}_rank - 1 - dim] <= int({{{name}}}_lengths[dim]));
                            assert({{{name}}}_meta->stride[{{{name}}}_rank - 1 - dim] <= {{{name}}}_strides[dim]);
                        } else {
                            assert({{{name}}}_meta->dim[{{{name}}}_rank - 1 - dim] == int({{{name}}}_lengths[dim]));
                            assert({{{name}}}_meta->stride[{{{name}}}_rank - 1 - dim] == {{{name}}}_strides[dim]);
                        }
                    }
                    //
                {{/isoutput}}
                {{#isoutput}}
                    // {{{name}}} is an output buffer: set metadata
                    std::shared_ptr<metadataObject> const {{{name}}}_mc =
                        args::{{{name}}} == args::E ?
                            output_ringbuf_signal->get_metadata(0) :
                            device.create_gpu_memory_array_metadata({{{name}}}_memname, gpu_frame_id, E_mc->parent_pool);
                    std::shared_ptr<chordMetadata> const {{{name}}}_meta = get_chord_metadata({{{name}}}_mc);
                    *{{{name}}}_meta = *Ein_meta;
                    std::strncpy({{{name}}}_meta->name, {{{name}}}_name, sizeof {{{name}}}_meta->name);
                    {{{name}}}_meta->type = {{{name}}}_type;
                    {{{name}}}_meta->dims = {{{name}}}_rank;
                    for (std::ptrdiff_t dim = 0; dim < {{{name}}}_rank; ++dim) {
                        std::strncpy({{{name}}}_meta->dim_name[{{{name}}}_rank - 1 - dim],
                                     {{{name}}}_labels[dim],
                                     sizeof {{{name}}}_meta->dim_name[{{{name}}}_rank - 1 - dim]);
                        {{{name}}}_meta->dim[{{{name}}}_rank - 1 - dim] = {{{name}}}_lengths[dim];
                        {{{name}}}_meta->stride[{{{name}}}_rank - 1 - dim] = {{{name}}}_strides[dim];
                    }
                    DEBUG("output {{{name}}} array: {:s} {:s}",
                          {{{name}}}_meta->get_type_string(),
                          {{{name}}}_meta->get_dimensions_string());
                    //
                {{/isoutput}}
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    record_start_event();

    DEBUG("gpu_frame_id: {}", gpu_frame_id);

    const char* exc_arg = "exception";
    {{#kernel_arguments}}
        {{^isscalar}}
            array_desc {{{name}}}_arg({{{name}}}_memory, {{{name}}}_length);
        {{/isscalar}}
        {{#isscalar}}
            std::{{{type}}}_t {{{name}}}_arg;
        {{/isscalar}}
    {{/kernel_arguments}}
    void* args[] = {
        &exc_arg,
        {{#kernel_arguments}}
            &{{{name}}}_arg,
        {{/kernel_arguments}}
    };

    // Set Ein_memory to beginning of input ring buffer
    Ein_arg = array_desc(Ein_memory, Ein_length);

    // Set E_memory to beginning of output ring buffer
    E_arg = array_desc(E_memory, E_length);

    // Ringbuffer size
    const std::ptrdiff_t Tin_ringbuf = input_ringbuf_signal->size / Ein_T_sample_bytes;
    const std::ptrdiff_t T_ringbuf = output_ringbuf_signal->size / E_T_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", Tin_ringbuf);
    DEBUG("Output ringbuffer size (samples): {:d}", T_ringbuf);

    const std::ptrdiff_t Tinlength = Tinmax - Tinmin;
    const std::ptrdiff_t Tlength = Tmax - Tmin;
    DEBUG("Processed input samples: {:d}", Tinlength);
    DEBUG("Produced output samples: {:d}", Tlength);

    DEBUG("Kernel arguments:");
    DEBUG("    Tinmin: {:d}", Tinmin);
    DEBUG("    Tinmax: {:d}", Tinmax);
    DEBUG("    Tmin:   {:d}", Tmin);
    DEBUG("    Tmax:   {:d}", Tmax);

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    Tinmin_arg = mod(Tinmin, Tin_ringbuf);
    Tinmax_arg = mod(Tinmin, Tin_ringbuf) + Tinlength;
    Tmin_arg = mod(Tmin, T_ringbuf);
    Tmax_arg = mod(Tmin, T_ringbuf) + Tlength;

    // Update metadata
    E_meta->dim[E_rank - 1 - E_index_T] = Tlength;
    assert(E_meta->dim[E_rank - 1 - E_index_T] <= int(E_lengths[E_index_T]));
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    // Copy inputs to device memory
    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                {{^isoutput}}
                    CHECK_CUDA_ERROR(cudaMemcpyAsync({{{name}}}_memory,
                                                     {{{name}}}_host.data(),
                                                     {{{name}}}_length,
                                                     cudaMemcpyHostToDevice,
                                                     device.getStream(cuda_stream_id)));
                {{/isoutput}}
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

#ifdef DEBUGGING
    // Initialize host-side buffer arrays
    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                {{#isoutput}}
                    CHECK_CUDA_ERROR(cudaMemsetAsync({{{name}}}_memory, 0xff, {{{name}}}_length, device.getStream(cuda_stream_id)));
                {{/isoutput}}
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}
#endif

#ifdef DEBUGGING
    // Poison outputs
    {
        DEBUG("begin poisoning");
        const int num_chunks = Tmax_arg <= T_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            DEBUG("poisoning chunk={}/{}", chunk, num_chunks);
            const std::ptrdiff_t Tstride = E_meta->stride[0];
            const std::ptrdiff_t Toffset = chunk == 0 ? Tmin_arg : 0;
            const std::ptrdiff_t Tlength = (num_chunks == 1 ?
                                            Tmax_arg - Tmin_arg :
                                            chunk == 0 ? T_ringbuf - Tmin_arg : Tmax_arg - T_ringbuf);
            DEBUG("before cudaMemset2DAsync.E");
            CHECK_CUDA_ERROR(cudaMemsetAsync((std::uint8_t*)E_memory + Toffset * Tstride,
                                             0x00,
                                             Tlength * Tstride,
                                             device.getStream(cuda_stream_id)));
        } // for chunk
        DEBUG("poisoning done.");
    }
#endif

    const std::string symname = "{{{kernel_name}}}_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA {{{kernel_name}}} on GPU frame {:d}", gpu_frame_id);
    const CUresult err =
        cuLaunchKernel(device.runtime_kernels[symname],
                       blocks, 1, 1, threads_x, threads_y, 1,
                       shmem_bytes,
                       device.getStream(cuda_stream_id),
                       args, NULL);

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        ERROR("cuLaunchKernel: Error number: {}: {}", (int)err, errStr);
    }

#ifdef DEBUGGING
    // Copy results back to host memory
    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                {{#isoutput}}
                    CHECK_CUDA_ERROR(cudaMemcpyAsync({{{name}}}_host.data(),
                                                     {{{name}}}_memory,
                                                     {{{name}}}_length,
                                                     cudaMemcpyDeviceToHost,
                                                     device.getStream(cuda_stream_id)));
                {{/isoutput}}
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
    DEBUG("Finished CUDA {{{kernel_name}}} on GPU frame {:d}", gpu_frame_id);

    // Check error codes
    const std::int32_t error_code = *std::max_element((const std::int32_t*)&*info_host.begin(),
                                                      (const std::int32_t*)&*info_host.end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    for (std::size_t i = 0; i < info_host.size(); ++i)
        if (info_host[i] != 0)
            ERROR("cuda{{{kernel_name}}} returned 'info' value {:d} at index {:d} (zero indicates no error)",
                info_host[i], i);
#endif

#ifdef DEBUGGING
    // Check outputs for poison
    {
        DEBUG("begin poison check");
        DEBUG("    Ein_dims={}", Ein_meta->dims);
        DEBUG("    Ein_dim[0]={}", Ein_meta->dim[0]);
        DEBUG("    Ein_dim[1]={}", Ein_meta->dim[1]);
        DEBUG("    Ein_dim[2]={}", Ein_meta->dim[2]);
        DEBUG("    Ein_dim[3]={}", Ein_meta->dim[3]);
        DEBUG("    Ein_stride[0]={}", Ein_meta->stride[0]);
        DEBUG("    Ein_stride[1]={}", Ein_meta->stride[1]);
        DEBUG("    Ein_stride[2]={}", Ein_meta->stride[2]);
        DEBUG("    Ein_stride[3]={}", Ein_meta->stride[3]);
        DEBUG("    E_dims={}", E_meta->dims);
        DEBUG("    E_dim[0]={}", E_meta->dim[0]);
        DEBUG("    E_dim[1]={}", E_meta->dim[1]);
        DEBUG("    E_dim[2]={}", E_meta->dim[2]);
        DEBUG("    E_dim[3]={}", E_meta->dim[3]);
        DEBUG("    E_stride[0]={}", E_meta->stride[0]);
        DEBUG("    E_stride[1]={}", E_meta->stride[1]);
        DEBUG("    E_stride[2]={}", E_meta->stride[2]);
        DEBUG("    E_stride[3]={}", E_meta->stride[3]);
        const int num_chunks = Tmax_arg <= T_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            DEBUG("poisoning chunk={}/{}", chunk, num_chunks);
            const std::ptrdiff_t Tstride = E_meta->stride[0];
            const std::ptrdiff_t Toffset = chunk == 0 ? Tmin_arg : 0;
            const std::ptrdiff_t Tlength = (num_chunks == 1 ?
                                            Tmax_arg - Tmin_arg :
                                            chunk == 0 ? T_ringbuf - Tmin_arg : Tmax_arg - T_ringbuf);
            DEBUG("    Tstride={}", Tstride);
            DEBUG("    Toffset={}", Toffset);
            DEBUG("    Tlength={}", Tlength);
            std::vector<std::uint8_t> E_buffer(Tlength * Tstride, 0x11);
            DEBUG("    E_buffer.size={}", E_buffer.size());
            DEBUG("before cudaMemcpy.E");
            CHECK_CUDA_ERROR(cudaMemcpy(E_buffer.data(),
                                        (const std::uint8_t*)E_memory + Toffset * Tstride,
                                        Tlength * Tstride,
                                        cudaMemcpyDeviceToHost));

            DEBUG("before memchr");
            const bool E_found_error = std::memchr(E_buffer.data(), 0x00, E_buffer.size());
            if (E_found_error) {
                for (std::ptrdiff_t t=0; t<Tlength; ++t) {
                  bool any_error = false;
                  for (std::ptrdiff_t n=0; n<Tstride; ++n) {
                    const auto val = E_buffer.at(t * Tstride + n); 
                    any_error |= val == 0x00;
                  }
                  if (any_error)
                    DEBUG("    [{}]={:#02x}", t, 0x00);
                }
            }
            assert(!E_found_error);
        } // for chunk
        DEBUG("poison check done.");
    }
#endif

    return record_end_event();
}

void cuda{{{kernel_name}}}::finalize_frame() {
    const std::ptrdiff_t Tinlength = Tinmax - Tinmin;
    const std::ptrdiff_t Tlength = Tmax - Tmin;

    // Advance the input ringbuffer
    const std::ptrdiff_t Tin_consumed = num_consumed_elements(Tinlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples: {:d}", Tin_consumed);
    DEBUG("    Consumed bytes:   {:d}", Tin_consumed * Ein_T_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, instance_num, Tin_consumed * Ein_T_sample_bytes);

    // Advance the output ringbuffer
    const std::ptrdiff_t T_produced = Tlength;
    DEBUG("Advancing output ringbuffer:");
    DEBUG("    Produced samples: {:d}", T_produced);
    DEBUG("    Produced bytes:   {:d}", T_produced * E_T_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, instance_num, T_produced * E_T_sample_bytes);

    cudaCommand::finalize_frame();
}
