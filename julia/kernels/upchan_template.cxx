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
                          bufferContainer& host_buffers, cudaDeviceInterface& device, const int inst);
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
    using kernel_arg = CuDeviceArray<int32_t, 1>;

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
            std::vector<std::uint8_t> {{{name}}}_host;
        {{/hasbuffer}}
    {{/kernel_arguments}}

    static constexpr std::size_t E_T_sample_bytes =
        chord_datatype_bytes(E_type) * E_lengths[E_index_D] * E_lengths[E_index_P] * E_lengths[E_index_F];
    static constexpr std::size_t Ebar_Tbar_sample_bytes =
         chord_datatype_bytes(Ebar_type) * Ebar_lengths[Ebar_index_D] * Ebar_lengths[Ebar_index_P] * Ebar_lengths[Ebar_index_Fbar];

    RingBuffer* input_ringbuf_signal;
    RingBuffer* output_ringbuf_signal;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::size_t Tmin, Tmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::size_t Tbarmin, Tbarmax;
};

REGISTER_CUDA_COMMAND(cuda{{{kernel_name}}});

cuda{{{kernel_name}}}::cuda{{{kernel_name}}}(Config& config,
                                             const std::string& unique_name,
                                             bufferContainer& host_buffers,
                                             cudaDeviceInterface& device,
                                             const int inst):
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
        "{{{kernel_name}}}", "{{{kernel_name}}}.ptx"),
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
            {{{name}}}_host({{{name}}}_length),
        {{/hasbuffer}}
    {{/kernel_arguments}}
    // Find input and output buffers used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")))),
    output_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "out_signal"))))
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

    // Only one of the instances of this pipeline stage needs to build the kernel
    if (inst == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "{{{kernel_name}}}_");
    }

    if (inst == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
        output_ringbuf_signal->register_producer(unique_name);
	output_ringbuf_signal->allocate_new_metadata_object(0);
    }
}

cuda{{{kernel_name}}}::~cuda{{{kernel_name}}}() {}

std::int64_t cuda{{{kernel_name}}}::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) - cuda_algorithm_overlap;
}
std::int64_t cuda{{{kernel_name}}}::num_produced_elements(std::int64_t num_available_elements) const {
    assert(num_consumed_elements(num_available_elements) % cuda_upchannelization_factor == 0);
    return num_consumed_elements(num_available_elements) / cuda_upchannelization_factor;
}

std::int64_t cuda{{{kernel_name}}}::num_processed_elements(std::int64_t num_available_elements) const {
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cuda{{{kernel_name}}}::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::size_t> val_in1 =
        input_ringbuf_signal->wait_without_claiming(unique_name);
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    if (!val_in1.has_value())
        return -1;
    const std::size_t input_bytes = val_in1.value();
    DEBUG("Input ring-buffer byte count: {:d}", input_bytes);

    // How many inputs samples are available?
    const std::size_t T_available = div_noremainder(input_bytes, E_T_sample_bytes);
    DEBUG("Available samples:      T_available: {:d}", T_available);

    // How many outputs will we process and consume?
    const std::size_t T_processed = num_processed_elements(T_available);
    const std::size_t T_consumed = num_consumed_elements(T_available);
    DEBUG("Will process (samples): T_processed: {:d}", T_processed);
    DEBUG("Will consume (samples): T_consumed:  {:d}", T_consumed);
    assert(T_processed > 0);
    assert(T_consumed <= T_processed);
    const std::size_t T_consumed2 = num_consumed_elements(T_processed);
    assert(T_consumed2 == T_consumed);

    const std::optional<std::size_t> val_in2 =
        input_ringbuf_signal->wait_and_claim_readable(unique_name, T_consumed * E_T_sample_bytes);
    if (!val_in2.has_value())
        return -1;
    const std::size_t input_cursor = val_in2.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    Tmin = div_noremainder(input_cursor, E_T_sample_bytes);
    Tmax = Tmin + T_processed;
    const std::size_t Tlength = Tmax - Tmin;
    DEBUG("Input samples:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);
    DEBUG("    Tlength: {:d}", Tlength);

    // How many outputs will we produce?
    const std::size_t Tbar_produced = num_produced_elements(T_available);
    DEBUG("Will produce (samples): Tbar_produced: {:d}", Tbar_produced);
    const std::size_t Tbarlength = Tbar_produced;

    // to bytes
    const std::size_t output_bytes = Tbarlength * Ebar_Tbar_sample_bytes;
    DEBUG("Will produce {:d} output bytes", output_bytes);

    // Wait for space to be available in our output ringbuffer...
    DEBUG("Waiting for output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::optional<std::size_t> val_out =
        output_ringbuf_signal->wait_for_writable(unique_name, output_bytes);
    DEBUG("Finished waiting for output for data frame {:d}.", gpu_frame_id);
    if (!val_out.has_value())
        return -1;
    const std::size_t output_cursor = val_out.value();
    DEBUG("Output ring-buffer byte offset {:d}", output_cursor);

    assert(mod(output_cursor, Ebar_Tbar_sample_bytes) == 0);
    Tbarmin = output_cursor / Ebar_Tbar_sample_bytes;
    Tbarmax = Tbarmin + Tbarlength;
    DEBUG("Output samples:");
    DEBUG("    Tbarmin:    {:d}", Tbarmin);
    DEBUG("    Tbarmax:    {:d}", Tbarmax);
    DEBUG("    Tbarlength: {:d}", Tbarlength);

    return 0;
}

cudaEvent_t cuda{{{kernel_name}}}::execute(cudaPipelineState& /*pipestate*/, const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    {{#kernel_arguments}}
        {{#hasbuffer}}
            void* const {{{name}}}_memory =
                args::{{{name}}} == args::E ?
                    device.get_gpu_memory({{{name}}}_memname, input_ringbuf_signal->size) :
                args::{{{name}}} == args::Ebar ?
                    device.get_gpu_memory({{{name}}}_memname, output_ringbuf_signal->size) :
                    device.get_gpu_memory_array({{{name}}}_memname, gpu_frame_id, _gpu_buffer_depth, {{{name}}}_length);
        {{/hasbuffer}}
        {{^hasbuffer}}
            void* const {{{name}}}_memory = device.get_gpu_memory({{{name}}}_memname, {{{name}}}_length);
        {{/hasbuffer}}
    {{/kernel_arguments}}

    {{#kernel_arguments}}
        {{#hasbuffer}}
            {{^isoutput}}
                // {{{name}}} is an input buffer: check metadata
                const std::shared_ptr<metadataObject> {{{name}}}_mc =
                    args::{{{name}}} == args::E ?
                        input_ringbuf_signal->get_metadata(0) :
                        device.get_gpu_memory_array_metadata({{{name}}}_memname, gpu_frame_id);
                assert({{{name}}}_mc);
                assert(metadata_is_chord({{{name}}}_mc));
                const std::shared_ptr<chordMetadata> {{{name}}}_meta = get_chord_metadata({{{name}}}_mc);
                INFO("input {{{name}}} array: {:s} {:s}",
                    {{{name}}}_meta->get_type_string(),
                    {{{name}}}_meta->get_dimensions_string());
                assert({{{name}}}_meta->type == {{{name}}}_type);
                assert({{{name}}}_meta->dims == {{{name}}}_rank);
                for (std::size_t dim = 0; dim < {{{name}}}_rank; ++dim) {
                    assert(std::strncmp({{{name}}}_meta->dim_name[dim],
                                        {{{name}}}_labels[{{{name}}}_rank - 1 - dim],
                                        sizeof {{{name}}}_meta->dim_name[dim]) == 0);
                    if (args::{{{name}}} == args::E && dim == 0)
                        assert({{{name}}}_meta->dim[dim] <= int({{{name}}}_lengths[{{{name}}}_rank - 1 - dim]));
                    else
                        assert({{{name}}}_meta->dim[dim] == int({{{name}}}_lengths[{{{name}}}_rank - 1 - dim]));
                }
                //
            {{/isoutput}}
            {{#isoutput}}
                // {{{name}}} is an output buffer: set metadata
                std::shared_ptr<metadataObject> const {{{name}}}_mc =
                    args::{{{name}}} == args::Ebar ?
                        output_ringbuf_signal->get_metadata(0) :
                        device.create_gpu_memory_array_metadata({{{name}}}_memname, gpu_frame_id, E_mc->parent_pool);
                std::shared_ptr<chordMetadata> const {{{name}}}_meta = get_chord_metadata({{{name}}}_mc);
                *{{{name}}}_meta = *E_meta;
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

    record_start_event();

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

    INFO("gpu_frame_id: {}", gpu_frame_id);

    // Set E_memory to beginning of input ring buffer
    E_arg = kernel_arg(E_memory, E_length);

    // Set Ebar_memory to beginning of output ring buffer
    Ebar_arg = kernel_arg(Ebar_memory, Ebar_length);

    // Ringbuffer size
    const std::size_t T_ringbuf = input_ringbuf_signal->size / E_T_sample_bytes;
    const std::size_t Tbar_ringbuf = output_ringbuf_signal->size / Ebar_Tbar_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", T_ringbuf);
    DEBUG("Output ringbuffer size (samples): {:d}", Tbar_ringbuf);

    const std::size_t Tlength = Tmax - Tmin;
    const std::size_t Tbarlength = Tbarmax - Tbarmin;
    DEBUG("Processed input samples: {:d}", Tlength);
    DEBUG("Produced output samples: {:d}", Tbarlength);

    DEBUG("Kernel arguments:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);
    DEBUG("    Tbarmin: {:d}", Tbarmin);
    DEBUG("    Tbarmax: {:d}", Tbarmax);

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    *(std::int32_t*)Tmin_host.data() = mod(Tmin, T_ringbuf);
    *(std::int32_t*)Tmax_host.data() = mod(Tmin, T_ringbuf) + Tlength;
    *(std::int32_t*)Tbarmin_host.data() = mod(Tbarmin, Tbar_ringbuf);
    *(std::int32_t*)Tbarmax_host.data() = mod(Tbarmin, Tbar_ringbuf) + Tbarlength;

    // Update metadata
    Ebar_meta->dim[0] = Tbarlength;
    assert(Ebar_meta->dim[0] <= int(Ebar_lengths[3]));

    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    assert(Ebar_meta->nfreq >= 0);
    assert(Ebar_meta->nfreq == E_meta->nfreq);
    for (int freq = 0; freq < Ebar_meta->nfreq; ++freq) {
        Ebar_meta->freq_upchan_factor[freq] = cuda_upchannelization_factor * E_meta->freq_upchan_factor[freq];
        Ebar_meta->half_fpga_sample0[freq] = E_meta->half_fpga_sample0[freq] + cuda_number_of_taps - 1;
	Ebar_meta->time_downsampling_fpga[freq] = cuda_upchannelization_factor * E_meta->time_downsampling_fpga[freq];
    }

    // Copy inputs to device memory
    // TODO: Pass scalar kernel arguments more efficiently, i.e. without a separate `cudaMemcpy`
    {{#kernel_arguments}}
    {{^hasbuffer}}
        {{^isoutput}}
            CHECK_CUDA_ERROR(cudaMemcpyAsync({{{name}}}_memory,
                                             {{{name}}}_host.data(),
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

    // // Initialize outputs
    // //     0x88 = (-8,-8), an unused value to detect uninitialized output
    // // TODO: Skip this for performance
    // CHECK_CUDA_ERROR(cudaMemsetAsync(Ebar_memory, 0x88, Ebar_length, device.getStream(cuda_stream_id)));

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

    // Copy results back to host memory
    // TODO: Skip this for performance
    {{#kernel_arguments}}
        {{^hasbuffer}}
            {{#isoutput}}
                CHECK_CUDA_ERROR(cudaMemcpyAsync({{{name}}}_host.data(),
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
    const std::int32_t error_code = *std::max_element((const std::int32_t*)&*info_host.begin(),
                                                      (const std::int32_t*)&*info_host.end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    for (std::size_t i = 0; i < info_host.size(); ++i)
        if (info_host[i] != 0)
            ERROR("cuda{{{kernel_name}}} returned 'info' value {:d} at index {:d} (zero indicates no error)",
                info_host[i], i);

    return record_end_event();
}

void cuda{{{kernel_name}}}::finalize_frame() {
    const std::size_t Tlength = Tmax - Tmin;
    const std::size_t Tbarlength = Tbarmax - Tbarmin;

    // Advance the input ringbuffer
    const std::size_t T_consumed = num_consumed_elements(Tlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples: {:d}", T_consumed);
    DEBUG("    Consumed bytes:   {:d}", T_consumed * E_T_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, T_consumed * E_T_sample_bytes);

    // Advance the output ringbuffer
    const std::size_t Tbar_produced = Tbarlength;
    DEBUG("Advancing output ringbuffer:");
    DEBUG("    Produced samples: {:d}", Tbar_produced);
    DEBUG("    Produced bytes:   {:d}", Tbar_produced * Ebar_Tbar_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, Tbar_produced * Ebar_Tbar_sample_bytes);

    cudaCommand::finalize_frame();
}
