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
    static constexpr int max_blocks = {{{num_blocks}}};
    static constexpr int shmem_bytes = {{{shmem_bytes}}};

    // Kernel name:
    static constexpr const char* kernel_symbol = "{{{kernel_symbol}}}";

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
            static_assert({{{name}}}_length == chord_datatype_bytes({{{name}}}_type) * {{{name}}}_strides[{{{name}}}_rank]);
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

    static constexpr std::ptrdiff_t Ebar_Tbar_sample_bytes =
        chord_datatype_bytes(Ebar_type) * Ebar_lengths[Ebar_index_D] * Ebar_lengths[Ebar_index_P] * Ebar_lengths[Ebar_index_Fbar];
    static constexpr std::ptrdiff_t I_Ttilde_sample_bytes =
        chord_datatype_bytes(I_type) * I_lengths[I_index_beamP] * I_lengths[I_index_beamQ] * I_lengths[I_index_Fbar];

    RingBuffer* const input_ringbuf_signal;
    RingBuffer* const output_ringbuf_signal;

    bool did_init_S_host;

    // How many frequencies we will process
    const int Fbar_in_min, Fbar_in_max;

    // How many frequencies we will produce
    const int Fbar_out_min, Fbar_out_max;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tbarmin, Tbarmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Ttildemin, Ttildemax;
};

REGISTER_CUDA_COMMAND(cuda{{{kernel_name}}});

cuda{{{kernel_name}}}::cuda{{{kernel_name}}}(Config& config,
                                             const std::string& unique_name,
                                             bufferContainer& host_buffers,
                                             cudaDeviceInterface& device,
                                             const int instance_num) :
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
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "out_signal")))),
    did_init_S_host(false),
    Fbar_in_min(config.get<int>(unique_name, "Fbar_in_min")),
    Fbar_in_max(config.get<int>(unique_name, "Fbar_in_max")),
    Fbar_out_min(config.get<int>(unique_name, "Fbar_out_min")),
    Fbar_out_max(config.get<int>(unique_name, "Fbar_out_max"))
{
    // Check ringbuffer sizes
    if (!(input_ringbuf_signal->size == Ebar_length))
        FATAL_ERROR("Need input_ringbuf_signal->size == Ebar_length, but have input_ringbuf_signal->size={:d}, Ebar_length={:d}",
                    input_ringbuf_signal->size, Ebar_length);
    if (!(output_ringbuf_signal->size == I_length))
        FATAL_ERROR("Need output_ringbuf_signal->size == I_length, but have output_ringbuf_signal->size={:d}, I_length={:d}",
                    output_ringbuf_signal->size, I_length);
    assert(input_ringbuf_signal->size == Ebar_length);
    assert(output_ringbuf_signal->size == I_length);

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
    return num_produced_elements(num_available_elements) * cuda_downsampling_factor;
}
std::int64_t cuda{{{kernel_name}}}::num_produced_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) / cuda_downsampling_factor;
}

std::int64_t cuda{{{kernel_name}}}::num_processed_elements(std::int64_t num_available_elements) const {
    assert(num_available_elements >= cuda_granularity_number_of_timesamples);
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
    const std::ptrdiff_t Tbar_available = div_noremainder(input_bytes, Ebar_Tbar_sample_bytes);
    DEBUG("Available samples:      Tbar_available: {:d}", Tbar_available);

    // How many outputs will we process and consume?
    const std::ptrdiff_t Tbar_processed = num_processed_elements(Tbar_available);
    const std::ptrdiff_t Tbar_consumed = num_consumed_elements(Tbar_available);
    DEBUG("Will process (samples): Tbar_processed: {:d}", Tbar_processed);
    DEBUG("Will consume (samples): Tbar_consumed:  {:d}", Tbar_consumed);
    assert(Tbar_processed > 0);
    assert(Tbar_consumed <= Tbar_processed);
    const std::ptrdiff_t Tbar_consumed2 = num_consumed_elements(Tbar_processed);
    assert(Tbar_consumed2 == Tbar_consumed);

    const std::optional<std::ptrdiff_t> val_in2 =
        input_ringbuf_signal->wait_and_claim_readable(unique_name, instance_num, Tbar_consumed * Ebar_Tbar_sample_bytes);
    if (!val_in2.has_value())
        return -1;
    const std::ptrdiff_t input_cursor = val_in2.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    Tbarmin = div_noremainder(input_cursor, Ebar_Tbar_sample_bytes);
    Tbarmax = Tbarmin + Tbar_processed;
    const std::ptrdiff_t Tbarlength = Tbarmax - Tbarmin;
    DEBUG("Input samples:");
    DEBUG("    Tbarmin:    {:d}", Tbarmin);
    DEBUG("    Tbarmax:    {:d}", Tbarmax);
    DEBUG("    Tbarlength: {:d}", Tbarlength);

    // How many outputs will we produce?
    const std::ptrdiff_t Ttilde_produced = num_produced_elements(Tbar_available);
    DEBUG("Will produce (samples): Ttilde_produced: {:d}", Ttilde_produced);
    const std::ptrdiff_t Ttildelength = Ttilde_produced;

    // to bytes
    const std::ptrdiff_t output_bytes = Ttildelength * I_Ttilde_sample_bytes;
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

    assert(mod(output_cursor, I_Ttilde_sample_bytes) == 0);
    Ttildemin = output_cursor / I_Ttilde_sample_bytes;
    Ttildemax = Ttildemin + Ttildelength;
    DEBUG("Output samples:");
    DEBUG("    Ttildemin:    {:d}", Ttildemin);
    DEBUG("    Ttildemax:    {:d}", Ttildemax);
    DEBUG("    Ttildelength: {:d}", Ttildelength);

    return 0;
}

cudaEvent_t cuda{{{kernel_name}}}::execute(cudaPipelineState& /*pipestate*/, const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    {{#kernel_arguments}}
        {{^isscalar}}
            {{#hasbuffer}}
                void* const {{{name}}}_memory =
                    args::{{{name}}} == args::Ebar ?
                        device.get_gpu_memory({{{name}}}_memname, input_ringbuf_signal->size) :
                    args::{{{name}}} == args::I ?
                        device.get_gpu_memory({{{name}}}_memname, output_ringbuf_signal->size) :
                    args::{{{name}}} == args::W ?
                        device.get_gpu_memory({{{name}}}_memname, {{{name}}}_length) :
                        device.get_gpu_memory_array({{{name}}}_memname, gpu_frame_id, _gpu_buffer_depth, {{{name}}}_length);
            {{/hasbuffer}}
            {{^hasbuffer}}
                {{{name}}}_host.resize({{{name}}}_length);
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
                        args::{{{name}}} == args::Ebar ?
                            input_ringbuf_signal->get_metadata(0) :
                            device.get_gpu_memory_array_metadata({{{name}}}_memname, gpu_frame_id);
                    assert({{{name}}}_mc);
                    assert(metadata_is_chord({{{name}}}_mc));
                    const std::shared_ptr<chordMetadata> {{{name}}}_meta = get_chord_metadata({{{name}}}_mc);
                    DEBUG("input {{{name}}} array: {:s} {:s}",
                          {{{name}}}_meta->get_type_string(),
                          {{{name}}}_meta->get_dimensions_string());
                    if (args::{{{name}}} == args::Ebar && {{{upchannelization_factor}}} == 1) {
                        // Replace "Ebar_U1" with "E" etc. because we don't run the upchannelizer for U=1
                        assert(std::strncmp({{{name}}}_meta->name, "E", sizeof {{{name}}}_meta->name) == 0);
                        assert({{{name}}}_meta->type == {{{name}}}_type);
                        assert({{{name}}}_meta->dims == 4);
                        assert(std::strncmp({{{name}}}_meta->dim_name[3], "D", sizeof {{{name}}}_meta->dim_name[3]) == 0);
                        assert({{{name}}}_meta->dim[3] == int(Ebar_lengths[0]));
                        assert({{{name}}}_meta->stride[3] == Ebar_strides[0]);
                        assert(std::strncmp({{{name}}}_meta->dim_name[2], "P", sizeof {{{name}}}_meta->dim_name[2]) == 0);
                        assert({{{name}}}_meta->dim[2] == int(Ebar_lengths[1]));
                        assert({{{name}}}_meta->stride[2] == Ebar_strides[1]);
                        assert(std::strncmp({{{name}}}_meta->dim_name[1], "F", sizeof {{{name}}}_meta->dim_name[1]) == 0);
                        assert({{{name}}}_meta->dim[1] == int(Ebar_lengths[2]));
                        assert({{{name}}}_meta->stride[1] == Ebar_strides[2]);
                        assert(std::strncmp({{{name}}}_meta->dim_name[0], "T", sizeof {{{name}}}_meta->dim_name[0]) == 0);
                        assert({{{name}}}_meta->dim[0] <= int(Ebar_lengths[3]));
                        assert({{{name}}}_meta->stride[0] == Ebar_strides[3]);
                    } else {
                        assert(std::strncmp({{{name}}}_meta->name, {{{name}}}_name, sizeof {{{name}}}_meta->name) == 0);
                        assert({{{name}}}_meta->type == {{{name}}}_type);
                        assert({{{name}}}_meta->dims == {{{name}}}_rank);
                        for (std::ptrdiff_t dim = 0; dim < {{{name}}}_rank; ++dim) {
                            assert(std::strncmp({{{name}}}_meta->dim_name[{{{name}}}_rank - 1 - dim],
                                                {{{name}}}_labels[dim],
                                                sizeof {{{name}}}_meta->dim_name[{{{name}}}_rank - 1 - dim]) == 0);
                            if ((args::{{{name}}} == args::Ebar && dim == Ebar_rank - 1) ||
                                (args::{{{name}}} == args::W && dim == W_rank - 1)) {
                                assert({{{name}}}_meta->dim[{{{name}}}_rank - 1 - dim] <= int({{{name}}}_lengths[dim]));
                                assert({{{name}}}_meta->stride[{{{name}}}_rank - 1 - dim] == {{{name}}}_strides[dim]);
                            } else {
                                assert({{{name}}}_meta->dim[{{{name}}}_rank - 1 - dim] == int({{{name}}}_lengths[dim]));
                                assert({{{name}}}_meta->stride[{{{name}}}_rank - 1 - dim] == {{{name}}}_strides[dim]);
                            }
                        }
                    }
                    //
                {{/isoutput}}
                {{#isoutput}}
                    // {{{name}}} is an output buffer: set metadata
                    std::shared_ptr<metadataObject> const {{{name}}}_mc =
                        args::{{{name}}} == args::I ?
                            output_ringbuf_signal->get_metadata(0) :
                            device.create_gpu_memory_array_metadata({{{name}}}_memname, gpu_frame_id, Ebar_mc->parent_pool);
                    std::shared_ptr<chordMetadata> const {{{name}}}_meta = get_chord_metadata({{{name}}}_mc);
                    *{{{name}}}_meta = *Ebar_meta;
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

    assert(Ebar_meta->ndishes == cuda_number_of_dishes);
    assert(Ebar_meta->n_dish_locations_ew == cuda_dish_layout_N);
    assert(Ebar_meta->n_dish_locations_ns == cuda_dish_layout_M);
    assert(Ebar_meta->dish_index);

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

    // Set Ebar_memory to beginning of input ring buffer
    Ebar_arg = array_desc(Ebar_memory, Ebar_length);

    // Set I_memory to beginning of output ring buffer
    I_arg = array_desc(I_memory, I_length);

    // Ringbuffer size
    const std::ptrdiff_t Tbar_ringbuf = input_ringbuf_signal->size / Ebar_Tbar_sample_bytes;
    const std::ptrdiff_t Ttilde_ringbuf = output_ringbuf_signal->size / I_Ttilde_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", Tbar_ringbuf);
    DEBUG("Output ringbuffer size (samples): {:d}", Ttilde_ringbuf);

    const std::ptrdiff_t Tbarlength = Tbarmax - Tbarmin;
    const std::ptrdiff_t Ttildelength = Ttildemax - Ttildemin;
    DEBUG("Processed input samples: {:d}", Tbarlength);
    DEBUG("Produced output samples: {:d}", Ttildelength);

    DEBUG("Kernel arguments:");
    DEBUG("    Tbarmin:   {:d}", Tbarmin);
    DEBUG("    Tbarmax:   {:d}", Tbarmax);
    DEBUG("    Ttildemin: {:d}", Ttildemin);
    DEBUG("    Ttildemax: {:d}", Ttildemax);

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    Tbarmin_arg = mod(Tbarmin, Tbar_ringbuf);
    Tbarmax_arg = mod(Tbarmin, Tbar_ringbuf) + Tbarlength;
    Ttildemin_arg = mod(Ttildemin, Ttilde_ringbuf);
    Ttildemax_arg = mod(Ttildemin, Ttilde_ringbuf) + Ttildelength;

    // Pass frequency spans to kernel
    Fbar_in_min_arg = Fbar_in_min;
    Fbar_in_max_arg = Fbar_in_max;
    Fbar_out_min_arg = Fbar_out_min;
    Fbar_out_max_arg = Fbar_out_max;

    // Update metadata
    I_meta->dim[I_rank - 1 - I_index_Ttilde] = Ttildelength;
    assert(I_meta->dim[I_rank - 1 - I_index_Ttilde] <= int(I_lengths[I_index_Ttilde]));
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    assert(I_meta->nfreq >= 0);
    assert(I_meta->nfreq == Ebar_meta->nfreq);
    for (int freq = 0; freq < I_meta->nfreq; ++freq) {
        I_meta->freq_upchan_factor[freq] = cuda_downsampling_factor * Ebar_meta->freq_upchan_factor[freq];
        // I_meta->half_fpga_sample0[freq] = Evar_meta->half_fpga_sample0[freq];
        I_meta->time_downsampling_fpga[freq] = cuda_downsampling_factor * Ebar_meta->time_downsampling_fpga[freq];
    }

    // Initialize `S` and copy it to the GPU
    if (!did_init_S_host) {
        // S maps dishes to locations.
        // The first `ndishes` dishes are real dishes,
        // the remaining dishes are not real and exist only to label the unoccupied dish locations.
        std::int16_t* __restrict__ const S = static_cast<std::int16_t*>(static_cast<void*>(S_host.data()));
        int surplus_dish_index = cuda_number_of_dishes;
        for (int locM = 0; locM < cuda_dish_layout_M; ++locM) {
            for (int locN = 0; locN < cuda_dish_layout_N; ++locN) {
                int dish_index = Ebar_meta->get_dish_index(locN, locM);
                if (dish_index >= 0) {
                    // This location holds a real dish, record its location
                    S[2 * dish_index + 0] = locM;
                    S[2 * dish_index + 1] = locN;
                } else {
                    // This location is empty, assign it a surplus dish index
                    S[2 * surplus_dish_index + 0] = locM;
                    S[2 * surplus_dish_index + 1] = locN;
                    ++surplus_dish_index;
                }
            }
        }
        assert(surplus_dish_index == cuda_dish_layout_M * cuda_dish_layout_N);
        // INFO("M={} N={}", cuda_dish_layout_M, cuda_dish_layout_N);
        // for (int i = 0; i < int(S_host.size() / 2); i += 2)
        //     INFO("    S[{}] = ({}, {})", i / 2, S[i], S[i+1]);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(S_memory, S_host.data(), S_length, cudaMemcpyHostToDevice,
                                         device.getStream(cuda_stream_id)));

        did_init_S_host = true;
    }

    // Copy inputs to device memory
    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                {{^isoutput}}
                    if constexpr (args::{{{name}}} != args::S)
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
        const int num_chunks = Ttildemax_arg <= Ttilde_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            DEBUG("poisoning chunk={}/{}", chunk, num_chunks);
            const std::ptrdiff_t Ttildestride = I_meta->stride[0];
            const std::ptrdiff_t Ttildeoffset = chunk == 0 ? Ttildemin_arg : 0;
            const std::ptrdiff_t Ttildelength = (num_chunks == 1 ?
                                                     Ttildemax_arg - Ttildemin_arg :
                                                     chunk == 0 ? Ttilde_ringbuf - Ttildemin_arg : Ttildemax_arg - Ttilde_ringbuf);
            const std::ptrdiff_t Fbar_out_stride = I_meta->stride[1];
            const std::ptrdiff_t Fbar_out_offset = Fbar_out_min;
            const std::ptrdiff_t Fbar_out_length = Fbar_out_max - Fbar_out_min;
            DEBUG("before cudaMemset2DAsync.I");
            CHECK_CUDA_ERROR(cudaMemset2DAsync((std::uint8_t*)I_memory +
                                                   2 * Ttildeoffset * Ttildestride + 2 * Fbar_out_offset * Fbar_out_stride,
                                               2 * Ttildestride,
                                               0x88,
                                               2 * Fbar_out_length * Fbar_out_stride,
                                               Ttildelength,
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
    assert(0 <= Fbar_in_min && Fbar_in_min <= Fbar_in_max);
    assert(0 <= Fbar_out_min && Fbar_out_min <= Fbar_out_max);
    assert(Fbar_out_max - Fbar_out_min == Fbar_in_max - Fbar_in_min);
    const int blocks = Fbar_in_max - Fbar_in_min;
    assert(0 <= blocks);
    assert(blocks <= max_blocks);
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

    for (std::size_t i = 0; i < info_host.size() * blocks / max_blocks; ++i)
        if (info_host[i] != 0)
            ERROR("cuda{{{kernel_name}}} returned 'info' value {:d} at index {:d} (zero indicates no error)",
                info_host[i], i);
#endif

#ifdef DEBUGGING
    // Check outputs for poison
    {
        DEBUG("begin poison check");
        DEBUG("    Ebar_dims={}", Ebar_meta->dims);
        DEBUG("    Ebar_dim[0]={}", Ebar_meta->dim[0]);
        DEBUG("    Ebar_dim[1]={}", Ebar_meta->dim[1]);
        DEBUG("    Ebar_dim[2]={}", Ebar_meta->dim[2]);
        DEBUG("    Ebar_dim[3]={}", Ebar_meta->dim[3]);
        DEBUG("    Ebar_stride[0]={}", Ebar_meta->stride[0]);
        DEBUG("    Ebar_stride[1]={}", Ebar_meta->stride[1]);
        DEBUG("    Ebar_stride[2]={}", Ebar_meta->stride[2]);
        DEBUG("    Ebar_stride[3]={}", Ebar_meta->stride[3]);
        DEBUG("    I_dims={}", I_meta->dims);
        DEBUG("    I_dim[0]={}", I_meta->dim[0]);
        DEBUG("    I_dim[1]={}", I_meta->dim[1]);
        DEBUG("    I_dim[2]={}", I_meta->dim[2]);
        DEBUG("    I_dim[3]={}", I_meta->dim[3]);
        DEBUG("    I_stride[0]={}", I_meta->stride[0]);
        DEBUG("    I_stride[1]={}", I_meta->stride[1]);
        DEBUG("    I_stride[2]={}", I_meta->stride[2]);
        DEBUG("    I_stride[3]={}", I_meta->stride[3]);
        const int num_chunks = Ttildemax_arg <= Ttilde_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            DEBUG("poisoning chunk={}/{}", chunk, num_chunks);
            const std::ptrdiff_t Ttildestride = I_meta->stride[0];
            const std::ptrdiff_t Ttildeoffset = chunk == 0 ? Ttildemin_arg : 0;
            const std::ptrdiff_t Ttildelength = (num_chunks == 1 ?
                                                     Ttildemax_arg - Ttildemin_arg :
                                                     chunk == 0 ? Ttilde_ringbuf - Ttildemin_arg : Ttildemax_arg - Ttilde_ringbuf);
            const std::ptrdiff_t Fbar_out_stride = I_meta->stride[1];
            const std::ptrdiff_t Fbar_out_offset = Fbar_out_min;
            const std::ptrdiff_t Fbar_out_length = Fbar_out_max - Fbar_out_min;
            DEBUG("    Ttildestride={}", Ttildestride);
            DEBUG("    Ttildeoffset={}", Ttildeoffset);
            DEBUG("    Ttildelength={}", Ttildelength);
            DEBUG("    Fbar_out_stride={}", Fbar_out_stride);
            DEBUG("    Fbar_out_offset={}", Fbar_out_offset);
            DEBUG("    Fbar_out_length={}", Fbar_out_length);
            std::vector<std::uint16_t> I_buffer(Ttildelength * Fbar_out_length * Fbar_out_stride, 0x1111);
            DEBUG("    I_buffer.size={}", I_buffer.size());
            DEBUG("before cudaMemcpy2D.I");
            CHECK_CUDA_ERROR(cudaMemcpy2D(I_buffer.data(),
                                          2 * Fbar_out_length * Fbar_out_stride,
                                          (const std::uint8_t*)I_memory +
                                              2 * Ttildeoffset * Ttildestride + 2 * Fbar_out_offset * Fbar_out_stride,
                                          2 * Ttildestride,
                                          2 * Fbar_out_length * Fbar_out_stride,
                                          Ttildelength,
                                          cudaMemcpyDeviceToHost));

            DEBUG("before memchr");
            bool I_found_error = false;
            for (std::ptrdiff_t ttilde=0; ttilde<Ttildelength; ++ttilde) {
                for (std::ptrdiff_t ftilde=0; ftilde<Fbar_out_length; ++ftilde) {
                    // for (std::ptrdiff_t n=0; n<Fbar_out_stride; ++n) {
                    //     const auto val = I_buffer.at(ttilde * (Fbar_out_length * Fbar_out_stride) + ftilde * Fbar_out_stride + n); 
                    //     if (val == 0x88) {
                    //         DEBUG("    U={{{upchannelization_factor}}} [{},{},{}]={:#02x}", ttilde, ftilde, n, 0x88);
                    //     }
                    // }
                    bool any_error = false, all_error = true;
                    for (std::ptrdiff_t n=0; n<Fbar_out_stride; ++n) {
                        const auto val = I_buffer.at(ttilde * (Fbar_out_length * Fbar_out_stride) + ftilde * Fbar_out_stride + n); 
                        any_error |= val == 0x8888;
                        all_error &= val == 0x8888;
                    }
                    if (any_error) {
                        DEBUG("    U={{{upchannelization_factor}}} [{},{}]=(any={},all={})",
                              ttilde, ftilde, any_error, all_error);
                    }
                    I_found_error |= any_error;
                }
            }
            assert(!I_found_error);
        } // for chunk
        DEBUG("poison check done.");
    }
#endif

    return record_end_event();
}

void cuda{{{kernel_name}}}::finalize_frame() {
    const std::ptrdiff_t Tbarlength = Tbarmax - Tbarmin;
    const std::ptrdiff_t Ttildelength = Ttildemax - Ttildemin;

    // Advance the input ringbuffer
    const std::ptrdiff_t Tbar_consumed = num_consumed_elements(Tbarlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples: {:d}", Tbar_consumed);
    DEBUG("    Consumed bytes:   {:d}", Tbar_consumed * Ebar_Tbar_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, instance_num, Tbar_consumed * Ebar_Tbar_sample_bytes);

    // Advance the output ringbuffer
    const std::ptrdiff_t Ttilde_produced = Ttildelength;
    DEBUG("Advancing output ringbuffer:");
    DEBUG("    Produced samples: {:d}", Ttilde_produced);
    DEBUG("    Produced bytes:   {:d}", Ttilde_produced * I_Ttilde_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, instance_num, Ttilde_produced * I_Ttilde_sample_bytes);

    cudaCommand::finalize_frame();
}
