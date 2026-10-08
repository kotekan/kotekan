/**
 * @file
 * @brief CUDA {{{kernel_name}}} kernel
 *
 * This file has been generated automatically.
 * Do not modify this C++ file, your changes will be lost.
 */

#include "DataType.hpp"
#include "NDArrayBuffer.hpp"
#include "NDArrayRingBuffer.hpp"
#include "Telescope.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaUtils.hpp"
#include "div.hpp"
#include "ringbuffer.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <fmt.hpp>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::round_down, kotekan::round_up, kotekan::div_noremainder, kotekan::div,
    kotekan::mod;

namespace {
template<typename T, std::size_t D>
std::array<T, D> reverse(const std::array<T, D>& values) {
    std::array<T, D> result;
    for (std::size_t d=0; d<D; ++d)
        result[d] = values[D - 1 - d];
    return result;
}

// Override the leading (slowest) dimension's scaling with a run-time value. Used for inputs
// that are valid for a configurable number of FPGA samples, such as the beamforming weights.
template<std::size_t D>
std::array<std::ptrdiff_t, D> with_leading_dimscaling(std::array<std::ptrdiff_t, D> dimscalings,
                                                      const std::ptrdiff_t dimscaling) {
    static_assert(D > 0);
    dimscalings[0] = dimscaling;
    return dimscalings;
}
}

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
    using array_desc = CuDeviceArray<std::int32_t, 1>;

    // Kernel design parameters:
    {{#kernel_design_parameters}}
        static constexpr {{{type}}} {{{name}}} = {{{value}}};
    {{/kernel_design_parameters}}
    // We are not using all the non-upchannelized frequencies.
    // But we are (should be!) using all the upchannelized ones.
    static_assert(cuda_upchannelization_factor > 1);

    // Each kernel invocation processes a multiple of this many `Tbar` samples: a multiple of the
    // kernel's granularity, and of the downsampling factor so that every sample read is also
    // consumed. The read head thus always stays on a multiple of `Tbar_quantum`, which is what
    // lets a read stop exactly at the end of a slowly varying input's lifetime.
    static constexpr std::ptrdiff_t Tbar_quantum =
        std::lcm(std::ptrdiff_t(cuda_granularity_number_of_timesamples),
                 std::ptrdiff_t(cuda_downsampling_factor));

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
    static constexpr int num_blocks = {{{num_blocks}}};
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
        static constexpr const char *{{{name}}}_quantity = "{{{name}}}";
        static constexpr kotekan::DataType {{{name}}}_type = kotekan::{{{type}}};
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
            static constexpr std::array<std::ptrdiff_t, {{{name}}}_rank> {{{name}}}_dimscalings = {
                {{#axes}}
                    {{{dimscaling}}},
                {{/axes}}
            };
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
            static constexpr std::ptrdiff_t {{{name}}}_length = {{{name}}}_strides[{{{name}}}_rank];
            static constexpr std::ptrdiff_t {{{name}}}_length_in_bytes = type_total_bytes({{{name}}}_type) * {{{name}}}_length;
        {{/isscalar}}
        //
    {{/kernel_arguments}}

    const bool poison_buffers;

    // Kotekan buffer names
    {{#kernel_arguments}}
        {{^isscalar}}
            const std::string {{{name}}}_name;
        {{/isscalar}}
    {{/kernel_arguments}}

    // Lifetimes of slowly varying inputs, in FPGA samples
    {{#kernel_arguments}}
        {{#haslifetime}}
            const std::ptrdiff_t {{{name}}}_lifetime_in_samples;
        {{/haslifetime}}
    {{/kernel_arguments}}

    // Host-side buffer arrays
    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                std::vector<std::uint8_t> {{{name}}}_host;
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    // Buffers
    {{#kernel_arguments}}
        {{^isscalar}}
            {{#hasbuffer}}
                {{#hasringbuffer}}
                    NDArrayRingBuffer<kotekan::GetType_t<{{{name}}}_type>, {{{name}}}_rank> {{{name}}}_buffer;
                {{/hasringbuffer}}
                {{^hasringbuffer}}
                    NDArrayBuffer<kotekan::GetType_t<{{{name}}}_type>, {{{name}}}_rank> {{{name}}}_buffer;
                {{/hasringbuffer}}
            {{/hasbuffer}}
            {{^hasbuffer}}
                NDArrayBuffer<kotekan::GetType_t<{{{name}}}_type>, {{{name}}}_rank> {{{name}}}_buffer;
                std::vector<kotekan::GetType_t<{{{name}}}_type>> host_{{{name}}}_buffer;
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    bool did_set_metadata;
};

REGISTER_CUDA_COMMAND(cuda{{{kernel_name}}});

cuda{{{kernel_name}}}::cuda{{{kernel_name}}}(Config& config,
                                             const std::string& unique_name,
                                             bufferContainer& host_buffers,
                                             cudaDeviceInterface& device,
                                             const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
        "{{{kernel_name}}}", "{{{kernel_name}}}.ptx"),

    poison_buffers(config.get_default<bool>(unique_name, "poison_buffers", false)),

    {{#kernel_arguments}}
        {{^isscalar}}
            {{#hasbuffer}}
                {{{name}}}_name(config.get<std::string>(unique_name, "{{{kotekan_name}}}")),
            {{/hasbuffer}}
            {{^hasbuffer}}
                {{{name}}}_name(unique_name + "/{{{kotekan_name}}}"),
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    {{#kernel_arguments}}
        {{#haslifetime}}
            {{{name}}}_lifetime_in_samples(config.get<std::int64_t>(unique_name, "{{{lifetime_config}}}")),
        {{/haslifetime}}
    {{/kernel_arguments}}

    {{#kernel_arguments}}
        {{^isscalar}}
            {{#hasbuffer}}
                {{#hasringbuffer}}
                    {{{name}}}_buffer(
                        {{{name}}}_name,
                        {{{name}}}_quantity,
                        reverse({{{name}}}_lengths),
                        reverse({{{name}}}_labels),
                        {{#haslifetime}}
                            with_leading_dimscaling(reverse({{{name}}}_dimscalings), {{{name}}}_lifetime_in_samples),
                        {{/haslifetime}}
                        {{^haslifetime}}
                            reverse({{{name}}}_dimscalings),
                        {{/haslifetime}}
                        *this
                    ),
                {{/hasringbuffer}}
                {{^hasringbuffer}}
                    {{{name}}}_buffer(
                        {{{name}}}_name,
                        {{{name}}}_quantity,
                        reverse({{{name}}}_lengths),
                        reverse({{{name}}}_labels),
                        reverse({{{name}}}_dimscalings),
                        *this
                        {{#do_once}}
                            , buffer_type_t::do_once
                        {{/do_once}}
                    ),
                {{/hasringbuffer}}
            {{/hasbuffer}}
            {{^hasbuffer}}
                {{{name}}}_buffer(
                    {{{name}}}_name,
                    {{{name}}}_quantity,
                    reverse({{{name}}}_lengths),
                    reverse({{{name}}}_labels),
                    reverse({{{name}}}_dimscalings),
                    *this
                ),
                host_{{{name}}}_buffer({{{name}}}_length),
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    did_set_metadata(false)
{
    // Register host memory
    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                CHECK_CUDA_ERROR(cudaHostRegister(host_{{{name}}}_buffer.data(),
                                                  host_{{{name}}}_buffer.size() * sizeof *host_{{{name}}}_buffer.data(),
                                                  0));
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    {{#kernel_arguments}}
        {{^isscalar}}
            {{#hasbuffer}}
                {{^isoutput}}
                    {{{name}}}_buffer.register_consumer();
                {{/isoutput}}
                {{#isoutput}}
                    {{{name}}}_buffer.register_producer();
                {{/isoutput}}
            {{/hasbuffer}}
            {{^hasbuffer}}
                register_gpu_buffer_user({.name = {{{name}}}_name, .is_array = true, .does_read = true, .does_write = true});
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    // Every invocation processes a multiple of `Tbar_quantum` samples, so at least that many
    // have to fit into one read, or the kernel would never make progress.
    {
        const std::ptrdiff_t Tbar_read_max = Ebar_buffer.get_ndarray().extent(0) / 4;
        if (Tbar_quantum > Tbar_read_max)
            FATAL_ERROR("Kernel {{{kernel_name}}} processes multiples of {:d} time samples "
                        "(the least common multiple of its granularity {:d} and its downsampling "
                        "factor {:d}), but reads at most {:d} time samples at a time",
                        Tbar_quantum, int(cuda_granularity_number_of_timesamples),
                        int(cuda_downsampling_factor), Tbar_read_max);
    }

    // Slowly varying inputs are held in a ring buffer and read without claiming, one element
    // per lifetime. A kernel invocation must not straddle the end of a lifetime, so a lifetime
    // has to be a whole number of processing quanta. (Unlike the baseband beamformer, the
    // output here is a ring buffer, so the reads can be shortened to stop at a lifetime's end.)
    {{#kernel_arguments}}
        {{#haslifetime}}
            {
                const std::ptrdiff_t quantum = cuda_upchannelization_factor * Tbar_quantum;
                if ({{{name}}}_lifetime_in_samples <= 0
                    || {{{name}}}_lifetime_in_samples % quantum != 0)
                    FATAL_ERROR("{{{lifetime_config}}} {:d} must be a positive multiple of {:d} "
                                "FPGA samples, the processing quantum of kernel {{{kernel_name}}} "
                                "(upchannelization factor {:d} times the least common multiple of "
                                "the granularity {:d} and the downsampling factor {:d})",
                                {{{name}}}_lifetime_in_samples, quantum,
                                int(cuda_upchannelization_factor),
                                int(cuda_granularity_number_of_timesamples),
                                int(cuda_downsampling_factor));
            }
        {{/haslifetime}}
    {{/kernel_arguments}}

    set_command_type(gpuCommandType::KERNEL);

    // Build the PTX once per device: the kernels live in this device's `runtime_kernels`, shared
    // by the `buffer_depth` instances of this command (building twice is fatal), while a stage on
    // another GPU has its own device. (A static flag would be shared by the stages of all GPUs.)
    if (!device.runtime_kernels.count("{{{kernel_name}}}_" + std::string(kernel_symbol))) {
        const std::vector<std::string> opts = {
            "--gpu-name={{{cuda_arch}}}",
            "--verbose",
        };
        device.build_ptx("lib/cuda/generated/{{{kernel_name}}}.ptx", {kernel_symbol}, opts, "{{{kernel_name}}}_");
    }
}

cuda{{{kernel_name}}}::~cuda{{{kernel_name}}}() {}

std::int64_t cuda{{{kernel_name}}}::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements);
}
std::int64_t cuda{{{kernel_name}}}::num_produced_elements(std::int64_t num_available_elements) const {
    return div_noremainder(num_processed_elements(num_available_elements), cuda_downsampling_factor);
}

std::int64_t cuda{{{kernel_name}}}::num_processed_elements(std::int64_t num_available_elements) const {
    return round_down(num_available_elements, Tbar_quantum);
}

int cuda{{{kernel_name}}}::wait_on_precondition() {
    {
        const int errcode = cudaCommand::wait_on_precondition();
        if (errcode < 0)
            return errcode;
    }

    const std::ptrdiff_t Tbar_ringbuf = Ebar_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t Tbar_read_max = Tbar_ringbuf / 4;

    // Where will our read begin? Ask the ringbuffer. We must not use our own `read_valid` for
    // this: every instance of this command shares one ringbuffer read head, so our own position
    // lags it by whatever the other instances have claimed since our previous frame.
    const std::ptrdiff_t Tbar_begin = Ebar_buffer.peek_read_head();
    if (Tbar_begin < 0)
        return -1; // shutting down
    // We only ever claim multiples of `Tbar_quantum`
    assert(Tbar_begin % Tbar_quantum == 0);

    // Slowly varying inputs: find the element covering the samples we are about to read, and do
    // not read past the end of its lifetime. (The constructor checked that lifetimes are
    // multiples of `Tbar_quantum`, so we can stop exactly there.)
    std::ptrdiff_t Tbar_read_limit = Tbar_read_max;
    {{#kernel_arguments}}
        {{#haslifetime}}
            // `Tbar` samples are `cuda_upchannelization_factor` FPGA samples apart
            const std::ptrdiff_t {{{name}}}_lifetime_in_Tbar =
                div_noremainder({{{name}}}_lifetime_in_samples, std::ptrdiff_t(cuda_upchannelization_factor));
            // (`kotekan::div` must be qualified; an unqualified `div` finds C's `::div`)
            const std::ptrdiff_t {{{name}}}_element = kotekan::div(Tbar_begin, {{{name}}}_lifetime_in_Tbar);
            const std::ptrdiff_t {{{name}}}_lifetime_end = ({{{name}}}_element + 1) * {{{name}}}_lifetime_in_Tbar;
            Tbar_read_limit = std::min(Tbar_read_limit, {{{name}}}_lifetime_end - Tbar_begin);
        {{/haslifetime}}
    {{/kernel_arguments}}
    assert(Tbar_read_limit >= Tbar_quantum);

    // Wait for data to be available in input ringbuffer
    std::ptrdiff_t Tbar_read = -1;
    {
        const int errcode = Ebar_buffer.wait_and_claim_readable([&](const std::ptrdiff_t Tbar_available) {
            using std::min;
            // `*_written` below is derived from this value, so it must include the clamp
            Tbar_read = num_processed_elements(min(Tbar_available, Tbar_read_limit));
            // If we cannot process a whole quantum then we read nothing, and wait for more data
            return read_descriptor_t{.claimed = num_consumed_elements(Tbar_read), .read = Tbar_read};
        });
        if (errcode < 0)
            return errcode;
    }
    const std::ptrdiff_t Tbar_end = Ebar_buffer.get_read_claimed().end();
    assert(Ebar_buffer.get_read_valid().begin() == Tbar_begin);
    assert(Ebar_buffer.get_read_valid().end() == Tbar_end);
    assert(Tbar_end <= Tbar_begin + Tbar_read_limit);

    // Slowly varying inputs: read the element covering these samples. We read the same element
    // on every invocation within its lifetime, and claim it only on the last one, so that the
    // producer can recycle it afterwards. That invocation is by construction the last one to use
    // the element, and `finalize_frame` runs in frame order, so releasing it there is safe.
    //
    // We are holding a claim on `Ebar` while we wait here. That is safe because the producer of
    // these inputs does not depend on `Ebar` being drained.
    {{#kernel_arguments}}
        {{#haslifetime}}
            {
                const bool last_use = Tbar_end == {{{name}}}_lifetime_end;
                DEBUG("Waiting for {{{name}}} input ringbuffer data for frame {:d}...", gpu_frame_id);
                const int errcode = {{{name}}}_buffer.wait_and_claim_readable(
                    [&](const std::ptrdiff_t available_elements) {
                        if (available_elements < 1)
                            return read_descriptor_t{.claimed = 0, .read = 0};
                        return read_descriptor_t{.claimed = last_use ? 1 : 0, .read = 1};
                    });
                if (errcode < 0)
                    return errcode;
                DEBUG("Done waiting for {{{name}}} input ringbuffer data for frame {:d}; "
                      "using element {:d}{:s}",
                      gpu_frame_id, {{{name}}}_element, last_use ? " (last use)" : "");
                // The two ringbuffers must agree on which element covers these samples
                if ({{{name}}}_buffer.get_read_valid().begin() != {{{name}}}_element)
                    FATAL_ERROR("Kernel {{{kernel_name}}}: samples [{:d},{:d}) of buffer Ebar are "
                                "covered by element {:d} of buffer {{{name}}}, but the {{{name}}} "
                                "ringbuffer is at element {:d}",
                                Tbar_begin, Tbar_end, {{{name}}}_element,
                                {{{name}}}_buffer.get_read_valid().begin());
            }
        {{/haslifetime}}
    {{/kernel_arguments}}

    const std::ptrdiff_t Ttilde_written = num_produced_elements(Tbar_read);

    // Wait for space to be available in output ringbuffer
    {
        const int errcode = I_buffer.wait_for_writable(Ttilde_written);
        if (errcode < 0)
            return errcode;
    }

    return 0;
}

cudaEvent_t cuda{{{kernel_name}}}::execute(cudaPipelineState& /*pipestate*/, const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    {{#kernel_arguments}}
        {{^isscalar}}
            void* const {{{name}}}_memory = {{{name}}}_buffer.get_ndarray().data();
        {{/isscalar}}
    {{/kernel_arguments}}

    // Since we use a ring buffer we need to set the metadata only once
    if (instance_num == 0 && !did_set_metadata) {
        did_set_metadata = true;

        {{#kernel_arguments}}
            {{#hasbuffer}}
                {{^isoutput}}
                    if (args::{{{name}}} == args::Ebar && cuda_upchannelization_factor == 1) {
                        // Replace "Ebar" with "E" etc. because we don't run the upchannelizer for U=1
                        // {{{name}}}_buffer.check_metadata();
                        const std::string quantity = "E";
                        const std::array<std::string, 4> dimname = {"T", "F", "P", "D"};
                        const std::shared_ptr<const chordMetadata> metadata = Ebar_buffer.get_metadata();
                        // A mismatch here means the kernel would index the buffer with a layout
                        // the producer did not use, silently producing wrong results, so these
                        // checks must hold in release builds as well.
                        if (!(metadata->get_name() == quantity))
                            FATAL_ERROR("buffer name: {:s}, quantity: {:s}, metadata name: {:s}", Ebar_buffer.get_buffer_name(),
                                        quantity, metadata->get_name());
                        const auto& ndarray = Ebar_buffer.get_ndarray();
                        if (!(metadata->type == ndarray.value_datatype))
                            FATAL_ERROR("buffer name: {:s}, metadata type: {:s}, ndarray type: {:s}", Ebar_buffer.get_buffer_name(),
                                        kotekan::type_to_string(metadata->type), kotekan::type_to_string(ndarray.value_datatype));
                        if (!(metadata->dims == int(ndarray.rank)))
                            FATAL_ERROR("buffer name: {:s}, metadata rank: {:d}, ndarray rank: {:d}", Ebar_buffer.get_buffer_name(),
                                        metadata->dims, int(ndarray.rank));
                        for (std::size_t d = 0; d < ndarray.rank; ++d) {
                            if (!(metadata->get_dimension_name(d) == dimname[d]))
                                FATAL_ERROR("buffer name: {:s}, dimension: {:d}: metadata dimension name: {:s}, expected: {:s}",
                                            Ebar_buffer.get_buffer_name(), d, metadata->get_dimension_name(d), dimname[d]);
                            // The ring buffer direction is special
                            if (d > 0 && !(metadata->dim[d] == int(ndarray.extent(d))))
                                FATAL_ERROR("buffer name: {:s}, dimension: {:d}: metadata extent: {:d}, ndarray extent: {:d}",
                                            Ebar_buffer.get_buffer_name(), d, metadata->dim[d], int(ndarray.extent(d)));
                            if (!(metadata->stride[d] == ndarray.stride(d)))
                                FATAL_ERROR("buffer name: {:s}, dimension: {:d}: metadata stride: {:d}, ndarray stride: {:d}",
                                            Ebar_buffer.get_buffer_name(), d, metadata->stride[d], ndarray.stride(d));
                        }
                    } else {
                        {{{name}}}_buffer.check_metadata();
                    }
                {{/isoutput}}
                {{#isoutput}}
                    if (args::{{{name}}} != args::I)
                        {{{name}}}_buffer.set_metadata(Ebar_buffer.get_metadata());
                {{/isoutput}}
            {{/hasbuffer}}
        {{/kernel_arguments}}

        const auto Ebar_meta = Ebar_buffer.get_metadata();
        // The kernel is compiled for a fixed dish grid. A larger telescope would place dishes
        // outside that grid and silently beamform the wrong sky.
        if (!(Telescope::instance().get_grid_size_x() <= std::uint64_t(cuda_dish_layout_N)
              && Telescope::instance().get_grid_size_y() <= std::uint64_t(cuda_dish_layout_M)))
            FATAL_ERROR("Telescope dish grid {:d}x{:d} does not fit the dish layout {:d}x{:d} "
                        "(N x M) for which kernel {{{kernel_name}}} was compiled",
                        Telescope::instance().get_grid_size_x(), Telescope::instance().get_grid_size_y(),
                        int(cuda_dish_layout_N), int(cuda_dish_layout_M));

        // Allocate metadata of I buffer only once
        const bool I_has_metadata = I_buffer.has_metadata();
        if (I_has_metadata)
            FATAL_ERROR("Output buffer I already has metadata; kernel {{{kernel_name}}} must be "
                        "its only producer");
        I_buffer.set_metadata(Ebar_meta);
        auto I_meta = I_buffer.get_metadata();

        const auto Ebar_nfreq = Ebar_meta->get_nfreq();
        const auto I_nfreq = I_meta->dim[I_rank - 1 - I_index_Fbar];
        if (I_nfreq < 0)
            FATAL_ERROR("Output buffer I reports a negative number of frequencies ({:d})", I_nfreq);

        const auto Ebar_freq_upchan_factor = Ebar_meta->get_freq_upchan_factor();
        if (Ebar_freq_upchan_factor.size() != static_cast<std::size_t>(Ebar_nfreq))
            FATAL_ERROR("Input buffer Ebar reports {:d} frequencies but its `freq_upchan_factor` "
                        "has {:d} entries",
                        Ebar_nfreq, Ebar_freq_upchan_factor.size());
        const auto& I_freq_upchan_factor = Ebar_freq_upchan_factor;
        I_meta->set_freq_upchan_factor(I_freq_upchan_factor);

        const auto Ebar_freq_upchan_index = Ebar_meta->get_freq_upchan_index();
        if (Ebar_freq_upchan_index.size() != static_cast<std::size_t>(Ebar_nfreq))
            FATAL_ERROR("Input buffer Ebar reports {:d} frequencies but its `freq_upchan_index` "
                        "has {:d} entries",
                        Ebar_nfreq, Ebar_freq_upchan_index.size());
        const auto& I_freq_upchan_index = Ebar_freq_upchan_index;
        I_meta->set_freq_upchan_index(I_freq_upchan_index);

        const auto Ebar_coarse_freq = Ebar_meta->get_coarse_freq();
        if (Ebar_coarse_freq.size() != static_cast<std::size_t>(Ebar_nfreq))
            FATAL_ERROR("Input buffer Ebar reports {:d} frequencies but its `coarse_freq` "
                        "has {:d} entries",
                        Ebar_nfreq, Ebar_coarse_freq.size());
        const auto& I_coarse_freq = Ebar_coarse_freq;
        I_meta->set_coarse_freq(I_coarse_freq);

        const auto Ebar_time_downsampling_fpga = Ebar_meta->get_time_downsampling_fpga();
        const auto I_time_downsampling_fpga = Ebar_time_downsampling_fpga * cuda_downsampling_factor;
        I_meta->set_time_downsampling_fpga(I_time_downsampling_fpga);

        const auto W_meta = W_buffer.get_metadata();
        const auto W_nfreq = W_meta->get_nfreq();
        // Mismatched weights would beamform each frequency with another frequency's gains.
        if (W_nfreq != I_nfreq)
            FATAL_ERROR("Weight buffer W holds {:d} frequencies, but kernel {{{kernel_name}}} "
                        "processes {:d}",
                        W_nfreq, I_nfreq);
        const auto W_coarse_freq = W_meta->get_coarse_freq();
        for (int freq = 0; freq < W_nfreq; ++freq)
            if (I_coarse_freq.at(freq) != W_coarse_freq.at(freq))
                FATAL_ERROR("Weight buffer W is for coarse frequency {:d} at index {:d}, but "
                            "kernel {{{kernel_name}}} processes coarse frequency {:d} there",
                            W_coarse_freq.at(freq), freq, I_coarse_freq.at(freq));

        // Element `k` of a slowly varying input covers the samples `k * lifetime` onwards,
        // counted from the voltage ring buffer's logical beginning -- so the two streams have
        // to start at the same sequence number. A misaligned input would be applied to the
        // wrong samples, silently corrupting the output. Ring buffer metadata are written once,
        // so checking once suffices.
        {{#kernel_arguments}}
            {{#haslifetime}}
                if ({{{name}}}_buffer.get_metadata()->get_fpga_seq_num() != Ebar_meta->get_fpga_seq_num())
                    FATAL_ERROR("Buffer {{{name}}} begins at FPGA sequence number {:d}, but the "
                                "voltage buffer Ebar begins at {:d}; kernel {{{kernel_name}}} "
                                "requires them to be aligned",
                                {{{name}}}_buffer.get_metadata()->get_fpga_seq_num(),
                                Ebar_meta->get_fpga_seq_num());
            {{/haslifetime}}
        {{/kernel_arguments}}

        // Since we use a ring buffer we do not need to update `meta->fpga_seq_num`
    } // if !did_set_metadata

    const auto Ebar_meta = Ebar_buffer.get_metadata();
    if (!I_buffer.has_metadata())
        FATAL_ERROR("Output buffer I has no metadata; kernel {{{kernel_name}}} cannot run");

    const char* exc_arg = "exception";
    {{#kernel_arguments}}
        {{^isscalar}}
            array_desc {{{name}}}_arg({{{name}}}_memory, {{{name}}}_length_in_bytes);
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
    Ebar_arg = array_desc(Ebar_memory, Ebar_length_in_bytes);

    // Set I_memory to beginning of output ring buffer
    I_arg = array_desc(I_memory, I_length_in_bytes);

    // Slowly varying inputs: the kernel wants a single element, not the whole ring buffer.
    {{#kernel_arguments}}
        {{#haslifetime}}
            {
                const std::ptrdiff_t ring_length = {{{name}}}_buffer.get_ndarray().extent(0);
                const std::ptrdiff_t element = {{{name}}}_buffer.get_read_valid().begin();
                {{{name}}}_arg = array_desc({{{name}}}_buffer.get_ndarray().data()
                                                + {{{name}}}_buffer.get_ndarray().stride(0)
                                                      * (element % ring_length),
                                            {{{name}}}_length_in_bytes / ring_length);
            }
        {{/haslifetime}}
    {{/kernel_arguments}}

    // Ringbuffer size
    const std::ptrdiff_t Tbar_ringbuf = Ebar_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t Ttilde_ringbuf = I_buffer.get_ndarray().extent(0);

    const std::ptrdiff_t Tbar_min = Ebar_buffer.get_read_valid().begin();
    const std::ptrdiff_t Tbar_max = Ebar_buffer.get_read_valid().end();
    const std::ptrdiff_t Ttilde_min = I_buffer.get_write_valid().begin();
    const std::ptrdiff_t Ttilde_max = I_buffer.get_write_valid().end();

    const std::ptrdiff_t Tbar_length = Tbar_max - Tbar_min;
    const std::ptrdiff_t Ttilde_length = Ttilde_max - Ttilde_min;

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    Tbar_min_arg = mod(Tbar_min, Tbar_ringbuf);
    Tbar_max_arg = mod(Tbar_min, Tbar_ringbuf) + Tbar_length;
    Ttilde_min_arg = mod(Ttilde_min, Ttilde_ringbuf);
    Ttilde_max_arg = mod(Ttilde_min, Ttilde_ringbuf) + Ttilde_length;

    // Copy inputs to device memory
    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                {{^isoutput}}
                    if constexpr (args::{{{name}}} != args::S)
                        CHECK_CUDA_ERROR(cudaMemcpyAsync({{{name}}}_memory,
                                                         host_{{{name}}}_buffer.data(),
                                                         {{{name}}}_length_in_bytes,
                                                         cudaMemcpyHostToDevice,
                                                         device.getStream(cuda_stream_id)));
                {{/isoutput}}
            {{/hasbuffer}}
        {{/isscalar}}
    {{/kernel_arguments}}

    if (poison_buffers) {
        I_buffer.set_to_poison(0xff); // 0xffff is NaN16
        info_buffer.set_to_poison(0xff);

        // Initialize host-side buffer arrays
        {{#kernel_arguments}}
            {{^isscalar}}
                {{^hasbuffer}}
                    {{#isoutput}}
                        CHECK_CUDA_ERROR(cudaMemsetAsync({{{name}}}_memory,
                                                         0xff,
                                                         {{{name}}}_length_in_bytes,
                                                         device.getStream(cuda_stream_id)));
                    {{/isoutput}}
                {{/hasbuffer}}
            {{/isscalar}}
        {{/kernel_arguments}}
    } // if (poison_buffers)

    const std::string symname = "{{{kernel_name}}}_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    const int blocks = num_blocks;
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

    if (poison_buffers) {
        // Copy results back to host memory
        {{#kernel_arguments}}
            {{^isscalar}}
                {{^hasbuffer}}
                    {{#isoutput}}
                        CHECK_CUDA_ERROR(cudaMemcpyAsync(host_{{{name}}}_buffer.data(),
                                                         {{{name}}}_memory,
                                                         {{{name}}}_length_in_bytes,
                                                         cudaMemcpyDeviceToHost,
                                                         device.getStream(cuda_stream_id)));
                    {{/isoutput}}
                {{/hasbuffer}}
            {{/isscalar}}
        {{/kernel_arguments}}

        CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));

        // Check error codes
        const std::uint32_t error_code = *std::max_element(
            (const std::uint32_t*)host_info_buffer.data(),
            (const std::uint32_t*)(host_info_buffer.data() +
                                   blocks * info_lengths[info_index_warp] * info_lengths[info_index_thread]));
        if (error_code != 0)
            ERROR("CUDA kernel {{{kernel_name}}} returned error code: {}", error_code);

        if (error_code != 0) {
            // TODO: Introduce a new "unbuffered" buffer; do this there
            // Our `info` buffer is too large (`blocks` vs. `max_blocks`)
            for (int block = 0; block < blocks; ++block) {
                for (int warp = 0; warp < info_lengths[info_index_warp]; ++warp) {
                    for (int thread = 0; thread < info_lengths[info_index_thread]; ++thread) {
                        const std::ptrdiff_t i =
                            info_strides[info_index_thread] * thread +
                            info_strides[info_index_warp] * warp +
                            info_strides[info_index_block] * block;
                        const std::uint32_t val = host_info_buffer.data()[i];
                        if (val != 0)
                            ERROR("CUDA kernel {{{kernel_name}}} returned 'info' value {:d} "
                                  "for thread {:d} warp {:d} block {:d} at index {:d} (zero indicates no error)",
                                  val, thread, warp, block, i);
                    }
                }
            }
        }

        I_buffer.check_for_poison(0xff);
    } // if (poison_buffers)

    return record_end_event();
}

void cuda{{{kernel_name}}}::finalize_frame() {
    // Advance the input ring buffers
    Ebar_buffer.finish_read();
    {{#kernel_arguments}}
        {{#haslifetime}}
            {{{name}}}_buffer.finish_read();
        {{/haslifetime}}
    {{/kernel_arguments}}

    // Advance the output ring buffer
    I_buffer.finish_write();

    cudaCommand::finalize_frame();
}
