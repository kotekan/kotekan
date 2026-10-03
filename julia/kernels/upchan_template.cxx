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
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaUtils.hpp"
#include "div.hpp"
#include "lifetimeWindows.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fmt.hpp>
#include <limits>
#include <mutex>
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
// that are valid for a configurable number of FPGA samples, such as the gains.
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
        CuDeviceArray(void* const ptr, const std::ptrdiff_t bytes) :
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

    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

    // We read at most a quarter of the input ring buffer per kernel invocation
    static constexpr std::int64_t max_granules_per_window =
        cuda_max_number_of_timesamples / 4 / cuda_granularity_number_of_timesamples;
    static_assert(kotekan::LifetimeWindows::valid(cuda_granularity_number_of_timesamples,
                                                  cuda_algorithm_overlap,
                                                  cuda_upchannelization_factor,
                                                  max_granules_per_window));
    // Chooses the input windows so that none of them straddles the end of a gain lifetime
    const kotekan::LifetimeWindows windows;

    // Kernel compile parameters:
    static constexpr int minthreads = {{{minthreads}}};
    static constexpr int blocks_per_sm = {{{num_blocks_per_sm}}};
    static constexpr int blocks_per_frequency = {{{num_blocks_per_frequency}}};

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

    // How many frequencies we will process
    const int Fmin, Fmax;

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

    // Set once, on the first frame; see `NDArrayRingBuffer::set_metadata`
    bool did_set_metadata;

    // To avoid trailing comma below
    int dummy;
};

REGISTER_CUDA_COMMAND(cuda{{{kernel_name}}});

cuda{{{kernel_name}}}::cuda{{{kernel_name}}}(Config& config,
                                             const std::string& unique_name,
                                             bufferContainer& host_buffers,
                                             cudaDeviceInterface& device,
                                             const int instance_num):
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
        "{{{kernel_name}}}", "{{{kernel_name}}}.ptx"),
    windows(cuda_granularity_number_of_timesamples, cuda_algorithm_overlap,
            cuda_upchannelization_factor, max_granules_per_window),

    Fmin(config.get<int>(unique_name, "Fmin")),
    Fmax(config.get<int>(unique_name, "Fmax")),

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

    did_set_metadata(false),
    dummy()                      // avoid trailing comma
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

    // The gains are held in a ring buffer, one element per lifetime, and read without claiming.
    // Unlike the baseband beamformer, whose output frame fixes the number of time samples per
    // invocation, we can shrink a read so that it ends exactly on a lifetime boundary. Because
    // of the PFB overlap not every size is possible; `LifetimeWindows` chooses them, and the
    // lifetime has to be a whole number of possible windows.
    if (G_lifetime_in_samples <= 0 || G_lifetime_in_samples % cuda_upchannelization_factor != 0
        || !windows.reachable(G_lifetime_in_samples / cuda_upchannelization_factor))
        FATAL_ERROR("upchan_gain_lifetime_in_samples {:d} must be a positive multiple of the "
                    "upchannelization factor {:d} that whole windows of kernel "
                    "{{{kernel_name}}} can tile exactly: a window reads n*{:d} input samples for "
                    "n in [{:d},{:d}] and produces n*{:d}-{:d} output samples",
                    G_lifetime_in_samples, int(cuda_upchannelization_factor),
                    int(cuda_granularity_number_of_timesamples), windows.min_granules(),
                    windows.max_granules(),
                    int(cuda_granularity_number_of_timesamples / cuda_upchannelization_factor),
                    int(cuda_algorithm_overlap / cuda_upchannelization_factor));

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
    if (num_processed_elements(num_available_elements) < cuda_algorithm_overlap)
        return 0;
    return num_processed_elements(num_available_elements) - cuda_algorithm_overlap;
}
std::int64_t cuda{{{kernel_name}}}::num_produced_elements(std::int64_t num_available_elements) const {
    return div_noremainder(num_consumed_elements(num_available_elements), cuda_upchannelization_factor);
}

std::int64_t cuda{{{kernel_name}}}::num_processed_elements(std::int64_t num_available_elements) const {
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cuda{{{kernel_name}}}::wait_on_precondition() {
    {
        const int errcode = cudaCommand::wait_on_precondition();
        if (errcode < 0)
            return errcode;
    }

    // Which gain element covers the data we are about to read? Ask the ring buffer where our
    // next read will begin. We must not use our own `read_valid` for this: every instance of
    // this command shares one ring buffer read head, so our own position lags it by whatever
    // the other instances have claimed since our previous frame. Output sample `Tbar` is
    // computed from the input samples starting at `U * Tbar`, and it uses the gain element
    // covering that input sample.
    const std::ptrdiff_t T_begin = E_buffer.peek_read_head();
    if (T_begin < 0)
        return -1; // shutting down
    // Each window claims a whole number of output samples' worth of input
    assert(T_begin % cuda_upchannelization_factor == 0);
    // (`kotekan::div` must be qualified; an unqualified `div` finds C's `::div`)
    const std::ptrdiff_t G_element = kotekan::div(T_begin, G_lifetime_in_samples);
    const std::ptrdiff_t G_lifetime_end = (G_element + 1) * G_lifetime_in_samples;
    // Output samples left until the end of this gain element's lifetime
    const std::ptrdiff_t Tbar_remaining =
        div_noremainder(G_lifetime_end - T_begin, cuda_upchannelization_factor);

    // Wait for data to be available in input ringbuffer
    const std::ptrdiff_t T_ringbuf = E_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t T_read_max = T_ringbuf / 4;
    assert(T_read_max == max_granules_per_window * cuda_granularity_number_of_timesamples);
    std::ptrdiff_t T_read = -1;
    {
        const int errcode = E_buffer.wait_and_claim_readable([&](const std::ptrdiff_t T_available) {
            using std::min;
            // Read as much as we can, but end exactly on the gain lifetime boundary when we
            // reach it, and never leave a remainder that whole windows cannot fill. If no
            // window fits the available data we read nothing and wait for more.
            const std::ptrdiff_t available_granules =
                min(T_available, T_read_max) / cuda_granularity_number_of_timesamples;
            T_read = windows.num_granules(Tbar_remaining, available_granules)
                     * cuda_granularity_number_of_timesamples;
            // Ensure that we make progress: If we cannot claim any elements then we
            // must not read any elements either, and instead wait for more data.
            const std::ptrdiff_t T_claimed = num_consumed_elements(T_read);
            const std::ptrdiff_t T_processed = T_claimed == 0 ? 0 : num_processed_elements(T_read);
            return read_descriptor_t{.claimed = T_claimed, .read = T_processed};
        });
        if (errcode < 0)
            return errcode;
    }
    const std::ptrdiff_t T_written = num_produced_elements(T_read);
    // The window must end on or before the end of the gain element's lifetime. This cannot
    // fail unless the scheduler is wrong, but a violation would silently apply the wrong gains.
    if (!(E_buffer.get_read_claimed().begin() == T_begin
          && E_buffer.get_read_claimed().end() <= G_lifetime_end))
        FATAL_ERROR("Kernel {{{kernel_name}}} claimed input samples [{:d},{:d}), which straddle "
                    "the end {:d} of gain element {:d} or do not begin at the read head {:d}",
                    E_buffer.get_read_claimed().begin(), E_buffer.get_read_claimed().end(),
                    G_lifetime_end, G_element, T_begin);

    // Read the gain element covering these data. We claim it only when we have reached the end
    // of its lifetime, i.e. when this is the last frame that will use it. Until then it stays
    // in the ring buffer and the following frames read it again.
    //
    // We are holding a claim on `E` while we wait here. That is safe because the gain producer
    // does not depend on `E` being drained: it is clocked only by the first voltage frame.
    {
        const bool G_last_use = E_buffer.get_read_claimed().end() == G_lifetime_end;
        DEBUG("Waiting for G input ringbuffer data for frame {:d}...", gpu_frame_id);
        const int errcode = G_buffer.wait_and_claim_readable(
            [&](const std::ptrdiff_t available_elements) {
                if (available_elements < 1)
                    return read_descriptor_t{.claimed = 0, .read = 0};
                return read_descriptor_t{.claimed = G_last_use ? 1 : 0, .read = 1};
            });
        if (errcode < 0)
            return errcode;
        DEBUG("Done waiting for G input ringbuffer data for frame {:d}; output samples "
              "[{:d},{:d}) using element {:d}{:s}",
              gpu_frame_id, T_begin / cuda_upchannelization_factor,
              T_begin / cuda_upchannelization_factor + T_written, G_element,
              G_last_use ? " (last use)" : "");
        // The two ring buffers must agree on which gain element covers these data
        assert(G_buffer.get_read_valid().begin() == G_element);
    }

    // Wait for space to be available in output ringbuffer
    {
        const int errcode = Ebar_buffer.wait_for_writable(T_written);
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
                    {{{name}}}_buffer.check_metadata();
                {{/isoutput}}
                {{#isoutput}}
                    {{{name}}}_buffer.set_metadata(E_buffer.get_metadata());
                {{/isoutput}}
            {{/hasbuffer}}
        {{/kernel_arguments}}

        const auto E_meta = E_buffer.get_metadata();
        auto Ebar_meta = Ebar_buffer.get_metadata();

        const auto E_nfreq = E_meta->get_nfreq();
        // `Fmin` and `Fmax` come from the config; they select the coarse frequencies this
        // kernel upchannelizes and must lie inside the input buffer.
        if (!(0 <= Fmin && Fmin <= Fmax && Fmax <= E_nfreq))
            FATAL_ERROR("Invalid frequency span [{:d},{:d}) for kernel {{{kernel_name}}}: input "
                        "buffer E holds {:d} frequencies",
                        Fmin, Fmax, E_nfreq);
        const auto Ebar_nfreq = cuda_upchannelization_factor * (Fmax - Fmin);

        const auto E_freq_upchan_factor = E_meta->get_freq_upchan_factor();
        std::vector<int> Ebar_freq_upchan_factor(Ebar_nfreq);
        for (int freq = 0; freq < Ebar_nfreq; ++freq) {
            const int coarse_freq = Fmin + freq / cuda_upchannelization_factor;
            assert(coarse_freq < Fmax);
            Ebar_freq_upchan_factor.at(freq) = E_freq_upchan_factor.at(coarse_freq) * cuda_upchannelization_factor;
        }
        Ebar_meta->set_freq_upchan_factor(Ebar_freq_upchan_factor);

        const auto E_freq_upchan_index = E_meta->get_freq_upchan_index();
        std::vector<int> Ebar_freq_upchan_index(Ebar_nfreq);
        for (int freq = 0; freq < Ebar_nfreq; ++freq) {
            const int upchan_index =  freq % cuda_upchannelization_factor;
            Ebar_freq_upchan_index.at(freq) = upchan_index;
        }
        Ebar_meta->set_freq_upchan_index(Ebar_freq_upchan_index);

        const auto E_time_downsampling_fpga = E_meta->get_time_downsampling_fpga();
        const auto Ebar_time_downsampling_fpga = E_time_downsampling_fpga * cuda_upchannelization_factor;
        Ebar_meta->set_time_downsampling_fpga(Ebar_time_downsampling_fpga);

        const auto E_coarse_freq = E_meta->get_coarse_freq();
        std::vector<int> Ebar_coarse_freq(Ebar_nfreq);
        for (int freq = 0; freq < Ebar_nfreq; ++freq) {
            const int coarse_freq = Fmin + freq / cuda_upchannelization_factor;
            assert(coarse_freq < Fmax);
            Ebar_coarse_freq.at(freq) = E_coarse_freq.at(coarse_freq);
        }
        Ebar_meta->set_coarse_freq(Ebar_coarse_freq);

        const auto G_meta = G_buffer.get_metadata();
        const auto G_nfreq = G_meta->get_nfreq();
        // Mismatched gains would scale each frequency by another frequency's gain.
        if (G_nfreq != Ebar_nfreq)
            FATAL_ERROR("Gain buffer G holds {:d} frequencies, but kernel {{{kernel_name}}} "
                        "produces {:d}",
                        G_nfreq, Ebar_nfreq);
        const auto G_coarse_freq = G_meta->get_coarse_freq();
        for (int freq = 0; freq < Ebar_nfreq; ++freq)
            if (Ebar_coarse_freq.at(freq) != G_coarse_freq.at(freq))
                FATAL_ERROR("Gain buffer G is for coarse frequency {:d} at index {:d}, but kernel "
                            "{{{kernel_name}}} produces coarse frequency {:d} there",
                            G_coarse_freq.at(freq), freq, Ebar_coarse_freq.at(freq));

        // The buffers must be large enough for the frequencies we are about to read and write.
        if (E_meta->dim[E_rank - 1 - E_index_F] != E_nfreq)
            FATAL_ERROR("Input buffer E reports {:d} frequencies, but its frequency dimension has "
                        "extent {:d}",
                        E_nfreq, E_meta->dim[E_rank - 1 - E_index_F]);
        if (G_meta->dim[G_rank - 1 - G_index_Fbar] < G_nfreq)
            FATAL_ERROR("Gain buffer G holds {:d} frequencies, but its frequency dimension has "
                        "extent {:d}",
                        G_nfreq, G_meta->dim[G_rank - 1 - G_index_Fbar]);
        if (Ebar_meta->dim[Ebar_rank - 1 - Ebar_index_Fbar] < Ebar_nfreq)
            FATAL_ERROR("Kernel {{{kernel_name}}} produces {:d} frequencies, but the frequency "
                        "dimension of its output buffer Ebar has extent {:d}",
                        Ebar_nfreq, Ebar_meta->dim[Ebar_rank - 1 - Ebar_index_Fbar]);

        // The gain lifetime is counted in input samples, so these must be FPGA samples
        if (E_meta->get_time_downsampling_fpga() != 1)
            FATAL_ERROR("Input buffer E has time_downsampling_fpga={:d}, but kernel "
                        "{{{kernel_name}}} counts the gain lifetime in input samples and requires "
                        "an input sampled at the FPGA rate",
                        E_meta->get_time_downsampling_fpga());

        // Element `k` of a slowly varying input covers the samples `k * lifetime` onwards,
        // counted from the voltage ring buffer's logical beginning -- so the two streams have
        // to start at the same sequence number. The metadata of both ring buffers are fixed,
        // so checking once suffices.
        {{#kernel_arguments}}
            {{#haslifetime}}
                if ({{{name}}}_buffer.get_metadata()->get_fpga_seq_num() != E_meta->get_fpga_seq_num())
                    FATAL_ERROR("Buffer {{{name}}} begins at FPGA sequence number {:d}, but the "
                                "voltage buffer E begins at {:d}; kernel {{{kernel_name}}} requires "
                                "them to be aligned",
                                {{{name}}}_buffer.get_metadata()->get_fpga_seq_num(),
                                E_meta->get_fpga_seq_num());
            {{/haslifetime}}
        {{/kernel_arguments}}

        // Since we use a ring buffer we do not need to update `meta->fpga_seq_num`
    } // if !did_set_metadata

    if (!Ebar_buffer.has_metadata())
        FATAL_ERROR("Output buffer Ebar has no metadata; kernel {{{kernel_name}}} cannot run");

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

    // Set E_memory to beginning of input ring buffer
    E_arg = array_desc(E_memory, E_length_in_bytes);

    // Set Ebar_memory to beginning of output ring buffer
    Ebar_arg = array_desc(Ebar_memory, Ebar_length_in_bytes);

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
    const std::ptrdiff_t T_ringbuf = E_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t Tbar_ringbuf = Ebar_buffer.get_ndarray().extent(0);

    const std::ptrdiff_t T_min = E_buffer.get_read_valid().begin();
    const std::ptrdiff_t T_max = E_buffer.get_read_valid().end();
    const std::ptrdiff_t Tbar_min = Ebar_buffer.get_write_valid().begin();
    const std::ptrdiff_t Tbar_max = Ebar_buffer.get_write_valid().end();

    const std::ptrdiff_t T_length = T_max - T_min;
    const std::ptrdiff_t Tbar_length = Tbar_max - Tbar_min;

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    T_min_arg = mod(T_min, T_ringbuf);
    T_max_arg = mod(T_min, T_ringbuf) + T_length;
    Tbar_min_arg = mod(Tbar_min, Tbar_ringbuf);
    Tbar_max_arg = mod(Tbar_min, Tbar_ringbuf) + Tbar_length;

    // Pass frequency spans to kernel
    Fmin_arg = Fmin;
    Fmax_arg = Fmax;
    const int blocks = blocks_per_frequency * (Fmax - Fmin);
    // `blocks` is both the CUDA grid size and the extent of the block dimension of the `info`
    // buffer. Launching more than `max_blocks` blocks would make the kernel write its status
    // words past the end of that device allocation.
    if (!(0 <= blocks && blocks <= max_blocks))
        FATAL_ERROR("Kernel {{{kernel_name}}} would launch {:d} blocks, but the `info` buffer "
                    "holds only {:d} (Fmin={:d}, Fmax={:d}, blocks_per_frequency={:d})",
                    blocks, int(max_blocks), Fmin, Fmax, int(blocks_per_frequency));

    // Copy inputs to device memory
    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                {{^isoutput}}
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
        Ebar_buffer.set_to_poison({{{ebar_poison_byte}}}, 0, cuda_upchannelization_factor * (Fmax - Fmin));
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

        Ebar_buffer.check_for_poison({{{ebar_poison_byte}}}, 0, cuda_upchannelization_factor * (Fmax - Fmin));
    } // if (poison_buffers)

    return record_end_event();
}

void cuda{{{kernel_name}}}::finalize_frame() {
    // Advance the input ring buffers
    E_buffer.finish_read();
    {{#kernel_arguments}}
        {{#haslifetime}}
            {{{name}}}_buffer.finish_read();
        {{/haslifetime}}
    {{/kernel_arguments}}

    // Advance the output ring buffer
    Ebar_buffer.finish_write();

    cudaCommand::finalize_frame();
}
