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
#include "div.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <fmt.hpp>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::round_down, kotekan::div_noremainder, kotekan::div, kotekan::mod;

namespace {
template<typename T, std::size_t D>
std::array<T, D> reverse(const std::array<T, D>& values) {
    std::array<T, D> result;
    for (std::size_t d=0; d<D; ++d)
        result[d] = values[D - 1 - d];
    return result;
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
        {{^isscalar}}
            {{#hasbuffer}}
                {{#hasringbuffer}}
                    {{{name}}}_buffer(
                        {{{name}}}_name,
                        {{{name}}}_quantity,
                        reverse({{{name}}}_lengths),
                        reverse({{{name}}}_labels),
                        reverse({{{name}}}_dimscalings),
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

    dummy()                      // avoid trailing comma
{
    // Register host memory
    {{#kernel_arguments}}
        {{^isscalar}}
            {{^hasbuffer}}
                {
                    const cudaError_t ierr = cudaHostRegister(host_{{{name}}}_buffer.data(),
                                                              host_{{{name}}}_buffer.size() * sizeof *host_{{{name}}}_buffer.data(),
                                                              0);
                    assert(ierr == cudaSuccess);
                }
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

    set_command_type(gpuCommandType::KERNEL);

    // Build the PTX only once
    static std::once_flag build_ptx_flag;
    std::call_once(build_ptx_flag, [&]() {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx("lib/cuda/generated/{{{kernel_name}}}.ptx", {kernel_symbol}, opts, "{{{kernel_name}}}_");
    });
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

    // Wait for data to be available in input ringbuffer
    const std::ptrdiff_t T_ringbuf = E_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t T_read_max = T_ringbuf / 4;
    std::ptrdiff_t T_read = -1;
    {
        const int errcode = E_buffer.wait_and_claim_readable([&](const std::ptrdiff_t T_available) {
            using std::min;
            T_read = min(T_available, T_read_max);
            return read_descriptor_t{.claimed = num_consumed_elements(T_read), .read = num_processed_elements(T_read)};
        });
        if (errcode < 0)
            return errcode;
    }
    const std::ptrdiff_t T_written = num_produced_elements(T_read);

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
    static bool did_set_metadata = false;
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
        const auto Ebar_nfreq = cuda_upchannelization_factor * (Fmax - Fmin);
        assert(Ebar_nfreq >= 0);
        assert(Ebar_nfreq <= cuda_upchannelization_factor * E_nfreq);

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
        assert(G_nfreq == Ebar_nfreq);
        const auto G_coarse_freq = G_meta->get_coarse_freq();
        for (int freq = 0; freq < Ebar_nfreq; ++freq)
            assert(Ebar_coarse_freq.at(freq) == G_coarse_freq.at(freq));

        assert(E_meta->dim[E_rank - 1 - E_index_F] == E_nfreq);
        assert(G_meta->dim[G_rank - 1 - G_index_Fbar] >= G_nfreq);
        assert(Ebar_meta->dim[Ebar_rank - 1 - Ebar_index_Fbar] >= Ebar_nfreq);

        // Since we use a ring buffer we do not need to update `meta->fpga_seq_num`
    } // if !did_set_metadata

    assert(Ebar_buffer.has_metadata());

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
    assert(0 <= blocks);
    assert(blocks <= max_blocks);

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
        Ebar_buffer.set_to_poison(0x00, 0, cuda_upchannelization_factor * (Fmax - Fmin));
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

        Ebar_buffer.check_for_poison(0x00, 0, cuda_upchannelization_factor * (Fmax - Fmin));
    } // if (poison_buffers)

    return record_end_event();
}

void cuda{{{kernel_name}}}::finalize_frame() {
    // Advance the input ring buffer
    E_buffer.finish_read();

    // Advance the output ring buffer
    Ebar_buffer.finish_write();

    cudaCommand::finalize_frame();
}
