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
#include "div.hpp"
#include "ringbuffer.hpp"

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
        {{^isscalar}}
            {{#hasbuffer}}
                {{#hasringbuffer}}
                    {{{name}}}_buffer(
                        {{{name}}}_name, {{{name}}}_quantity, reverse({{{name}}}_lengths), reverse({{{name}}}_labels), *this),
                {{/hasringbuffer}}
                {{^hasringbuffer}}
                    {{{name}}}_buffer(
                        {{{name}}}_name, {{{name}}}_quantity, reverse({{{name}}}_lengths), reverse({{{name}}}_labels), *this
                        {{#do_once}}
                            , buffer_type_t::do_once
                        {{/do_once}}
                        ),
                {{/hasringbuffer}}
            {{/hasbuffer}}
            {{^hasbuffer}}
                {{{name}}}_buffer(
                    {{{name}}}_name, {{{name}}}_quantity, reverse({{{name}}}_lengths), reverse({{{name}}}_labels), *this),
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
    return num_produced_elements(num_available_elements) * cuda_downsampling_factor;
}
std::int64_t cuda{{{kernel_name}}}::num_produced_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) / cuda_downsampling_factor;
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
    const std::ptrdiff_t Tbar_ringbuf = Ebar_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t Tbar_read_max = Tbar_ringbuf / 4;
    std::ptrdiff_t Tbar_read = -1;
    {
        const int errcode = Ebar_buffer.wait_and_claim_readable([&](const std::ptrdiff_t Tbar_available) {
            using std::min;
            Tbar_read = min(Tbar_available, Tbar_read_max);
            return read_descriptor_t{.claimed = num_consumed_elements(Tbar_read), .read = num_processed_elements(Tbar_read)};
        });
        if (errcode < 0)
            return errcode;
    }
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
                        if (!(metadata->get_name() == quantity))
                            ERROR("buffer name: {:s}, quantity: {:s}, metadata name: {:s}", Ebar_buffer.get_buffer_name(), quantity,
                                  metadata->get_name());
                        assert(metadata->get_name() == quantity);
                        const auto& ndarray = Ebar_buffer.get_ndarray();
                        assert(metadata->type == ndarray.value_datatype);
                        assert(metadata->dims == ndarray.rank);
                        for (std::size_t d = 0; d < ndarray.rank; ++d) {
                            assert(metadata->get_dimension_name(d) == dimname[d]);
                            // The ring buffer direction is special
                            if (d > 0)
                                assert(metadata->dim[d] == int(ndarray.extent(d)));
                            assert(metadata->stride[d] == ndarray.stride(d));
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
        assert(Telescope::instance().get_grid_size_x() <= cuda_dish_layout_N);
        assert(Telescope::instance().get_grid_size_y() <= cuda_dish_layout_M);

        // Allocate metadata of I buffer only once
        const bool I_has_metadata = I_buffer.has_metadata();
        assert(!I_has_metadata);
        I_buffer.set_metadata(Ebar_meta);
        auto I_meta = I_buffer.get_metadata();

        const auto Ebar_nfreq = Ebar_meta->get_nfreq();
        const auto I_nfreq = I_meta->dim[I_rank - 1 - I_index_Fbar];
        assert(I_nfreq >= 0);
        // We are not using all the non-upchannelized frequencies.
        // But we are (should be!) using all the upchannelized ones.
        assert(cuda_upchannelization_factor > 1);

        const auto Ebar_freq_upchan_factor = Ebar_meta->get_freq_upchan_factor();
        assert(Ebar_freq_upchan_factor.size() == static_cast<std::size_t>(Ebar_nfreq));
        const auto& I_freq_upchan_factor = Ebar_freq_upchan_factor;
        I_meta->set_freq_upchan_factor(I_freq_upchan_factor);

        const auto Ebar_freq_upchan_index = Ebar_meta->get_freq_upchan_index();
        assert(Ebar_freq_upchan_index.size() == static_cast<std::size_t>(Ebar_nfreq));
        const auto& I_freq_upchan_index = Ebar_freq_upchan_index;
        I_meta->set_freq_upchan_index(I_freq_upchan_index);

        const auto Ebar_coarse_freq = Ebar_meta->get_coarse_freq();
        assert(Ebar_coarse_freq.size() == static_cast<std::size_t>(Ebar_nfreq));
        const auto& I_coarse_freq = Ebar_coarse_freq;
        I_meta->set_coarse_freq(I_coarse_freq);

        const auto Ebar_time_downsampling_fpga = Ebar_meta->get_time_downsampling_fpga();
        const auto I_time_downsampling_fpga = Ebar_time_downsampling_fpga * cuda_downsampling_factor;
        I_meta->set_time_downsampling_fpga(I_time_downsampling_fpga);

        const auto W_meta = W_buffer.get_metadata();
        const auto W_nfreq = W_meta->get_nfreq();
        assert(W_nfreq == I_nfreq);
        const auto W_coarse_freq = W_meta->get_coarse_freq();
        for (int freq = 0; freq < W_nfreq; ++freq)
            assert(I_coarse_freq.at(freq) == W_coarse_freq.at(freq));

        // Since we use a ring buffer we do not need to update `meta->fpga_seq_num`
    } // if !did_set_metadata

    const auto Ebar_meta = Ebar_buffer.get_metadata();
    assert(I_buffer.has_metadata());

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
    // Advance the input ring buffer
    Ebar_buffer.finish_read();

    // Advance the output ring buffer
    I_buffer.finish_write();

    cudaCommand::finalize_frame();
}
