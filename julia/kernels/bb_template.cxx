/**
 * @file
 * @brief CUDA {{{kernel_name}}} kernel
 *
 * This file has been generated automatically.
 * Do not modify this C++ file, your changes will be lost.
 */

#include <DataType.hpp>
#include <NDArrayBuffer.hpp>
#include <NDArrayRingBuffer.hpp>
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

    // Kernel compile parameters:
    static constexpr int minthreads = {{{minthreads}}};
    static constexpr int blocks_per_sm = {{{num_blocks_per_sm}}};

    // Kernel call parameters:
    static constexpr int threads_x = {{{num_threads}}};
    static constexpr int threads_y = {{{num_warps}}};
    static constexpr int blocks = {{{num_blocks}}};
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
            static_assert({{{name}}}_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
        {{/isscalar}}
        //
    {{/kernel_arguments}}

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

    // Only one of the instances of this pipeline stage needs to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "{{{kernel_name}}}_");
    }
}

cuda{{{kernel_name}}}::~cuda{{{kernel_name}}}() {}

int cuda{{{kernel_name}}}::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    {
        const int errcode = E_buffer.wait_and_claim_readable([&](const std::ptrdiff_t T_available) {
            const std::ptrdiff_t T_read = round_down(T_available, cuda_granularity_number_of_timesamples);
            return read_descriptor_t{.claimed = T_read, .read = T_read};
        });
        if (errcode < 0)
            return errcode;
    }

    return 0;
}

cudaEvent_t cuda{{{kernel_name}}}::execute(cudaPipelineState& /*pipestate*/, const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    {{#kernel_arguments}}
        {{^isscalar}}
            void* const {{{name}}}_memory = {{{name}}}_buffer.get_ndarray().data();
        {{/isscalar}}
    {{/kernel_arguments}}

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

    record_start_event();

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

    // Ringbuffer size
    const std::ptrdiff_t T_ringbuf = E_buffer.get_ndarray().extent(0);

    const std::ptrdiff_t Tmin = E_buffer.get_begin_read_valid();
    const std::ptrdiff_t Tmax = E_buffer.get_end_read_valid();

    const std::ptrdiff_t Tlength = Tmax - Tmin;

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    Tmin_arg = mod(Tmin, T_ringbuf);
    Tmax_arg = mod(Tmin, T_ringbuf) + Tlength;

    // Update metadata
    {
        const std::shared_ptr<chordMetadata> J_meta = J_buffer.get_metadata();
    
        // Since we do not use a ring buffer we need to set `meta->sample0_offset`
        J_meta->sample0_offset = Tmin;
        assert(J_meta->offset_downsampling == 1);
    }

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

    J_buffer.set_to_poison(0x00);

#ifdef DEBUGGING
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
    DEBUG("Finished CUDA {{{kernel_name}}} on GPU frame {:d}", gpu_frame_id);

    // Check error codes
    // TODO: Introduce a new "unbuffered" buffer; do this there
    std::uint32_t error_code = 0;
    for (int block = 0; block < info_lengths[info_index_block]; ++block) {
        for (int warp = 0; warp < info_lengths[info_index_warp]; ++warp) {
            for (int thread = 0; thread < info_lengths[info_index_thread]; ++thread) {
                const std::ptrdiff_t i =
                    info_strides[info_index_thread] * thread +
                    info_strides[info_index_warp] * warp +
                    info_strides[info_index_block] * block;
                const std::uint32_t val = host_info_buffer.data()[i];
                using std::max;
                error_code = max(error_code, val);
            }
        }
    }
    if (error_code != 0)
        ERROR("CUDA kernel returned error code: {}", error_code);

    // TODO: Introduce a new "unbuffered" buffer; do this there
    for (int block = 0; block < info_lengths[info_index_block]; ++block) {
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

    // Check log codes
    const std::uint32_t log_code = *std::max_element((const std::uint32_t*)&*host_log_buffer.begin(),
                                                     (const std::uint32_t*)&*host_log_buffer.end());
    if (log_code != 0)
        WARN("CUDA kernel {{{kernel_name}}} returned log code cuLaunchKernel: {}", log_code);

    // TODO: Introduce a new "unbuffered" buffer; do this there
    for (std::size_t i = 0; i < host_log_buffer.size(); ++i) {
        const std::uint32_t val = host_log_buffer.data()[i];
        if (val != 0)
            WARN("CUDA kernel {{{kernel_name}}} returned 'log' value {:d} at index {:d} (zero "
                 "indicates success)",
                 val, i);
    }
#endif

    J_buffer.check_for_poison(0x00);

    return record_end_event();
}

void cuda{{{kernel_name}}}::finalize_frame() {
    // Advance the input ringbuffer
    E_buffer.finish_read();

    cudaCommand::finalize_frame();
}
