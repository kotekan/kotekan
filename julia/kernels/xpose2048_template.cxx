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
    const std::ptrdiff_t Tin_ringbuf = Ein_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t Tin_read_max = Tin_ringbuf / 4;
    std::ptrdiff_t Tin_read = -1;
    {
        const int errcode = Ein_buffer.wait_and_claim_readable([&](const std::ptrdiff_t Tin_available) {
            using std::min;
            Tin_read = round_down(min(Tin_available, Tin_read_max), cuda_granularity_number_of_timesamples);
                return read_descriptor_t{.claimed = Tin_read, .read = Tin_read};
        });
        if (errcode < 0)
            return errcode;
    }

    // Wait for space to be available in output ringbuffer
    const std::ptrdiff_t T_written = Tin_read;
    {
        const int errcode = E_buffer.wait_for_writable(T_written);
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

    {{#kernel_arguments}}
        {{#hasbuffer}}
            {{^isoutput}}
                if (args::{{{name}}} == args::Ein) {
                    // Replace "Ein" with "E" etc.
                    // {{{name}}}_buffer.check_metadata();
                    const std::string quantity = "E";
                    const std::array<std::string, 4> dimname = {"T", "F", "P", "D"};
                    const std::shared_ptr<const chordMetadata> metadata = Ein_buffer.get_metadata();
                    if (!(metadata->get_name() == quantity))
                        ERROR("buffer name: {:s}, quantity: {:s}, metadata name: {:s}", Ein_buffer.get_buffer_name(), quantity,
                              metadata->get_name());
                    assert(metadata->get_name() == quantity);
                    const auto& ndarray = Ein_buffer.get_ndarray();
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
                {{{name}}}_buffer.set_metadata(Ein_buffer.get_metadata());
            {{/isoutput}}
        {{/hasbuffer}}
    {{/kernel_arguments}}

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

    // Set Ein_memory to beginning of input ring buffer
    Ein_arg = array_desc(Ein_memory, Ein_length_in_bytes);

    // Set E_memory to beginning of output ring buffer
    E_arg = array_desc(E_memory, E_length_in_bytes);

    // Ringbuffer size
    const std::ptrdiff_t Tin_ringbuf = Ein_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t T_ringbuf = E_buffer.get_ndarray().extent(0);

    const std::ptrdiff_t Tin_min = Ein_buffer.get_read_valid().begin();
    const std::ptrdiff_t Tin_max = Ein_buffer.get_read_valid().end();
    const std::ptrdiff_t T_min = E_buffer.get_write_valid().begin();
    const std::ptrdiff_t T_max = E_buffer.get_write_valid().end();

    const std::ptrdiff_t Tin_length = Tin_max - Tin_min;
    const std::ptrdiff_t T_length = T_max - T_min;

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    Tin_min_arg = mod(Tin_min, Tin_ringbuf);
    Tin_max_arg = mod(Tin_min, Tin_ringbuf) + Tin_length;
    T_min_arg = mod(T_min, T_ringbuf);
    T_max_arg = mod(T_min, T_ringbuf) + T_length;

    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

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

    E_buffer.set_to_poison(0x00);

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

    E_buffer.set_to_poison(0x00);
    info_buffer.set_to_poison(0xff);

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

    // Check error codes
    const std::int32_t error_code = *std::max_element((const std::int32_t*)&*host_info_buffer.begin(),
                                                      (const std::int32_t*)&*host_info_buffer.end());
    if (error_code != 0)
        ERROR("CUDA kernel {{{kernel_name}}} returned error code: {}", error_code);

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
#endif

    E_buffer.check_for_poison(0x00);

    return record_end_event();
}

void cuda{{{kernel_name}}}::finalize_frame() {
    // Advance the input ring buffer
    Ein_buffer.finish_read();

    // Advance the output ring buffer
    E_buffer.finish_write();

    cudaCommand::finalize_frame();
}
