/**
 * @file
 * @brief CUDA Transpose2048_chime kernel
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
    for (std::size_t d = 0; d < D; ++d)
        result[d] = values[D - 1 - d];
    return result;
}
} // namespace

/**
 * @class cudaTranspose2048_chime
 * @brief cudaCommand for Transpose2048_chime
 */
class cudaTranspose2048_chime : public cudaCommand {
public:
    cudaTranspose2048_chime(Config& config, const std::string& unique_name,
                            bufferContainer& host_buffers, cudaDeviceInterface& device,
                            const int instance_num);
    virtual ~cudaTranspose2048_chime();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
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
            ptr(static_cast<T*>(ptr)), maxsize(bytes), dims{std::int64_t(maxsize / sizeof(T))},
            len(maxsize / sizeof(T)) {}
    };
    using array_desc = CuDeviceArray<std::int32_t, 1>;

    // Kernel design parameters:
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 1024;
    static constexpr int cuda_number_of_frequencies = 16;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_max_number_of_timesamples = 65536;
    static constexpr int cuda_granularity_number_of_timesamples = 32;

    // Kernel compile parameters:
    static constexpr int minthreads = 512;
    static constexpr int blocks_per_sm = 1;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 16;
    static constexpr int blocks = 128;
    static constexpr int shmem_bytes = 65536;

    // Kernel name:
    static constexpr const char* kernel_symbol =
        "_Z16xpose2048_kernel5Int32S_S_S_13CuDeviceArrayI6Int4x8Li1ELi1EES2_S0_IS_Li1ELi1EES3_";

    // Kernel arguments:
    enum class args { Tin_min, Tin_max, T_min, T_max, Ein, E, scatter_indices, info, count };

    // Tin_min: Tin_min
    static constexpr const char* Tin_min_quantity = "Tin_min";
    static constexpr kotekan::DataType Tin_min_type = kotekan::int32;
    //
    // Tin_max: Tin_max
    static constexpr const char* Tin_max_quantity = "Tin_max";
    static constexpr kotekan::DataType Tin_max_type = kotekan::int32;
    //
    // T_min: T_min
    static constexpr const char* T_min_quantity = "T_min";
    static constexpr kotekan::DataType T_min_type = kotekan::int32;
    //
    // T_max: T_max
    static constexpr const char* T_max_quantity = "T_max";
    static constexpr kotekan::DataType T_max_type = kotekan::int32;
    //
    // Ein: input_voltage_name
    static constexpr const char* Ein_quantity = "Ein";
    static constexpr kotekan::DataType Ein_type = kotekan::int4x2chime;
    enum Ein_indices {
        Ein_index_D,
        Ein_index_P,
        Ein_index_F,
        Ein_index_T,
        Ein_rank,
    };
    static constexpr std::array<const char*, Ein_rank> Ein_labels = {
        "D",
        "P",
        "F",
        "T",
    };
    static constexpr std::array<std::ptrdiff_t, Ein_rank> Ein_lengths = {
        1024,
        2,
        16,
        65536,
    };
    static constexpr auto Ein_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= Ein_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, Ein_rank + 1> Ein_strides = {
        Ein_calc_stride(Ein_index_D), Ein_calc_stride(Ein_index_P), Ein_calc_stride(Ein_index_F),
        Ein_calc_stride(Ein_index_T), Ein_calc_stride(Ein_rank),
    };
    static constexpr std::ptrdiff_t Ein_length = Ein_strides[Ein_rank];
    static constexpr std::ptrdiff_t Ein_length_in_bytes = type_total_bytes(Ein_type) * Ein_length;
    static_assert(Ein_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    //
    // E: output_voltage_name
    static constexpr const char* E_quantity = "E";
    static constexpr kotekan::DataType E_type = kotekan::int4x2chime;
    enum E_indices {
        E_index_D,
        E_index_P,
        E_index_F,
        E_index_T,
        E_rank,
    };
    static constexpr std::array<const char*, E_rank> E_labels = {
        "D",
        "P",
        "F",
        "T",
    };
    static constexpr std::array<std::ptrdiff_t, E_rank> E_lengths = {
        1024,
        2,
        16,
        65536,
    };
    static constexpr auto E_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= E_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, E_rank + 1> E_strides = {
        E_calc_stride(E_index_D), E_calc_stride(E_index_P), E_calc_stride(E_index_F),
        E_calc_stride(E_index_T), E_calc_stride(E_rank),
    };
    static constexpr std::ptrdiff_t E_length = E_strides[E_rank];
    static constexpr std::ptrdiff_t E_length_in_bytes = type_total_bytes(E_type) * E_length;
    static_assert(E_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    //
    // scatter_indices: scatter_indices_name
    static constexpr const char* scatter_indices_quantity = "scatter_indices";
    static constexpr kotekan::DataType scatter_indices_type = kotekan::int32;
    enum scatter_indices_indices {
        scatter_indices_index_D,
        scatter_indices_index_P,
        scatter_indices_rank,
    };
    static constexpr std::array<const char*, scatter_indices_rank> scatter_indices_labels = {
        "D",
        "P",
    };
    static constexpr std::array<std::ptrdiff_t, scatter_indices_rank> scatter_indices_lengths = {
        1024,
        2,
    };
    static constexpr auto scatter_indices_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= scatter_indices_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, scatter_indices_rank + 1> scatter_indices_strides =
        {
            scatter_indices_calc_stride(scatter_indices_index_D),
            scatter_indices_calc_stride(scatter_indices_index_P),
            scatter_indices_calc_stride(scatter_indices_rank),
        };
    static constexpr std::ptrdiff_t scatter_indices_length =
        scatter_indices_strides[scatter_indices_rank];
    static constexpr std::ptrdiff_t scatter_indices_length_in_bytes =
        type_total_bytes(scatter_indices_type) * scatter_indices_length;
    static_assert(scatter_indices_length_in_bytes
                  <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    //
    // info: gpu_mem_info
    static constexpr const char* info_quantity = "info";
    static constexpr kotekan::DataType info_type = kotekan::int32;
    enum info_indices {
        info_index_thread,
        info_index_warp,
        info_index_block,
        info_rank,
    };
    static constexpr std::array<const char*, info_rank> info_labels = {
        "thread",
        "warp",
        "block",
    };
    static constexpr std::array<std::ptrdiff_t, info_rank> info_lengths = {
        32,
        16,
        128,
    };
    static constexpr auto info_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= info_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, info_rank + 1> info_strides = {
        info_calc_stride(info_index_thread),
        info_calc_stride(info_index_warp),
        info_calc_stride(info_index_block),
        info_calc_stride(info_rank),
    };
    static constexpr std::ptrdiff_t info_length = info_strides[info_rank];
    static constexpr std::ptrdiff_t info_length_in_bytes =
        type_total_bytes(info_type) * info_length;
    static_assert(info_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    //

    // Kotekan buffer names
    const std::string Ein_name;
    const std::string E_name;
    const std::string scatter_indices_name;
    const std::string info_name;

    // Buffers
    NDArrayRingBuffer<kotekan::GetType_t<Ein_type>, Ein_rank> Ein_buffer;
    NDArrayRingBuffer<kotekan::GetType_t<E_type>, E_rank> E_buffer;
    NDArrayBuffer<kotekan::GetType_t<scatter_indices_type>, scatter_indices_rank>
        scatter_indices_buffer;
    NDArrayBuffer<kotekan::GetType_t<info_type>, info_rank> info_buffer;
    std::vector<kotekan::GetType_t<info_type>> host_info_buffer;

    // To avoid trailing comma below
    int dummy;
};

REGISTER_CUDA_COMMAND(cudaTranspose2048_chime);

cudaTranspose2048_chime::cudaTranspose2048_chime(Config& config, const std::string& unique_name,
                                                 bufferContainer& host_buffers,
                                                 cudaDeviceInterface& device,
                                                 const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "Transpose2048_chime", "Transpose2048_chime.ptx"),

    Ein_name(config.get<std::string>(unique_name, "input_voltage_name")),
    E_name(config.get<std::string>(unique_name, "output_voltage_name")),
    scatter_indices_name(config.get<std::string>(unique_name, "scatter_indices_name")),
    info_name(unique_name + "/gpu_mem_info"),

    Ein_buffer(Ein_name, Ein_quantity, reverse(Ein_lengths), reverse(Ein_labels), *this),
    E_buffer(E_name, E_quantity, reverse(E_lengths), reverse(E_labels), *this),
    scatter_indices_buffer(scatter_indices_name, scatter_indices_quantity,
                           reverse(scatter_indices_lengths), reverse(scatter_indices_labels), *this,
                           buffer_type_t::do_once),
    info_buffer(info_name, info_quantity, reverse(info_lengths), reverse(info_labels), *this),
    host_info_buffer(info_length),

    dummy() // avoid trailing comma
{
    // Register host memory
    {
        const cudaError_t ierr = cudaHostRegister(
            host_info_buffer.data(), host_info_buffer.size() * sizeof *host_info_buffer.data(), 0);
        assert(ierr == cudaSuccess);
    }

    Ein_buffer.register_consumer();
    E_buffer.register_producer();
    scatter_indices_buffer.register_consumer();
    register_gpu_buffer_user(
        {.name = info_name, .is_array = true, .does_read = true, .does_write = true});

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage needs to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "Transpose2048_chime_");
    }
}

cudaTranspose2048_chime::~cudaTranspose2048_chime() {}

int cudaTranspose2048_chime::wait_on_precondition() {
    {
        const int errcode = cudaCommand::wait_on_precondition();
        if (errcode < 0)
            return errcode;
    }

    // Wait for data to be available in input ringbuffer
    const std::ptrdiff_t Tin_ringbuf = Ein_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t Tin_read_max = Tin_ringbuf / 4;
    std::ptrdiff_t Tin_read = -1;
    {
        const int errcode =
            Ein_buffer.wait_and_claim_readable([&](const std::ptrdiff_t Tin_available) {
                using std::min;
                Tin_read = round_down(min(Tin_available, Tin_read_max),
                                      cuda_granularity_number_of_timesamples);
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

cudaEvent_t cudaTranspose2048_chime::execute(cudaPipelineState& /*pipestate*/,
                                             const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    void* const Ein_memory = Ein_buffer.get_ndarray().data();
    void* const E_memory = E_buffer.get_ndarray().data();
    void* const scatter_indices_memory = scatter_indices_buffer.get_ndarray().data();
    void* const info_memory = info_buffer.get_ndarray().data();

    if (args::Ein == args::Ein) {
        // Replace "Ein" with "E" etc.
        // Ein_buffer.check_metadata();
        const std::string quantity = "E";
        const std::array<std::string, 4> dimname = {"T", "F", "P", "D"};
        const std::shared_ptr<const chordMetadata> metadata = Ein_buffer.get_metadata();
        if (!(metadata->get_name() == quantity))
            ERROR("buffer name: {:s}, quantity: {:s}, metadata name: {:s}",
                  Ein_buffer.get_buffer_name(), quantity, metadata->get_name());
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
        Ein_buffer.check_metadata();
    }
    E_buffer.set_metadata(Ein_buffer.get_metadata());
    if (args::scatter_indices == args::Ein) {
        // Replace "Ein" with "E" etc.
        // scatter_indices_buffer.check_metadata();
        const std::string quantity = "E";
        const std::array<std::string, 4> dimname = {"T", "F", "P", "D"};
        const std::shared_ptr<const chordMetadata> metadata = Ein_buffer.get_metadata();
        if (!(metadata->get_name() == quantity))
            ERROR("buffer name: {:s}, quantity: {:s}, metadata name: {:s}",
                  Ein_buffer.get_buffer_name(), quantity, metadata->get_name());
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
        scatter_indices_buffer.check_metadata();
    }

    const char* exc_arg = "exception";
    std::int32_t Tin_min_arg;
    std::int32_t Tin_max_arg;
    std::int32_t T_min_arg;
    std::int32_t T_max_arg;
    array_desc Ein_arg(Ein_memory, Ein_length_in_bytes);
    array_desc E_arg(E_memory, E_length_in_bytes);
    array_desc scatter_indices_arg(scatter_indices_memory, scatter_indices_length_in_bytes);
    array_desc info_arg(info_memory, info_length_in_bytes);
    void* args[] = {
        &exc_arg, &Tin_min_arg, &Tin_max_arg,         &T_min_arg, &T_max_arg,
        &Ein_arg, &E_arg,       &scatter_indices_arg, &info_arg,
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

    E_buffer.set_to_poison(0x00);

#ifdef DEBUGGING
    // Initialize host-side buffer arrays
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length_in_bytes, device.getStream(cuda_stream_id)));
#endif

    E_buffer.set_to_poison(0x00);
    info_buffer.set_to_poison(0xff);

    const std::string symname = "Transpose2048_chime_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    const CUresult err =
        cuLaunchKernel(device.runtime_kernels[symname], blocks, 1, 1, threads_x, threads_y, 1,
                       shmem_bytes, device.getStream(cuda_stream_id), args, NULL);

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        ERROR("cuLaunchKernel: Error number: {}: {}", (int)err, errStr);
    }

#ifdef DEBUGGING
    // Copy results back to host memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_info_buffer.data(), info_memory, info_length_in_bytes,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));

    // Check error codes
    const std::int32_t error_code =
        *std::max_element((const std::int32_t*)&*host_info_buffer.begin(),
                          (const std::int32_t*)&*host_info_buffer.end());
    if (error_code != 0)
        ERROR("CUDA kernel Transpose2048_chime returned error code: {}", error_code);

    // TODO: Introduce a new "unbuffered" buffer; do this there
    for (int block = 0; block < info_lengths[info_index_block]; ++block) {
        for (int warp = 0; warp < info_lengths[info_index_warp]; ++warp) {
            for (int thread = 0; thread < info_lengths[info_index_thread]; ++thread) {
                const std::ptrdiff_t i = info_strides[info_index_thread] * thread
                                         + info_strides[info_index_warp] * warp
                                         + info_strides[info_index_block] * block;
                const std::uint32_t val = host_info_buffer.data()[i];
                if (val != 0)
                    ERROR("CUDA kernel Transpose2048_chime returned 'info' value {:d} "
                          "for thread {:d} warp {:d} block {:d} at index {:d} (zero indicates no "
                          "error)",
                          val, thread, warp, block, i);
            }
        }
    }
#endif

    E_buffer.check_for_poison(0x00);

    return record_end_event();
}

void cudaTranspose2048_chime::finalize_frame() {
    // Advance the input ring buffer
    Ein_buffer.finish_read();

    // Advance the output ring buffer
    E_buffer.finish_write();

    cudaCommand::finalize_frame();
}
