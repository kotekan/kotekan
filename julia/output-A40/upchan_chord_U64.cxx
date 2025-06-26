/**
 * @file
 * @brief CUDA Upchannelizer_chord_U64 kernel
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
 * @class cudaUpchannelizer_chord_U64
 * @brief cudaCommand for Upchannelizer_chord_U64
 */
class cudaUpchannelizer_chord_U64 : public cudaCommand {
public:
    cudaUpchannelizer_chord_U64(Config& config, const std::string& unique_name,
                                bufferContainer& host_buffers, cudaDeviceInterface& device,
                                const int instance_num);
    virtual ~cudaUpchannelizer_chord_U64();

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
        CuDeviceArray(void* const ptr, const std::ptrdiff_t bytes) :
            ptr(static_cast<T*>(ptr)), maxsize(bytes), dims{std::int64_t(maxsize / sizeof(T))},
            len(maxsize / sizeof(T)) {}
    };
    using array_desc = CuDeviceArray<std::int32_t, 1>;

    // Kernel design parameters:
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 512;
    static constexpr int cuda_number_of_frequencies = 48;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_taps = 4;
    static constexpr int cuda_max_number_of_timesamples = 32768;
    static constexpr int cuda_granularity_number_of_timesamples = 256;
    static constexpr int cuda_algorithm_overlap = 192;
    static constexpr int cuda_upchannelization_factor = 64;

    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

    // Kernel compile parameters:
    static constexpr int minthreads = 512;
    static constexpr int blocks_per_sm = 2;
    static constexpr int blocks_per_frequency = 8;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 16;
    static constexpr int max_blocks = 384;
    static constexpr int shmem_bytes = 33408;

    // Kernel name:
    static constexpr const char* kernel_symbol =
        "_Z8upchan645Int32S_S_S_S_S_13CuDeviceArrayI9Float16x2Li1ELi1EES0_I6Int4x8Li1ELi1EES4_S0_"
        "IS_Li1ELi1EE";

    // Kernel arguments:
    enum class args { T_min, T_max, Tbar_min, Tbar_max, Fmin, Fmax, G_U64, E, Ebar, info, count };

    // How many frequencies we will process
    const int Fmin, Fmax;

    // T_min: T_min
    static constexpr const char* T_min_quantity = "T_min";
    static constexpr kotekan::DataType T_min_type = kotekan::int32;
    //
    // T_max: T_max
    static constexpr const char* T_max_quantity = "T_max";
    static constexpr kotekan::DataType T_max_type = kotekan::int32;
    //
    // Tbar_min: Tbar_min
    static constexpr const char* Tbar_min_quantity = "Tbar_min";
    static constexpr kotekan::DataType Tbar_min_type = kotekan::int32;
    //
    // Tbar_max: Tbar_max
    static constexpr const char* Tbar_max_quantity = "Tbar_max";
    static constexpr kotekan::DataType Tbar_max_type = kotekan::int32;
    //
    // Fmin: Fmin
    static constexpr const char* Fmin_quantity = "Fmin";
    static constexpr kotekan::DataType Fmin_type = kotekan::int32;
    //
    // Fmax: Fmax
    static constexpr const char* Fmax_quantity = "Fmax";
    static constexpr kotekan::DataType Fmax_type = kotekan::int32;
    //
    // G_U64: upchan_U64_gain_name
    static constexpr const char* G_U64_quantity = "G_U64";
    static constexpr kotekan::DataType G_U64_type = kotekan::float16;
    enum G_U64_indices {
        G_U64_index_Fbar,
        G_U64_rank,
    };
    static constexpr std::array<const char*, G_U64_rank> G_U64_labels = {
        "Fbar",
    };
    static constexpr std::array<std::ptrdiff_t, G_U64_rank> G_U64_lengths = {
        256,
    };
    static constexpr auto G_U64_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= G_U64_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, G_U64_rank + 1> G_U64_strides = {
        G_U64_calc_stride(G_U64_index_Fbar),
        G_U64_calc_stride(G_U64_rank),
    };
    static constexpr std::ptrdiff_t G_U64_length = G_U64_strides[G_U64_rank];
    static constexpr std::ptrdiff_t G_U64_length_in_bytes =
        type_total_bytes(G_U64_type) * G_U64_length;
    static_assert(G_U64_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    //
    // E: voltage_name
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
        512,
        2,
        48,
        32768,
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
    // Ebar: upchan_U64_voltage_name
    static constexpr const char* Ebar_quantity = "Ebar";
    static constexpr kotekan::DataType Ebar_type = kotekan::int4x2chime;
    enum Ebar_indices {
        Ebar_index_D,
        Ebar_index_P,
        Ebar_index_Fbar,
        Ebar_index_Tbar,
        Ebar_rank,
    };
    static constexpr std::array<const char*, Ebar_rank> Ebar_labels = {
        "D",
        "P",
        "Fbar",
        "Tbar",
    };
    static constexpr std::array<std::ptrdiff_t, Ebar_rank> Ebar_lengths = {
        512,
        2,
        256,
        512,
    };
    static constexpr auto Ebar_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= Ebar_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, Ebar_rank + 1> Ebar_strides = {
        Ebar_calc_stride(Ebar_index_D),    Ebar_calc_stride(Ebar_index_P),
        Ebar_calc_stride(Ebar_index_Fbar), Ebar_calc_stride(Ebar_index_Tbar),
        Ebar_calc_stride(Ebar_rank),
    };
    static constexpr std::ptrdiff_t Ebar_length = Ebar_strides[Ebar_rank];
    static constexpr std::ptrdiff_t Ebar_length_in_bytes =
        type_total_bytes(Ebar_type) * Ebar_length;
    static_assert(Ebar_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
        384,
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
    const std::string G_U64_name;
    const std::string E_name;
    const std::string Ebar_name;
    const std::string info_name;

    // Buffers
    NDArrayBuffer<kotekan::GetType_t<G_U64_type>, G_U64_rank> G_U64_buffer;
    NDArrayRingBuffer<kotekan::GetType_t<E_type>, E_rank> E_buffer;
    NDArrayRingBuffer<kotekan::GetType_t<Ebar_type>, Ebar_rank> Ebar_buffer;
    NDArrayBuffer<kotekan::GetType_t<info_type>, info_rank> info_buffer;
    std::vector<kotekan::GetType_t<info_type>> host_info_buffer;

    // To avoid trailing comma below
    int dummy;
};

REGISTER_CUDA_COMMAND(cudaUpchannelizer_chord_U64);

cudaUpchannelizer_chord_U64::cudaUpchannelizer_chord_U64(Config& config,
                                                         const std::string& unique_name,
                                                         bufferContainer& host_buffers,
                                                         cudaDeviceInterface& device,
                                                         const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "Upchannelizer_chord_U64", "Upchannelizer_chord_U64.ptx"),
    Fmin(config.get<int>(unique_name, "Fmin")), Fmax(config.get<int>(unique_name, "Fmax")),

    G_U64_name(config.get<std::string>(unique_name, "upchan_U64_gain_name")),
    E_name(config.get<std::string>(unique_name, "voltage_name")),
    Ebar_name(config.get<std::string>(unique_name, "upchan_U64_voltage_name")),
    info_name(unique_name + "/gpu_mem_info"),

    G_U64_buffer(G_U64_name, G_U64_quantity, reverse(G_U64_lengths), reverse(G_U64_labels), *this,
                 buffer_type_t::do_once),
    E_buffer(E_name, E_quantity, reverse(E_lengths), reverse(E_labels), *this),
    Ebar_buffer(Ebar_name, Ebar_quantity, reverse(Ebar_lengths), reverse(Ebar_labels), *this),
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

    G_U64_buffer.register_consumer();
    E_buffer.register_consumer();
    Ebar_buffer.register_producer();
    register_gpu_buffer_user(
        {.name = info_name, .is_array = true, .does_read = true, .does_write = true});

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage needs to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "Upchannelizer_chord_U64_");
    }
}

cudaUpchannelizer_chord_U64::~cudaUpchannelizer_chord_U64() {}

std::int64_t
cudaUpchannelizer_chord_U64::num_consumed_elements(std::int64_t num_available_elements) const {
    if (num_processed_elements(num_available_elements) < cuda_algorithm_overlap)
        return 0;
    return num_processed_elements(num_available_elements) - cuda_algorithm_overlap;
}
std::int64_t
cudaUpchannelizer_chord_U64::num_produced_elements(std::int64_t num_available_elements) const {
    assert(num_consumed_elements(num_available_elements) % cuda_upchannelization_factor == 0);
    return num_consumed_elements(num_available_elements) / cuda_upchannelization_factor;
}

std::int64_t
cudaUpchannelizer_chord_U64::num_processed_elements(std::int64_t num_available_elements) const {
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cudaUpchannelizer_chord_U64::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    const std::ptrdiff_t T_ringbuf = E_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t T_read_max = T_ringbuf / 4;
    std::ptrdiff_t T_read = -1;
    {
        const int errcode = E_buffer.wait_and_claim_readable([&](const std::ptrdiff_t T_available) {
            using std::min;
            T_read = min(T_available, T_read_max);
            return read_descriptor_t{.claimed = num_consumed_elements(T_read),
                                     .read = num_processed_elements(T_read)};
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

cudaEvent_t cudaUpchannelizer_chord_U64::execute(cudaPipelineState& /*pipestate*/,
                                                 const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    void* const G_U64_memory = G_U64_buffer.get_ndarray().data();
    void* const E_memory = E_buffer.get_ndarray().data();
    void* const Ebar_memory = Ebar_buffer.get_ndarray().data();
    void* const info_memory = info_buffer.get_ndarray().data();

    G_U64_buffer.check_metadata();
    E_buffer.check_metadata();
    Ebar_buffer.set_metadata(E_buffer.get_metadata());

    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    const char* exc_arg = "exception";
    std::int32_t T_min_arg;
    std::int32_t T_max_arg;
    std::int32_t Tbar_min_arg;
    std::int32_t Tbar_max_arg;
    std::int32_t Fmin_arg;
    std::int32_t Fmax_arg;
    array_desc G_U64_arg(G_U64_memory, G_U64_length_in_bytes);
    array_desc E_arg(E_memory, E_length_in_bytes);
    array_desc Ebar_arg(Ebar_memory, Ebar_length_in_bytes);
    array_desc info_arg(info_memory, info_length_in_bytes);
    void* args[] = {
        &exc_arg,  &T_min_arg, &T_max_arg, &Tbar_min_arg, &Tbar_max_arg, &Fmin_arg,
        &Fmax_arg, &G_U64_arg, &E_arg,     &Ebar_arg,     &info_arg,
    };

    // Set E_memory to beginning of input ring buffer
    E_arg = array_desc(E_memory, E_length_in_bytes);

    // Set Ebar_memory to beginning of output ring buffer
    Ebar_arg = array_desc(Ebar_memory, Ebar_length_in_bytes);

    // Ringbuffer size
    const std::ptrdiff_t T_ringbuf = E_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t Tbar_ringbuf = Ebar_buffer.get_ndarray().extent(0);

    const std::ptrdiff_t T_min = E_buffer.get_begin_read_valid();
    const std::ptrdiff_t T_max = E_buffer.get_end_read_valid();
    const std::ptrdiff_t Tbar_min = Ebar_buffer.get_begin_write_valid();
    const std::ptrdiff_t Tbar_max = Ebar_buffer.get_end_write_valid();

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

    Ebar_buffer.set_to_poison(0x00);
    info_buffer.set_to_poison(0xff);

#ifdef DEBUGGING
    // Initialize host-side buffer arrays
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length_in_bytes, device.getStream(cuda_stream_id)));
#endif

    const std::string symname = "Upchannelizer_chord_U64_" + std::string(kernel_symbol);
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
    const std::uint32_t error_code = *std::max_element(
        (const std::uint32_t*)&host_info_buffer[0],
        (const std::uint32_t*)&host_info_buffer[blocks * info_lengths[info_index_warp]
                                                * info_lengths[info_index_thread]]);
    if (error_code != 0)
        ERROR("CUDA kernel Upchannelizer_chord_U64 returned error code: {}", error_code);

    if (error_code != 0) {
        // TODO: Introduce a new "unbuffered" buffer; do this there
        // Our `info` buffer is too large (`blocks` vs. `max_blocks`)
        for (int block = 0; block < blocks; ++block) {
            for (int warp = 0; warp < info_lengths[info_index_warp]; ++warp) {
                for (int thread = 0; thread < info_lengths[info_index_thread]; ++thread) {
                    const std::ptrdiff_t i = info_strides[info_index_thread] * thread
                                             + info_strides[info_index_warp] * warp
                                             + info_strides[info_index_block] * block;
                    const std::uint32_t val = host_info_buffer.data()[i];
                    if (val != 0)
                        ERROR("CUDA kernel Upchannelizer_chord_U64 returned 'info' value {:d} "
                              "for thread {:d} warp {:d} block {:d} at index {:d} (zero indicates "
                              "no error)",
                              val, thread, warp, block, i);
                }
            }
        }
    }
#endif

    E_buffer.check_for_poison(0x00);

    return record_end_event();
}

void cudaUpchannelizer_chord_U64::finalize_frame() {
    // Advance the input ring buffer
    E_buffer.finish_read();

    // Advance the output ring buffer
    Ebar_buffer.finish_write();

    cudaCommand::finalize_frame();
}
