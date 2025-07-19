/**
 * @file
 * @brief CUDA BasebandBeamformer_chord kernel
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
 * @class cudaBasebandBeamformer_chord
 * @brief cudaCommand for BasebandBeamformer_chord
 */
class cudaBasebandBeamformer_chord : public cudaCommand {
public:
    cudaBasebandBeamformer_chord(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, cudaDeviceInterface& device,
                                 const int instance_num);
    virtual ~cudaBasebandBeamformer_chord();

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
    static constexpr int cuda_number_of_beams = 96;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 512;
    static constexpr int cuda_number_of_frequencies = 48;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 32768;
    static constexpr int cuda_granularity_number_of_timesamples = 8192;
    static constexpr int cuda_shift_parameter_sigma = 3;

    // Kernel compile parameters:
    static constexpr int minthreads = 768;
    static constexpr int blocks_per_sm = 1;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 24;
    static constexpr int blocks = 96;
    static constexpr int shmem_bytes = 67712;

    // Kernel name:
    static constexpr const char* kernel_symbol =
        "_Z2bb5Int32S_13CuDeviceArrayI6Int8x4Li1ELi1EES0_I6Int4x8Li1ELi1EES0_IS_Li1ELi1EES4_S5_S5_";

    // Kernel arguments:
    enum class args { T_min, T_max, A, E, s, J, info, log, count };

    // T_min: T_min
    static constexpr const char* T_min_quantity = "T_min";
    static constexpr kotekan::DataType T_min_type = kotekan::int32;
    //
    // T_max: T_max
    static constexpr const char* T_max_quantity = "T_max";
    static constexpr kotekan::DataType T_max_type = kotekan::int32;
    //
    // A: bb_phase_name
    static constexpr const char* A_quantity = "A";
    static constexpr kotekan::DataType A_type = kotekan::int8;
    enum A_indices {
        A_index_C,
        A_index_D,
        A_index_B,
        A_index_P,
        A_index_F,
        A_rank,
    };
    static constexpr std::array<const char*, A_rank> A_labels = {
        "C", "D", "B", "P", "F",
    };
    static constexpr std::array<std::ptrdiff_t, A_rank> A_lengths = {
        2, 512, 96, 2, 48,
    };
    static constexpr auto A_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= A_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, A_rank + 1> A_strides = {
        A_calc_stride(A_index_C), A_calc_stride(A_index_D), A_calc_stride(A_index_B),
        A_calc_stride(A_index_P), A_calc_stride(A_index_F), A_calc_stride(A_rank),
    };
    static constexpr std::ptrdiff_t A_length = A_strides[A_rank];
    static constexpr std::ptrdiff_t A_length_in_bytes = type_total_bytes(A_type) * A_length;
    static_assert(A_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    // s: bb_shift_name
    static constexpr const char* s_quantity = "s";
    static constexpr kotekan::DataType s_type = kotekan::int32;
    enum s_indices {
        s_index_B,
        s_index_P,
        s_index_F,
        s_rank,
    };
    static constexpr std::array<const char*, s_rank> s_labels = {
        "B",
        "P",
        "F",
    };
    static constexpr std::array<std::ptrdiff_t, s_rank> s_lengths = {
        96,
        2,
        48,
    };
    static constexpr auto s_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= s_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, s_rank + 1> s_strides = {
        s_calc_stride(s_index_B),
        s_calc_stride(s_index_P),
        s_calc_stride(s_index_F),
        s_calc_stride(s_rank),
    };
    static constexpr std::ptrdiff_t s_length = s_strides[s_rank];
    static constexpr std::ptrdiff_t s_length_in_bytes = type_total_bytes(s_type) * s_length;
    static_assert(s_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    //
    // J: bb_beams_name
    static constexpr const char* J_quantity = "J";
    static constexpr kotekan::DataType J_type = kotekan::int4x2chime;
    enum J_indices {
        J_index_T,
        J_index_P,
        J_index_F,
        J_index_B,
        J_rank,
    };
    static constexpr std::array<const char*, J_rank> J_labels = {
        "T",
        "P",
        "F",
        "B",
    };
    static constexpr std::array<std::ptrdiff_t, J_rank> J_lengths = {
        8192,
        2,
        48,
        96,
    };
    static constexpr auto J_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= J_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, J_rank + 1> J_strides = {
        J_calc_stride(J_index_T), J_calc_stride(J_index_P), J_calc_stride(J_index_F),
        J_calc_stride(J_index_B), J_calc_stride(J_rank),
    };
    static constexpr std::ptrdiff_t J_length = J_strides[J_rank];
    static constexpr std::ptrdiff_t J_length_in_bytes = type_total_bytes(J_type) * J_length;
    static_assert(J_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
        24,
        96,
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
    // log: gpu_mem_log
    static constexpr const char* log_quantity = "log";
    static constexpr kotekan::DataType log_type = kotekan::int32;
    enum log_indices {
        log_index_block,
        log_rank,
    };
    static constexpr std::array<const char*, log_rank> log_labels = {
        "block",
    };
    static constexpr std::array<std::ptrdiff_t, log_rank> log_lengths = {
        96,
    };
    static constexpr auto log_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= log_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, log_rank + 1> log_strides = {
        log_calc_stride(log_index_block),
        log_calc_stride(log_rank),
    };
    static constexpr std::ptrdiff_t log_length = log_strides[log_rank];
    static constexpr std::ptrdiff_t log_length_in_bytes = type_total_bytes(log_type) * log_length;
    static_assert(log_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    //

    // Kotekan buffer names
    const std::string A_name;
    const std::string E_name;
    const std::string s_name;
    const std::string J_name;
    const std::string info_name;
    const std::string log_name;

    // Buffers
    NDArrayBuffer<kotekan::GetType_t<A_type>, A_rank> A_buffer;
    NDArrayRingBuffer<kotekan::GetType_t<E_type>, E_rank> E_buffer;
    NDArrayBuffer<kotekan::GetType_t<s_type>, s_rank> s_buffer;
    NDArrayBuffer<kotekan::GetType_t<J_type>, J_rank> J_buffer;
    NDArrayBuffer<kotekan::GetType_t<info_type>, info_rank> info_buffer;
    std::vector<kotekan::GetType_t<info_type>> host_info_buffer;
    NDArrayBuffer<kotekan::GetType_t<log_type>, log_rank> log_buffer;
    std::vector<kotekan::GetType_t<log_type>> host_log_buffer;

    // To avoid trailing comma below
    int dummy;
};

REGISTER_CUDA_COMMAND(cudaBasebandBeamformer_chord);

cudaBasebandBeamformer_chord::cudaBasebandBeamformer_chord(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& host_buffers,
                                                           cudaDeviceInterface& device,
                                                           const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "BasebandBeamformer_chord", "BasebandBeamformer_chord.ptx"),

    A_name(config.get<std::string>(unique_name, "bb_phase_name")),
    E_name(config.get<std::string>(unique_name, "voltage_name")),
    s_name(config.get<std::string>(unique_name, "bb_shift_name")),
    J_name(config.get<std::string>(unique_name, "bb_beams_name")),
    info_name(unique_name + "/gpu_mem_info"), log_name(unique_name + "/gpu_mem_log"),

    A_buffer(A_name, A_quantity, reverse(A_lengths), reverse(A_labels), *this,
             buffer_type_t::do_once),
    E_buffer(E_name, E_quantity, reverse(E_lengths), reverse(E_labels), *this),
    s_buffer(s_name, s_quantity, reverse(s_lengths), reverse(s_labels), *this,
             buffer_type_t::do_once),
    J_buffer(J_name, J_quantity, reverse(J_lengths), reverse(J_labels), *this),
    info_buffer(info_name, info_quantity, reverse(info_lengths), reverse(info_labels), *this),
    host_info_buffer(info_length),
    log_buffer(log_name, log_quantity, reverse(log_lengths), reverse(log_labels), *this),
    host_log_buffer(log_length),

    dummy() // avoid trailing comma
{
    // Register host memory
    {
        const cudaError_t ierr = cudaHostRegister(
            host_info_buffer.data(), host_info_buffer.size() * sizeof *host_info_buffer.data(), 0);
        assert(ierr == cudaSuccess);
    }
    {
        const cudaError_t ierr = cudaHostRegister(
            host_log_buffer.data(), host_log_buffer.size() * sizeof *host_log_buffer.data(), 0);
        assert(ierr == cudaSuccess);
    }

    A_buffer.register_consumer();
    E_buffer.register_consumer();
    s_buffer.register_consumer();
    J_buffer.register_producer();
    register_gpu_buffer_user(
        {.name = info_name, .is_array = true, .does_read = true, .does_write = true});
    register_gpu_buffer_user(
        {.name = log_name, .is_array = true, .does_read = true, .does_write = true});

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage needs to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "BasebandBeamformer_chord_");
    }
}

cudaBasebandBeamformer_chord::~cudaBasebandBeamformer_chord() {}

int cudaBasebandBeamformer_chord::wait_on_precondition() {
    {
        const int errcode = cudaCommand::wait_on_precondition();
        if (errcode < 0)
            return errcode;
    }

    // Wait for data to be available in input ringbuffer
    const std::ptrdiff_t T_ringbuf = E_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t T_read_max = T_ringbuf / 4;
    {
        const int errcode = E_buffer.wait_and_claim_readable([&](const std::ptrdiff_t T_available) {
            using std::min;
            const std::ptrdiff_t T_read =
                round_down(min(T_available, T_read_max), cuda_granularity_number_of_timesamples);
            return read_descriptor_t{.claimed = T_read, .read = T_read};
        });
        if (errcode < 0)
            return errcode;
    }

    return 0;
}

cudaEvent_t cudaBasebandBeamformer_chord::execute(cudaPipelineState& /*pipestate*/,
                                                  const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    void* const A_memory = A_buffer.get_ndarray().data();
    void* const E_memory = E_buffer.get_ndarray().data();
    void* const s_memory = s_buffer.get_ndarray().data();
    void* const J_memory = J_buffer.get_ndarray().data();
    void* const info_memory = info_buffer.get_ndarray().data();
    void* const log_memory = log_buffer.get_ndarray().data();

    A_buffer.check_metadata();
    E_buffer.check_metadata();
    s_buffer.check_metadata();
    J_buffer.set_metadata(E_buffer.get_metadata());

    const char* exc_arg = "exception";
    std::int32_t T_min_arg;
    std::int32_t T_max_arg;
    array_desc A_arg(A_memory, A_length_in_bytes);
    array_desc E_arg(E_memory, E_length_in_bytes);
    array_desc s_arg(s_memory, s_length_in_bytes);
    array_desc J_arg(J_memory, J_length_in_bytes);
    array_desc info_arg(info_memory, info_length_in_bytes);
    array_desc log_arg(log_memory, log_length_in_bytes);
    void* args[] = {
        &exc_arg, &T_min_arg, &T_max_arg, &A_arg, &E_arg, &s_arg, &J_arg, &info_arg, &log_arg,
    };

    // Set E_memory to beginning of input ring buffer
    E_arg = array_desc(E_memory, E_length_in_bytes);

    // Ringbuffer size
    const std::ptrdiff_t T_ringbuf = E_buffer.get_ndarray().extent(0);

    const std::ptrdiff_t T_min = E_buffer.get_read_valid().begin();
    const std::ptrdiff_t T_max = E_buffer.get_read_valid().end();

    const std::ptrdiff_t T_length = T_max - T_min;

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    T_min_arg = mod(T_min, T_ringbuf);
    T_max_arg = mod(T_min, T_ringbuf) + T_length;

    // Update metadata
    {
        const std::shared_ptr<chordMetadata> J_meta = J_buffer.get_metadata();

        // Since we do not use a ring buffer we need to set `meta->sample0_offset`
        J_meta->sample0_offset = T_min;
        assert(J_meta->offset_downsampling == 1);
    }

    // Copy inputs to device memory

    J_buffer.set_to_poison(0x00);
    info_buffer.set_to_poison(0xff);
    log_buffer.set_to_poison(0xff);

#ifdef DEBUGGING
    // Initialize host-side buffer arrays
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length_in_bytes, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(log_memory, 0xff, log_length_in_bytes, device.getStream(cuda_stream_id)));
#endif

    const std::string symname = "BasebandBeamformer_chord_" + std::string(kernel_symbol);
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
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_log_buffer.data(), log_memory, log_length_in_bytes,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));

    // Check error codes
    // TODO: Introduce a new "unbuffered" buffer; do this there
    const std::int32_t error_code =
        *std::max_element((const std::int32_t*)&*host_info_buffer.begin(),
                          (const std::int32_t*)&*host_info_buffer.end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code: {}", error_code);

    // TODO: Introduce a new "unbuffered" buffer; do this there
    for (int block = 0; block < info_lengths[info_index_block]; ++block) {
        for (int warp = 0; warp < info_lengths[info_index_warp]; ++warp) {
            for (int thread = 0; thread < info_lengths[info_index_thread]; ++thread) {
                const std::ptrdiff_t i = info_strides[info_index_thread] * thread
                                         + info_strides[info_index_warp] * warp
                                         + info_strides[info_index_block] * block;
                const std::uint32_t val = host_info_buffer.data()[i];
                if (val != 0)
                    ERROR("CUDA kernel BasebandBeamformer_chord returned 'info' value {:d} "
                          "for thread {:d} warp {:d} block {:d} at index {:d} (zero indicates no "
                          "error)",
                          val, thread, warp, block, i);
            }
        }
    }

    // Check log codes
    const std::uint32_t log_code =
        *std::max_element((const std::uint32_t*)&*host_log_buffer.begin(),
                          (const std::uint32_t*)&*host_log_buffer.end());
    if (log_code != 0)
        WARN("CUDA kernel BasebandBeamformer_chord returned log code cuLaunchKernel: {}", log_code);

    // TODO: Introduce a new "unbuffered" buffer; do this there
    for (std::size_t i = 0; i < host_log_buffer.size(); ++i) {
        const std::uint32_t val = host_log_buffer.data()[i];
        if (val != 0)
            WARN("CUDA kernel BasebandBeamformer_chord returned 'log' value {:d} at index {:d} "
                 "(zero "
                 "indicates success)",
                 val, i);
    }
#endif

    J_buffer.check_for_poison(0x00);

    return record_end_event();
}

void cudaBasebandBeamformer_chord::finalize_frame() {
    // Advance the input ring buffer
    E_buffer.finish_read();

    cudaCommand::finalize_frame();
}
