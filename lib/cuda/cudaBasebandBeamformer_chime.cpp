/**
 * @file
 * @brief CUDA BasebandBeamformer_chime kernel
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
 * @class cudaBasebandBeamformer_chime
 * @brief cudaCommand for BasebandBeamformer_chime
 */
class cudaBasebandBeamformer_chime : public cudaCommand {
public:
    cudaBasebandBeamformer_chime(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, cudaDeviceInterface& device,
                                 const int instance_num);
    virtual ~cudaBasebandBeamformer_chime();

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
    static constexpr int cuda_number_of_beams = 16;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 1024;
    static constexpr int cuda_number_of_frequencies = 16;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 65536;
    static constexpr int cuda_granularity_number_of_timesamples = 16384;
    static constexpr int cuda_shift_parameter_sigma = 4;

    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

    // Kernel compile parameters:
    static constexpr int minthreads = 128;
    static constexpr int blocks_per_sm = 8;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 4;
    static constexpr int blocks = 32;
    static constexpr int shmem_bytes = 43136;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z2bb5Int32S_13CuDeviceArrayI6Int8x4Li1ELi1EES0_I6Int4x8Li1ELi1EES0_IS_Li1ELi1EES4_S5_S5_";

    // Kernel arguments:
    enum class args { Tmin, Tmax, A, E, s, J, info, log, count };

    // Tmin: Tmin
    static constexpr const char* Tmin_quantity = "Tmin";
    static constexpr kotekan::DataType Tmin_type = kotekan::int32;
    //
    // Tmax: Tmax
    static constexpr const char* Tmax_quantity = "Tmax";
    static constexpr kotekan::DataType Tmax_type = kotekan::int32;
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
        2, 1024, 16, 2, 16,
    };
    static constexpr std::ptrdiff_t A_length = type_total_bytes(A_type) * 2 * 1024 * 16 * 2 * 16;
    static_assert(A_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    static_assert(A_length == type_total_bytes(A_type) * A_strides[A_rank]);
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
        1024,
        2,
        16,
        65536,
    };
    static constexpr std::ptrdiff_t E_length = type_total_bytes(E_type) * 1024 * 2 * 16 * 65536;
    static_assert(E_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    static_assert(E_length == type_total_bytes(E_type) * E_strides[E_rank]);
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
        16,
        2,
        16,
    };
    static constexpr std::ptrdiff_t s_length = type_total_bytes(s_type) * 16 * 2 * 16;
    static_assert(s_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    static_assert(s_length == type_total_bytes(s_type) * s_strides[s_rank]);
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
        16384,
        2,
        16,
        16,
    };
    static constexpr std::ptrdiff_t J_length = type_total_bytes(J_type) * 16384 * 2 * 16 * 16;
    static_assert(J_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    static_assert(J_length == type_total_bytes(J_type) * J_strides[J_rank]);
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
        4,
        32,
    };
    static constexpr std::ptrdiff_t info_length = type_total_bytes(info_type) * 32 * 4 * 32;
    static_assert(info_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    static_assert(info_length == type_total_bytes(info_type) * info_strides[info_rank]);
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
        32,
    };
    static constexpr std::ptrdiff_t log_length = type_total_bytes(log_type) * 32;
    static_assert(log_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    static_assert(log_length == type_total_bytes(log_type) * log_strides[log_rank]);
    //

    // Kotekan buffer names
    const std::string A_name;
    const std::string E_name;
    const std::string s_name;
    const std::string J_name;
    const std::string info_name;
    const std::string log_name;

    // Host-side buffer arrays
    std::vector<std::uint8_t> info_host;
    std::vector<std::uint8_t> log_host;

    static constexpr std::ptrdiff_t E_T_sample_bytes = type_total_bytes(E_type)
                                                       * E_lengths[E_index_D] * E_lengths[E_index_P]
                                                       * E_lengths[E_index_F];

    NDArrayBuffer<kotekan::GetType_t<A_type>, A_rank> A_buffer;
    NDArrayRingBuffer<kotekan::GetType_t<E_type>, E_rank> E_buffer;
    NDArrayBuffer<kotekan::GetType_t<s_type>, s_rank> s_buffer;
    NDArrayBuffer<kotekan::GetType_t<J_type>, J_rank> J_buffer;

    // To avoid trailing comma below
    int dummy;
};

REGISTER_CUDA_COMMAND(cudaBasebandBeamformer_chime);

cudaBasebandBeamformer_chime::cudaBasebandBeamformer_chime(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& host_buffers,
                                                           cudaDeviceInterface& device,
                                                           const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "BasebandBeamformer_chime", "BasebandBeamformer_chime.ptx"),
    A_name(config.get<std::string>(unique_name, "bb_phase_name")),
    E_name(config.get<std::string>(unique_name, "voltage_name")),
    s_name(config.get<std::string>(unique_name, "bb_shift_name")),
    J_name(config.get<std::string>(unique_name, "bb_beams_name")),
    info_name(unique_name + "/gpu_mem_info"), log_name(unique_name + "/gpu_mem_log"),

    info_host(info_length), log_host(log_length),

    A_buffer(A_name, A_quantity, reverse(A_lengths), reverse(A_labels), *this,
             buffer_type_t::do_once),
    E_buffer(E_name, E_quantity, reverse(E_lengths), reverse(E_labels), *this),
    s_buffer(s_name, s_quantity, reverse(s_lengths), reverse(s_labels), *this,
             buffer_type_t::do_once),
    J_buffer(J_name, J_quantity, reverse(J_lengths), reverse(J_labels), *this),

    dummy() // avoid trailing comma
{
    // Register host memory
    {
        const cudaError_t ierr = cudaHostRegister(info_host.data(), info_host.size(), 0);
        assert(ierr == cudaSuccess);
    }
    {
        const cudaError_t ierr = cudaHostRegister(log_host.data(), log_host.size(), 0);
        assert(ierr == cudaSuccess);
    }

    A_buffer.register_consumer();
    E_buffer.register_consumer();
    s_buffer.register_consumer();
    J_buffer.register_producer();
    gpu_buffers_used.push_back(std::make_tuple(info_name, false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(log_name, false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage needs to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "BasebandBeamformer_chime_");
    }
}

cudaBasebandBeamformer_chime::~cudaBasebandBeamformer_chime() {}

int cudaBasebandBeamformer_chime::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    {
        const int errcode = E_buffer.wait_and_claim_readable([&](const std::ptrdiff_t T_available) {
            const std::ptrdiff_t T_read =
                round_down(T_available, cuda_granularity_number_of_timesamples);
            return read_descriptor_t{.claimed = T_read, .read = T_read};
        });
        if (errcode < 0)
            return errcode;
    }

    return 0;
}

cudaEvent_t cudaBasebandBeamformer_chime::execute(cudaPipelineState& /*pipestate*/,
                                                  const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    void* const A_memory = A_buffer.get_ndarray().data();
    void* const E_memory = E_buffer.get_ndarray().data();
    void* const s_memory = s_buffer.get_ndarray().data();
    void* const J_memory = J_buffer.get_ndarray().data();
    const std::string info_memname = info_name + "_buffer";
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);
    const std::string log_memname = log_name + "_buffer";
    void* const log_memory = device.get_gpu_memory(log_memname, log_length);

    A_buffer.check_metadata();
    E_buffer.check_metadata();
    s_buffer.check_metadata();
    J_buffer.set_metadata(E_buffer.get_metadata());

    record_start_event();

    const char* exc_arg = "exception";
    std::int32_t Tmin_arg;
    std::int32_t Tmax_arg;
    array_desc A_arg(A_memory, A_length);
    array_desc E_arg(E_memory, E_length);
    array_desc s_arg(s_memory, s_length);
    array_desc J_arg(J_memory, J_length);
    array_desc info_arg(info_memory, info_length);
    array_desc log_arg(log_memory, log_length);
    void* args[] = {
        &exc_arg, &Tmin_arg, &Tmax_arg, &A_arg, &E_arg, &s_arg, &J_arg, &info_arg, &log_arg,
    };

    // Set E_memory to beginning of input ring buffer
    E_arg = array_desc(E_memory, E_length);

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
        std::shared_ptr<chordMetadata> const J_meta = J_buffer.get_metadata();

        // Since we do not use a ring buffer we need to set `meta->sample0_offset`
        J_meta->sample0_offset = Tmin;
        assert(J_meta->offset_downsampling == 1);
    }

    // Copy inputs to device memory

    J_buffer.set_to_poison(0x00);

#ifdef DEBUGGING
    // Initialize host-side buffer arrays
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(log_memory, 0xff, log_length, device.getStream(cuda_stream_id)));
#endif

    const std::string symname = "BasebandBeamformer_chime_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA BasebandBeamformer_chime on GPU frame {:d}", gpu_frame_id);
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
    CHECK_CUDA_ERROR(cudaMemcpyAsync(info_host.data(), info_memory, info_length,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(log_host.data(), log_memory, log_length,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
    DEBUG("Finished CUDA BasebandBeamformer_chime on GPU frame {:d}", gpu_frame_id);

    // Check error codes
    // TODO: Introduce a new "unbuffered" buffer; do this there
    std::uint32_t error_code = 0;
    for (int block = 0; block < info_lengths[info_index_block]; ++block) {
        for (int warp = 0; warp < info_lengths[info_index_warp]; ++warp) {
            for (int thread = 0; thread < info_lengths[info_index_thread]; ++thread) {
                const std::ptrdiff_t i = info_strides[info_index_thread] * thread
                                         + info_strides[info_index_warp] * warp
                                         + info_strides[info_index_block] * block;
                const std::uint32_t val = *(const std::uint32_t*)&info_host[i];
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
                const std::ptrdiff_t i = info_strides[info_index_thread] * thread
                                         + info_strides[info_index_warp] * warp
                                         + info_strides[info_index_block] * block;
                const std::uint32_t val = ((const std::uint32_t*)info_host.data())[i];
                if (val != 0)
                    ERROR("CUDA kernel BasebandBeamformer_chime returned 'info' value {:d} "
                          "for thread {:d} warp {:d} block {:d} at index {:d} (zero indicates no "
                          "error)",
                          val, thread, warp, block, i);
            }
        }
    }

    // Check log codes
    const std::uint32_t log_code = *std::max_element((const std::uint32_t*)&*log_host.begin(),
                                                     (const std::uint32_t*)&*log_host.end());
    if (log_code != 0)
        WARN("CUDA kernel BasebandBeamformer_chime returned log code cuLaunchKernel: {}", log_code);

    // TODO: Introduce a new "unbuffered" buffer; do this there
    for (std::size_t i = 0; i < log_host.size() / type_total_bytes(log_type); ++i) {
        const std::uint32_t val = ((const std::uint32_t*)log_host.data())[i];
        if (val != 0)
            WARN("CUDA kernel BasebandBeamformer_chime returned 'log' value {:d} at index {:d} "
                 "(zero "
                 "indicates success)",
                 val, i);
    }
#endif

    J_buffer.check_for_poison(0x00);

    return record_end_event();
}

void cudaBasebandBeamformer_chime::finalize_frame() {
    // Advance the input ringbuffer
    E_buffer.finish_read();

    cudaCommand::finalize_frame();
}
