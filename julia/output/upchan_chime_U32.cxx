/**
 * @file
 * @brief CUDA Upchannelizer_chime_U32 kernel
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
    for (std::size_t d = 0; d < D; ++d)
        result[d] = values[D - 1 - d];
    return result;
}
}

/**
 * @class cudaUpchannelizer_chime_U32
 * @brief cudaCommand for Upchannelizer_chime_U32
 */
class cudaUpchannelizer_chime_U32 : public cudaCommand {
public:
    cudaUpchannelizer_chime_U32(Config& config, const std::string& unique_name,
                                bufferContainer& host_buffers, cudaDeviceInterface& device,
                                const int instance_num);
    virtual ~cudaUpchannelizer_chime_U32();

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
    static constexpr int cuda_number_of_dishes = 1024;
    static constexpr int cuda_number_of_frequencies = 16;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_taps = 4;
    static constexpr int cuda_max_number_of_timesamples = 65536;
    static constexpr int cuda_granularity_number_of_timesamples = 256;
    static constexpr int cuda_algorithm_overlap = 96;
    static constexpr int cuda_upchannelization_factor = 32;

    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

    // Kernel compile parameters:
    static constexpr int minthreads = 512;
    static constexpr int blocks_per_sm = 2;
    static constexpr int blocks_per_frequency = 16;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 16;
    static constexpr int max_blocks = 256;
    static constexpr int shmem_bytes = 33920;

    // Kernel name:
    static constexpr const char* kernel_symbol =
        "_Z8upchan325Int32S_S_S_S_S_13CuDeviceArrayI9Float16x2Li1ELi1EES0_I6Int4x8Li1ELi1EES4_S0_"
        "IS_Li1ELi1EE";

    // Kernel arguments:
    enum class args { T_min, T_max, Tbar_min, Tbar_max, Fmin, Fmax, G, E, Ebar, info, count };

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
    // G: upchan_U32_gain_name
    static constexpr const char* G_quantity = "G";
    static constexpr kotekan::DataType G_type = kotekan::float16;
    enum G_indices {
        G_index_Fbar,
        G_rank,
    };
    static constexpr std::array<const char*, G_rank> G_labels = {
        "Fbar",
    };
    static constexpr std::array<std::ptrdiff_t, G_rank> G_lengths = {
        32,
    };
    static constexpr auto G_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= G_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, G_rank + 1> G_strides = {
        G_calc_stride(G_index_Fbar),
        G_calc_stride(G_rank),
    };
    static constexpr std::ptrdiff_t G_length = G_strides[G_rank];
    static constexpr std::ptrdiff_t G_length_in_bytes = type_total_bytes(G_type) * G_length;
    //
    // E: voltage_name
    static constexpr const char* E_quantity = "E";
    static constexpr kotekan::DataType E_type = kotekan::int4x2_swapped_withoffset;
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
    //
    // Ebar: upchan_U32_voltage_name
    static constexpr const char* Ebar_quantity = "Ebar";
    static constexpr kotekan::DataType Ebar_type = kotekan::int4x2_swapped_withoffset;
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
        1024,
        2,
        32,
        2048,
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
        256,
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
    //

    const bool poison_buffers;

    // Kotekan buffer names
    const std::string G_name;
    const std::string E_name;
    const std::string Ebar_name;
    const std::string info_name;

    // Buffers
    NDArrayBuffer<kotekan::GetType_t<G_type>, G_rank> G_buffer;
    NDArrayRingBuffer<kotekan::GetType_t<E_type>, E_rank> E_buffer;
    NDArrayRingBuffer<kotekan::GetType_t<Ebar_type>, Ebar_rank> Ebar_buffer;
    NDArrayBuffer<kotekan::GetType_t<info_type>, info_rank> info_buffer;
    std::vector<kotekan::GetType_t<info_type>> host_info_buffer;

    // To avoid trailing comma below
    int dummy;
};

REGISTER_CUDA_COMMAND(cudaUpchannelizer_chime_U32);

cudaUpchannelizer_chime_U32::cudaUpchannelizer_chime_U32(Config& config,
                                                         const std::string& unique_name,
                                                         bufferContainer& host_buffers,
                                                         cudaDeviceInterface& device,
                                                         const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "Upchannelizer_chime_U32", "Upchannelizer_chime_U32.ptx"),
    Fmin(config.get<int>(unique_name, "Fmin")), Fmax(config.get<int>(unique_name, "Fmax")),

    poison_buffers(config.get_default<bool>(unique_name, "poison_buffers", false)),

    G_name(config.get<std::string>(unique_name, "upchan_U32_gain_name")),
    E_name(config.get<std::string>(unique_name, "voltage_name")),
    Ebar_name(config.get<std::string>(unique_name, "upchan_U32_voltage_name")),
    info_name(unique_name + "/gpu_mem_info"),

    G_buffer(G_name, G_quantity, reverse(G_lengths), reverse(G_labels), *this,
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

    G_buffer.register_consumer();
    E_buffer.register_consumer();
    Ebar_buffer.register_producer();
    register_gpu_buffer_user(
        {.name = info_name, .is_array = true, .does_read = true, .does_write = true});

    set_command_type(gpuCommandType::KERNEL);

    // Build the PTX only once
    static std::once_flag build_ptx_flag;
    std::call_once(build_ptx_flag, [&]() {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx("lib/cuda/generated/Upchannelizer_chime_U32.ptx", {kernel_symbol}, opts,
                         "Upchannelizer_chime_U32_");
    });
}

cudaUpchannelizer_chime_U32::~cudaUpchannelizer_chime_U32() {}

std::int64_t
cudaUpchannelizer_chime_U32::num_consumed_elements(std::int64_t num_available_elements) const {
    if (num_processed_elements(num_available_elements) < cuda_algorithm_overlap)
        return 0;
    return num_processed_elements(num_available_elements) - cuda_algorithm_overlap;
}
std::int64_t
cudaUpchannelizer_chime_U32::num_produced_elements(std::int64_t num_available_elements) const {
    return div_noremainder(num_consumed_elements(num_available_elements),
                           cuda_upchannelization_factor);
}

std::int64_t
cudaUpchannelizer_chime_U32::num_processed_elements(std::int64_t num_available_elements) const {
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cudaUpchannelizer_chime_U32::wait_on_precondition() {
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

cudaEvent_t cudaUpchannelizer_chime_U32::execute(cudaPipelineState& /*pipestate*/,
                                                 const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    void* const G_memory = G_buffer.get_ndarray().data();
    void* const E_memory = E_buffer.get_ndarray().data();
    void* const Ebar_memory = Ebar_buffer.get_ndarray().data();
    void* const info_memory = info_buffer.get_ndarray().data();

    // Since we use a ring buffer we need to set the metadata only once
    static bool did_set_metadata = false;
    if (instance_num == 0 && !did_set_metadata) {
        did_set_metadata = true;

        G_buffer.check_metadata();
        E_buffer.check_metadata();
        Ebar_buffer.set_metadata(E_buffer.get_metadata());

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
            Ebar_freq_upchan_factor.at(freq) =
                E_freq_upchan_factor.at(coarse_freq) * cuda_upchannelization_factor;
        }
        Ebar_meta->set_freq_upchan_factor(Ebar_freq_upchan_factor);

        const auto E_freq_upchan_index = E_meta->get_freq_upchan_index();
        std::vector<int> Ebar_freq_upchan_index(Ebar_nfreq);
        for (int freq = 0; freq < Ebar_nfreq; ++freq) {
            const int upchan_index = freq % cuda_upchannelization_factor;
            Ebar_freq_upchan_index.at(freq) = upchan_index;
        }
        Ebar_meta->set_freq_upchan_index(Ebar_freq_upchan_index);

        const auto E_time_downsampling_fpga = E_meta->get_time_downsampling_fpga();
        const auto Ebar_time_downsampling_fpga =
            E_time_downsampling_fpga * cuda_upchannelization_factor;
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
    std::int32_t T_min_arg;
    std::int32_t T_max_arg;
    std::int32_t Tbar_min_arg;
    std::int32_t Tbar_max_arg;
    std::int32_t Fmin_arg;
    std::int32_t Fmax_arg;
    array_desc G_arg(G_memory, G_length_in_bytes);
    array_desc E_arg(E_memory, E_length_in_bytes);
    array_desc Ebar_arg(Ebar_memory, Ebar_length_in_bytes);
    array_desc info_arg(info_memory, info_length_in_bytes);
    void* args[] = {
        &exc_arg,  &T_min_arg, &T_max_arg, &Tbar_min_arg, &Tbar_max_arg, &Fmin_arg,
        &Fmax_arg, &G_arg,     &E_arg,     &Ebar_arg,     &info_arg,
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

    if (poison_buffers) {
        Ebar_buffer.set_to_poison(0x00, 0, cuda_upchannelization_factor * (Fmax - Fmin));
        info_buffer.set_to_poison(0xff);

        // Initialize host-side buffer arrays
        CHECK_CUDA_ERROR(cudaMemsetAsync(info_memory, 0xff, info_length_in_bytes,
                                         device.getStream(cuda_stream_id)));
    } // if (poison_buffers)

    const std::string symname = "Upchannelizer_chime_U32_" + std::string(kernel_symbol);
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

    if (poison_buffers) {
        // Copy results back to host memory
        CHECK_CUDA_ERROR(cudaMemcpyAsync(host_info_buffer.data(), info_memory, info_length_in_bytes,
                                         cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

        CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));

        // Check error codes
        const std::uint32_t error_code =
            *std::max_element((const std::uint32_t*)host_info_buffer.data(),
                              (const std::uint32_t*)(host_info_buffer.data()
                                                     + blocks * info_lengths[info_index_warp]
                                                           * info_lengths[info_index_thread]));
        if (error_code != 0)
            ERROR("CUDA kernel Upchannelizer_chime_U32 returned error code: {}", error_code);

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
                            ERROR("CUDA kernel Upchannelizer_chime_U32 returned 'info' value {:d} "
                                  "for thread {:d} warp {:d} block {:d} at index {:d} (zero "
                                  "indicates no error)",
                                  val, thread, warp, block, i);
                    }
                }
            }
        }

        Ebar_buffer.check_for_poison(0x00, 0, cuda_upchannelization_factor * (Fmax - Fmin));
    } // if (poison_buffers)

    return record_end_event();
}

void cudaUpchannelizer_chime_U32::finalize_frame() {
    // Advance the input ring buffer
    E_buffer.finish_read();

    // Advance the output ring buffer
    Ebar_buffer.finish_write();

    cudaCommand::finalize_frame();
}
