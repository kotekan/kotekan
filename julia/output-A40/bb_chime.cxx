/**
 * @file
 * @brief CUDA BasebandBeamformer_chime kernel
 *
 * This file has been generated automatically.
 * Do not modify this C++ file, your changes will be lost.
 */

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
#include <ringbuffer.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::round_down, kotekan::div_noremainder, kotekan::div, kotekan::mod;

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
    using array_desc = CuDeviceArray<int32_t, 1>;

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
        "_Z2bb5Int32S_13CuDeviceArrayI6Int8x4Li1ELi1EES0_I6Int4x8Li1ELi1EES0_IS_Li1ELi1EES0_IS2_"
        "Li1ELi1EES0_IS_Li1ELi1EES0_IS_Li1ELi1EE";

    // Kernel arguments:
    enum class args { Tmin, Tmax, A, E, s, J, info, log, count };

    // Tmin: Tmin
    static constexpr const char* Tmin_name = "Tmin";
    static constexpr chordDataType Tmin_type = int32;
    //
    // Tmax: Tmax
    static constexpr const char* Tmax_name = "Tmax";
    static constexpr chordDataType Tmax_type = int32;
    //
    // A: gpu_mem_phase
    static constexpr const char* A_name = "A";
    static constexpr chordDataType A_type = int8;
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
    static constexpr std::ptrdiff_t A_length =
        chord_datatype_bytes(A_type) * 2 * 1024 * 16 * 2 * 16;
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
    static_assert(A_length == chord_datatype_bytes(A_type) * A_strides[A_rank]);
    //
    // E: gpu_mem_voltage
    static constexpr const char* E_name = "E";
    static constexpr chordDataType E_type = int4p4;
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
    static constexpr std::ptrdiff_t E_length = chord_datatype_bytes(E_type) * 1024 * 2 * 16 * 65536;
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
    static_assert(E_length == chord_datatype_bytes(E_type) * E_strides[E_rank]);
    //
    // s: gpu_mem_output_scaling
    static constexpr const char* s_name = "s";
    static constexpr chordDataType s_type = int32;
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
    static constexpr std::ptrdiff_t s_length = chord_datatype_bytes(s_type) * 16 * 2 * 16;
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
    static_assert(s_length == chord_datatype_bytes(s_type) * s_strides[s_rank]);
    //
    // J: gpu_mem_formed_beams
    static constexpr const char* J_name = "J";
    static constexpr chordDataType J_type = int4p4;
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
    static constexpr std::ptrdiff_t J_length = chord_datatype_bytes(J_type) * 16384 * 2 * 16 * 16;
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
    static_assert(J_length == chord_datatype_bytes(J_type) * J_strides[J_rank]);
    //
    // info: gpu_mem_info
    static constexpr const char* info_name = "info";
    static constexpr chordDataType info_type = int32;
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
    static constexpr std::ptrdiff_t info_length = chord_datatype_bytes(info_type) * 32 * 4 * 32;
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
    static_assert(info_length == chord_datatype_bytes(info_type) * info_strides[info_rank]);
    //
    // log: gpu_mem_log
    static constexpr const char* log_name = "log";
    static constexpr chordDataType log_type = int32;
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
    static constexpr std::ptrdiff_t log_length = chord_datatype_bytes(log_type) * 32;
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
    static_assert(log_length == chord_datatype_bytes(log_type) * log_strides[log_rank]);
    //

    // Kotekan buffer names
    const std::string A_memname;
    const std::string E_memname;
    const std::string s_memname;
    const std::string J_memname;
    const std::string info_memname;
    const std::string log_memname;

    // Host-side buffer arrays
    std::vector<std::uint8_t> info_host;
    std::vector<std::uint8_t> log_host;

    static constexpr std::ptrdiff_t E_T_sample_bytes = chord_datatype_bytes(E_type)
                                                       * E_lengths[E_index_D] * E_lengths[E_index_P]
                                                       * E_lengths[E_index_F];

    RingBuffer* input_ringbuf_signal;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tmin, Tmax;
};

REGISTER_CUDA_COMMAND(cudaBasebandBeamformer_chime);

cudaBasebandBeamformer_chime::cudaBasebandBeamformer_chime(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& host_buffers,
                                                           cudaDeviceInterface& device,
                                                           const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "BasebandBeamformer_chime", "BasebandBeamformer_chime.ptx"),
    A_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    s_memname(config.get<std::string>(unique_name, "gpu_mem_output_scaling")),
    J_memname(config.get<std::string>(unique_name, "gpu_mem_formed_beams")),
    info_memname(unique_name + "/gpu_mem_info"), log_memname(unique_name + "/gpu_mem_log"),

    info_host(info_length), log_host(log_length),
    // Find input buffer used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")))) {
    // Check ringbuffer size
    assert(input_ringbuf_signal->size == E_length);

    // Register host memory
    {
        const cudaError_t ierr = cudaHostRegister(info_host.data(), info_host.size(), 0);
        assert(ierr == cudaSuccess);
    }
    {
        const cudaError_t ierr = cudaHostRegister(log_host.data(), log_host.size(), 0);
        assert(ierr == cudaSuccess);
    }

    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(A_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(s_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(J_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_log", false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage needs to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "BasebandBeamformer_chime_");
    }

    if (instance_num == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
    }
}

cudaBasebandBeamformer_chime::~cudaBasebandBeamformer_chime() {}

std::int64_t
cudaBasebandBeamformer_chime::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_produced_elements(num_available_elements);
}
std::int64_t
cudaBasebandBeamformer_chime::num_produced_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements);
}

std::int64_t
cudaBasebandBeamformer_chime::num_processed_elements(std::int64_t num_available_elements) const {
    assert(num_available_elements >= cuda_granularity_number_of_timesamples);
    return cuda_granularity_number_of_timesamples;
}

int cudaBasebandBeamformer_chime::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    const std::ptrdiff_t input_bytes = cuda_granularity_number_of_timesamples * E_T_sample_bytes;
    DEBUG("Input ring-buffer byte count: {:d}", input_bytes);
    DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::ptrdiff_t> val_in =
        input_ringbuf_signal->wait_and_claim_readable(unique_name, instance_num, input_bytes);
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    if (!val_in.has_value())
        return -1;
    const std::ptrdiff_t input_cursor = val_in.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);

    // How many inputs samples are available?
    const std::ptrdiff_t T_available = div_noremainder(input_bytes, E_T_sample_bytes);
    DEBUG("Available samples:      T_available: {:d}", T_available);

    // How many outputs will we process and consume?
    const std::ptrdiff_t T_processed = num_processed_elements(T_available);
    const std::ptrdiff_t T_consumed = num_consumed_elements(T_available);
    DEBUG("Will process (samples): T_processed: {:d}", T_processed);
    DEBUG("Will consume (samples): T_consumed:  {:d}", T_consumed);
    assert(T_processed > 0);
    assert(T_consumed <= T_processed);
    const std::ptrdiff_t T_consumed2 = num_consumed_elements(T_processed);
    assert(T_consumed2 == T_consumed);

    Tmin = div_noremainder(input_cursor, E_T_sample_bytes);
    Tmax = Tmin + T_processed;
    const std::ptrdiff_t Tlength = Tmax - Tmin;
    DEBUG("Input samples:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);
    DEBUG("    Tlength: {:d}", Tlength);

    return 0;
}

cudaEvent_t cudaBasebandBeamformer_chime::execute(cudaPipelineState& /*pipestate*/,
                                                  const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    void* const A_memory =
        args::A == args::E ? device.get_gpu_memory(A_memname, input_ringbuf_signal->size)
        : args::A == args::A || args::A == args::s
            ? device.get_gpu_memory(A_memname, A_length)
            : device.get_gpu_memory_array(A_memname, gpu_frame_id, _gpu_buffer_depth, A_length);
    void* const E_memory =
        args::E == args::E ? device.get_gpu_memory(E_memname, input_ringbuf_signal->size)
        : args::E == args::A || args::E == args::s
            ? device.get_gpu_memory(E_memname, E_length)
            : device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const s_memory =
        args::s == args::E ? device.get_gpu_memory(s_memname, input_ringbuf_signal->size)
        : args::s == args::A || args::s == args::s
            ? device.get_gpu_memory(s_memname, s_length)
            : device.get_gpu_memory_array(s_memname, gpu_frame_id, _gpu_buffer_depth, s_length);
    void* const J_memory =
        args::J == args::E ? device.get_gpu_memory(J_memname, input_ringbuf_signal->size)
        : args::J == args::A || args::J == args::s
            ? device.get_gpu_memory(J_memname, J_length)
            : device.get_gpu_memory_array(J_memname, gpu_frame_id, _gpu_buffer_depth, J_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);
    void* const log_memory = device.get_gpu_memory(log_memname, log_length);

    // A is an input buffer: check metadata
    const std::shared_ptr<metadataObject> A_mc =
        args::A == args::E ? input_ringbuf_signal->get_metadata(0)
                           : device.get_gpu_memory_array_metadata(A_memname, gpu_frame_id);
    assert(A_mc);
    assert(metadata_is_chord(A_mc));
    const std::shared_ptr<chordMetadata> A_meta = get_chord_metadata(A_mc);
    DEBUG("input A array: {:s} {:s}", A_meta->get_type_string(), A_meta->get_dimensions_string());
    assert(std::strncmp(A_meta->name, A_name, sizeof A_meta->name) == 0);
    assert(A_meta->type == A_type);
    assert(A_meta->dims == A_rank);
    for (std::ptrdiff_t dim = 0; dim < A_rank; ++dim) {
        assert(std::strncmp(A_meta->dim_name[A_rank - 1 - dim], A_labels[dim],
                            sizeof A_meta->dim_name[A_rank - 1 - dim])
               == 0);
        if (args::A == args::E) {
            assert(A_meta->dim[A_rank - 1 - dim] <= int(A_lengths[dim]));
            assert(A_meta->stride[A_rank - 1 - dim] == A_strides[dim]);
        } else {
            assert(A_meta->dim[A_rank - 1 - dim] == int(A_lengths[dim]));
            assert(A_meta->stride[A_rank - 1 - dim] == A_strides[dim]);
        }
    }
    //
    // E is an input buffer: check metadata
    const std::shared_ptr<metadataObject> E_mc =
        args::E == args::E ? input_ringbuf_signal->get_metadata(0)
                           : device.get_gpu_memory_array_metadata(E_memname, gpu_frame_id);
    assert(E_mc);
    assert(metadata_is_chord(E_mc));
    const std::shared_ptr<chordMetadata> E_meta = get_chord_metadata(E_mc);
    DEBUG("input E array: {:s} {:s}", E_meta->get_type_string(), E_meta->get_dimensions_string());
    assert(std::strncmp(E_meta->name, E_name, sizeof E_meta->name) == 0);
    assert(E_meta->type == E_type);
    assert(E_meta->dims == E_rank);
    for (std::ptrdiff_t dim = 0; dim < E_rank; ++dim) {
        assert(std::strncmp(E_meta->dim_name[E_rank - 1 - dim], E_labels[dim],
                            sizeof E_meta->dim_name[E_rank - 1 - dim])
               == 0);
        if (args::E == args::E) {
            assert(E_meta->dim[E_rank - 1 - dim] <= int(E_lengths[dim]));
            assert(E_meta->stride[E_rank - 1 - dim] == E_strides[dim]);
        } else {
            assert(E_meta->dim[E_rank - 1 - dim] == int(E_lengths[dim]));
            assert(E_meta->stride[E_rank - 1 - dim] == E_strides[dim]);
        }
    }
    //
    // s is an input buffer: check metadata
    const std::shared_ptr<metadataObject> s_mc =
        args::s == args::E ? input_ringbuf_signal->get_metadata(0)
                           : device.get_gpu_memory_array_metadata(s_memname, gpu_frame_id);
    assert(s_mc);
    assert(metadata_is_chord(s_mc));
    const std::shared_ptr<chordMetadata> s_meta = get_chord_metadata(s_mc);
    DEBUG("input s array: {:s} {:s}", s_meta->get_type_string(), s_meta->get_dimensions_string());
    assert(std::strncmp(s_meta->name, s_name, sizeof s_meta->name) == 0);
    assert(s_meta->type == s_type);
    assert(s_meta->dims == s_rank);
    for (std::ptrdiff_t dim = 0; dim < s_rank; ++dim) {
        assert(std::strncmp(s_meta->dim_name[s_rank - 1 - dim], s_labels[dim],
                            sizeof s_meta->dim_name[s_rank - 1 - dim])
               == 0);
        if (args::s == args::E) {
            assert(s_meta->dim[s_rank - 1 - dim] <= int(s_lengths[dim]));
            assert(s_meta->stride[s_rank - 1 - dim] == s_strides[dim]);
        } else {
            assert(s_meta->dim[s_rank - 1 - dim] == int(s_lengths[dim]));
            assert(s_meta->stride[s_rank - 1 - dim] == s_strides[dim]);
        }
    }
    //
    // J is an output buffer: set metadata
    std::shared_ptr<metadataObject> const J_mc =
        device.create_gpu_memory_array_metadata(J_memname, gpu_frame_id, E_mc->parent_pool);
    std::shared_ptr<chordMetadata> const J_meta = get_chord_metadata(J_mc);
    *J_meta = *E_meta;
    std::strncpy(J_meta->name, J_name, sizeof J_meta->name);
    J_meta->type = J_type;
    J_meta->dims = J_rank;
    for (std::ptrdiff_t dim = 0; dim < J_rank; ++dim) {
        std::strncpy(J_meta->dim_name[J_rank - 1 - dim], J_labels[dim],
                     sizeof J_meta->dim_name[J_rank - 1 - dim]);
        J_meta->dim[J_rank - 1 - dim] = J_lengths[dim];
        J_meta->stride[J_rank - 1 - dim] = J_strides[dim];
    }
    DEBUG("output J array: {:s} {:s}", J_meta->get_type_string(), J_meta->get_dimensions_string());
    //

    record_start_event();

    DEBUG("gpu_frame_id: {}", gpu_frame_id);

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
    const std::ptrdiff_t T_ringbuf = input_ringbuf_signal->size / E_T_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", T_ringbuf);

    const std::ptrdiff_t Tlength = Tmax - Tmin;
    DEBUG("Processed input samples: {:d}", Tlength);

    DEBUG("Kernel arguments:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    Tmin_arg = mod(Tmin, T_ringbuf);
    Tmax_arg = mod(Tmin, T_ringbuf) + Tlength;

    // Update metadata
    assert(J_meta->dim[J_rank - 1 - J_index_T] == int(Tlength));
    assert(J_meta->dim[J_rank - 1 - J_index_T] == int(J_lengths[J_index_T]));

    // Since we do not use a ring buffer we need to set `meta->sample0_offset`
    J_meta->sample0_offset = Tmin;

    assert(J_meta->nfreq >= 0);
    assert(J_meta->nfreq == J_meta->nfreq);
    for (int freq = 0; freq < J_meta->nfreq; ++freq) {
        J_meta->freq_upchan_factor[freq] = J_meta->freq_upchan_factor[freq];
        J_meta->time_downsampling_fpga[freq] = J_meta->time_downsampling_fpga[freq];
    }

    // Copy inputs to device memory

#ifdef DEBUGGING
    // Initialize host-side buffer arrays
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(log_memory, 0xff, log_length, device.getStream(cuda_stream_id)));
#endif

#ifdef DEBUGGING
    // Poison outputs
    CHECK_CUDA_ERROR(cudaMemsetAsync(J_memory, 0x88, J_length, device.getStream(cuda_stream_id)));
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
    const std::int32_t error_code = *std::max_element((const std::int32_t*)&*info_host.begin(),
                                                      (const std::int32_t*)&*info_host.end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    for (std::size_t i = 0; i < info_host.size(); ++i)
        if (info_host[i] != 0)
            ERROR("cudaBasebandBeamformer_chime returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host[i], i);

    // Check log codes
    const std::int32_t log_code = *std::max_element((const std::int32_t*)&*log_host.begin(),
                                                    (const std::int32_t*)&*log_host.end());
    if (log_code != 0)
        ERROR("CUDA kernel returned log code cuLaunchKernel: {}", log_code);

    for (std::size_t i = 0; i < log_host.size(); ++i)
        if (log_host[i] != 0)
            ERROR("cudaBasebandBeamformer_chime returned 'log' value {:d} at index {:d} (zero "
                  "indicates success)",
                  log_host[i], i);
#endif

#ifdef DEBUGGING
    // Check outputs for poison
    std::vector<std::uint8_t> J_buffer(J_length);
    CHECK_CUDA_ERROR(cudaMemcpy(J_buffer.data(), J_memory, J_length, cudaMemcpyDeviceToHost));

    const bool J_found_error = std::memchr(J_buffer.data(), 0x88, J_buffer.size());
    assert(!J_found_error);
#endif

    return record_end_event();
}

void cudaBasebandBeamformer_chime::finalize_frame() {
    const std::ptrdiff_t Tlength = Tmax - Tmin;

    // Advance the input ringbuffer
    const std::ptrdiff_t T_consumed = num_consumed_elements(Tlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples: {:d}", T_consumed);
    DEBUG("    Consumed bytes:   {:d}", T_consumed * E_T_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, instance_num, T_consumed * E_T_sample_bytes);

    cudaCommand::finalize_frame();
}
