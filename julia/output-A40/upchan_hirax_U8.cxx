/**
 * @file
 * @brief CUDA Upchannelizer_hirax_U8 kernel
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
 * @class cudaUpchannelizer_hirax_U8
 * @brief cudaCommand for Upchannelizer_hirax_U8
 */
class cudaUpchannelizer_hirax_U8 : public cudaCommand {
public:
    cudaUpchannelizer_hirax_U8(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device,
                               const int instance_num);
    virtual ~cudaUpchannelizer_hirax_U8();

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
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 256;
    static constexpr int cuda_number_of_frequencies = 64;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_taps = 4;
    static constexpr int cuda_max_number_of_timesamples = 65536;
    static constexpr int cuda_granularity_number_of_timesamples = 256;
    static constexpr int cuda_algorithm_overlap = 24;
    static constexpr int cuda_upchannelization_factor = 8;

    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

    // Kernel compile parameters:
    static constexpr int minthreads = 256;
    static constexpr int blocks_per_sm = 4;
    static constexpr int blocks_per_frequency = 4;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 8;
    static constexpr int max_blocks = 256;
    static constexpr int shmem_bytes = 36992;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z7upchan85Int32S_S_S_S_S_13CuDeviceArrayI9Float16x2Li1ELi1EES0_I6Int4x8Li1ELi1EES0_IS2_"
        "Li1ELi1EES0_IS_Li1ELi1EE";

    // Kernel arguments:
    enum class args { Tmin, Tmax, Tbarmin, Tbarmax, Fmin, Fmax, G_U8, E, Ebar, info, count };

    // Tmin: Tmin
    static constexpr const char* Tmin_name = "Tmin";
    static constexpr chordDataType Tmin_type = int32;
    //
    // Tmax: Tmax
    static constexpr const char* Tmax_name = "Tmax";
    static constexpr chordDataType Tmax_type = int32;
    //
    // Tbarmin: Tbarmin
    static constexpr const char* Tbarmin_name = "Tbarmin";
    static constexpr chordDataType Tbarmin_type = int32;
    //
    // Tbarmax: Tbarmax
    static constexpr const char* Tbarmax_name = "Tbarmax";
    static constexpr chordDataType Tbarmax_type = int32;
    //
    // Fmin: Fmin
    static constexpr const char* Fmin_name = "Fmin";
    static constexpr chordDataType Fmin_type = int32;
    //
    // Fmax: Fmax
    static constexpr const char* Fmax_name = "Fmax";
    static constexpr chordDataType Fmax_type = int32;
    //
    // G_U8: gpu_mem_gain
    static constexpr const char* G_U8_name = "G_U8";
    static constexpr chordDataType G_U8_type = float16;
    enum G_U8_indices {
        G_U8_index_Fbar,
        G_U8_rank,
    };
    static constexpr std::array<const char*, G_U8_rank> G_U8_labels = {
        "Fbar",
    };
    static constexpr std::array<std::ptrdiff_t, G_U8_rank> G_U8_lengths = {
        128,
    };
    static constexpr std::ptrdiff_t G_U8_length = chord_datatype_bytes(G_U8_type) * 128;
    static_assert(G_U8_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    static constexpr auto G_U8_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= G_U8_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, G_U8_rank + 1> G_U8_strides = {
        G_U8_calc_stride(G_U8_index_Fbar),
        G_U8_calc_stride(G_U8_rank),
    };
    static_assert(G_U8_length == chord_datatype_bytes(G_U8_type) * G_U8_strides[G_U8_rank]);
    //
    // E: gpu_mem_input_voltage
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
        256,
        2,
        64,
        65536,
    };
    static constexpr std::ptrdiff_t E_length = chord_datatype_bytes(E_type) * 256 * 2 * 64 * 65536;
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
    // Ebar: gpu_mem_output_voltage
    static constexpr const char* Ebar_name = "Ebar";
    static constexpr chordDataType Ebar_type = int4p4;
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
        256,
        2,
        128,
        8192,
    };
    static constexpr std::ptrdiff_t Ebar_length =
        chord_datatype_bytes(Ebar_type) * 256 * 2 * 128 * 8192;
    static_assert(Ebar_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    static_assert(Ebar_length == chord_datatype_bytes(Ebar_type) * Ebar_strides[Ebar_rank]);
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
        8,
        256,
    };
    static constexpr std::ptrdiff_t info_length = chord_datatype_bytes(info_type) * 32 * 8 * 256;
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

    // Kotekan buffer names
    const std::string G_U8_memname;
    const std::string E_memname;
    const std::string Ebar_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::uint8_t> info_host;

    static constexpr std::ptrdiff_t E_T_sample_bytes = chord_datatype_bytes(E_type)
                                                       * E_lengths[E_index_D] * E_lengths[E_index_P]
                                                       * E_lengths[E_index_F];
    static constexpr std::ptrdiff_t Ebar_Tbar_sample_bytes =
        chord_datatype_bytes(Ebar_type) * Ebar_lengths[Ebar_index_D] * Ebar_lengths[Ebar_index_P]
        * Ebar_lengths[Ebar_index_Fbar];

    RingBuffer* input_ringbuf_signal;
    RingBuffer* output_ringbuf_signal;

    // How many frequencies we will process
    const int Fmin, Fmax;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tmin, Tmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tbarmin, Tbarmax;
};

REGISTER_CUDA_COMMAND(cudaUpchannelizer_hirax_U8);

cudaUpchannelizer_hirax_U8::cudaUpchannelizer_hirax_U8(Config& config,
                                                       const std::string& unique_name,
                                                       bufferContainer& host_buffers,
                                                       cudaDeviceInterface& device,
                                                       const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "Upchannelizer_hirax_U8", "Upchannelizer_hirax_U8.ptx"),
    G_U8_memname(config.get<std::string>(unique_name, "gpu_mem_gain")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_input_voltage")),
    Ebar_memname(config.get<std::string>(unique_name, "gpu_mem_output_voltage")),
    info_memname(unique_name + "/gpu_mem_info"),

    info_host(info_length),
    // Find input and output buffers used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")))),
    output_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "out_signal")))),
    Fmin(config.get<int>(unique_name, "Fmin")), Fmax(config.get<int>(unique_name, "Fmax")) {
    // Check ringbuffer sizes
    assert(input_ringbuf_signal->size == E_length);
    assert(output_ringbuf_signal->size == Ebar_length);

    // Register host memory
    {
        const cudaError_t ierr = cudaHostRegister(info_host.data(), info_host.size(), 0);
        assert(ierr == cudaSuccess);
    }

    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(G_U8_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(Ebar_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage needs to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "Upchannelizer_hirax_U8_");
    }

    if (instance_num == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
        output_ringbuf_signal->register_producer(unique_name);
        output_ringbuf_signal->allocate_new_metadata_object(0);
    }
}

cudaUpchannelizer_hirax_U8::~cudaUpchannelizer_hirax_U8() {}

std::int64_t
cudaUpchannelizer_hirax_U8::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) - cuda_algorithm_overlap;
}
std::int64_t
cudaUpchannelizer_hirax_U8::num_produced_elements(std::int64_t num_available_elements) const {
    assert(num_consumed_elements(num_available_elements) % cuda_upchannelization_factor == 0);
    return num_consumed_elements(num_available_elements) / cuda_upchannelization_factor;
}

std::int64_t
cudaUpchannelizer_hirax_U8::num_processed_elements(std::int64_t num_available_elements) const {
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cudaUpchannelizer_hirax_U8::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::ptrdiff_t> val_in1 =
        input_ringbuf_signal->wait_without_claiming(unique_name, instance_num);
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    if (!val_in1.has_value())
        return -1;
    const std::ptrdiff_t input_bytes = val_in1.value();
    DEBUG("Input ring-buffer byte count: {:d}", input_bytes);

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

    const std::optional<std::ptrdiff_t> val_in2 = input_ringbuf_signal->wait_and_claim_readable(
        unique_name, instance_num, T_consumed * E_T_sample_bytes);
    if (!val_in2.has_value())
        return -1;
    const std::ptrdiff_t input_cursor = val_in2.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    Tmin = div_noremainder(input_cursor, E_T_sample_bytes);
    Tmax = Tmin + T_processed;
    const std::ptrdiff_t Tlength = Tmax - Tmin;
    DEBUG("Input samples:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);
    DEBUG("    Tlength: {:d}", Tlength);

    // How many outputs will we produce?
    const std::ptrdiff_t Tbar_produced = num_produced_elements(T_available);
    DEBUG("Will produce (samples): Tbar_produced: {:d}", Tbar_produced);
    const std::ptrdiff_t Tbarlength = Tbar_produced;

    // to bytes
    const std::ptrdiff_t output_bytes = Tbarlength * Ebar_Tbar_sample_bytes;
    DEBUG("Will produce {:d} output bytes", output_bytes);

    // Wait for space to be available in our output ringbuffer...
    DEBUG("Waiting for output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::optional<std::ptrdiff_t> val_out =
        output_ringbuf_signal->wait_for_writable(unique_name, instance_num, output_bytes);
    DEBUG("Finished waiting for output for data frame {:d}.", gpu_frame_id);
    if (!val_out.has_value())
        return -1;
    const std::ptrdiff_t output_cursor = val_out.value();
    DEBUG("Output ring-buffer byte offset {:d}", output_cursor);

    assert(mod(output_cursor, Ebar_Tbar_sample_bytes) == 0);
    Tbarmin = output_cursor / Ebar_Tbar_sample_bytes;
    Tbarmax = Tbarmin + Tbarlength;
    DEBUG("Output samples:");
    DEBUG("    Tbarmin:    {:d}", Tbarmin);
    DEBUG("    Tbarmax:    {:d}", Tbarmax);
    DEBUG("    Tbarlength: {:d}", Tbarlength);

    return 0;
}

cudaEvent_t cudaUpchannelizer_hirax_U8::execute(cudaPipelineState& /*pipestate*/,
                                                const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    void* const G_U8_memory =
        args::G_U8 == args::E ? device.get_gpu_memory(G_U8_memname, input_ringbuf_signal->size)
        : args::G_U8 == args::Ebar
            ? device.get_gpu_memory(G_U8_memname, output_ringbuf_signal->size)
        : args::G_U8 == args::G_U8 ? device.get_gpu_memory(G_U8_memname, G_U8_length)
                                   : device.get_gpu_memory_array(G_U8_memname, gpu_frame_id,
                                                                 _gpu_buffer_depth, G_U8_length);
    void* const E_memory =
        args::E == args::E      ? device.get_gpu_memory(E_memname, input_ringbuf_signal->size)
        : args::E == args::Ebar ? device.get_gpu_memory(E_memname, output_ringbuf_signal->size)
        : args::E == args::G_U8
            ? device.get_gpu_memory(E_memname, E_length)
            : device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const Ebar_memory =
        args::Ebar == args::E ? device.get_gpu_memory(Ebar_memname, input_ringbuf_signal->size)
        : args::Ebar == args::Ebar
            ? device.get_gpu_memory(Ebar_memname, output_ringbuf_signal->size)
        : args::Ebar == args::G_U8 ? device.get_gpu_memory(Ebar_memname, Ebar_length)
                                   : device.get_gpu_memory_array(Ebar_memname, gpu_frame_id,
                                                                 _gpu_buffer_depth, Ebar_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // G_U8 is an input buffer: check metadata
    const std::shared_ptr<metadataObject> G_U8_mc =
        args::G_U8 == args::E ? input_ringbuf_signal->get_metadata(0)
                              : device.get_gpu_memory_array_metadata(G_U8_memname, gpu_frame_id);
    assert(G_U8_mc);
    assert(metadata_is_chord(G_U8_mc));
    const std::shared_ptr<chordMetadata> G_U8_meta = get_chord_metadata(G_U8_mc);
    DEBUG("input G_U8 array: {:s} {:s}", G_U8_meta->get_type_string(),
          G_U8_meta->get_dimensions_string());
    assert(std::strncmp(G_U8_meta->name, G_U8_name, sizeof G_U8_meta->name) == 0);
    assert(G_U8_meta->type == G_U8_type);
    assert(G_U8_meta->dims == G_U8_rank);
    for (std::ptrdiff_t dim = 0; dim < G_U8_rank; ++dim) {
        assert(std::strncmp(G_U8_meta->dim_name[G_U8_rank - 1 - dim], G_U8_labels[dim],
                            sizeof G_U8_meta->dim_name[G_U8_rank - 1 - dim])
               == 0);
        if (args::G_U8 == args::E && dim == E_index_T) {
            assert(G_U8_meta->dim[G_U8_rank - 1 - dim] <= int(G_U8_lengths[dim]));
            assert(G_U8_meta->stride[G_U8_rank - 1 - dim] <= G_U8_strides[dim]);
        } else {
            assert(G_U8_meta->dim[G_U8_rank - 1 - dim] == int(G_U8_lengths[dim]));
            assert(G_U8_meta->stride[G_U8_rank - 1 - dim] == G_U8_strides[dim]);
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
        if (args::E == args::E && dim == E_index_T) {
            assert(E_meta->dim[E_rank - 1 - dim] <= int(E_lengths[dim]));
            assert(E_meta->stride[E_rank - 1 - dim] <= E_strides[dim]);
        } else {
            assert(E_meta->dim[E_rank - 1 - dim] == int(E_lengths[dim]));
            assert(E_meta->stride[E_rank - 1 - dim] == E_strides[dim]);
        }
    }
    //
    // Ebar is an output buffer: set metadata
    std::shared_ptr<metadataObject> const Ebar_mc =
        args::Ebar == args::Ebar ? output_ringbuf_signal->get_metadata(0)
                                 : device.create_gpu_memory_array_metadata(
                                     Ebar_memname, gpu_frame_id, E_mc->parent_pool);
    std::shared_ptr<chordMetadata> const Ebar_meta = get_chord_metadata(Ebar_mc);
    *Ebar_meta = *E_meta;
    std::strncpy(Ebar_meta->name, Ebar_name, sizeof Ebar_meta->name);
    Ebar_meta->type = Ebar_type;
    Ebar_meta->dims = Ebar_rank;
    for (std::ptrdiff_t dim = 0; dim < Ebar_rank; ++dim) {
        std::strncpy(Ebar_meta->dim_name[Ebar_rank - 1 - dim], Ebar_labels[dim],
                     sizeof Ebar_meta->dim_name[Ebar_rank - 1 - dim]);
        Ebar_meta->dim[Ebar_rank - 1 - dim] = Ebar_lengths[dim];
        Ebar_meta->stride[Ebar_rank - 1 - dim] = Ebar_strides[dim];
    }
    DEBUG("output Ebar array: {:s} {:s}", Ebar_meta->get_type_string(),
          Ebar_meta->get_dimensions_string());
    //

    record_start_event();

    DEBUG("gpu_frame_id: {}", gpu_frame_id);

    const char* exc_arg = "exception";
    std::int32_t Tmin_arg;
    std::int32_t Tmax_arg;
    std::int32_t Tbarmin_arg;
    std::int32_t Tbarmax_arg;
    std::int32_t Fmin_arg;
    std::int32_t Fmax_arg;
    array_desc G_U8_arg(G_U8_memory, G_U8_length);
    array_desc E_arg(E_memory, E_length);
    array_desc Ebar_arg(Ebar_memory, Ebar_length);
    array_desc info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg,  &Tmin_arg, &Tmax_arg, &Tbarmin_arg, &Tbarmax_arg, &Fmin_arg,
        &Fmax_arg, &G_U8_arg, &E_arg,    &Ebar_arg,    &info_arg,
    };

    // Set E_memory to beginning of input ring buffer
    E_arg = array_desc(E_memory, E_length);

    // Set Ebar_memory to beginning of output ring buffer
    Ebar_arg = array_desc(Ebar_memory, Ebar_length);

    // Ringbuffer size
    const std::ptrdiff_t T_ringbuf = input_ringbuf_signal->size / E_T_sample_bytes;
    const std::ptrdiff_t Tbar_ringbuf = output_ringbuf_signal->size / Ebar_Tbar_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", T_ringbuf);
    DEBUG("Output ringbuffer size (samples): {:d}", Tbar_ringbuf);

    const std::ptrdiff_t Tlength = Tmax - Tmin;
    const std::ptrdiff_t Tbarlength = Tbarmax - Tbarmin;
    DEBUG("Processed input samples: {:d}", Tlength);
    DEBUG("Produced output samples: {:d}", Tbarlength);

    DEBUG("Kernel arguments:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);
    DEBUG("    Tbarmin: {:d}", Tbarmin);
    DEBUG("    Tbarmax: {:d}", Tbarmax);

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    Tmin_arg = mod(Tmin, T_ringbuf);
    Tmax_arg = mod(Tmin, T_ringbuf) + Tlength;
    Tbarmin_arg = mod(Tbarmin, Tbar_ringbuf);
    Tbarmax_arg = mod(Tbarmin, Tbar_ringbuf) + Tbarlength;

    // Pass frequency spans to kernel
    Fmin_arg = Fmin;
    Fmax_arg = Fmax;

    // Update metadata
    Ebar_meta->dim[Ebar_rank - 1 - Ebar_index_Tbar] = Tbarlength;
    assert(Ebar_meta->dim[Ebar_rank - 1 - Ebar_index_Tbar] <= int(Ebar_lengths[Ebar_index_Tbar]));
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    assert(Ebar_meta->nfreq >= 0);
    assert(Ebar_meta->nfreq == E_meta->nfreq);
    for (int freq = 0; freq < Ebar_meta->nfreq; ++freq) {
        Ebar_meta->freq_upchan_factor[freq] =
            cuda_upchannelization_factor * E_meta->freq_upchan_factor[freq];
        Ebar_meta->half_fpga_sample0[freq] =
            E_meta->half_fpga_sample0[freq] + cuda_number_of_taps - 1;
        Ebar_meta->time_downsampling_fpga[freq] =
            cuda_upchannelization_factor * E_meta->time_downsampling_fpga[freq];
    }

    // Copy inputs to device memory

#ifdef DEBUGGING
    // Initialize host-side buffer arrays
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));
#endif

#ifdef DEBUGGING
    // Poison outputs
    {
        DEBUG("begin poisoning");
        const int num_chunks = Tbarmax_arg <= Tbar_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            DEBUG("poisoning chunk={}/{}", chunk, num_chunks);
            const std::ptrdiff_t Tbarstride = Ebar_meta->stride[0];
            const std::ptrdiff_t Tbaroffset = chunk == 0 ? Tbarmin_arg : 0;
            const std::ptrdiff_t Tbarlength = (num_chunks == 1 ? Tbarmax_arg - Tbarmin_arg
                                               : chunk == 0    ? Tbar_ringbuf - Tbarmin_arg
                                                               : Tbarmax_arg - Tbar_ringbuf);
            const std::ptrdiff_t Fbarstride = Ebar_meta->stride[1];
            const std::ptrdiff_t Fbaroffset = 0;
            const std::ptrdiff_t Fbarlength = cuda_upchannelization_factor * (Fmax - Fmin);
            DEBUG("before cudaMemset2DAsync.Ebar");
            CHECK_CUDA_ERROR(cudaMemset2DAsync((std::uint8_t*)Ebar_memory + Tbaroffset * Tbarstride
                                                   + Fbaroffset * Fbarstride,
                                               Tbarstride, 0x88, Fbarlength * Fbarstride,
                                               Tbarlength, device.getStream(cuda_stream_id)));
        } // for chunk
        DEBUG("poisoning done.");
    }
#endif

    const std::string symname = "Upchannelizer_hirax_U8_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA Upchannelizer_hirax_U8 on GPU frame {:d}", gpu_frame_id);
    const int blocks = blocks_per_frequency * (Fmax - Fmin);
    DEBUG("More kernel arguments:");
    DEBUG("    Fmin:   {:d}", Fmin);
    DEBUG("    Fmax:   {:d}", Fmax);
    DEBUG("    blocks: {:d}", blocks);
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

    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
    DEBUG("Finished CUDA Upchannelizer_hirax_U8 on GPU frame {:d}", gpu_frame_id);

    // Check error codes
    const std::int32_t error_code = *std::max_element((const std::int32_t*)&*info_host.begin(),
                                                      (const std::int32_t*)&*info_host.end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    for (std::size_t i = 0; i < info_host.size() * blocks / max_blocks; ++i)
        if (info_host[i] != 0)
            ERROR("cudaUpchannelizer_hirax_U8 returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host[i], i);
#endif

#ifdef DEBUGGING
    // Check outputs for poison
    {
        DEBUG("begin poison check");
        DEBUG("    E_dims={}", E_meta->dims);
        DEBUG("    E_dim[0]={}", E_meta->dim[0]);
        DEBUG("    E_dim[1]={}", E_meta->dim[1]);
        DEBUG("    E_dim[2]={}", E_meta->dim[2]);
        DEBUG("    E_dim[3]={}", E_meta->dim[3]);
        DEBUG("    E_stride[0]={}", E_meta->stride[0]);
        DEBUG("    E_stride[1]={}", E_meta->stride[1]);
        DEBUG("    E_stride[2]={}", E_meta->stride[2]);
        DEBUG("    E_stride[3]={}", E_meta->stride[3]);
        DEBUG("    Ebar_dims={}", Ebar_meta->dims);
        DEBUG("    Ebar_dim[0]={}", Ebar_meta->dim[0]);
        DEBUG("    Ebar_dim[1]={}", Ebar_meta->dim[1]);
        DEBUG("    Ebar_dim[2]={}", Ebar_meta->dim[2]);
        DEBUG("    Ebar_dim[3]={}", Ebar_meta->dim[3]);
        DEBUG("    Ebar_stride[0]={}", Ebar_meta->stride[0]);
        DEBUG("    Ebar_stride[1]={}", Ebar_meta->stride[1]);
        DEBUG("    Ebar_stride[2]={}", Ebar_meta->stride[2]);
        DEBUG("    Ebar_stride[3]={}", Ebar_meta->stride[3]);
        const int num_chunks = Tbarmax_arg <= Tbar_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            DEBUG("poisoning chunk={}/{}", chunk, num_chunks);
            const std::ptrdiff_t Tbarstride = Ebar_meta->stride[0];
            const std::ptrdiff_t Tbaroffset = chunk == 0 ? Tbarmin_arg : 0;
            const std::ptrdiff_t Tbarlength = (num_chunks == 1 ? Tbarmax_arg - Tbarmin_arg
                                               : chunk == 0    ? Tbar_ringbuf - Tbarmin_arg
                                                               : Tbarmax_arg - Tbar_ringbuf);
            const std::ptrdiff_t Fbarstride = Ebar_meta->stride[1];
            const std::ptrdiff_t Fbaroffset = 0;
            const std::ptrdiff_t Fbarlength = cuda_upchannelization_factor * (Fmax - Fmin);
            DEBUG("    Tbarstride={}", Tbarstride);
            DEBUG("    Tbaroffset={}", Tbaroffset);
            DEBUG("    Tbarlength={}", Tbarlength);
            DEBUG("    Fbarstride={}", Fbarstride);
            DEBUG("    Fbaroffset={}", Fbaroffset);
            DEBUG("    Fbarlength={}", Fbarlength);
            std::vector<std::uint8_t> Ebar_buffer(Tbarlength * Fbarlength * Fbarstride, 0x11);
            DEBUG("    Ebar_buffer.size={}", Ebar_buffer.size());
            DEBUG("before cudaMemcpy2D.Ebar");
            CHECK_CUDA_ERROR(cudaMemcpy2D(Ebar_buffer.data(), Fbarlength * Fbarstride,
                                          (const std::uint8_t*)Ebar_memory + Tbaroffset * Tbarstride
                                              + Fbaroffset * Fbarstride,
                                          Tbarstride, Fbarlength * Fbarstride, Tbarlength,
                                          cudaMemcpyDeviceToHost));

            DEBUG("before memchr");
            const bool Ebar_found_error = std::memchr(Ebar_buffer.data(), 0x88, Ebar_buffer.size());
            if (Ebar_found_error) {
                for (std::ptrdiff_t tbar = 0; tbar < Tbarlength; ++tbar) {
                    for (std::ptrdiff_t fbar = 0; fbar < Fbarlength; ++fbar) {
                        bool any_error = false;
                        for (std::ptrdiff_t n = 0; n < Fbarstride; ++n) {
                            const auto val = Ebar_buffer.at(tbar * (Fbarlength * Fbarstride)
                                                            + fbar * Fbarstride + n);
                            any_error |= val == 0x88;
                        }
                        if (any_error)
                            DEBUG("    U={} [{},{}]={:#02x}", cuda_upchannelization_factor, tbar,
                                  fbar, 0x88);
                    }
                }
            }
            assert(!Ebar_found_error);
        } // for chunk
        DEBUG("poison check done.");
    }
#endif

    return record_end_event();
}

void cudaUpchannelizer_hirax_U8::finalize_frame() {
    const std::ptrdiff_t Tlength = Tmax - Tmin;
    const std::ptrdiff_t Tbarlength = Tbarmax - Tbarmin;

    // Advance the input ringbuffer
    const std::ptrdiff_t T_consumed = num_consumed_elements(Tlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples: {:d}", T_consumed);
    DEBUG("    Consumed bytes:   {:d}", T_consumed * E_T_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, instance_num, T_consumed * E_T_sample_bytes);

    // Advance the output ringbuffer
    const std::ptrdiff_t Tbar_produced = Tbarlength;
    DEBUG("Advancing output ringbuffer:");
    DEBUG("    Produced samples: {:d}", Tbar_produced);
    DEBUG("    Produced bytes:   {:d}", Tbar_produced * Ebar_Tbar_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, instance_num,
                                        Tbar_produced * Ebar_Tbar_sample_bytes);

    cudaCommand::finalize_frame();
}
