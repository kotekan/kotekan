/**
 * @file
 * @brief CUDA TransposeKernel_chime kernel
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
 * @class cudaTransposeKernel_chime
 * @brief cudaCommand for TransposeKernel_chime
 */
class cudaTransposeKernel_chime : public cudaCommand {
public:
    cudaTransposeKernel_chime(Config& config, const std::string& unique_name,
                              bufferContainer& host_buffers, cudaDeviceInterface& device,
                              const int instance_num);
    virtual ~cudaTransposeKernel_chime();

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
    using array_desc = CuDeviceArray<int32_t, 1>;

    // Kernel design parameters:
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 1024;
    static constexpr int cuda_number_of_frequencies = 16;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_max_number_of_timesamples = 65536;
    static constexpr int cuda_granularity_number_of_timesamples = 32;

    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

    // Kernel compile parameters:
    static constexpr int minthreads = 512;
    static constexpr int blocks_per_sm = 1;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 16;
    static constexpr int blocks = 16;
    static constexpr int shmem_bytes = 65536;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z16xpose2048_kernel5Int32S_S_S_13CuDeviceArrayI6Int4x8Li1ELi1EES0_IS1_Li1ELi1EES0_IS_"
        "Li1ELi1EES0_IS_Li1ELi1EE";

    // Kernel arguments:
    enum class args { Tinmin, Tinmax, Tmin, Tmax, Ein, E, scatter_indices, info, count };

    // Tinmin: Tinmin
    static constexpr const char* Tinmin_name = "Tinmin";
    static constexpr chordDataType Tinmin_type = int32;
    //
    // Tinmax: Tinmax
    static constexpr const char* Tinmax_name = "Tinmax";
    static constexpr chordDataType Tinmax_type = int32;
    //
    // Tmin: Tmin
    static constexpr const char* Tmin_name = "Tmin";
    static constexpr chordDataType Tmin_type = int32;
    //
    // Tmax: Tmax
    static constexpr const char* Tmax_name = "Tmax";
    static constexpr chordDataType Tmax_type = int32;
    //
    // Ein: gpu_mem_voltage
    static constexpr const char* Ein_name = "Ein";
    static constexpr chordDataType Ein_type = int4p4chime;
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
    static constexpr std::ptrdiff_t Ein_length =
        chord_datatype_bytes(Ein_type) * 1024 * 2 * 16 * 65536;
    static_assert(Ein_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    // static_assert(Ein_length == chord_datatype_bytes(Ein_type) * Ein_strides[Ein_rank]);
    //
    // E: gpu_mem_voltage
    static constexpr const char* E_name = "E";
    static constexpr chordDataType E_type = int4p4chime;
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
    // static_assert(E_length == chord_datatype_bytes(E_type) * E_strides[E_rank]);
    //
    // scatter_indices: scatter_indices
    static constexpr const char* scatter_indices_name = "scatter_indices";
    static constexpr chordDataType scatter_indices_type = int32;
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
    static constexpr std::ptrdiff_t scatter_indices_length =
        chord_datatype_bytes(scatter_indices_type) * 1024 * 2;
    static_assert(scatter_indices_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    // static_assert(scatter_indices_length == chord_datatype_bytes(scatter_indices_type) *
    // scatter_indices_strides[scatter_indices_rank]);
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
        16,
        16,
    };
    static constexpr std::ptrdiff_t info_length = chord_datatype_bytes(info_type) * 32 * 16 * 16;
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
    // static_assert(info_length == chord_datatype_bytes(info_type) * info_strides[info_rank]);
    //

    // Kotekan buffer names
    const std::string Ein_memname;
    const std::string E_memname;
    const std::string scatter_indices_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::uint8_t> info_host;

    static constexpr std::ptrdiff_t Ein_T_sample_bytes =
        chord_datatype_bytes(Ein_type) * Ein_lengths[Ein_index_D] * Ein_lengths[Ein_index_P]
        * Ein_lengths[Ein_index_F];
    static constexpr std::ptrdiff_t E_T_sample_bytes = chord_datatype_bytes(E_type)
                                                       * E_lengths[E_index_D] * E_lengths[E_index_P]
                                                       * E_lengths[E_index_F];

    RingBuffer* input_ringbuf_signal;
    RingBuffer* output_ringbuf_signal;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tinmin, Tinmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tmin, Tmax;
};

REGISTER_CUDA_COMMAND(cudaTransposeKernel_chime);

cudaTransposeKernel_chime::cudaTransposeKernel_chime(Config& config, const std::string& unique_name,
                                                     bufferContainer& host_buffers,
                                                     cudaDeviceInterface& device,
                                                     const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "TransposeKernel_chime", "TransposeKernel_chime.ptx"),
    Ein_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    scatter_indices_memname(config.get<std::string>(unique_name, "scatter_indices")),
    info_memname(unique_name + "/gpu_mem_info"),

    info_host(info_length),
    // Find input and output buffers used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")))),
    output_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "out_signal")))) {
    // Check ringbuffer sizes
    assert(input_ringbuf_signal->size == Ein_length);
    assert(output_ringbuf_signal->size == E_length);

    // Register host memory
    {
        const cudaError_t ierr = cudaHostRegister(info_host.data(), info_host.size(), 0);
        assert(ierr == cudaSuccess);
    }

    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(Ein_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(scatter_indices_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage need to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "TransposeKernel_chime_");
    }

    if (instance_num == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
        output_ringbuf_signal->register_producer(unique_name);
        output_ringbuf_signal->allocate_new_metadata_object(0);
    }
}

cudaTransposeKernel_chime::~cudaTransposeKernel_chime() {}

std::int64_t
cudaTransposeKernel_chime::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements);
}
std::int64_t
cudaTransposeKernel_chime::num_produced_elements(std::int64_t num_available_elements) const {
    return num_consumed_elements(num_available_elements);
}

std::int64_t
cudaTransposeKernel_chime::num_processed_elements(std::int64_t num_available_elements) const {
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cudaTransposeKernel_chime::wait_on_precondition() {
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
    const std::ptrdiff_t Tin_available = div_noremainder(input_bytes, Ein_T_sample_bytes);
    DEBUG("Available samples:      Tin_available: {:d}", Tin_available);

    // How many outputs will we process and consume?
    const std::ptrdiff_t Tin_processed = num_processed_elements(Tin_available);
    const std::ptrdiff_t Tin_consumed = num_consumed_elements(Tin_available);
    DEBUG("Will process (samples): Tin_processed: {:d}", Tin_processed);
    DEBUG("Will consume (samples): Tin_consumed:  {:d}", Tin_consumed);
    assert(Tin_processed > 0);
    assert(Tin_consumed <= Tin_processed);
    const std::ptrdiff_t Tin_consumed2 = num_consumed_elements(Tin_processed);
    assert(Tin_consumed2 == Tin_consumed);

    const std::optional<std::ptrdiff_t> val_in2 = input_ringbuf_signal->wait_and_claim_readable(
        unique_name, instance_num, Tin_consumed * Ein_T_sample_bytes);
    if (!val_in2.has_value())
        return -1;
    const std::ptrdiff_t input_cursor = val_in2.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    Tinmin = div_noremainder(input_cursor, Ein_T_sample_bytes);
    Tinmax = Tinmin + Tin_processed;
    const std::ptrdiff_t Tinlength = Tinmax - Tinmin;
    DEBUG("Input samples:");
    DEBUG("    Tinmin:    {:d}", Tinmin);
    DEBUG("    Tinmax:    {:d}", Tinmax);
    DEBUG("    Tinlength: {:d}", Tinlength);

    // How many outputs will we produce?
    const std::ptrdiff_t T_produced = num_produced_elements(Tin_available);
    DEBUG("Will produce (samples): T_produced: {:d}", T_produced);
    const std::ptrdiff_t Tlength = T_produced;

    // to bytes
    const std::ptrdiff_t output_bytes = Tlength * E_T_sample_bytes;
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

    assert(mod(output_cursor, E_T_sample_bytes) == 0);
    Tmin = output_cursor / E_T_sample_bytes;
    Tmax = Tmin + Tlength;
    DEBUG("Output samples:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);
    DEBUG("    Tlength: {:d}", Tlength);

    return 0;
}

cudaEvent_t cudaTransposeKernel_chime::execute(cudaPipelineState& /*pipestate*/,
                                               const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    void* const Ein_memory =
        args::Ein == args::Ein ? device.get_gpu_memory(Ein_memname, input_ringbuf_signal->size)
        : args::Ein == args::E ? device.get_gpu_memory(Ein_memname, output_ringbuf_signal->size)
        : args::Ein == args::scatter_indices
            ? device.get_gpu_memory(Ein_memname, Ein_length)
            : device.get_gpu_memory_array(Ein_memname, gpu_frame_id, _gpu_buffer_depth, Ein_length);
    void* const E_memory =
        args::E == args::Ein ? device.get_gpu_memory(E_memname, input_ringbuf_signal->size)
        : args::E == args::E ? device.get_gpu_memory(E_memname, output_ringbuf_signal->size)
        : args::E == args::scatter_indices
            ? device.get_gpu_memory(E_memname, E_length)
            : device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const scatter_indices_memory =
        args::scatter_indices == args::Ein
            ? device.get_gpu_memory(scatter_indices_memname, input_ringbuf_signal->size)
        : args::scatter_indices == args::E
            ? device.get_gpu_memory(scatter_indices_memname, output_ringbuf_signal->size)
        : args::scatter_indices == args::scatter_indices
            ? device.get_gpu_memory(scatter_indices_memname, scatter_indices_length)
            : device.get_gpu_memory_array(scatter_indices_memname, gpu_frame_id, _gpu_buffer_depth,
                                          scatter_indices_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // Ein is an input buffer: check metadata
    const std::shared_ptr<metadataObject> Ein_mc =
        args::Ein == args::Ein ? input_ringbuf_signal->get_metadata(0)
                               : device.get_gpu_memory_array_metadata(Ein_memname, gpu_frame_id);
    assert(Ein_mc);
    assert(metadata_is_chord(Ein_mc));
    const std::shared_ptr<chordMetadata> Ein_meta = get_chord_metadata(Ein_mc);
    DEBUG("input Ein array: {:s} {:s}", Ein_meta->get_type_string(),
          Ein_meta->get_dimensions_string());
    assert(std::strncmp(Ein_meta->name, Ein_name, sizeof Ein_meta->name) == 0);
    assert(Ein_meta->type == Ein_type);
    assert(Ein_meta->dims == Ein_rank);
    for (std::ptrdiff_t dim = 0; dim < Ein_rank; ++dim) {
        assert(std::strncmp(Ein_meta->dim_name[Ein_rank - 1 - dim], Ein_labels[dim],
                            sizeof Ein_meta->dim_name[Ein_rank - 1 - dim])
               == 0);
        if (args::Ein == args::E && dim == E_index_T) {
            assert(Ein_meta->dim[Ein_rank - 1 - dim] <= int(Ein_lengths[dim]));
            assert(Ein_meta->stride[Ein_rank - 1 - dim] <= Ein_strides[dim]);
        } else {
            assert(Ein_meta->dim[Ein_rank - 1 - dim] == int(Ein_lengths[dim]));
            assert(Ein_meta->stride[Ein_rank - 1 - dim] == Ein_strides[dim]);
        }
    }
    //
    // E is an output buffer: set metadata
    std::shared_ptr<metadataObject> const E_mc =
        args::E == args::E
            ? output_ringbuf_signal->get_metadata(0)
            : device.create_gpu_memory_array_metadata(E_memname, gpu_frame_id, E_mc->parent_pool);
    std::shared_ptr<chordMetadata> const E_meta = get_chord_metadata(E_mc);
    *E_meta = *Ein_meta;
    std::strncpy(E_meta->name, E_name, sizeof E_meta->name);
    E_meta->type = E_type;
    E_meta->dims = E_rank;
    for (std::ptrdiff_t dim = 0; dim < E_rank; ++dim) {
        std::strncpy(E_meta->dim_name[E_rank - 1 - dim], E_labels[dim],
                     sizeof E_meta->dim_name[E_rank - 1 - dim]);
        E_meta->dim[E_rank - 1 - dim] = E_lengths[dim];
        E_meta->stride[E_rank - 1 - dim] = E_strides[dim];
    }
    DEBUG("output E array: {:s} {:s}", E_meta->get_type_string(), E_meta->get_dimensions_string());
    //
    // scatter_indices is an input buffer: check metadata
    const std::shared_ptr<metadataObject> scatter_indices_mc =
        args::scatter_indices == args::Ein
            ? input_ringbuf_signal->get_metadata(0)
            : device.get_gpu_memory_array_metadata(scatter_indices_memname, gpu_frame_id);
    assert(scatter_indices_mc);
    assert(metadata_is_chord(scatter_indices_mc));
    const std::shared_ptr<chordMetadata> scatter_indices_meta =
        get_chord_metadata(scatter_indices_mc);
    DEBUG("input scatter_indices array: {:s} {:s}", scatter_indices_meta->get_type_string(),
          scatter_indices_meta->get_dimensions_string());
    assert(std::strncmp(scatter_indices_meta->name, scatter_indices_name,
                        sizeof scatter_indices_meta->name)
           == 0);
    assert(scatter_indices_meta->type == scatter_indices_type);
    assert(scatter_indices_meta->dims == scatter_indices_rank);
    for (std::ptrdiff_t dim = 0; dim < scatter_indices_rank; ++dim) {
        assert(std::strncmp(scatter_indices_meta->dim_name[scatter_indices_rank - 1 - dim],
                            scatter_indices_labels[dim],
                            sizeof scatter_indices_meta->dim_name[scatter_indices_rank - 1 - dim])
               == 0);
        if (args::scatter_indices == args::E && dim == E_index_T) {
            assert(scatter_indices_meta->dim[scatter_indices_rank - 1 - dim]
                   <= int(scatter_indices_lengths[dim]));
            assert(scatter_indices_meta->stride[scatter_indices_rank - 1 - dim]
                   <= scatter_indices_strides[dim]);
        } else {
            assert(scatter_indices_meta->dim[scatter_indices_rank - 1 - dim]
                   == int(scatter_indices_lengths[dim]));
            assert(scatter_indices_meta->stride[scatter_indices_rank - 1 - dim]
                   == scatter_indices_strides[dim]);
        }
    }
    //

    record_start_event();

    DEBUG("gpu_frame_id: {}", gpu_frame_id);

    const char* exc_arg = "exception";
    std::int32_t Tinmin_arg;
    std::int32_t Tinmax_arg;
    std::int32_t Tmin_arg;
    std::int32_t Tmax_arg;
    array_desc Ein_arg(Ein_memory, Ein_length);
    array_desc E_arg(E_memory, E_length);
    array_desc scatter_indices_arg(scatter_indices_memory, scatter_indices_length);
    array_desc info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &Tinmin_arg, &Tinmax_arg,          &Tmin_arg, &Tmax_arg,
        &Ein_arg, &E_arg,      &scatter_indices_arg, &info_arg,
    };

    // Set Ein_memory to beginning of input ring buffer
    Ein_arg = array_desc(Ein_memory, Ein_length);

    // Set E_memory to beginning of output ring buffer
    E_arg = array_desc(E_memory, E_length);

    // Ringbuffer size
    const std::ptrdiff_t Tin_ringbuf = input_ringbuf_signal->size / Ein_T_sample_bytes;
    const std::ptrdiff_t T_ringbuf = output_ringbuf_signal->size / E_T_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", Tin_ringbuf);
    DEBUG("Output ringbuffer size (samples): {:d}", T_ringbuf);

    const std::ptrdiff_t Tinlength = Tinmax - Tinmin;
    const std::ptrdiff_t Tlength = Tmax - Tmin;
    DEBUG("Processed input samples: {:d}", Tinlength);
    DEBUG("Produced output samples: {:d}", Tlength);

    DEBUG("Kernel arguments:");
    DEBUG("    Tinmin: {:d}", Tinmin);
    DEBUG("    Tinmax: {:d}", Tinmax);
    DEBUG("    Tmin:   {:d}", Tmin);
    DEBUG("    Tmax:   {:d}", Tmax);

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    Tinmin_arg = mod(Tinmin, Tin_ringbuf);
    Tinmax_arg = mod(Tinmin, Tin_ringbuf) + Tinlength;
    Tmin_arg = mod(Tmin, T_ringbuf);
    Tmax_arg = mod(Tmin, T_ringbuf) + Tlength;

    // Update metadata
    E_meta->dim[E_rank - 1 - E_index_T] = Tlength;
    assert(E_meta->dim[E_rank - 1 - E_index_T] <= int(E_lengths[E_index_T]));
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

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
        const int num_chunks = Tmax_arg <= T_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            DEBUG("poisoning chunk={}/{}", chunk, num_chunks);
            const std::ptrdiff_t Tstride = E_meta->stride[0];
            const std::ptrdiff_t Toffset = chunk == 0 ? Tmin_arg : 0;
            const std::ptrdiff_t Tlength = (num_chunks == 1 ? Tmax_arg - Tmin_arg
                                            : chunk == 0    ? T_ringbuf - Tmin_arg
                                                            : Tmax_arg - T_ringbuf);
            DEBUG("before cudaMemset2DAsync.E");
            CHECK_CUDA_ERROR(cudaMemsetAsync((std::uint8_t*)E_memory + Toffset * Tstride, 0x88,
                                             Tlength * Tstride, device.getStream(cuda_stream_id)));
        } // for chunk
        DEBUG("poisoning done.");
    }
#endif

    const std::string symname = "TransposeKernel_chime_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA TransposeKernel_chime on GPU frame {:d}", gpu_frame_id);
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
    DEBUG("Finished CUDA TransposeKernel_chime on GPU frame {:d}", gpu_frame_id);

    // Check error codes
    const std::int32_t error_code = *std::max_element((const std::int32_t*)&*info_host.begin(),
                                                      (const std::int32_t*)&*info_host.end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    for (std::size_t i = 0; i < info_host.size(); ++i)
        if (info_host[i] != 0)
            ERROR("cudaTransposeKernel_chime returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host[i], i);
#endif

#ifdef DEBUGGING
    // Check outputs for poison
    {
        DEBUG("begin poison check");
        DEBUG("    Ein_dims={}", Ein_meta->dims);
        DEBUG("    Ein_dim[0]={}", Ein_meta->dim[0]);
        DEBUG("    Ein_dim[1]={}", Ein_meta->dim[1]);
        DEBUG("    Ein_dim[2]={}", Ein_meta->dim[2]);
        DEBUG("    Ein_dim[3]={}", Ein_meta->dim[3]);
        DEBUG("    Ein_stride[0]={}", Ein_meta->stride[0]);
        DEBUG("    Ein_stride[1]={}", Ein_meta->stride[1]);
        DEBUG("    Ein_stride[2]={}", Ein_meta->stride[2]);
        DEBUG("    Ein_stride[3]={}", Ein_meta->stride[3]);
        DEBUG("    E_dims={}", E_meta->dims);
        DEBUG("    E_dim[0]={}", E_meta->dim[0]);
        DEBUG("    E_dim[1]={}", E_meta->dim[1]);
        DEBUG("    E_dim[2]={}", E_meta->dim[2]);
        DEBUG("    E_dim[3]={}", E_meta->dim[3]);
        DEBUG("    E_stride[0]={}", E_meta->stride[0]);
        DEBUG("    E_stride[1]={}", E_meta->stride[1]);
        DEBUG("    E_stride[2]={}", E_meta->stride[2]);
        DEBUG("    E_stride[3]={}", E_meta->stride[3]);
        const int num_chunks = Tmax_arg <= T_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            DEBUG("poisoning chunk={}/{}", chunk, num_chunks);
            const std::ptrdiff_t Tstride = E_meta->stride[0];
            const std::ptrdiff_t Toffset = chunk == 0 ? Tmin_arg : 0;
            const std::ptrdiff_t Tlength = (num_chunks == 1 ? Tmax_arg - Tmin_arg
                                            : chunk == 0    ? T_ringbuf - Tmin_arg
                                                            : Tmax_arg - T_ringbuf);
            DEBUG("    Tstride={}", Tstride);
            DEBUG("    Toffset={}", Toffset);
            DEBUG("    Tlength={}", Tlength);
            std::vector<std::uint8_t> E_buffer(Tlength * Tstride, 0x11);
            DEBUG("    E_buffer.size={}", E_buffer.size());
            DEBUG("before cudaMemcpy.E");
            CHECK_CUDA_ERROR(cudaMemcpy(E_buffer.data(),
                                        (const std::uint8_t*)E_memory + Toffset * Tstride,
                                        Tlength * Tstride, cudaMemcpyDeviceToHost));

            DEBUG("before memchr");
            const bool E_found_error = std::memchr(E_buffer.data(), 0x88, E_buffer.size());
            if (E_found_error) {
                for (std::ptrdiff_t t = 0; t < Tlength; ++t) {
                    bool any_error = false;
                    for (std::ptrdiff_t n = 0; n < Tstride; ++n) {
                        const auto val = E_buffer.at(t * Tstride + n);
                        any_error |= val == 0x88;
                    }
                    if (any_error)
                        DEBUG("    [{}]={:#02x}", t, 0x88);
                }
            }
            assert(!E_found_error);
        } // for chunk
        DEBUG("poison check done.");
    }
#endif

    return record_end_event();
}

void cudaTransposeKernel_chime::finalize_frame() {
    const std::ptrdiff_t Tinlength = Tinmax - Tinmin;
    const std::ptrdiff_t Tlength = Tmax - Tmin;

    // Advance the input ringbuffer
    const std::ptrdiff_t Tin_consumed = num_consumed_elements(Tinlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples: {:d}", Tin_consumed);
    DEBUG("    Consumed bytes:   {:d}", Tin_consumed * Ein_T_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, instance_num, Tin_consumed * Ein_T_sample_bytes);

    // Advance the output ringbuffer
    const std::ptrdiff_t T_produced = Tlength;
    DEBUG("Advancing output ringbuffer:");
    DEBUG("    Produced samples: {:d}", T_produced);
    DEBUG("    Produced bytes:   {:d}", T_produced * E_T_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, instance_num, T_produced * E_T_sample_bytes);

    cudaCommand::finalize_frame();
}
