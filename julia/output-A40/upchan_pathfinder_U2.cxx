/**
 * @file
 * @brief CUDA Upchannelizer_pathfinder_U2 kernel
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
 * @class cudaUpchannelizer_pathfinder_U2
 * @brief cudaCommand for Upchannelizer_pathfinder_U2
 */
class cudaUpchannelizer_pathfinder_U2 : public cudaCommand {
public:
    cudaUpchannelizer_pathfinder_U2(Config& config, const std::string& unique_name,
                                    bufferContainer& host_buffers, cudaDeviceInterface& device,
                                    const int inst);
    virtual ~cudaUpchannelizer_pathfinder_U2();

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
    using kernel_arg = CuDeviceArray<int32_t, 1>;

    // Kernel design parameters:
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 64;
    static constexpr int cuda_number_of_frequencies = 128;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_taps = 4;
    static constexpr int cuda_max_number_of_timesamples = 8192;
    static constexpr int cuda_granularity_number_of_timesamples = 256;
    static constexpr int cuda_algorithm_overlap = 6;
    static constexpr int cuda_upchannelization_factor = 2;

    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

    // Kernel compile parameters:
    static constexpr int minthreads = 64;
    static constexpr int blocks_per_sm = 16;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 2;
    static constexpr int blocks = 128;
    static constexpr int shmem_bytes = 99328;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z6upchan13CuDeviceArrayI5Int32Li1ELi1EES_IS0_Li1ELi1EES_IS0_Li1ELi1EES_IS0_Li1ELi1EES_"
        "I9Float16x2Li1ELi1EES_I6Int4x8Li1ELi1EES_IS2_Li1ELi1EES_IS0_Li1ELi1EE";

    // Kernel arguments:
    enum class args { Tmin, Tmax, Tbarmin, Tbarmax, G, E, Ebar, info, count };

    // Tmin: Tmin
    static constexpr chordDataType Tmin_type = int32;
    enum Tmin_indices {
        Tmin_rank,
    };
    static constexpr std::array<const char*, Tmin_rank> Tmin_labels = {};
    static constexpr std::array<std::size_t, Tmin_rank> Tmin_lengths = {};
    static constexpr std::size_t Tmin_length = chord_datatype_bytes(Tmin_type);
    static_assert(Tmin_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // Tmax: Tmax
    static constexpr chordDataType Tmax_type = int32;
    enum Tmax_indices {
        Tmax_rank,
    };
    static constexpr std::array<const char*, Tmax_rank> Tmax_labels = {};
    static constexpr std::array<std::size_t, Tmax_rank> Tmax_lengths = {};
    static constexpr std::size_t Tmax_length = chord_datatype_bytes(Tmax_type);
    static_assert(Tmax_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // Tbarmin: Tbarmin
    static constexpr chordDataType Tbarmin_type = int32;
    enum Tbarmin_indices {
        Tbarmin_rank,
    };
    static constexpr std::array<const char*, Tbarmin_rank> Tbarmin_labels = {};
    static constexpr std::array<std::size_t, Tbarmin_rank> Tbarmin_lengths = {};
    static constexpr std::size_t Tbarmin_length = chord_datatype_bytes(Tbarmin_type);
    static_assert(Tbarmin_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // Tbarmax: Tbarmax
    static constexpr chordDataType Tbarmax_type = int32;
    enum Tbarmax_indices {
        Tbarmax_rank,
    };
    static constexpr std::array<const char*, Tbarmax_rank> Tbarmax_labels = {};
    static constexpr std::array<std::size_t, Tbarmax_rank> Tbarmax_lengths = {};
    static constexpr std::size_t Tbarmax_length = chord_datatype_bytes(Tbarmax_type);
    static_assert(Tbarmax_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // G: gpu_mem_gain
    static constexpr chordDataType G_type = float16;
    enum G_indices {
        G_index_Fbar,
        G_rank,
    };
    static constexpr std::array<const char*, G_rank> G_labels = {
        "Fbar",
    };
    static constexpr std::array<std::size_t, G_rank> G_lengths = {
        256,
    };
    static constexpr std::size_t G_length = chord_datatype_bytes(G_type) * 256;
    static_assert(G_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // E: gpu_mem_input_voltage
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
    static constexpr std::array<std::size_t, E_rank> E_lengths = {
        64,
        2,
        128,
        8192,
    };
    static constexpr std::size_t E_length = chord_datatype_bytes(E_type) * 64 * 2 * 128 * 8192;
    static_assert(E_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // Ebar: gpu_mem_output_voltage
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
    static constexpr std::array<std::size_t, Ebar_rank> Ebar_lengths = {
        64,
        2,
        256,
        4096,
    };
    static constexpr std::size_t Ebar_length =
        chord_datatype_bytes(Ebar_type) * 64 * 2 * 256 * 4096;
    static_assert(Ebar_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // info: gpu_mem_info
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
    static constexpr std::array<std::size_t, info_rank> info_lengths = {
        32,
        2,
        128,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 2 * 128;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //

    // Kotekan buffer names
    const std::string Tmin_memname;
    const std::string Tmax_memname;
    const std::string Tbarmin_memname;
    const std::string Tbarmax_memname;
    const std::string G_memname;
    const std::string E_memname;
    const std::string Ebar_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::uint8_t> Tmin_host;
    std::vector<std::uint8_t> Tmax_host;
    std::vector<std::uint8_t> Tbarmin_host;
    std::vector<std::uint8_t> Tbarmax_host;
    std::vector<std::uint8_t> info_host;

    static constexpr std::size_t E_T_sample_bytes = chord_datatype_bytes(E_type)
                                                    * E_lengths[E_index_D] * E_lengths[E_index_P]
                                                    * E_lengths[E_index_F];
    static constexpr std::size_t Ebar_Tbar_sample_bytes =
        chord_datatype_bytes(Ebar_type) * Ebar_lengths[Ebar_index_D] * Ebar_lengths[Ebar_index_P]
        * Ebar_lengths[Ebar_index_Fbar];

    RingBuffer* input_ringbuf_signal;
    RingBuffer* output_ringbuf_signal;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::size_t Tmin, Tmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::size_t Tbarmin, Tbarmax;
};

REGISTER_CUDA_COMMAND(cudaUpchannelizer_pathfinder_U2);

cudaUpchannelizer_pathfinder_U2::cudaUpchannelizer_pathfinder_U2(Config& config,
                                                                 const std::string& unique_name,
                                                                 bufferContainer& host_buffers,
                                                                 cudaDeviceInterface& device,
                                                                 const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "Upchannelizer_pathfinder_U2", "Upchannelizer_pathfinder_U2.ptx"),
    Tmin_memname(unique_name + "/Tmin"), Tmax_memname(unique_name + "/Tmax"),
    Tbarmin_memname(unique_name + "/Tbarmin"), Tbarmax_memname(unique_name + "/Tbarmax"),
    G_memname(config.get<std::string>(unique_name, "gpu_mem_gain")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_input_voltage")),
    Ebar_memname(config.get<std::string>(unique_name, "gpu_mem_output_voltage")),
    info_memname(unique_name + "/gpu_mem_info"),

    Tmin_host(Tmin_length), Tmax_host(Tmax_length), Tbarmin_host(Tbarmin_length),
    Tbarmax_host(Tbarmax_length), info_host(info_length),
    // Find input and output buffers used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")))),
    output_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "out_signal")))) {
    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_Tmin", false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_Tmax", false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_Tbarmin", false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_Tbarmax", false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(G_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(Ebar_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage needs to build the kernel
    if (inst == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "Upchannelizer_pathfinder_U2_");
    }

    if (inst == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
        output_ringbuf_signal->register_producer(unique_name);
        output_ringbuf_signal->allocate_new_metadata_object(0);
    }
}

cudaUpchannelizer_pathfinder_U2::~cudaUpchannelizer_pathfinder_U2() {}

std::int64_t
cudaUpchannelizer_pathfinder_U2::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) - cuda_algorithm_overlap;
}
std::int64_t
cudaUpchannelizer_pathfinder_U2::num_produced_elements(std::int64_t num_available_elements) const {
    assert(num_consumed_elements(num_available_elements) % cuda_upchannelization_factor == 0);
    return num_consumed_elements(num_available_elements) / cuda_upchannelization_factor;
}

std::int64_t
cudaUpchannelizer_pathfinder_U2::num_processed_elements(std::int64_t num_available_elements) const {
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cudaUpchannelizer_pathfinder_U2::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::size_t> val_in1 =
        input_ringbuf_signal->wait_without_claiming(unique_name);
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    if (!val_in1.has_value())
        return -1;
    const std::size_t input_bytes = val_in1.value();
    DEBUG("Input ring-buffer byte count: {:d}", input_bytes);

    // How many inputs samples are available?
    const std::size_t T_available = div_noremainder(input_bytes, E_T_sample_bytes);
    DEBUG("Available samples:      T_available: {:d}", T_available);

    // How many outputs will we process and consume?
    const std::size_t T_processed = num_processed_elements(T_available);
    const std::size_t T_consumed = num_consumed_elements(T_available);
    DEBUG("Will process (samples): T_processed: {:d}", T_processed);
    DEBUG("Will consume (samples): T_consumed:  {:d}", T_consumed);
    assert(T_processed > 0);
    assert(T_consumed <= T_processed);
    const std::size_t T_consumed2 = num_consumed_elements(T_processed);
    assert(T_consumed2 == T_consumed);

    const std::optional<std::size_t> val_in2 =
        input_ringbuf_signal->wait_and_claim_readable(unique_name, T_consumed * E_T_sample_bytes);
    if (!val_in2.has_value())
        return -1;
    const std::size_t input_cursor = val_in2.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    Tmin = div_noremainder(input_cursor, E_T_sample_bytes);
    Tmax = Tmin + T_processed;
    const std::size_t Tlength = Tmax - Tmin;
    DEBUG("Input samples:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);
    DEBUG("    Tlength: {:d}", Tlength);

    // How many outputs will we produce?
    const std::size_t Tbar_produced = num_produced_elements(T_available);
    DEBUG("Will produce (samples): Tbar_produced: {:d}", Tbar_produced);
    const std::size_t Tbarlength = Tbar_produced;

    // to bytes
    const std::size_t output_bytes = Tbarlength * Ebar_Tbar_sample_bytes;
    DEBUG("Will produce {:d} output bytes", output_bytes);

    // Wait for space to be available in our output ringbuffer...
    DEBUG("Waiting for output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::optional<std::size_t> val_out =
        output_ringbuf_signal->wait_for_writable(unique_name, output_bytes);
    DEBUG("Finished waiting for output for data frame {:d}.", gpu_frame_id);
    if (!val_out.has_value())
        return -1;
    const std::size_t output_cursor = val_out.value();
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

cudaEvent_t
cudaUpchannelizer_pathfinder_U2::execute(cudaPipelineState& /*pipestate*/,
                                         const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    void* const Tmin_memory = device.get_gpu_memory(Tmin_memname, Tmin_length);
    void* const Tmax_memory = device.get_gpu_memory(Tmax_memname, Tmax_length);
    void* const Tbarmin_memory = device.get_gpu_memory(Tbarmin_memname, Tbarmin_length);
    void* const Tbarmax_memory = device.get_gpu_memory(Tbarmax_memname, Tbarmax_length);
    void* const G_memory =
        args::G == args::E ? device.get_gpu_memory(G_memname, input_ringbuf_signal->size)
        : args::G == args::Ebar
            ? device.get_gpu_memory(G_memname, output_ringbuf_signal->size)
            : device.get_gpu_memory_array(G_memname, gpu_frame_id, _gpu_buffer_depth, G_length);
    void* const E_memory =
        args::E == args::E ? device.get_gpu_memory(E_memname, input_ringbuf_signal->size)
        : args::E == args::Ebar
            ? device.get_gpu_memory(E_memname, output_ringbuf_signal->size)
            : device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const Ebar_memory = args::Ebar == args::E
                                  ? device.get_gpu_memory(Ebar_memname, input_ringbuf_signal->size)
                              : args::Ebar == args::Ebar
                                  ? device.get_gpu_memory(Ebar_memname, output_ringbuf_signal->size)
                                  : device.get_gpu_memory_array(Ebar_memname, gpu_frame_id,
                                                                _gpu_buffer_depth, Ebar_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // G is an input buffer: check metadata
    const std::shared_ptr<metadataObject> G_mc =
        args::G == args::E ? input_ringbuf_signal->get_metadata(0)
                           : device.get_gpu_memory_array_metadata(G_memname, gpu_frame_id);
    assert(G_mc);
    assert(metadata_is_chord(G_mc));
    const std::shared_ptr<chordMetadata> G_meta = get_chord_metadata(G_mc);
    INFO("input G array: {:s} {:s}", G_meta->get_type_string(), G_meta->get_dimensions_string());
    assert(G_meta->type == G_type);
    assert(G_meta->dims == G_rank);
    for (std::size_t dim = 0; dim < G_rank; ++dim) {
        assert(std::strncmp(G_meta->dim_name[dim], G_labels[G_rank - 1 - dim],
                            sizeof G_meta->dim_name[dim])
               == 0);
        if (args::G == args::E && dim == 0)
            assert(G_meta->dim[dim] <= int(G_lengths[G_rank - 1 - dim]));
        else
            assert(G_meta->dim[dim] == int(G_lengths[G_rank - 1 - dim]));
    }
    //
    // E is an input buffer: check metadata
    const std::shared_ptr<metadataObject> E_mc =
        args::E == args::E ? input_ringbuf_signal->get_metadata(0)
                           : device.get_gpu_memory_array_metadata(E_memname, gpu_frame_id);
    assert(E_mc);
    assert(metadata_is_chord(E_mc));
    const std::shared_ptr<chordMetadata> E_meta = get_chord_metadata(E_mc);
    INFO("input E array: {:s} {:s}", E_meta->get_type_string(), E_meta->get_dimensions_string());
    assert(E_meta->type == E_type);
    assert(E_meta->dims == E_rank);
    for (std::size_t dim = 0; dim < E_rank; ++dim) {
        assert(std::strncmp(E_meta->dim_name[dim], E_labels[E_rank - 1 - dim],
                            sizeof E_meta->dim_name[dim])
               == 0);
        if (args::E == args::E && dim == 0)
            assert(E_meta->dim[dim] <= int(E_lengths[E_rank - 1 - dim]));
        else
            assert(E_meta->dim[dim] == int(E_lengths[E_rank - 1 - dim]));
    }
    //
    // Ebar is an output buffer: set metadata
    std::shared_ptr<metadataObject> const Ebar_mc =
        args::Ebar == args::Ebar ? output_ringbuf_signal->get_metadata(0)
                                 : device.create_gpu_memory_array_metadata(
                                     Ebar_memname, gpu_frame_id, E_mc->parent_pool);
    std::shared_ptr<chordMetadata> const Ebar_meta = get_chord_metadata(Ebar_mc);
    *Ebar_meta = *E_meta;
    Ebar_meta->type = Ebar_type;
    Ebar_meta->dims = Ebar_rank;
    for (std::size_t dim = 0; dim < Ebar_rank; ++dim) {
        std::strncpy(Ebar_meta->dim_name[dim], Ebar_labels[Ebar_rank - 1 - dim],
                     sizeof Ebar_meta->dim_name[dim]);
        Ebar_meta->dim[dim] = Ebar_lengths[Ebar_rank - 1 - dim];
    }
    INFO("output Ebar array: {:s} {:s}", Ebar_meta->get_type_string(),
         Ebar_meta->get_dimensions_string());
    //

    record_start_event();

    const char* exc_arg = "exception";
    kernel_arg Tmin_arg(Tmin_memory, Tmin_length);
    kernel_arg Tmax_arg(Tmax_memory, Tmax_length);
    kernel_arg Tbarmin_arg(Tbarmin_memory, Tbarmin_length);
    kernel_arg Tbarmax_arg(Tbarmax_memory, Tbarmax_length);
    kernel_arg G_arg(G_memory, G_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg Ebar_arg(Ebar_memory, Ebar_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &Tmin_arg, &Tmax_arg, &Tbarmin_arg, &Tbarmax_arg,
        &G_arg,   &E_arg,    &Ebar_arg, &info_arg,
    };

    INFO("gpu_frame_id: {}", gpu_frame_id);

    // Set E_memory to beginning of input ring buffer
    E_arg = kernel_arg(E_memory, E_length);

    // Set Ebar_memory to beginning of output ring buffer
    Ebar_arg = kernel_arg(Ebar_memory, Ebar_length);

    // Ringbuffer size
    const std::size_t T_ringbuf = input_ringbuf_signal->size / E_T_sample_bytes;
    const std::size_t Tbar_ringbuf = output_ringbuf_signal->size / Ebar_Tbar_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", T_ringbuf);
    DEBUG("Output ringbuffer size (samples): {:d}", Tbar_ringbuf);

    const std::size_t Tlength = Tmax - Tmin;
    const std::size_t Tbarlength = Tbarmax - Tbarmin;
    DEBUG("Processed input samples: {:d}", Tlength);
    DEBUG("Produced output samples: {:d}", Tbarlength);

    DEBUG("Kernel arguments:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);
    DEBUG("    Tbarmin: {:d}", Tbarmin);
    DEBUG("    Tbarmax: {:d}", Tbarmax);

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    *(std::int32_t*)Tmin_host.data() = mod(Tmin, T_ringbuf);
    *(std::int32_t*)Tmax_host.data() = mod(Tmin, T_ringbuf) + Tlength;
    *(std::int32_t*)Tbarmin_host.data() = mod(Tbarmin, Tbar_ringbuf);
    *(std::int32_t*)Tbarmax_host.data() = mod(Tbarmin, Tbar_ringbuf) + Tbarlength;

    // Update metadata
    Ebar_meta->dim[0] = Tbarlength;
    assert(Ebar_meta->dim[0] <= int(Ebar_lengths[3]));

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
    // TODO: Pass scalar kernel arguments more efficiently, i.e. without a separate `cudaMemcpy`
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tmin_memory, Tmin_host.data(), Tmin_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tmax_memory, Tmax_host.data(), Tmax_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tbarmin_memory, Tbarmin_host.data(), Tbarmin_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tbarmax_memory, Tbarmax_host.data(), Tbarmax_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));

    // Initialize host-side buffer arrays
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));

    // // Initialize outputs
    // //     0x88 = (-8,-8), an unused value to detect uninitialized output
    // // TODO: Skip this for performance
    // CHECK_CUDA_ERROR(cudaMemsetAsync(Ebar_memory, 0x88, Ebar_length,
    // device.getStream(cuda_stream_id)));

    const std::string symname = "Upchannelizer_pathfinder_U2_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA Upchannelizer_pathfinder_U2 on GPU frame {:d}", gpu_frame_id);
    const CUresult err =
        cuLaunchKernel(device.runtime_kernels[symname], blocks, 1, 1, threads_x, threads_y, 1,
                       shmem_bytes, device.getStream(cuda_stream_id), args, NULL);

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        ERROR("cuLaunchKernel: Error number: {}: {}", (int)err, errStr);
    }

    // Copy results back to host memory
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(cudaMemcpyAsync(info_host.data(), info_memory, info_length,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

    // Check error codes
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
    const std::int32_t error_code = *std::max_element((const std::int32_t*)&*info_host.begin(),
                                                      (const std::int32_t*)&*info_host.end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    for (std::size_t i = 0; i < info_host.size(); ++i)
        if (info_host[i] != 0)
            ERROR("cudaUpchannelizer_pathfinder_U2 returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host[i], i);

    return record_end_event();
}

void cudaUpchannelizer_pathfinder_U2::finalize_frame() {
    const std::size_t Tlength = Tmax - Tmin;
    const std::size_t Tbarlength = Tbarmax - Tbarmin;

    // Advance the input ringbuffer
    const std::size_t T_consumed = num_consumed_elements(Tlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples: {:d}", T_consumed);
    DEBUG("    Consumed bytes:   {:d}", T_consumed * E_T_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, T_consumed * E_T_sample_bytes);

    // Advance the output ringbuffer
    const std::size_t Tbar_produced = Tbarlength;
    DEBUG("Advancing output ringbuffer:");
    DEBUG("    Produced samples: {:d}", Tbar_produced);
    DEBUG("    Produced bytes:   {:d}", Tbar_produced * Ebar_Tbar_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, Tbar_produced * Ebar_Tbar_sample_bytes);

    cudaCommand::finalize_frame();
}
