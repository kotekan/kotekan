/**
 * @file
 * @brief CUDA Upchannelizer_chord_U32 kernel
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
 * @class cudaUpchannelizer_chord_U32
 * @brief cudaCommand for Upchannelizer_chord_U32
 */
class cudaUpchannelizer_chord_U32 : public cudaCommand {
public:
    cudaUpchannelizer_chord_U32(Config& config, const std::string& unique_name,
                                bufferContainer& host_buffers, cudaDeviceInterface& device,
                                const int instance_num);
    virtual ~cudaUpchannelizer_chord_U32();

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
    static constexpr int cuda_number_of_dishes = 512;
    static constexpr int cuda_number_of_frequencies = 48;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_taps = 4;
    static constexpr int cuda_max_number_of_timesamples = 32768;
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

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 16;
    static constexpr int blocks = 384;
    static constexpr int shmem_bytes = 67840;

    // Kernel name:
    const char* const kernel_symbol = "_Z6upchan5Int32S_S_S_13CuDeviceArrayI9Float16x2Li1ELi1EES0_"
                                      "I6Int4x8Li1ELi1EES0_IS2_Li1ELi1EES0_IS_Li1ELi1EE";

    // Kernel arguments:
    enum class args { Tmin, Tmax, Tbarmin, Tbarmax, G32, E, Ebar32, info, count };

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
    // G32: gpu_mem_gain
    static constexpr const char* G32_name = "G32";
    static constexpr chordDataType G32_type = float16;
    enum G32_indices {
        G32_index_Fbar,
        G32_rank,
    };
    static constexpr std::array<const char*, G32_rank> G32_labels = {
        "Fbar",
    };
    static constexpr std::array<std::ptrdiff_t, G32_rank> G32_lengths = {
        1536,
    };
    static constexpr std::ptrdiff_t G32_length = chord_datatype_bytes(G32_type) * 1536;
    static_assert(G32_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
        512,
        2,
        48,
        32768,
    };
    static constexpr std::ptrdiff_t E_length = chord_datatype_bytes(E_type) * 512 * 2 * 48 * 32768;
    static_assert(E_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    //
    // Ebar32: gpu_mem_output_voltage
    static constexpr const char* Ebar32_name = "Ebar32";
    static constexpr chordDataType Ebar32_type = int4p4;
    enum Ebar32_indices {
        Ebar32_index_D,
        Ebar32_index_P,
        Ebar32_index_Fbar,
        Ebar32_index_Tbar,
        Ebar32_rank,
    };
    static constexpr std::array<const char*, Ebar32_rank> Ebar32_labels = {
        "D",
        "P",
        "Fbar",
        "Tbar",
    };
    static constexpr std::array<std::ptrdiff_t, Ebar32_rank> Ebar32_lengths = {
        512,
        2,
        1536,
        1024,
    };
    static constexpr std::ptrdiff_t Ebar32_length =
        chord_datatype_bytes(Ebar32_type) * 512 * 2 * 1536 * 1024;
    static_assert(Ebar32_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
        384,
    };
    static constexpr std::ptrdiff_t info_length = chord_datatype_bytes(info_type) * 32 * 16 * 384;
    static_assert(info_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    //

    // Kotekan buffer names
    const std::string G32_memname;
    const std::string E_memname;
    const std::string Ebar32_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::uint8_t> info_host;

    static constexpr std::ptrdiff_t E_T_sample_bytes = chord_datatype_bytes(E_type)
                                                       * E_lengths[E_index_D] * E_lengths[E_index_P]
                                                       * E_lengths[E_index_F];
    static constexpr std::ptrdiff_t Ebar32_Tbar_sample_bytes =
        chord_datatype_bytes(Ebar32_type) * Ebar32_lengths[Ebar32_index_D]
        * Ebar32_lengths[Ebar32_index_P] * Ebar32_lengths[Ebar32_index_Fbar];

    RingBuffer* input_ringbuf_signal;
    RingBuffer* output_ringbuf_signal;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tmin, Tmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tbarmin, Tbarmax;
};

REGISTER_CUDA_COMMAND(cudaUpchannelizer_chord_U32);

cudaUpchannelizer_chord_U32::cudaUpchannelizer_chord_U32(Config& config,
                                                         const std::string& unique_name,
                                                         bufferContainer& host_buffers,
                                                         cudaDeviceInterface& device,
                                                         const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "Upchannelizer_chord_U32", "Upchannelizer_chord_U32.ptx"),
    G32_memname(config.get<std::string>(unique_name, "gpu_mem_gain")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_input_voltage")),
    Ebar32_memname(config.get<std::string>(unique_name, "gpu_mem_output_voltage")),
    info_memname(unique_name + "/gpu_mem_info"),

    info_host(info_length),
    // Find input and output buffers used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")))),
    output_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "out_signal")))) {
    // Check ringbuffer sizes
    assert(input_ringbuf_signal->size == E_length);
    assert(output_ringbuf_signal->size == Ebar32_length);

    // Register host memory
    {
        const cudaError_t ierr = cudaHostRegister(info_host.data(), info_host.size(), 0);
        assert(ierr == cudaSuccess);
    }

    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(G32_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(Ebar32_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage needs to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "Upchannelizer_chord_U32_");
    }

    if (instance_num == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
        output_ringbuf_signal->register_producer(unique_name);
        output_ringbuf_signal->allocate_new_metadata_object(0);
    }
}

cudaUpchannelizer_chord_U32::~cudaUpchannelizer_chord_U32() {}

std::int64_t
cudaUpchannelizer_chord_U32::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) - cuda_algorithm_overlap;
}
std::int64_t
cudaUpchannelizer_chord_U32::num_produced_elements(std::int64_t num_available_elements) const {
    assert(num_consumed_elements(num_available_elements) % cuda_upchannelization_factor == 0);
    return num_consumed_elements(num_available_elements) / cuda_upchannelization_factor;
}

std::int64_t
cudaUpchannelizer_chord_U32::num_processed_elements(std::int64_t num_available_elements) const {
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cudaUpchannelizer_chord_U32::wait_on_precondition() {
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
    const std::ptrdiff_t output_bytes = Tbarlength * Ebar32_Tbar_sample_bytes;
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

    assert(mod(output_cursor, Ebar32_Tbar_sample_bytes) == 0);
    Tbarmin = output_cursor / Ebar32_Tbar_sample_bytes;
    Tbarmax = Tbarmin + Tbarlength;
    DEBUG("Output samples:");
    DEBUG("    Tbarmin:    {:d}", Tbarmin);
    DEBUG("    Tbarmax:    {:d}", Tbarmax);
    DEBUG("    Tbarlength: {:d}", Tbarlength);

    return 0;
}

cudaEvent_t cudaUpchannelizer_chord_U32::execute(cudaPipelineState& /*pipestate*/,
                                                 const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    void* const G32_memory =
        args::G32 == args::E ? device.get_gpu_memory(G32_memname, input_ringbuf_signal->size)
        : args::G32 == args::Ebar32
            ? device.get_gpu_memory(G32_memname, output_ringbuf_signal->size)
        : args::G32 == args::G32
            ? device.get_gpu_memory(G32_memname, G32_length)
            : device.get_gpu_memory_array(G32_memname, gpu_frame_id, _gpu_buffer_depth, G32_length);
    void* const E_memory =
        args::E == args::E        ? device.get_gpu_memory(E_memname, input_ringbuf_signal->size)
        : args::E == args::Ebar32 ? device.get_gpu_memory(E_memname, output_ringbuf_signal->size)
        : args::E == args::G32
            ? device.get_gpu_memory(E_memname, E_length)
            : device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const Ebar32_memory =
        args::Ebar32 == args::E ? device.get_gpu_memory(Ebar32_memname, input_ringbuf_signal->size)
        : args::Ebar32 == args::Ebar32
            ? device.get_gpu_memory(Ebar32_memname, output_ringbuf_signal->size)
        : args::Ebar32 == args::G32 ? device.get_gpu_memory(Ebar32_memname, Ebar32_length)
                                    : device.get_gpu_memory_array(Ebar32_memname, gpu_frame_id,
                                                                  _gpu_buffer_depth, Ebar32_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // G32 is an input buffer: check metadata
    const std::shared_ptr<metadataObject> G32_mc =
        args::G32 == args::E ? input_ringbuf_signal->get_metadata(0)
                             : device.get_gpu_memory_array_metadata(G32_memname, gpu_frame_id);
    assert(G32_mc);
    assert(metadata_is_chord(G32_mc));
    const std::shared_ptr<chordMetadata> G32_meta = get_chord_metadata(G32_mc);
    DEBUG("input G32 array: {:s} {:s}", G32_meta->get_type_string(),
          G32_meta->get_dimensions_string());
    assert(std::strncmp(G32_meta->name, G32_name, sizeof G32_meta->name) == 0);
    assert(G32_meta->type == G32_type);
    assert(G32_meta->dims == G32_rank);
    for (std::ptrdiff_t dim = 0; dim < G32_rank; ++dim) {
        assert(std::strncmp(G32_meta->dim_name[G32_rank - 1 - dim], G32_labels[dim],
                            sizeof G32_meta->dim_name[G32_rank - 1 - dim])
               == 0);
        if (args::G32 == args::E && dim == E_index_T)
            assert(G32_meta->dim[G32_rank - 1 - dim] <= int(G32_lengths[dim]));
        else
            assert(G32_meta->dim[G32_rank - 1 - dim] == int(G32_lengths[dim]));
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
        if (args::E == args::E && dim == E_index_T)
            assert(E_meta->dim[E_rank - 1 - dim] <= int(E_lengths[dim]));
        else
            assert(E_meta->dim[E_rank - 1 - dim] == int(E_lengths[dim]));
    }
    //
    // Ebar32 is an output buffer: set metadata
    std::shared_ptr<metadataObject> const Ebar32_mc =
        args::Ebar32 == args::Ebar32 ? output_ringbuf_signal->get_metadata(0)
                                     : device.create_gpu_memory_array_metadata(
                                         Ebar32_memname, gpu_frame_id, E_mc->parent_pool);
    std::shared_ptr<chordMetadata> const Ebar32_meta = get_chord_metadata(Ebar32_mc);
    *Ebar32_meta = *E_meta;
    std::strncpy(Ebar32_meta->name, Ebar32_name, sizeof Ebar32_meta->name);
    Ebar32_meta->type = Ebar32_type;
    Ebar32_meta->dims = Ebar32_rank;
    for (std::ptrdiff_t dim = 0; dim < Ebar32_rank; ++dim) {
        std::strncpy(Ebar32_meta->dim_name[Ebar32_rank - 1 - dim], Ebar32_labels[dim],
                     sizeof Ebar32_meta->dim_name[Ebar32_rank - 1 - dim]);
        Ebar32_meta->dim[Ebar32_rank - 1 - dim] = Ebar32_lengths[dim];
    }
    DEBUG("output Ebar32 array: {:s} {:s}", Ebar32_meta->get_type_string(),
          Ebar32_meta->get_dimensions_string());
    //

    record_start_event();

    DEBUG("gpu_frame_id: {}", gpu_frame_id);

    const char* exc_arg = "exception";
    std::int32_t Tmin_arg;
    std::int32_t Tmax_arg;
    std::int32_t Tbarmin_arg;
    std::int32_t Tbarmax_arg;
    array_desc G32_arg(G32_memory, G32_length);
    array_desc E_arg(E_memory, E_length);
    array_desc Ebar32_arg(Ebar32_memory, Ebar32_length);
    array_desc info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &Tmin_arg, &Tmax_arg,   &Tbarmin_arg, &Tbarmax_arg,
        &G32_arg, &E_arg,    &Ebar32_arg, &info_arg,
    };

    // Set E_memory to beginning of input ring buffer
    E_arg = array_desc(E_memory, E_length);

    // Set Ebar_memory to beginning of output ring buffer
    Ebar32_arg = array_desc(Ebar32_memory, Ebar32_length);

    // Ringbuffer size
    const std::ptrdiff_t T_ringbuf = input_ringbuf_signal->size / E_T_sample_bytes;
    const std::ptrdiff_t Tbar_ringbuf = output_ringbuf_signal->size / Ebar32_Tbar_sample_bytes;
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

    // Update metadata
    Ebar32_meta->dim[Ebar32_rank - 1 - Ebar32_index_Tbar] = Tbarlength;
    assert(Ebar32_meta->dim[Ebar32_rank - 1 - Ebar32_index_Tbar]
           <= int(Ebar32_lengths[Ebar32_index_Tbar]));
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    assert(Ebar32_meta->nfreq >= 0);
    assert(Ebar32_meta->nfreq == E_meta->nfreq);
    for (int freq = 0; freq < Ebar32_meta->nfreq; ++freq) {
        Ebar32_meta->freq_upchan_factor[freq] =
            cuda_upchannelization_factor * E_meta->freq_upchan_factor[freq];
        Ebar32_meta->half_fpga_sample0[freq] =
            E_meta->half_fpga_sample0[freq] + cuda_number_of_taps - 1;
        Ebar32_meta->time_downsampling_fpga[freq] =
            cuda_upchannelization_factor * E_meta->time_downsampling_fpga[freq];
    }

    // Copy inputs to device memory

#ifdef DEBUGGING
    // Initialize host-side buffer arrays
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));
#endif

    const std::string symname = "Upchannelizer_chord_U32_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA Upchannelizer_chord_U32 on GPU frame {:d}", gpu_frame_id);
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

    // Check error codes
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
    const std::int32_t error_code = *std::max_element((const std::int32_t*)&*info_host.begin(),
                                                      (const std::int32_t*)&*info_host.end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    for (std::size_t i = 0; i < info_host.size(); ++i)
        if (info_host[i] != 0)
            ERROR("cudaUpchannelizer_chord_U32 returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host[i], i);
#endif

    return record_end_event();
}

void cudaUpchannelizer_chord_U32::finalize_frame() {
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
    DEBUG("    Produced bytes:   {:d}", Tbar_produced * Ebar32_Tbar_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, instance_num,
                                        Tbar_produced * Ebar32_Tbar_sample_bytes);

    cudaCommand::finalize_frame();
}
