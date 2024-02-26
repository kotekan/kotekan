/**
 * @file
 * @brief CUDA BasebandBeamformer_hirax kernel
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
 * @class cudaBasebandBeamformer_hirax
 * @brief cudaCommand for BasebandBeamformer_hirax
 */
class cudaBasebandBeamformer_hirax : public cudaCommand {
public:
    cudaBasebandBeamformer_hirax(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, cudaDeviceInterface& device,
                                 const int inst);
    virtual ~cudaBasebandBeamformer_hirax();

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
    static constexpr int cuda_number_of_beams = 16;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 256;
    static constexpr int cuda_number_of_frequencies = 64;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 8192;
    static constexpr int cuda_granularity_number_of_timesamples = 2048;
    static constexpr int cuda_shift_parameter_sigma = 3;

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
    static constexpr int blocks = 2048;
    static constexpr int shmem_bytes = 13440;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z2bb13CuDeviceArrayI5Int32Li1ELi1EES_IS0_Li1ELi1EES_I6Int8x4Li1ELi1EES_"
        "I6Int4x8Li1ELi1EES_IS0_Li1ELi1EES_IS2_Li1ELi1EES_IS0_Li1ELi1EE";

    // Kernel arguments:
    enum class args { Tmin, Tmax, A, E, s, J, info, count };

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
    // A: gpu_mem_phase
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
    static constexpr std::array<std::size_t, A_rank> A_lengths = {
        2, 256, 16, 2, 64,
    };
    static constexpr std::size_t A_length = chord_datatype_bytes(A_type) * 2 * 256 * 16 * 2 * 64;
    static_assert(A_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // E: gpu_mem_voltage
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
        256,
        2,
        64,
        8192,
    };
    static constexpr std::size_t E_length = chord_datatype_bytes(E_type) * 256 * 2 * 64 * 8192;
    static_assert(E_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // s: gpu_mem_output_scaling
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
    static constexpr std::array<std::size_t, s_rank> s_lengths = {
        16,
        2,
        64,
    };
    static constexpr std::size_t s_length = chord_datatype_bytes(s_type) * 16 * 2 * 64;
    static_assert(s_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // J: gpu_mem_formed_beams
    static constexpr chordDataType J_type = int4p4;
    enum J_indices {
        J_index_Tout,
        J_index_P,
        J_index_F,
        J_index_B,
        J_rank,
    };
    static constexpr std::array<const char*, J_rank> J_labels = {
        "Tout",
        "P",
        "F",
        "B",
    };
    static constexpr std::array<std::size_t, J_rank> J_lengths = {
        2048,
        2,
        64,
        16,
    };
    static constexpr std::size_t J_length = chord_datatype_bytes(J_type) * 2048 * 2 * 64 * 16;
    static_assert(J_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
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
        4,
        2048,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 4 * 2048;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //

    // Kotekan buffer names
    const std::string Tmin_memname;
    const std::string Tmax_memname;
    const std::string A_memname;
    const std::string E_memname;
    const std::string s_memname;
    const std::string J_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::uint8_t> Tmin_host;
    std::vector<std::uint8_t> Tmax_host;
    std::vector<std::uint8_t> info_host;

    static constexpr std::size_t E_T_sample_bytes = chord_datatype_bytes(E_type)
                                                    * E_lengths[E_index_D] * E_lengths[E_index_P]
                                                    * E_lengths[E_index_F];

    RingBuffer* input_ringbuf_signal;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::size_t Tmin, Tmax;
};

REGISTER_CUDA_COMMAND(cudaBasebandBeamformer_hirax);

cudaBasebandBeamformer_hirax::cudaBasebandBeamformer_hirax(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& host_buffers,
                                                           cudaDeviceInterface& device,
                                                           const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "BasebandBeamformer_hirax", "BasebandBeamformer_hirax.ptx"),
    Tmin_memname(unique_name + "/Tmin"), Tmax_memname(unique_name + "/Tmax"),
    A_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    s_memname(config.get<std::string>(unique_name, "gpu_mem_output_scaling")),
    J_memname(config.get<std::string>(unique_name, "gpu_mem_formed_beams")),
    info_memname(unique_name + "/gpu_mem_info"),

    Tmin_host(Tmin_length), Tmax_host(Tmax_length), info_host(info_length),
    // Find input buffer used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")))) {
    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_Tmin", false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_Tmax", false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(A_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(s_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(J_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage needs to build the kernel
    if (inst == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "BasebandBeamformer_hirax_");
    }

    if (inst == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
    }
}

cudaBasebandBeamformer_hirax::~cudaBasebandBeamformer_hirax() {}

std::int64_t
cudaBasebandBeamformer_hirax::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_produced_elements(num_available_elements);
}
std::int64_t
cudaBasebandBeamformer_hirax::num_produced_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements);
}

std::int64_t
cudaBasebandBeamformer_hirax::num_processed_elements(std::int64_t num_available_elements) const {
    assert(num_available_elements >= cuda_number_of_timesamples);
    return cuda_number_of_timesamples;
}

int cudaBasebandBeamformer_hirax::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    const std::size_t input_bytes = cuda_number_of_timesamples * E_T_sample_bytes;
    DEBUG("Input ring-buffer byte count: {:d}", input_bytes);
    DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::size_t> val_in =
        input_ringbuf_signal->wait_and_claim_readable(unique_name, input_bytes);
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    if (!val_in.has_value())
        return -1;
    const std::size_t input_cursor = val_in.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);

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

    Tmin = div_noremainder(input_cursor, E_T_sample_bytes);
    Tmax = Tmin + T_processed;
    const std::size_t Tlength = Tmax - Tmin;
    DEBUG("Input samples:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);
    DEBUG("    Tlength: {:d}", Tlength);

    return 0;
}

cudaEvent_t cudaBasebandBeamformer_hirax::execute(cudaPipelineState& /*pipestate*/,
                                                  const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    void* const Tmin_memory = device.get_gpu_memory(Tmin_memname, Tmin_length);
    void* const Tmax_memory = device.get_gpu_memory(Tmax_memname, Tmax_length);
    void* const A_memory =
        args::A == args::E
            ? device.get_gpu_memory(A_memname, input_ringbuf_signal->size)
            : device.get_gpu_memory_array(A_memname, gpu_frame_id, _gpu_buffer_depth, A_length);
    void* const E_memory =
        args::E == args::E
            ? device.get_gpu_memory(E_memname, input_ringbuf_signal->size)
            : device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const s_memory =
        args::s == args::E
            ? device.get_gpu_memory(s_memname, input_ringbuf_signal->size)
            : device.get_gpu_memory_array(s_memname, gpu_frame_id, _gpu_buffer_depth, s_length);
    void* const J_memory =
        args::J == args::E
            ? device.get_gpu_memory(J_memname, input_ringbuf_signal->size)
            : device.get_gpu_memory_array(J_memname, gpu_frame_id, _gpu_buffer_depth, J_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // A is an input buffer: check metadata
    const std::shared_ptr<metadataObject> A_mc =
        args::A == args::E ? input_ringbuf_signal->get_metadata(0)
                           : device.get_gpu_memory_array_metadata(A_memname, gpu_frame_id);
    assert(A_mc);
    assert(metadata_is_chord(A_mc));
    const std::shared_ptr<chordMetadata> A_meta = get_chord_metadata(A_mc);
    INFO("input A array: {:s} {:s}", A_meta->get_type_string(), A_meta->get_dimensions_string());
    assert(A_meta->type == A_type);
    assert(A_meta->dims == A_rank);
    for (std::size_t dim = 0; dim < A_rank; ++dim) {
        assert(std::strncmp(A_meta->dim_name[dim], A_labels[A_rank - 1 - dim],
                            sizeof A_meta->dim_name[dim])
               == 0);
        if (args::A == args::E)
            assert(A_meta->dim[dim] <= int(A_lengths[A_rank - 1 - dim]));
        else
            assert(A_meta->dim[dim] == int(A_lengths[A_rank - 1 - dim]));
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
        if (args::E == args::E)
            assert(E_meta->dim[dim] <= int(E_lengths[E_rank - 1 - dim]));
        else
            assert(E_meta->dim[dim] == int(E_lengths[E_rank - 1 - dim]));
    }
    //
    // s is an input buffer: check metadata
    const std::shared_ptr<metadataObject> s_mc =
        args::s == args::E ? input_ringbuf_signal->get_metadata(0)
                           : device.get_gpu_memory_array_metadata(s_memname, gpu_frame_id);
    assert(s_mc);
    assert(metadata_is_chord(s_mc));
    const std::shared_ptr<chordMetadata> s_meta = get_chord_metadata(s_mc);
    INFO("input s array: {:s} {:s}", s_meta->get_type_string(), s_meta->get_dimensions_string());
    assert(s_meta->type == s_type);
    assert(s_meta->dims == s_rank);
    for (std::size_t dim = 0; dim < s_rank; ++dim) {
        assert(std::strncmp(s_meta->dim_name[dim], s_labels[s_rank - 1 - dim],
                            sizeof s_meta->dim_name[dim])
               == 0);
        if (args::s == args::E)
            assert(s_meta->dim[dim] <= int(s_lengths[s_rank - 1 - dim]));
        else
            assert(s_meta->dim[dim] == int(s_lengths[s_rank - 1 - dim]));
    }
    //
    // J is an output buffer: set metadata
    std::shared_ptr<metadataObject> const J_mc =
        device.create_gpu_memory_array_metadata(J_memname, gpu_frame_id, E_mc->parent_pool);
    std::shared_ptr<chordMetadata> const J_meta = get_chord_metadata(J_mc);
    *J_meta = *E_meta;
    J_meta->type = J_type;
    J_meta->dims = J_rank;
    for (std::size_t dim = 0; dim < J_rank; ++dim) {
        std::strncpy(J_meta->dim_name[dim], J_labels[J_rank - 1 - dim],
                     sizeof J_meta->dim_name[dim]);
        J_meta->dim[dim] = J_lengths[J_rank - 1 - dim];
    }
    INFO("output J array: {:s} {:s}", J_meta->get_type_string(), J_meta->get_dimensions_string());
    //

    record_start_event();

    const char* exc_arg = "exception";
    kernel_arg Tmin_arg(Tmin_memory, Tmin_length);
    kernel_arg Tmax_arg(Tmax_memory, Tmax_length);
    kernel_arg A_arg(A_memory, A_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg s_arg(s_memory, s_length);
    kernel_arg J_arg(J_memory, J_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &Tmin_arg, &Tmax_arg, &A_arg, &E_arg, &s_arg, &J_arg, &info_arg,
    };

    INFO("gpu_frame_id: {}", gpu_frame_id);

    // Set E_memory to beginning of input ring buffer
    E_arg = kernel_arg(E_memory, E_length);

    // Ringbuffer size
    const std::size_t T_ringbuf = input_ringbuf_signal->size / E_T_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", T_ringbuf);

    const std::size_t Tlength = Tmax - Tmin;
    DEBUG("Processed input samples: {:d}", Tlength);

    DEBUG("Kernel arguments:");
    DEBUG("    Tmin:    {:d}", Tmin);
    DEBUG("    Tmax:    {:d}", Tmax);

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    *(std::int32_t*)Tmin_host.data() = mod(Tmin, T_ringbuf);
    *(std::int32_t*)Tmax_host.data() = mod(Tmin, T_ringbuf) + Tlength;

    // Copy inputs to device memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tmin_memory, Tmin_host.data(), Tmin_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tmax_memory, Tmax_host.data(), Tmax_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));

    // Initialize host-side buffer arrays
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));

    const std::string symname = "BasebandBeamformer_hirax_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA BasebandBeamformer_hirax on GPU frame {:d}", gpu_frame_id);
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
            ERROR("cudaBasebandBeamformer_hirax returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host[i], i);

    return record_end_event();
}

void cudaBasebandBeamformer_hirax::finalize_frame() {
    const std::size_t Tlength = Tmax - Tmin;

    // Advance the input ringbuffer
    const std::size_t T_consumed = num_consumed_elements(Tlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples: {:d}", T_consumed);
    DEBUG("    Consumed bytes:   {:d}", T_consumed * E_T_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, T_consumed * E_T_sample_bytes);

    cudaCommand::finalize_frame();
}
