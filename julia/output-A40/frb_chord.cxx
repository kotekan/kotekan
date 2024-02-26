/**
 * @file
 * @brief CUDA FRBBeamformer_chord kernel
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
 * @class cudaFRBBeamformer_chord
 * @brief cudaCommand for FRBBeamformer_chord
 */
class cudaFRBBeamformer_chord : public cudaCommand {
public:
    cudaFRBBeamformer_chord(Config& config, const std::string& unique_name,
                            bufferContainer& host_buffers, cudaDeviceInterface& device,
                            const int inst);
    virtual ~cudaFRBBeamformer_chord();

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
    static constexpr int cuda_beam_layout_M = 48;
    static constexpr int cuda_beam_layout_N = 48;
    static constexpr int cuda_dish_layout_M = 24;
    static constexpr int cuda_dish_layout_N = 24;
    static constexpr int cuda_downsampling_factor = 40;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 512;
    static constexpr int cuda_number_of_frequencies = 256;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 512;
    static constexpr int cuda_granularity_number_of_timesamples = 48;

    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

    // Kernel compile parameters:
    static constexpr int minthreads = 768;
    static constexpr int blocks_per_sm = 1;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 24;
    static constexpr int blocks = 256;
    static constexpr int shmem_bytes = 76896;

    // Kernel name:
    const char* const kernel_symbol =
        "_Z3frb13CuDeviceArrayI5Int32Li1ELi1EES_IS0_Li1ELi1EES_IS0_Li1ELi1EES_IS0_Li1ELi1EES_"
        "I7Int16x2Li1ELi1EES_I9Float16x2Li1ELi1EES_I6Int4x8Li1ELi1EES_IS2_Li1ELi1EES_IS0_Li1ELi1EE";

    // Kernel arguments:
    enum class args { Tmin, Tmax, Tbarmin, Tbarmax, S, W, E, I, info, count };

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
    // S: gpu_mem_dishlayout
    static constexpr chordDataType S_type = int16;
    enum S_indices {
        S_index_MN,
        S_index_D,
        S_rank,
    };
    static constexpr std::array<const char*, S_rank> S_labels = {
        "MN",
        "D",
    };
    static constexpr std::array<std::size_t, S_rank> S_lengths = {
        2,
        576,
    };
    static constexpr std::size_t S_length = chord_datatype_bytes(S_type) * 2 * 576;
    static_assert(S_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // W: gpu_mem_phase
    static constexpr chordDataType W_type = float16;
    enum W_indices {
        W_index_C,
        W_index_dishM,
        W_index_dishN,
        W_index_P,
        W_index_F,
        W_rank,
    };
    static constexpr std::array<const char*, W_rank> W_labels = {
        "C", "dishM", "dishN", "P", "F",
    };
    static constexpr std::array<std::size_t, W_rank> W_lengths = {
        2, 24, 24, 2, 256,
    };
    static constexpr std::size_t W_length = chord_datatype_bytes(W_type) * 2 * 24 * 24 * 2 * 256;
    static_assert(W_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
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
        512,
        2,
        256,
        512,
    };
    static constexpr std::size_t E_length = chord_datatype_bytes(E_type) * 512 * 2 * 256 * 512;
    static_assert(E_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // I: gpu_mem_beamgrid
    static constexpr chordDataType I_type = float16;
    enum I_indices {
        I_index_beamP,
        I_index_beamQ,
        I_index_F,
        I_index_Tbar,
        I_rank,
    };
    static constexpr std::array<const char*, I_rank> I_labels = {
        "beamP",
        "beamQ",
        "F",
        "Tbar",
    };
    static constexpr std::array<std::size_t, I_rank> I_lengths = {
        48,
        48,
        256,
        16,
    };
    static constexpr std::size_t I_length = chord_datatype_bytes(I_type) * 48 * 48 * 256 * 16;
    static_assert(I_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
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
        24,
        256,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 24 * 256;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //

    // Kotekan buffer names
    const std::string Tmin_memname;
    const std::string Tmax_memname;
    const std::string Tbarmin_memname;
    const std::string Tbarmax_memname;
    const std::string S_memname;
    const std::string W_memname;
    const std::string E_memname;
    const std::string I_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::uint8_t> Tmin_host;
    std::vector<std::uint8_t> Tmax_host;
    std::vector<std::uint8_t> Tbarmin_host;
    std::vector<std::uint8_t> Tbarmax_host;
    std::vector<std::uint8_t> S_host;
    std::vector<std::uint8_t> info_host;

    static constexpr std::size_t E_T_sample_bytes = chord_datatype_bytes(E_type)
                                                    * E_lengths[E_index_D] * E_lengths[E_index_P]
                                                    * E_lengths[E_index_F];
    static constexpr std::size_t I_Tbar_sample_bytes =
        chord_datatype_bytes(I_type) * I_lengths[I_index_beamP] * I_lengths[I_index_beamQ]
        * I_lengths[I_index_F];

    RingBuffer* input_ringbuf_signal;
    RingBuffer* output_ringbuf_signal;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::size_t Tmin, Tmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::size_t Tbarmin, Tbarmax;
};

REGISTER_CUDA_COMMAND(cudaFRBBeamformer_chord);

cudaFRBBeamformer_chord::cudaFRBBeamformer_chord(Config& config, const std::string& unique_name,
                                                 bufferContainer& host_buffers,
                                                 cudaDeviceInterface& device, const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "FRBBeamformer_chord", "FRBBeamformer_chord.ptx"),
    Tmin_memname(unique_name + "/Tmin"), Tmax_memname(unique_name + "/Tmax"),
    Tbarmin_memname(unique_name + "/Tbarmin"), Tbarmax_memname(unique_name + "/Tbarmax"),
    S_memname(unique_name + "/gpu_mem_dishlayout"),
    W_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    I_memname(config.get<std::string>(unique_name, "gpu_mem_beamgrid")),
    info_memname(unique_name + "/gpu_mem_info"),

    Tmin_host(Tmin_length), Tmax_host(Tmax_length), Tbarmin_host(Tbarmin_length),
    Tbarmax_host(Tbarmax_length), S_host(S_length), info_host(info_length),
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
    gpu_buffers_used.push_back(
        std::make_tuple(get_name() + "_gpu_mem_dishlayout", false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(W_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(E_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(I_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage need to build the kernel
    if (inst == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "FRBBeamformer_chord_");
    }

    if (inst == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
        output_ringbuf_signal->register_producer(unique_name);
        output_ringbuf_signal->allocate_new_metadata_object(0);
    }
}

cudaFRBBeamformer_chord::~cudaFRBBeamformer_chord() {}

std::int64_t
cudaFRBBeamformer_chord::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_produced_elements(num_available_elements) * cuda_downsampling_factor;
}
std::int64_t
cudaFRBBeamformer_chord::num_produced_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) / cuda_downsampling_factor;
}

std::int64_t
cudaFRBBeamformer_chord::num_processed_elements(std::int64_t num_available_elements) const {
    assert(num_available_elements >= cuda_number_of_timesamples);
    return cuda_number_of_timesamples;
}

int cudaFRBBeamformer_chord::wait_on_precondition() {
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

    // How many outputs will we produce?
    const std::size_t Tbar_produced = num_produced_elements(T_available);
    DEBUG("Will produce (samples): Tbar_produced: {:d}", Tbar_produced);
    const std::size_t Tbarlength = Tbar_produced;

    // to bytes
    const std::size_t output_bytes = Tbarlength * I_Tbar_sample_bytes;
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

    assert(mod(output_cursor, I_Tbar_sample_bytes) == 0);
    Tbarmin = output_cursor / I_Tbar_sample_bytes;
    Tbarmax = Tbarmin + Tbarlength;
    DEBUG("Output samples:");
    DEBUG("    Tbarmin:    {:d}", Tbarmin);
    DEBUG("    Tbarmax:    {:d}", Tbarmax);
    DEBUG("    Tbarlength: {:d}", Tbarlength);

    return 0;
}

cudaEvent_t cudaFRBBeamformer_chord::execute(cudaPipelineState& /*pipestate*/,
                                             const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    Tmin_host.resize(Tmin_length);
    void* const Tmin_memory = device.get_gpu_memory(Tmin_memname, Tmin_length);
    Tmax_host.resize(Tmax_length);
    void* const Tmax_memory = device.get_gpu_memory(Tmax_memname, Tmax_length);
    Tbarmin_host.resize(Tbarmin_length);
    void* const Tbarmin_memory = device.get_gpu_memory(Tbarmin_memname, Tbarmin_length);
    Tbarmax_host.resize(Tbarmax_length);
    void* const Tbarmax_memory = device.get_gpu_memory(Tbarmax_memname, Tbarmax_length);
    S_host.resize(S_length);
    void* const S_memory = device.get_gpu_memory(S_memname, S_length);
    void* const W_memory =
        args::W == args::E ? device.get_gpu_memory(W_memname, input_ringbuf_signal->size)
        : args::W == args::I
            ? device.get_gpu_memory(W_memname, output_ringbuf_signal->size)
            : device.get_gpu_memory_array(W_memname, gpu_frame_id, _gpu_buffer_depth, W_length);
    void* const E_memory =
        args::E == args::E ? device.get_gpu_memory(E_memname, input_ringbuf_signal->size)
        : args::E == args::I
            ? device.get_gpu_memory(E_memname, output_ringbuf_signal->size)
            : device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const I_memory =
        args::I == args::E ? device.get_gpu_memory(I_memname, input_ringbuf_signal->size)
        : args::I == args::I
            ? device.get_gpu_memory(I_memname, output_ringbuf_signal->size)
            : device.get_gpu_memory_array(I_memname, gpu_frame_id, _gpu_buffer_depth, I_length);
    info_host.resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // W is an input buffer: check metadata
    const std::shared_ptr<metadataObject> W_mc =
        args::W == args::E ? input_ringbuf_signal->get_metadata(0)
                           : device.get_gpu_memory_array_metadata(W_memname, gpu_frame_id);
    assert(W_mc);
    assert(metadata_is_chord(W_mc));
    const std::shared_ptr<chordMetadata> W_meta = get_chord_metadata(W_mc);
    INFO("input W array: {:s} {:s}", W_meta->get_type_string(), W_meta->get_dimensions_string());
    assert(W_meta->type == W_type);
    assert(W_meta->dims == W_rank);
    for (std::size_t dim = 0; dim < W_rank; ++dim) {
        assert(std::strncmp(W_meta->dim_name[dim], W_labels[W_rank - 1 - dim],
                            sizeof W_meta->dim_name[dim])
               == 0);
        if (args::W == args::E && dim == 0) {
            ERROR("dim={} meta->dim[]={} lengths[]={}", dim, W_meta->dim[dim],
                  int(W_lengths[W_rank - 1 - dim]));
            assert(W_meta->dim[dim] <= int(W_lengths[W_rank - 1 - dim]));
        } else
            assert(W_meta->dim[dim] == int(W_lengths[W_rank - 1 - dim]));
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
        if (args::E == args::E && dim == 0) {
            ERROR("dim={} meta->dim[]={} lengths[]={}", dim, E_meta->dim[dim],
                  int(E_lengths[E_rank - 1 - dim]));
            assert(E_meta->dim[dim] <= int(E_lengths[E_rank - 1 - dim]));
        } else
            assert(E_meta->dim[dim] == int(E_lengths[E_rank - 1 - dim]));
    }
    //
    // I is an output buffer: set metadata
    std::shared_ptr<metadataObject> const I_mc =
        args::I == args::I
            ? output_ringbuf_signal->get_metadata(0)
            : device.create_gpu_memory_array_metadata(I_memname, gpu_frame_id, E_mc->parent_pool);
    std::shared_ptr<chordMetadata> const I_meta = get_chord_metadata(I_mc);
    *I_meta = *E_meta;
    I_meta->type = I_type;
    I_meta->dims = I_rank;
    for (std::size_t dim = 0; dim < I_rank; ++dim) {
        std::strncpy(I_meta->dim_name[dim], I_labels[I_rank - 1 - dim],
                     sizeof I_meta->dim_name[dim]);
        I_meta->dim[dim] = I_lengths[I_rank - 1 - dim];
    }
    INFO("output I array: {:s} {:s}", I_meta->get_type_string(), I_meta->get_dimensions_string());
    //

    // TODO: Initialize (and copy to the GPU) S only once at the beginning
    {
        // S maps dishes to locations.
        // The first `ndishes` dishes are real dishes,
        // the remaining dishes are not real and exist only to label the unoccupied dish locations.
        assert(E_meta->ndishes == cuda_number_of_dishes);
        assert(E_meta->n_dish_locations_ew == cuda_dish_layout_N);
        assert(E_meta->n_dish_locations_ns == cuda_dish_layout_M);
        assert(E_meta->dish_index);
        std::int16_t* __restrict__ const S =
            static_cast<std::int16_t*>(static_cast<void*>(S_host.data()));
        int surplus_dish_index = cuda_number_of_dishes;
        for (int locM = 0; locM < cuda_dish_layout_M; ++locM) {
            for (int locN = 0; locN < cuda_dish_layout_N; ++locN) {
                int dish_index = E_meta->get_dish_index(locN, locM);
                if (dish_index >= 0) {
                    // This location holds a real dish, record its location
                    S[2 * dish_index + 0] = locM;
                    S[2 * dish_index + 1] = locN;
                } else {
                    // This location is empty, assign it a surplus dish index
                    S[2 * surplus_dish_index + 0] = locM;
                    S[2 * surplus_dish_index + 1] = locN;
                    ++surplus_dish_index;
                }
            }
        }
        assert(surplus_dish_index == cuda_dish_layout_M * cuda_dish_layout_N);
    }

    record_start_event();

    const char* exc_arg = "exception";
    kernel_arg Tmin_arg(Tmin_memory, Tmin_length);
    kernel_arg Tmax_arg(Tmax_memory, Tmax_length);
    kernel_arg Tbarmin_arg(Tbarmin_memory, Tbarmin_length);
    kernel_arg Tbarmax_arg(Tbarmax_memory, Tbarmax_length);
    kernel_arg S_arg(S_memory, S_length);
    kernel_arg W_arg(W_memory, W_length);
    kernel_arg E_arg(E_memory, E_length);
    kernel_arg I_arg(I_memory, I_length);
    kernel_arg info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &Tmin_arg, &Tmax_arg, &Tbarmin_arg, &Tbarmax_arg,
        &S_arg,   &W_arg,    &E_arg,    &I_arg,       &info_arg,
    };

    INFO("gpu_frame_id: {}", gpu_frame_id);

    // Set E_memory to beginning of input ring buffer
    E_arg = kernel_arg(E_memory, E_length);

    // Set I_memory to beginning of output ring buffer
    I_arg = kernel_arg(I_memory, I_length);

    // Ringbuffer size
    const std::size_t T_ringbuf = input_ringbuf_signal->size / E_T_sample_bytes;
    const std::size_t Tbar_ringbuf = output_ringbuf_signal->size / I_Tbar_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", T_ringbuf);
    DEBUG("Output ringbuffer size (samples): {:d}", Tbar_ringbuf);

    const std::size_t Tlength = Tmax - Tmin;
    const std::size_t Tbarlength = Tbarmax - Tbarmin;
    DEBUG("Processed input samples: {:d}", Tlength);
    DEBUG("Produced output samples: {:d}", Tbarlength);

    DEBUG("Kernel arguments:");
    DEBUG("    Tmin: {:d}", Tmin);
    DEBUG("    Tmax: {:d}", Tmax);
    DEBUG("    Tbarmin:  {:d}", Tbarmin);
    DEBUG("    Tbarmax:  {:d}", Tbarmax);

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    *(std::int32_t*)Tmin_host.data() = mod(Tmin, T_ringbuf);
    *(std::int32_t*)Tmax_host.data() = mod(Tmin, T_ringbuf) + Tlength;
    *(std::int32_t*)Tbarmin_host.data() = mod(Tbarmin, Tbar_ringbuf);
    *(std::int32_t*)Tbarmax_host.data() = mod(Tbarmin, Tbar_ringbuf) + Tbarlength;

    // Update metadata
    I_meta->dim[0] = Tbarlength;
    assert(I_meta->dim[0] <= int(I_lengths[3]));

    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    assert(I_meta->nfreq >= 0);
    assert(I_meta->nfreq == E_meta->nfreq);
    for (int freq = 0; freq < I_meta->nfreq; ++freq) {
        I_meta->freq_upchan_factor[freq] =
            cuda_downsampling_factor * E_meta->freq_upchan_factor[freq];
        I_meta->time_downsampling_fpga[freq] =
            cuda_downsampling_factor * E_meta->time_downsampling_fpga[freq];
    }

    // Copy inputs to device memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tmin_memory, Tmin_host.data(), Tmin_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tmax_memory, Tmax_host.data(), Tmax_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tbarmin_memory, Tbarmin_host.data(), Tbarmin_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tbarmax_memory, Tbarmax_host.data(), Tbarmax_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(S_memory, S_host.data(), S_length, cudaMemcpyHostToDevice,
                                     device.getStream(cuda_stream_id)));

    // Initialize host-side buffer arrays
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));

    const std::string symname = "FRBBeamformer_chord_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA FRBBeamformer_chord on GPU frame {:d}", gpu_frame_id);
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
            ERROR("cudaFRBBeamformer_chord returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host[i], i);

    return record_end_event();
}

void cudaFRBBeamformer_chord::finalize_frame() {
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
    DEBUG("    Produced bytes:   {:d}", Tbar_produced * I_Tbar_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, Tbar_produced * I_Tbar_sample_bytes);

    cudaCommand::finalize_frame();
}
