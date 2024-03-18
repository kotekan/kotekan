/**
 * @file
 * @brief CUDA Upchannelizer_chord_U128 kernel
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
#include <fmt.hpp>
#include <limits>
#include <ringbuffer.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;

namespace {
// Round down `x` to the next lower multiple of `y`
template<typename T, typename U>
auto round_down(T x, U y) {
    assert(x >= 0);
    assert(y > 0);
    auto r = x / y * y;
    assert(r % y == 0);
    assert(0 <= r && r <= x && r + y > x);
    return r;
}

// Calculate `x mod y`, returning `x` with `0 <= x < y`
template<typename T, typename U>
auto mod(T x, U y) {
    assert(y > 0);
    while (x < 0)
        x += y;
    auto r = x % y;
    assert(0 <= r && r < y);
    return r;
}
} // namespace

/**
 * @class cudaUpchannelizer_chord_U128
 * @brief cudaCommand for Upchannelizer_chord_U128
 */
class cudaUpchannelizer_chord_U128 : public cudaCommand {
public:
    cudaUpchannelizer_chord_U128(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, cudaDeviceInterface& device,
                                 const int inst);
    virtual ~cudaUpchannelizer_chord_U128();

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
    static constexpr int cuda_number_of_dishes = 512;
    static constexpr int cuda_number_of_frequencies = 16;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_taps = 4;
    static constexpr int cuda_max_number_of_timesamples = 8192;
    static constexpr int cuda_granularity_number_of_timesamples = 256;
    static constexpr int cuda_algorithm_overlap = 384;
    static constexpr int cuda_upchannelization_factor = 128;

    // Kernel compile parameters:
    static constexpr int minthreads = 512;
    static constexpr int blocks_per_sm = 2;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 16;
    static constexpr int blocks = 128;
    static constexpr int shmem_bytes = 66816;

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
        2048,
    };
    static constexpr std::size_t G_length = chord_datatype_bytes(G_type) * 2048;
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
        512,
        2,
        16,
        8192,
    };
    static constexpr std::size_t E_length = chord_datatype_bytes(E_type) * 512 * 2 * 16 * 8192;
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
        512,
        2,
        2048,
        64,
    };
    static constexpr std::size_t Ebar_length =
        chord_datatype_bytes(Ebar_type) * 512 * 2 * 2048 * 64;
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
        16,
        128,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 16 * 128;
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

    static constexpr size_t T_sample_bytes = E_lengths[E_index_D] * E_lengths[E_index_P]
                                             * E_lengths[E_index_F] * chord_datatype_bytes(E_type);
    static constexpr size_t Tbar_sample_bytes =
        Ebar_lengths[Ebar_index_D] * Ebar_lengths[Ebar_index_P] * Ebar_lengths[Ebar_index_Fbar]
        * chord_datatype_bytes(Ebar_type);

    RingBuffer* input_ringbuf_signal;
    RingBuffer* output_ringbuf_signal;

    size_t Tmin;
    size_t Tmax;
    size_t Tbarmin;
    size_t Tbarmax;
};

REGISTER_CUDA_COMMAND(cudaUpchannelizer_chord_U128);

cudaUpchannelizer_chord_U128::cudaUpchannelizer_chord_U128(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& host_buffers,
                                                           cudaDeviceInterface& device,
                                                           const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "Upchannelizer_chord_U128", "Upchannelizer_chord_U128.ptx"),
    Tmin_memname(unique_name + "/Tmin"), Tmax_memname(unique_name + "/Tmax"),
    Tbarmin_memname(unique_name + "/Tbarmin"), Tbarmax_memname(unique_name + "/Tbarmax"),
    G_memname(config.get<std::string>(unique_name, "gpu_mem_gain")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_input_voltage")),
    Ebar_memname(config.get<std::string>(unique_name, "gpu_mem_output_voltage")),
    info_memname(unique_name + "/gpu_mem_info"),

    Tmin_host(Tmin_length), Tmax_host(Tmax_length), Tbarmin_host(Tbarmin_length),
    Tbarmax_host(Tbarmax_length), info_host(info_length), Tmin(0), Tmax(0), Tbarmin(0), Tbarmax(0) {
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

    // Only one of the instances of this pipeline stage need to build the kernel
    if (inst == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "Upchannelizer_chord_U128_");
    }

    // Find input and output buffers used for signalling ring-buffer state
    input_ringbuf_signal = dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")));
    output_ringbuf_signal = dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "out_signal")));
    if (inst == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
        output_ringbuf_signal->register_producer(unique_name);
    }
}

cudaUpchannelizer_chord_U128::~cudaUpchannelizer_chord_U128() {}

int cudaUpchannelizer_chord_U128::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer...
    size_t input_bytes = E_length / _gpu_buffer_depth;
    // If we have leftovers from last frame, we can request fewer input bytes.
    size_t T_leftover = (gpu_frame_id == 0 ? 0 : cuda_algorithm_overlap);
    size_t req_input_bytes = input_bytes - T_leftover * T_sample_bytes;
    DEBUG("Waiting for input ringbuffer data for frame {:d}: {:d} bytes total, {:d} leftovers + "
          "{:d} new = {:d} time samples total, {:d} leftover + {:d} new...",
          gpu_frame_id, input_bytes, T_leftover * T_sample_bytes, req_input_bytes,
          input_bytes / T_sample_bytes, T_leftover, req_input_bytes / T_sample_bytes);
    std::optional<size_t> val =
        input_ringbuf_signal->wait_and_claim_readable(unique_name, req_input_bytes);
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    if (!val.has_value())
        return -1;
    // Where we should read from in the ring buffer, in *bytes*
    size_t input_cursor = val.value();
    // (that's the pointer to the 'new' part of the data -- we also have leftovers at the beginning
    // of the ring buffer -- we adjust for that below.)
    assert(mod(input_cursor, T_sample_bytes) == 0);

    // Convert from byte offset to time sample offset
    Tmin = input_cursor / T_sample_bytes;
    DEBUG("Input ring-buffer byte offset {:d} -> time sample offset {:d}", input_cursor, Tmin);

    // Subtract T_leftover samples, avoiding underflow
    size_t Tringbuf = input_ringbuf_signal->size / T_sample_bytes;
    Tmin = (Tmin + (Tringbuf - T_leftover)) % Tringbuf;

    DEBUG("After adjusting for leftover samples, input ring-buffer time sample offset is {:d}",
          Tmin);

    DEBUG("Input length: {:d} bytes, bytes per input sample: {:d}, input samples: {:d}",
          input_bytes, T_sample_bytes, input_bytes / T_sample_bytes);
    size_t nominal_Tlength = input_bytes / T_sample_bytes;
    assert(mod(input_bytes, T_sample_bytes) == 0);
    size_t Tlength = round_down(nominal_Tlength, cuda_granularity_number_of_timesamples);

    Tmax = Tmin + Tlength;
    assert(Tmax <= std::numeric_limits<int32_t>::max());

    // How many outputs will we produce?
    assert(Tlength > cuda_algorithm_overlap);
    size_t Tbarlength = (Tlength - cuda_algorithm_overlap) / cuda_upchannelization_factor;

    // to bytes
    size_t output_bytes = Tbarlength * Tbar_sample_bytes;
    DEBUG("Will produce {:d} output time samples, sample size {:d}, total {:d} bytes", Tbarlength,
          Tbar_sample_bytes, output_bytes);

    // Wait for space to be available in our output ringbuffer...
    DEBUG("Waiting for output ringbuffer space for frame {:d}: {:d} bytes ...", gpu_frame_id,
          output_bytes);
    val = output_ringbuf_signal->wait_for_writable(unique_name, output_bytes);
    DEBUG("Finished waiting for output for data frame {:d}.", gpu_frame_id);
    if (!val.has_value())
        return -1;
    size_t output_cursor = val.value();
    assert(mod(output_cursor, Tbar_sample_bytes) == 0);
    Tbarmin = output_cursor / Tbar_sample_bytes;
    DEBUG("Output ring-buffer byte offset {:d} -> tbar time sample offset {:d}", output_cursor,
          Tbarmin);

    Tbarmax = Tbarmin + Tbarlength;
    assert(Tbarmax <= std::numeric_limits<int32_t>::max());

    return 0;
}

cudaEvent_t cudaUpchannelizer_chord_U128::execute(cudaPipelineState& /*pipestate*/,
                                                  const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    void* const Tmin_memory = device.get_gpu_memory(Tmin_memname, Tmin_length);
    void* const Tmax_memory = device.get_gpu_memory(Tmax_memname, Tmax_length);
    void* const Tbarmin_memory = device.get_gpu_memory(Tbarmin_memname, Tbarmin_length);
    void* const Tbarmax_memory = device.get_gpu_memory(Tbarmax_memname, Tbarmax_length);

    void* const G_memory =
        (args::G == args::E) ? device.get_gpu_memory(G_memname, input_ringbuf_signal->size)
                             : ((args::G == args::Ebar)
                                    ? device.get_gpu_memory(G_memname, output_ringbuf_signal->size)
                                    : device.get_gpu_memory_array(G_memname, gpu_frame_id,
                                                                  _gpu_buffer_depth, G_length));

    void* const E_memory =
        (args::E == args::E) ? device.get_gpu_memory(E_memname, input_ringbuf_signal->size)
                             : ((args::E == args::Ebar)
                                    ? device.get_gpu_memory(E_memname, output_ringbuf_signal->size)
                                    : device.get_gpu_memory_array(E_memname, gpu_frame_id,
                                                                  _gpu_buffer_depth, E_length));

    void* const Ebar_memory =
        (args::Ebar == args::E)
            ? device.get_gpu_memory(Ebar_memname, input_ringbuf_signal->size)
            : ((args::Ebar == args::Ebar)
                   ? device.get_gpu_memory(Ebar_memname, output_ringbuf_signal->size)
                   : device.get_gpu_memory_array(Ebar_memname, gpu_frame_id, _gpu_buffer_depth,
                                                 Ebar_length));
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // G is an input buffer: check metadata
    const std::shared_ptr<metadataObject> G_mc =
        device.get_gpu_memory_array_metadata(G_memname, gpu_frame_id);
    assert(G_mc && metadata_is_chord(G_mc));
    const std::shared_ptr<chordMetadata> G_meta = get_chord_metadata(G_mc);
    INFO("input G array: {:s} {:s}", G_meta->get_type_string(), G_meta->get_dimensions_string());
    assert(G_meta->type == G_type);
    assert(G_meta->dims == G_rank);
    for (std::size_t dim = 0; dim < G_rank; ++dim) {
        assert(std::strncmp(G_meta->dim_name[dim], G_labels[G_rank - 1 - dim],
                            sizeof G_meta->dim_name[dim])
               == 0);
        if (args::G == args::E)
            assert(G_meta->dim[dim] <= int(G_lengths[G_rank - 1 - dim]));
        else
            assert(G_meta->dim[dim] == int(G_lengths[G_rank - 1 - dim]));
    }
    //
    // E is an input buffer: check metadata
    const std::shared_ptr<metadataObject> E_mc =
        device.get_gpu_memory_array_metadata(E_memname, gpu_frame_id);
    assert(E_mc && metadata_is_chord(E_mc));
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
    /// Ebar is an output buffer: set metadata
    std::shared_ptr<metadataObject> const Ebar_mc =
        device.create_gpu_memory_array_metadata(Ebar_memname, gpu_frame_id, E_mc->parent_pool);
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

    // TODO -- we need to tweak the metadata we get from the input ringbuffer!

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

    // We need an overlap of this many time samples, i.e. this many
    // time samples will need to be re-processed in the next
    // iteration. This also defines the number of time samples that
    // will be missing from the output.
    INFO("cuda_algorithm_overlap: {}", cuda_algorithm_overlap);
    // The input is processed in batches of this size. This means that
    // the input we provide must be a multiple of this number.
    INFO("cuda_granularity_number_of_timesamples: {}", cuda_granularity_number_of_timesamples);

    INFO("gpu_frame_id: {}", gpu_frame_id);

    // Set E_memory to beginning of input ring buffer
    E_arg = kernel_arg(E_memory, E_length);

    // Set Ebar_memory to beginning of output ring buffer
    Ebar_arg = kernel_arg(Ebar_memory, Ebar_length);

    const std::int64_t Tmin_wrapped = Tmin;
    const std::int64_t Tmax_wrapped = Tmax;
    const std::int64_t Tbarmin_wrapped = Tbarmin;
    const std::int64_t Tbarmax_wrapped = Tbarmax;
    const std::int64_t Tbarlength = (Tbarmax - Tbarmin);

    INFO("Tmin_wrapped: {}", Tmin_wrapped);
    INFO("Tmax_wrapped: {}", Tmax_wrapped);
    INFO("Tbarmin_wrapped: {}", Tbarmin_wrapped);
    INFO("Tbarmax_wrapped: {}", Tbarmax_wrapped);

    assert(Tmin_wrapped >= 0 && Tmin_wrapped <= Tmax_wrapped
           && Tmax_wrapped <= std::numeric_limits<int32_t>::max());
    assert(Tbarmin_wrapped >= 0 && Tbarmin_wrapped <= Tbarmax_wrapped
           && Tbarmax_wrapped <= std::numeric_limits<int32_t>::max());

    // // These are the conditions in the CUDA kernel
    // const int T = 32768 * 4;
    // const int Touter = 256;
    // const int U = 16;
    // const int M = 4;
    // assert(0 <= Tmin);
    // assert(Tmin <= Tmax);
    // assert(Tmax <= 2 * T);
    // assert((Tmax - Tmin) % Touter == 0);
    // assert(0 <= Tbarmin);
    // assert(Tbarmin <= Tbarmax);
    // assert(Tbarmax <= 2 * (T / U));
    // assert((Tbarmax - Tbarmin + (M - 1)) % (Touter / U) == 0);

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    *(std::int32_t*)Tmin_host.data() = Tmin_wrapped;
    *(std::int32_t*)Tmax_host.data() = Tmax_wrapped;
    *(std::int32_t*)Tbarmin_host.data() = Tbarmin_wrapped;
    *(std::int32_t*)Tbarmax_host.data() = Tbarmax_wrapped;

    // Update metadata
    // TODO -- check this!!
    Ebar_meta->dim[0] = Tbarlength;
    assert(Ebar_meta->dim[0] <= int(Ebar_lengths[3]));

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

    std::string symname = "Upchannelizer_chord_U128_" + std::string(kernel_symbol);
    DEBUG("kernel_symbol: {}", symname);
    DEBUG("runtime_kernels[kernel_symbol]: {}",
          static_cast<void*>(device.runtime_kernels[symname]));
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA Upchannelizer_chord_U128 on GPU frame {:d}", gpu_frame_id);
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
            ERROR("cudaUpchannelizer_chord_U128 returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host[i], i);

    return record_end_event();
}

void cudaUpchannelizer_chord_U128::finalize_frame() {
    // Advance the input ringbuffer
    size_t t_read = ((Tmax - Tmin) - cuda_algorithm_overlap);
    DEBUG("Advancing input ringbuffer: read {:d} samples, {:d} bytes", t_read,
          t_read * T_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, t_read * T_sample_bytes);

    // Advance the output ringbuffer
    size_t t_written = (Tbarmax - Tbarmin);
    DEBUG("Advancing output ringbuffer: wrote {:d} samples, {:d} bytes", t_written,
          t_written * Tbar_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, t_written * Tbar_sample_bytes);

    cudaCommand::finalize_frame();
}
