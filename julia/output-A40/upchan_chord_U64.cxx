/**
 * @file
 * @brief CUDA Upchannelizer_chord_U64 kernel
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
 * @class cudaUpchannelizer_chord_U64
 * @brief cudaCommand for Upchannelizer_chord_U64
 */
class cudaUpchannelizer_chord_U64 : public cudaCommand {
public:
    cudaUpchannelizer_chord_U64(Config& config, const std::string& unique_name,
                                bufferContainer& host_buffers, cudaDeviceInterface& device,
                                const int inst);
    virtual ~cudaUpchannelizer_chord_U64();

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
    static constexpr int cuda_algorithm_overlap = 192;
    static constexpr int cuda_upchannelization_factor = 64;

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
        1024,
    };
    static constexpr std::size_t G_length = chord_datatype_bytes(G_type) * 1024;
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
        1024,
        128,
    };
    static constexpr std::size_t Ebar_length =
        chord_datatype_bytes(Ebar_type) * 512 * 2 * 1024 * 128;
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
    std::vector<std::vector<std::uint8_t>> Tmin_host;
    std::vector<std::vector<std::uint8_t>> Tmax_host;
    std::vector<std::vector<std::uint8_t>> Tbarmin_host;
    std::vector<std::vector<std::uint8_t>> Tbarmax_host;
    std::vector<std::vector<std::uint8_t>> info_host;

    // Loop-carried information

    // How many time samples from the previous iteration still need to be processed (or processed
    // again)?
    std::int64_t unprocessed;
    // How many time samples were not provided in the previous iteration and need to be provided in
    // this iteration?
    std::int64_t unprovided;
};

REGISTER_CUDA_COMMAND(cudaUpchannelizer_chord_U64);

cudaUpchannelizer_chord_U64::cudaUpchannelizer_chord_U64(Config& config,
                                                         const std::string& unique_name,
                                                         bufferContainer& host_buffers,
                                                         cudaDeviceInterface& device,
                                                         const int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "Upchannelizer_chord_U64", "Upchannelizer_chord_U64.ptx"),
    Tmin_memname(unique_name + "/Tmin"), Tmax_memname(unique_name + "/Tmax"),
    Tbarmin_memname(unique_name + "/Tbarmin"), Tbarmax_memname(unique_name + "/Tbarmax"),
    G_memname(config.get<std::string>(unique_name, "gpu_mem_gain")),
    E_memname(config.get<std::string>(unique_name, "gpu_mem_input_voltage")),
    Ebar_memname(config.get<std::string>(unique_name, "gpu_mem_output_voltage")),
    info_memname(unique_name + "/gpu_mem_info"),

    Tmin_host(_gpu_buffer_depth), Tmax_host(_gpu_buffer_depth), Tbarmin_host(_gpu_buffer_depth),
    Tbarmax_host(_gpu_buffer_depth), info_host(_gpu_buffer_depth), unprocessed(0), unprovided(0) {
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
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts);
    }

    // // Create a ring buffer. Create it only once.
    // assert(E_length % _gpu_buffer_depth == 0);
    // const std::ptrdiff_t E_buffer_size = E_length / _gpu_buffer_depth;
    // const std::ptrdiff_t E_ringbuffer_size = E_length;
    // // if (cuda_upchannelization_factor == 128)
    //     device.create_gpu_memory_ringbuffer(E_memname + "_ringbuffer", E_ringbuffer_size,
    //                                         E_memname, 0, E_buffer_size);
}

cudaUpchannelizer_chord_U64::~cudaUpchannelizer_chord_U64() {}

cudaEvent_t cudaUpchannelizer_chord_U64::execute(cudaPipelineState& /*pipestate*/,
                                                 const std::vector<cudaEvent_t>& /*pre_events*/) {
    const int gpu_frame_index = gpu_frame_id % _gpu_buffer_depth;

    pre_execute();

    Tmin_host.at(gpu_frame_index).resize(Tmin_length);
    void* const Tmin_memory = device.get_gpu_memory(Tmin_memname, Tmin_length);
    Tmax_host.at(gpu_frame_index).resize(Tmax_length);
    void* const Tmax_memory = device.get_gpu_memory(Tmax_memname, Tmax_length);
    Tbarmin_host.at(gpu_frame_index).resize(Tbarmin_length);
    void* const Tbarmin_memory = device.get_gpu_memory(Tbarmin_memname, Tbarmin_length);
    Tbarmax_host.at(gpu_frame_index).resize(Tbarmax_length);
    void* const Tbarmax_memory = device.get_gpu_memory(Tbarmax_memname, Tbarmax_length);
    void* const G_memory =
        args::G == args::E || args::G == args::Ebar
            ? device.get_gpu_memory_array(G_memname, gpu_frame_id, _gpu_buffer_depth,
                                          G_length / _gpu_buffer_depth)
            : device.get_gpu_memory_array(G_memname, gpu_frame_id, _gpu_buffer_depth, G_length);
    void* const E_memory =
        args::E == args::E || args::E == args::Ebar
            ? device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth,
                                          E_length / _gpu_buffer_depth)
            : device.get_gpu_memory_array(E_memname, gpu_frame_id, _gpu_buffer_depth, E_length);
    void* const Ebar_memory =
        args::Ebar == args::E || args::Ebar == args::Ebar
            ? device.get_gpu_memory_array(Ebar_memname, gpu_frame_id, _gpu_buffer_depth,
                                          Ebar_length / _gpu_buffer_depth)
            : device.get_gpu_memory_array(Ebar_memname, gpu_frame_id, _gpu_buffer_depth,
                                          Ebar_length);
    info_host.at(gpu_frame_index).resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // G is an input buffer: check metadata
    const metadataContainer* const G_mc =
        device.get_gpu_memory_array_metadata(G_memname, gpu_frame_id);
    assert(G_mc && metadata_container_is_chord(G_mc));
    const chordMetadata* const G_meta = get_chord_metadata(G_mc);
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
    const metadataContainer* const E_mc =
        device.get_gpu_memory_array_metadata(E_memname, gpu_frame_id);
    assert(E_mc && metadata_container_is_chord(E_mc));
    const chordMetadata* const E_meta = get_chord_metadata(E_mc);
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
    // Ebar is an output buffer: set metadata
    metadataContainer* const Ebar_mc =
        device.create_gpu_memory_array_metadata(Ebar_memname, gpu_frame_id, E_mc->parent_pool);
    chordMetadata* const Ebar_meta = get_chord_metadata(Ebar_mc);
    chord_metadata_copy(Ebar_meta, E_meta);
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

    // We need to (re-)process some time samples, from the previous
    // iteration (`unprocessed`), and we still to provide some time
    // samples that were not provided in the previous iteration
    // (`unprovided`).

    // Although we told Kotekan in the last iteration that we wouldn't
    // need these inputs again, it's fine to look at them again
    // because they won't be overwritten yet, if the GPU buffer depth
    // is large enough.

    // We need an overlap of this many time samples, i.e. this many
    // time samples will need to be re-processed in the next
    // iteration. This also defines the number of time samples that
    // will be missing from the output.
    INFO("cuda_algorithm_overlap: {}", cuda_algorithm_overlap);
    // The input is processed in batches of this size. This means that
    // the input we provide must be a multiple of this number.
    INFO("cuda_granularity_number_of_timesamples: {}", cuda_granularity_number_of_timesamples);

    INFO("gpu_frame_id: {}", gpu_frame_id);

    // Beginning of the input ringbuffer
    void* const E_memory0 =
        device.get_gpu_memory_array(E_memname, 0, _gpu_buffer_depth, E_length / _gpu_buffer_depth);
    INFO("E_memory0: {}", E_memory0);

    // Beginning of the output ringbuffer
    void* const Ebar_memory0 = device.get_gpu_memory_array(Ebar_memname, 0, _gpu_buffer_depth,
                                                           Ebar_length / _gpu_buffer_depth);
    INFO("Ebar_memory0: {}", Ebar_memory0);

    // Set E_memory to beginning of input ring buffer
    E_arg = kernel_arg(E_memory0, E_length);

    // Set Ebar_memory to beginning of output ring buffer
    Ebar_arg = kernel_arg(Ebar_memory0, Ebar_length);

    // Current nominal index into input ringuffer
    const std::int64_t nominal_Tmin =
        gpu_frame_id * cuda_max_number_of_timesamples / _gpu_buffer_depth;
    INFO("nominal Tmin: {}", nominal_Tmin);

    // Current nominal index into output ringuffer
    const std::int64_t nominal_Tbarmin = gpu_frame_id * cuda_max_number_of_timesamples
                                         / cuda_upchannelization_factor / _gpu_buffer_depth;
    INFO("nominal Tbarmin: {}", nominal_Tbarmin);

    // Current unprocessed time samples
    INFO("unprocessed: {}", unprocessed);

    // Current unprovided time samples
    INFO("unprovided: {}", unprovided);

    // Actual index into input ringbuffer
    const std::int64_t Tmin = nominal_Tmin - unprocessed;
    INFO("Tmin: {}", Tmin);

    // Actual index into output ringbuffer
    const std::int64_t Tbarmin = nominal_Tbarmin - unprovided;
    INFO("Tbarmin: {}", Tbarmin);

    // Nominal end of input time span (given by available data)
    const std::int64_t nominal_Tmax = nominal_Tmin + E_meta->dim[0];
    INFO("nominal Tmax: {}", nominal_Tmax);

    // We cannot process all time samples because the input size needs
    // to be a multiple of `cuda_granularity_number_of_timesamples`.
    const std::int64_t nominal_Tlength = nominal_Tmax - Tmin;
    INFO("nominal Tlength: {}", nominal_Tlength);
    const std::int64_t Tlength =
        round_down(nominal_Tlength, cuda_granularity_number_of_timesamples);
    INFO("Tlength: {}", Tlength);

    // End of input time span
    const std::int64_t Tmax = Tmin + Tlength;
    INFO("Tmax: {}", Tmax);

    // Output time span (defined by input time span length)
    assert(Tlength % cuda_upchannelization_factor == 0);
    assert(cuda_algorithm_overlap % cuda_upchannelization_factor == 0);
    assert(Tlength >= cuda_algorithm_overlap);
    const std::int64_t Tbarlength =
        (Tlength - cuda_algorithm_overlap) / cuda_upchannelization_factor;
    INFO("Tbarlength: {}", Tbarlength);

    // End of output time span
    const std::int64_t Tbarmax = Tbarmin + Tbarlength;
    INFO("Tbarmax: {}", Tbarmax);

    // Wrap lower bounds into ringbuffer
    const std::int64_t Tmin_wrapped = mod(Tmin, cuda_max_number_of_timesamples);
    const std::int64_t Tmax_wrapped = Tmin_wrapped + Tmax - Tmin;
    const std::int64_t Tbarmin_wrapped =
        mod(Tbarmin, cuda_max_number_of_timesamples / cuda_upchannelization_factor);
    const std::int64_t Tbarmax_wrapped = Tbarmin_wrapped + Tbarmax - Tbarmin;
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
    *(std::int32_t*)Tmin_host.at(gpu_frame_index).data() = Tmin_wrapped;
    *(std::int32_t*)Tmax_host.at(gpu_frame_index).data() = Tmax_wrapped;
    *(std::int32_t*)Tbarmin_host.at(gpu_frame_index).data() = Tbarmin_wrapped;
    *(std::int32_t*)Tbarmax_host.at(gpu_frame_index).data() = Tbarmax_wrapped;

    // Update metadata
    Ebar_meta->dim[0] = Tbarlength - unprovided;
    assert(Ebar_meta->dim[0] <= int(Ebar_lengths[3]));

    // Calculate the number of  unprocessed time samples for the next iteration
    unprocessed -=
        Tlength - cuda_algorithm_overlap - cuda_max_number_of_timesamples / _gpu_buffer_depth;
    unprovided -=
        Tbarlength
        - cuda_max_number_of_timesamples / cuda_upchannelization_factor / _gpu_buffer_depth;
    INFO("new unprocessed: {}", unprocessed);
    INFO("new unprovided: {}", unprovided);
    assert(unprocessed >= 0 && unprocessed < cuda_max_number_of_timesamples / _gpu_buffer_depth);
    assert(unprovided >= 0
           && unprovided < cuda_max_number_of_timesamples / cuda_upchannelization_factor
                               / _gpu_buffer_depth);

    // Copy inputs to device memory
    // TODO: Pass scalar kernel arguments more efficiently, i.e. without a separate `cudaMemcpy`
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tmin_memory, Tmin_host.at(gpu_frame_index).data(), Tmin_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tmax_memory, Tmax_host.at(gpu_frame_index).data(), Tmax_length,
                                     cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tbarmin_memory, Tbarmin_host.at(gpu_frame_index).data(),
                                     Tbarmin_length, cudaMemcpyHostToDevice,
                                     device.getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(Tbarmax_memory, Tbarmax_host.at(gpu_frame_index).data(),
                                     Tbarmax_length, cudaMemcpyHostToDevice,
                                     device.getStream(cuda_stream_id)));

    // Initialize host-side buffer arrays
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));

    // // Initialize outputs
    // //     0x88 = (-8,-8), an unused value to detect uninitialized output
    // // TODO: Skip this for performance
    // CHECK_CUDA_ERROR(cudaMemsetAsync(Ebar_memory, 0x88, Ebar_length,
    // device.getStream(cuda_stream_id)));

    DEBUG("kernel_symbol: {}", kernel_symbol);
    DEBUG("runtime_kernels[kernel_symbol]: {}",
          static_cast<void*>(device.runtime_kernels[kernel_symbol]));
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[kernel_symbol],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA Upchannelizer_chord_U64 on GPU frame {:d}", gpu_frame_id);
    const CUresult err =
        cuLaunchKernel(device.runtime_kernels[kernel_symbol], blocks, 1, 1, threads_x, threads_y, 1,
                       shmem_bytes, device.getStream(cuda_stream_id), args, NULL);

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        ERROR("cuLaunchKernel: Error number: {}: {}", err, errStr);
    }

    // Copy results back to host memory
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(cudaMemcpyAsync(info_host.at(gpu_frame_index).data(), info_memory, info_length,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

    // Check error codes
    // TODO: Skip this for performance
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
    const std::int32_t error_code =
        *std::max_element((const std::int32_t*)&*info_host.at(gpu_frame_index).begin(),
                          (const std::int32_t*)&*info_host.at(gpu_frame_index).end());
    if (error_code != 0)
        ERROR("CUDA kernel returned error code cuLaunchKernel: {}", error_code);

    for (std::size_t i = 0; i < info_host.at(gpu_frame_index).size(); ++i)
        if (info_host.at(gpu_frame_index)[i] != 0)
            ERROR("cudaUpchannelizer_chord_U64 returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host.at(gpu_frame_index)[i], i);

    return record_end_event();
}

void cudaUpchannelizer_chord_U64::finalize_frame() {
    device.release_gpu_memory_array_metadata(G_memname, gpu_frame_id);
    device.release_gpu_memory_array_metadata(E_memname, gpu_frame_id);
    device.release_gpu_memory_array_metadata(Ebar_memname, gpu_frame_id);

    cudaCommand::finalize_frame();
}
