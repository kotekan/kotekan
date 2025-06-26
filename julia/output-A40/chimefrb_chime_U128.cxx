/**
 * @file
 * @brief CUDA CHIMEFRBBeamformer_chime_U128 kernel
 *
 * This file has been generated automatically.
 * Do not modify this C++ file, your changes will be lost.
 */

#include <DataType.hpp>
#include <NDArrayBuffer.hpp>
#include <NDArrayRingBuffer.hpp>
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
 * @class cudaCHIMEFRBBeamformer_chime_U128
 * @brief cudaCommand for CHIMEFRBBeamformer_chime_U128
 */
class cudaCHIMEFRBBeamformer_chime_U128 : public cudaCommand {
public:
    cudaCHIMEFRBBeamformer_chime_U128(Config& config, const std::string& unique_name,
                                      bufferContainer& host_buffers, cudaDeviceInterface& device,
                                      const int instance_num);
    virtual ~cudaCHIMEFRBBeamformer_chime_U128();

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
    using array_desc = CuDeviceArray<std::int32_t, 1>;

    // Kernel design parameters:
    static constexpr int cuda_beam_layout_M = 512;
    static constexpr int cuda_beam_layout_N = 8;
    static constexpr int cuda_dish_layout_M = 256;
    static constexpr int cuda_dish_layout_N = 4;
    static constexpr int cuda_downsampling_factor = 3;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 1024;
    static constexpr int cuda_number_of_frequencies = 2048;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 512;
    static constexpr int cuda_granularity_number_of_timesamples = 3;

    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

    // Kernel compile parameters:
    static constexpr int minthreads = 256;
    static constexpr int blocks_per_sm = 2;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 8;
    static constexpr int num_blocks = 2048;
    static constexpr int shmem_bytes = 16384;

    // Kernel name:
    static constexpr const char* kernel_symbol =
        "_Z8chimefrb5Int32S_S_S_13CuDeviceArrayI9Float16x2Li1ELi1EES0_I6Int4x8Li1ELi1EES2_S0_IS_"
        "Li1ELi1EE";

    // Kernel arguments:
    enum class args { Tbarmin, Tbarmax, Ttildemin, Ttildemax, W, Ebar, I, info, count };

    // Tbarmin: Tbarmin
    static constexpr const char* Tbarmin_quantity = "Tbarmin";
    static constexpr kotekan::DataType Tbarmin_type = kotekan::int32;
    //
    // Tbarmax: Tbarmax
    static constexpr const char* Tbarmax_quantity = "Tbarmax";
    static constexpr kotekan::DataType Tbarmax_type = kotekan::int32;
    //
    // Ttildemin: Ttildemin
    static constexpr const char* Ttildemin_quantity = "Ttildemin";
    static constexpr kotekan::DataType Ttildemin_type = kotekan::int32;
    //
    // Ttildemax: Ttildemax
    static constexpr const char* Ttildemax_quantity = "Ttildemax";
    static constexpr kotekan::DataType Ttildemax_type = kotekan::int32;
    //
    // W: frb_phase_name
    static constexpr const char* W_quantity = "W";
    static constexpr kotekan::DataType W_type = kotekan::float16;
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
    static constexpr std::array<std::ptrdiff_t, W_rank> W_lengths = {
        2, 256, 4, 2, 2048,
    };
    static constexpr std::ptrdiff_t W_length = type_total_bytes(W_type) * 2 * 256 * 4 * 2 * 2048;
    // static_assert(W_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    static constexpr auto W_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= W_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, W_rank + 1> W_strides = {
        W_calc_stride(W_index_C), W_calc_stride(W_index_dishM), W_calc_stride(W_index_dishN),
        W_calc_stride(W_index_P), W_calc_stride(W_index_F),     W_calc_stride(W_rank),
    };
    static_assert(W_length == type_total_bytes(W_type) * W_strides[W_rank]);
    //
    // Ebar: voltage_name
    static constexpr const char* Ebar_quantity = "Ebar";
    static constexpr kotekan::DataType Ebar_type = kotekan::int4x2chime;
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
        2048,
        512,
    };
    static constexpr std::ptrdiff_t Ebar_length =
        type_total_bytes(Ebar_type) * 1024 * 2 * 2048 * 512;
    // static_assert(Ebar_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    static_assert(Ebar_length == type_total_bytes(Ebar_type) * Ebar_strides[Ebar_rank]);
    //
    // I: frb_beamgrid_name
    static constexpr const char* I_quantity = "I";
    static constexpr kotekan::DataType I_type = kotekan::float16;
    enum I_indices {
        I_index_beamP,
        I_index_beamQ,
        I_index_Fbar,
        I_index_Ttilde,
        I_rank,
    };
    static constexpr std::array<const char*, I_rank> I_labels = {
        "beamP",
        "beamQ",
        "Fbar",
        "Ttilde",
    };
    static constexpr std::array<std::ptrdiff_t, I_rank> I_lengths = {
        512,
        8,
        2048,
        256,
    };
    static constexpr std::ptrdiff_t I_length = type_total_bytes(I_type) * 512 * 8 * 2048 * 256;
    // static_assert(I_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    static constexpr auto I_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= I_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, I_rank + 1> I_strides = {
        I_calc_stride(I_index_beamP),  I_calc_stride(I_index_beamQ), I_calc_stride(I_index_Fbar),
        I_calc_stride(I_index_Ttilde), I_calc_stride(I_rank),
    };
    static_assert(I_length == type_total_bytes(I_type) * I_strides[I_rank]);
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
        8,
        2048,
    };
    static constexpr std::ptrdiff_t info_length = type_total_bytes(info_type) * 32 * 8 * 2048;
    // static_assert(info_length <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
    static_assert(info_length == type_total_bytes(info_type) * info_strides[info_rank]);
    //

    // Kotekan buffer names
    const std::string W_name;
    const std::string Ebar_name;
    const std::string I_name;
    const std::string info_name;

    // Host-side buffer arrays
    std::vector<std::uint8_t> info_host;

    static constexpr std::ptrdiff_t Ebar_Tbar_sample_bytes =
        type_total_bytes(Ebar_type) * Ebar_lengths[Ebar_index_D] * Ebar_lengths[Ebar_index_P]
        * Ebar_lengths[Ebar_index_Fbar];
    static constexpr std::ptrdiff_t I_Ttilde_sample_bytes =
        type_total_bytes(I_type) * I_lengths[I_index_beamP] * I_lengths[I_index_beamQ]
        * I_lengths[I_index_Fbar];

    RingBuffer* const input_ringbuf_signal;
    RingBuffer* const output_ringbuf_signal;
    // NDArrayBuffer<kotekan::GetType_t<W_type>, W_rank> W_buffer;
    // TODO: NDArrayRingBuffer<kotekan::GetType_t<Ebar_type>, Ebar_rank> Ebar_buffer;
    // TODO: NDArrayRingBuffer<kotekan::GetType_t<I_type>, I_rank> I_buffer;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_on_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Tbarmin, Tbarmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_on_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t Ttildemin, Ttildemax;
};

REGISTER_CUDA_COMMAND(cudaCHIMEFRBBeamformer_chime_U128);

cudaCHIMEFRBBeamformer_chime_U128::cudaCHIMEFRBBeamformer_chime_U128(Config& config,
                                                                     const std::string& unique_name,
                                                                     bufferContainer& host_buffers,
                                                                     cudaDeviceInterface& device,
                                                                     const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "CHIMEFRBBeamformer_chime_U128", "CHIMEFRBBeamformer_chime_U128.ptx"),
    W_name(config.get<std::string>(unique_name, "frb_phase_name")),
    Ebar_name(config.get<std::string>(unique_name, "voltage_name")),
    I_name(config.get<std::string>(unique_name, "frb_beamgrid_name")),
    info_name(unique_name + "/gpu_mem_info"),

    info_host(info_length),
    // Find input and output buffers used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(host_buffers.get_generic_buffer(
        config.get<std::string>(unique_name, "voltage_signal_in")))),
    output_ringbuf_signal(dynamic_cast<RingBuffer*>(host_buffers.get_generic_buffer(
        config.get<std::string>(unique_name, "beamgrid_signal_out")))),

    // W_buffer(
    //     W_name, W_quantity, reverse(W_lengths), reverse(W_labels), *this),
    // Ebar_buffer(
    //     Ebar_name, Ebar_quantity, reverse(Ebar_lengths), reverse(Ebar_labels), *this),
    // I_buffer(
    //     I_name, I_quantity, reverse(I_lengths), reverse(I_labels), *this),

    Tbarmin() // avoid trailing comma
{
    // Check ringbuffer sizes
    if (!(input_ringbuf_signal->size == Ebar_length))
        FATAL_ERROR("Need input_ringbuf_signal->size == Ebar_length, but have "
                    "input_ringbuf_signal->size={:d}, Ebar_length={:d}",
                    input_ringbuf_signal->size, Ebar_length);
    if (!(output_ringbuf_signal->size == I_length))
        FATAL_ERROR("Need output_ringbuf_signal->size == I_length, but have "
                    "output_ringbuf_signal->size={:d}, I_length={:d}",
                    output_ringbuf_signal->size, I_length);
    assert(input_ringbuf_signal->size == Ebar_length);
    assert(output_ringbuf_signal->size == I_length);

    // Register host memory
    {
        const cudaError_t ierr = cudaHostRegister(info_host.data(), info_host.size(), 0);
        assert(ierr == cudaSuccess);
    }

    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(W_name, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(Ebar_name, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(I_name, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage need to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "CHIMEFRBBeamformer_chime_U128_");
    }

    if (instance_num == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
        output_ringbuf_signal->register_producer(unique_name);
        output_ringbuf_signal->allocate_new_metadata_object(0);
    }
}

cudaCHIMEFRBBeamformer_chime_U128::~cudaCHIMEFRBBeamformer_chime_U128() {}

std::int64_t cudaCHIMEFRBBeamformer_chime_U128::num_consumed_elements(
    std::int64_t num_available_elements) const {
    return num_produced_elements(num_available_elements) * cuda_downsampling_factor;
}
std::int64_t cudaCHIMEFRBBeamformer_chime_U128::num_produced_elements(
    std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) / cuda_downsampling_factor;
}

std::int64_t cudaCHIMEFRBBeamformer_chime_U128::num_processed_elements(
    std::int64_t num_available_elements) const {
    if (num_available_elements < cuda_granularity_number_of_timesamples)
        return 0;
    assert(num_available_elements >= cuda_granularity_number_of_timesamples);
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cudaCHIMEFRBBeamformer_chime_U128::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer

    // Check available samples
    DEBUG("Checking available input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::pair<std::ptrdiff_t, std::ptrdiff_t>> peeked =
        input_ringbuf_signal->peek_readable(unique_name, instance_num);
    if (!peeked.has_value())
        return -1;
    std::ptrdiff_t input_bytes = peeked.value().second;
    DEBUG("Input ring-buffer byte count: {:d}", input_bytes);

wait_for_data:

    // How many inputs samples are available?
    const std::ptrdiff_t Tbar_available = div_noremainder(input_bytes, Ebar_Tbar_sample_bytes);
    DEBUG("Available samples:      Tbar_available: {:d}", Tbar_available);

    // How many inputs will we process and consume?
    const std::ptrdiff_t Tbar_processed = num_processed_elements(Tbar_available);
    const std::ptrdiff_t Tbar_consumed = num_consumed_elements(Tbar_available);
    DEBUG("Will process (samples): Tbar_processed: {:d}", Tbar_processed);
    DEBUG("Will consume (samples): Tbar_consumed:  {:d}", Tbar_consumed);
    assert(Tbar_consumed <= Tbar_processed);
    const std::ptrdiff_t Tbar_consumed2 = num_consumed_elements(Tbar_processed);
    assert(Tbar_consumed2 == Tbar_consumed);

    // Can we make progress?
    if (Tbar_consumed <= 0) {
        // We cannot make progress, we need to wait
        DEBUG("We cannot make progress, we need to wait for more input");

        DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
        const std::optional<std::ptrdiff_t> waited =
            input_ringbuf_signal->wait_without_claiming(unique_name, instance_num, input_bytes + 1);
        DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
        if (!waited.has_value())
            return -1;
        input_bytes = waited.value();
        DEBUG("Input ring-buffer byte count: {:d}", input_bytes);

        goto wait_for_data;
    }

    // Claim inputs
    assert(Tbar_consumed > 0);
    const std::optional<std::ptrdiff_t> claimed = input_ringbuf_signal->wait_and_claim_readable(
        unique_name, instance_num, Tbar_consumed * Ebar_Tbar_sample_bytes);
    if (!claimed.has_value())
        return -1;
    const std::ptrdiff_t input_cursor = claimed.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    Tbarmin = div_noremainder(input_cursor, Ebar_Tbar_sample_bytes);
    Tbarmax = Tbarmin + Tbar_processed;
    const std::ptrdiff_t Tbarlength = Tbarmax - Tbarmin;
    DEBUG("Input samples:");
    DEBUG("    Tbarmin:    {:d}", Tbarmin);
    DEBUG("    Tbarmax:    {:d}", Tbarmax);
    DEBUG("    Tbarlength: {:d}", Tbarlength);

    // How many outputs will we produce?
    const std::ptrdiff_t Ttilde_produced = num_produced_elements(Tbar_available);
    DEBUG("Will produce (samples): Ttilde_produced: {:d}", Ttilde_produced);
    const std::ptrdiff_t Ttildelength = Ttilde_produced;

    // to bytes
    const std::ptrdiff_t output_bytes = Ttildelength * I_Ttilde_sample_bytes;
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

    assert(mod(output_cursor, I_Ttilde_sample_bytes) == 0);
    Ttildemin = output_cursor / I_Ttilde_sample_bytes;
    Ttildemax = Ttildemin + Ttildelength;
    DEBUG("Output samples:");
    DEBUG("    Ttildemin:    {:d}", Ttildemin);
    DEBUG("    Ttildemax:    {:d}", Ttildemax);
    DEBUG("    Ttildelength: {:d}", Ttildelength);

    return 0;
}

cudaEvent_t
cudaCHIMEFRBBeamformer_chime_U128::execute(cudaPipelineState& /*pipestate*/,
                                           const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    const std::string W_memname = W_name + "_buffer";
    void* const W_memory =
        args::W == args::Ebar ? device.get_gpu_memory(W_memname, input_ringbuf_signal->size)
        : args::W == args::I  ? device.get_gpu_memory(W_memname, output_ringbuf_signal->size)
        : args::W == args::W
            ? device.get_gpu_memory(W_memname, W_length)
            : device.get_gpu_memory_array(W_memname, gpu_frame_id, _gpu_buffer_depth, W_length);
    const std::string Ebar_memname = Ebar_name + "_buffer";
    void* const Ebar_memory =
        args::Ebar == args::Ebar ? device.get_gpu_memory(Ebar_memname, input_ringbuf_signal->size)
        : args::Ebar == args::I  ? device.get_gpu_memory(Ebar_memname, output_ringbuf_signal->size)
        : args::Ebar == args::W  ? device.get_gpu_memory(Ebar_memname, Ebar_length)
                                 : device.get_gpu_memory_array(Ebar_memname, gpu_frame_id,
                                                               _gpu_buffer_depth, Ebar_length);
    const std::string I_memname = I_name + "_buffer";
    void* const I_memory =
        args::I == args::Ebar ? device.get_gpu_memory(I_memname, input_ringbuf_signal->size)
        : args::I == args::I  ? device.get_gpu_memory(I_memname, output_ringbuf_signal->size)
        : args::I == args::W
            ? device.get_gpu_memory(I_memname, I_length)
            : device.get_gpu_memory_array(I_memname, gpu_frame_id, _gpu_buffer_depth, I_length);
    const std::string info_memname = info_name + "_buffer";
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // W is an input buffer: check metadata
    const std::shared_ptr<metadataObject> W_mc =
        args::W == args::Ebar ? input_ringbuf_signal->get_metadata(0)
                              : device.get_gpu_memory_array_metadata(W_memname, gpu_frame_id);
    assert(W_mc);
    assert(metadata_is_chord(W_mc));
    const std::shared_ptr<chordMetadata> W_meta = get_chord_metadata(W_mc);
    DEBUG("input W array: {:s} {:s}", W_meta->get_type_string(), W_meta->get_dimensions_string());
    if (args::W == args::Ebar && 128 == 1) {
        // Replace "Ebar_U1" with "E" etc. because we don't run the upchannelizer for U=1
        assert(std::strncmp(W_meta->name, "E", sizeof W_meta->name) == 0);
        assert(W_meta->type == W_type);
        assert(W_meta->dims == 4);
        assert(std::strncmp(W_meta->dim_name[3], "D", sizeof W_meta->dim_name[3]) == 0);
        assert(W_meta->dim[3] == int(Ebar_lengths[0]));
        assert(W_meta->stride[3] == Ebar_strides[0]);
        assert(std::strncmp(W_meta->dim_name[2], "P", sizeof W_meta->dim_name[2]) == 0);
        assert(W_meta->dim[2] == int(Ebar_lengths[1]));
        assert(W_meta->stride[2] == Ebar_strides[1]);
        assert(std::strncmp(W_meta->dim_name[1], "F", sizeof W_meta->dim_name[1]) == 0);
        assert(W_meta->dim[1] == int(Ebar_lengths[2]));
        assert(W_meta->stride[1] == Ebar_strides[2]);
        assert(std::strncmp(W_meta->dim_name[0], "T", sizeof W_meta->dim_name[0]) == 0);
        assert(W_meta->dim[0] <= int(Ebar_lengths[3]));
        assert(W_meta->stride[0] == Ebar_strides[3]);
    } else {
        assert(std::strncmp(W_meta->name, W_quantity, sizeof W_meta->name) == 0);
        assert(W_meta->type == W_type);
        assert(W_meta->dims == W_rank);
        for (std::ptrdiff_t dim = 0; dim < W_rank; ++dim) {
            assert(std::strncmp(W_meta->dim_name[W_rank - 1 - dim], W_labels[dim],
                                sizeof W_meta->dim_name[W_rank - 1 - dim])
                   == 0);
            if ((args::W == args::Ebar && dim == Ebar_rank - 1)
                || (args::W == args::W && dim == W_rank - 1)) {
                assert(W_meta->dim[W_rank - 1 - dim] <= int(W_lengths[dim]));
                assert(W_meta->stride[W_rank - 1 - dim] == W_strides[dim]);
            } else {
                assert(W_meta->dim[W_rank - 1 - dim] == int(W_lengths[dim]));
                assert(W_meta->stride[W_rank - 1 - dim] == W_strides[dim]);
            }
        }
    }
    //
    // Ebar is an input buffer: check metadata
    const std::shared_ptr<metadataObject> Ebar_mc =
        args::Ebar == args::Ebar ? input_ringbuf_signal->get_metadata(0)
                                 : device.get_gpu_memory_array_metadata(Ebar_memname, gpu_frame_id);
    assert(Ebar_mc);
    assert(metadata_is_chord(Ebar_mc));
    const std::shared_ptr<chordMetadata> Ebar_meta = get_chord_metadata(Ebar_mc);
    DEBUG("input Ebar array: {:s} {:s}", Ebar_meta->get_type_string(),
          Ebar_meta->get_dimensions_string());
    if (args::Ebar == args::Ebar && 128 == 1) {
        // Replace "Ebar_U1" with "E" etc. because we don't run the upchannelizer for U=1
        assert(std::strncmp(Ebar_meta->name, "E", sizeof Ebar_meta->name) == 0);
        assert(Ebar_meta->type == Ebar_type);
        assert(Ebar_meta->dims == 4);
        assert(std::strncmp(Ebar_meta->dim_name[3], "D", sizeof Ebar_meta->dim_name[3]) == 0);
        assert(Ebar_meta->dim[3] == int(Ebar_lengths[0]));
        assert(Ebar_meta->stride[3] == Ebar_strides[0]);
        assert(std::strncmp(Ebar_meta->dim_name[2], "P", sizeof Ebar_meta->dim_name[2]) == 0);
        assert(Ebar_meta->dim[2] == int(Ebar_lengths[1]));
        assert(Ebar_meta->stride[2] == Ebar_strides[1]);
        assert(std::strncmp(Ebar_meta->dim_name[1], "F", sizeof Ebar_meta->dim_name[1]) == 0);
        assert(Ebar_meta->dim[1] == int(Ebar_lengths[2]));
        assert(Ebar_meta->stride[1] == Ebar_strides[2]);
        assert(std::strncmp(Ebar_meta->dim_name[0], "T", sizeof Ebar_meta->dim_name[0]) == 0);
        assert(Ebar_meta->dim[0] <= int(Ebar_lengths[3]));
        assert(Ebar_meta->stride[0] == Ebar_strides[3]);
    } else {
        assert(std::strncmp(Ebar_meta->name, Ebar_quantity, sizeof Ebar_meta->name) == 0);
        assert(Ebar_meta->type == Ebar_type);
        assert(Ebar_meta->dims == Ebar_rank);
        for (std::ptrdiff_t dim = 0; dim < Ebar_rank; ++dim) {
            assert(std::strncmp(Ebar_meta->dim_name[Ebar_rank - 1 - dim], Ebar_labels[dim],
                                sizeof Ebar_meta->dim_name[Ebar_rank - 1 - dim])
                   == 0);
            if ((args::Ebar == args::Ebar && dim == Ebar_rank - 1)
                || (args::Ebar == args::W && dim == W_rank - 1)) {
                assert(Ebar_meta->dim[Ebar_rank - 1 - dim] <= int(Ebar_lengths[dim]));
                assert(Ebar_meta->stride[Ebar_rank - 1 - dim] == Ebar_strides[dim]);
            } else {
                assert(Ebar_meta->dim[Ebar_rank - 1 - dim] == int(Ebar_lengths[dim]));
                assert(Ebar_meta->stride[Ebar_rank - 1 - dim] == Ebar_strides[dim]);
            }
        }
    }
    //
    // I is an output buffer: set metadata
    std::shared_ptr<metadataObject> const I_mc =
        args::I == args::I ? output_ringbuf_signal->get_metadata(0)
                           : device.create_gpu_memory_array_metadata(I_memname, gpu_frame_id,
                                                                     Ebar_mc->parent_pool);
    std::shared_ptr<chordMetadata> const I_meta = get_chord_metadata(I_mc);
    *I_meta = *Ebar_meta;
    std::strncpy(I_meta->name, I_quantity, sizeof I_meta->name);
    I_meta->type = I_type;
    I_meta->dims = I_rank;
    for (std::ptrdiff_t dim = 0; dim < I_rank; ++dim) {
        std::strncpy(I_meta->dim_name[I_rank - 1 - dim], I_labels[dim],
                     sizeof I_meta->dim_name[I_rank - 1 - dim]);
        I_meta->dim[I_rank - 1 - dim] = I_lengths[dim];
        I_meta->stride[I_rank - 1 - dim] = I_strides[dim];
    }
    DEBUG("output I array: {:s} {:s}", I_meta->get_type_string(), I_meta->get_dimensions_string());
    //

    assert(Ebar_meta->ndishes == cuda_number_of_dishes);
    assert(Ebar_meta->n_dish_locations_ew == cuda_dish_layout_N);
    assert(Ebar_meta->n_dish_locations_ns == cuda_dish_layout_M);
    assert(Ebar_meta->dish_index);

    record_start_event();

    DEBUG("gpu_frame_id: {}", gpu_frame_id);

    const char* exc_arg = "exception";
    std::int32_t Tbarmin_arg;
    std::int32_t Tbarmax_arg;
    std::int32_t Ttildemin_arg;
    std::int32_t Ttildemax_arg;
    array_desc W_arg(W_memory, W_length);
    array_desc Ebar_arg(Ebar_memory, Ebar_length);
    array_desc I_arg(I_memory, I_length);
    array_desc info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &Tbarmin_arg, &Tbarmax_arg, &Ttildemin_arg, &Ttildemax_arg,
        &W_arg,   &Ebar_arg,    &I_arg,       &info_arg,
    };

    // Set Ebar_memory to beginning of input ring buffer
    Ebar_arg = array_desc(Ebar_memory, Ebar_length);

    // Set I_memory to beginning of output ring buffer
    I_arg = array_desc(I_memory, I_length);

    // Ringbuffer size
    const std::ptrdiff_t Tbar_ringbuf = input_ringbuf_signal->size / Ebar_Tbar_sample_bytes;
    const std::ptrdiff_t Ttilde_ringbuf = output_ringbuf_signal->size / I_Ttilde_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", Tbar_ringbuf);
    DEBUG("Output ringbuffer size (samples): {:d}", Ttilde_ringbuf);

    const std::ptrdiff_t Tbarlength = Tbarmax - Tbarmin;
    const std::ptrdiff_t Ttildelength = Ttildemax - Ttildemin;
    DEBUG("Processed input samples: {:d}", Tbarlength);
    DEBUG("Produced output samples: {:d}", Ttildelength);

    DEBUG("Kernel arguments:");
    DEBUG("    Tbarmin:   {:d}", Tbarmin);
    DEBUG("    Tbarmax:   {:d}", Tbarmax);
    DEBUG("    Ttildemin: {:d}", Ttildemin);
    DEBUG("    Ttildemax: {:d}", Ttildemax);

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    Tbarmin_arg = mod(Tbarmin, Tbar_ringbuf);
    Tbarmax_arg = mod(Tbarmin, Tbar_ringbuf) + Tbarlength;
    Ttildemin_arg = mod(Ttildemin, Ttilde_ringbuf);
    Ttildemax_arg = mod(Ttildemin, Ttilde_ringbuf) + Ttildelength;

    // Update metadata
    I_meta->dim[I_rank - 1 - I_index_Ttilde] = Ttildelength;
    assert(I_meta->dim[I_rank - 1 - I_index_Ttilde] <= int(I_lengths[I_index_Ttilde]));
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    assert(I_meta->nfreq >= 0);
    assert(I_meta->nfreq == Ebar_meta->nfreq);
    for (int freq = 0; freq < I_meta->nfreq; ++freq) {
        I_meta->freq_upchan_factor[freq] =
            cuda_downsampling_factor * Ebar_meta->freq_upchan_factor[freq];
        // I_meta->half_fpga_sample0[freq] = Evar_meta->half_fpga_sample0[freq];
        I_meta->time_downsampling_fpga[freq] =
            cuda_downsampling_factor * Ebar_meta->time_downsampling_fpga[freq];
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
        const int num_chunks = Ttildemax_arg <= Ttilde_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            DEBUG("poisoning chunk={}/{}", chunk, num_chunks);
            const std::ptrdiff_t Ttildestride = I_meta->stride[0];
            const std::ptrdiff_t Ttildeoffset = chunk == 0 ? Ttildemin_arg : 0;
            const std::ptrdiff_t Ttildelength = (num_chunks == 1 ? Ttildemax_arg - Ttildemin_arg
                                                 : chunk == 0    ? Ttilde_ringbuf - Ttildemin_arg
                                                                 : Ttildemax_arg - Ttilde_ringbuf);
            CHECK_CUDA_ERROR(
                cudaMemsetAsync((std::uint8_t*)I_memory + 2 * Ttildeoffset * Ttildestride,
                                0xff, // 0xffff is NaN16
                                2 * Ttildelength, device.getStream(cuda_stream_id)));
        } // for chunk
        DEBUG("poisoning done.");
    }
#endif

    const std::string symname = "CHIMEFRBBeamformer_chime_U128_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA CHIMEFRBBeamformer_chime_U128 on GPU frame {:d}", gpu_frame_id);
    const int blocks = num_blocks;
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
    DEBUG("Finished CUDA CHIMEFRBBeamformer_chime_U128 on GPU frame {:d}", gpu_frame_id);

    // Check error codes
    std::uint32_t error_code = 0;
    for (int block = 0; block < blocks; ++block) {
        for (int warp = 0; warp < info_lengths[info_index_warp]; ++warp) {
            for (int thread = 0; thread < info_lengths[info_index_thread]; ++thread) {
                const std::ptrdiff_t i = info_strides[info_index_thread] * thread
                                         + info_strides[info_index_warp] * warp
                                         + info_strides[info_index_block] * block;
                const std::uint32_t val = *(const std::uint32_t*)&info_host[i];
                using std::max;
                error_code = max(error_code, val);
            }
        }
    }
    if (error_code != 0)
        ERROR("CUDA kernel CHIMEFRBBeamformer_chime_U128 returned error code: {}", error_code);

    for (int block = 0; block < blocks; ++block) {
        for (int warp = 0; warp < info_lengths[info_index_warp]; ++warp) {
            for (int thread = 0; thread < info_lengths[info_index_thread]; ++thread) {
                const std::ptrdiff_t i = info_strides[info_index_thread] * thread
                                         + info_strides[info_index_warp] * warp
                                         + info_strides[info_index_block] * block;
                const std::uint32_t val = ((const std::uint32_t*)info_host.data())[i];
                if (val != 0)
                    ERROR("CUDA kernel CHIMEFRBBeamformer_chime_U128 returned 'info' value {:d} "
                          "for thread {:d} warp {:d} block {:d} at index {:d} (zero indicates no "
                          "error)",
                          val, thread, warp, block, i);
            }
        }
    }
#endif

#ifdef DEBUGGING
    // Check outputs for poison
    {
        DEBUG("begin poison check");
        DEBUG("    Ebar_dims={}", Ebar_meta->dims);
        DEBUG("    Ebar_dim[0]={}", Ebar_meta->dim[0]);
        DEBUG("    Ebar_dim[1]={}", Ebar_meta->dim[1]);
        DEBUG("    Ebar_dim[2]={}", Ebar_meta->dim[2]);
        DEBUG("    Ebar_dim[3]={}", Ebar_meta->dim[3]);
        DEBUG("    Ebar_stride[0]={}", Ebar_meta->stride[0]);
        DEBUG("    Ebar_stride[1]={}", Ebar_meta->stride[1]);
        DEBUG("    Ebar_stride[2]={}", Ebar_meta->stride[2]);
        DEBUG("    Ebar_stride[3]={}", Ebar_meta->stride[3]);
        DEBUG("    I_dims={}", I_meta->dims);
        DEBUG("    I_dim[0]={}", I_meta->dim[0]);
        DEBUG("    I_dim[1]={}", I_meta->dim[1]);
        DEBUG("    I_dim[2]={}", I_meta->dim[2]);
        DEBUG("    I_dim[3]={}", I_meta->dim[3]);
        DEBUG("    I_stride[0]={}", I_meta->stride[0]);
        DEBUG("    I_stride[1]={}", I_meta->stride[1]);
        DEBUG("    I_stride[2]={}", I_meta->stride[2]);
        DEBUG("    I_stride[3]={}", I_meta->stride[3]);
        const int num_chunks = Ttildemax_arg <= Ttilde_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            DEBUG("poison checking chunk={}/{}", chunk, num_chunks);
            const std::ptrdiff_t Ttildestride = I_meta->stride[0];
            const std::ptrdiff_t Ttildeoffset = chunk == 0 ? Ttildemin_arg : 0;
            const std::ptrdiff_t Ttildelength = num_chunks == 1 ? Ttildemax_arg - Ttildemin_arg
                                                : chunk == 0    ? Ttilde_ringbuf - Ttildemin_arg
                                                                : Ttildemax_arg - Ttilde_ringbuf;
            DEBUG("    Ttildestride={}", Ttildestride);
            DEBUG("    Ttildeoffset={}", Ttildeoffset);
            DEBUG("    Ttildelength={}", Ttildelength);
            std::vector<std::uint16_t> I_buffer(Ttildelength * Ttildestride, 0xfffe);
            DEBUG("    I_buffer.size={}", I_buffer.size());
            DEBUG("before cudaMemcpy2D.I");
            CHECK_CUDA_ERROR(cudaMemcpy(
                I_buffer.data(), (const std::uint8_t*)I_memory + 2 * Ttildeoffset * Ttildestride,
                2 * Ttildelength * Ttildestride, cudaMemcpyDeviceToHost));

            DEBUG("before memchr");
            bool I_found_error = false;
            for (std::ptrdiff_t ttilde = 0; ttilde < Ttildelength; ++ttilde) {
                bool any_error = false, all_error = true;
                for (std::ptrdiff_t n = 0; n < Ttildestride; ++n) {
                    const auto val = I_buffer.at(ttilde * Ttildestride + n);
                    const bool val_is_finite = (val & 0b0111110000000000) != 0b0111110000000000;
                    const bool val_is_nan = (val & 0b0111110000000000) == 0b0111110000000000
                                            && (val & 0b0000001111111111) != 0b0000000000000000;
                    // if (val_is_nan)
                    //     DEBUG("    U=16 [{},{}]=val={}", ttilde, n, val);
                    any_error |= val_is_nan;
                    all_error &= val_is_nan;
                }
                // if (any_error)
                //     DEBUG("    U=128 [{}]=(any={},all={})",
                //           ttilde, any_error, all_error);
                I_found_error |= any_error;
            }
            if (I_found_error)
                WARN("CUDA kernel CHIMEFRBBeamformer_chime_U128 returned produced non-finite "
                     "results");
        } // for chunk
        DEBUG("poison check done.");
    }
#endif

    return record_end_event();
}

void cudaCHIMEFRBBeamformer_chime_U128::finalize_frame() {
    const std::ptrdiff_t Tbarlength = Tbarmax - Tbarmin;
    const std::ptrdiff_t Ttildelength = Ttildemax - Ttildemin;

    // Advance the input ringbuffer
    const std::ptrdiff_t Tbar_consumed = num_consumed_elements(Tbarlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples: {:d}", Tbar_consumed);
    DEBUG("    Consumed bytes:   {:d}", Tbar_consumed * Ebar_Tbar_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, instance_num,
                                      Tbar_consumed * Ebar_Tbar_sample_bytes);

    // Advance the output ringbuffer
    const std::ptrdiff_t Ttilde_produced = Ttildelength;
    DEBUG("Advancing output ringbuffer:");
    DEBUG("    Produced samples: {:d}", Ttilde_produced);
    DEBUG("    Produced bytes:   {:d}", Ttilde_produced * I_Ttilde_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, instance_num,
                                        Ttilde_produced * I_Ttilde_sample_bytes);

    cudaCommand::finalize_frame();
}
