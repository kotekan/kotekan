/**
 * @file
 * @brief CUDA FRBBeamformer_pathfinder_U4 kernel
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
 * @class cudaFRBBeamformer_pathfinder_U4
 * @brief cudaCommand for FRBBeamformer_pathfinder_U4
 */
class cudaFRBBeamformer_pathfinder_U4 : public cudaCommand {
public:
    cudaFRBBeamformer_pathfinder_U4(Config& config, const std::string& unique_name,
                                    bufferContainer& host_buffers, cudaDeviceInterface& device,
                                    const int instance_num);
    virtual ~cudaFRBBeamformer_pathfinder_U4();

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
    static constexpr int cuda_beam_layout_M = 16;
    static constexpr int cuda_beam_layout_N = 24;
    static constexpr int cuda_dish_layout_M = 8;
    static constexpr int cuda_dish_layout_N = 12;
    static constexpr int cuda_downsampling_factor = 64;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 64;
    static constexpr int cuda_number_of_frequencies = 512;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 8192;
    static constexpr int cuda_granularity_number_of_timesamples = 48;

    // Kernel input and output sizes
    std::int64_t num_consumed_elements(std::int64_t num_available_elements) const;
    std::int64_t num_produced_elements(std::int64_t num_available_elements) const;

    std::int64_t num_processed_elements(std::int64_t num_available_elements) const;

    // Kernel compile parameters:
    static constexpr int minthreads = 192;
    static constexpr int blocks_per_sm = 4;

    // Kernel call parameters:
    static constexpr int threads_x = 32;
    static constexpr int threads_y = 6;
    static constexpr int max_blocks = 512;
    static constexpr int shmem_bytes = 13920;

    // Kernel name:
    static constexpr const char* kernel_symbol =
        "_Z3frb5Int32S_S_S_S_S_S_S_13CuDeviceArrayI7Int16x2Li1ELi1EES0_I9Float16x2Li1ELi1EES0_"
        "I6Int4x8Li1ELi1EES0_IS2_Li1ELi1EES0_IS_Li1ELi1EE";

    // Kernel arguments:
    enum class args {
        Tbarmin,
        Tbarmax,
        Ttildemin,
        Ttildemax,
        Fbarmin,
        Fbarmax,
        Ftildemin,
        Ftildemax,
        S,
        W_U4,
        Ebar_U4,
        I_U4,
        info,
        count
    };

    // Tbarmin: Tbarmin
    static constexpr const char* Tbarmin_name = "Tbarmin";
    static constexpr chordDataType Tbarmin_type = int32;
    //
    // Tbarmax: Tbarmax
    static constexpr const char* Tbarmax_name = "Tbarmax";
    static constexpr chordDataType Tbarmax_type = int32;
    //
    // Ttildemin: Ttildemin
    static constexpr const char* Ttildemin_name = "Ttildemin";
    static constexpr chordDataType Ttildemin_type = int32;
    //
    // Ttildemax: Ttildemax
    static constexpr const char* Ttildemax_name = "Ttildemax";
    static constexpr chordDataType Ttildemax_type = int32;
    //
    // Fbarmin: Fbarmin
    static constexpr const char* Fbarmin_name = "Fbarmin";
    static constexpr chordDataType Fbarmin_type = int32;
    //
    // Fbarmax: Fbarmax
    static constexpr const char* Fbarmax_name = "Fbarmax";
    static constexpr chordDataType Fbarmax_type = int32;
    //
    // Ftildemin: Ftildemin
    static constexpr const char* Ftildemin_name = "Ftildemin";
    static constexpr chordDataType Ftildemin_type = int32;
    //
    // Ftildemax: Ftildemax
    static constexpr const char* Ftildemax_name = "Ftildemax";
    static constexpr chordDataType Ftildemax_type = int32;
    //
    // S: gpu_mem_dishlayout
    static constexpr const char* S_name = "S";
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
        96,
    };
    static constexpr std::size_t S_length = chord_datatype_bytes(S_type) * 2 * 96;
    static_assert(S_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    static constexpr auto S_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= S_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, S_rank + 1> S_strides = {
        S_calc_stride(S_index_MN),
        S_calc_stride(S_index_D),
        S_calc_stride(S_rank),
    };
    static_assert(S_length == chord_datatype_bytes(S_type) * S_strides[S_rank]);
    //
    // W_U4: gpu_mem_phase
    static constexpr const char* W_U4_name = "W_U4";
    static constexpr chordDataType W_U4_type = float16;
    enum W_U4_indices {
        W_U4_index_C,
        W_U4_index_dishM,
        W_U4_index_dishN,
        W_U4_index_P,
        W_U4_index_Fbar_U4,
        W_U4_rank,
    };
    static constexpr std::array<const char*, W_U4_rank> W_U4_labels = {
        "C", "dishM", "dishN", "P", "Fbar_U4",
    };
    static constexpr std::array<std::size_t, W_U4_rank> W_U4_lengths = {
        2, 8, 12, 2, 512,
    };
    static constexpr std::size_t W_U4_length =
        chord_datatype_bytes(W_U4_type) * 2 * 8 * 12 * 2 * 512;
    static_assert(W_U4_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    static constexpr auto W_U4_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= W_U4_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, W_U4_rank + 1> W_U4_strides = {
        W_U4_calc_stride(W_U4_index_C),       W_U4_calc_stride(W_U4_index_dishM),
        W_U4_calc_stride(W_U4_index_dishN),   W_U4_calc_stride(W_U4_index_P),
        W_U4_calc_stride(W_U4_index_Fbar_U4), W_U4_calc_stride(W_U4_rank),
    };
    static_assert(W_U4_length == chord_datatype_bytes(W_U4_type) * W_U4_strides[W_U4_rank]);
    //
    // Ebar_U4: gpu_mem_voltage
    static constexpr const char* Ebar_U4_name = "Ebar_U4";
    static constexpr chordDataType Ebar_U4_type = int4p4;
    enum Ebar_U4_indices {
        Ebar_U4_index_D,
        Ebar_U4_index_P,
        Ebar_U4_index_Fbar_U4,
        Ebar_U4_index_Tbar_U4,
        Ebar_U4_rank,
    };
    static constexpr std::array<const char*, Ebar_U4_rank> Ebar_U4_labels = {
        "D",
        "P",
        "Fbar_U4",
        "Tbar_U4",
    };
    static constexpr std::array<std::size_t, Ebar_U4_rank> Ebar_U4_lengths = {
        64,
        2,
        512,
        8192,
    };
    static constexpr std::size_t Ebar_U4_length =
        chord_datatype_bytes(Ebar_U4_type) * 64 * 2 * 512 * 8192;
    static_assert(Ebar_U4_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    static constexpr auto Ebar_U4_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= Ebar_U4_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, Ebar_U4_rank + 1> Ebar_U4_strides = {
        Ebar_U4_calc_stride(Ebar_U4_index_D),       Ebar_U4_calc_stride(Ebar_U4_index_P),
        Ebar_U4_calc_stride(Ebar_U4_index_Fbar_U4), Ebar_U4_calc_stride(Ebar_U4_index_Tbar_U4),
        Ebar_U4_calc_stride(Ebar_U4_rank),
    };
    static_assert(Ebar_U4_length
                  == chord_datatype_bytes(Ebar_U4_type) * Ebar_U4_strides[Ebar_U4_rank]);
    //
    // I_U4: gpu_mem_beamgrid
    static constexpr const char* I_U4_name = "I_U4";
    static constexpr chordDataType I_U4_type = float16;
    enum I_U4_indices {
        I_U4_index_beamP,
        I_U4_index_beamQ,
        I_U4_index_Fbar_U4,
        I_U4_index_Ttilde_U4_Tds64,
        I_U4_rank,
    };
    static constexpr std::array<const char*, I_U4_rank> I_U4_labels = {
        "beamP",
        "beamQ",
        "Fbar_U4",
        "Ttilde_U4_Tds64",
    };
    static constexpr std::array<std::size_t, I_U4_rank> I_U4_lengths = {
        16,
        24,
        512,
        1024,
    };
    static constexpr std::size_t I_U4_length =
        chord_datatype_bytes(I_U4_type) * 16 * 24 * 512 * 1024;
    static_assert(I_U4_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    static constexpr auto I_U4_calc_stride = [](int dim) {
        std::ptrdiff_t str = 1;
        for (int d = 0; d < dim; ++d)
            str *= I_U4_lengths[d];
        return str;
    };
    static constexpr std::array<std::ptrdiff_t, I_U4_rank + 1> I_U4_strides = {
        I_U4_calc_stride(I_U4_index_beamP),   I_U4_calc_stride(I_U4_index_beamQ),
        I_U4_calc_stride(I_U4_index_Fbar_U4), I_U4_calc_stride(I_U4_index_Ttilde_U4_Tds64),
        I_U4_calc_stride(I_U4_rank),
    };
    static_assert(I_U4_length == chord_datatype_bytes(I_U4_type) * I_U4_strides[I_U4_rank]);
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
    static constexpr std::array<std::size_t, info_rank> info_lengths = {
        32,
        6,
        512,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 6 * 512;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
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
    const std::string S_memname;
    const std::string W_U4_memname;
    const std::string Ebar_U4_memname;
    const std::string I_U4_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::uint8_t> S_host;
    std::vector<std::uint8_t> info_host;

    static constexpr std::size_t Ebar_U4_Tbar_U4_sample_bytes =
        chord_datatype_bytes(Ebar_U4_type) * Ebar_U4_lengths[Ebar_U4_index_D]
        * Ebar_U4_lengths[Ebar_U4_index_P] * Ebar_U4_lengths[Ebar_U4_index_Fbar_U4];
    static constexpr std::size_t I_U4_Ttilde_U4_Tds64_sample_bytes =
        chord_datatype_bytes(I_U4_type) * I_U4_lengths[I_U4_index_beamP]
        * I_U4_lengths[I_U4_index_beamQ] * I_U4_lengths[I_U4_index_Fbar_U4];

    RingBuffer* const input_ringbuf_signal;
    RingBuffer* const output_ringbuf_signal;

    bool did_init_S_host;

    // How many frequencies we will process
    const int Fbarmin, Fbarmax;

    // How many frequencies we will produce
    const int Ftildemin, Ftildemax;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::size_t Tbarmin, Tbarmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::size_t Ttildemin, Ttildemax;
};

REGISTER_CUDA_COMMAND(cudaFRBBeamformer_pathfinder_U4);

cudaFRBBeamformer_pathfinder_U4::cudaFRBBeamformer_pathfinder_U4(Config& config,
                                                                 const std::string& unique_name,
                                                                 bufferContainer& host_buffers,
                                                                 cudaDeviceInterface& device,
                                                                 const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "FRBBeamformer_pathfinder_U4", "FRBBeamformer_pathfinder_U4.ptx"),
    S_memname(unique_name + "/gpu_mem_dishlayout"),
    W_U4_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    Ebar_U4_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    I_U4_memname(config.get<std::string>(unique_name, "gpu_mem_beamgrid")),
    info_memname(unique_name + "/gpu_mem_info"),

    S_host(S_length), info_host(info_length),
    // Find input and output buffers used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")))),
    output_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "out_signal")))),
    did_init_S_host(false), Fbarmin(config.get<int>(unique_name, "Fbarmin")),
    Fbarmax(config.get<int>(unique_name, "Fbarmax")),
    Ftildemin(config.get<int>(unique_name, "Ftildemin")),
    Ftildemax(config.get<int>(unique_name, "Ftildemax")) {
    // Check ringbuffer sizes
    assert(input_ringbuf_signal->size == Ebar_U4_length);
    assert(output_ringbuf_signal->size == I_U4_length);

    // Register host memory
    {
        const cudaError_t ierr = cudaHostRegister(S_host.data(), S_host.size(), 0);
        assert(ierr == cudaSuccess);
    }
    {
        const cudaError_t ierr = cudaHostRegister(info_host.data(), info_host.size(), 0);
        assert(ierr == cudaSuccess);
    }

    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(
        std::make_tuple(get_name() + "_gpu_mem_dishlayout", false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(W_U4_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(Ebar_U4_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(I_U4_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage need to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "FRBBeamformer_pathfinder_U4_");
    }

    if (instance_num == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
        output_ringbuf_signal->register_producer(unique_name);
        output_ringbuf_signal->allocate_new_metadata_object(0);
    }
}

cudaFRBBeamformer_pathfinder_U4::~cudaFRBBeamformer_pathfinder_U4() {}

std::int64_t
cudaFRBBeamformer_pathfinder_U4::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_produced_elements(num_available_elements) * cuda_downsampling_factor;
}
std::int64_t
cudaFRBBeamformer_pathfinder_U4::num_produced_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) / cuda_downsampling_factor;
}

std::int64_t
cudaFRBBeamformer_pathfinder_U4::num_processed_elements(std::int64_t num_available_elements) const {
    assert(num_available_elements >= cuda_granularity_number_of_timesamples);
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cudaFRBBeamformer_pathfinder_U4::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::size_t> val_in1 =
        input_ringbuf_signal->wait_without_claiming(unique_name, instance_num);
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    if (!val_in1.has_value())
        return -1;
    const std::size_t input_bytes = val_in1.value();
    DEBUG("Input ring-buffer byte count: {:d}", input_bytes);

    // How many inputs samples are available?
    const std::size_t Tbar_available = div_noremainder(input_bytes, Ebar_U4_Tbar_U4_sample_bytes);
    DEBUG("Available samples:      Tbar_available: {:d}", Tbar_available);

    // How many outputs will we process and consume?
    const std::size_t Tbar_processed = num_processed_elements(Tbar_available);
    const std::size_t Tbar_consumed = num_consumed_elements(Tbar_available);
    DEBUG("Will process (samples): Tbar_processed: {:d}", Tbar_processed);
    DEBUG("Will consume (samples): Tbar_consumed:  {:d}", Tbar_consumed);
    assert(Tbar_processed > 0);
    assert(Tbar_consumed <= Tbar_processed);
    const std::size_t Tbar_consumed2 = num_consumed_elements(Tbar_processed);
    assert(Tbar_consumed2 == Tbar_consumed);

    const std::optional<std::size_t> val_in2 = input_ringbuf_signal->wait_and_claim_readable(
        unique_name, instance_num, Tbar_consumed * Ebar_U4_Tbar_U4_sample_bytes);
    if (!val_in2.has_value())
        return -1;
    const std::size_t input_cursor = val_in2.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    Tbarmin = div_noremainder(input_cursor, Ebar_U4_Tbar_U4_sample_bytes);
    Tbarmax = Tbarmin + Tbar_processed;
    const std::size_t Tbarlength = Tbarmax - Tbarmin;
    DEBUG("Input samples:");
    DEBUG("    Tbarmin:    {:d}", Tbarmin);
    DEBUG("    Tbarmax:    {:d}", Tbarmax);
    DEBUG("    Tbarlength: {:d}", Tbarlength);

    // How many outputs will we produce?
    const std::size_t Ttilde_produced = num_produced_elements(Tbar_available);
    DEBUG("Will produce (samples): Ttilde_produced: {:d}", Ttilde_produced);
    const std::size_t Ttildelength = Ttilde_produced;

    // to bytes
    const std::size_t output_bytes = Ttildelength * I_U4_Ttilde_U4_Tds64_sample_bytes;
    DEBUG("Will produce {:d} output bytes", output_bytes);

    // Wait for space to be available in our output ringbuffer...
    DEBUG("Waiting for output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::optional<std::size_t> val_out =
        output_ringbuf_signal->wait_for_writable(unique_name, instance_num, output_bytes);
    DEBUG("Finished waiting for output for data frame {:d}.", gpu_frame_id);
    if (!val_out.has_value())
        return -1;
    const std::size_t output_cursor = val_out.value();
    DEBUG("Output ring-buffer byte offset {:d}", output_cursor);

    assert(mod(output_cursor, I_U4_Ttilde_U4_Tds64_sample_bytes) == 0);
    Ttildemin = output_cursor / I_U4_Ttilde_U4_Tds64_sample_bytes;
    Ttildemax = Ttildemin + Ttildelength;
    DEBUG("Output samples:");
    DEBUG("    Ttildemin:    {:d}", Ttildemin);
    DEBUG("    Ttildemax:    {:d}", Ttildemax);
    DEBUG("    Ttildelength: {:d}", Ttildelength);

    return 0;
}

cudaEvent_t
cudaFRBBeamformer_pathfinder_U4::execute(cudaPipelineState& /*pipestate*/,
                                         const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    S_host.resize(S_length);
    void* const S_memory = device.get_gpu_memory(S_memname, S_length);
    void* const W_U4_memory = args::W_U4 == args::Ebar_U4
                                  ? device.get_gpu_memory(W_U4_memname, input_ringbuf_signal->size)
                              : args::W_U4 == args::I_U4
                                  ? device.get_gpu_memory(W_U4_memname, output_ringbuf_signal->size)
                              : args::W_U4 == args::W_U4
                                  ? device.get_gpu_memory(W_U4_memname, W_U4_length)
                                  : device.get_gpu_memory_array(W_U4_memname, gpu_frame_id,
                                                                _gpu_buffer_depth, W_U4_length);
    void* const Ebar_U4_memory =
        args::Ebar_U4 == args::Ebar_U4
            ? device.get_gpu_memory(Ebar_U4_memname, input_ringbuf_signal->size)
        : args::Ebar_U4 == args::I_U4
            ? device.get_gpu_memory(Ebar_U4_memname, output_ringbuf_signal->size)
        : args::Ebar_U4 == args::W_U4
            ? device.get_gpu_memory(Ebar_U4_memname, Ebar_U4_length)
            : device.get_gpu_memory_array(Ebar_U4_memname, gpu_frame_id, _gpu_buffer_depth,
                                          Ebar_U4_length);
    void* const I_U4_memory = args::I_U4 == args::Ebar_U4
                                  ? device.get_gpu_memory(I_U4_memname, input_ringbuf_signal->size)
                              : args::I_U4 == args::I_U4
                                  ? device.get_gpu_memory(I_U4_memname, output_ringbuf_signal->size)
                              : args::I_U4 == args::W_U4
                                  ? device.get_gpu_memory(I_U4_memname, I_U4_length)
                                  : device.get_gpu_memory_array(I_U4_memname, gpu_frame_id,
                                                                _gpu_buffer_depth, I_U4_length);
    info_host.resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // W_U4 is an input buffer: check metadata
    const std::shared_ptr<metadataObject> W_U4_mc =
        args::W_U4 == args::Ebar_U4
            ? input_ringbuf_signal->get_metadata(0)
            : device.get_gpu_memory_array_metadata(W_U4_memname, gpu_frame_id);
    assert(W_U4_mc);
    assert(metadata_is_chord(W_U4_mc));
    const std::shared_ptr<chordMetadata> W_U4_meta = get_chord_metadata(W_U4_mc);
    DEBUG("input W_U4 array: {:s} {:s}", W_U4_meta->get_type_string(),
          W_U4_meta->get_dimensions_string());
    const auto output_meta_W_U4 = [&]() {
        std::ostringstream buf;
        buf << "    name: " << (W_U4_meta)->name << "\n"
            << "    type: " << chord_datatype_string((W_U4_meta)->type) << "\n"
            << "    dim: [";
        for (int d = 0; d < (W_U4_meta)->dims; ++d)
            buf << (W_U4_meta)->dim[d] << ", ";
        buf << "]\n"
            << "    stride: [";
        for (int d = 0; d < (W_U4_meta)->dims; ++d)
            buf << (W_U4_meta)->stride[d] << ", ";
        buf << "]\n";
        return buf.str();
    };
    if (args::W_U4 == args::Ebar_U4 && 4 == 1) {
        // Replace "Ebar_U1" with "E" etc. because we don't run the upchannelizer for U=1
        assert(std::strncmp(W_U4_meta->name, "E", sizeof W_U4_meta->name) == 0);
        assert(W_U4_meta->type == W_U4_type);
        assert(W_U4_meta->dims == 4);
        assert(std::strncmp(W_U4_meta->dim_name[3], "D", sizeof W_U4_meta->dim_name[3]) == 0);
        assert(W_U4_meta->dim[3] == int(Ebar_U4_lengths[0]));
        assert(W_U4_meta->stride[3] == Ebar_U4_strides[0]);
        assert(std::strncmp(W_U4_meta->dim_name[2], "P", sizeof W_U4_meta->dim_name[2]) == 0);
        assert(W_U4_meta->dim[2] == int(Ebar_U4_lengths[1]));
        assert(W_U4_meta->stride[2] == Ebar_U4_strides[1]);
        assert(std::strncmp(W_U4_meta->dim_name[1], "F", sizeof W_U4_meta->dim_name[1]) == 0);
        assert(W_U4_meta->dim[1] == int(Ebar_U4_lengths[2]));
        assert(W_U4_meta->stride[1] == Ebar_U4_strides[2]);
        assert(std::strncmp(W_U4_meta->dim_name[0], "T", sizeof W_U4_meta->dim_name[0]) == 0);
        assert(W_U4_meta->dim[0] <= int(Ebar_U4_lengths[3]));
        assert(W_U4_meta->stride[0] == Ebar_U4_strides[3]);
    } else {
        assert(std::strncmp(W_U4_meta->name, W_U4_name, sizeof W_U4_meta->name) == 0);
        assert(W_U4_meta->type == W_U4_type);
        assert(W_U4_meta->dims == W_U4_rank);
        for (std::size_t dim = 0; dim < W_U4_rank; ++dim) {
            assert(std::strncmp(W_U4_meta->dim_name[W_U4_rank - 1 - dim], W_U4_labels[dim],
                                sizeof W_U4_meta->dim_name[W_U4_rank - 1 - dim])
                   == 0);
            if ((args::W_U4 == args::Ebar_U4 && dim == Ebar_U4_rank - 1)
                || (args::W_U4 == args::W_U4 && dim == W_U4_rank - 1)) {
                assert(W_U4_meta->dim[W_U4_rank - 1 - dim] <= int(W_U4_lengths[dim]));
                assert(W_U4_meta->stride[W_U4_rank - 1 - dim] == W_U4_strides[dim]);
            } else {
                if (!(W_U4_meta->dim[W_U4_rank - 1 - dim] == int(W_U4_lengths[dim]))) {
                    ERROR("Will encounter failing assert");
                    ERROR("dim: {}", dim);
                    ERROR("context:\n{}", output_meta_W_U4());
                }
                assert(W_U4_meta->dim[W_U4_rank - 1 - dim] == int(W_U4_lengths[dim]));
                assert(W_U4_meta->stride[W_U4_rank - 1 - dim] == W_U4_strides[dim]);
            }
        }
    }
    //
    // Ebar_U4 is an input buffer: check metadata
    const std::shared_ptr<metadataObject> Ebar_U4_mc =
        args::Ebar_U4 == args::Ebar_U4
            ? input_ringbuf_signal->get_metadata(0)
            : device.get_gpu_memory_array_metadata(Ebar_U4_memname, gpu_frame_id);
    assert(Ebar_U4_mc);
    assert(metadata_is_chord(Ebar_U4_mc));
    const std::shared_ptr<chordMetadata> Ebar_U4_meta = get_chord_metadata(Ebar_U4_mc);
    DEBUG("input Ebar_U4 array: {:s} {:s}", Ebar_U4_meta->get_type_string(),
          Ebar_U4_meta->get_dimensions_string());
    const auto output_meta_Ebar_U4 = [&]() {
        std::ostringstream buf;
        buf << "    name: " << (Ebar_U4_meta)->name << "\n"
            << "    type: " << chord_datatype_string((Ebar_U4_meta)->type) << "\n"
            << "    dim: [";
        for (int d = 0; d < (Ebar_U4_meta)->dims; ++d)
            buf << (Ebar_U4_meta)->dim[d] << ", ";
        buf << "]\n"
            << "    stride: [";
        for (int d = 0; d < (Ebar_U4_meta)->dims; ++d)
            buf << (Ebar_U4_meta)->stride[d] << ", ";
        buf << "]\n";
        return buf.str();
    };
    if (args::Ebar_U4 == args::Ebar_U4 && 4 == 1) {
        // Replace "Ebar_U1" with "E" etc. because we don't run the upchannelizer for U=1
        assert(std::strncmp(Ebar_U4_meta->name, "E", sizeof Ebar_U4_meta->name) == 0);
        assert(Ebar_U4_meta->type == Ebar_U4_type);
        assert(Ebar_U4_meta->dims == 4);
        assert(std::strncmp(Ebar_U4_meta->dim_name[3], "D", sizeof Ebar_U4_meta->dim_name[3]) == 0);
        assert(Ebar_U4_meta->dim[3] == int(Ebar_U4_lengths[0]));
        assert(Ebar_U4_meta->stride[3] == Ebar_U4_strides[0]);
        assert(std::strncmp(Ebar_U4_meta->dim_name[2], "P", sizeof Ebar_U4_meta->dim_name[2]) == 0);
        assert(Ebar_U4_meta->dim[2] == int(Ebar_U4_lengths[1]));
        assert(Ebar_U4_meta->stride[2] == Ebar_U4_strides[1]);
        assert(std::strncmp(Ebar_U4_meta->dim_name[1], "F", sizeof Ebar_U4_meta->dim_name[1]) == 0);
        assert(Ebar_U4_meta->dim[1] == int(Ebar_U4_lengths[2]));
        assert(Ebar_U4_meta->stride[1] == Ebar_U4_strides[2]);
        assert(std::strncmp(Ebar_U4_meta->dim_name[0], "T", sizeof Ebar_U4_meta->dim_name[0]) == 0);
        assert(Ebar_U4_meta->dim[0] <= int(Ebar_U4_lengths[3]));
        assert(Ebar_U4_meta->stride[0] == Ebar_U4_strides[3]);
    } else {
        assert(std::strncmp(Ebar_U4_meta->name, Ebar_U4_name, sizeof Ebar_U4_meta->name) == 0);
        assert(Ebar_U4_meta->type == Ebar_U4_type);
        assert(Ebar_U4_meta->dims == Ebar_U4_rank);
        for (std::size_t dim = 0; dim < Ebar_U4_rank; ++dim) {
            assert(std::strncmp(Ebar_U4_meta->dim_name[Ebar_U4_rank - 1 - dim], Ebar_U4_labels[dim],
                                sizeof Ebar_U4_meta->dim_name[Ebar_U4_rank - 1 - dim])
                   == 0);
            if ((args::Ebar_U4 == args::Ebar_U4 && dim == Ebar_U4_rank - 1)
                || (args::Ebar_U4 == args::W_U4 && dim == W_U4_rank - 1)) {
                assert(Ebar_U4_meta->dim[Ebar_U4_rank - 1 - dim] <= int(Ebar_U4_lengths[dim]));
                assert(Ebar_U4_meta->stride[Ebar_U4_rank - 1 - dim] == Ebar_U4_strides[dim]);
            } else {
                if (!(Ebar_U4_meta->dim[Ebar_U4_rank - 1 - dim] == int(Ebar_U4_lengths[dim]))) {
                    ERROR("Will encounter failing assert");
                    ERROR("dim: {}", dim);
                    ERROR("context:\n{}", output_meta_Ebar_U4());
                }
                assert(Ebar_U4_meta->dim[Ebar_U4_rank - 1 - dim] == int(Ebar_U4_lengths[dim]));
                assert(Ebar_U4_meta->stride[Ebar_U4_rank - 1 - dim] == Ebar_U4_strides[dim]);
            }
        }
    }
    //
    // I_U4 is an output buffer: set metadata
    std::shared_ptr<metadataObject> const I_U4_mc =
        args::I_U4 == args::I_U4 ? output_ringbuf_signal->get_metadata(0)
                                 : device.create_gpu_memory_array_metadata(
                                     I_U4_memname, gpu_frame_id, Ebar_U4_mc->parent_pool);
    std::shared_ptr<chordMetadata> const I_U4_meta = get_chord_metadata(I_U4_mc);
    *I_U4_meta = *Ebar_U4_meta;
    std::strncpy(I_U4_meta->name, I_U4_name, sizeof I_U4_meta->name);
    I_U4_meta->type = I_U4_type;
    I_U4_meta->dims = I_U4_rank;
    for (std::size_t dim = 0; dim < I_U4_rank; ++dim) {
        std::strncpy(I_U4_meta->dim_name[I_U4_rank - 1 - dim], I_U4_labels[dim],
                     sizeof I_U4_meta->dim_name[I_U4_rank - 1 - dim]);
        I_U4_meta->dim[I_U4_rank - 1 - dim] = I_U4_lengths[dim];
        I_U4_meta->stride[I_U4_rank - 1 - dim] = I_U4_strides[dim];
    }
    DEBUG("output I_U4 array: {:s} {:s}", I_U4_meta->get_type_string(),
          I_U4_meta->get_dimensions_string());
    //

    assert(Ebar_U4_meta->ndishes == cuda_number_of_dishes);
    assert(Ebar_U4_meta->n_dish_locations_ew == cuda_dish_layout_N);
    assert(Ebar_U4_meta->n_dish_locations_ns == cuda_dish_layout_M);
    assert(Ebar_U4_meta->dish_index);

    record_start_event();

    DEBUG("gpu_frame_id: {}", gpu_frame_id);

    const char* exc_arg = "exception";
    std::int32_t Tbarmin_arg;
    std::int32_t Tbarmax_arg;
    std::int32_t Ttildemin_arg;
    std::int32_t Ttildemax_arg;
    std::int32_t Fbarmin_arg;
    std::int32_t Fbarmax_arg;
    std::int32_t Ftildemin_arg;
    std::int32_t Ftildemax_arg;
    array_desc S_arg(S_memory, S_length);
    array_desc W_U4_arg(W_U4_memory, W_U4_length);
    array_desc Ebar_U4_arg(Ebar_U4_memory, Ebar_U4_length);
    array_desc I_U4_arg(I_U4_memory, I_U4_length);
    array_desc info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg,     &Tbarmin_arg, &Tbarmax_arg,   &Ttildemin_arg, &Ttildemax_arg,
        &Fbarmin_arg, &Fbarmax_arg, &Ftildemin_arg, &Ftildemax_arg, &S_arg,
        &W_U4_arg,    &Ebar_U4_arg, &I_U4_arg,      &info_arg,
    };

    // Set Ebar_memory to beginning of input ring buffer
    Ebar_U4_arg = array_desc(Ebar_U4_memory, Ebar_U4_length);

    // Set I_memory to beginning of output ring buffer
    I_U4_arg = array_desc(I_U4_memory, I_U4_length);

    // Ringbuffer size
    const std::size_t Tbar_ringbuf = input_ringbuf_signal->size / Ebar_U4_Tbar_U4_sample_bytes;
    const std::size_t Ttilde_ringbuf =
        output_ringbuf_signal->size / I_U4_Ttilde_U4_Tds64_sample_bytes;
    DEBUG("Input ringbuffer size (samples):  {:d}", Tbar_ringbuf);
    DEBUG("Output ringbuffer size (samples): {:d}", Ttilde_ringbuf);

    const std::size_t Tbarlength = Tbarmax - Tbarmin;
    const std::size_t Ttildelength = Ttildemax - Ttildemin;
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

    // Pass frequency spans to kernel
    Fbarmin_arg = Fbarmin;
    Fbarmax_arg = Fbarmax;
    Ftildemin_arg = Ftildemin;
    Ftildemax_arg = Ftildemax;

    // Update metadata
    I_U4_meta->dim[I_U4_rank - 1 - I_U4_index_Ttilde_U4_Tds64] = Ttildelength;
    assert(I_U4_meta->dim[I_U4_rank - 1 - I_U4_index_Ttilde_U4_Tds64]
           <= int(I_U4_lengths[I_U4_index_Ttilde_U4_Tds64]));
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    assert(I_U4_meta->nfreq >= 0);
    assert(I_U4_meta->nfreq == Ebar_U4_meta->nfreq);
    for (int freq = 0; freq < I_U4_meta->nfreq; ++freq) {
        I_U4_meta->freq_upchan_factor[freq] =
            cuda_downsampling_factor * Ebar_U4_meta->freq_upchan_factor[freq];
        // I_meta->half_fpga_sample0[freq] = Evar_meta->half_fpga_sample0[freq];
        I_U4_meta->time_downsampling_fpga[freq] =
            cuda_downsampling_factor * Ebar_U4_meta->time_downsampling_fpga[freq];
    }

    // Initialize `S` and copy it to the GPU
    if (!did_init_S_host) {
        // S maps dishes to locations.
        // The first `ndishes` dishes are real dishes,
        // the remaining dishes are not real and exist only to label the unoccupied dish locations.
        std::int16_t* __restrict__ const S =
            static_cast<std::int16_t*>(static_cast<void*>(S_host.data()));
        int surplus_dish_index = cuda_number_of_dishes;
        for (int locM = 0; locM < cuda_dish_layout_M; ++locM) {
            for (int locN = 0; locN < cuda_dish_layout_N; ++locN) {
                int dish_index = Ebar_U4_meta->get_dish_index(locN, locM);
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
        INFO("M={} N={}", cuda_dish_layout_M, cuda_dish_layout_N);
        for (int i = 0; i < int(S_host.size() / 2); i += 2)
            INFO("    S[{}] = ({}, {})", i / 2, S[i], S[i + 1]);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(S_memory, S_host.data(), S_length, cudaMemcpyHostToDevice,
                                         device.getStream(cuda_stream_id)));

        did_init_S_host = true;
    }

    // Copy inputs to device memory
    if constexpr (args::S != args::S)
        CHECK_CUDA_ERROR(cudaMemcpyAsync(S_memory, S_host.data(), S_length, cudaMemcpyHostToDevice,
                                         device.getStream(cuda_stream_id)));

#ifdef DEBUGGING
    // Initialize host-side buffer arrays
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length, device.getStream(cuda_stream_id)));
#endif

    const std::string symname = "FRBBeamformer_pathfinder_U4_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA FRBBeamformer_pathfinder_U4 on GPU frame {:d}", gpu_frame_id);
    assert(0 <= Fbarmin && Fbarmin <= Fbarmax);
    assert(0 <= Ftildemin && Ftildemin <= Ftildemax);
    assert(Ftildemax - Ftildemin == Fbarmax - Fbarmin);
    const int blocks = Fbarmax - Fbarmin;
    assert(0 <= blocks);
    assert(blocks <= max_blocks);
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

    for (std::size_t i = 0; i < info_host.size() * blocks / max_blocks; ++i)
        if (info_host[i] != 0)
            ERROR("cudaFRBBeamformer_pathfinder_U4 returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host[i], i);
#endif

    return record_end_event();
}

void cudaFRBBeamformer_pathfinder_U4::finalize_frame() {
    const std::size_t Tbarlength = Tbarmax - Tbarmin;
    const std::size_t Ttildelength = Ttildemax - Ttildemin;

    // Advance the input ringbuffer
    const std::size_t Tbar_consumed = num_consumed_elements(Tbarlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples: {:d}", Tbar_consumed);
    DEBUG("    Consumed bytes:   {:d}", Tbar_consumed * Ebar_U4_Tbar_U4_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, instance_num,
                                      Tbar_consumed * Ebar_U4_Tbar_U4_sample_bytes);

    // Advance the output ringbuffer
    const std::size_t Ttilde_produced = Ttildelength;
    DEBUG("Advancing output ringbuffer:");
    DEBUG("    Produced samples: {:d}", Ttilde_produced);
    DEBUG("    Produced bytes:   {:d}", Ttilde_produced * I_U4_Ttilde_U4_Tds64_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, instance_num,
                                        Ttilde_produced * I_U4_Ttilde_U4_Tds64_sample_bytes);

    cudaCommand::finalize_frame();
}
