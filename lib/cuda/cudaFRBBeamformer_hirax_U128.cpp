/**
 * @file
 * @brief CUDA FRBBeamformer_hirax_U128 kernel
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
 * @class cudaFRBBeamformer_hirax_U128
 * @brief cudaCommand for FRBBeamformer_hirax_U128
 */
class cudaFRBBeamformer_hirax_U128 : public cudaCommand {
public:
    cudaFRBBeamformer_hirax_U128(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, cudaDeviceInterface& device,
                                 const int instance_num);
    virtual ~cudaFRBBeamformer_hirax_U128();

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
    static constexpr int cuda_beam_layout_M = 32;
    static constexpr int cuda_beam_layout_N = 32;
    static constexpr int cuda_dish_layout_M = 16;
    static constexpr int cuda_dish_layout_N = 16;
    static constexpr int cuda_downsampling_factor = 25;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 256;
    static constexpr int cuda_number_of_frequencies = 8192;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 256;
    static constexpr int cuda_granularity_number_of_timesamples = 64;

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
    static constexpr int blocks = 8192;
    static constexpr int shmem_bytes = 34944;

    // Kernel name:
    static constexpr const char* kernel_symbol =
        "_Z3frb5Int32S_S_S_13CuDeviceArrayI7Int16x2Li1ELi1EES0_I9Float16x2Li1ELi1EES0_"
        "I6Int4x8Li1ELi1EES0_IS2_Li1ELi1EES0_IS_Li1ELi1EE";

    // Kernel arguments:
    enum class args { Tbarmin, Tbarmax, Ttildemin, Ttildemax, S, W128, Ebar128, I128, info, count };

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
        256,
    };
    static constexpr std::size_t S_length = chord_datatype_bytes(S_type) * 2 * 256;
    static_assert(S_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // W128: gpu_mem_phase
    static constexpr const char* W128_name = "W128";
    static constexpr chordDataType W128_type = float16;
    enum W128_indices {
        W128_index_C,
        W128_index_dishM,
        W128_index_dishN,
        W128_index_P,
        W128_index_Fbar,
        W128_rank,
    };
    static constexpr std::array<const char*, W128_rank> W128_labels = {
        "C", "dishM", "dishN", "P", "Fbar",
    };
    static constexpr std::array<std::size_t, W128_rank> W128_lengths = {
        2, 16, 16, 2, 8192,
    };
    static constexpr std::size_t W128_length =
        chord_datatype_bytes(W128_type) * 2 * 16 * 16 * 2 * 8192;
    static_assert(W128_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // Ebar128: gpu_mem_voltage
    static constexpr const char* Ebar128_name = "Ebar128";
    static constexpr chordDataType Ebar128_type = int4p4;
    enum Ebar128_indices {
        Ebar128_index_D,
        Ebar128_index_P,
        Ebar128_index_Fbar,
        Ebar128_index_Tbar,
        Ebar128_rank,
    };
    static constexpr std::array<const char*, Ebar128_rank> Ebar128_labels = {
        "D",
        "P",
        "Fbar",
        "Tbar",
    };
    static constexpr std::array<std::size_t, Ebar128_rank> Ebar128_lengths = {
        256,
        2,
        8192,
        256,
    };
    static constexpr std::size_t Ebar128_length =
        chord_datatype_bytes(Ebar128_type) * 256 * 2 * 8192 * 256;
    static_assert(Ebar128_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // I128: gpu_mem_beamgrid
    static constexpr const char* I128_name = "I128";
    static constexpr chordDataType I128_type = float16;
    enum I128_indices {
        I128_index_beamP,
        I128_index_beamQ,
        I128_index_Fbar,
        I128_index_Ttilde,
        I128_rank,
    };
    static constexpr std::array<const char*, I128_rank> I128_labels = {
        "beamP",
        "beamQ",
        "Fbar",
        "Ttilde",
    };
    static constexpr std::array<std::size_t, I128_rank> I128_lengths = {
        32,
        32,
        8192,
        32,
    };
    static constexpr std::size_t I128_length =
        chord_datatype_bytes(I128_type) * 32 * 32 * 8192 * 32;
    static_assert(I128_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
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
        16,
        8192,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 16 * 8192;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //

    // Kotekan buffer names
    const std::string S_memname;
    const std::string W128_memname;
    const std::string Ebar128_memname;
    const std::string I128_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::uint8_t> S_host;
    std::vector<std::uint8_t> info_host;

    static constexpr std::size_t Ebar128_Tbar_sample_bytes =
        chord_datatype_bytes(Ebar128_type) * Ebar128_lengths[Ebar128_index_D]
        * Ebar128_lengths[Ebar128_index_P] * Ebar128_lengths[Ebar128_index_Fbar];
    static constexpr std::size_t I128_Ttilde_sample_bytes =
        chord_datatype_bytes(I128_type) * I128_lengths[I128_index_beamP]
        * I128_lengths[I128_index_beamQ] * I128_lengths[I128_index_Fbar];

    RingBuffer* const input_ringbuf_signal;
    RingBuffer* const output_ringbuf_signal;

    bool did_init_S_host;

    // How many samples we will process from the input ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::size_t Tbarmin, Tbarmax;

    // How many samples we will produce in the output ringbuffer
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::size_t Ttildemin, Ttildemax;
};

REGISTER_CUDA_COMMAND(cudaFRBBeamformer_hirax_U128);

cudaFRBBeamformer_hirax_U128::cudaFRBBeamformer_hirax_U128(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& host_buffers,
                                                           cudaDeviceInterface& device,
                                                           const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "FRBBeamformer_hirax_U128", "FRBBeamformer_hirax_U128.ptx"),
    S_memname(unique_name + "/gpu_mem_dishlayout"),
    W128_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    Ebar128_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    I128_memname(config.get<std::string>(unique_name, "gpu_mem_beamgrid")),
    info_memname(unique_name + "/gpu_mem_info"),

    S_host(S_length), info_host(info_length),
    // Find input and output buffers used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")))),
    output_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "out_signal")))),
    did_init_S_host(false) {
    // Check ringbuffer sizes
    assert(input_ringbuf_signal->size == Ebar128_length);
    assert(output_ringbuf_signal->size == I128_length);

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
    gpu_buffers_used.push_back(std::make_tuple(W128_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(Ebar128_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(I128_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_gpu_mem_info", false, true, true));

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage need to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "FRBBeamformer_hirax_U128_");
    }

    if (instance_num == 0) {
        input_ringbuf_signal->register_consumer(unique_name);
        output_ringbuf_signal->register_producer(unique_name);
        output_ringbuf_signal->allocate_new_metadata_object(0);
    }
}

cudaFRBBeamformer_hirax_U128::~cudaFRBBeamformer_hirax_U128() {}

std::int64_t
cudaFRBBeamformer_hirax_U128::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_produced_elements(num_available_elements) * cuda_downsampling_factor;
}
std::int64_t
cudaFRBBeamformer_hirax_U128::num_produced_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) / cuda_downsampling_factor;
}

std::int64_t
cudaFRBBeamformer_hirax_U128::num_processed_elements(std::int64_t num_available_elements) const {
    assert(num_available_elements >= cuda_granularity_number_of_timesamples);
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cudaFRBBeamformer_hirax_U128::wait_on_precondition() {
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
    const std::size_t Tbar_available = div_noremainder(input_bytes, Ebar128_Tbar_sample_bytes);
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
        unique_name, instance_num, Tbar_consumed * Ebar128_Tbar_sample_bytes);
    if (!val_in2.has_value())
        return -1;
    const std::size_t input_cursor = val_in2.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    Tbarmin = div_noremainder(input_cursor, Ebar128_Tbar_sample_bytes);
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
    const std::size_t output_bytes = Ttildelength * I128_Ttilde_sample_bytes;
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

    assert(mod(output_cursor, I128_Ttilde_sample_bytes) == 0);
    Ttildemin = output_cursor / I128_Ttilde_sample_bytes;
    Ttildemax = Ttildemin + Ttildelength;
    DEBUG("Output samples:");
    DEBUG("    Ttildemin:    {:d}", Ttildemin);
    DEBUG("    Ttildemax:    {:d}", Ttildemax);
    DEBUG("    Ttildelength: {:d}", Ttildelength);

    return 0;
}

cudaEvent_t cudaFRBBeamformer_hirax_U128::execute(cudaPipelineState& /*pipestate*/,
                                                  const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    S_host.resize(S_length);
    void* const S_memory = device.get_gpu_memory(S_memname, S_length);
    void* const W128_memory = args::W128 == args::Ebar128
                                  ? device.get_gpu_memory(W128_memname, input_ringbuf_signal->size)
                              : args::W128 == args::I128
                                  ? device.get_gpu_memory(W128_memname, output_ringbuf_signal->size)
                              : args::W128 == args::W128
                                  ? device.get_gpu_memory(W128_memname, W128_length)
                                  : device.get_gpu_memory_array(W128_memname, gpu_frame_id,
                                                                _gpu_buffer_depth, W128_length);
    void* const Ebar128_memory =
        args::Ebar128 == args::Ebar128
            ? device.get_gpu_memory(Ebar128_memname, input_ringbuf_signal->size)
        : args::Ebar128 == args::I128
            ? device.get_gpu_memory(Ebar128_memname, output_ringbuf_signal->size)
        : args::Ebar128 == args::W128
            ? device.get_gpu_memory(Ebar128_memname, Ebar128_length)
            : device.get_gpu_memory_array(Ebar128_memname, gpu_frame_id, _gpu_buffer_depth,
                                          Ebar128_length);
    void* const I128_memory = args::I128 == args::Ebar128
                                  ? device.get_gpu_memory(I128_memname, input_ringbuf_signal->size)
                              : args::I128 == args::I128
                                  ? device.get_gpu_memory(I128_memname, output_ringbuf_signal->size)
                              : args::I128 == args::W128
                                  ? device.get_gpu_memory(I128_memname, I128_length)
                                  : device.get_gpu_memory_array(I128_memname, gpu_frame_id,
                                                                _gpu_buffer_depth, I128_length);
    info_host.resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // W128 is an input buffer: check metadata
    const std::shared_ptr<metadataObject> W128_mc =
        args::W128 == args::Ebar128
            ? input_ringbuf_signal->get_metadata(0)
            : device.get_gpu_memory_array_metadata(W128_memname, gpu_frame_id);
    assert(W128_mc);
    assert(metadata_is_chord(W128_mc));
    const std::shared_ptr<chordMetadata> W128_meta = get_chord_metadata(W128_mc);
    DEBUG("input W128 array: {:s} {:s}", W128_meta->get_type_string(),
          W128_meta->get_dimensions_string());
    if (args::W128 == args::Ebar128 && 128 == 1) {
        // Replace "Ebar1" with "E" etc. because we don't run the upchannelizer for U=1
        assert(std::strncmp(W128_meta->name, "E", sizeof W128_meta->name) == 0);
        assert(W128_meta->type == W128_type);
        assert(W128_meta->dims == 4);
        assert(std::strncmp(W128_meta->dim_name[3], "D", sizeof W128_meta->dim_name[3]) == 0);
        assert(W128_meta->dim[3] == int(Ebar128_lengths[0]));
        assert(std::strncmp(W128_meta->dim_name[2], "P", sizeof W128_meta->dim_name[2]) == 0);
        assert(W128_meta->dim[2] == int(Ebar128_lengths[1]));
        assert(std::strncmp(W128_meta->dim_name[1], "F", sizeof W128_meta->dim_name[1]) == 0);
        assert(W128_meta->dim[1] == int(Ebar128_lengths[2]));
        assert(std::strncmp(W128_meta->dim_name[0], "T", sizeof W128_meta->dim_name[0]) == 0);
        assert(W128_meta->dim[0] <= int(Ebar128_lengths[3]));
    } else {
        assert(std::strncmp(W128_meta->name, W128_name, sizeof W128_meta->name) == 0);
        assert(W128_meta->type == W128_type);
        assert(W128_meta->dims == W128_rank);
        for (std::size_t dim = 0; dim < W128_rank; ++dim) {
            assert(std::strncmp(W128_meta->dim_name[W128_rank - 1 - dim], W128_labels[dim],
                                sizeof W128_meta->dim_name[W128_rank - 1 - dim])
                   == 0);
            if (args::W128 == args::Ebar128 && dim == Ebar128_index_Tbar)
                assert(W128_meta->dim[W128_rank - 1 - dim] <= int(W128_lengths[dim]));
            else
                assert(W128_meta->dim[W128_rank - 1 - dim] == int(W128_lengths[dim]));
        }
    }
    //
    // Ebar128 is an input buffer: check metadata
    const std::shared_ptr<metadataObject> Ebar128_mc =
        args::Ebar128 == args::Ebar128
            ? input_ringbuf_signal->get_metadata(0)
            : device.get_gpu_memory_array_metadata(Ebar128_memname, gpu_frame_id);
    assert(Ebar128_mc);
    assert(metadata_is_chord(Ebar128_mc));
    const std::shared_ptr<chordMetadata> Ebar128_meta = get_chord_metadata(Ebar128_mc);
    DEBUG("input Ebar128 array: {:s} {:s}", Ebar128_meta->get_type_string(),
          Ebar128_meta->get_dimensions_string());
    if (args::Ebar128 == args::Ebar128 && 128 == 1) {
        // Replace "Ebar1" with "E" etc. because we don't run the upchannelizer for U=1
        assert(std::strncmp(Ebar128_meta->name, "E", sizeof Ebar128_meta->name) == 0);
        assert(Ebar128_meta->type == Ebar128_type);
        assert(Ebar128_meta->dims == 4);
        assert(std::strncmp(Ebar128_meta->dim_name[3], "D", sizeof Ebar128_meta->dim_name[3]) == 0);
        assert(Ebar128_meta->dim[3] == int(Ebar128_lengths[0]));
        assert(std::strncmp(Ebar128_meta->dim_name[2], "P", sizeof Ebar128_meta->dim_name[2]) == 0);
        assert(Ebar128_meta->dim[2] == int(Ebar128_lengths[1]));
        assert(std::strncmp(Ebar128_meta->dim_name[1], "F", sizeof Ebar128_meta->dim_name[1]) == 0);
        assert(Ebar128_meta->dim[1] == int(Ebar128_lengths[2]));
        assert(std::strncmp(Ebar128_meta->dim_name[0], "T", sizeof Ebar128_meta->dim_name[0]) == 0);
        assert(Ebar128_meta->dim[0] <= int(Ebar128_lengths[3]));
    } else {
        assert(std::strncmp(Ebar128_meta->name, Ebar128_name, sizeof Ebar128_meta->name) == 0);
        assert(Ebar128_meta->type == Ebar128_type);
        assert(Ebar128_meta->dims == Ebar128_rank);
        for (std::size_t dim = 0; dim < Ebar128_rank; ++dim) {
            assert(std::strncmp(Ebar128_meta->dim_name[Ebar128_rank - 1 - dim], Ebar128_labels[dim],
                                sizeof Ebar128_meta->dim_name[Ebar128_rank - 1 - dim])
                   == 0);
            if (args::Ebar128 == args::Ebar128 && dim == Ebar128_index_Tbar)
                assert(Ebar128_meta->dim[Ebar128_rank - 1 - dim] <= int(Ebar128_lengths[dim]));
            else
                assert(Ebar128_meta->dim[Ebar128_rank - 1 - dim] == int(Ebar128_lengths[dim]));
        }
    }
    //
    // I128 is an output buffer: set metadata
    std::shared_ptr<metadataObject> const I128_mc =
        args::I128 == args::I128 ? output_ringbuf_signal->get_metadata(0)
                                 : device.create_gpu_memory_array_metadata(
                                     I128_memname, gpu_frame_id, Ebar128_mc->parent_pool);
    std::shared_ptr<chordMetadata> const I128_meta = get_chord_metadata(I128_mc);
    *I128_meta = *Ebar128_meta;
    std::strncpy(I128_meta->name, I128_name, sizeof I128_meta->name);
    I128_meta->type = I128_type;
    I128_meta->dims = I128_rank;
    for (std::size_t dim = 0; dim < I128_rank; ++dim) {
        std::strncpy(I128_meta->dim_name[I128_rank - 1 - dim], I128_labels[dim],
                     sizeof I128_meta->dim_name[I128_rank - 1 - dim]);
        I128_meta->dim[I128_rank - 1 - dim] = I128_lengths[dim];
    }
    DEBUG("output I128 array: {:s} {:s}", I128_meta->get_type_string(),
          I128_meta->get_dimensions_string());
    //

    assert(Ebar128_meta->ndishes == cuda_number_of_dishes);
    assert(Ebar128_meta->n_dish_locations_ew == cuda_dish_layout_N);
    assert(Ebar128_meta->n_dish_locations_ns == cuda_dish_layout_M);
    assert(Ebar128_meta->dish_index);

    record_start_event();

    DEBUG("gpu_frame_id: {}", gpu_frame_id);

    const char* exc_arg = "exception";
    std::int32_t Tbarmin_arg;
    std::int32_t Tbarmax_arg;
    std::int32_t Ttildemin_arg;
    std::int32_t Ttildemax_arg;
    array_desc S_arg(S_memory, S_length);
    array_desc W128_arg(W128_memory, W128_length);
    array_desc Ebar128_arg(Ebar128_memory, Ebar128_length);
    array_desc I128_arg(I128_memory, I128_length);
    array_desc info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &Tbarmin_arg, &Tbarmax_arg, &Ttildemin_arg, &Ttildemax_arg,
        &S_arg,   &W128_arg,    &Ebar128_arg, &I128_arg,      &info_arg,
    };

    // Set Ebar128_memory to beginning of input ring buffer
    Ebar128_arg = array_desc(Ebar128_memory, Ebar128_length);

    // Set I_memory to beginning of output ring buffer
    I128_arg = array_desc(I128_memory, I128_length);

    // Ringbuffer size
    const std::size_t Tbar_ringbuf = input_ringbuf_signal->size / Ebar128_Tbar_sample_bytes;
    const std::size_t Ttilde_ringbuf = output_ringbuf_signal->size / I128_Ttilde_sample_bytes;
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

    // Update metadata
    I128_meta->dim[I128_rank - 1 - I128_index_Ttilde] = Ttildelength;
    assert(I128_meta->dim[I128_rank - 1 - I128_index_Ttilde]
           <= int(I128_lengths[I128_index_Ttilde]));
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    assert(I128_meta->nfreq >= 0);
    assert(I128_meta->nfreq == Ebar128_meta->nfreq);
    for (int freq = 0; freq < I128_meta->nfreq; ++freq) {
        I128_meta->freq_upchan_factor[freq] =
            cuda_downsampling_factor * Ebar128_meta->freq_upchan_factor[freq];
        // I_meta->half_fpga_sample0[freq] = Evar_meta->half_fpga_sample0[freq];
        I128_meta->time_downsampling_fpga[freq] =
            cuda_downsampling_factor * Ebar128_meta->time_downsampling_fpga[freq];
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
                int dish_index = Ebar128_meta->get_dish_index(locN, locM);
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

    const std::string symname = "FRBBeamformer_hirax_U128_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA FRBBeamformer_hirax_U128 on GPU frame {:d}", gpu_frame_id);
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
            ERROR("cudaFRBBeamformer_hirax_U128 returned 'info' value {:d} at index {:d} (zero "
                  "indicates no error)",
                  info_host[i], i);
#endif

    return record_end_event();
}

void cudaFRBBeamformer_hirax_U128::finalize_frame() {
    const std::size_t Tbarlength = Tbarmax - Tbarmin;
    const std::size_t Ttildelength = Ttildemax - Ttildemin;

    // Advance the input ringbuffer
    const std::size_t Tbar_consumed = num_consumed_elements(Tbarlength);
    DEBUG("Advancing input ringbuffer:");
    DEBUG("    Consumed samples: {:d}", Tbar_consumed);
    DEBUG("    Consumed bytes:   {:d}", Tbar_consumed * Ebar128_Tbar_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, instance_num,
                                      Tbar_consumed * Ebar128_Tbar_sample_bytes);

    // Advance the output ringbuffer
    const std::size_t Ttilde_produced = Ttildelength;
    DEBUG("Advancing output ringbuffer:");
    DEBUG("    Produced samples: {:d}", Ttilde_produced);
    DEBUG("    Produced bytes:   {:d}", Ttilde_produced * I128_Ttilde_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, instance_num,
                                        Ttilde_produced * I128_Ttilde_sample_bytes);

    cudaCommand::finalize_frame();
}
