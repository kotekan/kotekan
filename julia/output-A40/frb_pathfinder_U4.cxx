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
    static constexpr int cuda_number_of_frequencies = 1536;
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
    static constexpr int blocks = 1536;
    static constexpr int shmem_bytes = 13920;

    // Kernel name:
    static constexpr const char* kernel_symbol =
        "_Z3frb5Int32S_S_S_13CuDeviceArrayI7Int16x2Li1ELi1EES0_I9Float16x2Li1ELi1EES0_"
        "I6Int4x8Li1ELi1EES0_IS2_Li1ELi1EES0_IS_Li1ELi1EE";

    // Kernel arguments:
    enum class args { Tbarmin, Tbarmax, Ttildemin, Ttildemax, S, W4, Ebar4, I4, info, count };

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
        96,
    };
    static constexpr std::size_t S_length = chord_datatype_bytes(S_type) * 2 * 96;
    static_assert(S_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // W4: gpu_mem_phase
    static constexpr const char* W4_name = "W4";
    static constexpr chordDataType W4_type = float16;
    enum W4_indices {
        W4_index_C,
        W4_index_dishM,
        W4_index_dishN,
        W4_index_P,
        W4_index_Fbar,
        W4_rank,
    };
    static constexpr std::array<const char*, W4_rank> W4_labels = {
        "C", "dishM", "dishN", "P", "Fbar",
    };
    static constexpr std::array<std::size_t, W4_rank> W4_lengths = {
        2, 8, 12, 2, 1536,
    };
    static constexpr std::size_t W4_length = chord_datatype_bytes(W4_type) * 2 * 8 * 12 * 2 * 1536;
    static_assert(W4_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // Ebar4: gpu_mem_voltage
    static constexpr const char* Ebar4_name = "Ebar4";
    static constexpr chordDataType Ebar4_type = int4p4;
    enum Ebar4_indices {
        Ebar4_index_D,
        Ebar4_index_P,
        Ebar4_index_Fbar,
        Ebar4_index_Tbar,
        Ebar4_rank,
    };
    static constexpr std::array<const char*, Ebar4_rank> Ebar4_labels = {
        "D",
        "P",
        "Fbar",
        "Tbar",
    };
    static constexpr std::array<std::size_t, Ebar4_rank> Ebar4_lengths = {
        64,
        2,
        1536,
        8192,
    };
    static constexpr std::size_t Ebar4_length =
        chord_datatype_bytes(Ebar4_type) * 64 * 2 * 1536 * 8192;
    static_assert(Ebar4_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //
    // I4: gpu_mem_beamgrid
    static constexpr const char* I4_name = "I4";
    static constexpr chordDataType I4_type = float16;
    enum I4_indices {
        I4_index_beamP,
        I4_index_beamQ,
        I4_index_Fbar,
        I4_index_Ttilde,
        I4_rank,
    };
    static constexpr std::array<const char*, I4_rank> I4_labels = {
        "beamP",
        "beamQ",
        "Fbar",
        "Ttilde",
    };
    static constexpr std::array<std::size_t, I4_rank> I4_lengths = {
        16,
        24,
        1536,
        1024,
    };
    static constexpr std::size_t I4_length = chord_datatype_bytes(I4_type) * 16 * 24 * 1536 * 1024;
    static_assert(I4_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
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
        1536,
    };
    static constexpr std::size_t info_length = chord_datatype_bytes(info_type) * 32 * 6 * 1536;
    static_assert(info_length <= std::size_t(std::numeric_limits<int>::max()) + 1);
    //

    // Kotekan buffer names
    const std::string S_memname;
    const std::string W4_memname;
    const std::string Ebar4_memname;
    const std::string I4_memname;
    const std::string info_memname;

    // Host-side buffer arrays
    std::vector<std::uint8_t> S_host;
    std::vector<std::uint8_t> info_host;

    static constexpr std::size_t Ebar4_Tbar_sample_bytes =
        chord_datatype_bytes(Ebar4_type) * Ebar4_lengths[Ebar4_index_D]
        * Ebar4_lengths[Ebar4_index_P] * Ebar4_lengths[Ebar4_index_Fbar];
    static constexpr std::size_t I4_Ttilde_sample_bytes =
        chord_datatype_bytes(I4_type) * I4_lengths[I4_index_beamP] * I4_lengths[I4_index_beamQ]
        * I4_lengths[I4_index_Fbar];

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

REGISTER_CUDA_COMMAND(cudaFRBBeamformer_pathfinder_U4);

cudaFRBBeamformer_pathfinder_U4::cudaFRBBeamformer_pathfinder_U4(Config& config,
                                                                 const std::string& unique_name,
                                                                 bufferContainer& host_buffers,
                                                                 cudaDeviceInterface& device,
                                                                 const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "FRBBeamformer_pathfinder_U4", "FRBBeamformer_pathfinder_U4.ptx"),
    S_memname(unique_name + "/gpu_mem_dishlayout"),
    W4_memname(config.get<std::string>(unique_name, "gpu_mem_phase")),
    Ebar4_memname(config.get<std::string>(unique_name, "gpu_mem_voltage")),
    I4_memname(config.get<std::string>(unique_name, "gpu_mem_beamgrid")),
    info_memname(unique_name + "/gpu_mem_info"),

    S_host(S_length), info_host(info_length),
    // Find input and output buffers used for signalling ring-buffer state
    input_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")))),
    output_ringbuf_signal(dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "out_signal")))),
    did_init_S_host(false) {
    // Check ringbuffer sizes
    assert(input_ringbuf_signal->size == Ebar4_length);
    assert(output_ringbuf_signal->size == I4_length);

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
    gpu_buffers_used.push_back(std::make_tuple(W4_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(Ebar4_memname, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(I4_memname, true, true, false));
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
    const std::size_t Tbar_available = div_noremainder(input_bytes, Ebar4_Tbar_sample_bytes);
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
        unique_name, instance_num, Tbar_consumed * Ebar4_Tbar_sample_bytes);
    if (!val_in2.has_value())
        return -1;
    const std::size_t input_cursor = val_in2.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    Tbarmin = div_noremainder(input_cursor, Ebar4_Tbar_sample_bytes);
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
    const std::size_t output_bytes = Ttildelength * I4_Ttilde_sample_bytes;
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

    assert(mod(output_cursor, I4_Ttilde_sample_bytes) == 0);
    Ttildemin = output_cursor / I4_Ttilde_sample_bytes;
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
    void* const W4_memory =
        args::W4 == args::Ebar4 ? device.get_gpu_memory(W4_memname, input_ringbuf_signal->size)
        : args::W4 == args::I4  ? device.get_gpu_memory(W4_memname, output_ringbuf_signal->size)
        : args::W4 == args::W4
            ? device.get_gpu_memory(W4_memname, W4_length)
            : device.get_gpu_memory_array(W4_memname, gpu_frame_id, _gpu_buffer_depth, W4_length);
    void* const Ebar4_memory =
        args::Ebar4 == args::Ebar4
            ? device.get_gpu_memory(Ebar4_memname, input_ringbuf_signal->size)
        : args::Ebar4 == args::I4
            ? device.get_gpu_memory(Ebar4_memname, output_ringbuf_signal->size)
        : args::Ebar4 == args::W4 ? device.get_gpu_memory(Ebar4_memname, Ebar4_length)
                                  : device.get_gpu_memory_array(Ebar4_memname, gpu_frame_id,
                                                                _gpu_buffer_depth, Ebar4_length);
    void* const I4_memory =
        args::I4 == args::Ebar4 ? device.get_gpu_memory(I4_memname, input_ringbuf_signal->size)
        : args::I4 == args::I4  ? device.get_gpu_memory(I4_memname, output_ringbuf_signal->size)
        : args::I4 == args::W4
            ? device.get_gpu_memory(I4_memname, I4_length)
            : device.get_gpu_memory_array(I4_memname, gpu_frame_id, _gpu_buffer_depth, I4_length);
    info_host.resize(info_length);
    void* const info_memory = device.get_gpu_memory(info_memname, info_length);

    // W4 is an input buffer: check metadata
    const std::shared_ptr<metadataObject> W4_mc =
        args::W4 == args::Ebar4 ? input_ringbuf_signal->get_metadata(0)
                                : device.get_gpu_memory_array_metadata(W4_memname, gpu_frame_id);
    assert(W4_mc);
    assert(metadata_is_chord(W4_mc));
    const std::shared_ptr<chordMetadata> W4_meta = get_chord_metadata(W4_mc);
    DEBUG("input W4 array: {:s} {:s}", W4_meta->get_type_string(),
          W4_meta->get_dimensions_string());
    if (args::W4 == args::Ebar4 && 4 == 1) {
        // Replace "Ebar1" with "E" etc. because we don't run the upchannelizer for U=1
        assert(std::strncmp(W4_meta->name, "E", sizeof W4_meta->name) == 0);
        assert(W4_meta->type == W4_type);
        assert(W4_meta->dims == 4);
        assert(std::strncmp(W4_meta->dim_name[3], "D", sizeof W4_meta->dim_name[3]) == 0);
        assert(W4_meta->dim[3] == int(Ebar4_lengths[0]));
        assert(std::strncmp(W4_meta->dim_name[2], "P", sizeof W4_meta->dim_name[2]) == 0);
        assert(W4_meta->dim[2] == int(Ebar4_lengths[1]));
        assert(std::strncmp(W4_meta->dim_name[1], "F", sizeof W4_meta->dim_name[1]) == 0);
        assert(W4_meta->dim[1] == int(Ebar4_lengths[2]));
        assert(std::strncmp(W4_meta->dim_name[0], "T", sizeof W4_meta->dim_name[0]) == 0);
        assert(W4_meta->dim[0] <= int(Ebar4_lengths[3]));
    } else {
        assert(std::strncmp(W4_meta->name, W4_name, sizeof W4_meta->name) == 0);
        assert(W4_meta->type == W4_type);
        assert(W4_meta->dims == W4_rank);
        for (std::size_t dim = 0; dim < W4_rank; ++dim) {
            assert(std::strncmp(W4_meta->dim_name[W4_rank - 1 - dim], W4_labels[dim],
                                sizeof W4_meta->dim_name[W4_rank - 1 - dim])
                   == 0);
            if (args::W4 == args::Ebar4 && dim == Ebar4_index_Tbar)
                assert(W4_meta->dim[W4_rank - 1 - dim] <= int(W4_lengths[dim]));
            else
                assert(W4_meta->dim[W4_rank - 1 - dim] == int(W4_lengths[dim]));
        }
    }
    //
    // Ebar4 is an input buffer: check metadata
    const std::shared_ptr<metadataObject> Ebar4_mc =
        args::Ebar4 == args::Ebar4
            ? input_ringbuf_signal->get_metadata(0)
            : device.get_gpu_memory_array_metadata(Ebar4_memname, gpu_frame_id);
    assert(Ebar4_mc);
    assert(metadata_is_chord(Ebar4_mc));
    const std::shared_ptr<chordMetadata> Ebar4_meta = get_chord_metadata(Ebar4_mc);
    DEBUG("input Ebar4 array: {:s} {:s}", Ebar4_meta->get_type_string(),
          Ebar4_meta->get_dimensions_string());
    if (args::Ebar4 == args::Ebar4 && 4 == 1) {
        // Replace "Ebar1" with "E" etc. because we don't run the upchannelizer for U=1
        assert(std::strncmp(Ebar4_meta->name, "E", sizeof Ebar4_meta->name) == 0);
        assert(Ebar4_meta->type == Ebar4_type);
        assert(Ebar4_meta->dims == 4);
        assert(std::strncmp(Ebar4_meta->dim_name[3], "D", sizeof Ebar4_meta->dim_name[3]) == 0);
        assert(Ebar4_meta->dim[3] == int(Ebar4_lengths[0]));
        assert(std::strncmp(Ebar4_meta->dim_name[2], "P", sizeof Ebar4_meta->dim_name[2]) == 0);
        assert(Ebar4_meta->dim[2] == int(Ebar4_lengths[1]));
        assert(std::strncmp(Ebar4_meta->dim_name[1], "F", sizeof Ebar4_meta->dim_name[1]) == 0);
        assert(Ebar4_meta->dim[1] == int(Ebar4_lengths[2]));
        assert(std::strncmp(Ebar4_meta->dim_name[0], "T", sizeof Ebar4_meta->dim_name[0]) == 0);
        assert(Ebar4_meta->dim[0] <= int(Ebar4_lengths[3]));
    } else {
        assert(std::strncmp(Ebar4_meta->name, Ebar4_name, sizeof Ebar4_meta->name) == 0);
        assert(Ebar4_meta->type == Ebar4_type);
        assert(Ebar4_meta->dims == Ebar4_rank);
        for (std::size_t dim = 0; dim < Ebar4_rank; ++dim) {
            assert(std::strncmp(Ebar4_meta->dim_name[Ebar4_rank - 1 - dim], Ebar4_labels[dim],
                                sizeof Ebar4_meta->dim_name[Ebar4_rank - 1 - dim])
                   == 0);
            if (args::Ebar4 == args::Ebar4 && dim == Ebar4_index_Tbar)
                assert(Ebar4_meta->dim[Ebar4_rank - 1 - dim] <= int(Ebar4_lengths[dim]));
            else
                assert(Ebar4_meta->dim[Ebar4_rank - 1 - dim] == int(Ebar4_lengths[dim]));
        }
    }
    //
    // I4 is an output buffer: set metadata
    std::shared_ptr<metadataObject> const I4_mc =
        args::I4 == args::I4 ? output_ringbuf_signal->get_metadata(0)
                             : device.create_gpu_memory_array_metadata(I4_memname, gpu_frame_id,
                                                                       Ebar4_mc->parent_pool);
    std::shared_ptr<chordMetadata> const I4_meta = get_chord_metadata(I4_mc);
    *I4_meta = *Ebar4_meta;
    std::strncpy(I4_meta->name, I4_name, sizeof I4_meta->name);
    I4_meta->type = I4_type;
    I4_meta->dims = I4_rank;
    for (std::size_t dim = 0; dim < I4_rank; ++dim) {
        std::strncpy(I4_meta->dim_name[I4_rank - 1 - dim], I4_labels[dim],
                     sizeof I4_meta->dim_name[I4_rank - 1 - dim]);
        I4_meta->dim[I4_rank - 1 - dim] = I4_lengths[dim];
    }
    DEBUG("output I4 array: {:s} {:s}", I4_meta->get_type_string(),
          I4_meta->get_dimensions_string());
    //

    assert(Ebar4_meta->ndishes == cuda_number_of_dishes);
    assert(Ebar4_meta->n_dish_locations_ew == cuda_dish_layout_N);
    assert(Ebar4_meta->n_dish_locations_ns == cuda_dish_layout_M);
    assert(Ebar4_meta->dish_index);

    record_start_event();

    DEBUG("gpu_frame_id: {}", gpu_frame_id);

    const char* exc_arg = "exception";
    std::int32_t Tbarmin_arg;
    std::int32_t Tbarmax_arg;
    std::int32_t Ttildemin_arg;
    std::int32_t Ttildemax_arg;
    array_desc S_arg(S_memory, S_length);
    array_desc W4_arg(W4_memory, W4_length);
    array_desc Ebar4_arg(Ebar4_memory, Ebar4_length);
    array_desc I4_arg(I4_memory, I4_length);
    array_desc info_arg(info_memory, info_length);
    void* args[] = {
        &exc_arg, &Tbarmin_arg, &Tbarmax_arg, &Ttildemin_arg, &Ttildemax_arg,
        &S_arg,   &W4_arg,      &Ebar4_arg,   &I4_arg,        &info_arg,
    };

    // Set Ebar4_memory to beginning of input ring buffer
    Ebar4_arg = array_desc(Ebar4_memory, Ebar4_length);

    // Set I_memory to beginning of output ring buffer
    I4_arg = array_desc(I4_memory, I4_length);

    // Ringbuffer size
    const std::size_t Tbar_ringbuf = input_ringbuf_signal->size / Ebar4_Tbar_sample_bytes;
    const std::size_t Ttilde_ringbuf = output_ringbuf_signal->size / I4_Ttilde_sample_bytes;
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
    I4_meta->dim[I4_rank - 1 - I4_index_Ttilde] = Ttildelength;
    assert(I4_meta->dim[I4_rank - 1 - I4_index_Ttilde] <= int(I4_lengths[I4_index_Ttilde]));
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    assert(I4_meta->nfreq >= 0);
    assert(I4_meta->nfreq == Ebar4_meta->nfreq);
    for (int freq = 0; freq < I4_meta->nfreq; ++freq) {
        I4_meta->freq_upchan_factor[freq] =
            cuda_downsampling_factor * Ebar4_meta->freq_upchan_factor[freq];
        // I_meta->half_fpga_sample0[freq] = Evar_meta->half_fpga_sample0[freq];
        I4_meta->time_downsampling_fpga[freq] =
            cuda_downsampling_factor * Ebar4_meta->time_downsampling_fpga[freq];
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
                int dish_index = Ebar4_meta->get_dish_index(locN, locM);
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

    const std::string symname = "FRBBeamformer_pathfinder_U4_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

    DEBUG("Running CUDA FRBBeamformer_pathfinder_U4 on GPU frame {:d}", gpu_frame_id);
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
    DEBUG("    Consumed bytes:   {:d}", Tbar_consumed * Ebar4_Tbar_sample_bytes);
    input_ringbuf_signal->finish_read(unique_name, instance_num,
                                      Tbar_consumed * Ebar4_Tbar_sample_bytes);

    // Advance the output ringbuffer
    const std::size_t Ttilde_produced = Ttildelength;
    DEBUG("Advancing output ringbuffer:");
    DEBUG("    Produced samples: {:d}", Ttilde_produced);
    DEBUG("    Produced bytes:   {:d}", Ttilde_produced * I4_Ttilde_sample_bytes);
    output_ringbuf_signal->finish_write(unique_name, instance_num,
                                        Ttilde_produced * I4_Ttilde_sample_bytes);

    cudaCommand::finalize_frame();
}
