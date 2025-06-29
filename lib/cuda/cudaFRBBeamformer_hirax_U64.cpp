/**
 * @file
 * @brief CUDA FRBBeamformer_hirax_U64 kernel
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

namespace {
template<typename T, std::size_t D>
std::array<T, D> reverse(const std::array<T, D>& values) {
    std::array<T, D> result;
    for (std::size_t d = 0; d < D; ++d)
        result[d] = values[D - 1 - d];
    return result;
}
} // namespace

/**
 * @class cudaFRBBeamformer_hirax_U64
 * @brief cudaCommand for FRBBeamformer_hirax_U64
 */
class cudaFRBBeamformer_hirax_U64 : public cudaCommand {
public:
    cudaFRBBeamformer_hirax_U64(Config& config, const std::string& unique_name,
                                bufferContainer& host_buffers, cudaDeviceInterface& device,
                                const int instance_num);
    virtual ~cudaFRBBeamformer_hirax_U64();

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
    static constexpr int cuda_beam_layout_M = 32;
    static constexpr int cuda_beam_layout_N = 32;
    static constexpr int cuda_dish_layout_M = 16;
    static constexpr int cuda_dish_layout_N = 16;
    static constexpr int cuda_upchannelization_factor = 64;
    static constexpr int cuda_downsampling_factor = 6;
    static constexpr int cuda_number_of_complex_components = 2;
    static constexpr int cuda_number_of_dishes = 256;
    static constexpr int cuda_number_of_frequencies = 64;
    static constexpr int cuda_number_of_polarizations = 2;
    static constexpr int cuda_number_of_timesamples = 1024;
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
    static constexpr int max_blocks = 64;
    static constexpr int shmem_bytes = 34944;

    // Kernel name:
    static constexpr const char* kernel_symbol =
        "_Z3frb5Int32S_S_S_S_S_S_S_13CuDeviceArrayI7Int16x2Li1ELi1EES0_I9Float16x2Li1ELi1EES0_"
        "I6Int4x8Li1ELi1EES4_S0_IS_Li1ELi1EE";

    // How many frequencies we will process
    const int Fbar_in_min, Fbar_in_max;
    const int Fbar_out_min, Fbar_out_max;

    // Kernel arguments:
    enum class args {
        Tbar_min,
        Tbar_max,
        Ttilde_min,
        Ttilde_max,
        Fbar_in_min,
        Fbar_in_max,
        Fbar_out_min,
        Fbar_out_max,
        S,
        W,
        Ebar,
        I,
        info,
        count
    };

    // Tbar_min: Tbar_min
    static constexpr const char* Tbar_min_quantity = "Tbar_min";
    static constexpr kotekan::DataType Tbar_min_type = kotekan::int32;
    //
    // Tbar_max: Tbar_max
    static constexpr const char* Tbar_max_quantity = "Tbar_max";
    static constexpr kotekan::DataType Tbar_max_type = kotekan::int32;
    //
    // Ttilde_min: Ttilde_min
    static constexpr const char* Ttilde_min_quantity = "Ttilde_min";
    static constexpr kotekan::DataType Ttilde_min_type = kotekan::int32;
    //
    // Ttilde_max: Ttilde_max
    static constexpr const char* Ttilde_max_quantity = "Ttilde_max";
    static constexpr kotekan::DataType Ttilde_max_type = kotekan::int32;
    //
    // Fbar_in_min: Fbar_in_min
    static constexpr const char* Fbar_in_min_quantity = "Fbar_in_min";
    static constexpr kotekan::DataType Fbar_in_min_type = kotekan::int32;
    //
    // Fbar_in_max: Fbar_in_max
    static constexpr const char* Fbar_in_max_quantity = "Fbar_in_max";
    static constexpr kotekan::DataType Fbar_in_max_type = kotekan::int32;
    //
    // Fbar_out_min: Fbar_out_min
    static constexpr const char* Fbar_out_min_quantity = "Fbar_out_min";
    static constexpr kotekan::DataType Fbar_out_min_type = kotekan::int32;
    //
    // Fbar_out_max: Fbar_out_max
    static constexpr const char* Fbar_out_max_quantity = "Fbar_out_max";
    static constexpr kotekan::DataType Fbar_out_max_type = kotekan::int32;
    //
    // S: frb_dishlayout_name
    static constexpr const char* S_quantity = "S";
    static constexpr kotekan::DataType S_type = kotekan::int16;
    enum S_indices {
        S_index_MN,
        S_index_D,
        S_rank,
    };
    static constexpr std::array<const char*, S_rank> S_labels = {
        "MN",
        "D",
    };
    static constexpr std::array<std::ptrdiff_t, S_rank> S_lengths = {
        2,
        256,
    };
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
    static constexpr std::ptrdiff_t S_length = S_strides[S_rank];
    static constexpr std::ptrdiff_t S_length_in_bytes = type_total_bytes(S_type) * S_length;
    // We allow the `I` buffer to be large. We have checked the sizes and 64-bit code in the GPU
    // kernels where necessary.
    static_assert(args::S == args::I
                      ? true
                      : S_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
        2, 16, 16, 2, 64,
    };
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
    static constexpr std::ptrdiff_t W_length = W_strides[W_rank];
    static constexpr std::ptrdiff_t W_length_in_bytes = type_total_bytes(W_type) * W_length;
    // We allow the `I` buffer to be large. We have checked the sizes and 64-bit code in the GPU
    // kernels where necessary.
    static_assert(args::W == args::I
                      ? true
                      : W_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
        256,
        2,
        64,
        1024,
    };
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
    static constexpr std::ptrdiff_t Ebar_length = Ebar_strides[Ebar_rank];
    static constexpr std::ptrdiff_t Ebar_length_in_bytes =
        type_total_bytes(Ebar_type) * Ebar_length;
    // We allow the `I` buffer to be large. We have checked the sizes and 64-bit code in the GPU
    // kernels where necessary.
    static_assert(args::Ebar == args::I
                      ? true
                      : Ebar_length_in_bytes
                            <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
        32,
        32,
        2048,
        1024,
    };
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
    static constexpr std::ptrdiff_t I_length = I_strides[I_rank];
    static constexpr std::ptrdiff_t I_length_in_bytes = type_total_bytes(I_type) * I_length;
    // We allow the `I` buffer to be large. We have checked the sizes and 64-bit code in the GPU
    // kernels where necessary.
    static_assert(args::I == args::I
                      ? true
                      : I_length_in_bytes <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
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
        16,
        64,
    };
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
    static constexpr std::ptrdiff_t info_length = info_strides[info_rank];
    static constexpr std::ptrdiff_t info_length_in_bytes =
        type_total_bytes(info_type) * info_length;
    // We allow the `I` buffer to be large. We have checked the sizes and 64-bit code in the GPU
    // kernels where necessary.
    static_assert(args::info == args::I
                      ? true
                      : info_length_in_bytes
                            <= std::ptrdiff_t(std::numeric_limits<int>::max()) + 1);
    //

    // Kotekan buffer names
    const std::string S_name;
    const std::string W_name;
    const std::string Ebar_name;
    const std::string I_name;
    const std::string info_name;

    // Host-side buffer arrays
    std::vector<std::uint8_t> S_host;
    std::vector<std::uint8_t> info_host;

    // Buffers
    NDArrayBuffer<kotekan::GetType_t<S_type>, S_rank> S_buffer;
    std::vector<kotekan::GetType_t<S_type>> host_S_buffer;
    NDArrayBuffer<kotekan::GetType_t<W_type>, W_rank> W_buffer;
    NDArrayRingBuffer<kotekan::GetType_t<Ebar_type>, Ebar_rank> Ebar_buffer;
    NDArrayRingBuffer<kotekan::GetType_t<I_type>, I_rank> I_buffer;
    NDArrayBuffer<kotekan::GetType_t<info_type>, info_rank> info_buffer;
    std::vector<kotekan::GetType_t<info_type>> host_info_buffer;

    bool did_init_host_S_buffer;
};

REGISTER_CUDA_COMMAND(cudaFRBBeamformer_hirax_U64);

cudaFRBBeamformer_hirax_U64::cudaFRBBeamformer_hirax_U64(Config& config,
                                                         const std::string& unique_name,
                                                         bufferContainer& host_buffers,
                                                         cudaDeviceInterface& device,
                                                         const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "FRBBeamformer_hirax_U64", "FRBBeamformer_hirax_U64.ptx"),
    Fbar_in_min(config.get<int>(unique_name, "Fbar_in_min")),
    Fbar_in_max(config.get<int>(unique_name, "Fbar_in_max")),
    Fbar_out_min(config.get<int>(unique_name, "Fbar_out_min")),
    Fbar_out_max(config.get<int>(unique_name, "Fbar_out_max")),

    S_name(unique_name + "/frb_dishlayout_name"),
    W_name(config.get<std::string>(unique_name, "frb_phase_name")),
    Ebar_name(config.get<std::string>(unique_name, "voltage_name")),
    I_name(config.get<std::string>(unique_name, "frb_beamgrid_name")),
    info_name(unique_name + "/gpu_mem_info"),

    S_buffer(S_name, S_quantity, reverse(S_lengths), reverse(S_labels), *this),
    host_S_buffer(S_length), W_buffer(W_name, W_quantity, reverse(W_lengths), reverse(W_labels),
                                      *this, buffer_type_t::do_once),
    Ebar_buffer(Ebar_name, Ebar_quantity, reverse(Ebar_lengths), reverse(Ebar_labels), *this),
    I_buffer(I_name, I_quantity, reverse(I_lengths), reverse(I_labels), *this),
    info_buffer(info_name, info_quantity, reverse(info_lengths), reverse(info_labels), *this),
    host_info_buffer(info_length),

    did_init_host_S_buffer(false) {
    // Register host memory
    {
        const cudaError_t ierr = cudaHostRegister(
            host_S_buffer.data(), host_S_buffer.size() * sizeof *host_S_buffer.data(), 0);
        assert(ierr == cudaSuccess);
    }
    {
        const cudaError_t ierr = cudaHostRegister(
            host_info_buffer.data(), host_info_buffer.size() * sizeof *host_info_buffer.data(), 0);
        assert(ierr == cudaSuccess);
    }

    register_gpu_buffer_user(
        {.name = S_name, .is_array = true, .does_read = true, .does_write = true});
    W_buffer.register_consumer();
    Ebar_buffer.register_consumer();
    I_buffer.register_producer();
    register_gpu_buffer_user(
        {.name = info_name, .is_array = true, .does_read = true, .does_write = true});

    set_command_type(gpuCommandType::KERNEL);

    // Only one of the instances of this pipeline stage need to build the kernel
    if (instance_num == 0) {
        const std::vector<std::string> opts = {
            "--gpu-name=sm_86",
            "--verbose",
        };
        device.build_ptx(kernel_file_name, {kernel_symbol}, opts, "FRBBeamformer_hirax_U64_");
    }
}

cudaFRBBeamformer_hirax_U64::~cudaFRBBeamformer_hirax_U64() {}

std::int64_t
cudaFRBBeamformer_hirax_U64::num_consumed_elements(std::int64_t num_available_elements) const {
    return num_produced_elements(num_available_elements) * cuda_downsampling_factor;
}
std::int64_t
cudaFRBBeamformer_hirax_U64::num_produced_elements(std::int64_t num_available_elements) const {
    return num_processed_elements(num_available_elements) / cuda_downsampling_factor;
}

std::int64_t
cudaFRBBeamformer_hirax_U64::num_processed_elements(std::int64_t num_available_elements) const {
    return round_down(num_available_elements, cuda_granularity_number_of_timesamples);
}

int cudaFRBBeamformer_hirax_U64::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    const std::ptrdiff_t Tbar_ringbuf = Ebar_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t Tbar_read_max = Tbar_ringbuf / 4;
    std::ptrdiff_t Tbar_read = -1;
    {
        const int errcode =
            Ebar_buffer.wait_and_claim_readable([&](const std::ptrdiff_t Tbar_available) {
                using std::min;
                Tbar_read = min(Tbar_available, Tbar_read_max);
                return read_descriptor_t{.claimed = num_consumed_elements(Tbar_read),
                                         .read = num_processed_elements(Tbar_read)};
            });
        if (errcode < 0)
            return errcode;
    }
    const std::ptrdiff_t Ttilde_written = num_produced_elements(Tbar_read);

    // Wait for space to be available in output ringbuffer
    {
        const int errcode = I_buffer.wait_for_writable(Ttilde_written);
        if (errcode < 0)
            return errcode;
    }

    return 0;
}

cudaEvent_t cudaFRBBeamformer_hirax_U64::execute(cudaPipelineState& /*pipestate*/,
                                                 const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    void* const S_memory = S_buffer.get_ndarray().data();
    void* const W_memory = W_buffer.get_ndarray().data();
    void* const Ebar_memory = Ebar_buffer.get_ndarray().data();
    void* const I_memory = I_buffer.get_ndarray().data();
    void* const info_memory = info_buffer.get_ndarray().data();

    if (args::W == args::Ebar && cuda_upchannelization_factor == 1) {
        // Replace "Ebar" with "E" etc. because we don't run the upchannelizer for U=1
        // W_buffer.check_metadata();
        const std::string quantity = "E";
        const std::array<std::string, 4> dimname = {"T", "F", "P", "D"};
        const std::shared_ptr<const chordMetadata> metadata = Ebar_buffer.get_metadata();
        if (!(metadata->get_name() == quantity))
            ERROR("buffer name: {:s}, quantity: {:s}, metadata name: {:s}",
                  Ebar_buffer.get_buffer_name(), quantity, metadata->get_name());
        assert(metadata->get_name() == quantity);
        const auto& ndarray = Ebar_buffer.get_ndarray();
        assert(metadata->type == ndarray.value_datatype);
        assert(metadata->dims == ndarray.rank);
        for (std::size_t d = 0; d < ndarray.rank; ++d) {
            assert(metadata->get_dimension_name(d) == dimname[d]);
            // The ring buffer direction is special
            if (d > 0)
                assert(metadata->dim[d] == int(ndarray.extent(d)));
            assert(metadata->stride[d] == ndarray.stride(d));
        }
    } else {
        W_buffer.check_metadata();
    }
    if (args::Ebar == args::Ebar && cuda_upchannelization_factor == 1) {
        // Replace "Ebar" with "E" etc. because we don't run the upchannelizer for U=1
        // Ebar_buffer.check_metadata();
        const std::string quantity = "E";
        const std::array<std::string, 4> dimname = {"T", "F", "P", "D"};
        const std::shared_ptr<const chordMetadata> metadata = Ebar_buffer.get_metadata();
        if (!(metadata->get_name() == quantity))
            ERROR("buffer name: {:s}, quantity: {:s}, metadata name: {:s}",
                  Ebar_buffer.get_buffer_name(), quantity, metadata->get_name());
        assert(metadata->get_name() == quantity);
        const auto& ndarray = Ebar_buffer.get_ndarray();
        assert(metadata->type == ndarray.value_datatype);
        assert(metadata->dims == ndarray.rank);
        for (std::size_t d = 0; d < ndarray.rank; ++d) {
            assert(metadata->get_dimension_name(d) == dimname[d]);
            // The ring buffer direction is special
            if (d > 0)
                assert(metadata->dim[d] == int(ndarray.extent(d)));
            assert(metadata->stride[d] == ndarray.stride(d));
        }
    } else {
        Ebar_buffer.check_metadata();
    }
    I_buffer.set_metadata(Ebar_buffer.get_metadata());

    const auto Ebar_meta = Ebar_buffer.get_metadata();
    assert(Ebar_meta->ndishes == cuda_number_of_dishes);
    assert(Ebar_meta->n_dish_locations_ew == cuda_dish_layout_N);
    assert(Ebar_meta->n_dish_locations_ns == cuda_dish_layout_M);
    assert(Ebar_meta->dish_index);

    auto I_meta = I_buffer.get_metadata();
    assert(I_meta->nfreq >= 0);
    assert(I_meta->nfreq == Ebar_meta->nfreq);
    for (int freq = 0; freq < I_meta->nfreq; ++freq) {
        I_meta->freq_upchan_factor[freq] *= cuda_downsampling_factor;
        I_meta->time_downsampling_fpga[freq] *= cuda_downsampling_factor;
    }
    // Since we use a ring buffer we do not need to update `meta->sample0_offset`

    const char* exc_arg = "exception";
    std::int32_t Tbar_min_arg;
    std::int32_t Tbar_max_arg;
    std::int32_t Ttilde_min_arg;
    std::int32_t Ttilde_max_arg;
    std::int32_t Fbar_in_min_arg;
    std::int32_t Fbar_in_max_arg;
    std::int32_t Fbar_out_min_arg;
    std::int32_t Fbar_out_max_arg;
    array_desc S_arg(S_memory, S_length_in_bytes);
    array_desc W_arg(W_memory, W_length_in_bytes);
    array_desc Ebar_arg(Ebar_memory, Ebar_length_in_bytes);
    array_desc I_arg(I_memory, I_length_in_bytes);
    array_desc info_arg(info_memory, info_length_in_bytes);
    void* args[] = {
        &exc_arg,
        &Tbar_min_arg,
        &Tbar_max_arg,
        &Ttilde_min_arg,
        &Ttilde_max_arg,
        &Fbar_in_min_arg,
        &Fbar_in_max_arg,
        &Fbar_out_min_arg,
        &Fbar_out_max_arg,
        &S_arg,
        &W_arg,
        &Ebar_arg,
        &I_arg,
        &info_arg,
    };

    // Set Ebar_memory to beginning of input ring buffer
    Ebar_arg = array_desc(Ebar_memory, Ebar_length_in_bytes);

    // Set I_memory to beginning of output ring buffer
    I_arg = array_desc(I_memory, I_length_in_bytes);

    // Ringbuffer size
    const std::ptrdiff_t Tbar_ringbuf = Ebar_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t Ttilde_ringbuf = I_buffer.get_ndarray().extent(0);

    const std::ptrdiff_t Tbar_min = Ebar_buffer.get_read_valid().begin();
    const std::ptrdiff_t Tbar_max = Ebar_buffer.get_read_valid().end();
    const std::ptrdiff_t Ttilde_min = I_buffer.get_write_valid().begin();
    const std::ptrdiff_t Ttilde_max = I_buffer.get_write_valid().end();

    const std::ptrdiff_t Tbar_length = Tbar_max - Tbar_min;
    const std::ptrdiff_t Ttilde_length = Ttilde_max - Ttilde_min;

    // Pass time spans to kernel
    // The kernel will wrap the upper bounds to make them fit into the ringbuffer
    Tbar_min_arg = mod(Tbar_min, Tbar_ringbuf);
    Tbar_max_arg = mod(Tbar_min, Tbar_ringbuf) + Tbar_length;
    Ttilde_min_arg = mod(Ttilde_min, Ttilde_ringbuf);
    Ttilde_max_arg = mod(Ttilde_min, Ttilde_ringbuf) + Ttilde_length;

    // Pass frequency spans to kernel
    assert(0 <= Fbar_in_min && Fbar_in_min <= Fbar_in_max);
    assert(0 <= Fbar_out_min && Fbar_out_min <= Fbar_out_max);
    assert(Fbar_out_max - Fbar_out_min == Fbar_in_max - Fbar_in_min);
    Fbar_in_min_arg = Fbar_in_min;
    Fbar_in_max_arg = Fbar_in_max;
    Fbar_out_min_arg = Fbar_out_min;
    Fbar_out_max_arg = Fbar_out_max;
    const int blocks = Fbar_out_max - Fbar_out_min;
    assert(0 <= blocks);
    assert(blocks <= max_blocks);

    // Initialize `S` and copy it to the GPU
    if (!did_init_host_S_buffer) {
        // S maps dishes to locations.
        // The first `ndishes` dishes are real dishes,
        // the remaining dishes are not real and exist only to label the unoccupied dish locations.
        std::int16_t* __restrict__ const S = host_S_buffer.data();
        int surplus_dish_index = cuda_number_of_dishes;
        for (int locM = 0; locM < cuda_dish_layout_M; ++locM) {
            for (int locN = 0; locN < cuda_dish_layout_N; ++locN) {
                int dish_index = Ebar_meta->get_dish_index(locN, locM);
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

        CHECK_CUDA_ERROR(cudaMemcpyAsync(S_memory, host_S_buffer.data(), S_length_in_bytes,
                                         cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));

        did_init_host_S_buffer = true;
    }

    // Copy inputs to device memory
    if constexpr (args::S != args::S)
        CHECK_CUDA_ERROR(cudaMemcpyAsync(S_memory, host_S_buffer.data(), S_length_in_bytes,
                                         cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));

        // We do not write all frequencies
        // I_buffer.set_to_poison(0xff); // 0xffff is NaN16

#ifdef DEBUGGING
    // Poison outputs
    {
        const int num_chunks = Ttilde_max_arg <= Ttilde_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            const std::ptrdiff_t Ttilde_stride = I_meta->stride[0];
            const std::ptrdiff_t Ttilde_offset = chunk == 0 ? Ttilde_min_arg : 0;
            const std::ptrdiff_t Ttilde_length = (num_chunks == 1 ? Ttilde_max_arg - Ttilde_min_arg
                                                  : chunk == 0    ? Ttilde_ringbuf - Ttilde_min_arg
                                                               : Ttilde_max_arg - Ttilde_ringbuf);
            const std::ptrdiff_t Fbar_out_stride = I_meta->stride[1];
            const std::ptrdiff_t Fbar_out_offset = Fbar_out_min;
            const std::ptrdiff_t Fbar_out_length = Fbar_out_max - Fbar_out_min;
            CHECK_CUDA_ERROR(cudaMemset2DAsync((std::uint8_t*)I_memory
                                                   + 2 * Ttilde_offset * Ttilde_stride
                                                   + 2 * Fbar_out_offset * Fbar_out_stride,
                                               2 * Ttilde_stride,
                                               0xff, // 0xffff is NaN16
                                               2 * Fbar_out_length * Fbar_out_stride, Ttilde_length,
                                               device.getStream(cuda_stream_id)));
        } // for chunk
    }
#endif

    info_buffer.set_to_poison(0xff);

#ifdef DEBUGGING
    // Initialize host-side buffer arrays
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(info_memory, 0xff, info_length_in_bytes, device.getStream(cuda_stream_id)));
#endif

    const std::string symname = "FRBBeamformer_hirax_U64_" + std::string(kernel_symbol);
    CHECK_CU_ERROR(cuFuncSetAttribute(device.runtime_kernels[symname],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shmem_bytes));

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
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_info_buffer.data(), info_memory, info_length_in_bytes,
                                     cudaMemcpyDeviceToHost, device.getStream(cuda_stream_id)));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));

    // Check error codes
    const std::uint32_t error_code = *std::max_element(
        (const std::uint32_t*)&host_info_buffer[0],
        (const std::uint32_t*)&host_info_buffer[blocks * info_lengths[info_index_warp]
                                                * info_lengths[info_index_thread]]);
    if (error_code != 0)
        ERROR("CUDA kernel FRBBeamformer_hirax_U64 returned error code: {}", error_code);

    if (error_code != 0) {
        // TODO: Introduce a new "unbuffered" buffer; do this there
        // Our `info` buffer is too large (`blocks` vs. `max_blocks`)
        for (int block = 0; block < blocks; ++block) {
            for (int warp = 0; warp < info_lengths[info_index_warp]; ++warp) {
                for (int thread = 0; thread < info_lengths[info_index_thread]; ++thread) {
                    const std::ptrdiff_t i = info_strides[info_index_thread] * thread
                                             + info_strides[info_index_warp] * warp
                                             + info_strides[info_index_block] * block;
                    const std::uint32_t val = host_info_buffer.data()[i];
                    if (val != 0)
                        ERROR("CUDA kernel FRBBeamformer_hirax_U64 returned 'info' value {:d} "
                              "for thread {:d} warp {:d} block {:d} at index {:d} (zero indicates "
                              "no error)",
                              val, thread, warp, block, i);
                }
            }
        }
    }
#endif

    // We do not write all frequencies
    // I_buffer.check_for_poison(0xff);

#ifdef DEBUGGING
    // Check outputs for poison
    {
        const int num_chunks = Ttilde_max_arg <= Ttilde_ringbuf ? 1 : 2;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            DEBUG("poison check chunk={}/{}", chunk, num_chunks);
            const std::ptrdiff_t Ttilde_stride = I_meta->stride[0];
            const std::ptrdiff_t Ttilde_offset = chunk == 0 ? Ttilde_min_arg : 0;
            const std::ptrdiff_t Ttilde_length = num_chunks == 1 ? Ttilde_max_arg - Ttilde_min_arg
                                                 : chunk == 0    ? Ttilde_ringbuf - Ttilde_min_arg
                                                                 : Ttilde_max_arg - Ttilde_ringbuf;
            const std::ptrdiff_t Fbar_out_stride = I_meta->stride[1];
            const std::ptrdiff_t Fbar_out_offset = Fbar_out_min;
            const std::ptrdiff_t Fbar_out_length = Fbar_out_max - Fbar_out_min;
            std::vector<std::uint16_t> I_buffer(Ttilde_length * Fbar_out_length * Fbar_out_stride,
                                                0xfffe);
            CHECK_CUDA_ERROR(cudaMemcpy2D(I_buffer.data(), 2 * Fbar_out_length * Fbar_out_stride,
                                          (const std::uint8_t*)I_memory
                                              + 2 * Ttilde_offset * Ttilde_stride
                                              + 2 * Fbar_out_offset * Fbar_out_stride,
                                          2 * Ttilde_stride, 2 * Fbar_out_length * Fbar_out_stride,
                                          Ttilde_length, cudaMemcpyDeviceToHost));

            bool I_found_error = false;
            for (std::ptrdiff_t ttilde = 0; ttilde < Ttilde_length; ++ttilde) {
                for (std::ptrdiff_t ftilde = 0; ftilde < Fbar_out_length; ++ftilde) {
                    bool any_error = false, all_error = true;
                    for (std::ptrdiff_t n = 0; n < Fbar_out_stride; ++n) {
                        const auto val = I_buffer.at(ttilde * (Fbar_out_length * Fbar_out_stride)
                                                     + ftilde * Fbar_out_stride + n);
                        const bool val_is_finite = (val & 0b0111110000000000) != 0b0111110000000000;
                        any_error |= !val_is_finite;
                        all_error &= !val_is_finite;
                    }
                    I_found_error |= any_error;
                }
            }
            if (I_found_error)
                WARN("CUDA kernel FRBBeamformer_hirax_U64 produced non-finite results");
        } // for chunk
    }
#endif

    return record_end_event();
}

void cudaFRBBeamformer_hirax_U64::finalize_frame() {
    // Advance the input ring buffer
    Ebar_buffer.finish_read();

    // Advance the output ring buffer
    I_buffer.finish_write();

    cudaCommand::finalize_frame();
}
