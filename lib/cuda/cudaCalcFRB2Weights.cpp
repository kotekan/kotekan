/**
 * @file
 * @brief Calculate the FRB2 beamforming weights on the GPU
 *  - cudaCalcFRB2Weights : public cudaCommand
 */

#include "Config.hpp"                    // for Config
#include "DataType.hpp"                  // for float16_t
#include "NDArrayBuffer.hpp"             // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"         // for NDArrayRingBuffer
#include "Telescope.hpp"                 // for Telescope, freq_id_t
#include "UpchannelizationSchedule.hpp"  // for UpchannelizationSchedule, wait_for_coarse_freq
#include "buffer.hpp"                    // for Buffer
#include "bufferContainer.hpp"           // for bufferContainer
#include "chordMetadata.hpp"             // for chordMetadata
#include "cudaCalcFRB2WeightsKernel.hpp" // for cuda_calc_frb2_weights
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "div.hpp"                 // for mod
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG, INFO, FATAL_ERROR

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>      // for min
#include <array>          // for array
#include <cassert>        // for assert
#include <cstddef>        // for ptrdiff_t, size_t
#include <cstdint>        // for int64_t
#include <cuda_runtime.h> // for cudaMemcpyAsync
#include <memory>         // for shared_ptr, make_shared
#include <optional>       // for optional
#include <string>         // for string
#include <vector>         // for vector

using kotekan::mod;

/**
 * @class cudaCalcFRB2Weights
 * @brief cudaCommand calculating the FRB2 beamforming weights `W2` directly into their GPU ring
 *        buffer.
 *
 * Each invocation turns one frame of FRB2 beam positions into one weights matrix, i.e. one element
 * of the `W2` ring buffer consumed by cudaFRBBeamReformer. A positions frame, and hence a weights
 * matrix, is valid for `frb2_weights_lifetime_in_samples` FPGA samples; its `fpga_seq_num` is
 * `seq0 + k * frb2_weights_lifetime_in_samples`, where `k` counts the frames. The positions have
 * to be copied to the GPU beforehand, by a (non-`do_once`) cudaInputData followed by cudaSyncInput
 * in the same cudaProcess. The weights never leave the GPU.
 *
 * The weights depend on the frequencies this GPU handles. These are not in the configuration,
 * which is the same for every GPU: they come from the `coarse_freq` metadata of the first frame of
 * the `metadata_source` buffers, typically the voltage buffers, which are read once, on the first
 * invocation, and then released.
 *
 * The calculation is the same as calcFRB2Weights' (see there for the conventions), up to
 * floating-point differences (see cudaCalcFRB2WeightsKernel.hpp).
 *
 * @par GPU Memory
 * @gpu_mem <frb2_beam_positions_name>_buffer  Input beam positions, filled by cudaInputData
 *     @gpu_mem_type   array
 *     @gpu_mem_format float32 [R, X/Y]
 * @gpu_mem <frb2_weights_name>_buffer  Output weights ring buffer
 *     @gpu_mem_type   ring buffer
 *     @gpu_mem_format float16 [TW2, Fbar, R, beamQ, beamP]
 * @gpu_mem <frb2_weights_name>_frequencies_buffer  The frequencies in Hz, uploaded once
 *     @gpu_mem_type   static
 *     @gpu_mem_format float32 [Fbar]
 *
 * @par Buffers
 * @buffer metadata_source  Buffer or list of buffers whose first frame's `coarse_freq`
 *                          metadata lists the coarse frequency channels this GPU handles. Must
 *                          also be listed in the cudaProcess's `in_buffers`.
 * @buffer host_<frb2_weights_name>_ringbuffer  Signal buffer for the weights ring buffer
 *
 * @conf frb2_beam_positions_name  String. Name of the beam positions (without `_buffer`).
 * @conf frb2_weights_name  String. Name of the weights (without `_buffer`).
 * @conf frb2_weights_lifetime_in_samples  Int. FPGA samples one weights matrix is valid for. Must
 *                          equal the positions' `time_downsampling_fpga`.
 * @conf frb2_weights_ring_depth  Int (default 2). Number of weights matrices in the ring buffer.
 * @conf frb2_num_beams_x   Int. Number of FRB2 beams in the x direction.
 * @conf frb2_num_beams_y   Int. Number of FRB2 beams in the y direction.
 * @conf frb2_num_frequencies  Int. Number of (upchannelized) frequencies.
 * @conf frb1_swap_MN       Bool (default false). See calcFRB2Weights.
 * @conf upchannelization_schedule_name  String (default ""). Config path of the upchannelization
 *                          schedule.
 */
class cudaCalcFRB2Weights : public cudaCommand {
public:
    cudaCalcFRB2Weights(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                        int instance_num);
    ~cudaCalcFRB2Weights() {}
    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    // Calculate the frequencies from the upchannelization schedule (instance 0 only)
    bool setup_frequencies();

    // Upchannelization setup
    const std::string upchannelization_schedule_name;

    // FRB1 beamformer setup
    // See calcFRB2Weights for the meaning of the directions M/N and P/Q and of `frb1_swap_MN`.
    const int num_dishes_x;
    const int num_dishes_y;
    const bool frb1_swap_MN;
    const int num_dishes_M;
    const int num_dishes_N;
    const int frb1_num_beams_P;
    const int frb1_num_beams_Q;

    // FRB2 beamformer setup
    const int frb2_num_beams_x;
    const int frb2_num_beams_y;
    const int frb2_num_beams;
    const int frb2_num_frequencies;

    // Lifetime of a weights matrix in FPGA samples, and the number of matrices in the ring buffer
    const std::ptrdiff_t frb2_weights_lifetime_in_samples;
    const int frb2_weights_ring_depth;

    // Kotekan buffer names
    const std::string frb2_beam_positions_name;
    const std::string frb2_weights_name;
    const std::string frb2_frequencies_name;

    // Buffers
    std::vector<Buffer*> metadata_sources;
    NDArrayBuffer<float, 2> frb2_beam_positions_buffer;
    NDArrayRingBuffer<float16_t, 5> frb2_weights_buffer;
    float* const frb2_frequencies_memory;

    // Frequency setup, known to instance 0 after its first `wait_on_precondition`
    bool did_setup_frequencies;
    std::vector<int> coarse_freq;
    std::vector<int> freq_upchan_factor;
    std::vector<int> freq_upchan_index;
    std::vector<float> frequencies;

    // Set once, on the first frame; see `NDArrayRingBuffer::set_metadata`
    bool did_set_metadata;
};

REGISTER_CUDA_COMMAND(cudaCalcFRB2Weights);

cudaCalcFRB2Weights::cudaCalcFRB2Weights(kotekan::Config& config, const std::string& unique_name,
                                         kotekan::bufferContainer& host_buffers,
                                         cudaDeviceInterface& device, const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaCalcFRB2Weights"),

    upchannelization_schedule_name(
        config.get_default<std::string>(unique_name, "upchannelization_schedule_name", "")),

    num_dishes_x(Telescope::instance().get_grid_size_x()),
    num_dishes_y(Telescope::instance().get_grid_size_y()),
    frb1_swap_MN(config.get_default<bool>(unique_name, "frb1_swap_MN", false)),
    num_dishes_M(frb1_swap_MN ? num_dishes_y : num_dishes_x),
    num_dishes_N(frb1_swap_MN ? num_dishes_x : num_dishes_y), frb1_num_beams_P(2 * num_dishes_M),
    frb1_num_beams_Q(2 * num_dishes_N),

    frb2_num_beams_x(config.get<int>(unique_name, "frb2_num_beams_x")),
    frb2_num_beams_y(config.get<int>(unique_name, "frb2_num_beams_y")),
    frb2_num_beams(frb2_num_beams_x * frb2_num_beams_y),
    frb2_num_frequencies(config.get<int>(unique_name, "frb2_num_frequencies")),

    frb2_weights_lifetime_in_samples(
        config.get<std::int64_t>(unique_name, "frb2_weights_lifetime_in_samples")),
    frb2_weights_ring_depth(config.get_default<int>(unique_name, "frb2_weights_ring_depth", 2)),

    frb2_beam_positions_name(config.get<std::string>(unique_name, "frb2_beam_positions_name")),
    frb2_weights_name(config.get<std::string>(unique_name, "frb2_weights_name")),
    frb2_frequencies_name(frb2_weights_name + "_frequencies_buffer"),

    frb2_beam_positions_buffer(frb2_beam_positions_name, "frb2_beam_positions",
                               std::array<std::ptrdiff_t, 2>{frb2_num_beams, 2},
                               std::array<std::string, 2>{"R", "X/Y"}, {1, 1}, *this),
    frb2_weights_buffer(frb2_weights_name, "W2",
                        std::array<std::ptrdiff_t, 5>{frb2_weights_ring_depth, frb2_num_frequencies,
                                                      frb2_num_beams, frb1_num_beams_Q,
                                                      frb1_num_beams_P},
                        std::array<std::string, 5>{"TW2", "Fbar", "R", "beamQ", "beamP"},
                        {frb2_weights_lifetime_in_samples, 1, 1, 1, 1}, *this),
    frb2_frequencies_memory(static_cast<float*>(
        device.get_gpu_memory(frb2_frequencies_name, frb2_num_frequencies * sizeof(float)))),

    did_setup_frequencies(false), did_set_metadata(false)
//
{
    if (frb2_weights_lifetime_in_samples <= 0)
        FATAL_ERROR("frb2_weights_lifetime_in_samples {:d} must be positive",
                    frb2_weights_lifetime_in_samples);
    if (frb2_weights_ring_depth <= 0)
        FATAL_ERROR("frb2_weights_ring_depth {:d} must be positive", frb2_weights_ring_depth);

    // Only instance 0 handles frame 0 and thus reads the metadata sources; the other instances
    // must not register, or the sources' producers would wait for them forever.
    if (instance_num == 0) {
        // A single buffer name or a list, as for `Stage::get_buffer_or_array`
        const std::vector<std::string> names =
            config.get_value(unique_name, "metadata_source").is_array()
                ? config.get<std::vector<std::string>>(unique_name, "metadata_source")
                : std::vector<std::string>{config.get<std::string>(unique_name, "metadata_source")};
        for (const std::string& name : names) {
            Buffer* const metadata_source = host_buffers.get_buffer(name);
            if (!metadata_source)
                FATAL_ERROR("metadata_source {:s} is not a frame buffer listed in the "
                            "cudaProcess's in_buffers",
                            name);
            metadata_source->register_consumer(unique_name);
            metadata_sources.push_back(metadata_source);
        }
    }

    frb2_beam_positions_buffer.register_consumer();
    frb2_weights_buffer.register_producer();
    if (instance_num == 0)
        register_gpu_buffer_user({.name = frb2_frequencies_name,
                                  .is_array = false,
                                  .does_read = true,
                                  .does_write = true});

    set_command_type(gpuCommandType::KERNEL);
}

bool cudaCalcFRB2Weights::setup_frequencies() {
    // The coarse frequency channels handled by this GPU. These are local to a GPU and thus cannot
    // come from the configuration, which is the same for every GPU. This blocks until the
    // metadata sources have produced their first frame; the voltage buffers do not depend on
    // anything downstream of them, so this cannot deadlock.
    const std::optional<std::vector<int>> local_coarse_freq =
        wait_for_coarse_freq(metadata_sources, unique_name);
    if (!local_coarse_freq)
        return false;

    const UpchannelizationSchedule upchan_schedule(config, upchannelization_schedule_name,
                                                   *local_coarse_freq, unique_name);

    const Telescope& telescope = Telescope::instance();
    for (const int channel : upchan_schedule.get_frequency_channels()) {
        const float frequency = telescope.to_freq_MHz(freq_id_t(channel)) * 1.0e+6f;
        const float frequency_spacing = telescope.freq_width_MHz(freq_id_t(channel)) * 1.0e+6f;
        const auto& upchan_factors = upchan_schedule.get_upchan_factors(channel);
        if (upchan_factors.empty()) {
            // Assume we keep the frequency itself
            coarse_freq.push_back(channel);
            freq_upchan_factor.push_back(1);
            freq_upchan_index.push_back(0);
            frequencies.push_back(frequency);
        } else {
            // Assume we do not keep the frequency itself, we only process the upchannelized ones
            for (const int upchan_factor : upchan_factors) {
                for (int upchan_index = 0; upchan_index < upchan_factor; ++upchan_index) {
                    const float upchan_frequency =
                        frequency
                        + frequency_spacing * ((upchan_index + 0.5f) / upchan_factor - 0.5f);
                    coarse_freq.push_back(channel);
                    freq_upchan_factor.push_back(upchan_factor);
                    freq_upchan_index.push_back(upchan_index);
                    frequencies.push_back(upchan_frequency);
                }
            }
        }
    }
    if (frequencies.size() != std::size_t(frb2_num_frequencies))
        FATAL_ERROR("The upchannelization schedule yields {:d} frequencies, but "
                    "frb2_num_frequencies is {:d}",
                    frequencies.size(), frb2_num_frequencies);

    return true;
}

int cudaCalcFRB2Weights::wait_on_precondition() {
    {
        const int errcode = cudaCommand::wait_on_precondition();
        if (errcode < 0)
            return errcode;
    }

    // Instance 0 handles frame 0, and it needs the frequencies for its first `execute`, which
    // uploads them for all instances. This may block, which is allowed here.
    if (instance_num == 0 && !did_setup_frequencies) {
        if (!setup_frequencies())
            return -1;
        did_setup_frequencies = true;
    }

    DEBUG("Waiting for {:s} output ringbuffer space for frame {:d}...", frb2_weights_name,
          gpu_frame_id);
    const int errcode = frb2_weights_buffer.wait_for_writable(1);
    if (errcode < 0)
        return errcode;
    DEBUG("Done waiting for {:s} output ringbuffer space for frame {:d}; writing element {:d}",
          frb2_weights_name, gpu_frame_id, frb2_weights_buffer.get_write_valid().begin());

    return 0;
}

cudaEvent_t cudaCalcFRB2Weights::execute(cudaPipelineState& /*pipestate*/,
                                         const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    const cudaStream_t stream = device.getStream(cuda_stream_id);

    frb2_beam_positions_buffer.check_metadata();
    const std::shared_ptr<const chordMetadata> positions_meta =
        frb2_beam_positions_buffer.get_metadata();
    if (positions_meta->get_time_downsampling_fpga() != frb2_weights_lifetime_in_samples)
        FATAL_ERROR("The {:s} stream has time_downsampling_fpga {:d}, but "
                    "frb2_weights_lifetime_in_samples is {:d}",
                    frb2_beam_positions_name, positions_meta->get_time_downsampling_fpga(),
                    frb2_weights_lifetime_in_samples);

    const std::ptrdiff_t element = frb2_weights_buffer.get_write_valid().begin();

    // Set the ring buffer metadata once; see `NDArrayRingBuffer::set_metadata`. The ring buffer
    // starts where the positions stream starts, and each element covers one lifetime.
    if (instance_num == 0 && !did_set_metadata) {
        did_set_metadata = true;
        assert(did_setup_frequencies);
        assert(element == 0);
        const std::shared_ptr<chordMetadata> weights_meta = std::make_shared<chordMetadata>();
        weights_meta->deepCopy(positions_meta);
        weights_meta->dim[0] = 1; // one weights matrix per positions frame
        weights_meta->set_fpga_seq_num(positions_meta->get_fpga_seq_num());
        weights_meta->set_time_downsampling_fpga(frb2_weights_lifetime_in_samples);
        weights_meta->set_coarse_freq(coarse_freq);
        weights_meta->set_freq_upchan_factor(freq_upchan_factor);
        weights_meta->set_freq_upchan_index(freq_upchan_index);
        frb2_weights_buffer.set_metadata(weights_meta);

        // The frequencies are the same for every weights matrix. Later instances run on the same
        // stream, after this copy.
        CHECK_CUDA_ERROR(cudaMemcpyAsync(frb2_frequencies_memory, frequencies.data(),
                                         frb2_num_frequencies * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));

        INFO("Calculating {:s} weights matrices of {:d} bytes each, one per {:d} FPGA samples, in "
             "a ring buffer of {:d} matrices",
             frb2_weights_name,
             frb2_weights_buffer.get_ndarray().get_stride(0) * std::ptrdiff_t(sizeof(float16_t)),
             frb2_weights_lifetime_in_samples, frb2_weights_ring_depth);
    }
    frb2_weights_buffer.check_metadata();

    // Positions frame `k` has to describe weights matrix `k`
    const std::shared_ptr<const chordMetadata> weights_meta = frb2_weights_buffer.get_metadata();
    const std::int64_t expected_seq_num =
        weights_meta->get_fpga_seq_num() + element * frb2_weights_lifetime_in_samples;
    if (positions_meta->get_fpga_seq_num() != expected_seq_num)
        FATAL_ERROR("{:s} frame {:d} has fpga_seq_num {:d}, expected {:d} (the stream must be "
                    "contiguous with a cadence of frb2_weights_lifetime_in_samples {:d})",
                    frb2_beam_positions_name, element, positions_meta->get_fpga_seq_num(),
                    expected_seq_num, frb2_weights_lifetime_in_samples);
    DEBUG("Calculating {:s} element {:d} for fpga_seq_num {:d}", frb2_weights_name, element,
          positions_meta->get_fpga_seq_num());

    // Vectors giving the feed separation in each axis direction in meters. These vectors are in
    // the GRID frame, where 'x' and 'y' are aligned with the feed grid array and are also
    // orthogonal. This makes the vectors very simple, with a single component in the x and y
    // directions respectively.
    const Telescope& telescope = Telescope::instance();
    const float sigmaM_x = frb1_swap_MN ? 0 : telescope.get_feed_separation_x_m();
    const float sigmaM_y = frb1_swap_MN ? telescope.get_feed_separation_y_m() : 0;
    const float sigmaM_z = 0;
    const float sigmaN_x = frb1_swap_MN ? telescope.get_feed_separation_x_m() : 0;
    const float sigmaN_y = frb1_swap_MN ? 0 : telescope.get_feed_separation_y_m();
    const float sigmaN_z = 0;

    // Write directly into the ring buffer element
    kotekan::NDArray<float16_t, 5>& W2 = frb2_weights_buffer.get_ndarray();
    assert(std::string(W2.get_dimname(1)) == "Fbar");
    float16_t* const W2_element = W2.data() + W2.get_stride(0) * mod(element, W2.get_extent(0));
    const float* const positions = frb2_beam_positions_buffer.get_ndarray().data();

    // The kernel's grid has one row per frequency, and gridDim.y must be < 65536
    const int max_chunk = 65535;
    for (int freq0 = 0; freq0 < frb2_num_frequencies; freq0 += max_chunk) {
        const int nfreq = std::min(max_chunk, frb2_num_frequencies - freq0);
        cuda_calc_frb2_weights(frb2_frequencies_memory + freq0, positions,
                               W2_element + freq0 * W2.get_stride(1), nfreq, frb2_num_beams,
                               num_dishes_M, num_dishes_N, sigmaM_x, sigmaM_y, sigmaM_z, sigmaN_x,
                               sigmaN_y, sigmaN_z, stream);
    }

    return record_end_event();
}

void cudaCalcFRB2Weights::finalize_frame() {
    // Publish the weights matrix
    frb2_weights_buffer.finish_write();

    cudaCommand::finalize_frame();
}
