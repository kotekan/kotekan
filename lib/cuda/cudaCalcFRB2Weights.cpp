// GPU version of calcFRB2Weights: calculates the FRB2 beamforming weights on a CUDA device.
// This is an independent alternative to the (slow) CPU stage calcFRB2Weights; both produce
// the same weights up to floating-point differences (see cudaCalcFRB2WeightsKernel.hpp).
//
// This is a cudaCommand. Each invocation turns one frame of beam positions into one weights
// matrix, written directly into its element of the `W2` ring buffer that cudaFRBBeamReformer
// reads; the weights never leave the GPU. A positions frame, and hence a weights matrix, is valid
// for `frb2_weights_lifetime_in_samples` FPGA samples, and frame `k` must have the
// `fpga_seq_num` `seq0 + k * frb2_weights_lifetime_in_samples`. The positions have to be copied
// to the GPU beforehand by a (non-`do_once`) cudaInputData and cudaSyncInput in the same
// cudaProcess. The configuration keys `frb2_beam_positions_name` and `frb2_weights_name` name the
// GPU buffers (without `_buffer`), and `frb2_weights_ring_depth` (default 2) is the number of
// weights matrices in the ring buffer. The `metadata_source` buffers must also be listed in the
// cudaProcess's `in_buffers`.

#include "Config.hpp"                   // for Config
#include "DataType.hpp"                 // for float16_t
#include "NDArray.hpp"                  // for NDArray
#include "NDArrayBuffer.hpp"            // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"        // for NDArrayRingBuffer
#include "Telescope.hpp"                // for Telescope, freq_id_t
#include "UpchannelizationSchedule.hpp" // for UpchannelizationSchedule, wait_for_coarse_freq
#include "buffer.hpp"                   // for Buffer
#include "bufferContainer.hpp"          // for bufferContainer
#include "chordMetadata.hpp"            // for chordMetadata
#include "cudaCalcFRB2WeightsKernel.hpp"
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "div.hpp"                 // for mod
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG, INFO, FATAL_ERROR

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm> // for min
#include <array>     // for array
#include <cassert>   // for assert
#include <cstddef>   // for ptrdiff_t, size_t
#include <cstdint>   // for int64_t
#include <cuda_runtime.h>
#include <memory>   // for __shared_ptr_access, shared_ptr, make_shared
#include <optional> // for optional
#include <string>   // for allocator, basic_string, string
#include <vector>   // for vector

class cudaCalcFRB2Weights : public cudaCommand {
    // Telescope setup
    const int num_dishes = config.get<int>(unique_name, "num_dishes");

    // Upchannelization setup
    const std::string upchannelization_schedule_name =
        config.get_default<std::string>(unique_name, "upchannelization_schedule_name", "");

    // FRB1 beamformer setup

    // See calcFRB2Weights for the meaning of the directions M/N and P/Q and of `frb1_swap_MN`.
    const int num_dishes_x = Telescope::instance().get_grid_size_x();
    const int num_dishes_y = Telescope::instance().get_grid_size_y();
    const bool frb1_swap_MN = config.get_default<bool>(unique_name, "frb1_swap_MN", false);
    const int num_dishes_M = frb1_swap_MN ? num_dishes_y : num_dishes_x;
    const int num_dishes_N = frb1_swap_MN ? num_dishes_x : num_dishes_y;
    const int frb1_num_beams_P = 2 * num_dishes_M;
    const int frb1_num_beams_Q = 2 * num_dishes_N;

    // FRB2 beamformer setup
    const int frb2_num_beams_x = config.get<int>(unique_name, "frb2_num_beams_x");
    const int frb2_num_beams_y = config.get<int>(unique_name, "frb2_num_beams_y");
    const int frb2_num_beams = frb2_num_beams_x * frb2_num_beams_y;
    const int frb2_num_frequencies = config.get<int>(unique_name, "frb2_num_frequencies");

    // Lifetime of a weights matrix in FPGA samples, and the number of matrices in the ring buffer
    const std::int64_t frb2_weights_lifetime_in_samples =
        config.get<std::int64_t>(unique_name, "frb2_weights_lifetime_in_samples");
    const int frb2_weights_ring_depth =
        config.get_default<int>(unique_name, "frb2_weights_ring_depth", 2);

    const std::ptrdiff_t frb2_beam_positions_frame_size [[maybe_unused]] =
        sizeof(float) * 2 * frb2_num_beams;
    const std::ptrdiff_t W2_frame_size [[maybe_unused]] = sizeof(float16_t) * frb1_num_beams_P
                                                          * frb1_num_beams_Q * frb2_num_beams
                                                          * frb2_num_frequencies;

    // GPU buffer names
    const std::string frb2_beam_positions_name =
        config.get<std::string>(unique_name, "frb2_beam_positions_name");
    const std::string frb2_weights_name = config.get<std::string>(unique_name, "frb2_weights_name");

    NDArrayBuffer<float, 2> frb2_beam_positions_buffer{frb2_beam_positions_name,
                                                       "frb2_beam_positions",
                                                       {frb2_num_beams, 2},
                                                       std::array<std::string, 2>{"R", "X/Y"},
                                                       {1, 1},
                                                       *this};
    NDArrayRingBuffer<float16_t, 5> W2_buffer{
        frb2_weights_name,
        "W2",
        {frb2_weights_ring_depth, frb2_num_frequencies, frb2_num_beams, frb1_num_beams_Q,
         frb1_num_beams_P},
        std::array<std::string, 5>{"TW2", "Fbar", "R", "beamQ", "beamP"},
        {frb2_weights_lifetime_in_samples, 1, 1, 1, 1},
        *this};
    // Only instance 0 handles frame 0 and thus reads the metadata sources; the other instances must
    // not register, or the sources' producers would wait for them forever.
    const std::vector<Buffer*> metadata_sources =
        instance_num == 0 ? get_buffer_or_array("metadata_source") : std::vector<Buffer*>{};

    // The frequencies are calculated by instance 0 before its first `execute`, which uploads them
    // for all instances
    float* const d_frequencies = static_cast<float*>(device.get_gpu_memory(
        frb2_weights_name + "_frequencies_buffer", frb2_num_frequencies * sizeof(float)));
    bool did_calculate_frequencies = false;
    std::vector<int> coarse_freq;
    std::vector<int> freq_upchan_factor;
    std::vector<int> freq_upchan_index;
    std::vector<float> frequencies;

    // Set once, on the first frame; see `NDArrayRingBuffer::set_metadata`
    bool did_set_metadata = false;

    // A single buffer name or a list, as for `Stage::get_buffer_or_array`
    std::vector<Buffer*> get_buffer_or_array(const std::string& name) {
        const std::vector<std::string> buffer_names =
            config.get_value(unique_name, name).is_array()
                ? config.get<std::vector<std::string>>(unique_name, name)
                : std::vector<std::string>{config.get<std::string>(unique_name, name)};
        std::vector<Buffer*> buffers;
        for (const std::string& buffer_name : buffer_names)
            buffers.push_back(host_buffers.get_buffer(buffer_name));
        return buffers;
    }

public:
    cudaCalcFRB2Weights(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                        const int instance_num) :
        cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                    "cudaCalcFRB2Weights")
    //
    {
        if (frb2_weights_lifetime_in_samples <= 0)
            FATAL_ERROR("frb2_weights_lifetime_in_samples {:d} must be positive",
                        frb2_weights_lifetime_in_samples);
        if (frb2_weights_ring_depth <= 0)
            FATAL_ERROR("frb2_weights_ring_depth {:d} must be positive", frb2_weights_ring_depth);
        frb2_beam_positions_buffer.register_consumer();
        W2_buffer.register_producer();
        for (Buffer* const metadata_source : metadata_sources) {
            assert(metadata_source);
            metadata_source->register_consumer(unique_name);
        }

        set_command_type(gpuCommandType::KERNEL);
    }

    virtual ~cudaCalcFRB2Weights() {}

    int wait_on_precondition() override {
        {
            const int errcode = cudaCommand::wait_on_precondition();
            if (errcode < 0)
                return errcode;
        }

        // This may block, which is allowed here, but not in `execute`
        if (instance_num == 0 && !did_calculate_frequencies) {
            if (!calculate_frequencies())
                return -1;
            did_calculate_frequencies = true;
        }

        DEBUG("[{:s}/{:d}] Waiting for ring buffer space...", frb2_weights_name, gpu_frame_id);
        return W2_buffer.wait_for_writable(1);
    }

    // Returns false if the pipeline is shutting down
    bool calculate_frequencies() {
        // Telescope
        const Telescope& telescope = Telescope::instance();

        // Upchannelization schedule
        // The coarse frequency channels handled by this GPU. These are local to
        // a GPU and thus cannot come from the configuration, which is the same
        // for every GPU. This waits for the first frame of the metadata sources;
        // these are the voltage buffers, which do not depend on anything
        // downstream, so this cannot deadlock.
        const auto local_coarse_freq = wait_for_coarse_freq(metadata_sources, unique_name);
        if (!local_coarse_freq)
            return false;

        const UpchannelizationSchedule upchan_schedule(config, upchannelization_schedule_name,
                                                       *local_coarse_freq, unique_name);

        // Calculate frequencies
        const auto& frequency_channels = upchan_schedule.get_frequency_channels();
        for (const int channel : frequency_channels) {
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
                // Assume we do not keep the frequency itself, we only process the upchannelized
                // ones
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

    cudaEvent_t execute(cudaPipelineState& /*pipestate*/,
                        const std::vector<cudaEvent_t>& /*pre_events*/) override {
        pre_execute();
        record_start_event();

        // Telescope
        const Telescope& telescope = Telescope::instance();

        const cudaStream_t stream = device.getStream(cuda_stream_id);

        // The ring buffer element we write
        const std::ptrdiff_t W2_element = W2_buffer.get_write_valid().begin();

        // Check buffer sizes
        assert(W2_buffer.get_ndarray().get_stride(0) * std::ptrdiff_t(sizeof(float16_t))
               == W2_frame_size);

        // Check metadata
        frb2_beam_positions_buffer.check_metadata();
        const std::shared_ptr<const chordMetadata> positions_meta =
            frb2_beam_positions_buffer.get_metadata();
        if (positions_meta->get_time_downsampling_fpga() != frb2_weights_lifetime_in_samples)
            FATAL_ERROR("{:s} has time_downsampling_fpga {:d}, but "
                        "frb2_weights_lifetime_in_samples is {:d}",
                        frb2_beam_positions_name, positions_meta->get_time_downsampling_fpga(),
                        frb2_weights_lifetime_in_samples);

        // Set metadata, once: the ring buffer starts where the positions stream starts, and each
        // element covers one lifetime. Also upload the frequencies, which are the same for every
        // weights matrix; later instances run on the same stream, after this copy.
        if (instance_num == 0 && !did_set_metadata) {
            did_set_metadata = true;
            assert(did_calculate_frequencies);
            assert(W2_element == 0);
            const auto W2_meta = std::make_shared<chordMetadata>();
            W2_meta->deepCopy(positions_meta);
            W2_meta->dim[0] = 1; // one weights matrix per positions frame
            W2_meta->set_fpga_seq_num(positions_meta->get_fpga_seq_num());
            W2_meta->set_time_downsampling_fpga(frb2_weights_lifetime_in_samples);
            W2_meta->set_coarse_freq(coarse_freq);
            W2_meta->set_freq_upchan_factor(freq_upchan_factor);
            W2_meta->set_freq_upchan_index(freq_upchan_index);
            W2_buffer.set_metadata(W2_meta);

            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_frequencies, frequencies.data(),
                                             frb2_num_frequencies * sizeof(float),
                                             cudaMemcpyHostToDevice, stream));

            INFO("Calculating {:s}: one weights matrix of {:d} bytes per {:d} FPGA samples, in a "
                 "ring buffer of {:d} matrices",
                 frb2_weights_name, W2_frame_size, frb2_weights_lifetime_in_samples,
                 frb2_weights_ring_depth);
        }
        W2_buffer.check_metadata();

        // Positions frame `k` has to describe weights matrix `k`
        const std::int64_t expected_seq_num = W2_buffer.get_metadata()->get_fpga_seq_num()
                                              + W2_element * frb2_weights_lifetime_in_samples;
        if (positions_meta->get_fpga_seq_num() != expected_seq_num)
            FATAL_ERROR("{:s} frame {:d} has fpga_seq_num {:d}, expected {:d}",
                        frb2_beam_positions_name, W2_element, positions_meta->get_fpga_seq_num(),
                        expected_seq_num);

        // Set W2
        {
            DEBUG("Calculating FRB2 beam weights for element {:d} (fpga_seq_num {:d})...",
                  W2_element, positions_meta->get_fpga_seq_num());

            const std::ptrdiff_t str_beamP = 1;
            const std::ptrdiff_t str_beamQ = str_beamP * frb1_num_beams_P;
            const std::ptrdiff_t str_beamR = str_beamQ * frb1_num_beams_Q;
            const std::ptrdiff_t str_freq = str_beamR * frb2_num_beams;

            // Vectors giving the feed separation in each axis direction in meters.
            // These vectors are in the GRID frame, where 'x' and 'y' are aligned
            // with the feed grid array and are also orthogonal. This makes the
            // vectors very simple, with a single component in the x and y directions
            // respectively.
            const float sigmaM_x = frb1_swap_MN ? 0 : telescope.get_feed_separation_x_m();
            const float sigmaM_y = frb1_swap_MN ? telescope.get_feed_separation_y_m() : 0;
            const float sigmaM_z = 0;

            const float sigmaN_x = frb1_swap_MN ? telescope.get_feed_separation_x_m() : 0;
            const float sigmaN_y = frb1_swap_MN ? 0 : telescope.get_feed_separation_y_m();
            const float sigmaN_z = 0;

            // Split the calculation into chunks of frequencies such that the grid fits into
            // gridDim.y
            const int max_chunk = std::min(frb2_num_frequencies, 65535);

            // Write directly into the ring buffer element
            const float* const d_positions = frb2_beam_positions_buffer.get_ndarray().data();
            float16_t* const d_W2 =
                W2_buffer.get_ndarray().data()
                + W2_buffer.get_ndarray().get_stride(0)
                      * kotekan::mod(W2_element, W2_buffer.get_ndarray().get_extent(0));

            for (int freq0 = 0; freq0 < frb2_num_frequencies; freq0 += max_chunk) {
                const int nfreq = std::min(max_chunk, frb2_num_frequencies - freq0);
                cuda_calc_frb2_weights(d_frequencies + freq0, d_positions, d_W2 + freq0 * str_freq,
                                       nfreq, frb2_num_beams, num_dishes_M, num_dishes_N, sigmaM_x,
                                       sigmaM_y, sigmaM_z, sigmaN_x, sigmaN_y, sigmaN_z, stream);
            }
        }

        return record_end_event();
    }

    void finalize_frame() override {
        // Publish the weights matrix
        W2_buffer.finish_write();

        cudaCommand::finalize_frame();
    }
};

REGISTER_CUDA_COMMAND(cudaCalcFRB2Weights);
