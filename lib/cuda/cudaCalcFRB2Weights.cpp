// GPU version of calcFRB2Weights: calculates the FRB2 beamforming weights on a CUDA device.
// This is an independent alternative to the (slow) CPU stage calcFRB2Weights; both produce
// the same weights up to floating-point differences (see cudaCalcFRB2WeightsKernel.hpp).

#include "Config.hpp"                   // for Config
#include "DataType.hpp"                 // for float16_t
#include "NDArray.hpp"                  // for GenericNDArray
#include "Stage.hpp"                    // for Stage
#include "StageFactory.hpp"             // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"                // for Telescope, freq_id_t
#include "UpchannelizationSchedule.hpp" // for UpchannelizationSchedule
#include "buffer.hpp"                   // for Buffer
#include "bufferContainer.hpp"          // for bufferContainer
#include "chordMetadata.hpp"            // for chordMetadata, get_chord_metadata
#include "cudaCalcFRB2WeightsKernel.hpp"
#include "cudaUtils.hpp"      // for CHECK_CUDA_ERROR
#include "kotekanLogging.hpp" // for DEBUG, INFO, WARN, FATAL_ERROR
#include "visUtil.hpp"        // for current_time

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm> // for clamp, min
#include <cassert>   // for assert
#include <cstddef>   // for ptrdiff_t, size_t
#include <cstdint>   // for int64_t
#include <cuda_runtime.h>
#include <functional> // for function
#include <memory>     // for __shared_ptr_access, shared_ptr
#include <string>     // for allocator, basic_string, string
#include <unistd.h>   // for sleep
#include <vector>     // for vector

class cudaCalcFRB2Weights : public kotekan::Stage {
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

    // Upper limit for the GPU scratch buffer holding W2 before it is copied to the host; the
    // calculation is split into chunks of frequencies to respect this limit
    const std::int64_t gpu_max_chunk_bytes =
        config.get_default<std::int64_t>(unique_name, "gpu_max_chunk_bytes", std::int64_t(2) << 30);

    const std::ptrdiff_t frb2_beam_positions_frame_size [[maybe_unused]] =
        sizeof(float) * 2 * frb2_num_beams;
    const std::ptrdiff_t W2_frame_size [[maybe_unused]] = sizeof(float16_t) * frb1_num_beams_P
                                                          * frb1_num_beams_Q * frb2_num_beams
                                                          * frb2_num_frequencies;

    Buffer* const frb2_beam_positions_buffer;
    Buffer* const W2_buffer;
    Buffer* const metadata_buffer;

public:
    cudaCalcFRB2Weights(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        frb2_beam_positions_buffer(get_buffer("frb2_beam_positions")),
        W2_buffer(get_buffer("frb2_weights")),
        metadata_buffer(get_buffer("metadata"))
    //
    {
        assert(frb2_beam_positions_buffer);
        assert(W2_buffer);
        if (gpu_max_chunk_bytes <= 0)
            FATAL_ERROR("gpu_max_chunk_bytes {:d} must be positive", gpu_max_chunk_bytes);
        frb2_beam_positions_buffer->register_consumer(unique_name);
        W2_buffer->register_producer(unique_name);
        metadata_buffer->register_consumer(unique_name);

        frb2_beam_positions_buffer->require_frame_desc(kotekan::NDArray<float, 2>::describe(
            "frb2_beam_positions", {frb2_num_beams, 2}, {"R", "X/Y"}, {1, 1}));
        W2_buffer->require_frame_desc(kotekan::NDArray<float16_t, 4>::describe(
            "W2", {frb2_num_frequencies, frb2_num_beams, frb1_num_beams_Q, frb1_num_beams_P},
            {"Fbar", "R", "beamQ", "beamP"}, {1, 1, 1, 1}));
    }

    virtual ~cudaCalcFRB2Weights() {}

    void main_thread() override {
        // Only calculate a single frame
        const int frame_index = 0;
        const int frame_id = frame_index;

        if (stop_thread)
            return;

        // Telescope
        const Telescope& telescope = Telescope::instance();

        // Upchannelization schedule
        uint8_t const * const metadata_frame = metadata_buffer->wait_for_full_frame(unique_name, 0);
        if(metadata_frame == nullptr)
            return;
        const auto& upchan_schedule = *[&]() {
            const auto coarse_freq = get_chord_metadata(metadata_buffer, 0)->get_coarse_freq();
            return &UpchannelizationSchedule::instance(config, upchannelization_schedule_name, coarse_freq);
        }();
        metadata_buffer->mark_frame_empty(unique_name, 0);
        metadata_buffer->unregister_consumer(unique_name);

        // Calculate frequencies
        const auto& frequency_channels = upchan_schedule.get_frequency_channels();
        std::vector<int> coarse_freq;
        std::vector<int> freq_upchan_factor;
        std::vector<int> freq_upchan_index;
        std::vector<float> frequencies;
        for (const int channel : frequency_channels) {
            const float frequency = telescope.to_freq_MHz(freq_id_t(channel)) * 1.0e+6f;
            const float frequency_spacing = telescope.freq_width_MHz(freq_id_t(channel)) * 1.0e+6f;
            const auto& upchan_factors = upchan_schedule.get_upchan_factors(channel);
            if (upchan_factors.empty()) {
                // Assume we keep the frequency itself
                coarse_freq.push_back(channel);
                freq_upchan_factor.push_back(1);
                freq_upchan_index.push_back(1);
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
        assert(frequencies.size() == std::size_t(frb2_num_frequencies));

        // Wait for buffers
        DEBUG("[{:s}/{:d}] Waiting for buffer...", frb2_beam_positions_buffer->buffer_name,
              frame_index);
        float* const frb2_beam_positions_frame = static_cast<float*>(static_cast<void*>(
            frb2_beam_positions_buffer->wait_for_full_frame(unique_name, frame_id)));
        if (!frb2_beam_positions_frame)
            return;

        DEBUG("[{:s}/{:d}] Waiting for buffer...", W2_buffer->buffer_name, frame_index);
        float16_t* const W2_frame = static_cast<float16_t*>(
            static_cast<void*>(W2_buffer->wait_for_empty_frame(unique_name, frame_id)));
        if (!W2_frame)
            return;

        // Check buffer sizes
        assert(std::ptrdiff_t(frb2_beam_positions_buffer->frame_size)
               == frb2_beam_positions_frame_size);
        assert(std::ptrdiff_t(W2_buffer->frame_size) == W2_frame_size);

        // Set metadata
        W2_buffer->allocate_new_metadata_object(frame_id);
        const auto& W2_meta = get_chord_metadata(W2_buffer->get_metadata(frame_id));
        W2_meta->set_from_frame_desc(W2_buffer->get_frame_desc<kotekan::GenericNDArray>());
        W2_meta->set_fpga_seq_num(0);           // ???
        W2_meta->set_time_downsampling_fpga(1); // ???
        W2_meta->set_coarse_freq(coarse_freq);
        W2_meta->set_freq_upchan_factor(freq_upchan_factor);
        W2_meta->set_freq_upchan_index(freq_upchan_index);

        // Set W2
        {
            // Start timer
            DEBUG("Calculating FRB2 beam weights...");
            const double t0 = current_time();

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

            // Split the calculation into chunks of frequencies such that the GPU scratch
            // buffer stays below `gpu_max_chunk_bytes` (and the grid fits into gridDim.y)
            const std::ptrdiff_t slab_bytes = str_freq * std::ptrdiff_t(sizeof(float16_t));
            const int max_chunk =
                std::clamp<std::ptrdiff_t>(gpu_max_chunk_bytes / slab_bytes, 1,
                                           std::min<std::ptrdiff_t>(frb2_num_frequencies, 65535));

            float* d_frequencies = nullptr;
            float* d_positions = nullptr;
            float16_t* d_W2 = nullptr;
            CHECK_CUDA_ERROR(cudaMalloc(&d_frequencies, frb2_num_frequencies * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_positions, frb2_beam_positions_buffer->frame_size));
            CHECK_CUDA_ERROR(cudaMalloc(&d_W2, max_chunk * slab_bytes));

            CHECK_CUDA_ERROR(cudaMemcpy(d_frequencies, frequencies.data(),
                                        frb2_num_frequencies * sizeof(float),
                                        cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_positions, frb2_beam_positions_frame,
                                        frb2_beam_positions_buffer->frame_size,
                                        cudaMemcpyHostToDevice));

            // Pinning the output frame speeds up the device-to-host copies below, but
            // pinning can fail for very large frames (locked-memory limits); fall back to
            // pageable copies in that case
            const cudaError_t register_error =
                cudaHostRegister(W2_frame, W2_buffer->frame_size, cudaHostRegisterDefault);
            const bool host_registered = register_error == cudaSuccess;
            if (!host_registered) {
                (void)cudaGetLastError(); // clear the error
                WARN("Could not pin the W2 frame ({:s}); using slower pageable copies",
                     cudaGetErrorString(register_error));
            }

            for (int freq0 = 0; freq0 < frb2_num_frequencies; freq0 += max_chunk) {
                const int nfreq = std::min(max_chunk, frb2_num_frequencies - freq0);
                cuda_calc_frb2_weights(d_frequencies + freq0, d_positions, d_W2, nfreq,
                                       frb2_num_beams, num_dishes_M, num_dishes_N, sigmaM_x,
                                       sigmaM_y, sigmaM_z, sigmaN_x, sigmaN_y, sigmaN_z, nullptr);
                // This synchronizes with the kernel (same stream, synchronous copy). No
                // copy/compute overlap: this stage runs only once, a few seconds suffice.
                CHECK_CUDA_ERROR(cudaMemcpy(W2_frame + freq0 * str_freq, d_W2, nfreq * slab_bytes,
                                            cudaMemcpyDeviceToHost));
                INFO("cudaCalcFRB2Weights: freqs: {:d}/{:d}...", freq0 + nfreq,
                     frb2_num_frequencies);
            }

            if (host_registered)
                CHECK_CUDA_ERROR(cudaHostUnregister(W2_frame));
            CHECK_CUDA_ERROR(cudaFree(d_W2));
            CHECK_CUDA_ERROR(cudaFree(d_positions));
            CHECK_CUDA_ERROR(cudaFree(d_frequencies));

            const double t1 = current_time();
            const double elapsed = t1 - t0;
            DEBUG("Calculated FRB2 beam weights in {} seconds", elapsed);
        }

        // Mark buffers as full
        DEBUG("[{:s}/{:d}] Marking buffer as empty...", frb2_beam_positions_buffer->buffer_name,
              frame_index);
        frb2_beam_positions_buffer->mark_frame_empty(unique_name, frame_id);

        DEBUG("[{:s}/{:d}] Marking buffer as full...", W2_buffer->buffer_name, frame_index);
        W2_buffer->mark_frame_full(unique_name, frame_id);

        // Wait for shutdown (don't trigger a shutdown)
        while (!stop_thread)
            sleep(1);
    }
};

REGISTER_KOTEKAN_STAGE(cudaCalcFRB2Weights);
