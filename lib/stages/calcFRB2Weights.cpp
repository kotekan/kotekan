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
#include "kotekanLogging.hpp"           // for DEBUG, FATAL_ERROR
#include "visUtil.hpp"                  // for current_time

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm> // for max
#include <atomic>
#include <cassert> // for assert
#include <cassert>
#include <cmath>   // for sin, cos, sqrt, M_PI
#include <cstddef> // for ptrdiff_t, size_t
#include <cstdint>
#include <cstdio>
#include <functional> // for function
#include <memory>     // for __shared_ptr_access, shared_ptr
#include <set>        // for set, operator!=, _Rb_tree_const_iterator
#ifdef WITH_OMP
#include <omp.h>
#endif
#include <string>   // for allocator, basic_string, string
#include <unistd.h> // for sleep
#include <vector>   // for vector

class calcFRB2Weights : public kotekan::Stage {
    // Telescope setup
    const int num_dishes = config.get<int>(unique_name, "num_dishes");

    // Upchannelization setup
    const std::string upchannelization_schedule_name =
        config.get_default<std::string>(unique_name, "upchannelization_schedule_name", "");

    // FRB1 beamformer setup

    // The directions "x" and "y" are the geographic direction
    // east-west and north-south. The directions we call "M" and "N"
    // are really just describing the in-memory layout of the dishes
    // (antennae) in the dish array. For CHORD, the physical
    // x-direction (east-west) runs fastest, but for CHIME, the
    // y-direction (north-south) runs fastest. This is necessary to
    // get an efficient FRB1 beamformer given the different shapes of
    // the dish (antenna) layouts.
    //
    // Internally (in the FRB1 kernels) we call the dish (antenna)
    // directions M and N, where M runs fastest, and the respective
    // beam directions P and Q, where P runs fastest. For CHORD,
    // M=P=x=east/west and N=Q=y=north/south. For CHIME it's the
    // converse, M=P=y=north/south and N=Q=x=east/west.
    //
    // We need to take this into account when creating the FRB2
    // beamforming weights. For CHORD we have `frb1_swap_MN=false`,
    // and for CHIME we have `frb1_swap_MN=true`.
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

    const int num_threads = config.get_default<int>(unique_name, "num_threads", 1);

    const std::ptrdiff_t frb2_beam_positions_frame_size [[maybe_unused]] =
        sizeof(float) * 2 * frb2_num_beams;
    const std::ptrdiff_t W2_frame_size [[maybe_unused]] = sizeof(float16_t) * frb1_num_beams_P
                                                          * frb1_num_beams_Q * frb2_num_beams
                                                          * frb2_num_frequencies;

    Buffer* const frb2_beam_positions_buffer;
    Buffer* const W2_buffer;

public:
    calcFRB2Weights(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        frb2_beam_positions_buffer(get_buffer("frb2_beam_positions")),
        W2_buffer(get_buffer("frb2_weights"))
    //
    {
        assert(frb2_beam_positions_buffer);
        assert(W2_buffer);
        if (num_threads < 0)
            FATAL_ERROR("num_threads %d must be positive", num_threads);
        frb2_beam_positions_buffer->register_consumer(unique_name);
        W2_buffer->register_producer(unique_name);

        frb2_beam_positions_buffer->require_frame_desc(kotekan::NDArray<float, 2>::describe(
            "frb2_beam_positions", {frb2_num_beams, 2}, {"R", "X/Y"}, {1, 1}));
        W2_buffer->require_frame_desc(kotekan::NDArray<float16_t, 4>::describe(
            "W2", {frb2_num_frequencies, frb2_num_beams, frb1_num_beams_Q, frb1_num_beams_P},
            {"Fbar", "R", "beamQ", "beamP"}, {1, 1, 1, 1}));
    }

    virtual ~calcFRB2Weights() {}

    void main_thread() override {
        // Only calculate a single frame
        const int frame_index = 0;
        const int frame_id = frame_index;

        if (stop_thread)
            return;

        // Telescope
        const Telescope& telescope = Telescope::instance();

        // Upchannelization schedule
        const auto& upchan_schedule =
            UpchannelizationSchedule::instance(config, upchannelization_schedule_name);

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

            using std::cos, std::sin, std::sqrt;

            const float c0 = 299792458.0f; // speed of light in vacuum [m/s]

            // This matches a function defined in Kendrick's beamforming note (eqn
            // 7/8).
            const auto Ufunc = [](int p, int M, float theta) {
                float acc = 0.0f;
                for (int s = 0; s <= M; ++s) {
                    float A = s == 0 || s == M ? 0.5f : 1.0f;
                    acc += A * cos(float(M_PI) * (2 * theta - p) * s / M);
                }
                return acc / M;
            };

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

            std::atomic<int> nfreqs_done = 0;

#ifdef WITH_OMP
#pragma omp parallel num_threads(num_threads)
#endif
            {
                std::vector<float> Up(frb1_num_beams_P);
                std::vector<float> Uq(frb1_num_beams_Q);

#ifdef WITH_OMP
#pragma omp for
#endif
                for (int freq = 0; freq < frb2_num_frequencies; ++freq) {
                    // Calculate physical frequency from channel index
                    const float afreq = frequencies.at(freq);
                    const float wavelength = c0 / afreq;

                    for (int beamR = 0; beamR < frb2_num_beams; ++beamR) {
                        // Unit vector pointing to sky location in the GRID frame
                        const float nx = frb2_beam_positions_frame[2 * beamR + 0];
                        const float ny = frb2_beam_positions_frame[2 * beamR + 1];
                        const float nz = sqrt(1 - (nx * nx + ny * ny));

                        // Kendrick's FRB beamforming notes, equation 7:
                        //   theta = M (nhat ⋅ sigma) / lambda
                        // where nhat is the unit vector in the direction of the sky location
                        // sigma is the dish displacement in meters East-West.
                        const float theta_M = num_dishes_M
                                              * (nx * sigmaM_x + ny * sigmaM_y + nz * sigmaM_z)
                                              / wavelength;
                        const float theta_N = num_dishes_N
                                              * (nx * sigmaN_x + ny * sigmaN_y + nz * sigmaN_z)
                                              / wavelength;

                        for (int i = 0; i < frb1_num_beams_P; ++i)
                            Up[i] = Ufunc(i, num_dishes_M, theta_M);
                        for (int j = 0; j < frb1_num_beams_Q; ++j)
                            Uq[j] = Ufunc(j, num_dishes_N, theta_N);

                        for (int beamQ = 0; beamQ < frb1_num_beams_Q; ++beamQ) {
                            for (int beamP = 0; beamP < frb1_num_beams_P; ++beamP) {
                                const std::ptrdiff_t idx = str_beamP * beamP + str_beamQ * beamQ
                                                           + str_beamR * beamR + str_freq * freq;
                                assert(idx >= 0
                                       && idx < std::ptrdiff_t(W2_buffer->frame_size
                                                               / sizeof *W2_frame));
                                W2_frame[idx] = float16_t(Up[beamP] * Uq[beamQ]);
                            }
                        }
                    }
                    ++nfreqs_done;
                    std::printf("\rcalcFRB2Weights: freqs: %d/%d...", int(nfreqs_done),
                                frb2_num_frequencies);
                    std::fflush(stdout);
                }
            }
            std::printf("\n");

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

REGISTER_KOTEKAN_STAGE(calcFRB2Weights);
