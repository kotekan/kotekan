#include <unistd.h>             // for sleep
#include <cassert>              // for assert
#include <cstdint>              // for int8_t, int32_t
#include <string>               // for basic_string, string
#include <vector>               // for vector
#include <algorithm>            // for clamp
#include <array>                // for array
#include <cmath>                // for sin, lrint, sqrt, M_PI
#include <complex>              // for complex, imag, polar, real
#include <cstddef>              // for ptrdiff_t
#include <functional>           // for function
#include <memory>               // for allocator, __shared_ptr_access, shared_ptr

#include "Config.hpp"           // for Config
#include "Stage.hpp"            // for Stage
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "chordMetadata.hpp"    // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"   // for DEBUG
#include "Telescope.hpp"        // for Telescope
#include "fmt.hpp"              // for compile_string_to_view, format


class calcBBPhase : public kotekan::Stage {
    // Telescope setup
    const int num_components = config.get<int>(unique_name, "num_components");
    const int num_dishes = config.get<int>(unique_name, "num_dishes");
    const int num_polarizations = config.get<int>(unique_name, "num_polarizations");
    const int num_frequencies = config.get<int>(unique_name, "num_frequencies");
    const int num_times = config.get<int>(unique_name, "num_times");
    const ElementOrder input_order = config.get<ElementOrder>(unique_name, "input_order");
    const int num_elements = num_dishes * num_polarizations;

    const std::vector<int> frequency_channels =
        config.get<std::vector<int>>(unique_name, "frequency_channels");

    // Baseband beamformer setup
    const int bb_num_beams = config.get<int>(unique_name, "bb_num_beams");
    const int bb_scale = config.get<int>(unique_name, "bb_scale");

    const std::ptrdiff_t bb_beam_positions_frame_size [[maybe_unused]] =
        sizeof(float) * 2 * bb_num_beams;
    const std::ptrdiff_t A_frame_size [[maybe_unused]] = sizeof(std::int8_t) * num_components
                                                         * num_dishes * bb_num_beams
                                                         * num_polarizations * num_frequencies;
    const std::ptrdiff_t s_frame_size [[maybe_unused]] =
        sizeof(std::int32_t) * bb_num_beams * num_polarizations * num_frequencies;

    // Buffers
    Buffer* const bb_beam_positions_buffer;
    Buffer* const A_buffer;
    Buffer* const s_buffer;

public:
    calcBBPhase(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        bb_beam_positions_buffer(get_buffer("bb_beam_positions")), A_buffer(get_buffer("bb_phase")),
        s_buffer(get_buffer("bb_shift"))
    //
    {
        assert(bb_beam_positions_buffer);
        assert(A_buffer);
        assert(s_buffer);
        bb_beam_positions_buffer->register_consumer(unique_name);
        A_buffer->register_producer(unique_name);
        s_buffer->register_producer(unique_name);

        bb_beam_positions_buffer->allocate_ndarray_frame_desc<float, 2>(
            "bb_beam_positions", {bb_num_beams, 2}, {"B", "X/Y"});
        
        A_buffer->allocate_ndarray_frame_desc<std::int8_t, 5>(
            "A", {num_frequencies, num_polarizations, bb_num_beams, num_dishes, num_components},
            {"F", "P", "B", "D", "C"});

        s_buffer->allocate_ndarray_frame_desc<std::int32_t, 3>(
            "s", {num_frequencies, num_polarizations, bb_num_beams}, {"F", "P", "B"});
    }

    virtual ~calcBBPhase() {}

    void main_thread() override {
        // Only calculate a single frame
        const int frame_index = 0;
        const int frame_id = frame_index;

        if (stop_thread)
            return;

        // Telescope
        const Telescope& telescope = Telescope::instance();

        // Get dish positions (in the Telescope's GRID frame in meters).
        std::vector<vec3d_t> feed_pos_m = telescope.get_feed_positions_m(num_elements, input_order);
        assert(std::ptrdiff_t(feed_pos_m.size()) == num_elements);

        // Get frequencies
        std::vector<float> frequencies(num_frequencies); // [Hz]
        for (int freq = 0; freq < num_frequencies; ++freq) {
            const freq_id_t channel = static_cast<freq_id_t>(frequency_channels.at(freq));
            frequencies.at(freq) = telescope.to_freq_MHz(channel) * 1.0e+6f;
        }
        const std::vector<int> freq_upchan_factor(frequency_channels.size(), 1);
        const std::vector<int> freq_upchan_index(frequency_channels.size(), 0);

        // Wait for buffers
        DEBUG("[{:s}/{:d}] Waiting for buffer...", bb_beam_positions_buffer->buffer_name,
              frame_index);
        float* const bb_beam_positions_frame = static_cast<float*>(static_cast<void*>(
            bb_beam_positions_buffer->wait_for_full_frame(unique_name, frame_id)));
        if (!bb_beam_positions_frame)
            return;

        DEBUG("[{:s}/{:d}] Waiting for buffer...", A_buffer->buffer_name, frame_index);
        std::complex<std::int8_t>* const A_frame = static_cast<std::complex<std::int8_t>*>(
            static_cast<void*>(A_buffer->wait_for_empty_frame(unique_name, frame_id)));
        if (!A_frame)
            return;

        DEBUG("[{:s}/{:d}] Waiting for buffer...", s_buffer->buffer_name, frame_index);
        std::int32_t* const s_frame = static_cast<std::int32_t*>(
            static_cast<void*>(s_buffer->wait_for_empty_frame(unique_name, frame_id)));
        if (!s_frame)
            return;

        // Check buffer sizes
        assert(std::ptrdiff_t(bb_beam_positions_buffer->frame_size)
               == bb_beam_positions_frame_size);
        assert(std::ptrdiff_t(A_buffer->frame_size) == A_frame_size);
        assert(std::ptrdiff_t(s_buffer->frame_size) == s_frame_size);


        // Get timing info from beam positions buffer.
        const auto& bb_beam_positions_meta = get_chord_metadata(bb_beam_positions_buffer->get_metadata(frame_id));
        uint64_t seq_num = bb_beam_positions_meta->get_fpga_seq_num();
        uint64_t time_downsampling = bb_beam_positions_meta->get_time_downsampling_fpga();

        // Set metadata
        A_buffer->allocate_new_metadata_object(frame_id);
        const auto& A_meta = get_chord_metadata(A_buffer->get_metadata(frame_id));
        A_meta->set_from_frame_desc(A_buffer->get_ndarray_frame_desc());
        A_meta->set_fpga_seq_num(seq_num);
        A_meta->set_time_downsampling_fpga(time_downsampling);
        A_meta->set_coarse_freq(frequency_channels);
        A_meta->set_freq_upchan_factor(freq_upchan_factor);
        A_meta->set_freq_upchan_index(freq_upchan_index);

        s_buffer->allocate_new_metadata_object(frame_id);
        const auto& s_meta = get_chord_metadata(s_buffer->get_metadata(frame_id));
        s_meta->set_from_frame_desc(s_buffer->get_ndarray_frame_desc());
        s_meta->set_fpga_seq_num(seq_num);
        s_meta->set_time_downsampling_fpga(time_downsampling);
        s_meta->set_coarse_freq(frequency_channels);
        s_meta->set_freq_upchan_factor(freq_upchan_factor);
        s_meta->set_freq_upchan_index(freq_upchan_index);

        // Set A
        {
            const float c0 = 299792458.0f; // speed of light in vacuum [m/s]
            const std::ptrdiff_t str_dish = 1;
            const std::ptrdiff_t str_beam = str_dish * num_dishes;
            const std::ptrdiff_t str_polr = str_beam * bb_num_beams;
            const std::ptrdiff_t str_freq = str_polr * num_polarizations;
            for (int freq = 0; freq < num_frequencies; ++freq) {
                for (int polr = 0; polr < num_polarizations; ++polr) {
                    for (int beam = 0; beam < bb_num_beams; ++beam) {
                        for (int dish = 0; dish < num_dishes; ++dish) {
                            const std::ptrdiff_t idx = str_dish * dish + str_beam * beam
                                                       + str_polr * polr + str_freq * freq;
                            // We choose A independent of polarization
                            using std::clamp, std::lrint, std::polar, std::sqrt;
                            const auto pow2 = [](auto x) { return x * x; };
                            const int element = dish + polr * num_dishes;
                            // Dish positions are cartesian components in GRID frame in meters.
                            const float dish_x = feed_pos_m.at(element)[0];
                            const float dish_y = feed_pos_m.at(element)[1];
                            const float dish_z = feed_pos_m.at(element)[2];
                            // Buffered beam positions are nx & ny cartesian components in GRID frame.
                            // |n| = 1.0
                            const float n_x = bb_beam_positions_frame[2 * beam + 0];
                            const float n_y = bb_beam_positions_frame[2 * beam + 1];
                            const float n_z = sqrt(1 - (pow2(n_x) + pow2(n_y)));
                            const float deltat = n_x * dish_x / c0
                                                 + n_y * dish_y / c0
                                                 + n_z * dish_z / c0;
                            const float f = frequencies.at(freq);
                            const float phi = 2 * float(M_PI) * f * deltat;
                            const std::complex<float> A = polar(127.5f, phi);
                            const std::complex<std::int8_t> iA(
                                clamp(int(lrint(real(A))), -127, +127),
                                clamp(int(lrint(imag(A))), -127, +127));
                            assert(idx >= 0
                                   && idx < std::ptrdiff_t(A_buffer->frame_size / sizeof *A_frame));
                            A_frame[idx] = iA;
                        }
                    }
                }
            }
        }

        // Set s
        {
            const std::ptrdiff_t str_beam = 1;
            const std::ptrdiff_t str_polr = str_beam * bb_num_beams;
            const std::ptrdiff_t str_freq = str_polr * num_polarizations;
            for (int freq = 0; freq < num_frequencies; ++freq) {
                for (int polr = 0; polr < num_polarizations; ++polr) {
                    for (int beam = 0; beam < bb_num_beams; ++beam) {
                        const std::ptrdiff_t idx =
                            str_beam * beam + str_polr * polr + str_freq * freq;
                        assert(idx >= 0
                               && idx < std::ptrdiff_t(s_buffer->frame_size / sizeof *s_frame));
                        s_frame[idx] = bb_scale;
                    }
                }
            }
        }

        // Mark buffers as full
        DEBUG("[{:s}/{:d}] Marking buffer as empty...", bb_beam_positions_buffer->buffer_name,
              frame_index);
        bb_beam_positions_buffer->mark_frame_empty(unique_name, frame_id);

        DEBUG("[{:s}/{:d}] Marking buffer as full...", A_buffer->buffer_name, frame_index);
        A_buffer->mark_frame_full(unique_name, frame_id);

        DEBUG("[{:s}/{:d}] Marking buffer as full...", s_buffer->buffer_name, frame_index);
        s_buffer->mark_frame_full(unique_name, frame_id);

        // Wait for shutdown (don't trigger a shutdown)
        while (!stop_thread)
            sleep(1);
    }
};

REGISTER_KOTEKAN_STAGE(calcBBPhase);
