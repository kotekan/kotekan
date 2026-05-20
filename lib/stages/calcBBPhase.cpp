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

#include "CHORDTelescope.hpp"   // for CHORDTelescope, dishInfo, dishGrid
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

    const std::vector<int> frequency_channels =
        config.get<std::vector<int>>(unique_name, "frequency_channels");

    // Baseband beamformer setup
    const int bb_num_beams_x = config.get<int>(unique_name, "bb_num_beams_x");
    const int bb_num_beams_y = config.get<int>(unique_name, "bb_num_beams_y");
    const int bb_num_beams = bb_num_beams_x * bb_num_beams_y;
    const float bb_beam_separation_x = config.get<float>(unique_name, "bb_beam_separation_x");
    const float bb_beam_separation_y = config.get<float>(unique_name, "bb_beam_separation_y");
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
        bb_beam_positions_buffer->register_producer(unique_name);
        A_buffer->register_producer(unique_name);
        s_buffer->register_producer(unique_name);
    }

    virtual ~calcBBPhase() {}

    void main_thread() override {
        // Only calculate a single frame
        const int frame_index = 0;
        const int frame_id = frame_index;

        if (stop_thread)
            return;

        // Telescope
        const auto& chord_telescope = Telescope::instance().cast<CHORDTelescope>();

        // Calculate dish positions
        const float dish_separation_x = chord_telescope.get_feed_sep_x_m();
        const float dish_separation_y = chord_telescope.get_feed_sep_y_m();
        const auto& dish_grid = chord_telescope.get_dish_grid();
        assert(std::ptrdiff_t(chord_telescope.get_num_dishes()) == num_dishes);
        std::vector<float> dish_loc_x(num_dishes, -1), dish_loc_y(num_dishes, -1),
            dish_loc_z(num_dishes, -1);
        assert(std::ptrdiff_t(dish_grid.get_dish_indices().size()) >= num_dishes);
        for (const auto dish_index : dish_grid.get_dish_indices()) {
            if (dish_index >= 0) {
                const dishInfo& dish_info = chord_telescope.get_dish_at_idx(dish_index);
                assert(dish_loc_x.at(dish_info.idx) == -1);
                dish_loc_x.at(dish_info.idx) =
                    dish_info.grid_x_idx * dish_separation_x + dish_info.feed_pos_disp_m.at(0);
                dish_loc_y.at(dish_info.idx) =
                    dish_info.grid_y_idx * dish_separation_y + dish_info.feed_pos_disp_m.at(1);
                dish_loc_z.at(dish_info.idx) = dish_info.feed_pos_disp_m.at(2);
            }
        }
        assert(std::ptrdiff_t(dish_loc_x.size()) == num_dishes);

        // Get frequencies
        std::vector<float> frequencies(num_frequencies); // [Hz]
        for (int freq = 0; freq < num_frequencies; ++freq) {
            const int channel = frequency_channels.at(freq);
            frequencies.at(freq) = chord_telescope.to_freq_MHz(channel) * 1.0e+6f;
        }
        const std::vector<int> freq_upchan_factor(frequency_channels.size(), 1);
        const std::vector<int> freq_upchan_index(frequency_channels.size(), 0);

        // Wait for buffers
        DEBUG("[{:s}/{:d}] Waiting for buffer...", bb_beam_positions_buffer->buffer_name,
              frame_index);
        float* const bb_beam_positions_frame = static_cast<float*>(static_cast<void*>(
            bb_beam_positions_buffer->wait_for_empty_frame(unique_name, frame_id)));
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

        // Set metadata
        bb_beam_positions_buffer->allocate_ndarray_frame_desc<float, 2>(
            "bb_beam_positions", {bb_num_beams, 2}, {"B", "X/Y"});
        bb_beam_positions_buffer->allocate_new_metadata_object(frame_id);
        const auto& bb_beam_positions_meta =
            get_chord_metadata(bb_beam_positions_buffer->get_metadata(frame_id));
        bb_beam_positions_meta->set_from_frame_desc(
            bb_beam_positions_buffer->get_ndarray_frame_desc());
        bb_beam_positions_meta->set_fpga_seq_num(frame_index * num_times);
        bb_beam_positions_meta->set_time_downsampling_fpga(1);

        A_buffer->allocate_ndarray_frame_desc<std::int8_t, 5>(
            "A", {num_frequencies, num_polarizations, bb_num_beams, num_dishes, num_components},
            {"F", "P", "B", "D", "C"});
        A_buffer->allocate_new_metadata_object(frame_id);
        const auto& A_meta = get_chord_metadata(A_buffer->get_metadata(frame_id));
        A_meta->set_from_frame_desc(A_buffer->get_ndarray_frame_desc());
        A_meta->set_fpga_seq_num(frame_index * num_times);
        A_meta->set_time_downsampling_fpga(1);
        A_meta->set_coarse_freq(frequency_channels);
        A_meta->set_freq_upchan_factor(freq_upchan_factor);
        A_meta->set_freq_upchan_index(freq_upchan_index);

        s_buffer->allocate_ndarray_frame_desc<std::int32_t, 3>(
            "s", {num_frequencies, num_polarizations, bb_num_beams}, {"F", "P", "B"});
        s_buffer->allocate_new_metadata_object(frame_id);
        const auto& s_meta = get_chord_metadata(s_buffer->get_metadata(frame_id));
        s_meta->set_from_frame_desc(s_buffer->get_ndarray_frame_desc());
        s_meta->set_fpga_seq_num(frame_index * num_times);
        s_meta->set_time_downsampling_fpga(1);
        s_meta->set_coarse_freq(frequency_channels);
        s_meta->set_freq_upchan_factor(freq_upchan_factor);
        s_meta->set_freq_upchan_index(freq_upchan_index);

        // Set bb_beam_positions
        {
            // Find centre
            const float i_x0 = (bb_num_beams_x - 1) / 2.0f;
            const float i_y0 = (bb_num_beams_y - 1) / 2.0f;
            for (int i_y = 0; i_y < bb_num_beams_y; ++i_y) {
                for (int i_x = 0; i_x < bb_num_beams_x; ++i_x) {
                    const int beam = i_x + bb_num_beams_x * i_y;
                    const int idx = 2 * beam;
                    const float theta_x = bb_beam_separation_x * (i_x - i_x0);
                    const float theta_y = bb_beam_separation_y * (i_y - i_y0);
                    assert(idx >= 0
                           && idx < std::ptrdiff_t(bb_beam_positions_buffer->frame_size
                                                   / sizeof *bb_beam_positions_frame));
                    bb_beam_positions_frame[idx + 0] = theta_x;
                    bb_beam_positions_frame[idx + 1] = theta_y;
                }
            }
        }

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
                            const float dish_x = dish_loc_x.at(dish);
                            const float dish_y = dish_loc_y.at(dish);
                            const float dish_z = dish_loc_z.at(dish);
                            const float theta_x = bb_beam_positions_frame[2 * beam + 0];
                            const float theta_y = bb_beam_positions_frame[2 * beam + 1];
                            const float theta_z = sqrt(1 - (pow2(theta_x) + pow2(theta_y)));
                            const float deltat = sin(theta_x) * dish_x / c0
                                                 + sin(theta_y) * dish_y / c0
                                                 + sin(theta_z) * dish_z / c0;
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
        DEBUG("[{:s}/{:d}] Marking buffer as full...", bb_beam_positions_buffer->buffer_name,
              frame_index);
        bb_beam_positions_buffer->mark_frame_full(unique_name, frame_id);

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
