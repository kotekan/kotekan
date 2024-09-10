#ifndef F_ENGINE_STAGE_H
#define F_ENGINE_STAGE_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <array>

#ifdef WITH_CUDA
#include <nvToolsExt.h>
#endif

/**
 * @class FEngine
 * @brief A stage that simulates the CHORD F-engine hardware and a simple scene (eg, a
 * single radio source), producing data useful for end-to-end testing of our kernels.
 * (The actual computational core is implemented in Julia, in FEngine.jl)
 *
 * @conf num_components Int (=2) complex components
 * @conf num_polations  Int (=2) polarizations
 * @conf source_amplitude  Float  simulated radio source amplitude
 * @conf source_frequency  Float  simulated radio source frequency (in Hz)
 * @conf source_position_x  Float  simulated radio source position (east-west angle in radians)
 * @conf source_position_y  Float  simulated radio source position (north-south angle in radians)
 * @conf num_dish_locations_M Int  east-west dish grid size
 * @conf num_dish_locations_N Int  north-south dish grid size
 * @conf dish_separation_x  Float east-west dish separation (in meters)
 * @conf dish_separation_y  Float north-south dish separation (in meters)
 * @conf num_dishes         Int   number of dishes
 * @conf dish_locations     vector of Int  A pair of integers for each dish, giving its position in
 * the M,N grid.
 * @conf adc_frequency      Float  how many samples per second does the ADC perform?
 * @conf num_taps           Int    how many taps in the F-engine's polyphase filter bank?
 * @conf num_frequencies    Int    how many frequencies are produced by the F-engine?
 * @conf num_times          Int    how many time samples per chunk?
 * @conf bb_num_dishes_M    Int    Baseband beamformer: input dish grid size
 * @conf bb_num_dishes_N    Int    Baseband beamformer: input dish grid size
 * @conf bb_num_beams_P    Int    Baseband beamformer: output beam grid size
 * @conf bb_num_beams_Q    Int    Baseband beamformer: output beam grid size
 * @conf bb_beam_separation_x Float Baseband beamformer: output beam spacing east-west (radians?)
 * @conf bb_beam_separation_y Float Baseband beamformer: output beam spacing north-south (radians?)
 * @conf upchannelization_factor Int Upchan
 * @conf num_frames              Int how many frames of data to produce
 */
class FEngine : public kotekan::Stage {
    const std::string unique_name;

    const bool skip_julia;

    // Basic constants
    const int num_components;
    const int num_polarizations;

    // Sky
    const float noise_amplitude;
    const std::vector<float> source_channels;
    const std::vector<float> source_amplitudes;
    const float dispersed_source_start_time;
    const float dispersed_source_end_time;
    const float dispersed_source_start_frequency;
    const float dispersed_source_stop_frequency;
    const float dispersed_source_linewidth;
    const float dispersed_source_amplitude;
    const float source_position_ew;
    const float source_position_ns;

    // Dishes
    const int num_dish_locations_ew;
    const int num_dish_locations_ns;
    const int num_dish_locations;
    const float dish_separation_ew;
    const float dish_separation_ns;
    const int num_dishes;
    const std::vector<int> dish_indices;
    std::vector<int> dish_locations; // (ew, ns)
    int* dish_indices_ptr;

    // ADC
    const float adc_frequency;
    const int num_samples_per_frame;
    const int num_taps;
    const int num_frequencies;
    const std::vector<int> frequency_channels;
    const int num_times;

    // Dish reordering
    const std::vector<int> scatter_indices;

    // Baseband beamformer setup
    const int bb_num_beams_ew;
    const int bb_num_beams_ns;
    const float bb_beam_separation_ew;
    const float bb_beam_separation_ns;
    const int bb_num_beams;
    const int bb_scale;

    // Upchannelizer setup
    const int upchannelization_factor;

    enum upchan_factor_t { U1, U2, U4, U8, U16, U32, U64, Usize };
    constexpr int upchan_factor(upchan_factor_t U) {
        return std::array<int, Usize>{1, 2, 4, 8, 16, 32, 64}.at(U);
    }
    const std::array<int, Usize> upchan_max_num_channelss, upchan_min_channels, upchan_max_channels;
    const int upchan_all_max_num_output_channels, upchan_all_min_output_channel,
        upchan_all_max_output_channel;
    const std::array<std::vector<float>, Usize> upchan_gainss;

    // FRB beamformer setup
    const int frb1_num_beams_P;
    const int frb1_num_beams_Q;
    const int frb2_num_beams_ew;
    const int frb2_num_beams_ns;
    const float frb2_bore_z;
    const float frb2_opening_angle_ew;
    const float frb2_opening_angle_ns;
    const int Tds = 40;
    const int frb_num_times;

    // Pipeline
    const int num_frames;
    const int repeat_count;

    // Kotekan
    const std::int64_t dish_positions_frame_size;
    const std::int64_t E_frame_size;
    const std::int64_t scatter_indices_frame_size;
    const std::int64_t bb_beam_positions_frame_size;
    const std::int64_t A_frame_size;
    const std::int64_t s_frame_size;
    const std::int64_t J_frame_size;
    const std::array<std::int64_t, Usize> G_frame_sizes;
    const std::array<std::int64_t, Usize> W1_frame_sizes;
    const std::int64_t W2_frame_size;
    const std::int64_t I1_frame_size;

    Buffer* const dish_positions_buffer;
    Buffer* const E_buffer;
    Buffer* const scatter_indices_buffer;
    Buffer* const bb_beam_positions_buffer;
    Buffer* const A_buffer;
    Buffer* const s_buffer;
    Buffer* const J_buffer;
    std::array<Buffer*, Usize> const G_buffers;
    std::array<Buffer*, Usize> const W1_buffers;
    Buffer* const W2_buffer;
    Buffer* const I1_buffer;

public:
    FEngine(kotekan::Config& config, const std::string& unique_name,
            kotekan::bufferContainer& buffer_conainer);
    virtual ~FEngine();
    void main_thread() override;
};

static void profile_mark([[maybe_unused]] const char* mark_name) {
#ifdef WITH_CUDA
    nvtxMarkA(mark_name);
#endif
}
static void profile_range_push([[maybe_unused]] const char* range_name) {
#ifdef WITH_CUDA
    nvtxRangePushA(range_name);
#endif
}
static void profile_range_pop() {
#ifdef WITH_CUDA
    nvtxRangePop();
#endif
}

#endif
