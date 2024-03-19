#ifndef F_ENGINE_STAGE_H
#define F_ENGINE_STAGE_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

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
    const float source_amplitude;
    const float source_frequency;
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
    std::vector<int> dish_locations;
    int* dish_indices_ptr;

    // ADC
    const float adc_frequency;
    const int num_taps;
    const int num_frequencies;
    const int num_times;

    // Baseband beamformer setup
    const int bb_num_dishes_M;
    const int bb_num_dishes_N;
    const int bb_num_beams_P;
    const int bb_num_beams_Q;
    const float bb_beam_separation_x;
    const float bb_beam_separation_y;
    const int bb_num_beams;

    // Upchannelizer setup
    const int upchannelization_factor;

    // FRB beamformer setup
    const int frb_num_beams_P;
    const int frb_num_beams_Q;
    const int Tds = 40;
    const int frb_num_times;

    // Pipeline
    const int num_frames;

    // Kotekan
    const std::int64_t E_frame_size;
    const std::int64_t A_frame_size;
    const std::int64_t s_frame_size;
    const std::int64_t J_frame_size;
    const std::int64_t G_frame_size;
    const std::int64_t W_frame_size;
    const std::int64_t I_frame_size;

    Buffer* const E_buffer;
    Buffer* const A_buffer;
    Buffer* const s_buffer;
    Buffer* const J_buffer;
    Buffer* const G_buffer;
    Buffer* const W_buffer;
    Buffer* const I_buffer;

public:
    FEngine(kotekan::Config& config, const std::string& unique_name,
            kotekan::bufferContainer& buffer_conainer);
    virtual ~FEngine();
    void main_thread() override;
};

#endif
