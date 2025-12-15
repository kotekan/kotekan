#ifndef F_ENGINE_STAGE_H
#define F_ENGINE_STAGE_H

#include "CHORDTelescope.hpp"
#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
// Block IWYU from suggesting NVTX3’s private impl header.
// (Use both forms to be safe.)
// IWYU pragma: no_include "nvtx3/nvtxDetail/nvtxImplCore.h"
// IWYU pragma: no_include <nvtx3/nvtxDetail/nvtxImplCore.h>

#include <array>   // for array
#include <cstdint> // for int64_t
#ifdef WITH_CUDA
#include <nvtx3/nvToolsExt.h> // IWYU pragma: keep
#endif
#include <string> // for string, basic_string
#include <vector> // for vector

/**
 * @class FEngine
 * @brief A stage that simulates the CHORD F-engine hardware and a simple scene (eg, a
 * single radio source), producing data useful for end-to-end testing of our kernels.
 * (The actual computational core is implemented in Julia, in FEngine.jl)
 *
 * @par Buffers
 * @buffer E_buffer_chord / E_buffers_chime Input voltage buffers (CHORD single / CHIME per freq).
 *     @buffer_format Depends on mode (complex voltages)
 *     @buffer_metadata chordMetadata
 * @buffer bf_mask_buffer Input beamforming mask buffer.
 *     @buffer_format int8 mask
 * @buffer pl_mask_buffer Input RFI/packet-loss mask buffer.
 *     @buffer_format bool mask
 * @buffer scatter_indices_buffer Scatter index buffer.
 * @buffer bb_beam_positions_buffer Beam position buffer.
 * @buffer A_buffer/s_buffer/G/W1/W2 Buffer set used by kernels (see code for shapes).
 *
 * @conf skip_julia             Bool. Default false. If true, skip Julia setup.
 * @conf receive_chime          Bool. Default false. Use CHIME buffer layout (one per freq) instead
 *                              of CHORD layout.
 * @conf repeat_count           Int. Default 1. Number of times to loop the generated frames.
 *
 * @conf num_frames             Int. Number of frames to produce.
 * @conf num_components         Int. Complex components per sample.
 * @conf num_polarizations      Int. Number of polarizations.
 * @conf num_dishes             Int. Number of dishes.
 * @conf num_samples_per_frame  Int. Samples per input frame.
 * @conf num_taps               Int. Polyphase filter taps.
 * @conf num_frequencies        Int. Number of coarse channels.
 * @conf num_times              Int. Time samples per chunk.
 * @conf frequency_channels     Array<Int>. Channel indices for each coarse channel.
 * @conf scatter_indices        Array<Int>. Optional remapping of dish/pol order (length
 *                              num_dishes*num_polarizations).
 *
 * @conf source_channels        Array<Float>. Channel indices for simulated sources.
 * @conf source_amplitudes      Array<Float>. Amplitudes per source channel.
 * @conf noise_amplitude        Float. Default 0. Additive noise amplitude.
 * @conf dispersed_source_start_time       Float. Default 0.
 * @conf dispersed_source_end_time         Float. Default 1.
 * @conf dispersed_source_start_frequency  Float. Default 2.
 * @conf dispersed_source_stop_frequency   Float. Default 1.
 * @conf dispersed_source_linewidth        Float. Default 1.
 * @conf dispersed_source_amplitude        Float. Default 0.
 * @conf frb_source_start_time             Float. Default 0.
 * @conf frb_source_stop_time              Float. Default 0.
 * @conf frb_source_start_frequency        Float. Default 0.
 * @conf frb_source_stop_frequency         Float. Default 0.
 * @conf frb_source_scale                  Float. Default 0. Scaling applied to FRB pulse.
 * @conf frb_source_time_envelope_centre   Float. Default 0.
 * @conf frb_source_time_envelope_width    Float. Default 0.
 * @conf frb_source_frequency_envelope_lo_centre Float. Default 0.
 * @conf frb_source_frequency_envelope_lo_width  Float. Default 0.
 * @conf frb_source_frequency_envelope_hi_centre Float. Default 0.
 * @conf frb_source_frequency_envelope_hi_width  Float. Default 0.
 * @conf frb_source_amplitude              Float. Default 0.
 * @conf source_position_ew                Float. Source position east-west angle (radians).
 * @conf source_position_ns                Float. Source position north-south angle (radians).
 *
 * @conf adc_frequency         Float. ADC sampling rate (Hz).
 * @conf bb_num_beams_ew       Int. Baseband beams in east-west.
 * @conf bb_num_beams_ns       Int. Baseband beams in north-south.
 * @conf bb_beam_separation_ew Float. Baseband beam spacing east-west.
 * @conf bb_beam_separation_ns Float. Baseband beam spacing north-south.
 * @conf bb_scale              Int. Baseband amplitude scaling.
 *
 * @conf upchannelization_factor          Int. Upchannelisation factor (power of 2).
 * @conf upchan_U1_max_num_channels       Int. Max local channels for U1.
 * @conf upchan_U2_max_num_channels       Int. Max local channels for U2.
 * @conf upchan_U4_max_num_channels       Int. Max local channels for U4.
 * @conf upchan_U8_max_num_channels       Int. Max local channels for U8.
 * @conf upchan_U16_max_num_channels      Int. Max local channels for U16.
 * @conf upchan_U32_max_num_channels      Int. Max local channels for U32.
 * @conf upchan_U64_max_num_channels      Int. Max local channels for U64.
 * @conf upchan_U1_min_channel            Int. First channel for U1.
 * @conf upchan_U2_min_channel            Int. First channel for U2.
 * @conf upchan_U4_min_channel            Int. First channel for U4.
 * @conf upchan_U8_min_channel            Int. First channel for U8.
 * @conf upchan_U16_min_channel           Int. First channel for U16.
 * @conf upchan_U32_min_channel           Int. First channel for U32.
 * @conf upchan_U64_min_channel           Int. First channel for U64.
 * @conf upchan_U1_max_channel            Int. Last channel (exclusive) for U1.
 * @conf upchan_U2_max_channel            Int. Last channel (exclusive) for U2.
 * @conf upchan_U4_max_channel            Int. Last channel (exclusive) for U4.
 * @conf upchan_U8_max_channel            Int. Last channel (exclusive) for U8.
 * @conf upchan_U16_max_channel           Int. Last channel (exclusive) for U16.
 * @conf upchan_U32_max_channel           Int. Last channel (exclusive) for U32.
 * @conf upchan_U64_max_channel           Int. Last channel (exclusive) for U64.
 * @conf upchan_all_max_num_output_channels Int. Total output channels available.
 * @conf upchan_all_min_output_channel      Int. First global output channel.
 * @conf upchan_all_max_output_channel      Int. Last global output channel.
 * @conf upchan_U2_gains                  Array<Float>. Complex gain factors for U2 (interleaved).
 * @conf upchan_U4_gains                  Array<Float>. Complex gain factors for U4 (interleaved).
 * @conf upchan_U8_gains                  Array<Float>. Complex gain factors for U8 (interleaved).
 * @conf upchan_U16_gains                 Array<Float>. Complex gain factors for U16 (interleaved).
 * @conf upchan_U32_gains                 Array<Float>. Complex gain factors for U32 (interleaved).
 * @conf upchan_U64_gains                 Array<Float>. Complex gain factors for U64 (interleaved).
 *
 * @conf frb1_input_scale    Float. Scale for FRB stage 1 input.
 * @conf frb_num_beams_ew    Int. FRB beamformer beams east-west.
 * @conf frb_num_beams_ns    Int. FRB beamformer beams north-south.
 * @conf frb_bore_z          Float. FRB bore-sight z position.
 * @conf frb_opening_angle_ew Float. FRB east-west opening angle.
 * @conf frb_opening_angle_ns Float. FRB north-south opening angle.
 *
 * @par Example
 * @code
 * f_engine:
 *   kotekan_stage: FEngine
 *   receive_chime: true
 *   num_frames: 10
 *   num_components: 2
 *   num_polarizations: 2
 *   num_frequencies: 1024
 *   num_times: 16384
 *   adc_frequency: 800e6
 *   num_taps: 8
 *   # ... other telescope/beam parameters ...
 * @endcode
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
    const float frb_source_start_time;
    const float frb_source_stop_time;
    const float frb_source_start_frequency;
    const float frb_source_stop_frequency;
    const int frb_source_scale;
    const float frb_source_time_envelope_centre;
    const float frb_source_time_envelope_width;
    const float frb_source_frequency_envelope_lo_centre;
    const float frb_source_frequency_envelope_lo_width;
    const float frb_source_frequency_envelope_hi_centre;
    const float frb_source_frequency_envelope_hi_width;
    const float frb_source_amplitude;
    const float source_position_ew;
    const float source_position_ns;

    // Telescope
    const CHORDTelescope& chord_telescope;

    // Dishes
    const int num_dishes;
    const dishGrid& dish_grid;

    // ADC
    const float adc_frequency;
    const int num_samples_per_frame;
    const int num_taps;
    const int num_frequencies;
    const std::vector<int> frequency_channels;
    const int num_times;

    // Dish reordering
    const std::vector<int> scatter_indices;

    // Input buffer layout (CHIME or CHORD)
    const bool receive_chime;

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
    const float frb1_input_scale;
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
    const std::int64_t bf_mask_frame_size;
    const std::int64_t pl_mask_frame_size;
    const std::int64_t E_frame_size;
    const std::int64_t scatter_indices_frame_size;
    const std::int64_t bb_beam_positions_frame_size;
    const std::int64_t A_frame_size;
    const std::int64_t s_frame_size;
    [[maybe_unused]] const std::int64_t J_frame_size;
    const std::array<std::int64_t, Usize> G_frame_sizes;
    const std::array<std::int64_t, Usize> W1_frame_sizes;
    const std::int64_t W2_frame_size;
    [[maybe_unused]] const std::int64_t I1_frame_size;

    // int8 bf_mask[dish][polr]
    Buffer* const bf_mask_buffer; // 0=bad, 1=good
    // bool pl_mask[time / 2 % 64][dish][polr][freq / 4][time / 2 / 64]
    Buffer* const pl_mask_buffer;               // 0=bad, 1=good
    Buffer* const E_buffer_chord;               // CHORD uses a single input buffer
    std::vector<Buffer*> const E_buffers_chime; // CHIME uses one input buffer per frequency
    Buffer* const scatter_indices_buffer;
    Buffer* const bb_beam_positions_buffer;
    Buffer* const A_buffer;
    Buffer* const s_buffer;
    // Buffer* const J_buffer;
    std::array<Buffer*, Usize> const G_buffers;
    std::array<Buffer*, Usize> const W1_buffers;
    Buffer* const W2_buffer;
    // Buffer* const I1_buffer;

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
