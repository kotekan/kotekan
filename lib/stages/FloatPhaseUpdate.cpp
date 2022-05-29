#include "FloatPhaseUpdate.hpp"

#include "StageFactory.hpp"

REGISTER_KOTEKAN_STAGE(FloatPhaseUpdate);

FloatPhaseUpdate::FloatPhaseUpdate(kotekan::Config& config, const std::string& unique_name,
                                   kotekan::bufferContainer& buffer_container) :
    BeamformingPhaseUpdate(config, unique_name, buffer_container) {}

void FloatPhaseUpdate::compute_phases(uint8_t* out_frame, const timespec& gps_time,
                                     const std::vector<freq_id_t>& frequencies_in_frame,
                                     uint8_t* gains_frame) {
    // These lines are just to suppress warnings, remove once function uses them
    (void)out_frame;
    (void)gps_time;
    (void)frequencies_in_frame;
    (void)gains_frame;

    // Code to generate phases goes here.
    // Can access configuration parameters from BeamformingPhaseUpdate
    // e.g. _inst_lat and _inst_long, _beam_coord, _num_beams, etc.
}

