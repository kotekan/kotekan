#include "TCPhaseUpdate.hpp"

#include "StageFactory.hpp"

REGISTER_KOTEKAN_STAGE(TCPhaseUpdate);

TCPhaseUpdate::TCPhaseUpdate(kotekan::Config& config, const std::string& unique_name,
                             kotekan::bufferContainer& buffer_container) :
    BeamformingPhaseUpdate(config, unique_name, buffer_container) {}

void TCPhaseUpdate::compute_phases(uint8_t* out_frame, const timespec& gps_time,
                                   const std::vector<float>& frequencies_in_frame,
                                   uint32_t beam_offset, uint8_t* gains_frame) {
    // These lines are just to suppress warnings, remove once function uses them
    (void)out_frame;
    (void)gps_time;
    (void)frequencies_in_frame;
    (void)beam_offset;
    // Keep this one, since it isn't used for the TC version
    (void)gains_frame;

    // Code to generate phases goes here.
    // Can access configuration parameters from BeamformingPhaseUpdate
    // e.g. _inst_lat and _inst_long, _beam_coord, _num_beams, etc.
    // Note for this version the `gains_frame` will be set to nullptr
    // and isn't used, since gains are loaded into the GPU separately.
}
