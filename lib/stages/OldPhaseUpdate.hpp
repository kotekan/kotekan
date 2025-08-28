#ifndef OLD_PHASE_UPDATE_HPP
#define OLD_PHASE_UPDATE_HPP

#include "BeamformingPhaseUpdate.hpp"

class OldPhaseUpdate : public BeamformingPhaseUpdate {
public:
    OldPhaseUpdate(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    ~OldPhaseUpdate() = default;

protected:
    virtual void compute_phases(uint8_t* out_frame, const timespec& gps_time,
                                const std::vector<float>& frequencies_in_frame,
                                uint32_t beam_offset, uint8_t* gains_frame) override;

    /// N-S feed separation in m
    float _feed_sep_NS;
    /// E-W feed separation in m
    float _feed_sep_EW;

};


#endif // OLD_PHASE_UPDATE_HPP
