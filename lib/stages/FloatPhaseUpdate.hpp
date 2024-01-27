#ifndef FLOAT_PHASE_UPDATE_HPP
#define FLOAT_PHASE_UPDATE_HPP

#include "BeamformingPhaseUpdate.hpp"

class FloatPhaseUpdate : public BeamformingPhaseUpdate {
public:
    FloatPhaseUpdate(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    ~FloatPhaseUpdate() = default;

protected:
    virtual void compute_phases(uint8_t* out_frame, const timespec& gps_time,
                                const std::vector<float>& frequencies_in_frame,
                                uint8_t* gains_frame) override;
};


#endif // FLOAT_PHASE_UPDATE_HPP
