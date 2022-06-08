#ifndef TC_PHASE_UPDATE_HPP
#define TC_PHASE_UPDATE_HPP

#include "BeamformingPhaseUpdate.hpp"

class TCPhaseUpdate : public BeamformingPhaseUpdate {
public:
    TCPhaseUpdate(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    ~TCPhaseUpdate() = default;

protected:
    virtual void compute_phases(uint8_t* out_frame, const timespec& gps_time,
                                const std::vector<freq_id_t>& frequencies_in_frame,
                                uint8_t* gains_frame) override;
};


#endif // TC_PHASE_UPDATE_HPP
