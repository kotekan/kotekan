#ifndef FRBBEAMFORMER_H
#define FRBBEAMFORMER_H

#include <string>

#include "bufferContainer.hpp"
#include "Config.hpp"
#include "Stage.hpp"

class FRBBeamformer : public kotekan::Stage {
public:
    FRBBeamformer(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    virtual ~FRBBeamformer();
    void main_thread() override;

private:
};

#endif /* FRBBEAMFORMER_H */
