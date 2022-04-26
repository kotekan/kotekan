#include "FRBBeamformer.hpp"
#include "StageFactory.hpp"

#include "errors.h"

#include <functional>

using kotekan::Config;
using kotekan::Stage;
using kotekan::bufferContainer;

REGISTER_KOTEKAN_STAGE(FRBBeamformer);

FRBBeamformer::FRBBeamformer(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&FRBBeamformer::main_thread, this)) {}

FRBBeamformer::~FRBBeamformer() {}

void FRBBeamformer::main_thread() {
    INFO("FRB beamformer Process, reached main_thread!");
    while (!stop_thread) {
        INFO("In thread!");
    }
}
