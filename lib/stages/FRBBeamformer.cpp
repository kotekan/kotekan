#include <atomic>
#include <functional>

#include "FRBBeamformer.hpp"
#include "kotekanLogging.hpp"
#include "StageFactory.hpp"

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(FRBBeamformer);

FRBBeamformer::FRBBeamformer(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&FRBBeamformer::main_thread, this)) {}

FRBBeamformer::~FRBBeamformer() {}

void FRBBeamformer::main_thread() {
    INFO("FRB beamformer process, reached main_thread!");
    while (!stop_thread) {
        INFO("In thread!");
    }
}
