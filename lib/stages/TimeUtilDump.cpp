#include "TimeUtilDump.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp" // for INFO
#include "timeUtil.hpp"

// Include the classes we will be using
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

// Register the stage with the stage factory.
REGISTER_KOTEKAN_STAGE(TimeUtilDump);

TimeUtilDump::TimeUtilDump(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&TimeUtilDump::main_thread, this)) {

    long gps_sec = config.get_default<long>(unique_name, "gps_sec", 0);
    long gps_nsec = config.get_default<long>(unique_name, "gps_nsec", 0);

    _gps_time = timespec{.tv_sec = gps_sec, .tv_nsec = gps_nsec};
}


TimeUtilDump::~TimeUtilDump() {}

// Framework managed pthread
void TimeUtilDump::main_thread() {
    // Logging function
    INFO("Reached main_thread!");

    // Until the thread is stopped
    while (!stop_thread) {

        // Logging
        INFO("GPS Time: {:d} s {:d} ns", _gps_time.tv_sec, _gps_time.tv_nsec);

        timespec tai = get_TAI_from_GPS(_gps_time);
        INFO("TAI Time: {:d} s {:d} ns", tai.tv_sec, tai.tv_nsec);

        break;
    }
}
