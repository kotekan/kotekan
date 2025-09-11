#include "configTrackerWriter.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "configTracker.hpp"   // for ConfigTracker
#include "kotekanLogging.hpp"  // for logging macros

#include <chrono>
#include <thread>

using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(configTrackerWriter);

configTrackerWriter::configTrackerWriter(Config& config, const std::string& unique_name,
                                         kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&configTrackerWriter::main_thread, this)),
    _base_dir(config.get<std::string>(unique_name, "base_dir")) {}

configTrackerWriter::~configTrackerWriter() {}

void configTrackerWriter::main_thread() {
    using namespace std::chrono_literals;

    std::string last_hash;

    while (!stop_thread) {
        const std::string cur_hash = kotekan::ConfigTracker::instance().getTrackerHash();

        if (cur_hash != last_hash) {
            // Update before writing to avoid duplicate writes if this loop is slow
            last_hash = cur_hash;

            size_t n_written = kotekan::ConfigTracker::instance().writeConfigsToDisk(_base_dir);
            INFO("configTrackerWriter: tracker hash changed; wrote {} config file(s) to {}",
                 n_written, _base_dir);
        }

        // Lightweight polling to avoid busy-waiting
        std::this_thread::sleep_for(250ms);
    }

    INFO("configTrackerWriter: exiting main thread");
}

