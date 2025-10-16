#include "configTrackerWriter.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "configTracker.hpp"  // for ConfigTracker
#include "kotekanLogging.hpp" // for logging macros

#include <chrono>
#include <filesystem>
#include <thread>

using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(configTrackerWriter);

configTrackerWriter::configTrackerWriter(Config& config, const std::string& unique_name,
                                         kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&configTrackerWriter::main_thread, this)),
    _base_dir(config.get<std::string>(unique_name, "base_dir")) {

    namespace fs = std::filesystem;

    // Resolve absolute path and ensure directory exists
    try {
        fs::path base_path(_base_dir);
        fs::path abs_path = base_path.is_absolute() ? base_path : fs::absolute(base_path);

        std::error_code ec;
        if (!fs::exists(abs_path, ec)) {
            if (!fs::create_directories(abs_path, ec)) {
                const std::string msg = ec ? ec.message() : std::string("unknown error");
                FATAL_ERROR("configTrackerWriter: Failed to create directory '{}': {}",
                            abs_path.string(), msg);
            }
        } else if (!fs::is_directory(abs_path, ec)) {
            FATAL_ERROR("configTrackerWriter: Path is not a directory: '{}'", abs_path.string());
        }

        _output_dir_abs = abs_path.lexically_normal().string();
        INFO("configTrackerWriter: writing tracked configs to {}", _output_dir_abs);

    } catch (const std::exception& e) {
        FATAL_ERROR("configTrackerWriter: exception setting up output directory '{}': {}",
                    _base_dir, e.what());
    }
}

configTrackerWriter::~configTrackerWriter() {}

void configTrackerWriter::main_thread() {
    using namespace std::chrono_literals;

    std::string last_hash = "";

    while (!stop_thread) {
        const std::string cur_hash = kotekan::ConfigTracker::instance().getTrackerHash();

        if (cur_hash != last_hash) {
            // Update before writing to avoid duplicate writes if this loop is slow
            last_hash = cur_hash;

            size_t n_written =
                kotekan::ConfigTracker::instance().writeConfigsToDisk(_output_dir_abs);
            INFO("configTrackerWriter: tracker hash changed; wrote {} config file(s) to {}",
                 n_written, _output_dir_abs);
        }

        // Lightweight polling to avoid busy-waiting
        std::this_thread::sleep_for(250ms);
    }

    INFO("configTrackerWriter: exiting main thread");
}
