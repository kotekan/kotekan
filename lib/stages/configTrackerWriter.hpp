/**
 * @file
 * @brief Stage that writes tracked configs to disk on change.
 *
 * In its main thread, this stage polls the global ConfigTracker and, whenever the
 * combined tracker hash changes, writes all tracked configuration payloads to disk via
 * ConfigTracker::writeConfigsToDisk().
 *
 * - configTrackerWriter : public kotekan::Stage
 *
 * @conf base_dir  String. Directory to write JSON files into. If it doesn't
 *                 exist, it will be created. Logs the absolute path.
 */
#ifndef CONFIG_TRACKER_WRITER_HPP
#define CONFIG_TRACKER_WRITER_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "bufferContainer.hpp"

#include <string>

class configTrackerWriter : public kotekan::Stage {
public:
    configTrackerWriter(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);
    ~configTrackerWriter() override;

    void main_thread() override;

private:
    // Output directory for config files
    const std::string _base_dir;
    // Absolute, normalized path resolved from _base_dir
    std::string _output_dir_abs;
};

#endif // CONFIG_TRACKER_WRITER_HPP
