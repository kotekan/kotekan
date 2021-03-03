/*****************************************
@file
@brief Baseband BasebandWriter stage.
- BasebandWriter : public
*****************************************/
#ifndef BASEBAND_WRITER_HPP
#define BASEBAND_WRITER_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <cstdint>             // for uint64_t
#include <map>                 // for map

class BasebandWriter : public kotekan::Stage {
public:
    BasebandWriter(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    void main_thread() override;

private:
    void write_data(Buffer* in_buf, int frame_id);

    // Parameters saved from the config file
    std::string _root_path;

    /// Input buffer to read from
    struct Buffer* in_buf;

    /// The set of active baseband dump file descriptors, keyed by their event
    /// id to frequency map
    std::unordered_map<uint64_t, std::unordered_map<uint32_t, int>> baseband_events;
};

#endif // BASEBAND_WRITER_HPP
