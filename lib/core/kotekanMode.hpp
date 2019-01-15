#ifndef KOTEKAN_MODE_HPP
#define KOTEKAN_MODE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "metadata.h"
#include "restServer.hpp"

#include <map>

using std::map;
using std::string;

namespace kotekan {

class kotekanMode {
public:
    kotekanMode(Config& config);
    virtual ~kotekanMode();

    // Allocate memory for the processes and get the configuration.
    virtual void initalize_processes();

    // Call start on all the processes.
    void start_processes();

    // Stop all the processes.
    void stop_processes();

    // Join blocks until all processes have stopped.
    void join();

private:
    Config& config;
    bufferContainer buffer_container;

    map<string, Stage*> processes;
    map<string, struct metadataPool*> metadata_pools;
    map<string, struct Buffer*> buffers;
};

} // namespace kotekan

#endif /* CHIME_HPP */
