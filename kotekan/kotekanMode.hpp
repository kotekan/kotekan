#ifndef KOTEKAN_MODE_HPP
#define KOTEKAN_MODE_HPP

#include "Config.hpp"
#include "KotekanProcess.hpp"
#include "metadata.h"

#include <map>

using std::map;
using std::string;

class kotekanMode {
public:
    kotekanMode(Config &config);
    virtual ~kotekanMode();

    // Allocate memory for the processes and get the configuration.
    virtual void initalize_processes();

    // Call start on all the processes.
    void start_processes();

    // Stop all the processes.
    void stop_processes();

    // Join blocks until all processes have stopped.
    void join();

protected:
    Config &config;
    bufferContainer buffer_container;

private:
    map<string, KotekanProcess *> processes;
    map<string, struct metadataPool *> metadata_pools;
    map<string, struct Buffer *> buffers;
};


#endif /* CHIME_HPP */