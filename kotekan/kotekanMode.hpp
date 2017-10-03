#ifndef KOTEKAN_MODE_HPP
#define KOTEKAN_MODE_HPP

#include "Config.hpp"
#include "KotekanProcess.hpp"
#include "metadata.h"

#include <vector>

using std::vector;

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

    void add_process(KotekanProcess * process);
    void add_buffer(struct Buffer * buffer);
    void add_metadata_pool(struct metadataPool * metadata_pool);

private:
    vector<KotekanProcess *> processes;
    vector<struct metadataPool *> metadata_pools;
};


#endif /* CHIME_HPP */