#ifndef KOTEKAN_MODE_HPP
#define KOTEKAN_MODE_HPP

#include "Config.hpp"
#include "KotekanProcess.hpp"

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

    void add_process(KotekanProcess * process);
    void add_buffer(struct Buffer * buffer);
    void add_info_object_pool(struct InfoObjectPool * info_pool);

private:
    vector<KotekanProcess *> processes;
    vector<struct Buffer *> buffers;
    vector<struct InfoObjectPool *> info_pools;
};


#endif /* CHIME_HPP */