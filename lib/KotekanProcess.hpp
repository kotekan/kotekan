#ifndef KOTEKANPROCESS_H
#define KOTEKANPROCESS_H

#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <vector>
#include "Config.hpp"
#include "bufferContainer.hpp"
#ifdef MAC_OSX
    #include "osxBindCPU.hpp"
    #include <immintrin.h>
#endif

class KotekanProcess {
public:
    KotekanProcess(Config &config, const string& unique_name,
                    bufferContainer &buffer_container,
                    std::function<void(const KotekanProcess&)> main_thread_ref);
    virtual ~KotekanProcess();
    virtual void start();
    virtual void main_thread();
    // This should only happen if config.update_needed(fpga_seq) is true.
    virtual void apply_config(uint64_t fpga_seq) = 0;
    void join();
    void stop();
protected:
    std::atomic_bool stop_thread;
    Config &config;

    std::string unique_name;

    // Set the cores the main thread is allowed to run on to the
    // cores given in cpu_affinity_
    // Also applies the list to the main thread if it is running.
    void set_cpu_affinity(const std::vector<int> &cpu_affinity_);

    // Applies the cpu_list to the thread affinity of the main thread.
    void apply_cpu_affinity();

    // Helper function
    struct Buffer * get_buffer(const std::string &name);
private:
    std::thread this_thread;
    std::function<void(const KotekanProcess&)> main_thread_fn;

    // List of CPU cores that the main thread is allowed to run on.
    // CPU core numbers are zero based.
    std::vector<int> cpu_affinity;

    // Lock for changing or using the cpu_affinity variable.
    std::mutex cpu_affinity_lock;

    // Should we allow direct access?
    // Currently we use the get_buffer helper function for getting a buffer.
    bufferContainer &buffer_container;

};

#endif /* KOTEKANPROCESS_H */