#ifndef KOTEKANPROCESS_H
#define KOTEKANPROCESS_H

#include <thread>
#include <atomic>
#include <functional>
#include "Config.hpp"

class KotekanProcess {
public:
    KotekanProcess(Config &config,
                    std::function<void(const KotekanProcess&)> main_thread_ref);
    virtual ~KotekanProcess();
    virtual void start();
    virtual void main_thread();
    // This should only happen if config.update_needed(fpga_seq) is true.
    virtual void apply_config(uint64_t fpga_seq) = 0;
    void join();
protected:
    std::atomic_bool stop_thread;
    Config &config;
private:
    std::thread this_thread;
    std::function<void(const KotekanProcess&)> main_thread_fn;
};

#endif /* KOTEKANPROCESS_H */