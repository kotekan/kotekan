#ifndef KOTEKANPROCESS_H
#define KOTEKANPROCESS_H

#include <thread>
#include <atomic>
#include <functional>
#include "config.h"

class KotekanProcess {
public:
    KotekanProcess(struct Config &config,
                    std::function<void(const KotekanProcess&)> main_thread_ref);
    virtual ~KotekanProcess();
    virtual void start() final;
    virtual void main_thread();
protected:
    std::atomic_bool stop_thread;
    struct Config &config;
private:
    std::thread this_thread;
    std::function<void(const KotekanProcess&)> main_thread_fn;
};

#endif /* KOTEKANPROCESS_H */

