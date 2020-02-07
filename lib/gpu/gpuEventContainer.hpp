#ifndef GPU_EVENT_CONTAINER_H
#define GPU_EVENT_CONTAINER_H

#include <condition_variable> // for condition_variable
#include <mutex>              // for mutex

class gpuEventContainer {

public:
    gpuEventContainer();
    gpuEventContainer(const gpuEventContainer& obj);
    virtual ~gpuEventContainer();

    // Clear the variables to default state
    void reset();

    // Set the signal and notify anyone waiting on them.
    void set_signal(void* sig);
    void* get_signal(void);

    // Wait for the signal to become ready to sleep on then
    // wait for the hsa signal itself to reach zero
    // If `stopping` is set, then sleep on signals, but exit when
    // there are no signals to wait on.  (to clear the packet queue).
    int wait_for_signal();

    // Wait for the signal object to be in its default state
    // i.e. no signal set to wait on, and signal_set == false;
    void wait_for_free_slot();

    // Causes set_signal to exit and return -1 if there are no signals
    // to wait for, otherwise we wait for the signal since we don't
    // want to exit while there are packets in the GPU queues.
    void stop();

    virtual void set(void* sig) = 0;
    virtual void* get() = 0;
    virtual void unset() = 0;
    virtual void wait() = 0;

private:
    std::condition_variable cond_var;
    std::mutex mux;

    bool signal_set;
    bool stopping;
};

#endif // GPU_EVENT_CONTAINER_H
