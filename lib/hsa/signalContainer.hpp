#ifndef GPU_SIGNAL_CONTAINER_H
#define GPU_SIGNAL_CONTAINER_H

#include <condition_variable>
#include <mutex>
#include <thread>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_finalize.h"
#include "hsa/hsa_ext_amd.h"

class signalContainer {

public:
    signalContainer();
    signalContainer( const signalContainer &obj);
    ~signalContainer();

    // Clear the variables to default state
    void reset();

    // Set the signal and notify anyone waiting on them.
    void set_signal(hsa_signal_t sig);

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

private:
    std::condition_variable cond_var;
    std::mutex mux;

    hsa_signal_t signal;
    bool signal_set;
    bool stopping;
};

#endif
