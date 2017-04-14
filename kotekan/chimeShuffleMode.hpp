#ifndef CHIME_SHUFFLE_HPP
#define CHIME_SHUFFLE_HPP

#include "kotekanMode.hpp"
#include "bufferContainer.hpp"

#include "hsaBase.h"

// Make this dynamic
#define NUM_GPUS 4

class chimeShuffleMode : public kotekanMode {

public:
    chimeShuffleMode(Config &config);
    virtual ~chimeShuffleMode();

    void initalize_processes();

private:

    bufferContainer host_buffers[NUM_GPUS];
};

#endif /* CHIME_SHUFFLE_HPP */
