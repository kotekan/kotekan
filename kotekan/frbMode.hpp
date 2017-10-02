#ifndef FRB_HPP
#define FRB_HPP

#include "kotekanMode.hpp"
#include "bufferContainer.hpp"

#include "hsaBase.h"

// Make this dynamic
#define NUM_GPUS 4

class frbMode : public kotekanMode {

public:
    frbMode(Config &config);
    virtual ~frbMode();

    void initalize_processes();

private:

    bufferContainer host_buffers[NUM_GPUS];
};

#endif /* FRB_HPP */
