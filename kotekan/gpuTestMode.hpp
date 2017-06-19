#ifndef GPU_TEST_MODE_HPP
#define GPU_TEST_MODE_HPP

#include "kotekanMode.hpp"
#include "bufferContainer.hpp"
#ifdef WITH_HSA
#include "hsaBase.h"

class gpuTestMode : public kotekanMode {

public:
    gpuTestMode(Config &config);
    virtual ~gpuTestMode();

    void initalize_processes();

private:
};

#endif /* WITH_HSA */

#endif /* GPU_TEST_MODE_HPP */
