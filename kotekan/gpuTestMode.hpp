#ifndef GPU_TEST_MODE_HPP
#define GPU_TEST_MODE_HPP

#include "kotekanMode.hpp"
#include "bufferContainer.hpp"
#ifdef WITH_HSA
    #include "hsaBase.h"
#endif /* WITH_HSA */


class gpuTestMode : public kotekanMode {

public:
    gpuTestMode(Config &config);
    virtual ~gpuTestMode();

    void initalize_processes();

private:
};

#endif /* GPU_TEST_MODE_HPP */
