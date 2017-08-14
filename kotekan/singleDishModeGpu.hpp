#ifndef SINGLE_DISH_MODE_GPU_HPP
#define SINGLE_DISH_MODE_GPU_HPP

#include "kotekanMode.hpp"
#include "bufferContainer.hpp"

#include "hsaBase.h"

class singleDishModeGpu : public kotekanMode {

public:
    singleDishModeGpu(Config &config);
    virtual ~singleDishModeGpu();

    void initalize_processes();

private:

    bufferContainer host_buffers;
};

#endif /* RFI_MODE_HPP */
