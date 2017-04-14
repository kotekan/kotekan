#ifndef SINGLE_DISH_VDIF_MODE_HPP
#define SINGLE_DISH_VDIF_MODE_HPP

#include "kotekanMode.hpp"

class singleDishVDIFMode : public kotekanMode {

public:
    singleDishVDIFMode(Config &config);
    virtual ~singleDishVDIFMode();

    void initalize_processes();

private:

};

#endif /* GPU_TEST_MODE_HPP */
