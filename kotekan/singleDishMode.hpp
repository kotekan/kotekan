#ifndef SINGLE_DISH_MODE_HPP
#define SINGLE_DISH_MODE_HPP

#include "kotekanMode.hpp"
#include "bufferContainer.hpp"

class singleDishMode : public kotekanMode {

public:
    singleDishMode(Config &config);
    virtual ~singleDishMode();

    void initalize_processes();

private:

    bufferContainer host_buffers;
};

#endif /* SINGLE_DISH_MODE_HPP */
