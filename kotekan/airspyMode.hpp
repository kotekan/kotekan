#ifndef AIRSPY_MODE_HPP
#define AIRSPY_MODE_HPP

#include "kotekanMode.hpp"
#include "bufferContainer.hpp"

class airspyMode : public kotekanMode {

public:
    airspyMode(Config &config);
    virtual ~airspyMode();

    void initalize_processes();

private:

    bufferContainer host_buffers;
};

#endif
