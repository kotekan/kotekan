#ifndef INTENSITY_RECEIVER_MODE_HPP
#define INTENSITY_RECEIVER_MODE_HPP

#include "kotekanMode.hpp"
#include "bufferContainer.hpp"

class intensityReceiverMode : public kotekanMode {

public:
    intensityReceiverMode(Config &config);
    virtual ~intensityReceiverMode();

    void initalize_processes();

private:

    bufferContainer host_buffers;
};

#endif /* INTENSITY_RECEIVER_MODE_HPP */
