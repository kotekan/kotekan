#ifndef VDIF_RECEIVER_MODE_HPP
#define VDIF_RECEIVER_MODE_HPP

#include "kotekanMode.hpp"
#include "bufferContainer.hpp"

class vdifReceiverMode : public kotekanMode {

public:
    vdifReceiverMode(Config &config);
    virtual ~vdifReceiverMode();

    void initalize_processes();

private:

    bufferContainer host_buffers;
};

#endif /* VDIF_RECEIVER_MODE_HPP */
