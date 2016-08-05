#ifndef VDIF_STREAM
#define VDIF_STREAM

#include "config.h"
#include "buffers.h"
#include "KotekanProcess.hpp"

class vdifStream : public KotekanProcess {
public:
    vdifStream(struct Config &config, struct Buffer &buf);
    virtual ~vdifStream();
    void main_thread();
private:
    struct Buffer &buf;
};

#endif