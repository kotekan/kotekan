#ifndef ACQ_UPLINK_H
#define ACQ_UPLINK_H

#include "KotekanProcess.hpp"

class chrxUplink : public KotekanProcess {
public:
    chrxUplink(struct Config &config,
                  struct Buffer &buf,
                  struct Buffer &gate_buf);
    virtual ~chrxUplink();
    void main_thread();
private:
    struct Buffer &vis_buf;
    struct Buffer &gate_buf;
};

#endif