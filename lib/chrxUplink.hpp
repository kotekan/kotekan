#ifndef ACQ_UPLINK_H
#define ACQ_UPLINK_H

#include <string>

#include "KotekanProcess.hpp"

using string = std::string;

class chrxUplink : public KotekanProcess {
public:
    chrxUplink(struct Config &config,
                  struct Buffer &buf,
                  struct Buffer &gate_buf);
    virtual ~chrxUplink();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);

private:
    struct Buffer &vis_buf;
    struct Buffer &gate_buf;

    // Config variables
    string _collection_server_ip;
    int32_t _collection_server_port;
    bool _enable_gating;

};

#endif