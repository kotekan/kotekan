#ifndef FULL_PACKET_DUMP_HPP
#define FULL_PACKET_DUMP_HPP

#include "KotekanProcess.hpp"

class fullPacketDump : public KotekanProcess {
public:
    fullPacketDump(Config &config, struct Buffer &buf, int link_id);
    virtual ~fullPacketDump();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);

    uint8_t * packet_grab_callback(int num_packets, int &len);

private:
    struct Buffer &buf;
    int link_id;

    bool got_packets = false;

    int _packet_size;
    uint8_t * _packet_frame;
};

#endif