#ifndef NETWORK_POWER_STREAM_H
#define NETWORK_POWER_STREAM_H

#include <sys/socket.h>
#include "Config.hpp"
#include "buffers.h"
#include "KotekanProcess.hpp"

class networkPowerStream : public KotekanProcess {
public:
    networkPowerStream(Config &config, struct Buffer &buf);
    virtual ~networkPowerStream();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:
	void tcpConnect();

    struct Buffer &buf;

    uint32_t dest_port;
    string dest_server_ip;
    string dest_protocol;

    int socket_fd;
    bool tcp_connected=false;
    bool tcp_connecting=false;
	std::thread connect_thread;

    int freqs;
    int times;
};

#endif