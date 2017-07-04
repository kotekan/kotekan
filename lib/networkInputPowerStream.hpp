#ifndef NETWORK_INPUT_POWER_STREAM_H
#define NETWORK_INPUT_POWER_STREAM_H

#include "powerStreamUtil.hpp"
#include <sys/socket.h>
#include "Config.hpp"
#include "buffers.h"
#include "KotekanProcess.hpp"
#include <atomic>

class networkInputPowerStream : public KotekanProcess {
public:
    networkInputPowerStream(Config &config, struct Buffer &buf);
    virtual ~networkInputPowerStream();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);
    void receive_packet(void *buffer, int length);

private:
	void tcpConnect();

    struct Buffer &buf;

    uint32_t port;
    string server_ip;
    string protocol;

    int socket_fd;
    bool tcp_connected=false;
    bool tcp_connecting=false;
	std::thread connect_thread;
    std::atomic_flag socket_lock;

    int freqs;
    int times;
    int elems;

    int id;

    uint frame_idx=0;

    uint64_t handshake_idx=-1;
    double   handshake_utc=-1;

	IntensityHeader header;
};

#endif