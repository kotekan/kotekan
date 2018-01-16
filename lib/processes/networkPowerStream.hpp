#ifndef NETWORK_POWER_STREAM_H
#define NETWORK_POWER_STREAM_H

#include "powerStreamUtil.hpp"
#include <sys/socket.h>
#include "Config.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include <atomic>


class networkPowerStream : public KotekanProcess {
public:
    networkPowerStream(Config& config,
                       const string& unique_name,
                       bufferContainer& buffer_container);
    virtual ~networkPowerStream();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:
	void tcpConnect();

    struct Buffer *buf;

    uint32_t dest_port;
    string dest_server_ip;
    string dest_protocol;

    int socket_fd;
    bool tcp_connected=false;
    bool tcp_connecting=false;
	std::thread connect_thread;
    std::atomic_flag socket_lock;

    int freqs;
    int times;
    int elems;

    float freq0;
    float sample_bw;

    int id;

    uint frame_idx=0;

    uint64_t handshake_idx=-1;
    double   handshake_utc=-1;

	IntensityHeader header;
};

#endif