#ifndef RFI_BROADCAST_H
#define RFI_BROADCAST_H

#include "powerStreamUtil.hpp"
#include <sys/socket.h>
#include "Config.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include <atomic>
#include "chimeMetadata.h"

class rfiBroadcast : public KotekanProcess {
public:
    rfiBroadcast(Config& config,
                       const string& unique_name,
                       bufferContainer& buffer_container);
    virtual ~rfiBroadcast();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:

	void tcpConnect();

    struct Buffer *rfi_buf;
    int _num_local_freq;
    int _num_elements;
    int _samples_per_data_set;
    int _sk_step;
    int _buf_depth;
    uint8_t slot_id;
    uint8_t link_id;
    int frames_per_packet;

    stream_id_t stream_ID;
    int64_t fpga_seq_num;

    uint32_t dest_port;
    string dest_server_ip;
    string dest_protocol;

    int socket_fd;
    bool tcp_connected=false;
    bool tcp_connecting=false;
	std::thread connect_thread;
    std::atomic_flag socket_lock;

    int id;
    int slot;
    uint frame_idx=0;

    uint64_t handshake_idx=-1;
    double   handshake_utc=-1;

};

#endif
