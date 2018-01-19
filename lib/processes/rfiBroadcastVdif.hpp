#ifndef RFI_BROADCAST_VDIF_H
#define RFI_BROADCAST_VDIF_H

#include "powerStreamUtil.hpp"
#include <sys/socket.h>
#include "Config.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "chimeMetadata.h"

class rfiBroadcastVdif : public KotekanProcess {
public:
    rfiBroadcastVdif(Config& config,
                       const string& unique_name,
                       bufferContainer& buffer_container);
    virtual ~rfiBroadcastVdif();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:

    struct Buffer *rfi_buf;

    int _num_freq;
    int _num_elements;
    int _samples_per_data_set;

    int _sk_step;
    bool COMBINED;

    uint32_t dest_port;
    string dest_server_ip;
    string dest_protocol;

    int socket_fd;
};

#endif
