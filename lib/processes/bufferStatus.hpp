#ifndef BUFFER_STATUS_H
#define BUFFER_STATUS_H

#include "KotekanProcess.hpp"
#include "bufferContainer.hpp"

class bufferStatus : public KotekanProcess {
public:
    bufferStatus(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);
    virtual ~bufferStatus();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);
private:
    map<string, Buffer*> buffers;
    int time_delay;
};

#endif
