#ifndef SIM_VDIF_DATA_H
#define SIM_VDIF_DATA_H

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "vdif_functions.h"

class simVdifData : public KotekanProcess {
public:
    simVdifData(Config& config, const string& unique_name,
                        bufferContainer &buffer_container);
    ~simVdifData();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread() override;
private:
    struct Buffer *buf;
    double start_time, stop_time;

};

#endif