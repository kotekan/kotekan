#ifndef SIM_VDIF_DATA_H
#define SIM_VDIF_DATA_H

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "vdif_functions.h"

class simVdifData : public kotekan::KotekanProcess {
public:
    simVdifData(kotekan::Config& config, const string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~simVdifData();
    void main_thread() override;

private:
    struct Buffer* buf;
    double start_time, stop_time;
};

#endif
