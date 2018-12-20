#ifndef SIM_VDIF_DATA_H
#define SIM_VDIF_DATA_H

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "vdif_functions.h"

class simVdifData : public KotekanProcess {
public:
    simVdifData(Config& config, const string& unique_name, bufferContainer& buffer_container);
    ~simVdifData();
    void main_thread() override;

private:
    struct Buffer* buf;
    double start_time, stop_time;
};

#endif