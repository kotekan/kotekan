#ifndef SIM_VDIF_DATA_H
#define SIM_VDIF_DATA_H

#include "buffer.h"
#include "stage.hpp"
#include "vdif_functions.h"

class simVdifData : public kotekan::Stage {
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
