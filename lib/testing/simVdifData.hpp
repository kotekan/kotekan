#ifndef SIM_VDIF_DATA_H
#define SIM_VDIF_DATA_H

#include <string>               // for string

#include "Config.hpp"           // for Config
#include "Stage.hpp"            // for Stage
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer

class simVdifData : public kotekan::Stage {
public:
    simVdifData(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~simVdifData();
    void main_thread() override;

private:
    Buffer* buf;
    double start_time, stop_time;
};

#endif
