#ifndef SIM_VDIF_DATA_H
#define SIM_VDIF_DATA_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <string> // for string

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
