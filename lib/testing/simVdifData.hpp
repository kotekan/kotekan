#ifndef SIM_VDIF_DATA_H
#define SIM_VDIF_DATA_H

#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // IWYU pragma: keep

#include <string> // for string

namespace kotekan {
class Config;
} // namespace kotekan

class simVdifData : public kotekan::Stage {
public:
    simVdifData(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~simVdifData();
    void main_thread() override;

private:
    struct Buffer* buf;
    double start_time, stop_time;
};

#endif
