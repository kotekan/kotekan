#ifndef NETWORK_OUTPUT_SIM
#define NETWORK_OUTPUT_SIM

#define SIM_CONSTANT 0
#define SIM_FULL_RANGE 1
#define SIM_SINE 2

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "Telescope.hpp"       // for stream_t
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t
#include <string>   // for string

class networkOutputSim : public kotekan::Stage {
public:
    networkOutputSim(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    virtual ~networkOutputSim();
    void main_thread() override;

private:
    struct Buffer* buf;
    int num_links_in_group;
    int link_id;
    int pattern;
    stream_t stream_id;

    // Config variables.
    int32_t _samples_per_data_set;
    int32_t _num_local_freq;
    int32_t _num_elem;
};

#endif
