#ifndef BEAMFORMING_POST_PROCESS
#define BEAMFORMING_POST_PROCESS

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for uint32_t, int32_t
#include <string>   // for string
#include <vector>   // for vector


class beamformingPostProcess : public kotekan::Stage {
public:
    beamformingPostProcess(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container);
    virtual ~beamformingPostProcess();
    void main_thread() override;

private:
    void fill_headers(unsigned char* out_buf, struct VDIFHeader* vdif_header, const uint32_t second,
                      const uint32_t fpga_seq_num, const uint32_t num_links, uint32_t* thread_id);

    struct Buffer** in_buf;
    struct Buffer* vdif_buf;

    // Config variables
    uint32_t _num_fpga_links;
    uint32_t _samples_per_data_set;
    uint32_t _num_data_sets;
    std::vector<int32_t> _link_map;
    uint32_t _num_local_freq;
    uint32_t _num_gpus;
};


#endif
