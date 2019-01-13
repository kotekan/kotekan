#ifndef BEAMFORMING_POST_PROCESS
#define BEAMFORMING_POST_PROCESS

#include "Config.hpp"
#include "buffer.h"
#include "chimeMetadata.h"
#include "stage.hpp"

#include <vector>

using std::vector;

class beamformingPostProcess : public kotekan::Stage {
public:
    beamformingPostProcess(kotekan::Config& config, const string& unique_name,
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
    vector<int32_t> _link_map;
    uint32_t _num_local_freq;
    uint32_t _num_gpus;
};


#endif