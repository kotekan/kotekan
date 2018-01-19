#ifndef VDIF_RFI_H
#define VDIF_RFI_H

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "vdif_functions.h"

class vdifRFI : public KotekanProcess {
public:
    vdifRFI(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_containter);
    ~vdifRFI();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *buf_in;
    struct Buffer *buf_out;

    uint32_t num_disks;
    uint32_t num_elements;
    uint32_t num_frequencies;
    uint32_t num_timesteps;
    bool COMBINED;
    int SK_STEP;
};

#endif
