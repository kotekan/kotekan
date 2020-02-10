#ifndef HSA_CORRELATOR_KERNEL_H
#define HSA_CORRELATOR_KERNEL_H

#include "Config.hpp"             // for Config
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface
#include "hsaSubframeCommand.hpp" // for hsaSubframeCommand

#include <stdint.h> // for int32_t, uint32_t
#include <string>   // for string

#pragma pack(4)
struct corr_kernel_config_t {
    uint32_t n_elem;
    uint32_t n_intg;
    uint32_t n_iter;
    uint32_t n_blk;
};
#pragma pack(0)

class hsaCorrelatorKernel : public hsaSubframeCommand {
public:
    hsaCorrelatorKernel(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    virtual ~hsaCorrelatorKernel();

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    int32_t input_frame_len;
    int32_t presum_len;
    int32_t corr_frame_len;
    int32_t block_map_len;

    uint32_t* host_block_map;
    corr_kernel_config_t* host_kernel_args;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _num_blocks;
    int32_t _n_intg;
};

#endif
