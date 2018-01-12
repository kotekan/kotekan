#ifndef HSA_CORRELATOR_KERNEL_H
#define HSA_CORRELATOR_KERNEL_H

#include "hsaCommand.hpp"

#pragma pack(4)
struct corr_kernel_config_t {
    uint32_t n_elem;
    uint32_t n_intg;
    uint32_t n_iter;
    uint32_t n_blk;
};
#pragma pack(0)

class hsaCorrelatorKernel: public hsaCommand
{
public:

    hsaCorrelatorKernel(const string &kernel_name, const string &kernel_file_name,
                        hsaDeviceInterface &device, Config &config,
                        bufferContainer &host_buffers, const string &unique_name);

    virtual ~hsaCorrelatorKernel();

    void apply_config(const uint64_t& fpga_seq) override;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:
    int32_t input_frame_len;
    int32_t presum_len;
    int32_t corr_frame_len;
    int32_t block_map_len;

    uint32_t * host_block_map;
    corr_kernel_config_t * host_kernel_args;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _num_blocks;
    int32_t _n_intg;
};

#endif