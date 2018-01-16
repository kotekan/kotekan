#ifndef HSA_BEAMFORM_REORDER_H
#define HSA_BEAMFORM_REORDER_H

#include "hsaCommand.hpp"


class hsaBeamformReorder: public hsaCommand
{
public:
    hsaBeamformReorder(const string &kernel_name, const string &kernel_file_name,
                         hsaDeviceInterface &device, Config &config,
			 bufferContainer &host_buffers,
			 const string &unique_name);

    virtual ~hsaBeamformReorder();

    void apply_config(const uint64_t& fpga_seq) override;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:
    int32_t input_frame_len;
    int32_t output_frame_len;
    int32_t map_len;

    int32_t _num_elements;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    vector<int32_t> _reorder_map;
    int * _reorder_map_c;
};

#endif
