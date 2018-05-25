
#ifndef OUTPUT_BEAMFORM_RESULT_H
#define OUTPUT_BEAMFORM_RESULT_H

#include "clCommand.hpp"

class clOutputBeamformResult: public clCommand
{
public:
    clOutputBeamformResult(Config &config, const string &unique_name,
                      bufferContainer &host_buffers, clDeviceInterface &device);
    ~clOutputBeamformResult();
    virtual void build() override;
    virtual cl_event execute(int param_bufferID, const uint64_t& fpga_seq, cl_event param_PrecedeEvent) override;
protected:

private:
    int32_t _num_local_freq;
    int32_t _num_data_sets;
    int32_t _samples_per_data_set;
    Buffer *beam_out_buf;
};

#endif

