#ifndef CL_OUTPUT_DATA_ZERO_H
#define CL_OUTPUT_DATA_ZERO_H

#include "clCommand.hpp"

class clOutputDataZero: public clCommand
{
public:
    clOutputDataZero(Config &config, const string &unique_name,
                      bufferContainer &host_buffers, clDeviceInterface &device);
    ~clOutputDataZero();
    cl_event execute(int gpu_frame_id, const uint64_t& fpga_seq, cl_event pre_event) override;

private:
    int32_t presum_len;
    void * presum_zeros;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_local_freq;
    int32_t _num_elements;

};

#endif // CL_OUTPUT_DATA_ZERO_H

