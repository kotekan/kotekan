#ifndef CL_PRESUM_ZERO_H
#define CL_PRESUM_ZERO_H

#include "clCommand.hpp"
#include <sys/mman.h>

class clPresumZero: public clCommand
{
public:
    clPresumZero(Config &config, const string &unique_name,
                      bufferContainer &host_buffers, clDeviceInterface &device);
    ~clPresumZero();
    cl_event execute(int param_bufferID, cl_event param_PrecedeEvent) override;

private:
    int32_t presum_len;
    void * presum_zeros;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_local_freq;
    int32_t _num_elements;

};

#endif // CL_PRESUM_ZERO_H

