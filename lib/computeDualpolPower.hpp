#ifndef COMPUTE_DUALPOL_POWER
#define COMPUTE_DUALPOL_POWER


#include "buffers.h"
#include "errors.h"
#include "KotekanProcess.hpp"

class computeDualpolPower : public KotekanProcess {
public:
    computeDualpolPower(Config &config,
                     struct Buffer &buf_in,
                     struct Buffer &buf_out);
    virtual ~computeDualpolPower();
    void main_thread();

private:
    void fastSqSumVdif(unsigned char *data, int *temp_buf, float *output);

    struct Buffer &buf_in;
    struct Buffer &buf_out;

    int num_freq;
    int timesteps_in;
    int timesteps_out;
    int integration_length;

};

#endif