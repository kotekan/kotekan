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
    ~computeDualpolPower();
    void main_thread();
    void apply_config(uint64_t fpga_seq);

private:
    inline void fastSqSumVdif(unsigned char *data, int *temp_buf, float *output);
    void parallelSqSumVdif(int loop_idx, int loop_length);
    struct Buffer &buf_in;
    struct Buffer &buf_out;

    int num_freq;
    int timesteps_in;
    int timesteps_out;
    int integration_length;

    unsigned char *in_local;
    unsigned char *out_local;
};

#endif