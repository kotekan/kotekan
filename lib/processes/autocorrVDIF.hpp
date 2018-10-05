#ifndef AUTOCORR_VDIF
#define AUTOCORR_VDIF

#include "buffer.h"
#include "errors.h"
#include "KotekanProcess.hpp"

class autocorrVDIF : public KotekanProcess {
public:
    autocorrVDIF(Config &config, const string& unique_name,
                        bufferContainer &buffer_container);
    ~autocorrVDIF();
    void main_thread();
    void apply_config(uint64_t fpga_seq);

private:
    inline void fastSqSumVdif(uint8_t *input, uint32_t *output, uint32_t *outcount);
    void parallelSqSumVdif(int loop_idx, int loop_length);
    struct Buffer *buf_in;
    struct Buffer *buf_out;

    int num_pol;
    int num_freq;
    int num_elem;
    int timesteps_in;
    int timesteps_out;
    int integration_length;
    unsigned char *in_local;
    unsigned char *out_local;
    uint32_t *out, *outcount;
    int foldbins;
    float foldperiod;
};

#endif
