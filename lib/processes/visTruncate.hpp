#ifndef VISTRUNCATE
#define VISTRUNCATE

#include "KotekanProcess.hpp"
#include "buffer.h"
#include <xmmintrin.h>
#include <immintrin.h>

class visTruncate : public KotekanProcess {
public:
    /// Constructor; loads parameters from config
    visTruncate(Config &config, const string& unique_name, bufferContainer &buffer_container);
    ~visTruncate();

    /// Main loop over buffer frames
    void main_thread() override;

    void apply_config(uint64_t fpga_seq) override;
private:
    // Buffers
    Buffer * in_buf;
    Buffer * out_buf;

    // Truncation parameters
    float err_sq_lim;
    float w_prec;
    float vis_prec;

    // Timing
    double start_time;;
    double wait_time = 0.;
    double truncate_time = 0.;
    double copy_time = 0.;
    double last_time;

};

#endif
