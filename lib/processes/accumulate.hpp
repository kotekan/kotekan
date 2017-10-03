#ifndef ACCUMULATE_HPP
#define ACCUMULATE_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"

struct rawGPUFrameHeader {
    uint64_t fpga_seq_num;
    uint32_t epoch_time_sec;
    uint32_t epoch_time_usec;
    uint32_t stream_id;
    uint32_t lost_frames;
    uint64_t unused;
};

class accumulate : public KotekanProcess {
public:
    accumulate(Config& config,
                const string& unique_name,
                bufferContainer &buffer_container);
    ~accumulate();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *in_buf;
    struct Buffer *out_buf;

    int32_t _samples_per_data_set;
    int32_t _num_gpu_frames;
};

#endif