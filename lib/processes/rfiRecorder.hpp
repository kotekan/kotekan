#ifndef RFI_RECORDER_H
#define RFI_RECORDER_H

#include "KotekanProcess.hpp"

class rfiRecorder : public KotekanProcess {
public:
    rfiRecorder(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);
    virtual ~rfiRecorder();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);
private:
    struct Buffer *rfi_buf;
    int _num_local_freq;
    int _num_elements;
    int _samples_per_data_set;
    int _sk_step;
    int _buf_depth;
    int slot_id;
    int link_id;
    FILE *f;
};

#endif
