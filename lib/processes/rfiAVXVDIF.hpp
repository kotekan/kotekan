#ifndef RFI_AVX_VDIF_HPP
#define RFI_AVX_VDIF_HPP

#include "buffer.h"
#include "errors.h"
#include "KotekanProcess.hpp"

class rfiAVXVDIF : public KotekanProcess {
public:
    rfiAVXVDIF(Config &config, const string& unique_name,
                        bufferContainer &buffer_container);
    ~rfiAVXVDIF();
    void main_thread();
    void apply_config(uint64_t fpga_seq);

private:

    inline void fastSKVDIF(uint8_t *data, uint32_t *temp_buf, uint32_t *sq_temp_buf, float *output);
    void parallelSpectralKurtosis(uint32_t loop_idx, uint32_t loop_length);
    struct Buffer *buf_in;
    struct Buffer *buf_out;

    uint32_t _num_local_freq;
    uint32_t _num_elements;
    uint32_t _samples_per_data_set;
    uint32_t _sk_step;
    bool _rfi_combined;
//    uint32_t *integration_count;
    uint8_t *in_local;
    uint8_t *out_local;
};

#endif
