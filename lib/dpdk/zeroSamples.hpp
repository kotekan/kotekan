#ifndef ZERO_SAMPLES_HPP
#define ZERO_SAMPLES_HPP

#include "KotekanProcess.hpp"

/**
 * @brief Zeros samples in the @c out_buf based on flags in the @lost_samples_buf
 *
 * Note the synchronization is a little non-standard here.  We wait for the buffer
 * which contains the flags to be full and register as a consumer on that buffer.
 * Because we know that will only happen once the data buffer is full, we can use
 * that as the synchronization on the data, and so can start zeroing data in the data
 * buffer (which we operate on as a producer).
 *
 * @author Andre Renard
 */
class zeroSamples : public KotekanProcess {
public:

    zeroSamples(Config& config, const string& unique_name,
             bufferContainer &buffer_container);
    ~zeroSamples();

    void main_thread();
    void apply_config(uint64_t fpga_seq) {};

private:

    struct Buffer * out_buf;
    struct Buffer * lost_samples_buf;

    int32_t out_buf_frame_id = 0;
    int32_t lost_samples_buf_frame_id = 0;

    uint32_t sample_size;
};

#endif /* ZERO_SAMPLES_HPP */