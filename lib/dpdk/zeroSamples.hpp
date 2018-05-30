#ifndef ZERO_SAMPLES_HPP
#define ZERO_SAMPLES_HPP

#include "KotekanProcess.hpp"

class zeroSamples : public KotekanProcess {
public:

    zeroSamples(Config& config, const string& unique_name,
             bufferContainer &buffer_container);
    ~zeroSamples();

    void main_thread();
    void apply_config(uint64_t fpga_seq) {};

private:

    struct Buffer &in_buf;
};

#endif /* ZERO_SAMPLES_HPP */