#ifndef SIMPLE_AUTOCORR_HPP
#define SIMPLE_AUTOCORR_HPP
#include <unistd.h>

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

#define BYTES_PER_SAMPLE 2

#include <string>
using std::string;

class simpleAutocorr : public KotekanProcess {
public:
    simpleAutocorr(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);
    virtual ~simpleAutocorr();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);

private:
    struct Buffer *buf_in;
    struct Buffer *buf_out;

    int frame_in;
    int frame_out;

    //options
    int spectrum_length;
    int integration_length;
    float *spectrum_out;
};


#endif 
