#ifndef FFTW_ENGINE_HPP
#define FFTW_ENGINE_HPP
#include <unistd.h>

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"
#include <fftw3.h>

#define BYTES_PER_SAMPLE 2

#include <string>
using std::string;

class fftwEngine : public KotekanProcess {
public:
    fftwEngine(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);
    virtual ~fftwEngine();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);

private:
    struct Buffer *buf_in;
    struct Buffer *buf_out;

    int frame_in;
    int frame_out;

    //options
    fftwf_complex *samples;
    fftwf_complex *spectrum;
    fftwf_plan fft_plan;
    int spectrum_length;
    int double_spectrum_length;
};


#endif 
