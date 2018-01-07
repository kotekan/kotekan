#ifndef AIRSPY_INPUT_HPP
#define AIRSPY_INPUT_HPP
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//#include <time.h>
//#include <sys/time.h>
//#include <math.h>

#include <unistd.h>
#include <libairspy/airspy.h>

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

#define BYTES_PER_SAMPLE 2

#include <string>
using std::string;

class airspyInput : public KotekanProcess {
public:
    airspyInput(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);
    virtual ~airspyInput();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);

    struct airspy_device *init_device();
    static int airspy_callback(airspy_transfer_t* transfer);
    void airspy_producer(airspy_transfer_t* transfer);

private:
    struct Buffer *buf;
    struct airspy_device* a_device;

    unsigned char* buf_ptr;
    unsigned int frame_loc;
    int frame_id;
    pthread_mutex_t recv_busy;


    //options
    int freq; //Hz
    int sample_bw; //Hz
    int gain_lna;
    int gain_mix;
    int gain_if;

    int biast_power;
};


#endif 
