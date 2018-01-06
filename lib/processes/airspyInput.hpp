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
#define NUM_COMMANDS 8
#define N_INPUT 2
#define BLOCK_LENGTH (65536*4)
#define BUF_BLOCKS 40
#define CMD_CHKCAL 6
#define CMD_AUTOCAL 7

#include <string>
using std::string;

struct ring_buffer{
    int head;
    int head_pos;
    int tail;
    int64_t sample_counter;
    pthread_mutex_t lock;
    void *blocks[BUF_BLOCKS];
};

struct command_queue{
    int cmd[NUM_COMMANDS];
    int cur,len;
};

struct msg_header{
    int type;
};


//int my_callback(airspy_transfer_t* transfer);

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
};


#endif 
