#include "simVdifData.hpp"
#include <random>
#include "errors.h"
#include <sys/time.h>
#include <unistd.h>

simVdifData::simVdifData(Config& config, const string& unique_name,
                        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&simVdifData::main_thread, this))
    {
    buf = buffer_container.get_buffer("network_buf");
}

simVdifData::~simVdifData() {

}

void simVdifData::apply_config(uint64_t fpga_seq) {

}


double e_time(void)
{
  static struct timeval now;
  gettimeofday(&now, NULL);
  return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

void simVdifData::main_thread() {
    int times = config.get_int(unique_name, "samples_per_data_set");
    int elements = config.get_int(unique_name, "num_elements");
    int freqs = config.get_int(unique_name, "num_local_freq");

    double time_available =  2.56e-6 * times; //microseconds
    int buf_id = 0;

    struct VDIFHeader header = {
            0,      //uint32_t seconds : 30;
            0,      //uint32_t legacy : 1;
            0,      //uint32_t invalid : 1;
            0,      //uint32_t data_frame : 24;
            0,      //uint32_t ref_epoch : 6;
            0,      //uint32_t unused : 2;
            0,      //uint32_t frame_len : 24;
            0,      //uint32_t log_num_chan : 5;
            0,      //uint32_t vdif_version : 3;
            0,      //uint32_t station_id : 16;
            0,      //uint32_t thread_id : 10;
            0,      //uint32_t bits_depth : 5;
            0,      //uint32_t data_type : 1;
            0,      //uint32_t eud1 : 24;
            0,      //uint32_t edv : 8;
            0,      //uint32_t eud2 : 32;
            0,      //uint32_t eud3 : 32;
            0       //uint32_t eud4 : 32;
    };
    int frame_length = sizeof(header) + freqs * sizeof(char);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    start_time = e_time();
//    for (int ct=0; ct<100; ct++) {
    for (;;) {
        wait_for_empty_buffer(buf, buf_id);
        stop_time = e_time();
        double dt = stop_time-start_time;
        if (dt < time_available) {
            usleep((time_available-dt) * 1e6);
            stop_time = e_time();
        }
        unsigned char* buf_ptr = buf->data[buf_id];
        for (int t = 0; t < times; t++) {
            for (int e = 0; e < elements; e++){
                memcpy(buf_ptr,(void*)&header,sizeof(header));
                buf_ptr+=sizeof(header);
                memset(buf_ptr,(unsigned char)dis(gen),freqs*sizeof(char));
//                for (int f = 0; f < freqs; f++) buf_ptr[f] = (f/4 % 16);
//                for (int f = 0; f < freqs; f++) buf_ptr[f] = (unsigned char)dis(gen);
                buf_ptr+=freqs;
            }
        }
//        INFO("Generated a test data set in %s[%d]", buf.buffer_name, buf_id);

        mark_buffer_full(buf, buf_id);
        buf_id = (buf_id + 1) % buf->num_buffers;
        header.data_frame++;

        INFO("%4.1f%% of %6.4fs available.\n",dt/time_available*100,time_available);
        start_time=stop_time;
    }
    mark_producer_done(buf, 0);
}

