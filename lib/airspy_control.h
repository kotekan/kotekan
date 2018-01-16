#ifndef AIRSPY_CONTROL_H
#define AIRSPY_CONTROL_H

#include "buffer.h"
#include "errors.h"

//#include <libairspy/airspy.h>
//#include <fftw3.h>
#ifdef __cplusplus
extern "C" {
#endif

#define BYTES_PER_SAMPLE 2
#define NUM_COMMANDS 8
#define N_INPUT 2
#define BLOCK_LENGTH (65536*4)
#define BUF_BLOCKS 40
#define CMD_CHKCAL 6
#define CMD_AUTOCAL 7
/*
pthread_cond_t new_data;
pthread_mutex_t readlock;
pthread_mutex_t action_lock;
*/
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

struct airspy_device *init_device();

void dispatch_message(void *msg, int length);

//void auto_corr(fftwf_complex *spec_samp, float *spectrum, int nfreq);

//void cross_corr(fftwf_complex *spec_a, fftwf_complex *spec_b, fftwf_complex *spectrum, int nfreq);

void set_lag(int n_samples);

//int airspy_producer(airspy_transfer_t* transfer);

int push_command(int new_cmd);

#ifdef __cplusplus
}
#endif

#endif
