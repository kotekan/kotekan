//#include <libairspy/airspy.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#include <pthread.h>
//#include <fftw3.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#include "airspy_control.h"


struct airspy_device *init_device(){
	/*int result;
	//airspy_read_partid_serialno_t read_partid_serialno;
	uint8_t board_id = AIRSPY_BOARD_ID_INVALID;

	struct airspy_device *dev;
	result = airspy_open(&dev);
	if( result != AIRSPY_SUCCESS ) {
		printf("airspy_open() failed: %s (%d)\n", airspy_error_name(result), result);
		airspy_exit();
	}

	int sample_rate_val=2500000;
//	int sample_rate_val=10000000;
	result = airspy_set_samplerate(dev, sample_rate_val);
	if (result != AIRSPY_SUCCESS) {
		printf("airspy_set_samplerate() failed: %s (%d)\n", airspy_error_name(result), result);
		airspy_close(dev);
		airspy_exit();
	}
	
	result = airspy_set_sample_type(dev, 5);
	if (result != AIRSPY_SUCCESS) {
		printf("airspy_set_sample_type() failed: %s (%d)\n", airspy_error_name(result), result);
		airspy_close(dev);
		airspy_exit();
	}

	result = airspy_set_vga_gain(dev, 15); //MAX:15
	if( result != AIRSPY_SUCCESS ) {
		printf("airspy_set_vga_gain() failed: %s (%d)\n", airspy_error_name(result), result);
	}

	result = airspy_set_freq(dev, 1420000000);
	if( result != AIRSPY_SUCCESS ) {
		printf("airspy_set_freq() failed: %s (%d)\n", airspy_error_name(result), result);
	}

	result = airspy_set_mixer_gain(dev, 15); //MAX: 15
	if( result != AIRSPY_SUCCESS ) {
		printf("airspy_set_mixer_gain() failed: %s (%d)\n", airspy_error_name(result), result);
	}
	result = airspy_set_mixer_agc(dev, 0); //Auto gain control: 0/1
	if( result != AIRSPY_SUCCESS ) {
		printf("airspy_set_mixer_agc() failed: %s (%d)\n", airspy_error_name(result), result);
	}

	result = airspy_set_lna_gain(dev, 14); //MAX: 14
	if( result != AIRSPY_SUCCESS ) {
		printf("airspy_set_lna_gain() failed: %s (%d)\n", airspy_error_name(result), result);
	}

	
	result = airspy_set_rf_bias(device, biast_val);
	if( result != AIRSPY_SUCCESS ) {
		printf("airspy_set_rf_bias() failed: %s (%d)\n", airspy_error_name(result), result);
		airspy_close(device);
		airspy_exit();
		return EXIT_FAILURE;
	}

	result = airspy_board_id_read(dev, &board_id);
	if (result != AIRSPY_SUCCESS) {
		fprintf(stderr, "airspy_board_id_read() failed: %s (%d)\n",
			airspy_error_name(result), result);
	}
	printf("Board ID Number: %d (%s)\n", board_id,
		airspy_board_id_name(board_id));

	result = airspy_board_partid_serialno_read(dev, &read_partid_serialno);
	if (result != AIRSPY_SUCCESS) {
		fprintf(stderr, "airspy_board_partid_serialno_read() failed: %s (%d)\n",
			airspy_error_name(result), result);
	}
	printf("Part ID Number: 0x%08X 0x%08X\n",
		read_partid_serialno.part_id[0],
		read_partid_serialno.part_id[1]);
	printf("Serial Number: 0x%08X%08X\n",
		read_partid_serialno.serial_no[2],
		read_partid_serialno.serial_no[3]);

	return dev;*/
}

void dispatch_message(void *msg, int length){
/*	static int init = 1;
	static int s;
	static struct sockaddr_in si;
	if (init){
		s=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
		setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &(int){ 1 }, sizeof(int));
		memset((char *) &si, 0, sizeof(si));
		si.sin_family = AF_INET;
		si.sin_port = htons(6547);
//		si.sin_addr.s_addr = inet_addr("127.0.0.1");
		si.sin_addr.s_addr = htonl(INADDR_ANY);
		bind(s,(struct sockaddr *)&si,sizeof(si));
		init=0;
	}
	sendto(s, msg, length, 0, (struct sockaddr *)&si, sizeof(si));
//	printf("Message sent, length %i.\n",length);
	printf("+");
	fflush(stdout);*/
}

/*
void auto_corr(fftwf_complex *spec_samp, float *spectrum, int nfreq){
	for (int i=0; i<nfreq; i++){
		spectrum[i]+= (spec_samp[i][0]*spec_samp[i][0] + spec_samp[i][1]*spec_samp[i][1]);
	}
}
void cross_corr(fftwf_complex *spec_a, fftwf_complex *spec_b, fftwf_complex *spectrum, int nfreq){
	for (int i=0; i<nfreq; i++){
		spectrum[i][0]+= (spec_a[i][0]*spec_b[i][0] + spec_a[i][1]*spec_b[i][1]);
		spectrum[i][1]+= (spec_a[i][1]*spec_b[i][0] - spec_a[i][0]*spec_b[i][1]);
	}
}*/

void set_lag(int n_samples){
	/*
	int n_blocks = abs(n_samples) / BLOCK_LENGTH;
	int n_offset = abs(n_samples) % BLOCK_LENGTH;

//	printf("Offsetting: %i, %i, %i\n", n_samples, n_blocks, n_offset);

	if (n_samples > 0) {
		pthread_mutex_lock(&buf[0].lock);
		buf[0].head = (buf[0].head + n_blocks + ((buf[0].head_pos+n_offset > BLOCK_LENGTH)? 1:0)) % BUF_BLOCKS;
		buf[0].head_pos = (buf[0].head_pos + n_offset*BYTES_PER_SAMPLE) % BLOCK_LENGTH;
//		printf("Head offset 0: %i, %i, %i\n",buf[0].head_pos, n_offset, (buf[0].head_pos + n_offset*BYTES_PER_SAMPLE) % BLOCK_LENGTH);
		pthread_mutex_unlock(&buf[0].lock);
	}
	else{
		pthread_mutex_lock(&buf[1].lock);
		buf[1].head = (buf[1].head + n_blocks + ((buf[1].head_pos+n_offset > BLOCK_LENGTH)? 1:0)) % BUF_BLOCKS;
		buf[1].head_pos = (buf[1].head_pos + n_offset*BYTES_PER_SAMPLE) % BLOCK_LENGTH;
//		printf("Head offset 1: %i, %i, %i\n",buf[0].head_pos, n_offset, (buf[0].head_pos + n_offset*BYTES_PER_SAMPLE) % BLOCK_LENGTH);
		pthread_mutex_unlock(&buf[1].lock);
	}*/
}

int push_command(int new_cmd) {
	/*
	pthread_mutex_lock(&action_lock);
	if (cmd.len < NUM_COMMANDS) {
		int cmd_new = (cmd.cur + cmd.len) % NUM_COMMANDS;
		cmd.cmd[cmd_new]=new_cmd;
		cmd.len++;
	} else printf("Oops, command buffer full! Dropping command! %i\n",new_cmd);
	pthread_mutex_unlock(&action_lock);
	return 0;*/
}


