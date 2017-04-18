#ifndef NETWORK_POWER_STREAM_H
#define NETWORK_POWER_STREAM_H

#include <sys/socket.h>
#include "Config.hpp"
#include "buffers.h"
#include "KotekanProcess.hpp"
#include <atomic>

struct  __attribute__((__packed__)) IntensityHeader {
	int packet_length;		// - packet length
	int header_length;		// - header length
	int samples_per_packet;	// - number of samples in packet (or dimensions, n_freq x n_time x n_stream?)
	int sample_type;		// - data type of samples in packet
	double raw_cadence;		// - raw sample cadence
	int num_freqs;			// - freq list / map
	int samples_summed;		// - samples summed for each datum
	uint handshake_idx;		// - frame idx at handshake
	double handshake_utc;	// - UTC time at handshake
	char stokes_type; 		// - description of stream (e.g. V / H pol, Stokes-I / Q / U / V)
							//	-8	-7	-6	-5	-4	-3	-2	-1	1	2	3	4
							//	YX	XY	YY	XX	LR	RL	LL	RR	I	Q	U	V
};

struct  __attribute__((__packed__)) IntensityPacketHeader {
	uint frame_idx;			//- frame idx
	uint samples_summed;	//- number of samples integrated
};


class networkPowerStream : public KotekanProcess {
public:
    networkPowerStream(Config &config, struct Buffer &buf);
    virtual ~networkPowerStream();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:
	void tcpConnect();

    struct Buffer &buf;

    uint32_t dest_port;
    string dest_server_ip;
    string dest_protocol;

    int socket_fd;
    bool tcp_connected=false;
    bool tcp_connecting=false;
	std::thread connect_thread;
    std::atomic_flag socket_lock;

    int freqs;
    int times;

    uint frame_idx=0;

    uint64_t handshake_idx=-1;
    double handshake_utc=-1;

	IntensityHeader header;
};

#endif