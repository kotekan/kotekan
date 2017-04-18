import time
import threading
import socket
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import struct

#struct IntensityHeader {
#	int packet_length;		// - packet length
#	int header_length;		// - header length
#	int samples_per_packet;	// - number of samples in packet (or dimensions, n_freq x n_time x n_stream?)
#	int sample_type;		// - data type of samples in packet
#	double raw_cadence;		// - raw sample cadence
#	int num_freqs;			// - freq list / map
#	int samples_summed;		// - samples summed for each datum
#	uint handshake_idx;		// - frame idx at handshake
#	double handshake_utc;	// - UTC time at handshake
#	char stokes_type; 		// - description of stream (e.g. V / H pol, Stokes-I / Q / U / V)
#							//	-8	-7	-6	-5	-4	-3	-2	-1	1	2	3	4
#							//	YX	XY	YY	XX	LR	RL	LL	RR	I	Q	U	V
#};
header_fmt = '=iiiidiiIdb'

TCP_IP="127.0.0.1"
TCP_PORT = 2054
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_IP, TCP_PORT))
sock.listen(1)

def updatefig(*args):
    global waterfall
    p.set_data(waterfall)
    return p,

connection, client_address = sock.accept()
packed_header = connection.recv(1024)
tcp_header  = struct.unpack(header_fmt,packed_header)

pkt_length  = tcp_header[0] # packet_length
pkt_header  = tcp_header[1] # header_length
pkt_samples = tcp_header[2] # samples_per_packet
pkt_dtype   = tcp_header[3] # sample_type
pkt_raw_cad = tcp_header[4] # raw_cadence
pkt_freqs   = tcp_header[5] # num_freqs
pkt_int_len = tcp_header[6] # samples_summed
pkt_idx0	= tcp_header[7] # handshake_idx
pkt_utc0	= tcp_header[8] # handshake_utc
pkt_stokes  = tcp_header[9] # stokes_type

plot_freqs=pkt_freqs/4
plot_times=128
plot_integration=64

waterfall = np.zeros((plot_times,plot_freqs),dtype=np.float32)

def receive(connection,length):
    chunks = []
    bytes_recd = 0
    while bytes_recd < length:
        chunk = connection.recv(min(length - bytes_recd, 2048))
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_recd = bytes_recd + len(chunk)
    return b''.join(chunks)


def data_listener():
	global waterfall
	data_pkt_frame_idx = 0;
	data_pkt_samples_summed = 1;
	idx=0
	while True:
		d=np.zeros(pkt_freqs)
		for i in np.arange(plot_integration):
			data = receive(connection,pkt_length+pkt_header)
			if (len(data) != pkt_length+pkt_header):
				print("Lost Conenction!")
				connection.close()
				return;
			data_pkt_frame_idx, data_pkt_samples_summed = struct.unpack('ii',data[:pkt_header])
			d += np.fromstring(data[pkt_header:],dtype=np.float32) / data_pkt_samples_summed
		idx=(data_pkt_frame_idx/plot_integration) % plot_times
		waterfall = np.roll(waterfall,1,axis=0)
		waterfall[0,:]=10*np.log10(d.reshape(-1,pkt_freqs / plot_freqs).mean(axis=1)/plot_integration) 

thread = threading.Thread(target=data_listener)
thread.daemon = True
thread.start()

time.sleep(1)

f, ax = plt.subplots()
plt.ioff()
p=ax.imshow(waterfall,aspect='auto',animated=True,origin='lower',interpolation='nearest', \
			cmap='gray',vmin=16, vmax=17)
c = f.colorbar(p)
ax.set_xlabel('Freq')
ax.set_ylabel('Time')
c.set_label('Power (dB, arbitrary)')
ani = animation.FuncAnimation(f, updatefig, frames=100, interval=100)
f.show()













