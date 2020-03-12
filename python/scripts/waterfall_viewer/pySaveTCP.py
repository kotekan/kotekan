# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from future import standard_library

standard_library.install_aliases()
import time
import threading
import socket
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as md
import datetime
import struct

# struct IntensityHeader {
#   int packet_length;      // - packet length
#   int header_length;      // - header length
#   int samples_per_packet; // - number of samples in packet (or dimensions, n_freq x n_time x n_stream?)
#   int sample_type;        // - data type of samples in packet
#   double raw_cadence;     // - raw sample cadence
#   int num_freqs;          // - freq list / map
#   int samples_summed;     // - samples summed for each datum
#   uint handshake_idx;     // - frame idx at handshake
#   double handshake_utc;   // - UTC time at handshake
#   char stokes_type;       // - description of stream (e.g. V / H pol, Stokes-I / Q / U / V)
#                           //  -8  -7  -6  -5  -4  -3  -2  -1  1   2   3   4
#                           //  YX  XY  YY  XX  LR  RL  LL  RR  I   Q   U   V
# };
dest_file = "test.dat"
output_file = open(dest_file, "wb")

header_fmt = "=iiiidiiiId"
TCP_IP = "0.0.0.0"
TCP_PORT = 12051
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_IP, TCP_PORT))
sock.listen(1)


def updatefig(*args):
    global waterfall, times, medsub, colorscale
    tmin = md.date2num(datetime.datetime.fromtimestamp(np.amin(times)))
    tmax = md.date2num(datetime.datetime.fromtimestamp(np.amax(times)))
    for i in np.arange(pkt_elems):
        if medsub:
            p[i].set_data(
                waterfall[:, :, i]
                - np.nanmedian(waterfall[:, :, i], axis=0)[np.newaxis, :]
            )
        else:
            p[i].set_data(waterfall[:, :, i])
        p[i].set_extent([freqlist[0, 0], freqlist[-1, -1], tmin, tmax])
        p[i].set_clim(vmin=colorscale[0], vmax=colorscale[1])
    return (p,)


connection, client_address = sock.accept()
packed_header = connection.recv(48)
print(len(packed_header), packed_header)
output_file.write(packed_header)

tcp_header = struct.unpack(header_fmt, packed_header)

pkt_length = tcp_header[0]  # packet_length
pkt_header = tcp_header[1]  # header_length
pkt_samples = tcp_header[2]  # samples_per_packet
pkt_dtype = tcp_header[3]  # sample_type
pkt_raw_cad = tcp_header[4]  # raw_cadence
pkt_freqs = tcp_header[5]  # num_freqs
pkt_elems = tcp_header[6]  # num_freqs
pkt_int_len = tcp_header[7]  # samples_summed
pkt_idx0 = tcp_header[8]  # handshake_idx
pkt_utc0 = tcp_header[9]  # handshake_utc

sec_per_pkt_frame = pkt_raw_cad * pkt_int_len

info_header = connection.recv(pkt_freqs * 4 * 2 + pkt_elems * 1)
output_file.write(info_header)
output_file.close()

freqlist = np.fromstring(info_header[: pkt_freqs * 4 * 2], dtype=np.float32).reshape(
    -1, 2
)  # .mean(axis=1)
freqlist = freqlist / 1e6
elemlist = np.fromstring(info_header[pkt_freqs * 4 * 2 :], dtype=np.uint8)

plot_freqs = pkt_freqs // 8
# freqlist = freqlist.reshape(-1,plot_freqs).mean(axis=1)

plot_times = 128 * 4

plot_integration = 2

waterfall = np.zeros((plot_times, plot_freqs, 2), dtype=np.float32) + np.nan
times = np.zeros(plot_times)


def receive(connection, length):
    chunks = []
    bytes_recd = 0
    while bytes_recd < length:
        chunk = connection.recv(min(length - bytes_recd, 2048))
        if chunk == b"":
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_recd = bytes_recd + len(chunk)
    return b"".join(chunks)


def data_listener():
    global waterfall, times
    last_idx = pkt_idx0
    data_pkt_frame_idx = 0
    data_pkt_samples_summed = 1
    idx = 0
    while True:
        d = np.zeros([pkt_freqs, pkt_elems])
        t = np.zeros(plot_times)
        for i in np.arange(plot_integration * pkt_elems):
            data = receive(connection, pkt_length + pkt_header)
            if len(data) != pkt_length + pkt_header:
                print("Lost Connection!")
                connection.close()
                return
            output_file = open(dest_file, "ab")
            output_file.write(data)
            output_file.close()
            (
                data_pkt_frame_idx,
                data_pkt_elem_idx,
                data_pkt_samples_summed,
            ) = struct.unpack("III", data[:pkt_header])
            d[:, data_pkt_elem_idx] += (
                np.fromstring(data[pkt_header:], dtype=np.uint32)
                * 1.0
                / plot_integration
                / data_pkt_samples_summed
            )
        times = np.roll(times, (data_pkt_frame_idx - last_idx) // plot_integration)
        times[0] = sec_per_pkt_frame * (data_pkt_frame_idx - pkt_idx0) + pkt_utc0
        waterfall = np.roll(
            waterfall, (data_pkt_frame_idx - last_idx) // plot_integration, axis=0
        )
        waterfall[0, :, :] = 10 * np.log10(
            d.reshape(-1, pkt_freqs // plot_freqs, pkt_elems).mean(axis=1)
        )
        last_idx = data_pkt_frame_idx


thread = threading.Thread(target=data_listener)
thread.daemon = True
thread.start()

time.sleep(1)

f, ax = plt.subplots(1, pkt_elems)
f.subplots_adjust(right=0.8)
if pkt_elems == 1:
    ax = [ax]
plt.ioff()
p = []
tmin = md.date2num(
    datetime.datetime.fromtimestamp(
        pkt_utc0 - plot_times * plot_integration * sec_per_pkt_frame
    )
)
tmax = md.date2num(datetime.datetime.fromtimestamp(pkt_utc0))
times = pkt_utc0 - np.arange(plot_times) * plot_integration * sec_per_pkt_frame
# waterfall[:,:,:] = times[:,np.newaxis,np.newaxis]
date_format = md.DateFormatter("%H:%M:%S")
medsub = True
colorscale = [-0.5, 0.5]

for i in np.arange(pkt_elems):
    p.append(
        ax[i].imshow(
            waterfall[:, :, i],
            aspect="auto",
            animated=True,
            origin="upper",
            interpolation="nearest",
            cmap="gray",
            vmin=colorscale[0],
            vmax=colorscale[1],
            extent=[freqlist[0, 0], freqlist[-1, -1], tmin, tmax],
        )
    )
    ax[i].set_xlabel("Freq (MHz)")
    ax[i].set_yticklabels([])
    ax[i].yaxis_date()

ax[0].set_ylabel("Local Time")
ax[0].yaxis_date()
ax[0].yaxis.set_major_formatter(date_format)

cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
c = f.colorbar(p[0], cax=cbar_ax)
c.set_label("Power (dB, arbitrary)")

from matplotlib.widgets import Slider, Button

rax = plt.axes([0.82, 0.03, 0.15, 0.04])
check = Button(rax, "Med Subtract")


def func(event):
    global medsub, check, colorscale
    medsub = not medsub
    if medsub:
        check.label.set_text("Med Subtracted")
        colorscale = [-0.5, 0.5]
    else:
        check.label.set_text("Raw Power")
        colorscale = [-10, 10]


check.on_clicked(func)

ani = animation.FuncAnimation(f, updatefig, frames=100, interval=100)
f.show()
