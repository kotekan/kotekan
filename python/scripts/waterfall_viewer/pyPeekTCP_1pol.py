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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as md
import datetime
import struct

np.seterr(divide="ignore", invalid="ignore")

# struct IntensityHeader {
#    int packet_length;      // - packet length
#    int header_length;      // - header length
#    int samples_per_packet; // - number of samples in packet (or dimensions, n_freq x n_time x n_stream?)
#    int sample_type;        // - data type of samples in packet
#    double raw_cadence;     // - raw sample cadence
#    int num_freqs;          // - freq list / map
#    int samples_summed;     // - samples summed for each datum
#    uint handshake_idx;     // - frame idx at handshake
#    double handshake_utc;   // - UTC time at handshake
#    char stokes_type;       // - description of stream (e.g. V / H pol, Stokes-I / Q / U / V)
#                            //    -8    -7    -6    -5    -4    -3    -2    -1    1    2    3    4
#                            //    YX    XY    YY    XX    LR    RL    LL    RR    I    Q    U    V
# };
header_fmt = "=iiiidiiiId"
stokes_lookup = ["YX", "XY", "YY", "XX", "LR", "RL", "LL", "RR", "I", "Q", "U", "V"]

TCP_IP = "0.0.0.0"
TCP_PORT = 23401
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_IP, TCP_PORT))
sock.listen(1)


def updatefig(*args):
    global waterfall, times, medsub, colorscale, p, ax
    global spec_baseline
    global pkt_elems
    tmin = md.date2num(datetime.datetime.fromtimestamp(np.amin(times)))
    tmax = md.date2num(datetime.datetime.fromtimestamp(np.amax(times)))
    if medsub:
        p[0].set_data(
            waterfall[:, :, 0] - np.nanmedian(waterfall[:, :, 0], axis=0)[np.newaxis, :]
        )
    else:
        p[0].set_data(waterfall[:, :, 0] - spec_baseline[np.newaxis, :])
    p[0].set_extent([freqlist[0, 0], freqlist[-1, -1], tmin, tmax])
    p[0].set_clim(vmin=colorscale[0], vmax=colorscale[1])

    d = np.nanmean(waterfall[:, :, 0] - spec_baseline[np.newaxis, :], axis=1)
    p[1].set_data(d, times)
    ax[0, 1].set_xlim([np.nanmin(d), np.nanmax(d)])
    ax[0, 1].set_ylim([np.amin(times), np.amax(times)])

    d = np.nanmean(waterfall[:, :, 0], axis=0)
    p[2].set_data(
        freqlist.reshape(plot_freqs, -1).mean(axis=1), d - spec_baseline[np.newaxis, :]
    )
    ax[1, 0].set_ylim(colorscale)
    return (p,)


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


connection, client_address = sock.accept()
packed_header = receive(connection, 48)
print(len(packed_header), packed_header)
tcp_header = struct.unpack(header_fmt, packed_header)

pkt_length = tcp_header[0]  # packet_length
pkt_header = tcp_header[1]  # header_length
pkt_samples = tcp_header[2]  # samples_per_packet
pkt_dtype = tcp_header[3]  # sample_type (?in bytes?)
pkt_raw_cad = tcp_header[4]  # raw_cadence
pkt_freqs = tcp_header[5]  # num_freqs
pkt_elems = tcp_header[6]  # num_elems
pkt_int_len = tcp_header[7]  # samples_summed
pkt_idx0 = tcp_header[8]  # handshake_idx
pkt_utc0 = tcp_header[9]  # handshake_utc

print(tcp_header)

sec_per_pkt_frame = pkt_raw_cad * pkt_int_len

info_header = receive(connection, pkt_freqs * 4 * 2 + pkt_elems * 1)
freqlist = np.fromstring(info_header[: pkt_freqs * 4 * 2], dtype=np.float32).reshape(
    -1, 2
)  # .mean(axis=1)
freqlist = freqlist / 1e6
elemlist = np.fromstring(info_header[pkt_freqs * 4 * 2 :], dtype=np.int8)

print(freqlist, elemlist)

plot_freqs = 256
plot_times = 256
total_integration = 64 * 8

if pkt_int_len > total_integration:
    print("Pre-integrated to longer than desired time!")
    print("{} vs {}".format(pkt_int_len, total_integration))
    print("Resetting integration length to {}".format(pkt_int_len))
    total_integration = pkt_int_len
local_integration = total_integration // pkt_int_len

waterfall = np.zeros((plot_times, plot_freqs, pkt_elems), dtype=np.float32) + np.nan
times = np.zeros(plot_times)
spec_baseline = np.ones(plot_freqs)


def data_listener():
    global connection, sock
    global waterfall
    global times, total_integration, pkt_idx0
    last_idx = pkt_idx0
    data_pkt_frame_idx = 0
    data_pkt_samples_summed = 1
    idx = 0
    while True:
        try:
            d = np.zeros([pkt_freqs, pkt_elems])
            n = np.zeros([pkt_freqs, pkt_elems])
            t = np.zeros(plot_times)
            for _ in np.arange(local_integration * pkt_elems):
                data = receive(connection, pkt_length + pkt_header)
                if len(data) != pkt_length + pkt_header:
                    print("Lost Connection!")
                    connection.close()
                    return
                (
                    data_pkt_frame_idx,
                    data_pkt_elem_idx,
                    data_pkt_samples_summed,
                ) = struct.unpack("III", data[:pkt_header])
                d[:, data_pkt_elem_idx] += (
                    np.fromstring(data[pkt_header:], dtype=np.uint32) * 1.0
                )
                n[:, data_pkt_elem_idx] += data_pkt_samples_summed * 1.0
            roll_idx = (data_pkt_frame_idx - last_idx) // local_integration
            times = np.roll(times, roll_idx)
            times[0] = sec_per_pkt_frame * (data_pkt_frame_idx - pkt_idx0) + pkt_utc0
            waterfall = np.roll(waterfall, roll_idx, axis=0)
            waterfall[0, :, :] = 10 * np.log10(
                (d / n).reshape(-1, pkt_freqs // plot_freqs, pkt_elems).mean(axis=1)
            )
            if np.mean(n) != total_integration:
                print(np.mean(n), np.std(n))
            last_idx = data_pkt_frame_idx
        #        except socket.error, exc:
        except:
            connection, client_address = sock.accept()
            packed_header = receive(connection, 48)
            info_header = receive(connection, pkt_freqs * 4 * 2 + pkt_elems * 1)
            print("Reconnected!")


thread = threading.Thread(target=data_listener)
thread.daemon = True
thread.start()

time.sleep(1)
f, ax = plt.subplots(
    2, 2, gridspec_kw={"height_ratios": [4, 1], "width_ratios": [4, 1]}
)
f.subplots_adjust(right=0.8, top=0.95, wspace=0.0, hspace=0.0)
ax[-1, -1].axis("off")


plt.ioff()
p = []
tmin = md.date2num(
    datetime.datetime.fromtimestamp(
        pkt_utc0 - plot_times * local_integration * sec_per_pkt_frame
    )
)
tmax = md.date2num(datetime.datetime.fromtimestamp(pkt_utc0))
times = pkt_utc0 - np.arange(plot_times) * local_integration * sec_per_pkt_frame
date_format = md.DateFormatter("%H:%M:%S")
medsub = False
med_range = [-1, 1]
full_range = [40, 60]

colorscale = med_range if medsub else full_range
oc = colorscale

p.append(
    ax[0][0].imshow(
        waterfall[:, :, 0],
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
ax[0][0].set_yticklabels([])
ax[0][0].yaxis_date()
ax[0][0].set_title(stokes_lookup[elemlist[0] + 8])
ax[0][0].set_ylabel("Local Time")
ax[0][0].yaxis_date()
ax[0][0].xaxis.set_visible(False)
ax[0][0].yaxis.set_major_formatter(date_format)

ax[0][1].yaxis.set_visible(False)

d = np.nanmean(waterfall[:, :, 0], axis=1)
#ax[0][1].set_xlim(np.amin(d), np.amax(d))
ax[0][1].set_ylim([tmin, tmax])
(im,) = ax[0][1].plot(d, times, ".")
ax[0][1].set_xlabel("Power (dB, arb)")
p.append(im)

ax[1][0].set_xlim(freqlist[0, 0], freqlist[-1, -1])
ax[1][0].set_ylim(colorscale)
(im,) = ax[1][0].plot(
    freqlist.reshape(plot_freqs, -1).mean(axis=1),
    np.nanmean(waterfall[:, :, 0], axis=0),
    ".",
)
p.append(im)
ax[1][0].set_xlabel("Frequency (MHz)")
ax[1][0].set_ylabel("Power (dB, arb)")

cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.8])
c = f.colorbar(p[0], cax=cbar_ax)
c.set_label("Power (dB, arbitrary)")

ani = animation.FuncAnimation(f, updatefig, frames=100, interval=100)
f.show()


# UI
from matplotlib.widgets import Button

import pickle


def save(event):
    global check, waterfall, times, freqlist
    freqs = freqlist.reshape(plot_freqs, -1).mean(axis=1)
    data = 10 ** (waterfall / 10)
    fn = time.strftime("MP_%Y%m%d-%H%M%S.pkl")
    pickle.dump({"freqs": freqs, "times": times, "data": data}, open(fn, "wb"))
    print("Data Saved to {}".format(fn))


rax = plt.axes([0.82, 0.03, 0.13, 0.04])
save_btn = Button(rax, "Save Data")
save_btn.on_clicked(save)


def set_spec_baseline(event):
    global spec_baseline, waterfall, colorscale, oc
    spec_baseline = np.nanmean(waterfall[:, :, 0], axis=0)
    colorscale = [-1, 1]
    print("Set a new spectral baseline")


rax = plt.axes([0.7, 0.07, 0.13, 0.04])
sb_btn = Button(rax, "Renorm Spec")
sb_btn.on_clicked(set_spec_baseline)


def reset_spec_baseline(event):
    global spec_baseline, plot_freqs, colorscale, oc
    spec_baseline = np.ones(plot_freqs)
    colorscale = oc
    print("Removed spectral baseline")


rax = plt.axes([0.82, 0.07, 0.13, 0.04])
rsb_btn = Button(rax, "Un-Renorm")
rsb_btn.on_clicked(reset_spec_baseline)
