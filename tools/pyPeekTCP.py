import time
import threading
import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as md
import datetime
import struct
import json

np.seterr(divide="ignore", invalid="ignore")

# target = 'B2111+46'
# target = 'B0329+54'
target = "B1133+16"

# struct IntensityHeader {
# 	int packet_length;		// - packet length
# 	int header_length;		// - header length
# 	int samples_per_packet;	// - number of samples in packet (or dimensions, n_freq x n_time x n_stream?)
# 	int sample_type;		// - data type of samples in packet
# 	double raw_cadence;		// - raw sample cadence
# 	int num_freqs;			// - freq list / map
# 	int samples_summed;		// - samples summed for each datum
# 	uint handshake_idx;		// - frame idx at handshake
# 	double handshake_utc;	// - UTC time at handshake
# 	char stokes_type; 		// - description of stream (e.g. V / H pol, Stokes-I / Q / U / V)
# 							//	-8	-7	-6	-5	-4	-3	-2	-1	1	2	3	4
# 							//	YX	XY	YY	XX	LR	RL	LL	RR	I	Q	U	V
# };
header_fmt = "=iiiidiiiId"
stokes_lookup = ["YX", "XY", "YY", "XX", "LR", "RL", "LL", "RR", "I", "Q", "U", "V"]

TCP_IP = "0.0.0.0"
TCP_PORT = 2061
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_IP, TCP_PORT))
sock.listen(1)

psrcat = json.load(open("psrcat_b.json"))["pulsars"]
psrdata = psrcat[target]


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
            tmpdata = 10 * np.log10(waterfold[:, :, i] / countfold[:, :, i])
            p[pkt_elems + i].set_data(
                tmpdata - np.median(tmpdata, axis=0)[np.newaxis, :]
            )
        else:
            p[i].set_data(waterfall[:, :, i])
            tmpdata = 10 * np.log10(waterfold[:, :, i] / countfold[:, :, i])
            p[pkt_elems + i].set_data(tmpdata)
        p[i].set_extent([freqlist[0, 0], freqlist[-1, -1], tmin, tmax])
        p[i].set_clim(vmin=colorscale[0], vmax=colorscale[1])
        p[pkt_elems + i].set_clim(vmin=colorscale[0] / 10, vmax=colorscale[1] / 10)
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
pkt_dtype = tcp_header[3]  # sample_type
pkt_raw_cad = tcp_header[4]  # raw_cadence
pkt_freqs = tcp_header[5]  # num_freqs
pkt_elems = tcp_header[6]  # num_freqs
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

plot_freqs = pkt_freqs / 8
# freqlist = freqlist.reshape(-1,plot_freqs).mean(axis=1)

plot_times = 256 * 4
plot_phase = 128
total_integration = 1024 * 8

if pkt_int_len > total_integration:
    print("Pre-integrated to longer than desired time!")
    print("{} vs {}".format(pkt_int_len, total_integration))
    print("Resetting integration length to {}".format(pkt_int_len))
    total_integration = pkt_int_len
local_integration = total_integration / pkt_int_len

waterfall = np.zeros((plot_times, plot_freqs, pkt_elems), dtype=np.float32) + np.nan
countfold = np.zeros((plot_phase, plot_freqs, pkt_elems), dtype=np.float32)
fold_period = 1.0 / psrdata["frequency"]
waterfold = np.zeros((plot_phase, plot_freqs, pkt_elems), dtype=np.float32)
times = np.zeros(plot_times)


def data_listener():
    global connection, sock
    global waterfall, waterfold, countfold
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
            waterfold *= 0.999
            countfold *= 0.999
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
                fold_idx = np.array(
                    (
                        (sec_per_pkt_frame * data_pkt_frame_idx + 0.5 * fold_period)
                        % fold_period
                    )
                    / fold_period
                    * plot_phase,
                    dtype=np.int32,
                )
                waterfold[fold_idx, :, data_pkt_elem_idx] += (
                    np.fromstring(data[pkt_header:], dtype=np.uint32)
                    .reshape(-1, pkt_freqs / plot_freqs)
                    .mean(axis=1)
                )
                countfold[fold_idx, :, data_pkt_elem_idx] += data_pkt_samples_summed
            roll_idx = (data_pkt_frame_idx - last_idx) / local_integration
            times = np.roll(times, roll_idx)
            times[0] = sec_per_pkt_frame * (data_pkt_frame_idx - pkt_idx0) + pkt_utc0
            # 		print(d,n)
            waterfall = np.roll(waterfall, roll_idx, axis=0)
            waterfall[0, :, :] = 10 * np.log10(
                (d / n).reshape(-1, pkt_freqs / plot_freqs, pkt_elems).mean(axis=1)
            )
            if np.mean(n) != total_integration:
                print(np.mean(n), np.std(n))
            last_idx = data_pkt_frame_idx
        # 		except socket.error, exc:
        except:
            connection, client_address = sock.accept()
            packed_header = receive(connection, 48)
            info_header = receive(connection, pkt_freqs * 4 * 2 + pkt_elems * 1)
            print("Reconnected!")


thread = threading.Thread(target=data_listener)
thread.daemon = True
thread.start()

time.sleep(1)
f, ax = plt.subplots(2, pkt_elems, gridspec_kw={"height_ratios": [2, 1]})
f.subplots_adjust(right=0.8)
if pkt_elems == 1:
    ax = [ax]
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
medsub = True
colorscale = [-0.5, 0.5]

for i in np.arange(pkt_elems):
    p.append(
        ax[0, i].imshow(
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
    ax[0, i].set_yticklabels([])
    ax[0, i].yaxis_date()

ax[0, 0].set_title(stokes_lookup[elemlist[0] + 8])
ax[0, 1].set_title(stokes_lookup[elemlist[1] + 8])

ax[0, 0].set_ylabel("Local Time")
ax[0, 0].yaxis_date()
ax[0, 0].yaxis.set_major_formatter(date_format)

for i in np.arange(pkt_elems):
    p.append(
        ax[1, i].imshow(
            waterfold[:, :, i],
            aspect="auto",
            animated=True,
            origin="upper",
            interpolation="nearest",
            cmap="gray",
            vmin=colorscale[0],
            vmax=colorscale[1],
            extent=[freqlist[0, 0], freqlist[-1, -1], 0, 1],
        )
    )
    ax[1, i].set_xlabel("Freq (MHz)")

ax[1, 0].set_ylabel("Pulse Phase")


cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
c = f.colorbar(p[0], cax=cbar_ax)
c.set_label("Power (dB, arbitrary)")

from matplotlib.widgets import Button

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
