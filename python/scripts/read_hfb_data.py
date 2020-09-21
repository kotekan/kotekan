import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from kotekan import hfbbuffer


def main():

    in_file = sys.argv[1]

    print("\nReading file: {}".format(in_file))

    freq_id = int(sys.argv[2])
    beam_id = int(sys.argv[3])
    nframes = 128
    nsubfreq = 128

    f = hfbbuffer.HFBRaw.from_file(in_file)

    metadata = f.metadata
    data = f.data["hfb"]

    print("\nNo. of valid frames in the file: %d" % np.sum(f.valid_frames))

    print("\nMetadata")
    print("--------")
    print(metadata.dtype)
    print(metadata[0][freq_id])

    plot_sky_map(metadata, data, nsubfreq, freq_id)
    #plot_one_beam_freq_over_time(metadata, data, nframes, nsubfreq, freq_id, beam_id)


def plot_sky_map(metadata, data, nsubfreq, freq_id):
    nframes = 512
    nbeams = 1024
    ns_beams = 256

    freq_start = 800.0 - float(freq_id) * 400.0 / 1024.0

    hfb_square = np.zeros((nframes, ns_beams), dtype=np.float32)

    # Get the time in UTC
    ts = int(metadata["ctime"][0][freq_id][0])
    date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    # Filter specific beam from data
    ctr = 0
    for d in data[:nframes]:
        beam_sum = np.zeros(nbeams, dtype=np.float32)
        beam_ns_sum = np.zeros(ns_beams, dtype=np.float32)

        for beam_id in range(0, nbeams):
            beam_sum[beam_id] += np.sum(d[freq_id][beam_id * nsubfreq : (beam_id + 1) * nsubfreq])
                    
        for beam_id in range(0, ns_beams):
            beam_ns_sum[beam_id] += beam_sum[beam_id * 4]

        hfb_square[ctr] = beam_ns_sum
        ctr = ctr + 1

    # Plot data
    plt.imshow(np.log(np.transpose(hfb_square)), interpolation="none")
    plt.ylabel("N-S Beam Index")
    plt.yticks(np.arange(0, ns_beams, 50))

    tick_loc = np.arange(0, nframes, 120)
    #plt.xticks(tick_loc, xticks, rotation=45)
    xticks = np.arange(ts, ts + nframes * 10, 1200)
    new_xticks = [datetime.utcfromtimestamp(num).strftime("%H:%M:%S") for num in xticks]
    
    plt.xticks(tick_loc, new_xticks)
    plt.xlabel("Time (UTC)")
    plt.title(
        "Freq ID: %d, Freq: %.3fMHz, \nDate: %s"
        % (freq_id, freq_start, date)
    )

    cbar = plt.colorbar()
    cbar.set_label("log(Power)")
    plt.gcf().subplots_adjust(bottom=0.20)
    plt.show()
    #file_name = "hfb_data_sky_map_ns_beams_freq_" + str(freq_id) + ".pdf"
    #plt.savefig(file_name)


def plot_one_beam_freq_over_time(metadata, data, nframes, nsubfreq, freq_id, beam_id):

    freq_start = 800.0 - float(freq_id) * 400.0 / 1024.0

    hfb_square = np.zeros((nframes, nsubfreq), dtype=np.float32)

    # Get the time in UTC
    ts = int(metadata["ctime"][0][freq_id][0])
    date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    # Filter specific beam from data
    ctr = 0
    for d in data[:nframes]:
        hfb_square[ctr] = d[freq_id][beam_id * nsubfreq : (beam_id + 1) * nsubfreq]
        ctr = ctr + 1

    # Plot data
    plt.imshow(np.log(hfb_square), interpolation="none")
    plt.xlabel("Frequency MHz")
    plt.xticks(np.arange(7))
    xticks = [
        round(num, 3)
        for num in np.arange(freq_start, freq_start + 0.39, 20 * 0.39 / 128.0)
    ]
    tick_loc = np.arange(0, nsubfreq, 20)
    plt.xticks(tick_loc, xticks, rotation=45)

    yticks = np.arange(ts, ts + nframes * 10, 200)
    new_yticks = [datetime.utcfromtimestamp(num).strftime("%H:%M:%S") for num in yticks]
    plt.yticks(tick_loc, new_yticks)
    plt.ylabel("Time (UTC)")
    plt.title(
        "Freq ID: %d, Freq range: %.3f - %.3fMHz, \nBeam: %d, Date: %s"
        % (freq_id, freq_start, freq_start + 0.39, beam_id, date)
    )

    cbar = plt.colorbar()
    cbar.set_label("log(Power)")
    plt.gcf().subplots_adjust(bottom=0.20)
    plt.show()
    #file_name = "hfb_data_freq_" + str(freq_id) + "_beam_" + str(beam_id) + ".pdf"
    #plt.savefig(file_name)


if __name__ == "__main__":
    main()
