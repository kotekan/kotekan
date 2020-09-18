import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import hfbbuffer

def main():

    in_file = sys.argv[1]

    print("Reading file: {}".format(in_file))

    freq_id = int(sys.argv[2])
    beam_id = int(sys.argv[3])
    freq_start = 800.0 - float(freq_id) * 400.0 / 1024.0
    nframes = 128

    f = hfbbuffer.HFBRaw.from_file(in_file) 

    metadata = f.metadata
    data = f.data['hfb']

    print("No. of valid frames in the file: %d" % np.sum(f.valid_frames))

    print("Metadata")
    print("--------")
    print(metadata.dtype)
    print(metadata[0][freq_id])

    vis_square = np.zeros((nframes, 128), dtype=np.float32)

    # Get the time in UTC
    ts = int(metadata["ctime"][0][freq_id][0])
    date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    # Filter specific beam from data
    ctr = 0
    for d in data[:nframes]:
        vis_square[ctr] = d[freq_id][beam_id * 128 : (beam_id + 1) * 128]
        ctr = ctr + 1

    # Plot data
    plt.imshow(np.log(vis_square), interpolation="none")
    plt.xlabel("Frequency MHz")
    plt.xticks(np.arange(7))
    xticks = [
        round(num, 3)
        for num in np.arange(freq_start, freq_start + 0.39, 20 * 0.39 / 128.0)
    ]
    tick_loc = np.arange(0, 128, 20)
    plt.xticks(tick_loc, xticks, rotation=45)

    yticks = np.arange(ts, ts + 128 * 10, 200)
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
    #plt.show()
    file_name = "hfb_data_freq_" + str(freq_id) + "_beam_" + str(beam_id) + ".pdf"
    plt.savefig(file_name)


if __name__ == "__main__":
    main()
