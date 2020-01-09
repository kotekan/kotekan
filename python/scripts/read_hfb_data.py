import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Metadata format
metadata_t = np.dtype(
    [
        ("metadata_size", np.uint32),
        ("fpga_seq_num", np.int64),
        ("gps_time", [("s", np.int64), ("ns", np.int64)]),
        ("gps_time_flag", np.uint32),
        ("freq_bin_num", np.uint32),
        ("norm_frac", np.float32),
        ("num_samples_integrated", np.uint32),
        ("num_samples_expected", np.uint32),
        ("compressed_data_size", np.uint32),
    ]
)


def main():

    in_file = sys.argv[1]

    print("Reading file: {}".format(in_file))

    freq_bin_num = int(sys.argv[2])
    beam_num = int(sys.argv[3])
    freq_start = 800.0 - float(freq_bin_num) * 400.0 / 1024.0

    dt = np.dtype([("metadata", metadata_t), ("data", np.float32, (1024 * 128,))])

    data = np.fromfile(in_file, dtype=dt)

    print("No. of frames in the file: %d" % len(data["metadata"]))

    print("Metadata")
    print("--------")
    print(metadata_t.names)
    print(data["metadata"][0:5])

    vis_square = np.zeros((128, 128), dtype=np.float32)

    # Get the time in UTC
    ts = int(data["metadata"]["gps_time"][0]["s"])
    date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    # Find specific frequency bin in data and filter
    index = data["metadata"]["freq_bin_num"] == freq_bin_num
    freq = data["data"][index]

    # Filter specific beam from data
    ctr = 0
    for i in range(0, len(freq)):
        vis_square[ctr] = freq[i][beam_num * 128 : (beam_num + 1) * 128]
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
        "Freq bin: %d, Freq range: %.3f - %.3fMHz, \nBeam: %d, Date: %s"
        % (freq_bin_num, freq_start, freq_start + 0.39, beam_num, date)
    )

    cbar = plt.colorbar()
    cbar.set_label("log(Power)")
    plt.gcf().subplots_adjust(bottom=0.20)
    # plt.show()
    file_name = "hfb_data_freq_" + str(freq_bin_num) + "_beam_" + str(beam_num) + ".pdf"
    plt.savefig(file_name)


if __name__ == "__main__":
    main()
