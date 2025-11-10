#!/opt/anaconda/bin/python

# Imports

# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import sys
import matplotlib

matplotlib.use("Agg")
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

rcParams["figure.figsize"] = (10, 10)

# Load data

data_format = np.dtype([("vis", [("real", np.int32), ("imag", np.int32)], 32896)])
data = np.fromfile(sys.argv[1], dtype=data_format)

vis_real = empty((len(data), 256, 256))
vis_real[:] = NAN
vis_imag = empty((len(data), 256, 256))
vis_imag[:] = NAN
vis_mag = empty((len(data), 256, 256))
vis_mag[:] = NAN

ut_indices = triu_indices(256)

for i in range(0, len(data)):
    vis_real[i][ut_indices] = data[i]["vis"]["real"]

for i in range(0, len(data)):
    vis_imag[i][ut_indices] = data[i]["vis"]["imag"]

vis_mag = sqrt(vis_real**2 + vis_imag**2)

# Generate graphs

centers = [800 - i * 400.0 / 1024.0 for i in range(1024)]


def bin_number(node, link, index):
    return (node - 1) + 16 * link + 128 * index


def frequency(bin_num):
    return centers[bin_num]


pp = PdfPages("visibilities.pdf")
cmap = cm.jet
cmap.set_bad(color="w")

for i in range(0, 64):
    fig = figure()
    ax = fig.add_subplot(111)
    ma = ax.matshow(vis_real[i], cmap=cmap, aspect="equal", interpolation="none")
    bin_num = bin_number(16, i // 8, i % 8)
    title(
        "Real visibility matrix for slot 16, link "
        + str(i // 8)
        + ", index "
        + str(i % 8)
        + ", frequency "
        + "{:4.1f}".format(frequency(bin_num))
        + "MHz, bin_num: "
        + str(bin_num)
    )
    xlabel("Element Number")
    ylabel("Element Number")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar(ma, cax=cax)
    pp.savefig(fig)
    close(fig)

    fig = figure()
    ax = fig.add_subplot(111)
    ma = ax.matshow(vis_imag[i], cmap=cmap, aspect="equal", interpolation="none")
    bin_num = bin_number(16, i // 8, i % 8)
    title(
        "Imaginary visibility matrix for slot 16, link "
        + str(i // 8)
        + ", index "
        + str(i % 8)
        + ", frequency "
        + "{:4.1f}".format(frequency(bin_num))
        + "MHz, bin_num: "
        + str(bin_num)
    )
    xlabel("Element Number")
    ylabel("Element Number")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar(ma, cax=cax)
    pp.savefig(fig)
    close(fig)

pp.close()
