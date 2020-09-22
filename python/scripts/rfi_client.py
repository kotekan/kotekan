"""
/*********************************************************************************
* RFI Documentation Header Block
* File: rfi_client.py
* Purpose: A client to receive and display real-time rfi data from rfi_receiver.py
* Python Version: 3.6 
* Dependencies: Matplotlib, yaml, numpy, argparse
* Help: Run "python3 rfi_client.py" -H (or --Help) for how to use.
*********************************************************************************/
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from future import standard_library

standard_library.install_aliases()
import threading
import socket
import numpy as np
import matplotlib.animation as animation
from matplotlib.widgets import Button
import datetime
import matplotlib.pyplot as plt
import time
import matplotlib.dates as mdates
import os
import argparse
import yaml


def parse_dict(cmd, _dict):
    for key, value in _dict.items():
        if type(value) == dict:
            parse_dict(cmd, value)
        else:
            if key in cmd.config:
                if type(cmd.config[key]) == type(value):
                    print("Setting Config Paramter %s to %s" % (key, str(value)))
                    cmd.config[key] = value


class CommandLine(object):
    def __init__(self):

        # Defaults
        self.TCP_IP = "127.0.0.1"
        self.TCP_PORT = 2901
        self.config = {
            "samples_per_data_set": 32768,
            "timestep": 2.56e-6,
            "waterfallX": 1024,
            "num_elements": 2048,
            "waterfallY": 1024,
            "waterfall_request_delay": 10,
            "colorscale": 1.0,
            "num_global_freq": 1024,
            "sk_step": 256,
            "bi_frames_per_packet": 10,
        }
        self.mode = "pathfinder"
        self.supportedModes = ["vdif", "pathfinder", "chime", "badinput"]
        parser = argparse.ArgumentParser(description="RFI Receiver Script")
        parser.add_argument(
            "-H", "--Help", help="Example: Help argument", required=False, default=""
        )
        parser.add_argument(
            "-r",
            "--receive",
            help="Example: 127.0.0.1:2900",
            required=False,
            default="",
        )
        parser.add_argument(
            "-c",
            "--config",
            help="Example: ../kotekan/kotekan_opencl_rfi.yaml",
            required=False,
            default="",
        )
        parser.add_argument(
            "-m", "--mode", help="Example: vdif, pathfinder", required=False, default=""
        )
        argument = parser.parse_args()
        status = False

        if argument.Help:
            print(
                "You have used '-H' or '--Help' with argument: {0}".format(
                    argument.Help
                )
            )
            status = True
        if argument.receive:
            print(
                "You have used '-r' or '--receive' with argument: {0}".format(
                    argument.receive
                )
            )
            self.TCP_IP = argument.receive[: argument.receive.index(":")]
            self.TCP_PORT = int(argument.receive[argument.receive.index(":") + 1 :])
            print("Setting TCP IP: %s PORT: %d" % (self.TCP_IP, self.TCP_PORT))
            status = True
        if argument.config:
            print(
                "You have used '-c' or '--config' with argument: {0}".format(
                    argument.config
                )
            )
            parse_dict(self, yaml.load(open(argument.config)))
            print(self.config)
            status = True
        if argument.mode:
            print(
                "You have used '-m' or '--mode' with argument: {0}".format(
                    argument.mode
                )
            )
            if argument.mode in self.supportedModes:
                self.mode = argument.mode
                print("Setting mode to %s mode." % (argument.mode))
            else:
                print("This mode in currently not supported, reverting to default")
                print("Supported Modes Include:")
                for mode in self.supportedModes:
                    print("- ", mode)
            status = True
        if not status:
            print("Maybe you want to use -H or -s or -p or -p as arguments ?")


def init():
    im.set_data(waterfall)
    if app.mode == "badinput":
        med_plot.set_ydata(
            100.0
            * np.nanmedian(waterfall, axis=0)
            / float(app.config["bi_frames_per_packet"])
        )
        med_plot_input.set_xdata(np.nanmean(waterfall, axis=1))
    else:
        med_plot.set_xdata(np.linspace(x_lims[0], x_lims[1], num=waterfall.shape[1]))
        med_plot.set_ydata(np.nanmedian(waterfall, axis=0))
        med_plot_input.set_xdata(np.nanmedian(waterfall, axis=1))


def animate(i):
    im.set_data(waterfall)
    if app.mode != "badinput":
        x_lims = mdates.date2num(
            [
                t_min,
                t_min
                + datetime.timedelta(
                    seconds=waterfall.shape[1]
                    * app.config["samples_per_data_set"]
                    * app.config["timestep"]
                ),
            ]
        )
        im.set_extent([x_lims[0], x_lims[1], 400, 800])
        med_plot.set_xdata(np.linspace(x_lims[0], x_lims[1], num=waterfall.shape[1]))
        med_plot.set_ydata(np.nanmedian(waterfall, axis=0))
        ax[1, 0].set_xlim([x_lims[0], x_lims[1]])
        med_plot_input.set_xdata(np.nanmedian(waterfall, axis=1))
    else:
        med_plot.set_ydata(
            100.0
            * np.nanmedian(waterfall, axis=0)
            / float(app.config["bi_frames_per_packet"])
        )
        med_plot_input.set_xdata(np.nanmean(waterfall, axis=1))
    return im


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def data_listener():

    global sock_tcp, waterfall, addr, t_min, app

    WATERFALLMESSAGE = "W"
    TIMEMESSAGE = "T"

    timesize = len(t_min.strftime("%d-%m-%YT%H:%M:%S:%f"))
    waterfallsize = 8 * waterfall.size  # Bytes
    delay = app.config["waterfall_request_delay"]

    if app.mode == "badinput":
        WATERFALLMESSAGE = "w"
        TIMEMESSAGE = "t"

    while True:

        sock_tcp.send(WATERFALLMESSAGE.encode())
        print("Trying to receive waterfall of size: %d" % (waterfallsize))
        data = recvall(sock_tcp, waterfallsize)
        if data is None:
            print("Connection to %s:%s Broken... Exiting" % (addr[0], str(addr[1])))
            break

        waterfall = np.fromstring(data).reshape(waterfall.shape)
        waterfall[waterfall == -1] = np.nan
        if app.mode == "badinput":
            print(
                np.where(
                    100.0
                    * np.nanmedian(waterfall, axis=0)
                    / float(app.config["bi_frames_per_packet"])
                    > 10
                )[0]
            )
            print(
                np.where(
                    100.0
                    * np.nanmedian(waterfall, axis=0)
                    / float(app.config["bi_frames_per_packet"])
                    > 10
                )[0].size
            )

        sock_tcp.send(TIMEMESSAGE.encode())

        data = recvall(sock_tcp, timesize).decode()

        if data is None:
            print("Connection to %s:%s Broken... Exiting" % (addr[0], str(addr[1])))
            break

        t_min = datetime.datetime.strptime(data, "%d-%m-%YT%H:%M:%S:%f")
        print(t_min)

        time.sleep(delay)


class Callback(object):
    def SaveData(self, event):
        if not os.path.exists("RFIData"):
            os.makedirs("RFIData")
        newDir = "RFIData/" + datetime.datetime.utcnow().strftime(
            "%Y%m%dT%H:%M:%S"
        )  # Create New Folder
        os.makedirs(newDir)
        np.save(newDir + "/data.npy", waterfall)
        plt.savefig(newDir + "/image.png")
        f = open(newDir + "/info.txt", "w")  # Create Info Text File
        f.write("X Lims: %f - %f\n" % (x_lims[0], x_lims[1]))
        f.close()
        print("Saved Data! Path: ./" + newDir)


def to_confidence(app, ticks):
    M = app.config["sk_step"] * app.config["num_elements"]
    sigma = np.sqrt(4 / M)
    return np.round(np.abs(1 - ticks) / (sigma), 1)


if __name__ == "__main__":

    app = CommandLine()

    plt.ion()

    # Initialize Plot
    nx, ny = app.config["waterfallY"], app.config["waterfallX"]
    t_min = datetime.datetime.utcnow()
    if app.mode == "badinput":
        waterfall = -1 * np.ones(
            [app.config["num_global_freq"], app.config["num_elements"]]
        )
    else:
        waterfall = -1 * np.ones([nx, ny])

    #    fig = plt.figure()
    fig, ax = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [4, 1], "width_ratios": [4, 1]},
    )
    if app.mode == "badinput":
        x_lims = [0, app.config["num_elements"]]
        im = ax[0, 0].imshow(
            waterfall,
            aspect="auto",
            cmap="viridis",
            extent=[x_lims[0], x_lims[1], 400, 800],
            vmin=0,
            vmax=app.config["bi_frames_per_packet"],
        )
        cbar_ax = fig.add_axes([0.915, 0.32, 0.03, 0.58])
        cbar = fig.colorbar(im, cax=cbar_ax, label="Median Faulty Frames")
        (med_plot,) = ax[1, 0].plot(
            np.arange(waterfall.shape[1]), np.nanmedian(waterfall, axis=0)
        )
        (med_plot_input,) = ax[0, 1].plot(
            np.nanmean(waterfall, axis=1),
            800 - 400.0 / 1024.0 * np.arange(waterfall.shape[0]),
            "o",
        )
        ax[1, 0].set_xlabel("Input")
        ax[1, 0].set_xlim([0, waterfall.shape[1]])
        ax[1, 0].set_ylim([0, 100])
        ax[1, 0].set_ylabel("Likelyhood of Faultiness")
        ax[0, 1].set_xlabel("Median Across Inputs")
        ax[0, 1].set_xlim([0, 10])
    else:
        x_lims = mdates.date2num(
            [
                t_min,
                t_min
                + datetime.timedelta(
                    seconds=waterfall.shape[1]
                    * app.config["samples_per_data_set"]
                    * app.config["timestep"]
                ),
            ]
        )
        im = ax[0, 0].imshow(
            waterfall,
            aspect="auto",
            cmap="viridis",
            extent=[x_lims[0], x_lims[1], 400, 800],
            vmin=1 - app.config["colorscale"],
            vmax=1 + app.config["colorscale"],
        )
        ticks = np.linspace(
            1 - app.config["colorscale"], 1 + app.config["colorscale"], num=10
        )
        cbar_ax = fig.add_axes([0.913, 0.32, 0.03, 0.58])
        cbar = fig.colorbar(im, cax=cbar_ax, label="SK Value", ticks=ticks)
        # cbar = fig.colorbar(im, cax=cbar_ax, label = "Detection Confidence", ticks = ticks)
        # cbar.ax.set_yticklabels(to_confidence(app, ticks))
        ax[0, 0].xaxis_date()
        date_format = mdates.DateFormatter("%H:%M:%S")
        ax[0, 0].xaxis.set_major_formatter(date_format)
        ax[1, 0].xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        (med_plot,) = ax[1, 0].plot(
            np.linspace(x_lims[0], x_lims[1], num=waterfall.shape[1]),
            np.nanmedian(waterfall, axis=0),
        )
        (med_plot_input,) = ax[0, 1].plot(
            np.nanmean(waterfall, axis=1),
            800 - 400.0 / 1024.0 * np.arange(waterfall.shape[0]),
        )
        ax[1, 0].set_xlabel("Time")
        ax[1, 0].set_xlim([x_lims[0], x_lims[1]])
        ax[1, 0].set_ylim(
            [1 - app.config["colorscale"] / 2.0, 1 + app.config["colorscale"] / 2.0]
        )
        ax[1, 0].set_ylabel("Median SK Value")
        ax[0, 1].set_xlabel("Median SK Value")
        ax[0, 1].set_xlim([1 - app.config["colorscale"], 1 + app.config["colorscale"]])

    ax[0, 0].set_ylabel("Frequency[MHz]")
    ax[0, 0].set_title("RFI Viewer (Mode: " + app.mode + ")")
    ax[0, 1].set_ylim([400, 800])
    ax[1, 1].axis("off")

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=waterfall.size, interval=50
    )

    np.warnings.filterwarnings("ignore")

    time.sleep(1)

    # Intialize TCP
    TCP_IP = app.TCP_IP
    TCP_PORT = app.TCP_PORT
    addr = (TCP_IP, TCP_PORT)
    sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Trying to connect to %s:%s" % (addr[0], addr[1]))
    Connected = False

    while not Connected:
        try:
            sock_tcp.connect(addr)
            Connected = True
        except Exception:
            print(
                "Could not connect to %s:%s Trying again in 5 seconds"
                % (addr[0], addr[1])
            )
            time.sleep(5)

    thread = threading.Thread(target=data_listener)
    thread.daemon = True
    thread.start()

    callback = Callback()
    buttonLocation = plt.axes([0.75, 0.1, 0.2, 0.15])
    saveButton = Button(buttonLocation, "Save")
    saveButton.on_clicked(callback.SaveData)

    eval(input())
