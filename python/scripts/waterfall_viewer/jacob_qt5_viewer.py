"""
Welcome to the ARO Live-Viewer!
Documentation By: Jacob Taylor
Developers: Keith Vanderlinde, Jacob Taylor
Date: August 2017
Compatibility: Python 2.7 & Python 3.5
Dependencies: Astropy, PyQt5, Matplotlib, Numpy
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility


# Imports
from future import standard_library

standard_library.install_aliases()
import threading
import socket
import sys
import numpy as np
import matplotlib.animation as animation
import matplotlib.dates as md
import datetime
import struct
import json
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import random
import time
import cmath
import math
import subprocess
from astropy.coordinates import SkyCoord, EarthLocation, Angle, AltAz
from astropy import units as u
from astropy.time import Time

"""
Settings Menu Class
Widgets:
    - timer: A Timer which triggers the graph to update when the timer runs out 
    - mode: Toggles Median Subtraction
    - dispersed_button: Toggles dispersion correction
    - pul_label: Label Reading "Pulsar: "
    - pls_text: Text box for editing pulsar information
    - dm_text: The Dispersion Measure of the current pulsar
    - fold_text: Label Reading "Folding Period: "
    - fold_edit: Textbox for editing the Folding period of the pulsar
    - PulseProfileRadio: Radio Button for toggling what the Graph displays, Pulse Profile
    - GainProfileRadio: Radio Button for toggling what the Graph displays, Gain Profile
    - CalibrationRadio: Radio Button for toggling what the Graph displays, Focus Calibration 
    - GraphSlider: Slider used to change Graph's Y axis
    - settingscanvas: The Matplotlib Graph
    - Brightness_Text: Label Reading "Current Frequency For Calibration (MHz): "
    - Brightness_Edit: Text box to edit the brightness
    - WriteButton: Button that write the current graph to disk
Methods:
    WriteGraphToDisk - Function that triggers the data on the graph to be written
    ChangeGraphState - Function to change what the graph is showing
    ChangeDispersion - Toggle dispersion correction
    UpdateSettings - Update the pulsar information
    UpdateFreq - Update the frequency being looked at during calibration
    UpdateFold - Update folding period
    UpdateGraph - Update the graph
"""


class Settings(QDialog):
    def __init__(self, parent):
        super(Settings, self).__init__(parent)
        self.main = parent  # define the main window
        # Initiate Variables
        self.freq_upper = float(self.main.freqlist[0, 0])
        self.freq_lower = float(self.main.freqlist[-1, -1])
        self.total_bandwidth = self.freq_upper - self.freq_lower

        self.Bright_Freq_index = self.main.waterfall.shape[1] // 2
        self.GraphState = 0
        self.Bright_Freq = self.freq_upper - self.Bright_Freq_index * (
            self.freq_lower / self.main.waterfall.shape[1]
        )
        self.BrightnessMeasures = []
        self.BrightnessTimes = []
        self.WriteTrigger = False
        self.settingsfigure = plt.figure(2)
        self.settingscanvas = FigureCanvas(self.settingsfigure)

        # Intialize Widgets and Add them to screen
        self.layout = QVBoxLayout()  # Main layout, Vertical

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.UpdateGraph)
        self.timer.start(1000)
        self.kotekanTimer = QTimer()
        self.kotekanTimer.timeout.connect(self.CheckKotekanStatus)
        self.kotekanTimer.start(5000)
        # CheckBoxes
        hbox = QHBoxLayout()
        self.mode = QCheckBox("Median Subtract")
        self.mode.setChecked(self.main.medsub)
        self.mode.stateChanged.connect(parent.ChangeMode)
        hbox.addWidget(self.mode)
        self.dispersed_button = QCheckBox("Dispersion Correction")
        self.dispersed_button.setChecked(self.main.DeDisperse)
        self.dispersed_button.stateChanged.connect(self.ChangeDispersion)
        hbox.addWidget(self.dispersed_button)
        self.centerPulse = QCheckBox("Center Pulse")
        self.centerPulse.setChecked(self.main.centerPulsar)
        self.centerPulse.stateChanged.connect(self.TogglePulseCenter)
        hbox.addWidget(self.centerPulse)
        self.kotekanStatusText = QLabel("Kotekan Status Indicator:")
        hbox.addWidget(self.kotekanStatusText)
        self.col = QColor(255, 0, 0)
        self.kotekanStatus = QFrame(self)
        self.kotekanStatus.setFixedWidth(20)
        self.kotekanStatus.setFixedHeight(20)
        self.kotekanStatus.setStyleSheet(
            "QWidget { background-color: %s }" % self.col.name()
        )
        hbox.addWidget(self.kotekanStatus)
        self.layout.addLayout(hbox)

        # Pulsar Text/Edit
        hbox = QHBoxLayout()
        self.pul_label = QLabel("Pulsar: ")
        hbox.addWidget(self.pul_label)
        self.pls_text = QLineEdit(self.main.target)
        self.pls_text.textChanged[str].connect(self.UpdateSettings)
        hbox.addWidget(self.pls_text)
        self.layout.addLayout(hbox)

        # Pulsar Info
        hbox = QHBoxLayout()
        self.dm_text = QLabel(
            "Dispersion Measure: " + str(self.main.psrdata["dmeasure"])
        )
        hbox.addWidget(self.dm_text)
        self.fold_text = QLabel("Folding Period: ")  # + str(self.main.fold_period))
        hbox.addWidget(self.fold_text)
        self.fold_edit = QLineEdit(str(self.main.fold_period))
        self.fold_edit.textChanged[str].connect(self.UpdateFold)
        hbox.addWidget(self.fold_edit)
        self.layout.addLayout(hbox)

        # Mooncake Info
        hbox = QHBoxLayout()
        self.MoonIP_Text = QLabel("Pointing Server IP: ")
        hbox.addWidget(self.MoonIP_Text)
        self.MoonIP_Edit = QLineEdit(str(self.main.mooncakeIP))
        self.MoonIP_Edit.textChanged[str].connect(self.UpdateMoonIP)
        hbox.addWidget(self.MoonIP_Edit)
        self.MoonPort_Text = QLabel("Port: ")
        hbox.addWidget(self.MoonPort_Text)
        self.MoonPort_Edit = QLineEdit(str(self.main.mooncakePort))
        self.MoonPort_Edit.textChanged[str].connect(self.UpdateMoonPort)
        hbox.addWidget(self.MoonPort_Edit)
        self.layout.addLayout(hbox)

        # Graph Modes
        self.ButtonModeList = []
        hbox = QHBoxLayout()
        self.PulseProfileRadio = QRadioButton("Pulse Profile")
        self.PulseProfileRadio.setChecked(True)
        self.ButtonModeList.append(self.PulseProfileRadio)
        self.PulseProfileRadio.toggled.connect(self.ChangeGraphState)
        self.GainProfileRadio = QRadioButton("Gain Profile")
        self.GainProfileRadio.toggled.connect(self.ChangeGraphState)
        self.ButtonModeList.append(self.GainProfileRadio)
        self.CalibrationRadio = QRadioButton("Calibration Mode")
        self.CalibrationRadio.toggled.connect(self.ChangeGraphState)
        self.ButtonModeList.append(self.CalibrationRadio)
        hbox.addWidget(self.PulseProfileRadio)
        hbox.addWidget(self.GainProfileRadio)
        hbox.addWidget(self.CalibrationRadio)
        self.layout.addLayout(hbox)

        # Graph and Slider
        hbox = QHBoxLayout()
        self.GraphSlider = QSlider(Qt.Vertical)
        self.GraphSlider.setMinimum(1)
        self.GraphSlider.setMaximum(10)
        self.GraphSlider.setValue(3)
        self.GraphSlider.setFocusPolicy(Qt.StrongFocus)
        self.GraphSlider.setTickPosition(QSlider.TicksBothSides)
        self.GraphSlider.setTickInterval(1)
        self.GraphSlider.valueChanged.connect(self.UpdateGraph)
        hbox.addWidget(self.GraphSlider)
        hbox.addWidget(self.settingscanvas)
        self.layout.addLayout(hbox)

        # Calibration Frequency
        hbox = QHBoxLayout()
        self.Brightness_Text = QLabel("Center Frequency For Calibration (MHz): ")
        hbox.addWidget(self.Brightness_Text)
        self.Brightness_Edit = QLineEdit(str(self.Bright_Freq))
        self.Brightness_Edit.textChanged[str].connect(self.UpdateFreq)
        hbox.addWidget(self.Brightness_Edit)
        self.Width_Text = QLabel("Calibration Bandwidth (MHz):")
        hbox.addWidget(self.Width_Text)
        self.Width_index = 200
        self.Width_Edit = QLineEdit(str(self.Width_index))
        self.Width_Edit.textChanged[str].connect(self.UpdateWidth)
        hbox.addWidget(self.Width_Edit)
        self.layout.addLayout(hbox)

        # Write Button
        self.WriteButton = QPushButton("Write To Disk")
        self.WriteButton.clicked.connect(self.WriteGraphToDisk)
        self.layout.addWidget(self.WriteButton)

        self.setLayout(self.layout)  # Add layout
        self.UpdateGraph()  # Create/Intialize Graph
        self.CheckKotekanStatus()

    def CheckKotekanStatus(self):
        tunnelCommand = "ssh %s@%s" % (
            self.main.main_startup.Receive_User,
            self.main.main_startup.Receive_Ip,
        )
        kotekanStatusCommand = """ 'ps -ef | grep kotekan' """
        process = subprocess.Popen(
            tunnelCommand + kotekanStatusCommand,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        output, error = process.communicate()
        if "./kotekan" in output.decode("utf-8"):
            self.col.setRed(0)
            self.col.setGreen(255)
            self.kotekanStatus.setStyleSheet(
                "QFrame { background-color: %s }" % self.col.name()
            )
        else:
            self.col.setRed(255)
            self.col.setGreen(0)
            self.kotekanStatus.setStyleSheet(
                "QFrame { background-color: %s }" % self.col.name()
            )

    def WriteGraphToDisk(self):
        self.WriteTrigger = True  # Toggle Trigger for Update Graph

    def ChangeGraphState(self, state):
        if state:  # When a Radio Button is checked
            states = [button.isChecked() for button in self.ButtonModeList]
            for i in range(len(states)):  # Find which one was checked
                if states[i]:  # Update
                    self.GraphState = i
                    if i == 2:
                        self.BrightnessMeasures = []
                        self.BrightnessTimes = []
                        self.t0 = datetime.datetime.fromtimestamp(
                            np.amin(self.main.times)
                        )
                    break

    def ChangeDispersion(self):
        self.main.DeDisperse = not self.main.DeDisperse  # Toggler Dispersion Flag

    def TogglePulseCenter(self):
        self.main.centerPulsar = not self.main.centerPulsar  # Toggler Pulsar Centering

    def UpdateSettings(self, text):
        try:
            x = self.main.psrcat[text]  # Try to find the pulsar

        except:
            # If you cant find it Update the label on the Main screen
            self.main.temp_target = text
            self.main.UpdateLabel()
            return
        # If you can find it, Update
        self.main.target = text
        self.main.psrdata = self.main.psrcat[self.main.target]
        try:
            self.main.fold_period = 1.0 / self.main.psrdata["frequency"]
        except:
            self.main.fold_period = self.main.psrdata["period"]
        self.dm_text.setText(
            "Dispersion Measure: " + str(self.main.psrdata["dmeasure"])
        )
        self.fold_edit.setText(str(self.main.fold_period))
        self.main.temp_target = ""
        self.main.UpdateLabel()

    def UpdateMoonIP(self, text):
        try:
            self.main.mooncakeIP = text
            return
        except:
            return

    def UpdateMoonPort(self, text):
        try:
            self.main.mooncakePort = text
            return
        except:
            return

    def UpdateFreq(self, text):
        try:  # Is it a valid entry?
            x = float(text)
            if x > self.freq_upper or x < self.freq_lower:
                return

        except:
            return
        # If so, Update
        self.Bright_Freq_index = np.round(
            (self.freq_upper - float(text)) * (self.main.waterfall.shape[1] / 400.0)
        )
        self.Bright_Freq = float(text)
        if self.Bright_Freq_index < 0:
            self.Bright_Freq_index = 0
        elif self.Bright_Freq_index >= self.main.waterfall.shape[1]:
            self.Bright_Freq_index = self.main.waterfall.shape[1] - 1
        self.BrightnessMeasures = []
        self.BrightnessTimes = []
        self.t0 = datetime.datetime.fromtimestamp(np.amin(self.main.times))

    def UpdateWidth(self, text):
        print(text)
        try:  # Is it a valid entry?
            print("TESTING VALIDITY")
            x = float(text)
            if (
                x + self.Bright_Freq > self.freq_upper
                or self.Bright_Freq - x < self.freq_lower
            ):
                print(x, self.Bright_Freq, x + self.Bright_Freq, x - self.Bright_Freq)
                return
        except:
            print("NOT VALID")
            return
        print("VALID")
        # If so, Update
        freqBinSize = 400.0 / self.main.waterfall.shape[1]
        self.Width_index = int(float(text) / freqBinSize)
        print(freqBinSize, self.Width_index)
        if self.Width_index < 0:
            self.Width_index = 0
        elif self.Bright_Freq_index + self.Width_index >= self.main.waterfall.shape[1]:
            self.Width_index = self.main.waterfall.shape[1] - 1 - self.Bright_Freq_index
        self.BrightnessMeasures = []
        self.BrightnessTimes = []
        self.t0 = datetime.datetime.fromtimestamp(np.amin(self.main.times))

    def UpdateFold(self, text):
        try:  # Is it a Valid entry?
            self.main.fold_period = float(text)
            self.main.UpdateLabel()  # Update

        except:
            return

    def UpdateGraph(self):

        self.settingsfigure.clear()  # Remove old graph
        if (self.WriteTrigger) and (not os.path.exists("ViewerGraphsAndData")):
            os.makedirs(
                "ViewerGraphsAndData"
            )  # Insure Directory is made for Graph data
            print("Directory Made")

        if self.GraphState == 0:  # Pulse Profile
            x = np.arange(0, 1, 1.0 / self.main.dedispersed.shape[0])  # phase, x axis
            y_lower = -0.03  # -1*(self.GraphSlider.value()/100.0) #Set Y scale
            y_upper = 2 * self.GraphSlider.value() / 100.0
            plt.ylim([y_lower, y_upper])
            plt.title("Pulse Profile")  # Labels
            plt.xlabel("Phase")
            plt.ylabel("Power (arb)")
            y = np.zeros_like(self.main.MedSubbed[:, 0])
            for i in range(self.main.MedSubbed.shape[0]):  # Compute Pulse
                y[y.size - 1 - i] = np.mean(
                    self.main.MedSubbed[i, :][
                        (self.main.MedSubbed[i, :] > (5 * y_lower))
                        * (self.main.MedSubbed[i, :] < (5 * y_upper))
                    ]
                )
            # y /= self.main.MedSubbed.shape[1]
            p10 = np.poly1d(np.polyfit(x, y, 10))  # Create Line of best fit
            plt.plot(x, y, ".", x, p10(x))  # plot
        elif self.GraphState == 1:  # Gain Profile
            x = np.linspace(
                self.freq_upper, self.freq_lower, num=self.main.waterfall.shape[1]
            )  # X axis, frequency
            y = np.array(
                [
                    np.median(
                        self.main.waterfall[:, self.main.waterfall.shape[1] - t - 1, 0]
                    )
                    for t in range(self.main.waterfall.shape[1])
                ]
            )  # Compute Median Values for each freq
            y_lower = (
                np.round(np.median(y)) - 2 * self.GraphSlider.value()
            )  # Y axis scale
            y_upper = np.round(np.median(y)) + 2 * self.GraphSlider.value()
            plt.ylim([y_lower, y_upper])
            plt.title("Power vs Frequency")  # Labels
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Power (arb, dB)")
            plt.plot(x, y[::-1])  # Plot
        else:  # Calibration Mode
            y = self.main.waterfall[
                :,
                int(self.Bright_Freq_index)
                - self.Width_index : int(self.Bright_Freq_index)
                + self.Width_index,
                0,
            ]
            self.BrightnessMeasures.append(np.median(y))  # Make measurement
            self.BrightnessTimes.append(
                (
                    datetime.datetime.fromtimestamp(np.amin(self.main.times)) - self.t0
                ).total_seconds()
            )  # Record Time
            x = np.array(
                self.BrightnessTimes
            )  # np.arange(len(self.BrightnessMeasures))
            y = np.array(self.BrightnessMeasures)
            plt.title("Power vs Time (single freq)")  # Labels
            plt.xlabel("Time (s)")
            plt.ylabel("Power (arb, dB)")
            y_lower = np.round(np.median(y)) - 2 * self.GraphSlider.value()  # Y Scale
            y_upper = np.round(np.median(y)) + 2 * self.GraphSlider.value()
            plt.ylim([y_lower, y_upper])
            plt.xlim([0, int(x[-1]) + 1])  # X scale, dynamic
            plt.plot(x, y, ".")  # Plot

        if self.WriteTrigger:  # Handles Writting data to file
            FolderName = "ViewerGraphsAndData/" + datetime.datetime.utcnow().strftime(
                "%Y%m%dT%H:%M:%S"
            )  # Create New Folder
            os.makedirs(FolderName)
            f = open(FolderName + "/info.txt", "w")  # Create Info Text File
            f.write(
                "Current Graph State: %d\nTime of Write: %s\nPlease Find X and Y Axis Data in Current Folder\n\n"
                % (self.GraphState, FolderName[20:])
            )
            if self.GraphState == 2:
                f.write(
                    "Reference Time for Calibration: t0 = %s\n\n"
                    % (self.t0.strftime("%Y%m%dT%H:%M:%S"))
                )
            f.write(
                "Graph State Legend\n0 - Pulsar Profile\n1 - Gain\n2 - Calibration\n\nEnjoy!\n"
            )
            f.close()
            np.save(FolderName + "/x", x)  # Save x and y axes
            np.save(FolderName + "/y", y)
            self.WriteTrigger = False

        self.settingscanvas.draw()  # Draw new graph


"""
Main Window Class
Widgets:
    - pos_text_1: Pointing textbox
    - pos_text_2: Pointing textbox
    - info_text_1: Pulsar Info Textbox
    - info_text_2: Pulsar Info Textbox
    - canvas: The main matplotlib Graph widget
    - ColorSlider: Slider that changes the colorbar scale
    - title_text: Label for What Pulsar you are looking at
    - set_button: Button to launch settings menu
Methods:
    UpdateColorbar - Updates Colorbar Scale
    ShowSettings - Launces Settings Menu
    ChangeMode - Changes Viewing mode
    UpdateLabel - Updates the Text Labels at the top of the screen
    Updatefig - Animates the waterfall plots
"""


class Window(QDialog):

    # Receive Data from Handshake
    def receive(self, connection, length):
        chunks = []
        bytes_recd = 0
        while bytes_recd < length:
            chunk = connection.recv(min(length - bytes_recd, 2048))
            if chunk == b"":
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return b"".join(chunks)

    def __init__(self, main_startup, parent=None):  # Constructor
        super(Window, self).__init__(parent)

        # Intialize Variables
        self.header_fmt = "=iiiidiiiId"  # Header Format
        self.stokes_lookup = [
            "YX",
            "XY",
            "YY",
            "XX",
            "LR",
            "RL",
            "LL",
            "RR",
            "I",
            "Q",
            "U",
            "V",
        ]
        self.curtime = 0  # Pointing Information
        self.curpoint = 0
        self.DeDisperse = True  # Toggles de-Dispersion
        self.medsub = True  # Toggles the viewing mode (Median Subtraction)
        self.centerPulsar = False
        self.colorscale = [-0.5, 0.5]  # Intial Color Scale
        self.mooncakeIP = "192.168.3.105"
        self.mooncakePort = 6350
        self.main_startup = main_startup

        # Intialize Handshake and Variables
        self.TCP_IP = "127.0.0.1"
        self.TCP_PORT = main_startup.Receive_Port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.TCP_IP, self.TCP_PORT))
        self.sock.listen(1)
        print("Wating for Kotekan...")
        self.connection, self.client_address = self.sock.accept()
        print("Connected to Kotekan")
        self.packed_header = self.receive(self.connection, 48)
        self.tcp_header = struct.unpack(self.header_fmt, self.packed_header)
        self.pkt_length = self.tcp_header[0]  # packet_length
        self.pkt_header = self.tcp_header[1]  # header_length
        self.pkt_samples = self.tcp_header[2]  # samples_per_packet
        self.pkt_dtype = self.tcp_header[3]  # sample_type
        self.pkt_raw_cad = self.tcp_header[4]  # raw_cadence
        self.pkt_freqs = self.tcp_header[5]  # num_freqs
        self.pkt_elems = self.tcp_header[6]  # num_freqs
        self.pkt_int_len = self.tcp_header[7]  # samples_summed
        self.pkt_idx0 = self.tcp_header[8]  # handshake_idx
        self.pkt_utc0 = self.tcp_header[9]  # handshake_utc

        # Get Intial Pulsar Intformation
        self.psrcat = json.load(open("psrcat/psrcat_b.json"))["pulsars"]
        self.target = "B0329+54"
        self.temp_target = ""
        self.psrdata = self.psrcat[self.target]
        # Compute some helpful values
        self.sec_per_pkt_frame = self.pkt_raw_cad * self.pkt_int_len
        self.info_header = self.receive(
            self.connection, self.pkt_freqs * 4 * 2 + self.pkt_elems * 1
        )
        self.freqlist = np.fromstring(
            self.info_header[: int(self.pkt_freqs) * 4 * 2], dtype=np.float32
        ).reshape(-1, 2)
        self.freqlist = self.freqlist / 1e6
        self.elemlist = np.fromstring(
            self.info_header[int(self.pkt_freqs) * 4 * 2 :], dtype=np.int8
        )
        self.plot_freqs = self.pkt_freqs // 8
        self.plot_times = 256
        self.plot_phase = 128
        self.total_integration = 1024 * 8

        if self.pkt_int_len > self.total_integration:  # Correct integration length
            print("Pre-integrated to longer than desired time!")
            print("{} vs {}".format(self.pkt_int_len, self.total_integration))
            print("Resetting integration length to {}".format(self.pkt_int_len))
            self.total_integration = self.pkt_int_len

        self.local_integration = self.total_integration // self.pkt_int_len

        # Intialize waterfall arrays
        self.waterfall = (
            np.zeros(
                (self.plot_times, int(self.plot_freqs), int(self.pkt_elems)),
                dtype=np.float32,
            )
            + np.nan
        )
        self.countfold = np.zeros(
            (self.plot_phase, int(self.plot_freqs), int(self.pkt_elems)),
            dtype=np.float32,
        )
        try:
            self.fold_period = 1.0 / self.psrdata["frequency"]
        except:
            self.fold_period = self.psrdata["period"]
        self.waterfold = np.zeros(
            (self.plot_phase, int(self.plot_freqs), int(self.pkt_elems)),
            dtype=np.float32,
        )
        self.times = np.zeros(self.plot_times)

        time.sleep(1)

        # Intialize plots
        self.f, self.ax = plt.subplots(
            2, self.pkt_elems, gridspec_kw={"height_ratios": [2, 1]}
        )
        self.f.subplots_adjust(right=0.8)
        if self.pkt_elems == 1:
            self.ax = [self.ax]
        plt.ioff()

        self.p = []  # list to hold graphs

        # Intialize Time information
        self.tmin = md.date2num(
            datetime.datetime.fromtimestamp(
                self.pkt_utc0
                - self.plot_times * self.local_integration * self.sec_per_pkt_frame
            )
        )
        self.tmax = md.date2num(datetime.datetime.fromtimestamp(self.pkt_utc0))
        self.times = (
            self.pkt_utc0
            - np.arange(self.plot_times)
            * self.local_integration
            * self.sec_per_pkt_frame
        )
        self.date_format = md.DateFormatter("%H:%M:%S")

        # Create Waterfall Plots and add to list
        for i in np.arange(self.pkt_elems):
            self.p.append(
                self.ax[0, i].imshow(
                    self.waterfall[:, :, i],
                    aspect="auto",
                    animated=True,
                    origin="upper",
                    interpolation="nearest",
                    cmap="viridis",
                    vmin=self.colorscale[0],
                    vmax=self.colorscale[1],
                    extent=[
                        self.freqlist[0, 0],
                        self.freqlist[-1, -1],
                        self.tmin,
                        self.tmax,
                    ],
                )
            )
            self.ax[0, i].set_yticklabels([])
            self.ax[0, i].yaxis_date()

        # Plot Labels
        self.ax[0, 0].set_title(self.stokes_lookup[self.elemlist[0] + 8])
        self.ax[0, 1].set_title(self.stokes_lookup[self.elemlist[1] + 8])

        self.ax[0, 0].set_ylabel("Local Time")
        self.ax[0, 0].yaxis_date()
        self.ax[0, 0].yaxis.set_major_formatter(self.date_format)

        # Create Folded Waterall Plots and add to list
        for i in np.arange(self.pkt_elems):
            self.p.append(
                self.ax[1, i].imshow(
                    self.waterfold[:, :, i],
                    aspect="auto",
                    animated=True,
                    origin="upper",
                    interpolation="nearest",
                    cmap="viridis",
                    vmin=self.colorscale[0],
                    vmax=self.colorscale[1],
                    extent=[self.freqlist[0, 0], self.freqlist[-1, -1], 0, 1],
                )
            )
            self.ax[1, i].set_xlabel("Freq (MHz)")
        # Plot Labels
        self.ax[1, 0].set_ylabel("Pulse Phase")
        # Colorbar Labels
        self.cbar_ax = self.f.add_axes([0.85, 0.15, 0.05, 0.7])
        self.c = self.f.colorbar(self.p[0], cax=self.cbar_ax)
        self.c.set_label("Power (dB, arbitrary)")

        # Add graph to screen and animate it
        self.figure = self.f
        self.canvas = FigureCanvas(self.figure)
        self.ani = animation.FuncAnimation(
            self.f, self.Updatefig, frames=100, interval=100
        )  # Animated by self.Updatefig

        # Create UI, intialize widgets
        self.set_button = QPushButton("Show Settings")
        self.set_button.clicked.connect(self.ShowSettings)
        newfont = QFont("Times", 15, QFont.Bold)  # Font
        self.title_text = QLabel(self)
        self.title_text.setAlignment(Qt.AlignCenter)
        self.title_text.setFont(newfont)
        self.title_text.setFixedHeight(25)
        hbox1 = QHBoxLayout()
        self.pos_text_1 = QLabel(self)
        self.pos_text_1.setFont(newfont)
        self.pos_text_1.setFixedHeight(25)
        self.pos_text_1.setAlignment(Qt.AlignLeft)
        hbox1.addWidget(self.pos_text_1)
        self.pos_text_2 = QLabel(self)
        self.pos_text_2.setFont(newfont)
        self.pos_text_2.setFixedHeight(25)
        self.pos_text_2.setAlignment(Qt.AlignRight)
        hbox1.addWidget(self.pos_text_2)
        hbox2 = QHBoxLayout()
        self.info_text_1 = QLabel(self)
        self.info_text_1.setFont(newfont)
        self.info_text_1.setFixedHeight(25)
        self.info_text_1.setAlignment(Qt.AlignLeft)
        hbox2.addWidget(self.info_text_1)
        self.info_text_2 = QLabel(self)
        self.info_text_2.setFont(newfont)
        self.info_text_2.setFixedHeight(25)
        self.info_text_2.setAlignment(Qt.AlignRight)
        hbox2.addWidget(self.info_text_2)
        hbox3 = QHBoxLayout()
        self.ColorSlider = QSlider(Qt.Vertical)
        self.ColorSlider.setMinimum(1)
        self.ColorSlider.setMaximum(10)
        self.ColorSlider.setValue(5)
        self.ColorSlider.setFocusPolicy(Qt.StrongFocus)
        self.ColorSlider.setTickPosition(QSlider.TicksBothSides)
        self.ColorSlider.setTickInterval(1)
        self.ColorSlider.valueChanged.connect(self.UpdateColorbar)
        hbox3.addWidget(self.canvas)
        hbox3.addWidget(self.ColorSlider)

        # Add Widgets to the layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.title_text)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addWidget(self.set_button)

        self.UpdateLabel()  # Update the labels
        self.setLayout(vbox)  # Add the layout to the screen
        self.showMaximized()  # Maximize the window

    def UpdateColorbar(self):  # Update Colobar scale

        if self.medsub:  # If Median Subtracted
            self.colorscale = [
                -1 * self.ColorSlider.value() / 10.0,
                self.ColorSlider.value() / 10.0,
            ]  # Compute Scale based on Slider
        else:
            self.colorscale = [
                np.median(self.waterfall) - 1 * self.ColorSlider.value() * 2,
                np.median(self.waterfall) + self.ColorSlider.value() * 2,
            ]  # Compute Scale based on Slider

    def ShowSettings(self):  # Launch Settings Menu
        self.new_child = Settings(self)
        self.new_child.show()

    def ChangeMode(self):  # Toggle Viewing Mode

        self.medsub = not self.medsub  # Toggle viewing mode
        if self.medsub:  # Update Color bar
            self.colorscale = [
                -1 * self.ColorSlider.value() / 10.0,
                self.ColorSlider.value() / 10.0,
            ]
        else:
            self.colorscale = [
                np.median(self.waterfall) - 1 * self.ColorSlider.value() * 2,
                np.median(self.waterfall) + self.ColorSlider.value() * 2,
            ]

    def UpdateLabel(self):  # Handles Labels at top of Main window

        if self.curtime != 0:  # Calulate J2000 pointing
            Pointing = (
                self.curtime.isot + "        " + self.curpoint.altaz.to_string("dms")
            )
            c = SkyCoord(
                self.curpoint.altaz.to_string("hmsdms").split(" ")[0],
                self.curpoint.altaz.to_string("hmsdms").split(" ")[1],
                frame="icrs",
            )
            c = c.transform_to("fk5").to_string("hmsdms")
        else:
            Pointing = "Unavailiable"
            c = "Unavailiable"
        if self.temp_target == "":  # Handle the Pulsar information
            self.title_text.setText(self.target)
        else:
            self.title_text.setText(self.temp_target + " (PSRCAT: " + self.target + ")")
        self.pos_text_1.setText("Pointing: " + Pointing)  # Pointing labels
        self.pos_text_2.setText("FK5: " + c)
        self.info_text_1.setText(
            "Dispersion Measure: " + str(self.psrdata["dmeasure"])
        )  # Pulsar information labels
        self.info_text_2.setText("Fold Period: " + str(self.fold_period))

    def Updatefig(self, *args):  # Animates the waterfall plots

        self.tmin = md.date2num(
            datetime.datetime.fromtimestamp(np.amin(self.times))
        )  # Find new time bounds of waterfalls
        self.tmax = md.date2num(datetime.datetime.fromtimestamp(np.amax(self.times)))
        self.dedispersed = np.copy(
            self.waterfold
        )  # Create a copy of the folded array to dedisperse
        self.dedispersed_count = np.copy(self.countfold)
        self.MedSubbed = np.zeros_like(self.dedispersed[:, :, 0])
        for i in np.arange(self.pkt_elems):  # For each polirization
            if self.DeDisperse:  # De-disperse
                for j in range(
                    self.dedispersed.shape[1]
                ):  # Iterate through frequencies
                    time = (
                        (4.148808 / 1000)
                        * (
                            (1 / 0.4) ** 2
                            - (
                                1
                                / (0.4 + (128 - j) * (0.4 / self.dedispersed.shape[1]))
                            )
                            ** 2
                        )
                        * self.psrdata["dmeasure"]
                    )  # Compute Time lag
                    location = 1 * int(
                        (time / self.fold_period) * self.dedispersed.shape[0]
                    )  # Compute Index
                    self.dedispersed[:, j, i] = np.roll(
                        self.dedispersed[:, j, i], location
                    )  # Adjust the frequency by the tim lag
                    self.dedispersed_count[:, j, i] = np.roll(
                        self.dedispersed_count[:, j, i], location
                    )
            if self.medsub:  # If median subtracted mode selected
                self.p[i].set_data(
                    self.waterfall[:, :, i]
                    - np.nanmedian(self.waterfall[:, :, i], axis=0)[np.newaxis, :]
                )  # subtract Median and update data
                tmpdata = 10 * np.log10(
                    self.dedispersed[:, :, i] / self.dedispersed_count[:, :, i]
                )  # Update Folded Graph
                if self.centerPulsar:
                    pulse_max = np.max(np.median(tmpdata, axis=1))
                    pulse_max_index = np.where(np.median(tmpdata, axis=1) == pulse_max)[
                        0
                    ]
                    if pulse_max_index.size > 0:
                        tmpdata = np.roll(
                            tmpdata,
                            int(np.abs(pulse_max_index - tmpdata.shape[0] // 2)),
                            axis=0,
                        )
                self.p[self.pkt_elems + i].set_data(
                    tmpdata - np.median(tmpdata, axis=0)[np.newaxis, :]
                )

            else:
                self.p[i].set_data(self.waterfall[:, :, i])  # Update watefall
                tmpdata = 10 * np.log10(
                    self.dedispersed[:, :, i] / self.dedispersed_count[:, :, i]
                )  # Update Folded Waterfall
                self.p[self.pkt_elems + i].set_data(tmpdata)
            self.MedSubbed += (
                (self.dedispersed[:, :, i] / self.dedispersed_count[:, :, i])
                - np.median(
                    (self.dedispersed[:, :, i] / self.dedispersed_count[:, :, i]),
                    axis=0,
                )[np.newaxis, :]
            )
            self.p[i].set_extent(
                [self.freqlist[0, 0], self.freqlist[-1, -1], self.tmin, self.tmax]
            )  # Update Graph Scale
            self.p[i].set_clim(
                vmin=self.colorscale[0], vmax=self.colorscale[1]
            )  # Update colorbar Scale
            self.p[self.pkt_elems + i].set_clim(
                vmin=self.colorscale[0] / 10, vmax=self.colorscale[1] / 10
            )
        return self.p


class Startup(QDialog):
    def __init__(self, app, parent=None):  # Constructor
        super(Startup, self).__init__(parent)

        self.Receive_User = "natasha"
        self.Receive_Ip = "192.168.52.35"
        self.Receive_Port = 2054
        self.app = app

        newfont = QFont("Times", 15, QFont.Bold)  # Font
        self.startup_welcome = QLabel(self)
        self.startup_welcome.setAlignment(Qt.AlignCenter)
        self.startup_welcome.setFont(newfont)
        self.startup_welcome.setFixedHeight(25)
        self.startup_welcome.setText("Welcome to ARO Live-Viewer!")

        self.ReciveIP = QLabel("Where is Kotekan Running?")
        self.ReciveUser_Edit = QLineEdit(self.Receive_User)
        self.ReciveIP_Edit = QLineEdit(self.Receive_Ip)
        self.RecivePort = QLabel("What Port is it Streaming to?")
        self.ReceivePort_Edit = QLineEdit(str(self.Receive_Port))
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.ExitApp)

        vbox = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.ReciveIP)
        hbox1.addWidget(self.ReciveUser_Edit)
        hbox1.addWidget(self.ReciveIP_Edit)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.RecivePort)
        hbox2.addWidget(self.ReceivePort_Edit)
        vbox.addWidget(self.startup_welcome)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addWidget(self.start_button)
        self.setLayout(vbox)

    def ExitApp(self):
        self.Receive_User = self.ReciveUser_Edit.text()
        self.Receive_Port = int(self.ReceivePort_Edit.text())
        self.Receive_Ip = self.ReciveIP_Edit.text()
        bashCommand = "ps -ef | grep ssh"
        tunnelCommand = "ssh -R %d:localhost:2054 %s@%s" % (
            self.Receive_Port,
            self.Receive_User,
            self.Receive_Ip,
        )
        process = subprocess.Popen(
            bashCommand, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
        )
        output, error = process.communicate()
        if output is not None and tunnelCommand in output.decode("utf-8"):
            print("A tunnel has already been opened")
        elif output is not None:
            print("Opening tunnel to %s" % (self.Receive_User))
            subprocess.call(["gnome-terminal", "-e", tunnelCommand])
            # process = subprocess.Popen(tunnelCommand, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell = True)
        else:
            print(error, output)
        self.app.quit()
        self.hide()


def data_listener():  # Listens to Data and updates
    # Intialize Vairiables
    global main
    last_idx = main.pkt_idx0
    data_pkt_frame_idx = 0
    data_pkt_samples_summed = 1
    idx = 0

    while True:
        try:
            d = np.zeros(
                [int(main.pkt_freqs), main.pkt_elems]
            )  # Declare array to hold incoming data
            n = np.zeros([int(main.pkt_freqs), main.pkt_elems])
            t = np.zeros(main.plot_times)
            main.waterfold *= 0.999  # Reduce Old Folded Data
            main.countfold *= 0.999
            for i in np.arange(int(main.local_integration * main.pkt_elems)):
                data = main.receive(
                    main.connection, main.pkt_length + main.pkt_header
                )  # Receive Data from Port
                if len(data) != main.pkt_length + main.pkt_header:
                    print("Lost Connection to port, exiting...")
                    main.connection.close()
                    return
                (
                    data_pkt_frame_idx,
                    data_pkt_elem_idx,
                    data_pkt_samples_summed,
                ) = struct.unpack("III", data[: main.pkt_header])
                data_array = np.fromstring(
                    data[int(main.pkt_header) :], dtype=np.uint32
                )  # Unpack Data
                d[:, data_pkt_elem_idx] += data_array * 1.0  # Store Data
                n[:, data_pkt_elem_idx] += (
                    data_pkt_samples_summed * 1.0
                )  # Store Integration lengths
                # n[:,data_pkt_elem_idx][data_array != 0] += data_pkt_samples_summed * 1.0
                fold_idx = np.array(
                    int(
                        (
                            (
                                main.sec_per_pkt_frame * data_pkt_frame_idx
                                + 0.5 * main.fold_period
                            )
                            % main.fold_period
                        )
                        / main.fold_period
                        * main.plot_phase
                    ),
                    dtype=np.int32,
                )
                main.waterfold[
                    int(fold_idx), :, int(data_pkt_elem_idx)
                ] += data_array.reshape(
                    -1, int(main.pkt_freqs // main.plot_freqs)
                ).mean(
                    axis=1
                )  # Fold Waterfall
                main.countfold[
                    int(fold_idx), :, int(data_pkt_elem_idx)
                ] += data_pkt_samples_summed
                # main.countfold[:,15:70,:] = 0
            roll_idx = (data_pkt_frame_idx - last_idx) // main.local_integration
            main.times = np.roll(main.times, int(roll_idx))  # Roll times
            main.times[0] = (
                main.sec_per_pkt_frame * (data_pkt_frame_idx - main.pkt_idx0)
                + main.pkt_utc0
            )
            main.waterfall = np.roll(main.waterfall, int(roll_idx), axis=0)
            main.waterfall[0, :, :] = 10 * np.log10(
                (d / n)
                .reshape(
                    -1, int(main.pkt_freqs // main.plot_freqs), int(main.pkt_elems)
                )
                .mean(axis=1)
            )  # Add data to waterfall
            # main.waterfall[0,:,:] = (d/1000).reshape(-1,main.pkt_freqs / main.plot_freqs,main.pkt_elems).mean(axis=1)
            last_idx = data_pkt_frame_idx

        except:  # When Connection is lost, try to reconnect
            print("Lost connection to Kotekan, trying to reconnect...")
            main.connection, main.client_address = main.sock.accept()
            main.packed_header = main.receive(main.connection, 48)
            main.info_header = main.receive(
                main.connection, main.pkt_freqs * 4 * 2 + main.pkt_elems * 1
            )
            main.pkt_length = main.tcp_header[0]  # packet_length
            main.pkt_header = main.tcp_header[1]  # header_length
            main.pkt_samples = main.tcp_header[2]  # samples_per_packet
            main.pkt_dtype = main.tcp_header[3]  # sample_type
            main.pkt_raw_cad = main.tcp_header[4]  # raw_cadence
            main.pkt_freqs = main.tcp_header[5]  # num_freqs
            main.pkt_elems = main.tcp_header[6]  # num_freqs
            main.pkt_int_len = main.tcp_header[7]  # samples_summed
            main.pkt_idx0 = main.tcp_header[8]  # handshake_idx
            main.pkt_utc0 = main.tcp_header[9]  # handshake_utc

            main.sec_per_pkt_frame = main.pkt_raw_cad * main.pkt_int_len

            # main.waterfall = np.zeros((self.plot_times,self.plot_freqs,self.pkt_elems),dtype=np.float32) + np.nan;
            print("Reconnected!")


def get_pointing():  # Function to get Pointing information

    global main
    while True:
        ARO = EarthLocation(
            lat=45.95550333 * u.deg, lon=-78.073040402778 * u.deg, height=260.4 * u.m
        )  # Define current loction (ARO)
        try:  # Try to connect to "Mooncake" and get pointing information
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.mooncakeIP, self.mooncakePort))
            s.send("A")
            b = s.recv(100).rstrip()
            # Format Current Time
            main.curtime = Time(
                "{}-{}-{}:{}:{}".format(
                    b.split(" ")[0][0:4],
                    b.split(" ")[0][4:6],
                    b.split(" ")[0][6:11],
                    b.split(" ")[0][11:13],
                    b.split(" ")[0][13:-1],
                )
            )
            # Calculate pointing in alt az
            main.curpoint = SkyCoord(
                alt=Angle(b.split(" ")[2], unit=u.deg),
                az=Angle(b.split(" ")[1], unit=u.deg),
                obstime=main.curtime,
                frame="altaz",
                location=ARO,
            )
            s.close()
            main.UpdateLabel()  # Change Labels
            """
            for k,v in self.psrcat.iteritems():
            if v['RA'][0:6] == '01h47m':
                print k
            """
            time.sleep(1)
        except:
            print("disconnected from mooncake, retrying in 30 seconds...")
            time.sleep(30)


# Main program
if __name__ == "__main__":

    np.seterr(divide="ignore", invalid="ignore")  # Remove numpy divide error messages
    startup = QApplication(sys.argv)
    main_startup = Startup(startup)
    main_startup.show()
    startup.exec_()
    # Launch main Window
    app = QApplication(sys.argv)
    main = Window(main_startup)
    main.show()
    # Launch data listening thread
    thread = threading.Thread(target=data_listener)
    thread.daemon = True
    thread.start()
    # Launch Pointing thread
    thread2 = threading.Thread(target=get_pointing)
    thread2.daemon = True
    thread2.start()
    # Exit Upon exiting window
    sys.exit(app.exec_())
