#!/usr/bin/env python3
"""
/*********************************************************************************
* RFI Documentation Header Block
* File: rfi_receiver.py
* Purpose: A server to receive real-time rfi data from kotekan and send to rfi_client.py
* Python Version: 3.6
* Dependencies: yaml, numpy, argparse
* Help: Run "python3 rfi_receiver.py" -H (or --Help) for how to use.
*********************************************************************************/
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility


from future import standard_library

standard_library.install_aliases()
from comet import Manager, CometError
from prometheus_client import start_http_server, Gauge
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import threading
import socket
import numpy as np
import datetime
import os
import time
import argparse
import yaml
import subprocess
import requests
import json
import imp
from ch_util import ephemeris
from kotekan import __version__


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_dict(cmd, _dict):
    for key, value in _dict.items():
        if type(value) == dict:
            parse_dict(cmd, value)
        else:
            if key in cmd.config:
                if type(cmd.config[key]) == type(value):
                    logger.debug(
                        "Setting Config Parameter %s to %s" % (key, str(value))
                    )
                    cmd.config[key] = value


class CommandLine(object):
    def __init__(self):

        # Defaults
        self.startup_time = datetime.datetime.utcnow()

        self.git_version = __version__

        self.UDP_IP = "0.0.0.0"
        self.UDP_PORT = 2900
        self.TCP_IP = "10.10.10.2"
        self.TCP_PORT = 41214
        self.mode = "chime"
        self.debug = False
        self.min_seq = -1
        self.max_seq = -1
        self.config = {
            "frames_per_packet": 1,
            "num_global_freq": 1024,
            "num_local_freq": 1,
            "samples_per_data_set": 49152,
            "num_elements": 2048,
            "timestep": 2.56e-6,
            "bytes_per_freq": 16,
            "waterfallX": 1024,
            "waterfallY": 1024,
            "bi_frames_per_packet": 10,
            "sk_step": 256,
            "chime_rfi_header_size": 35,
            "num_receive_threads": 4,
            "use_dataset_broker": True,
            "ds_broker_host": "10.1.50.11",
            "ds_broker_port": 12050,
            "solar_transit_downtime_m": 60,
        }
        self.supportedModes = ["vdif", "pathfinder", "chime"]
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
            "-s", "--send", help="Example: 10.10.10.2:41214", required=False, default=""
        )
        parser.add_argument(
            "-c",
            "--config",
            help="Example: ../kotekan/kotekan_opencl_rfi.yaml",
            required=False,
            default="",
        )
        parser.add_argument(
            "-m", "--mode", help="Example: vdif, chime", required=False, default=""
        )
        parser.add_argument(
            "-d",
            "--debug",
            help="Launch with warnings",
            required=False,
            default="",
            action="store_true",
        )

        argument = parser.parse_args()
        status = False

        if argument.Help:
            logger.debug(
                "You have used '-H' or '--Help' with argument: {0}".format(
                    argument.Help
                )
            )
            status = True
        if argument.debug:
            logger.info("You have used '-d' , Enabling debug mode")
            self.debug = True
            status = True
            logging.basicConfig(level=logging.DEBUG)
        if argument.send:
            logger.debug(
                "You have used '-s' or '--send' with argument: {0}".format(
                    argument.send
                )
            )
            self.TCP_IP = argument.send[: argument.send.index(":")]
            self.TCP_PORT = int(argument.send[argument.send.index(":") + 1 :])
            logger.info("Setting TCP IP: %s PORT: %d" % (self.TCP_IP, self.TCP_PORT))
            status = True
        if argument.receive:
            logger.debug(
                "You have used '-r' or '--receive' with argument: {0}".format(
                    argument.receive
                )
            )
            self.UDP_IP = argument.receive[: argument.receive.index(":")]
            self.UDP_PORT = int(argument.receive[argument.receive.index(":") + 1 :])
            logger.info("Setting UDP IP: %s PORT: %d" % (self.UDP_IP, self.UDP_PORT))
            status = True
        if argument.config:
            logger.debug(
                "You have used '-c' or '--config' with argument: {0}".format(
                    argument.config
                )
            )
            parse_dict(self, yaml.safe_load(open(argument.config)))
            logger.info(self.config)
            self.register_config(self.config)
            status = True
        if argument.mode:
            logger.debug(
                "You have used '-m' or '--mode' with argument: {0}".format(
                    argument.mode
                )
            )
            if argument.mode in self.supportedModes:
                self.mode = argument.mode
                logger.info("Setting mode to %s mode." % (argument.mode))
            else:
                logger.warning(
                    "This mode in currently not supported, reverting to default"
                )
                logger.warning("Supported Modes Include:")
                for mode in self.supportedModes:
                    logger.warning("- ", mode)
            status = True
        if not status:
            logger.info(
                "Remember: You can use -H or - Help to see configuration options"
            )
        self.bad_input_mask = [0] * self.config["num_elements"]

    def register_config(self, config):
        # Register config with comet broker
        try:
            enable_comet = config["use_dataset_broker"]
        except KeyError:
            logger.error("Missing config value 'dataset_manager/use_dataset_broker'.")
            exit(1)
        if enable_comet:
            try:
                comet_host = config["ds_broker_host"]
                comet_port = config["ds_broker_port"]
            except KeyError as exc:
                logger.error(
                    "Failure registering initial config with comet broker: '{}' not defined in "
                    "config.".format(exc[0])
                )
                exit(1)
            comet = Manager(comet_host, comet_port)
            try:
                comet.register_start(self.startup_time, self.git_version)
                comet.register_config(config)
            except CometError as exc:
                logger.error("Comet failed registering initial config: {}".format(exc))
                exit(1)
        else:
            logger.warning("Config registration DISABLED. This is only OK for testing.")


class Stream(object):
    def __init__(self, thread_id, mode, header, freq_bins, known_streams):

        encoded_stream_id = header["encoded_stream_ID"][0]
        if encoded_stream_id not in known_streams:
            known_streams.append(encoded_stream_id)
            self.link_id = encoded_stream_id & 0x000F
            self.slot_id = (encoded_stream_id & 0x00F0) >> 4
            self.crate = (encoded_stream_id & 0x0F00) >> 8
            self.unused = (encoded_stream_id & 0xF000) >> 12
            if mode == "pathfinder":
                self.bins = [
                    self.slot_id + self.link_id * 16 + i * 128
                    for i in range(header["num_local_freq"][0])
                ]
            elif mode == "chime":
                self.bins = freq_bins
            elif mode == "vdif":
                self.bins = list(range(header["num_local_freq"][0]))
            self.freqs = [800.0 - float(b) * 400.0 / 1024.0 for b in self.bins]
            self.bins = np.array(self.bins).astype(np.int)
            self.freqs = np.array(self.freqs)
            # logger.debug("Thread id %d Stream Created %d %d %d %d %d"%(thread_id, encoded_stream_id, self.slot_id, self.link_id, self.crate, self.unused))
            # logger.debug(self.bins, self.freqs)
        else:
            logger.warning("Stream Creation Warning: Known Stream Creation Attempt")


def HeaderCheck(header, app):

    if header["combined_flag"] != 1:
        logger.error("Header Error: Only Combined RFI values are currently supported ")
        return False
    if header["sk_step"] != app.config["sk_step"]:
        logger.error(
            "Header Error: SK Step does not match config; Got value %d"
            % (header["sk_step"])
        )
        return False
    if header["num_elements"] != app.config["num_elements"]:
        logger.error(
            "Header Error: Number of Elements does not match config; Got value %d"
            % (header["num_elements"])
        )
        return False
    if header["num_timesteps"] != app.config["samples_per_data_set"]:
        logger.error(
            "Header Error: Samples per Dataset does not match config; Got value %d"
            % (header["num_timesteps"])
        )
        return False
    if header["num_global_freq"] != app.config["num_global_freq"]:
        logger.error(
            "Header Error: Number of Global Frequencies does not match config; Got value %d"
            % (header["num_global_freq"])
        )
        return False
    if header["num_local_freq"][0] != app.config["num_local_freq"]:
        logger.error(
            "Header Error: Number of Local Frequencies does not match config; Got value %d"
            % (header["num_local_freq"][0])
        )
        return False
    if header["fpga_seq_num"] < 0:
        logger.error(
            "Header Error: Invalid FPGA sequence Number; Got value %d"
            % (header["fpga_seq_num"])
        )
        return False
    if (
        header["frames_per_packet"] != app.config["frames_per_packet"]
        and header["frames_per_packet"] != app.config["bi_frames_per_packet"]
    ):
        logger.error(
            "Header Error: Frames per Packet does not match config; Got value %d"
            % (header["frames_per_packet"])
        )
        return False

    # logger.debug("First Packet Received, Valid Chime Header Confirmed.")
    return True


# Listen for UDP packets from Kotekan
def data_listener(thread_id):

    global waterfall, t_min, app, sk_receive_watchdogs, InitialKotekanConnection

    UDP_PORT = app.UDP_PORT + thread_id
    socket_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket_udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_udp.bind((app.UDP_IP, UDP_PORT))

    # Config Variables
    frames_per_packet = app.config["frames_per_packet"]
    local_freq = app.config["num_local_freq"]
    timesteps_per_frame = app.config["samples_per_data_set"]
    timestep = app.config["timestep"]
    bytesPerFreq = app.config["bytes_per_freq"]
    global_freq = app.config["num_global_freq"]
    sk_step = app.config["sk_step"]
    RFIHeaderSize = app.config["chime_rfi_header_size"]
    mode = app.mode
    firstPacket = True
    vdifPacketSize = global_freq * 4 + RFIHeaderSize
    # CHIME packet: RFIHeader + frequency_bins[local_freq] + rfi_avg[local_freq]
    chimePacketSize = RFIHeaderSize + 4 * local_freq + 4 * local_freq

    # Packet data type received from Kotekan
    HeaderDataType = np.dtype(
        [
            ("combined_flag", np.uint8, 1),
            ("sk_step", np.uint32, 1),
            ("num_elements", np.uint32, 1),
            ("num_timesteps", np.uint32, 1),
            ("num_global_freq", np.uint32, 1),
            ("num_local_freq", np.uint32, 1),
            ("frames_per_packet", np.uint32, 1),
            ("fpga_seq_num", np.int64, 1),
            ("encoded_stream_ID", np.uint16, 1),
        ]
    )
    stream_dict = dict()
    known_streams = []
    packetCounter = 0
    freq_bins_set = set()

    while True:

        # sk_receive_watchdog = datetime.datetime.now()
        # Receive packet from port
        packet, addr = socket_udp.recvfrom(chimePacketSize)

        if not InitialKotekanConnection:
            sk_receive_watchdogs[thread_id] = datetime.datetime.now()
            InitialKotekanConnection = True
            logger.info("Connected to Kotekan")

        # A packet is received on each stream roughly every ~0.126s
        if packet != "":

            # Print frequency bins received every ~19s
            if packetCounter % (150 * len(stream_dict) + 1) == 0:
                logger.debug(
                    "data_listener: Thread id: %d, Streams: %d, Receiving frequency bins: %s"
                    % (thread_id, len(stream_dict), freq_bins_set)
                )
                freq_bins_set.clear()

            packetCounter += 1

            header = np.fromstring(packet[:RFIHeaderSize], dtype=HeaderDataType)
            freq_bins = np.fromstring(
                packet[RFIHeaderSize : RFIHeaderSize + 4 * local_freq], dtype=np.uint32
            )
            freq_bins_set.update(freq_bins)
            data = np.fromstring(
                packet[RFIHeaderSize + 4 * local_freq :], dtype=np.float32
            )

            # Create a new stream object each time a new stream connects
            if header["encoded_stream_ID"][0] not in known_streams:
                # Check that the new stream is providing the correct data
                if HeaderCheck(header, app) == False:
                    break
                # Add to the dictionary of Streams
                stream_dict[header["encoded_stream_ID"][0]] = Stream(
                    thread_id, mode, header, freq_bins, known_streams
                )

            # On first packet received by any stream
            if app.min_seq == -1:

                # Set up waterfall parameters
                t_min = datetime.datetime.utcnow()
                app.min_seq = header["fpga_seq_num"][0]
                app.max_seq = (
                    app.min_seq
                    + (waterfall.shape[1] - 1) * timesteps_per_frame * frames_per_packet
                )
                firstPacket = False

            else:

                if header["fpga_seq_num"][0] > app.max_seq:

                    roll_amount = int(
                        -1
                        * max(
                            (header["fpga_seq_num"][0] - app.max_seq)
                            // (timesteps_per_frame * frames_per_packet),
                            waterfall.shape[1] // 8,
                        )
                    )
                    # If the roll is larger than the whole waterfall (kotekan dies and rejoins)
                    if -1 * roll_amount > waterfall.shape[1]:
                        # Reset Waterfall
                        t_min = datetime.datetime.utcnow()
                        waterfall[:, :] = -1  # np.nan
                        app.min_seq = header["fpga_seq_num"][0]
                        app.max_seq = (
                            app.min_seq
                            + (waterfall.shape[1] - 1)
                            * timesteps_per_frame
                            * frames_per_packet
                        )
                    else:
                        # DO THE ROLL, Note: Roll Amount is negative
                        waterfall = np.roll(waterfall, roll_amount, axis=1)
                        waterfall[:, roll_amount:] = -1  # np.nan
                        app.min_seq -= (
                            roll_amount * timesteps_per_frame * frames_per_packet
                        )
                        app.max_seq = (
                            app.min_seq
                            + (waterfall.shape[1] - 1)
                            * timesteps_per_frame
                            * frames_per_packet
                        )
                        # Adjust Time
                        t_min += datetime.timedelta(
                            seconds=-1
                            * roll_amount
                            * timestep
                            * timesteps_per_frame
                            * frames_per_packet
                        )
            # if(thread_id == 1):
            # logger.debug(header['fpga_seq_num'][0],min_seq,timesteps_per_frame,frames_per_packet, (header['fpga_seq_num'][0]-min_seq)/(float(timesteps_per_frame)*frames_per_packet), np.median(data))
            idx = int(
                (header["fpga_seq_num"][0] - app.min_seq)
                // (timesteps_per_frame * frames_per_packet)
            )
            if idx >= 0 and idx < waterfall.shape[1]:
                waterfall[stream_dict[header["encoded_stream_ID"][0]].bins, idx] = data
                sk_receive_watchdogs[thread_id] = datetime.datetime.now()
            elif app.debug:
                logger.warning("Invalid Packet Location (Ignore on Startup)")
                logger.warning(datetime.datetime.utcnow())
                logger.warning(
                    "- IDX %d Header Seq %d Min Seq %d Max Seq %d Timesteps Per Frame %d Frames Per Packet %d"
                    % (
                        idx,
                        header["fpga_seq_num"][0],
                        app.min_seq,
                        app.max_seq,
                        timesteps_per_frame,
                        frames_per_packet,
                    )
                )


def bad_input_listener(thread_id):

    global bi_waterfall, bi_t_min, max_t_pos, app, bi_receive_watchdog, InitialKotekanConnection

    UDP_PORT = app.UDP_PORT + app.config["num_receive_threads"]
    socket_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket_udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_udp.bind((app.UDP_IP, UDP_PORT))

    # Config Variables
    frames_per_packet = app.config["bi_frames_per_packet"]
    local_freq = app.config["num_local_freq"]
    num_elements = app.config["num_elements"]
    timesteps_per_frame = app.config["samples_per_data_set"]
    timestep = app.config["timestep"]
    bytesPerFreq = app.config["bytes_per_freq"]
    global_freq = app.config["num_global_freq"]
    sk_step = app.config["sk_step"]
    RFIHeaderSize = app.config["chime_rfi_header_size"]
    mode = app.mode
    firstPacket = True
    # CHIME packet: RFIHeader + frequency_bins[local_freq] + faulty_counter[local_freq * num_elements]
    PacketSize = RFIHeaderSize + 4 * local_freq + local_freq * num_elements
    HeaderDataType = np.dtype(
        [
            ("combined_flag", np.uint8, 1),
            ("sk_step", np.uint32, 1),
            ("num_elements", np.uint32, 1),
            ("num_timesteps", np.uint32, 1),
            ("num_global_freq", np.uint32, 1),
            ("num_local_freq", np.uint32, 1),
            ("frames_per_packet", np.uint32, 1),
            ("fpga_seq_num", np.int64, 1),
            ("encoded_stream_ID", np.uint16, 1),
        ]
    )
    stream_dict = dict()
    known_streams = []
    packetCounter = 0

    while True:
        # Receive packet from port
        packet, addr = socket_udp.recvfrom(PacketSize)
        if not InitialKotekanConnection:
            bi_receive_watchdog = datetime.datetime.now()
            InitialKotekanConnection = True
            logger.info("Connected to Kotekan")

        # If we get something not empty
        if packet != "":
            # Every so often print that we are receiving packets
            if packetCounter % (25 * len(stream_dict) + 1) == 0:
                logger.warning(
                    "Bad Input Thread (id %d): Receiving Packets from %d Streams"
                    % (thread_id, len(stream_dict))
                )
            packetCounter += 1
            # Read the header
            header = np.fromstring(packet[:RFIHeaderSize], dtype=HeaderDataType)
            # Read the frequency bins
            freq_bins = np.fromstring(
                packet[RFIHeaderSize : RFIHeaderSize + 4 * local_freq], dtype=np.uint32
            )
            # Read the data
            data = np.fromstring(
                packet[RFIHeaderSize + 4 * local_freq :], dtype=np.uint8
            )
            # Create a new stream object each time a new stream connects
            if header["encoded_stream_ID"][0] not in known_streams:
                # logger.debug("New Stream Detected")
                # Check that the new stream is providing the correct data
                if HeaderCheck(header, app) == False:
                    break
                # Add to the dictionary of Streams
                stream_dict[header["encoded_stream_ID"][0]] = Stream(
                    thread_id, mode, header, freq_bins, known_streams
                )
            # On first packet received by any stream
            if firstPacket:
                # Set up waterfall parameters
                bi_t_min = datetime.datetime.utcnow()
                bi_min_seq = header["fpga_seq_num"][0]
                firstPacket = False
            # Add data to waterfall
            fq = stream_dict[header["encoded_stream_ID"][0]].bins
            t = (
                int(
                    (header["fpga_seq_num"][0] - bi_min_seq)
                    // (timesteps_per_frame * frames_per_packet)
                )
                % bi_waterfall.shape[2]
            )
            if t > max_t_pos:
                max_t_pos = t
            bi_waterfall[fq, :, t] = data
            bi_receive_watchdog = datetime.datetime.now()
            # if(223 in stream_dict[header['encoded_stream_ID'][0]].bins):
            #    logger.debug(np.where(data == 10)[0].size, t, fq)


def TCP_stream():

    global sock_tcp, waterfall, t_min, max_t_pos, app, tcp_connected

    sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_tcp.bind((app.TCP_IP, app.TCP_PORT))

    sock_tcp.listen(1)

    while True:

        conn, addr = sock_tcp.accept()
        logger.info("Established Connection to %s:%s" % (addr[0], addr[1]))
        tcp_connected = True

        while True:

            MESSAGE = conn.recv(1).decode()  # Client Message

            if not MESSAGE:
                break

            elif MESSAGE == "W":
                logger.debug(
                    "Sending Watefall Data %d ..." % (len(waterfall.tostring()))
                )
                conn.send(waterfall.tostring())  # Send Watefall
            elif MESSAGE == "T":
                logger.debug(
                    "Sending Time Data ...", len(t_min.strftime("%d-%m-%YT%H:%M:%S:%f"))
                )
                conn.send(t_min.strftime("%d-%m-%YT%H:%M:%S:%f").encode())
            elif MESSAGE == "w":
                temp_bi_waterfall = (
                    np.sum(bi_waterfall[:, :, :max_t_pos], axis=2)
                    + np.count_nonzero(bi_waterfall[:, :, :max_t_pos] == -1, axis=2)
                ).astype(float)
                temp_bi_waterfall /= max_t_pos - np.count_nonzero(
                    bi_waterfall[:, :, :max_t_pos] == -1, axis=2
                )
                # logger.debug(np.count_nonzero(bi_waterfall[:,:,:max_t_pos]==-1, axis = 2).shape, np.min(np.count_nonzero(bi_waterfall[:,:,:max_t_pos]==-1, axis = 2)), np.max(np.count_nonzero(bi_waterfall[:,:,:max_t_pos]==-1, axis = 2)))
                # logger.debug(np.nanmin(temp_bi_waterfall), np.nanmax(temp_bi_waterfall), np.nanmean(temp_bi_waterfall))
                # logger.debug(np.where(temp_bi_waterfall[223,:] > 2)[0].size, temp_bi_waterfall[223,143], np.nanmax(temp_bi_waterfall[223,:]), np.nanmin(temp_bi_waterfall[223,:]))
                logger.debug(
                    "Sending Bad Input Watefall Data %d ..."
                    % (len(temp_bi_waterfall.tostring()))
                )
                conn.send(temp_bi_waterfall.tostring())  # Send Watefall
            elif MESSAGE == "t":
                logger.debug(
                    "Sending Bad Input Time Data ...",
                    len(bi_t_min.strftime("%d-%m-%YT%H:%M:%S:%f")),
                )
                conn.send(bi_t_min.strftime("%d-%m-%YT%H:%M:%S:%f").encode())
        logger.info("Closing Connection to %s:%s ..." % (addr[0], str(addr[1])))
        conn.close()


def compute_metrics(bi_waterfall, waterfall, metric_dict, max_t_pos, app):

    if not app.debug:
        np.warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        np.warnings.filterwarnings("ignore", r"Mean of empty slice")
        np.warnings.filterwarnings(
            "ignore",
            r"invalid value encountered in (greater|true_divide|double_scalars)",
        )

    # Bad Input Metrics
    mean_bi_waterfall = (
        np.sum(bi_waterfall[:, :, :max_t_pos], axis=2)
        + np.count_nonzero(bi_waterfall[:, :, :max_t_pos] == -1, axis=2)
    ).astype(float)
    mean_bi_waterfall /= max_t_pos - np.count_nonzero(
        bi_waterfall[:, :, :max_t_pos] == -1, axis=2
    )
    # mean_bi_waterfall = np.mean(bi_waterfall[:,:,:max_t_pos], axis = 2)
    mean_bi_waterfall[mean_bi_waterfall < 0] = np.nan
    bad_input_band = (
        100.0
        * np.nanmedian(mean_bi_waterfall, axis=0)
        / float(app.config["bi_frames_per_packet"])
    )
    bad_input_mask = []
    for i in range(bad_input_band.size):
        if np.isnan(bad_input_band[i]):
            metric_dict["rfi_input_mask"].labels(i + 10000, i).set(-1)
            bad_input_mask.append(-1)
        else:
            metric_dict["rfi_input_mask"].labels(i + 10000, i).set(bad_input_band[i])
            bad_input_mask.append(np.round(bad_input_band[i], 2))
    app.bad_input_mask = bad_input_mask
    num_bad_inputs = bad_input_band[bad_input_band > 10.0].size
    n = app.config["sk_step"]
    M = float(n * (app.config["num_elements"] - num_bad_inputs))
    if np.isnan(num_bad_inputs):
        num_bad_inputs = -1
        M = float(n * (app.config["num_elements"]))
    metric_dict["overall_rfi_bad_input"].set(num_bad_inputs)

    # RFI metrics
    # Find which timesteps are not populated yet in the waterfall
    bad_locs = np.where(np.sum(waterfall, axis=0) == -1 * waterfall.shape[0])[0]
    if bad_locs.size > 0:
        max_pos = bad_locs[0]
    else:
        max_pos = waterfall.shape[1]
    band = np.nanmedian(waterfall[:, :max_pos], axis=1)
    med = np.nanmedian(
        band[band != -1]
    )  # ((M+1)/(M-1))*(2.0*n**2/((n-2)*(n-1)) - 8.0/n - 1)
    std = 2.0 / np.sqrt(M)
    confidence = np.abs(waterfall[:, :max_pos] - med) / std
    rfi_mask = np.zeros_like(confidence)
    rfi_mask[confidence > 3.0] = 1.0
    rfi_mask[waterfall[:, :max_pos] == -1] = np.nan
    band_perc = 100.0 * np.nanmean(rfi_mask, axis=1)
    fbins_mhz = np.round(
        np.array([800.0 - float(b) * 400.0 / 1024.0 for b in np.arange(band.size)]),
        decimals=2,
    )
    fbins = np.arange(band_perc.size)
    for i in range(band_perc.size):
        if np.isnan(band[i]) or band[i] < 0:
            metric_dict["rfi_band"].labels(fbins_mhz[i], fbins[i]).set(np.nan)
        else:
            metric_dict["rfi_band"].labels(fbins_mhz[i], fbins[i]).set(band_perc[i])
    overall_rfi = (
        100.0
        * np.nanmean(rfi_mask[waterfall[:, :max_pos]])
    )
    if np.isnan(overall_rfi):
        overall_rfi = -1
    metric_dict["overall_rfi_sk"].set(overall_rfi)

    logger.debug("Metrics Log:")
    logger.debug(
        "    - mean_bi_waterfall: min %.2f max %.2f"
        % (np.nanmin(mean_bi_waterfall), np.nanmax(mean_bi_waterfall))
    )
    logger.debug(
        "    - Bad Input Band Computed: min %.2f max %.2f"
        % (np.nanmin(bad_input_band), np.nanmax(bad_input_band))
    )
    logger.debug("    - num_bad_inputs: %d" % (num_bad_inputs))
    logger.debug(np.where(bad_input_band > 10.0)[0])
    logger.debug(bad_input_band[np.where(bad_input_band > 10.0)])
    logger.debug("    - max_pos: %d" % (max_pos))
    logger.debug(
        "    - Band Computed: min %.2f max %.2f" % (np.nanmin(band), np.nanmax(band))
    )
    logger.debug("    - Expectation of SK: %.5f Deviation of SK: %.5f" % (med, std))
    logger.debug(
        "    - Band Percent Computed: min %.2f max %.2f"
        % (np.nanmin(band_perc), np.nanmax(band_perc))
    )
    logger.debug("    - Overall RFI: %.2f" % (overall_rfi))


class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_GET(self):

        global app, logger
        self._set_headers()
        self.wfile.write(
            str.encode("rfi_bad_input_mask = %s\n" % (str(app.bad_input_mask)))
        )


def metric_thread():

    global bi_waterfall, waterfall, max_t_pos, app, InitialKotekanConnection
    logger.info("Starting Metrics Thread")
    metric_dict = dict()
    # Create Metrics:
    metric_dict["overall_rfi_sk"] = Gauge("overall_rfi_sk", "percent_masked")
    metric_dict["overall_rfi_bad_input"] = Gauge(
        "overall_rfi_bad_input", "number_of_bad_inputs"
    )
    metric_dict["rfi_band"] = Gauge("rfi_band", "med_sk", ["freq", "freq_bin"])
    metric_dict["rfi_input_mask"] = Gauge(
        "rfi_input_mask", "defect_likelihood", ["input_nu", "input_num"]
    )
    while not InitialKotekanConnection:
        time.sleep(1)
    logger.info("Starting HTTP Server")
    # Start up the server to expose the metrics.
    start_http_server(7341)  # RFI1
    # Generate some requests.
    time.sleep(5)
    while True:
        compute_metrics(bi_waterfall, waterfall, metric_dict, max_t_pos, app)
        time.sleep(10)


def watchdog_thread():

    global sk_receive_watchdogs, bi_receive_watchdog, EXIT, InitialKotekanConnection
    logger.info("Starting Watchdog Thread")
    while not InitialKotekanConnection:
        time.sleep(1)
    while True:
        # logger.debug((datetime.datetime.now() - sk_receive_watchdog).total_seconds())
        for i in range(len(sk_receive_watchdogs)):
            if (datetime.datetime.now() - sk_receive_watchdogs[i]).total_seconds() > 10:
                logger.error("Watchdog Failed: Is kotekan running?")
                EXIT = True
        if (datetime.datetime.now() - bi_receive_watchdog).total_seconds() > 10:
            logger.error("Bad Input Watchdog Failed: Is kotekan running?")
            EXIT = True
        time.sleep(1)


def http_server2():

    server_address = ("", 7342)  # RFI2
    httpd = HTTPServer(server_address, S)
    logger.info("Starting HTTP Server 2")
    httpd.serve_forever()


# Sends message to coco to turn RFI zeroing on/off
def set_rfi_zeroing(zeroing_on):

    global rfi_zeroing_url, rfi_zeroing_headers

    # Create payload
    payload = {"rfi_zeroing": zeroing_on}
    try:
        r = requests.post(
            rfi_zeroing_url, data=json.dumps(payload), headers=rfi_zeroing_headers
        )
        state = "on" if zeroing_on else "off"
        if not r.ok:
            logger.error(
                f"RFI Solar Transit Toggle: Failed to turn RFI zeroing {state}. Something went wrong in the request."
            )
        else:
            logger.info(
                f"RFI Solar Transit Toggle: Successfully turned RFI zeroing {state}."
            )
            return True
    except:
        logger.info("RFI Solar Transit Toggle: Failure to contact coco, is it running?")
    return False


# Disables RFI zeroing during a solar transit
def rfi_zeroing():

    global InitialKotekanConnection

    # Downtime of RFI zeroing
    downtime_m = app.config["solar_transit_downtime_m"]
    downtime_s = downtime_m * 60
    half_window_s = 0.5 * downtime_s

    logger.info("RFI Solar Transit Toggle: Starting thread")
    while not InitialKotekanConnection:
        time.sleep(1)
    while True:
        # Wait until the correct UTC time of the solar transit at DRAO (deals with daylight savings time)
        time_now = ephemeris.ensure_unix(datetime.datetime.utcnow())

        # Get the *next* transit in the future
        time_to_next_transit = ephemeris.solar_transit(time_now) - time_now

        # Get the *nearest* transit which we need to determine if we are still in the window
        time_to_nearest_transit = (
            ephemeris.solar_transit(time_now - 12 * 3600) - time_now
        )

        logger.info(
            "RFI Solar Transit Toggle: Time of next transit: {}".format(
                datetime.datetime.fromtimestamp(time_to_next_transit + time_now)
            )
        )
        logger.info(
            "RFI Solar Transit Toggle: Time of nearest transit: {}".format(
                datetime.datetime.fromtimestamp(time_to_nearest_transit + time_now)
            )
        )

        new_zeroing_state = True

        # Check if we are within the current transit window and wait until the end of it
        if abs(time_to_nearest_transit) < half_window_s:
            new_zeroing_state = False
            downtime_s = half_window_s + time_to_nearest_transit
            logger.info(
                "RFI Solar Transit Toggle: Within solar transit window, disabling zeroing and sleeping for {} seconds until end of window.".format(
                    downtime_s
                )
            )
        # Otherwise, we wait until the start of the next transit window
        else:
            new_zeroing_state = True
            downtime_s = time_to_next_transit - half_window_s
            logger.info(
                "RFI Solar Transit Toggle: Outside solar transit window, enabling zeroing and sleeping for {} seconds until next window.".format(
                    downtime_s
                )
            )

        # Set new RFI zeroing state
        success = set_rfi_zeroing(new_zeroing_state)

        # If we failed to set new RFI zeroing state sleep for a few seconds
        if not success:

            logger.info(
                "RFI Solar Transit Toggle: Failed to set new RFI zeroing state. Will wait for a few seconds and try again."
            )

            time.sleep(5)
            continue

        # Sleep until end of transit window or until the next one occurs
        time.sleep(downtime_s)


if __name__ == "__main__":

    app = CommandLine()

    EXIT = False
    InitialKotekanConnection = False
    tcp_connected = False

    # Intialize Time
    t_min = datetime.datetime.utcnow()
    bi_t_min = t_min
    max_t_pos = 0
    # Initialize Plot
    nx, ny = app.config["waterfallY"], app.config["waterfallX"]
    waterfall = np.empty([nx, ny], dtype=float)
    waterfall[:, :] = -1  # np.nan
    bi_waterfall = np.empty(
        [app.config["num_global_freq"], app.config["num_elements"], 64], dtype=np.int8
    )
    bi_waterfall[:, :, :] = -1  # np.nan
    time.sleep(1)

    # Spawn threads to receive UDP packets from Kotekan
    sk_receive_watchdogs = [datetime.datetime.now()] * app.config["num_receive_threads"]
    receive_threads = []
    for i in range(app.config["num_receive_threads"]):
        receive_threads.append(threading.Thread(target=data_listener, args=(i,)))
        receive_threads[i].daemon = True
        receive_threads[i].start()

    bi_thread = threading.Thread(
        target=bad_input_listener, args=(app.config["num_receive_threads"],)
    )
    bi_thread.daemon = True
    bi_thread.start()

    thread2 = threading.Thread(target=TCP_stream)
    thread2.daemon = True
    thread2.start()

    metricsThread = threading.Thread(target=metric_thread)
    metricsThread.daemon = True
    metricsThread.start()

    httpServer2 = threading.Thread(target=http_server2)
    httpServer2.daemon = True
    httpServer2.start()

    bi_receive_watchdog = datetime.datetime.now()

    watchdogThread = threading.Thread(target=watchdog_thread)
    watchdogThread.daemon = True
    watchdogThread.start()

    # Endpoint parameters
    rfi_zeroing_url = "http://csBfs:54323/rfi-zeroing-toggle"
    rfi_zeroing_headers = {
        "content-type": "application/json",
        "Accept-Charset": "UTF-8",
    }

    rfi_zeroingThread = threading.Thread(target=rfi_zeroing)
    rfi_zeroingThread.daemon = True
    rfi_zeroingThread.start()

    while not EXIT:
        time.sleep(1)

    os._exit(1)
