#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import time
import requests
import numpy as np
import generateEOPTable


def broadcast_kotekan_eop_table(host, port, eop_table, timeout, protocol="http://"):
    r"""
    Send a new EOP table to a running Kotekan instance.

    Parameters
    ----------
    base_url : String
        The URL at which to find the kotekan instance, no trailing "/". For
        instance "http://localhost".
    port : int
        The port at which to find the kotekan instance. For instance 12048.
    eop_table : List of dicts, each an EOP table entry
        The EOP table. A list of entries, each a dict with entries
        "time_inst_ns", "delta_UT1_inst", "x_pm", and "y_pm"
    timeout : float
        Timeout in seconds for the request.
    protocol : String, optional
        Prefix for the URL, for instance "http://" (the default)

    Returns
    -------
    time0_ns : int
        The UNIX timestamp in nanoseconds received from kotekan.

    Raises
    ------
    Exceptions from requests.
    """

    payload = {"earth_orientation_parameter_table": eop_table}

    resp = generateEOPTable.make_rest_post_request(host, port, "earth_rotation_data",
                                                   payload, timeout, protocol)

    return resp


def parse_broadcast_list(broadcast_list, default_host, default_port):

    hostports = []

    current_host = None

    for word in broadcast_list:
        isPort = False
        try:
            port = int(word)
            isPort = True
        except ValueError:
            isPort = False


        if isPort:
            if port < 0:
                raise ValueError("Bad port value:", port)
            host = current_host if current_host is not None else default_host
            hostports.append((host, port))
            current_host = None
        else:
            if current_host is None:
                current_host = word
            else:
                hostports.append((current_host, default_port))
                current_host = word

    if current_host is not None:
        hostports.append((current_host, default_port))

    if len(hostports) == 0:
        hostports.append((default_host, default_port))

    return hostports


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            prog='EOP Table Broadcaster',
            description='Send an Earth Orientation Parameter (EOP) table to kotekan')

    parser.add_argument("--broadcast-list", nargs="*", default=[])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=12048, type=int)
    parser.add_argument("--protocol", default="http://")
    parser.add_argument("--timeout", default=30.0, type=float)
    parser.add_argument("input-json-file", nargs=1)

    args = parser.parse_args()

    hostports = parse_broadcast_list(args.broadcast_list, args.host, args.port)

    print(hostports)

    print("Checking {:d} Kotekan instances are running.".format(len(hostports)))

    bad_instances = []

    for hostport in hostports:
        host, port = hostport
        if generateEOPTable.is_kotekan_alive(host, port, args.timeout):
            print("{:s}:{:d} OK".format(host, port))
        else:
            print("{:s}:{:d} down".format(host, port))
            bad_instances.append((host, port))

    if len(bad_instances) > 0:
        print("The locations:", bad_instances, "did not have Kotekan running.")
        print("Will not continue.")
        sys.exit()


    sys.exit()

    # Send table to Kotekan
    # broadcast_kotekan_eop_table(kotekan_host, kotekan_port, eop_table)
