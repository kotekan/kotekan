#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import generateEOPTable


def broadcast_kotekan_eop_table(host, port, eop_table, eop_endpoint, timeout, protocol="http://"):
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

    resp = generateEOPTable.make_rest_post_request(host, port, eop_endpoint,
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
    parser.add_argument("--eop-post-endpoint", default="earth_rotation_data")
    parser.add_argument("input-json-file", nargs=1)

    args = parser.parse_args()

    hostports = parse_broadcast_list(args.broadcast_list, args.host, args.port)

    print("Checking {:d} Kotekan instances are running.".format(len(hostports)))

    bad_instances = []

    full_endpoint = "/" + args.eop_post_endpoint

    for host, port in hostports:
        if generateEOPTable.is_kotekan_alive(host, port, args.timeout, args.protocol):
            endpoints = generateEOPTable.get_kotekan_endpoints(host, port, args.timeout,
                                                               args.protocol)
            if len(endpoints) > 0 and isinstance(endpoints, dict) and 'POST' in endpoints.keys() and full_endpoint in endpoints['POST']:
                print("{:s}:{:d} OK".format(host, port))
            else:
                print("{:s}:{:d} is not accepting POST to {:s}".format(host, port, full_endpoint))
                bad_instances.append((host, port))

        else:
            print("{:s}:{:d} down".format(host, port))
            bad_instances.append((host, port))

    if len(bad_instances) > 0:
        print("The locations:", bad_instances, "failed.")
        print("Will not continue.")
        sys.exit()

    filepath = Path(args.input_json_file)
    print("Reading eop_table from", filepath)
    with open(args.input_json_file, "r") as f:
        eop_table = json.load(f)

    print(eop_table)

    sys.exit()
    
    for host, port in hostports:
        broadcast_kotekan_eop_table(host, port, eop_table, args.eop_post_endpoint,
                                    args.protocol)
    # Send table to Kotekan
