#!/usr/bin/env python3
"""
/*********************************************************************************
* Earth Orientation Parameter Tools 
* File: broadcastEOPTable.py
* Purpose: Send the given EOP table to running kotekan instance(s).
* Python Version: 3.12
* Dependencies: argparse, eop_utils
* Authors: Geoffrey Ryan
*********************************************************************************/
"""
import argparse
import json
from pathlib import Path
import sys
import eop_utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="EOP Table Broadcaster",
        description="Send an Earth Orientation Parameter (EOP) update table (BareEOP) to Kotekan",
    )

    parser.add_argument(
        "--broadcast-list",
        nargs="*",
        default=[],
        help="List of hosts and ports running kotekan. Adjacent hosts and ports are paired, hosts (ports) without a port (host) pair use default_port (default_port)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="The default host to use. default: localhost",
    )
    parser.add_argument(
        "--port",
        default=12048,
        type=int,
        help="The default port to use. default: 12048",
    )
    parser.add_argument(
        "--protocol",
        default="http://",
        help="Protocol to use for REST requests, default: 'http://'",
    )
    parser.add_argument(
        "--timeout",
        default=30.0,
        type=float,
        help="REST timeout in seconds, default: 30.0",
    )
    parser.add_argument(
        "--eop-post-endpoint",
        default="earth_rotation_data",
        help="Endpoint to receive EOP table, no leading '/', default: 'earth_rotation_data'",
    )
    parser.add_argument(
        "input_json_file",
        nargs=1,
        help='Input file name, a JSON file containing the EOP update table to send: An object with the member "earth_orientation_paramer_table" whose value is a list of BareEOP objects.',
    )

    args = parser.parse_args()

    filepath = Path(args.input_json_file[0])
    print("Reading eop_table from", filepath)
    with open(filepath, "r") as f:
        eop_table = json.load(f)

    print(json.dumps(eop_table, indent=4))

    hostports = eop_utils.parse_hostport_list(args.broadcast_list, args.host, args.port)

    print("Checking {:d} Kotekan instances are running.".format(len(hostports)))

    bad_instances = []

    full_endpoint = "/" + args.eop_post_endpoint

    for host, port in hostports:
        if eop_utils.is_kotekan_alive(host, port, args.timeout, args.protocol):
            endpoints = eop_utils.get_kotekan_endpoints(
                host, port, args.timeout, args.protocol
            )
            if (
                len(endpoints) > 0
                and isinstance(endpoints, dict)
                and "POST" in endpoints.keys()
                and full_endpoint in endpoints["POST"]
            ):
                print("{:s}:{:d} OK".format(host, port))
            else:
                print(
                    "{:s}:{:d} is not accepting POST to {:s}".format(
                        host, port, full_endpoint
                    )
                )
                bad_instances.append((host, port))

        else:
            print("{:s}:{:d} down".format(host, port))
            bad_instances.append((host, port))

    if len(bad_instances) > 0:
        print("The locations:", bad_instances, "failed.")
        print("Will not continue.")
        sys.exit()

    # Send table to Kotekan
    for host, port in hostports:
        print("Sending EOP update to {}:{}...".format(host, port), end="")
        eop_utils.broadcast_kotekan_eop_table(
            host, port, args.eop_post_endpoint, eop_table, args.timeout, args.protocol
        )
        print(" OK")

    print("All updates sent")
