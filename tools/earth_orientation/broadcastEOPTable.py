#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import eop_utils


def parse_broadcast_list(broadcast_list, default_host, default_port):
    r"""
    Parse the broadcast list (list of hosts and ports) into a list of
    hostport pairs.  

    The broadcast list is a list of hosts (strings) and ports (positive
    integers). If a port appears after a host, the two form a host-port pair.
    If a host or port appear alone, the default host or default port is used
    to make a pair.

    Ex.
    [ host1 port1 host2 host3 port2 port3 ]
    becomes
    [ (host1, port1), (host2, default_port), (host3, port2),
     (default_host, port3) ]

    Parameters
    ----------
    broadcast_list : List of str
        The list of hosts and ports
    default_host : str
        Default host to use
    default_port : int
        Default port to use
    
    Returns
    -------
    hostports : List of (str, int)
        The list of host-port pairs

    Raises
    ------
    ValueError if a negative port is received
    """

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
        prog="EOP Table Broadcaster",
        description="Send an Earth Orientation Parameter (EOP) update table to Kotekan",
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
        help='Input file name, a JSON file containing the EOP update table to send: An object with the member "earth_orientation_paramer_table" whose value is a list of EOP update objects.',
    )

    args = parser.parse_args()

    filepath = Path(args.input_json_file[0])
    print("Reading eop_table from", filepath)
    with open(filepath, "r") as f:
        eop_table = json.load(f)

    print(json.dumps(eop_table, indent=4))

    hostports = parse_broadcast_list(args.broadcast_list, args.host, args.port)

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
        eop_utils.broadcast_kotekan_eop_table(
            host, port, args.eop_post_endpoint, eop_table, args.timeout, args.protocol
        )
