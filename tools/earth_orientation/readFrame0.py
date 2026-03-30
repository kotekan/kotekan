#!/usr/bin/env python3
"""
/*********************************************************************************
* Earth Orientation Parameter Tools 
* File: readFrame0.py
* Purpose: Contact running fpga_master and/or kotekan instances via REST and output their values for `frame0_nano`.
* Python Version: 3.12
* Dependencies: argparse, eop_utils
* Authors: Geoffrey Ryan
*********************************************************************************/
"""
import argparse
import eop_utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Read Frame0 Time",
        description="Read Frame0 Time from kotekan and fpga_master",
    )

    parser.add_argument(
        "--kotekan-host-port-list",
        nargs="*",
        default=[],
        help="List of hosts and ports running kotekan. Adjacent hosts and ports are paired, hosts (ports) without a port (host) pair use default_port (default_port)",
    )
    parser.add_argument(
        "-kh",
        "--kotekan_host",
        default="localhost",
        help="The default host to use. default: localhost",
    )
    parser.add_argument(
        "-kp",
        "--kotekan_port",
        default=12048,
        type=int,
        help="The default port to use. default: 12048",
    )
    parser.add_argument(
        "-fh",
        "--fpga_host",
        default=None,
        help="The default host to use. default: localhost",
    )
    parser.add_argument(
        "-fp",
        "--fpga_port",
        default=54321,
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

    args = parser.parse_args()

    print("Frame0 times")

    if args.fpga_host is not None:
        t0_ns = eop_utils.read_fpga_master_frame0_ns(
            args.fpga_host, args.fpga_port, args.timeout
        )
        print(
            "fpga_master {:d}".format(t0_ns),
            eop_utils.calc_astropy_time_from_unix_ns(t0_ns).isot,
            "{:s}:{:d}".format(args.fpga_host, args.fpga_port),
        )

    hostports = eop_utils.parse_hostport_list(
        args.kotekan_host_port_list, args.kotekan_host, args.kotekan_port
    )

    for host, port in hostports:
        if not eop_utils.is_kotekan_alive(host, port, args.timeout):
            print("kotekan {:s}:{:d} down".format(host, port))
            continue
        t0_ns = eop_utils.read_kotekan_frame0_ns(host, port, args.timeout)
        print(
            "kotekan     {:d}".format(t0_ns),
            eop_utils.calc_astropy_time_from_unix_ns(t0_ns).isot,
            "{:s}:{:d}".format(host, port),
        )
