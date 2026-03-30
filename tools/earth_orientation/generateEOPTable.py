#!/usr/bin/env python3
"""
/*********************************************************************************
* Earth Orientation Parameter Tools 
* File: generateEOPTable.py
* Purpose: Generate an Earth Orientation Parameter (EOP) Table suitable for broadcasting to Kotekan, compatible with the current frame0 time and any currently loaded EOP table.
* Python Version: 3.12
* Dependencies: argparse, eop_utils, astropy
* Authors: Geoffrey Ryan
*********************************************************************************/
"""
import argparse
import json
from pathlib import Path
import sys
from astropy.time import Time, TimeDelta
import astropy.utils.iers
import astropy.utils.data
import eop_utils

# Ensure Astropy will download new IERS data when needed
astropy.utils.iers.conf.auto_download = True
# Set the Astropy IERS Refresh time to the minimum allowed (10 days)
astropy.utils.iers.conf.auto_max_age = 10.0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="EOP Table Updater",
        description="Read, compute, and output an Earth Orientation Parameter (EOP) table for kotekan",
    )

    parser.add_argument(
        "--frame0-src",
        choices=["fpga_master", "kotekan", "manual"],
        default="manual",
        help="Where to obtain frame0 time",
    )
    parser.add_argument(
        "--frame0-ns",
        default=None,
        type=int,
        help="A manual frame0 time, only used if frame0-src is 'manual'",
    )
    parser.add_argument(
        "--current-time",
        default=None,
        help="The current time, or reference time to build EOP table for, as an Astropy-parsable string (e.g. ISOT). If unset, astropy.time.Time.now() is used.",
    )
    parser.add_argument(
        "-kh",
        "--kotekan-host",
        default="localhost",
        help="A host running Kotekan, for frame0 or a current EOP table, default 'localhost'.",
    )
    parser.add_argument(
        "-kp",
        "--kotekan-port",
        default=12048,
        type=int,
        help="The port to contact Kotekan on, default 12048.",
    )
    parser.add_argument(
        "-kprot",
        "--kotekan-protocol",
        default="http://",
        help="Protocol to issue GET requests to kotekan, default: 'http://'",
    )
    parser.add_argument(
        "-fh",
        "--fpga-master-host",
        default="localhost",
        help="A host running fpga_master, default: localhost",
    )
    parser.add_argument(
        "-fp",
        "--fpga-master-port",
        default=54321,
        type=int,
        help="A port to contact fpga_master on, default: 54321",
    )
    parser.add_argument(
        "-fprot",
        "--fpga-master-protocol",
        default="http://",
        help="Protocol to issue GET requests to fpga_master, default: 'http://'",
    )
    parser.add_argument(
        "--timeout",
        default=30.0,
        type=float,
        help="REST timeout in seconds, default 30.0.",
    )
    parser.add_argument(
        "-nb",
        "--num-intervals-before",
        default=2,
        type=int,
        help="Number of intervals before current interval to include in EOP table, at least 0, default 2.",
    )
    parser.add_argument(
        "-na",
        "--num-intervals-after",
        default=3,
        type=int,
        help="Number of intervals after current interval to include in EOP table, at least 0, default 3.",
    )
    parser.add_argument(
        "-dt",
        "--interval-length-days",
        default=1.0,
        type=float,
        help="Length of interval, e.g. time between EOP table entries in UTC days, default 1.0.",
    )
    parser.add_argument(
        "--force-iers-download",
        action="store_true",
        help="Download a fresh set of IERS data, otherwise use cached data (which is not more than 10 days out)",
    )
    parser.add_argument(
        "--enforce-continuity",
        choices=["yes", "no"],
        default="yes",
        help="If 'yes', read the current EOP table in Kotekan and merge it with the new table so all EOP will be interpolated continuously in time. Only new entries ahead of the current interval (plus cushion time) will be included in the new table, otherwise old entries will be used. Some old entries sufficiently in the past will be dropped.",
    )
    parser.add_argument(
        "--merge-cushion-dt",
        default="1hr",
        help="Cushion time to add to the current (reference) time when determining the current interval when merging eop tables to enforce continuity. An astropy-parsable TimeDelta string, default `1hr`",
    )
    parser.add_argument(
        "-o",
        "--out-json-file",
        default=None,
        help="File to write EOP table to in JSON format. File is suitable input for `broadcastEOPTable.py` and POSTing to Kotekan update endpoint.",
    )

    args = parser.parse_args()

    # Before anything else, set the astropy download options if necessary
    if args.force_iers_download:
        astropy.utils.data.clear_download_cache()

    # Extract the location of kotekan and fpga_master
    kotekan_protocol = args.kotekan_protocol
    kotekan_host = args.kotekan_host
    kotekan_port = args.kotekan_port

    fpga_master_protocol = args.fpga_master_protocol
    fpga_master_host = args.fpga_master_host
    fpga_master_port = args.fpga_master_port

    timeout = args.timeout

    # Determine how to set t0_ns.
    if args.frame0_src != "manual" and args.frame0_ns is not None:
        raise RuntimeError("Do not specify frame0_ns if frame0_src is not 'manual'.")

    if args.frame0_src == "manual" and args.frame0_ns is None:
        raise RuntimeError("If frame0_src is 'manual', must set frame0_ns")

    # Set t0_ns, this may make a REST call and could take time or fail.
    if args.frame0_src == "fpga_master":
        t0_ns = eop_utils.read_fpga_master_frame0_ns(
            fpga_master_host, fpga_master_port, timeout
        )
    elif args.frame0_src == "kotekan":
        if not eop_utils.is_kotekan_alive(kotekan_host, kotekan_port, timeout):
            print(
                "ERROR: Kotekan could not be reached on {}:{}. Exiting.".format(
                    kotekan_host, kotekan_port
                )
            )
            sys.exit()
        t0_ns = eop_utils.read_kotekan_frame0_ns(kotekan_host, kotekan_port, timeout)
    elif args.frame0_src == "manual":
        t0_ns = args.frame0_ns
    else:
        # Should be unneccessary, but just in case.
        raise ValueError("Unknown frame0_src: {:s}".format(args.frame0_src))

    # Get a Time from the t0_ns
    t0 = eop_utils.calc_astropy_time_from_unix_ns(t0_ns)
    print("frame0_ns is: {0:d} ns   (from {1:s})".format(t0_ns, args.frame0_src))
    print("frame0 time is: ", t0.utc.isot, "(UTC)")

    # Set reference (likely current) time.
    if args.current_time is not None:
        t_ref = Time(args.current_time, precision=9)
    else:
        t_ref = Time.now()
        t_ref.precision = 9

    print("Current time is:", t_ref.utc.isot, "(UTC)")

    # Extract parameters for building table entry times
    num_intervals_before = args.num_intervals_before
    num_intervals_after = args.num_intervals_after
    interval_length_days = args.interval_length_days

    if num_intervals_before < 0:
        raise ValueError(
            "num_intervals_before must be positive. Recieved: {:d}".format(
                num_intervals_before
            )
        )
    if num_intervals_after < 0:
        raise ValueError(
            "num_intervals_after must be positive. Recieved: {:d}".format(
                num_intervals_after
            )
        )

    # Build the array of times to generate EOP entries for
    ts = eop_utils.build_time_array(
        t_ref,
        num_intervals_before,
        num_intervals_after,
        interval_length_days,
        snap_to_grid=True,
    )
    print("t_ref (mjd):", t_ref.mjd)
    print("times in table:", ts.iso)
    print("times in table (mjd):", ts.mjd)

    # Build the table, use astropy's automatic IERS table
    iers = astropy.utils.iers.IERS_Auto.open()
    # This is a list of BareEOP objects
    new_eop_table = eop_utils.build_EOP_table(ts, t0_ns, iers)
    iers.close()

    if args.enforce_continuity == "yes":
        enforce_continuity = True
    elif args.enforce_continuity == "no":
        enforce_continuity = False
    else:
        raise ValueError("enforce_continuity must be yes or no")

    # Set the final table, possibly by blending with the current table to ensure
    # continuity of EOP values
    if enforce_continuity:
        merge_cushion_dt = TimeDelta(args.merge_cushion_dt, scale="tai")

        # Get the current table loaded into Kotekan
        if not eop_utils.is_kotekan_alive(kotekan_host, kotekan_port, timeout):
            print(
                "ERROR: Kotekan could not be reached on {}:{}. Exiting.".format(
                    kotekan_host, kotekan_port
                )
            )
            sys.exit()

        # This is a table of EOP objects
        current_eop_table = eop_utils.read_kotekan_eop_table(
            kotekan_host, kotekan_port, timeout, kotekan_protocol
        )

        # Merge the tables
        eop_table = eop_utils.merge_eop_tables(
            current_eop_table,
            new_eop_table,
            t_ref,
            t0_ns,
            num_intervals_before,
            merge_cushion_dt,
        )
    else:
        eop_table = new_eop_table

    print(
        "The final EOP table contains {:d} entries from {} to {}".format(
            len(eop_table),
            eop_utils.calc_astropy_time_from_inst_ns(
                eop_table[0]["t_inst_ns"], t0_ns
            ).utc.isot,
            eop_utils.calc_astropy_time_from_inst_ns(
                eop_table[-1]["t_inst_ns"], t0_ns
            ).utc.isot,
        )
    )

    # Print the table
    eop_utils.print_eop_table(eop_table)

    # Write the table to a file if asked.
    if args.out_json_file is not None and len(args.out_json_file) > 0:
        eop_utils.output_json_eop_table(eop_table, args.out_json_file)
