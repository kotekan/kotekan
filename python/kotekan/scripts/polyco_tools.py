# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility


import click
import yaml
import json
import time
from datetime import datetime
from tempfile import TemporaryFile
import requests
from subprocess import check_call
from os import path
from kotekan.pulsar_timing import Timespec, unix2mjd, mjd2unix, PolycoFile


def parse_parfile(fname):
    """Read the parameters from a pulsar parfile."""
    param = {}
    with open(fname, "r") as f:
        line = f.readline()
        while line != "":
            if line[0] != "#":
                line = [l.strip() for l in line.split()]
                if len(line) > 1:
                    param[line[0]] = line[1:] if len(line[1:]) > 1 else line[1]
            line = f.readline()
    return param


@click.group()
def cli():
    pass


@click.command()
@click.argument("unixtime", required=False, default=None, type=float)
def mjd(unixtime):
    """ Convert unix time to MJD.
        Will print MJD now if no time is provided. """
    if unixtime is None:
        ts = Timespec(time.time())
    else:
        ts = Timespec(unixtime)
    click.echo(unix2mjd(ts))


@click.command()
@click.argument("fname", type=str)
@click.argument("start-time", type=float)
@click.option(
    "--load-polyco",
    is_flag=True,
    help="If enabled, input file must be a pregenerated polyco file.",
)
@click.option(
    "--end-time",
    type=float,
    default=None,
    help="Specify an end time to generate polyco / match segment with maximum overlap. Defaults to start time + 1.",
)
@click.option(
    "--dm",
    type=float,
    default=None,
    help="The DM in cm^-3/pc. If not specified will try and get from file.",
)
@click.option(
    "--name",
    type=str,
    default=None,
    help="Pulsar name to use. If not specified will try and get from file.",
)
@click.option(
    "--width",
    type=float,
    default=None,
    help="Pulse width to use. If not specified will try and get from file.",
)
@click.option(
    "--segment",
    type=float,
    default=300,
    help="(generate-polyco) Length of polyco segments in minutes (default 300).",
)
@click.option(
    "--ncoeff",
    type=int,
    default=12,
    help="(generate-polyco) Number of polyco coefficients to generate.",
)
@click.option(
    "--max_ha",
    type=float,
    default=12.0,
    help="(generate-polyco) Maximum hour angle for timing solution to span.",
)
@click.option(
    "--format",
    type=click.Choice(["yaml", "json", "dict"]),
    default="yaml",
    help="Config format to print out.",
)
@click.option(
    "--offset",
    type=float,
    default=0.0,
    help="Add an offset (s) to the polyco phase solution.",
)
@click.option("--send-update", is_flag=True, help="Send the update to kotekan.")
@click.option(
    "--no-confirm",
    is_flag=True,
    help="Don't ask for confirmation before sending update.",
)
@click.option(
    "--url",
    type=str,
    default="http://csBfs:54323/update-pulsar-gating",
    help="URL of kotekan master pulsar gating endpoint.",
)
@click.option(
    "--recv-url",
    type=str,
    default="http://recv2:12048/updatable_config/26m_gated",
    help="URL of 26m_gated toggle endpoint on receiver node.",
)
@click.option(
    "--tempo-dir",
    type=str,
    default="/usr/local/tempo2/",
    help="TEMPO2 runtime directory",
)
@click.option(
    "--schedule",
    is_flag=True,
    help="Schedule enabling and disabling gating using specified start and end times. Will also enable writing of 26m_gated dataset.",
)
def update_polyco(
    fname,
    start_time,
    load_polyco,
    end_time,
    dm,
    name,
    width,
    segment,
    ncoeff,
    max_ha,
    format,
    offset,
    send_update,
    no_confirm,
    url,
    recv_url,
    tempo_dir,
    schedule,
):
    """Generate a gating polyco update from a parfile and send to kotekan.
    Required arguments are the path to the parfile and the start time for the polyco
    (enter 'now' to use current time minus 0.2 days).
    """
    fname = path.abspath(fname)
    if not load_polyco:
        if start_time == "now":
            start_time = unix2mjd(Timespec(time.time())) - 0.2
        else:
            start_time = float(start_time)
        if end_time is None:
            end_time = start_time + 1.0
        pfile = PolycoFile.generate(
            start_time, end_time, fname, dm, segment, ncoeff, max_ha, tempo_dir
        )
        # Read DM and name from parfile since TEMPO mangles them
        parfile = parse_parfile(fname)
    else:
        pfile = PolycoFile(fname)
        parfile = {}

    # ensure time is a float
    start_time = float(start_time)

    if pfile is None or len(pfile.polycos) == 0:
        print("\nCould not generate/read polyco file.")
        return

    # parse parameters from args/file
    if dm is None and "DM" in parfile:
        if isinstance(parfile["DM"], (list, tuple)):
            dm = parfile["DM"][0]
        else:
            dm = parfile["DM"]
        pfile.dm = float(dm)
    elif dm is not None:
        pfile.dm = dm
    if name is None and "PSRJ" in parfile:
        pfile.name = parfile["PSRJ"]
    elif name is not None:
        pfile.name = name
    if width is None and "W10" in parfile:
        if isinstance(parfile["W10"], (list, tuple)):
            pfile.width = float(parfile["W10"][0])
        else:
            pfile.width = float(parfile["W10"])
    elif width is not None:
        pfile.width = width
    else:
        print("No pulse width provided and can't find in parfile. Aborting.")
        return

    if offset != 0.0:
        for p in pfile.polycos:
            p.phase_ref += offset * p.f0

    update = pfile.config_block(start_time, end_time)
    print("\nConfig update:\n")
    formatter = (
        yaml.dump if format == "yaml" else json.dumps if format == "json" else repr
    )
    print(formatter(update))

    if send_update or schedule:
        if not no_confirm:
            confirm = input(
                "{} this update? (y/N) ".format("Schedule" if schedule else "Send")
            )
            if confirm.lower().strip() in ["yes", "y"]:
                pass
            else:
                return
        if schedule:
            if mjd2unix(start_time).tv_sec <= time.time():
                raise ValueError("Cannot schedule a gating start time in the past.")
            start_str = datetime.fromtimestamp(mjd2unix(start_time).tv_sec).strftime(
                "%H:%M %Y-%m-%d"
            )
            end_str = datetime.fromtimestamp(mjd2unix(end_time).tv_sec).strftime(
                "%H:%M %Y-%m-%d"
            )
            enable_cmd = (
                'curl {} -X POST -H "Content-Type: application/json" -d \'{}\' &&'
                'curl {} -X POST -H "Content-Type: application/json" '
                "-d '{{\"enabled\":true}}'"
            ).format(url, json.dumps(update), recv_url)
            update["enabled"] = False
            disable_cmd = (
                'curl {} -X POST -H "Content-Type: application/json" -d \'{}\' &&'
                'curl {} -X POST -H "Content-Type: application/json" '
                "-d '{{\"enabled\":false}}'"
            ).format(url, json.dumps(update), recv_url)
            print("Scheduling gating between {} and {}.".format(start_str, end_str))
            with TemporaryFile(mode="w+") as tmpf:
                tmpf.write(enable_cmd)
                tmpf.seek(0)
                tmpf.flush()
                check_call(["at", start_str], stdin=tmpf)
            with TemporaryFile(mode="w+") as tmpf:
                tmpf.write(disable_cmd)
                tmpf.seek(0)
                tmpf.flush()
                check_call(["at", end_str], stdin=tmpf)
        else:
            print("Sending update to {}...".format(url))
            r = requests.post(url, json=update)
            r.raise_for_status()
            print("Received: ({}) {}".format(r.status_code, r.content))


@click.command()
@click.option(
    "--url",
    type=str,
    default="http://csBfs:54323/update-pulsar-gating",
    help="URL of kotekan master pulsar gating endpoint.",
)
def disable_gating(url):
    """Send an update to kotekan disabling the pulsar gating."""
    empty_config = {
        "coeff": [[0.0]],
        "t_ref": [0.0],
        "phase_ref": [0.0],
        "rot_freq": 0.0,
        "dm": 0.0,
        "segment": 0.0,
        "enabled": False,
        "pulsar_name": "none",
        "pulse_width": 0.0,
    }
    print("Sending update to {}...".format(url))
    r = requests.post(url, json=empty_config)
    r.raise_for_status()
    print("Received: ({}) {}".format(r.status_code, r.content))


cli.add_command(update_polyco)
cli.add_command(disable_gating)
cli.add_command(mjd)

if __name__ == "__main__":
    cli()
