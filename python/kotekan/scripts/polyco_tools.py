import click
import yaml
import json
import time
import re
from datetime import datetime
from tempfile import TemporaryFile
import requests
from subprocess import check_call
from os import path
from kotekan.pulsar_timing import Timespec, unix2mjd, mjd2unix, PolycoFile
from ch_util import ephemeris as ephem

COCO_URL = "http://csBfs:54323"
PARFILE_DIR = "/mnt/gong/parfiles/"
TEMPO_DIR = "/usr/local/tempo2/"


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


def parse_26m_sched(fname, n_ant=8):
    sched = []
    with open(fname, "r") as f:
        line = f.readline()
        while line != "":
            if line[0] != "#":
                line = [l.strip() for l in line.split()]
                if not len(line) == 71:
                    raise ValueError(
                        "File {} contains line with unexpected number of entries.".format(
                            fname
                        )
                    )
                # skip index
                _ = line.pop[0]
                name = re.match("^OBS(.+)$", line.pop(0))
                if name is None:
                    line = f.readline()
                    continue  # not a source observation
                name = name.group(1)
                # skip RA and dec
                for i in range(2 * n_ant):
                    _ = line.pop[0]
                # skip random characters
                for i in range(3):
                    _ = line.pop[0]
                lst = float(line.pop[0])
                dur = float(line.pop[0])
                sched.append({"name": name, "lst": lst, "duration": dur})
            line = f.readline()
    return sched


@click.group()
def cli():
    pass


@click.command()
@click.argument("unixtime", required=False, default=None, type=float)
def mjd(unixtime):
    """Convert unix time to MJD.
    Will print MJD now if no time is provided."""
    if unixtime is None:
        ts = Timespec(time.time())
    else:
        ts = Timespec(unixtime)
    click.echo(unix2mjd(ts))


@click.command()
@click.argument("unixtime", required=False, default=None, type=float)
def lst(unixtime):
    """Convert unix time to LST (in hours).
    Will print LST now if no time is provided."""
    if unixtime is None:
        unixtime = time.time()
    click.echo(ephem.unix_to_lsa(unixtime) * 24.0 / 360.0)


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
    help="Specify an end time to generate polyco / match segment with maximum overlap.",
    show_default="start_time + 1 day",
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
    help="(generate-polyco) Length of polyco segments in minutes.",
    show_default=True,
)
@click.option(
    "--ncoeff",
    type=int,
    default=12,
    help="(generate-polyco) Number of polyco coefficients to generate.",
    show_default=True,
)
@click.option(
    "--max_ha",
    type=float,
    default=12.0,
    help="(generate-polyco) Maximum hour angle for timing solution to span.",
    show_default=True,
)
@click.option(
    "--format",
    type=click.Choice(["yaml", "json", "dict"]),
    default="yaml",
    help="Config format to print out.",
    show_default=True,
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
    "--url", type=str, default=COCO_URL, help="URL for coco.", show_default=True,
)
@click.option(
    "--tempo-dir",
    type=str,
    default=TEMPO_DIR,
    help="TEMPO2 runtime directory",
    show_default=True,
)
@click.option(
    "--schedule",
    is_flag=True,
    help="Schedule enabling and disabling gating using specified start and end times. Will also enable writing of 26m_gated dataset.",
)
@click.option(
    "--time-spec",
    type=click.Choice(["MJD", "LST", "UNIX"]),
    help="How to interpret start and end times. If non-absolute LST is used, will reference to the time the command is run.",
    default="MJD",
    show_default=True,
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
    tempo_dir,
    schedule,
    time_spec,
):
    """Generate a gating polyco update from a parfile and send to kotekan.
    Required arguments are the path to the parfile and the start time for the polyco
    (enter 'now' to use current time minus 0.2 days).
    """
    url = url.strip("/")
    fname = path.abspath(fname)

    # parse times
    def ensure_mjd(t):
        if time_spec == "LST":
            return unix2mjd(Timespec(ephem.lsa_to_unix(t / 24.0 * 360.0, time.time())))
        elif time_spec == "UNIX":
            return unix2mjd(Timespec(t))
        else:
            return t

    if start_time == "now":
        start_time = unix2mjd(Timespec(time.time())) - 0.2
    else:
        start_time = ensure_mjd(float(start_time))
    if end_time is None:
        end_time = start_time + 1.0
    else:
        end_time = ensure_mjd(end_time)
    if end_time <= start_time:
        raise ValueError("Cannot use end time before start time")

    if not load_polyco:
        pfile = PolycoFile.generate(
            start_time, end_time, fname, dm, segment, ncoeff, max_ha, tempo_dir,
        )
        # Read DM and name from parfile since TEMPO mangles them
        parfile = parse_parfile(fname)
    else:
        pfile = PolycoFile(fname)
        parfile = {}

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
    if offset == 0.0 and "DPHASE" in parfile:
        if isinstance(parfile["DPHASE"], (list, tuple)):
            offset = float(parfile["DPHASE"][0])
        else:
            offset = float(parfile["DPHASE"])

    for p in pfile.polycos:
        p.phase_ref += offset
        # remove potentially large phase wrappings
        p.phase_ref = p.phase_ref % 1.0

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
                "curl {} -X POST -H \"Content-Type: application/json\" -d '{}' &&"
                'curl {} -X POST -H "Content-Type: application/json" '
                "-d '{{\"enabled\":true}}'"
            ).format(
                url + "/update-pulsar-gating", json.dumps(update), url + "/26m-gated"
            )
            update["enabled"] = False
            disable_cmd = (
                "curl {} -X POST -H \"Content-Type: application/json\" -d '{}' &&"
                'curl {} -X POST -H "Content-Type: application/json" '
                "-d '{{\"enabled\":false}}'"
            ).format(
                url + "/update-pulsar-gating", json.dumps(update), url + "/26m-gated"
            )
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
            r = requests.post(url + "/update-pulsar-gating", json=update)
            r.raise_for_status()
            print("Received: ({}) {}".format(r.status_code, r.content))


@click.command()
@click.argument("fname", type=str)
@click.option(
    "--url", type=str, default=COCO_URL, help="URL for coco.",
)
@click.option(
    "--tempo-dir", type=str, default=TEMPO_DIR, help="TEMPO2 runtime directory",
)
@click.option(
    "--parfile-dir",
    type=str,
    default=PARFILE_DIR,
    help="Directory containing pulsar parfiles.",
)
@click.option(
    "--reference", type=float, default=None, help="Reference time for observations.",
)
@click.pass_context
def import_schedule(ctx, fname, url, tempo_dir, parfile_dir, reference):
    """NOT IMPLEMENTED YET"""
    print("Not implemented yet. Aborting.")
    return

    # WIP
    if reference is None:
        cur_t = time.time()
    else:
        cur_t = ephem.ensure_unix(reference)
    sched = parse_26m_sched(fname)
    if len(sched) == 0:
        raise ValueError("Found no observations in schedule file {}.".format(fname))
    for obs in sched:
        pfile = path.join(parfile_dir, obs["name"])
        if not path.isfile(pfile):
            print(
                "Could not find parfile for {} in directory {}. Skipping observation.".format(
                    obs["name"], parfile_dir
                )
            )
            continue
        # Update time from end of previous observation
        cur_t = ephem.lsa_to_unix(obs["lst"] / 24.0 * 360.0, cur_t)
        end = cur_t + obs["duration"] * 3600.0
        ctx.invoke(
            update_polyco,
            pfile,
            unix2mjd(Timespec(cur_t)),
            url=url,
            tempo_dir=tempo_dir,
            end_time=unix2mjd(Timespec(end)),
            name=obs["name"],
            schedule=True,
        )
        cur_t = end


@click.command()
@click.option("--url", type=str, default=COCO_URL, help="URL for coco.")
def disable_gating(url):
    """Send an update to kotekan disabling the pulsar gating."""
    url = url.strip("/")
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
    r = requests.post(url + "/update-pulsar-gating", json=empty_config)
    r.raise_for_status()
    print("Received: ({}) {}".format(r.status_code, r.content))


cli.add_command(update_polyco)
cli.add_command(disable_gating)
cli.add_command(import_schedule)
cli.add_command(mjd)
cli.add_command(lst)

if __name__ == "__main__":
    cli()
