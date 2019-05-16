# Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

import click
import yaml
import json
import time
from kotekan.pulsar_timing import Timespec, unix2mjd, PolycoFile


@click.group()
def cli():
    pass


@click.command()
@click.argument('unixtime', required=False, default=None, type=float)
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
@click.option("--generate-polyco", is_flag=True,
              help="If enabled, input file must be a pulsar parfile.")
@click.option("--end-time", type=float, default=None,
              help="Specify an end time to generate polyco / match segment with maximum overlap")
@click.option("--dm", type=float, default=None,
              help="The DM in cm^-3/pc. If not specified will try and get from file.")
@click.option("--segment", type=float, default=300,
              help="(generate-polyco) Length of polyco segments in minutes (default 300).")
@click.option("--ncoeff", type=int, default=12,
              help="(generate-polyco) Number of polyco coefficients to generate.")
@click.option("--max_ha", type=float, default=12.,
              help="(generate-polyco) Maximum hour angle for timing solution to span.")
@click.option("--format", type=click.Choice(['yaml', 'json', 'dict']), default='json',
              help="Config format to print out.")
@click.option("--offset", type=float, default=0.,
              help="Add an offset (s) to the polyco phase solution.")
def polyco_config(fname, start_time, generate_polyco, end_time, dm, segment, ncoeff, max_ha,
                  format, offset):
    if generate_polyco:
        if end_time is None:
            end = start_time + 1.
        else:
            end = end_time
        pfile = PolycoFile.generate(start_time, end, fname, dm, segment, ncoeff, max_ha)
    else:
        pfile = PolycoFile(fname)

    if pfile is None or len(pfile.polycos) == 0:
        print("\nCould not generate/read polyco file.")
        return

    if dm is not None:
        pfile.dm = dm

    if offset != 0.:
        for p in pfile.polycos:
            p.phase_ref += offset * p.f0

    print("\nConfig update:\n")
    formatter = yaml.dump if format == "yaml" else json.dumps if format == "json" else repr
    print(formatter(pfile.config_block(start_time, end_time)))


if __name__ == '__main__':
    cli.add_command(polyco_config)
    cli.add_command(mjd)

    cli()
