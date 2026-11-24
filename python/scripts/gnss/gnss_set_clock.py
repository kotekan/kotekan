#!/usr/bin/env python3
"""Correct the capture's time anchor when the host clock is wrong and no NTP is reachable.

    python3 gnss_set_clock.py --sky --yaml chains_l5.yaml MARKS.json ... [--restart]
    python3 gnss_set_clock.py --set SECONDS [--restart]

bladeRFInput adds the offset in --file (its clock_offset_file) to the system clock when it places
the first sample. --sky measures the anchor error from the CNAV time marks (gnss_skytime.py) and
moves the offset so the error becomes --target-ms: by default the +0.9 ms the capture shows with
NTP good (11-23), so a corrected run is labelled as a synchronised one. --set writes an offset
outright: 0 clears it, and a nonzero one fakes a wrong host clock for a test. Either takes
effect when kotekan restarts (--restart restarts gnss-kotekan, and the broker with it).
"""

import argparse
import json
import os
import subprocess
import sys

import gnss_skytime

FILE = "/home/lwlab/gnss/bladerf/clock_offset.json"


def read_offset(path):
    try:
        with open(path) as f:
            return float(json.load(f)["offset_s"])
    except FileNotFoundError:
        return 0.0


def write_offset(path, offset_s):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"offset_s": offset_s}, f)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "marks", nargs="*", help="nav_decode.py --timemarks-file outputs (--sky)"
    )
    ap.add_argument(
        "--sky", action="store_true", help="correct by the sky's measurement"
    )
    ap.add_argument("--set", type=float, metavar="SECONDS", help="write this offset")
    ap.add_argument("--yaml", help="the broker yaml holding the site (--sky)")
    ap.add_argument(
        "--site-json", help="or a gnss_fix.py --json fix as the site (--sky)"
    )
    ap.add_argument(
        "--decoded-eph",
        help="comma-separated decoded ephemeris files, not the BRDC (--sky)",
    )
    ap.add_argument("--target-ms", type=float, default=0.9)
    ap.add_argument("--file", default=FILE, help="bladeRFInput's clock_offset_file")
    ap.add_argument("--restart", action="store_true", help="restart gnss-kotekan")
    a = ap.parse_args(argv)
    old = read_offset(a.file)
    if a.set is not None:
        new = a.set
    elif a.sky:
        if not a.marks:
            ap.error("--sky needs the marks files")
        res, _ = gnss_skytime.run(
            a.marks,
            a.yaml,
            a.site_json,
            a.decoded_eph.split(",") if a.decoded_eph else None,
        )
        if len({r[1] for r in res}) < 2:
            print("fewer than two satellites' marks; nothing changed")
            return 1
        err = sorted(r[2] for r in res)[len(res) // 2]
        print(
            "anchor %+.6f s from %d marks of %d satellites"
            % (err, len(res), len({r[1] for r in res}))
        )
        new = old - (err - a.target_ms * 1e-3)
    else:
        ap.error("give --sky or --set")
    write_offset(a.file, new)
    print("%s: offset %+.6f s (was %+.6f s)" % (a.file, new, old))
    if a.restart:
        subprocess.run(["systemctl", "--user", "restart", "gnss-kotekan"], check=False)
        print("restarted gnss-kotekan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
