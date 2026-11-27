#!/usr/bin/env python3
"""Decoded ephemerides of satellites in no broadcast product, as daily RINEX 3.04 nav files.

CHORD has no broadcast records for some satellites it sees (the GLONASS spare COSMOS 2433 on
slot 25): their own navigation messages are the only source. nav_decode decodes them here
(gnss_oddsat_pointing.py points the broker at them) into its ephemeris caches; this writes the
decoded records of the satellites named, one file per UTC day of their epochs
(gx10_decoded_YYYYDDD.rnx in --outbox, each rewritten through a temporary name), and copies
any file that changed to --dest with rsync, which renames into place there too.

GLONASS records are the immediate data of strings 1-4 as decoded. Two fields are not in those
strings: the frequency channel, which the almanac carries only for slots 1-24, comes from
--sats (the carrier the tracker measured); and RINEX 3.04's one health field holds the MSB of
B_n, as nav_decode reads it.

    gnss_oddsat_rinex.py --sats R25:8 --decoded ~/.cache/kotekan_gps/decoded --outbox DIR
        [--dest HOST:DIR --ssh-config FILE] [--every-s 60] [--once]
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import decoded_eph
import gnss_ephemeris as ge

KEEP_DAYS = 3


def parse_sats(spec):
    """{(sys, prn): frequency channel or None} from "R25:8"."""
    out = {}
    for item in spec.split(","):
        f = item.strip().split(":")
        if f[0]:
            out[(f[0][0].upper(), int(f[0][1:]))] = int(f[1]) if len(f) > 1 else None
    return out


def records(paths, sats):
    """{(sys, prn, toe_gpst): eph} of the decoded GLONASS records of the satellites named."""
    out = {}
    for path in decoded_eph.expand(paths):
        try:
            with open(path) as f:
                rows = json.load(f).get("eph", [])
        except (OSError, ValueError):
            continue
        for r in rows:
            key = (r.get("sys"), int(r.get("prn", -1)))
            if key in sats and r.get("signal") == "GLO_L2OF":
                e = r["eph"]
                out.setdefault(key + (round(e["toe_gpst"], 3),), e)
    return out


def _num(x):
    return "%19.12E" % x


def _utc(t_gpst):
    return t_gpst - ge.gpst_of_utc(0.0)


def glonass_lines(prn, e, k):
    """The RINEX 3.04 R record of a decoded GLONASS ephemeris: tb in UTC, -tau_n, +gamma_n,
    the frame time t_k in seconds of the UTC week, then X/Y/Z (km) with health, k and age.
    """
    tb = _utc(e["toe_gpst"])
    # t_k is the Moscow (UTC + 3 h) time of day of the frame start, shortly before tb.
    sod = (e["frame_time"] - 10800.0) % 86400.0
    tk = math.floor(tb / 86400.0) * 86400.0 + sod
    tk += 86400.0 * round((tb - tk) / 86400.0)
    tk_sow = (tk - 315964800.0) % 604800.0
    dt = datetime.fromtimestamp(tb, timezone.utc)
    head = "R%02d %04d %02d %02d %02d %02d %02d" % (
        prn,
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second,
    )
    lines = [head + _num(e["af0"]) + _num(e["af1"]) + _num(tk_sow)]
    for ax, extra in ((0, e["health"]), (1, k), (2, e.get("age", 0.0))):
        vals = (
            e["pos"][ax] / 1e3,
            e["vel"][ax] / 1e3,
            e["acc"][ax] / 1e3,
            float(extra),
        )
        lines.append("    " + "".join(_num(v) for v in vals))
    return "\n".join(lines) + "\n"


def header(sats):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d %H%M%S UTC")
    lines = [
        "%-60sRINEX VERSION / TYPE"
        % "     3.04           N: GNSS NAV DATA    M: MIXED",
        "%-20s%-20s%-20sPGM / RUN BY / DATE" % ("gnss_oddsat_rinex", "gx10", stamp),
        "%-60sCOMMENT" % "DECODED FROM THE SATELLITES' OWN NAV MESSAGES ON GX10",
        "%-60sCOMMENT" % "GLONASS HEALTH = MSB OF B_N (RINEX 3.04)",
    ]
    for (sysc, prn), k in sorted(sats.items()):
        if sysc == "R":
            lines.append(
                "%-60sCOMMENT"
                % ("R%02d FREQ NUMBER %d FROM THE MEASURED CARRIER" % (prn, k))
            )
    lines.append("%-60sEND OF HEADER" % "")
    return "\n".join(lines) + "\n"


def day_files(recs, sats, now):
    """{file name: text} for each UTC day of the last KEEP_DAYS holding a record."""
    by_day = {}
    for (sysc, prn, _), e in sorted(recs.items(), key=lambda kv: (kv[0][2], kv[0][1])):
        tb = _utc(e["toe_gpst"])
        if now - tb > KEEP_DAYS * 86400.0:
            continue
        name = datetime.fromtimestamp(tb, timezone.utc).strftime(
            "gx10_decoded_%Y%j.rnx"
        )
        by_day.setdefault(name, []).append(glonass_lines(prn, e, sats[(sysc, prn)]))
    return {name: header(sats) + "".join(body) for name, body in by_day.items()}


def write_changed(outbox, files):
    """Write each file whose records changed (the header's date aside); the names written."""
    os.makedirs(outbox, exist_ok=True)
    changed = []
    for name, text in files.items():
        path = os.path.join(outbox, name)
        try:
            with open(path) as f:
                old = f.read()
        except OSError:
            old = None
        if (
            old is not None
            and old.split("END OF HEADER")[-1] == text.split("END OF HEADER")[-1]
        ):
            continue
        tmp = os.path.join(outbox, "." + name + ".tmp")
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
        changed.append(name)
    return changed


def push(outbox, names, dest, ssh_config):
    ssh = "ssh -o BatchMode=yes" + (" -F " + ssh_config if ssh_config else "")
    cmd = ["rsync", "-a", "--chmod=F644", "-e", ssh] + [
        os.path.join(outbox, n) for n in names
    ]
    subprocess.run(cmd + [dest], check=True, timeout=120)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sats", required=True)
    ap.add_argument(
        "--decoded", required=True, help="decoded-ephemeris files or cache dirs"
    )
    ap.add_argument("--outbox", required=True)
    ap.add_argument("--dest", default="", help="rsync destination, HOST:DIR/")
    ap.add_argument("--ssh-config", default="")
    ap.add_argument("--every-s", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    sats = parse_sats(a.sats)
    if any(s == "R" and k is None for (s, _), k in sats.items()):
        raise SystemExit(
            "--sats: a GLONASS satellite needs its frequency channel (R25:8)"
        )
    paths = [p for p in a.decoded.split(",") if p]
    unsent = set()
    while True:
        try:
            now = time.time()
            changed = write_changed(
                a.outbox, day_files(records(paths, sats), sats, now)
            )
            if changed:
                print("wrote %s" % " ".join(changed), flush=True)
            unsent |= set(changed)
            if a.dest and unsent:
                push(a.outbox, sorted(unsent), a.dest, a.ssh_config)
                print("sent %s to %s" % (" ".join(sorted(unsent)), a.dest), flush=True)
                unsent.clear()
        except Exception as e:  # noqa: BLE001 -- a failed pass is retried with the next
            print("pass failed: %s" % e, flush=True)
        for path in glob.glob(os.path.join(a.outbox, "gx10_decoded_*.rnx")):
            if time.time() - os.path.getmtime(path) > (KEEP_DAYS + 1) * 86400.0:
                os.remove(path)
        if a.once:
            return 0
        time.sleep(a.every_s)


if __name__ == "__main__":
    sys.exit(main())
