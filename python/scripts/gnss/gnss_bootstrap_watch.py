#!/usr/bin/env python3
"""Run gnss_bootstrap when the receiver has lost the whole sky.

    python3 gnss_bootstrap_watch.py [--quiet-s 300] [--poll-s 30] [-- gnss_bootstrap options]

Every --poll-s, count the satellites locked on each bootstrap band's combiner (gnss_bootstrap
.BANDS: the deep SNR it calls locked). None locked on any band for --quiet-s, while kotekan
answers and the broker has run that long, means the broker's model is wrong about the sky: its
clock, its site or its orbits, as after a long power-off at a new site with no network. Then run
gnss_bootstrap --apply, which finds all three from the sky alone.

After a run the next waits --backoff-s, doubling with each run up to --backoff-max-s, until a
lock is seen: an antenna fault or a dead front end looks the same as a wrong model, and there a
bootstrap finds nothing. Nothing runs while the broker's hold file exists (a bootstrap, or a
person, holds the broker).
"""

import argparse
import os
import subprocess
import sys
import time

import gnss_bootstrap as gb


def locked():
    """Satellites locked per band kotekan answers for, or None when it answers for none."""
    out = {}
    for band, c in gb.BANDS.items():
        try:
            rows = gb.get("/%s/get_status" % c["combine"])
        except Exception:
            continue
        rows = rows if isinstance(rows, list) else rows.get("prns", [])
        out[band] = sum(
            1
            for r in rows
            if isinstance(r, dict) and r.get("deep_snr", 0.0) >= c["lock"]
        )
    return out or None


def broker_up_s():
    """How long the broker has been active (s), or None when it is not."""
    r = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            "gnss-broker",
            "-p",
            "ActiveState",
            "-p",
            "ActiveEnterTimestampMonotonic",
        ],
        capture_output=True,
        text=True,
    )
    v = dict(l.split("=", 1) for l in r.stdout.split() if "=" in l)
    if v.get("ActiveState") != "active":
        return None
    return time.monotonic() - int(v.get("ActiveEnterTimestampMonotonic", 0)) / 1e6


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    extra = []
    if "--" in argv:
        extra = argv[argv.index("--") + 1 :]
        argv = argv[: argv.index("--")]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quiet-s", type=float, default=300.0)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--backoff-s", type=float, default=1800.0)
    ap.add_argument("--backoff-max-s", type=float, default=4 * 3600.0)
    ap.add_argument(
        "--hold", default=gb.B + "/broker/HOLD", help="the broker's hold file"
    )
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    backoff = a.backoff_s
    # the last time anything was locked, or the watch could not tell
    t_lock = time.monotonic()
    t_next = 0.0  # no run before this
    gb.log(
        "watching %s: bootstrap after %.0f s with none locked"
        % (", ".join(gb.BANDS), a.quiet_s)
    )
    while True:
        now = time.monotonic()
        n = locked()
        up = broker_up_s()
        if n is None or up is None or os.path.exists(a.hold):
            t_lock = now
        elif sum(n.values()):
            t_lock = now
            backoff = a.backoff_s
        elif now - t_lock >= a.quiet_s and up >= a.quiet_s and now >= t_next:
            gb.log("none locked for %.0f s; bootstrapping" % (now - t_lock))
            rc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(here, "gnss_bootstrap.py"),
                    "--apply",
                    *extra,
                ]
            ).returncode
            gb.log("bootstrap exited %d; the next waits %.0f s" % (rc, backoff))
            t_next = time.monotonic() + backoff
            backoff = min(2.0 * backoff, a.backoff_max_s)
            t_lock = time.monotonic()
        time.sleep(a.poll_s)


if __name__ == "__main__":
    sys.exit(main())
