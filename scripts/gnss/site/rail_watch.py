#!/usr/bin/env python3
"""Log the ADC rail (clip) fraction and band power at a fixed cadence, fleet-wide.

    rail_watch.py record [--out FILE] [--endpoints ...]   # per channel, every pass (1 Hz)

Bright-transit clipping: a satellite passes near boresight, total band power rises, the 4+4b
quantiser starts railing, and every other satellite is desensitised by small-signal
suppression while the bright one is barely touched.

The broker logs `RF PATH` every ~5 min and `RF CLIPPING` only above 1%, so its sampling
density follows the value being measured: enough to show the shape of an event, useless for
a lag or a cross-correlation. A fixed cadence is the point; do not make this event-triggered.

What it records, per instance per pass, one JSON line (the exact keys are in record()):
  * clip_hi / clip_lo  -- per-channel rail fraction
  * elem_clip          -- per-element rail fraction (128), which localises a clipping event to
                          a feed rather than the band
  * power              -- the band power. If clipping rises because total power rises, the
                          cause is a source or an interferer; if power is flat while clipping
                          rises, it is a gain or quantiser-scaling change.
  * age_s, fpga_seq, passes, enabled -- so a frozen monitor shows as frozen: a stale endpoint
    can answer 200 with plausible numbers. `enabled` false is recorded as off, never as
    zeros, and unreachable is a third state.

Standalone: no broker change, no node change, nothing in the control path.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

CONF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "config"
)
REC_OUT = "/home/kvand/gnss/fixtures/obs/rf_chan_%Y%m%d.jsonl"


def fleet_taps():
    """Both GPUs' search taps on every fleet host (config/gnss_fleet_chord.yaml nodes:, read by
    config/gnss_fleet.py, which needs PyYAML: apt's python3-yaml under /usr/bin/python3)."""
    sys.path.insert(0, CONF)
    try:
        import gnss_fleet

        fleet = gnss_fleet.Fleet()
    except Exception as e:
        sys.exit(
            "no fleet host list from %s/gnss_fleet.py (%s: %s); name the taps with --endpoints"
            % (CONF, type(e).__name__, e)
        )
    return [
        "%s/gnss%d_srch_tap" % (fleet.rest_url(n), g)
        for n in fleet.hosts
        for g in (0, 1)
    ]


def _r(v, n=4):
    """Compact float for the per-channel arrays: 4 significant figures, exact 0 stays "0"."""
    return [0 if x == 0 else float("%.*g" % (n, x)) for x in v]


def record(args):
    """Every rf_stats pass, per channel, one JSON line per (instance, pass).

    The node computes one pass per band_power_period_s (1 s in the manifest) from one frame, and
    `passes` counts them. Polling at twice that rate and writing only when `passes` moves keeps
    every pass exactly once whatever the phase between the two clocks; a pass missed because
    the poll came late is visible as a jump in `pass`, never as a silent gap.

    A line:  {"t", "inst": "<node>.<gpu>", "pass", "seq" (fpga_seq of the measured frame),
              "age" (s since the pass), "hi"/"lo" (fraction of nibbles at +7 / -8, per
              channel), "pw" (mean |x|^2 per channel)}
    plus "f" (the channels' freq_ids, from the frame) on an instance's first line of each file
    and whenever it changes, "ec" (per-element clip, 128) every 10th pass, and a {"state"} line
    whenever an instance goes unreachable / off / back. freq_ids are carried, never assumed:
    local comb index means a different frequency on every node.
    """
    from concurrent.futures import ThreadPoolExecutor

    eps = args.endpoints.split(",") if args.endpoints else fleet_taps()
    name = {
        u: u.split("//")[1].split(":")[0] + "." + u.rsplit("gnss", 1)[1][0] for u in eps
    }
    last_pass, last_f, last_state, cur_path = {}, {}, {}, [None]
    print(
        "rail_watch record: %d endpoint(s) -> %s" % (len(eps), args.out),
        file=sys.stderr,
        flush=True,
    )

    def get(u):
        try:
            with urllib.request.urlopen(u + "/rf_stats", timeout=args.timeout) as r:
                return u, json.load(r)
        except Exception as e:
            return u, {"_err": str(e)[:80]}

    with ThreadPoolExecutor(len(eps)) as pool:
        while True:
            t0 = time.time()
            path = time.strftime(args.out, time.gmtime(t0))
            if path != cur_path[0]:  # new file: every instance restates its freq_ids
                cur_path[0] = path
                last_f.clear()
                os.makedirs(os.path.dirname(path), exist_ok=True)
            lines = []
            for u, d in pool.map(get, eps):
                inst = name[u]
                state = (
                    "unreachable"
                    if "_err" in d
                    else "off"
                    if not d.get("enabled")
                    else "ok"
                )
                if state != last_state.get(inst):
                    last_state[inst] = state
                    lines.append(
                        {
                            "t": round(t0, 3),
                            "inst": inst,
                            "state": state,
                            **({"err": d["_err"]} if "_err" in d else {}),
                        }
                    )
                if state != "ok" or d.get("passes") == last_pass.get(inst):
                    continue
                if not d.get("clip_hi"):  # enabled but no frame measured yet
                    continue
                last_pass[inst] = d["passes"]
                rec = {
                    "t": round(t0, 3),
                    "inst": inst,
                    "pass": d["passes"],
                    "seq": d.get("fpga_seq"),
                    "age": round(d.get("age_s", -1), 3),
                    "hi": _r(d["clip_hi"]),
                    "lo": _r(d["clip_lo"]),
                    "pw": _r(d["power"]),
                }
                f = d.get("freq_ids") or []
                if f != last_f.get(inst):
                    last_f[inst] = f
                    rec["f"] = f
                if d["passes"] % 10 == 0 and d.get("elem_clip"):
                    rec["ec"] = _r(d["elem_clip"], 3)
                lines.append(rec)
            if lines:
                with open(path, "a") as fh:
                    fh.write(
                        "".join(
                            json.dumps(x, separators=(",", ":")) + "\n" for x in lines
                        )
                    )
            time.sleep(max(0.05, args.poll - (time.time() - t0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["record"])
    ap.add_argument(
        "--out",
        default=REC_OUT,
        help="JSONL output path, expanded with strftime in UTC on every poll, so the default "
        "starts a new file each UTC day; default %s" % REC_OUT.replace("%", "%%"),
    )
    ap.add_argument(
        "--poll",
        type=float,
        default=0.5,
        help="seconds between rf_stats polls (default %(default)s); keep it at most half the "
        "nodes' band_power_period_s (the generator's --rf-stats-period-s) so that every "
        "pass is read",
    )
    ap.add_argument(
        "--endpoints",
        default="",
        help="comma list of srch_tap URLs; default both GPUs' taps on every fleet host "
        "(config/gnss_fleet.py hosts)",
    )
    ap.add_argument("--timeout", type=float, default=6.0)
    record(ap.parse_args())
