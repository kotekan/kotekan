#!/usr/bin/env python3
"""Log the ADC rail (clip) fraction and band power at a FIXED cadence, fleet-wide.

    rail_watch.py record [--out FILE] [--endpoints ...]   # per channel, every pass (1 Hz)

WHY THIS EXISTS. Bright-transit clipping events happen a few times a day: a satellite
passes near boresight, total band power rises, the 4+4b quantiser starts railing, and every
OTHER satellite is desensitised by small-signal suppression while the bright one is barely
touched. Measured 2026-08-22 00:38-01:05 UTC -- clip 0.085% -> 5.57%, G9's q RISING to 5.4
(r = +0.86 vs clip) while E8 collapsed 3.0 -> 1.0 and sat there.

⚠️ THAT EVENT HAD TO BE RECONSTRUCTED FROM LOG LINES, and it is why this exists. The only
history was the broker's `RF PATH` line at ~5 min cadence plus the `⚠️ RF CLIPPING` warnings,
which fire only ABOVE 1%. So the sampling density was correlated with the value being
measured -- fine for showing the shape, useless for a lag or a cross-correlation. A fixed
cadence is the whole point; do not make this event-triggered.

WHAT IT RECORDS, per instance per pass, one JSON line (the exact keys are in record()):
  * clip_hi / clip_lo  -- per-CHANNEL rail fraction
  * elem_clip          -- per-ELEMENT rail fraction (128), which localises a clipping event to
                          a feed rather than the band
  * power              -- the band power itself. THIS IS THE DISCRIMINATOR the reconstruction
                          could not settle: if clipping rises because total power rises, the
                          cause is a source or an interferer; if power is flat while clip
                          rises, it is a gain or quantiser-scaling change.
  * age_s, fpga_seq, passes, enabled -- so a FROZEN monitor is visible as frozen. A stale
    endpoint answering 200 with plausible numbers is this project's most repeated trap
    (#70/#87, the wedged gather, the stranded capture windows); `enabled` false is recorded
    as off, never as zeros, and unreachable is a third state again.

Deliberately standalone: no broker change, no node change, nothing in the control path.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

NODES = ["cx19", "cx27", "cx42", "cx43", "cx44", "cx51"]
DEF_EP = [f"http://{n}:12048/gnss{g}_srch_tap" for n in NODES for g in (0, 1)]
REC_OUT = "/home/kvand/gnss/fixtures/obs/rf_chan_%Y%m%d.jsonl"


def _r(v, n=4):
    """Compact float for the per-channel arrays: 4 significant figures, exact 0 stays "0"."""
    return [0 if x == 0 else float("%.*g" % (n, x)) for x in v]


def record(args):
    """EVERY rf_stats pass, per channel, one JSON line per (instance, pass).

    The node computes one pass per band_power_period_s (1 s in the manifest) from ONE frame, and
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

    eps = args.endpoints.split(",") if args.endpoints else DEF_EP
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
        help="strftime path; default %s" % REC_OUT.replace("%", "%%"),
    )
    ap.add_argument(
        "--poll",
        type=float,
        default=0.5,
        help="record: seconds between polls; half the node's pass period, so no "
        "pass is skipped",
    )
    ap.add_argument("--endpoints", default="")
    ap.add_argument("--timeout", type=float, default=6.0)
    record(ap.parse_args())
