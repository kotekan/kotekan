#!/usr/bin/env python3
"""Log the ADC rail (clip) fraction and band power at a FIXED cadence, fleet-wide.

    rail_watch.py watch [--out FILE] [--period 10] [--endpoints ...]
    rail_watch.py plot  [--out FILE] [--since 01:30] [--prn G9,E8] [--png x.png]

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

WHAT IT RECORDS, per instance per poll, one JSON line:
  * clip_hi / clip_lo  -- per-CHANNEL rail fraction (max and mean over monitored channels)
  * elem_clip          -- per-ELEMENT rail fraction (max/mean over 128), which localises a
                          clipping event to a feed rather than the band
  * power / elem_power -- the band power itself. THIS IS THE DISCRIMINATOR the reconstruction
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
DEF_OUT = "/home/kvand/gnss/fixtures/obs/rf_rail_%Y%m%d.jsonl"


def _stats(v):
    """max/mean of a numeric list, or None -- never 0.0, which would read as 'quiet'."""
    v = [x for x in (v or []) if isinstance(x, (int, float))]
    if not v:
        return None, None
    return max(v), sum(v) / len(v)


def poll_one(url, timeout=6.0):
    try:
        with urllib.request.urlopen(url + "/rf_stats", timeout=timeout) as r:
            d = json.load(r)
    except Exception as e:
        return {"url": url, "state": "unreachable", "err": str(e)[:80]}
    if not d.get("enabled", False):
        return {"url": url, "state": "off"}
    ch_hi, ch_hi_m = _stats(d.get("clip_hi"))
    ch_lo, _ = _stats(d.get("clip_lo"))
    el, el_m = _stats(d.get("elem_clip"))
    pw, pw_m = _stats(d.get("power"))
    ep, ep_m = _stats(d.get("elem_power"))
    return {"url": url, "state": "ok",
            "age_s": d.get("age_s"), "fpga_seq": d.get("fpga_seq"),
            "passes": d.get("passes"), "period_s": d.get("period_s"),
            "clip_hi_max": ch_hi, "clip_hi_mean": ch_hi_m, "clip_lo_max": ch_lo,
            "elem_clip_max": el, "elem_clip_mean": el_m,
            "power_max": pw, "power_mean": pw_m,
            "elem_power_max": ep, "elem_power_mean": ep_m,
            "n_chan": len(d.get("chans") or []), "n_elem": len(d.get("elem_clip") or [])}


def watch(args):
    eps = args.endpoints.split(",") if args.endpoints else DEF_EP
    print("rail_watch: %d endpoint(s), period %.1f s -> %s" % (len(eps), args.period, args.out),
          file=sys.stderr, flush=True)
    while True:
        t = time.time()
        rows = [poll_one(u, args.timeout) for u in eps]
        path = time.strftime(args.out, time.gmtime(t))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ok = [r for r in rows if r.get("state") == "ok"]
        rec = {"t": round(t, 2), "n_ok": len(ok), "n": len(rows), "inst": rows}
        if ok:   # the fleet-wide headline, so a plot needs no per-instance reduction
            rec["clip_worst"] = max(r["clip_hi_max"] for r in ok if r["clip_hi_max"] is not None)
            rec["elem_clip_worst"] = max(r["elem_clip_max"] for r in ok
                                         if r["elem_clip_max"] is not None)
            _p = [r["elem_power_mean"] for r in ok if r["elem_power_mean"] is not None]
            rec["elem_power_mean"] = sum(_p) / len(_p) if _p else None
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(max(1.0, args.period - (time.time() - t)))


def plot(args):
    import datetime as dt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    path = time.strftime(args.out, time.gmtime())
    rows = []
    with open(path) as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    if args.since:
        h, m = (args.since.split(":") + ["0"])[:2]
        cut = dt.datetime.now(dt.timezone.utc).replace(hour=int(h), minute=int(m),
                                                       second=0, microsecond=0).timestamp()
        rows = [r for r in rows if r["t"] >= cut]
    if not rows:
        sys.exit("no rows in %s for the requested window" % path)
    t = [dt.datetime.fromtimestamp(r["t"], dt.timezone.utc) for r in rows]

    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.patch.set_facecolor("white")
    ax[0].plot(t, [100 * (r.get("clip_worst") or 0) for r in rows], lw=1.6, color="#c65d21",
               label="worst per-channel clip (% of nibbles)")
    ax[0].plot(t, [100 * (r.get("elem_clip_worst") or 0) for r in rows], lw=1.2, alpha=.8,
               color="#d64550", label="worst per-element clip")
    ax[0].axhline(1.0, ls="--", lw=1, color="#888")
    ax[0].set_ylabel("rail / clip  (%)")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].grid(alpha=.25)
    ax[0].set_title("ADC rail fraction and band power, fixed %.0f s cadence"
                    % (rows[0].get("inst", [{}])[0].get("period_s") or 10), weight="bold")
    ax[1].plot(t, [r.get("elem_power_mean") for r in rows], lw=1.6, color="#4d9de0",
               label="mean per-element band power (arb.)")
    ax[1].set_ylabel("band power (arb.)")
    ax[1].legend(loc="upper left", fontsize=9)
    ax[1].grid(alpha=.25)
    ax[1].set_xlabel("UTC")
    for a in ax:
        a.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=dt.timezone.utc))
    plt.tight_layout()
    plt.savefig(args.png, dpi=125)
    print("wrote %s  (%d samples, %s..%s)"
          % (args.png, len(rows), t[0].strftime("%H:%M"), t[-1].strftime("%H:%M")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["watch", "plot"])
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--endpoints", default="")
    ap.add_argument("--period", type=float, default=10.0,
                    help="seconds; the node integrates over period_s (10 s), so faster buys "
                         "nothing but load")
    ap.add_argument("--timeout", type=float, default=6.0)
    ap.add_argument("--since", default="", help="plot: UTC HH:MM to start from")
    ap.add_argument("--png", default="/tmp/rf_rail.png")
    a = ap.parse_args()
    (watch if a.mode == "watch" else plot)(a)
