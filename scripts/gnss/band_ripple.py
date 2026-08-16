#!/usr/bin/env python3
"""THE RIPPLE ON TOP OF THE RAMP: an ECHO, or the mod-8 INTERLEAVE showing through?

KV spotted a clean sinusoid riding the linear phase ramp. There are two candidates and they
predict DIFFERENT PERIODS, so the period decides it.

PHYSICAL -- a delayed echo (multipath, or a cable/impedance reflection). An echo of relative
amplitude `a` at delay `tau_e` multiplies the band by (1 + a e^{-2i pi f tau_e}), which is a
ripple of period 1/tau_e in frequency. ⚠️ ITS SIGNATURE IS THAT AMPLITUDE AND PHASE RIPPLE
TOGETHER, at the SAME period and in QUADRATURE, with depths tied by the same `a`
(phase ~ a sin(theta), amplitude ~ 1 + a cos(theta)). Phase-only ripple is NOT an echo.

ARTIFACT -- the transport grouping. An instance is `freq_id mod 8` and the two GPUs on a node
split that residue at stride 16, so a constant phase per despread PROCESS is EXACTLY PERIODIC
in frequency with period 8 channels (1.5625 MHz) and/or 16 (3.1250 MHz). Plotted against
frequency it is a sawtooth by construction -- no physics required, and it is the bug already
established by the dfid=1 vs dfid=16 coherence split.

    period 8 chan  = 1.5625 MHz  ->  delay-domain peak at  640.0 ns
    period 16 chan = 3.1250 MHz  ->  delay-domain peak at  320.0 ns
    an echo        = whatever it is, and NOT sitting on that lattice

METHOD. Take the reference-free per-channel complex gain (the leading eigenvector of the
diagonal-deleted channel covariance -- see band_phase_ramp.py), remove the fitted linear ramp,
and transform the RESIDUAL to delay. Do it for the phase residual AND for the amplitude
residual separately, because only an echo puts a matching peak in both.

⚠️ THE CHANNEL COMB IS INCOMPLETE, so this transform has its own lobes. The probes are the
null for "is there a peak at all", and the 320/640/1280 ns marks are printed so a lattice hit
is visible rather than inferred.

    ./band_ripple.py --chain gps_l5 --plot /tmp/ripple.png          # ON cf06
"""
import argparse
import cmath
import json
import math
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker import telem  # noqa: E402
sys.path.insert(0, HERE)
from band_phase_ramp import (collect, eigen_phase, fit_ramp,  # noqa: E402
                             CHAN_HZ, CHIP_HZ, FID_REF)


def delay_spectrum(fids, vals, lo_ns=0.0, hi_ns=2600.0, step_ns=2.0):
    """[(tau_ns, |X|)] -- transform an irregularly sampled series over frequency to delay.

    `vals` is complex per channel and is transformed as-is: sum_k vals_k exp(+2i pi f_k tau).
    Mean-removed first so a DC offset cannot dominate the low-delay end.
    """
    mu = sum(vals) / len(vals)
    xs = [(f - FID_REF) * CHAN_HZ for f in fids]
    ys = [v - mu for v in vals]
    out = []
    n = int((hi_ns - lo_ns) / step_ns)
    for i in range(n + 1):
        tau = (lo_ns + i * step_ns) * 1e-9
        s = 0j
        for x, y in zip(xs, ys):
            s += y * cmath.exp(2j * math.pi * x * tau)
        out.append(((lo_ns + i * step_ns), abs(s) / len(ys)))
    return out


def top_peaks(spec, n=5, sep_ns=60.0):
    pk = sorted(spec, key=lambda t: -t[1])
    tops, seen = [], []
    for t, v in pk:
        if all(abs(t - s) > sep_ns for s in seen):
            tops.append((t, v))
            seen.append(t)
        if len(tops) >= n:
            break
    return tops


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--broker", default="http://127.0.0.1:12060")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--seconds", type=float, default=50.0)
    ap.add_argument("--windows", type=int, default=72)
    ap.add_argument("--min-duty", type=float, default=0.5)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()

    with urllib.request.urlopen("%s/%s/get_status" % (a.broker.rstrip("/"), a.chain),
                                timeout=10) as r:
        rows = json.loads(r.read().decode())
    probes = {int(x["prn"]) for x in rows if x.get("noise_probe")}
    held = {int(x["prn"]) for x in rows
            if not x.get("noise_probe") and x.get("cn0_prompt_db") is not None
            and (x.get("cn0_prompt_duty") or 0) >= a.min_duty}
    cn0 = {int(x["prn"]): x.get("cn0_prompt_db") for x in rows}

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    time.sleep(a.seconds)
    wins = cl.windows(a.chain, lag=1)[-a.windows:]
    ser, owner = collect(cl, a.chain, wins, held | probes)
    cl.stop()
    if not ser:
        raise SystemExit("no records -- run this ON cf06 (--gather is HOST:PORT)")

    lat = {"8 chan (mod-8 node interleave)": 1e9 / (8 * CHAN_HZ),
           "16 chan (GPU split within a node)": 1e9 / (16 * CHAN_HZ),
           "4 chan": 1e9 / (4 * CHAN_HZ)}
    print("\n  THE GROUPING LATTICE (what an artifact must land on):")
    for k, v in sorted(lat.items(), key=lambda t: t[1]):
        print("    %-38s -> %7.1f ns" % (k, v))
    print()

    res = {}
    print("  %-5s %-7s | %-34s | %-24s"
          % ("prn", "cn0", "PHASE-residual delay peaks (ns)", "AMPLITUDE-residual"))
    for prn in sorted(ser):
        fids, recs = ser[prn]
        if len(recs) < 60:
            continue
        v, _l, _lo = eigen_phase(fids, recs)
        tau, rms, _ph, _un = fit_ramp(fids, v)
        # residual complex gain with the fitted ramp taken out
        r = []
        for f, x in zip(fids, v):
            ramp = -2 * math.pi * (f - FID_REF) * CHAN_HZ * tau
            r.append(abs(x) * cmath.exp(1j * (cmath.phase(x) - ramp)))
        mean_amp = sum(abs(z) for z in r) / len(r)
        # PHASE-only residual: unit modulus, so amplitude structure cannot leak in
        ph_only = [z / abs(z) if z != 0j else 0j for z in r]
        # AMPLITUDE-only residual: real, so phase structure cannot leak in
        am_only = [complex(abs(z) / mean_amp, 0.0) for z in r]
        sp = delay_spectrum(fids, ph_only)
        sa = delay_spectrum(fids, am_only)
        res[prn] = (fids, r, tau, sp, sa)
        print("  %-5d %-7s | %-34s | %-24s %s"
              % (prn, ("%.1f" % cn0[prn]) if cn0.get(prn) is not None else "--",
                 " ".join("%.0f(%.2f)" % (t, v_) for t, v_ in top_peaks(sp, 3)),
                 " ".join("%.0f(%.2f)" % (t, v_) for t, v_ in top_peaks(sa, 3)),
                 "probe" if prn in probes else ""))

    print("\n  READ IT LIKE THIS:")
    print("   * a peak ON the lattice (320 / 640 ns) and NOT matched in amplitude")
    print("     => the transport grouping showing through. Not physics.")
    print("   * the SAME peak in phase AND amplitude, OFF the lattice")
    print("     => a real echo at that delay; ripple period = 1/tau.")
    print("   * the probes say what a peak height means -- they have no signal at all.")

    sats = [p for p in res if p not in probes]
    if sats:
        print("\n  LATTICE TEST -- residual power at the grouping delays vs the median:")
        print("    %-5s %-10s %-10s %-10s %-10s"
              % ("prn", "320ns", "640ns", "median", "320/med"))
        for prn in sorted(sats):
            sp = res[prn][3]
            def at(t):
                return min(sp, key=lambda z: abs(z[0] - t))[1]
            med = sorted(v for _t, v in sp)[len(sp) // 2]
            print("    %-5d %-10.3f %-10.3f %-10.3f %-10.2f"
                  % (prn, at(320.0), at(640.0), med, at(320.0) / med if med else float("nan")))

    if a.plot and res:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                                        # noqa: BLE001
            print("  (no plot: %s)" % e)
            return 0
        show = sorted(sats or res)
        fig, axes = plt.subplots(len(show), 2, figsize=(13, 2.7 * len(show)), squeeze=False)
        for row, prn in enumerate(show):
            fids, r, tau, sp, sa = res[prn]
            ax = axes[row][0]
            fr = [f * CHAN_HZ / 1e6 for f in fids]
            ax.plot(fr, [cmath.phase(z) for z in r], "o", ms=4)
            ax.set_title("PRN %d -- phase residual after the linear ramp (%.1f ns)"
                         % (prn, tau * 1e9), fontsize=9)
            ax.set_ylabel("rad")
            ax.grid(alpha=0.3)
            ax = axes[row][1]
            ax.plot([t for t, _v in sp], [v for _t, v in sp], "-", lw=1.0, label="phase")
            ax.plot([t for t, _v in sa], [v for _t, v in sa], "-", lw=1.0, label="amplitude")
            for t, lab in ((320.0, "16 chan"), (640.0, "8 chan"), (1280.0, "4 chan")):
                ax.axvline(t, color="r", ls="--", lw=0.8)
                ax.text(t, ax.get_ylim()[1] * 0.95, lab, fontsize=6, color="r", rotation=90)
            ax.set_title("PRN %d -- residual transformed to DELAY" % prn, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        axes[-1][0].set_xlabel("RF frequency (MHz)")
        axes[-1][1].set_xlabel("delay (ns)")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)
        print("\n  wrote %s" % a.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
