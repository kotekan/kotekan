#!/usr/bin/env python3
"""#61: is the COMMANDED carrier following the satellite, or standing still?

    scripts/gnss/cmd_vs_resid.py [--chain gal_e5a] [--prn N] [--seconds 40]

NO SPECULATION NEEDED -- WE SHIP THE COMMAND. Every record carries what we told the tracker to
despread with:
    REC_DOPPLER  (slot 1)   the reported/commanded Doppler for that record, Hz
    REC_CPHASE   (slot 15)  the commanded carrier-phase INCREMENT since this PRN's previous
                            record, cycles -> /dt is the commanded frequency, exactly
and the prompt's phase is the residual AFTER that command, so its rate is (f_true - f_cmd).

THE IDENTITY THIS CHECKS. If the command were perfect the residual rate would be zero. If the
command's FREQUENCY is static while the sky moves at dop_rate, the residual rate grows as
dop_rate*t -- which is the smooth ~0.3 Hz/s drift rate_vs_time.py measured. So:

    d(f_cmd)/dt + d(residual)/dt  ==  d(f_true)/dt  ==  dop_rate

Measuring both terms separately says which one is missing. It is a decomposition of our own
numbers, not a model fit, and it cannot be argued with.
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python", "scripts", "gnss"))

from gnss_broker import telem  # noqa: E402
from coh_time import HOPS_PER_SEC, _fit_rate, fleet_series, ladder  # noqa: E402


def lsq_slope(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return (sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs))) / sxx) if sxx else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11061)
    ap.add_argument("--chain", default="gal_e5a")
    ap.add_argument("--prn", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--sub", type=int, default=32)
    ap.add_argument("--min-coh", type=float, default=0.45)
    a = ap.parse_args()

    need = int(a.seconds / 0.0419)
    c = telem.TelemClient(host=a.host, port=a.port, depth=need + 32, retry_s=1.0,
                          chains={a.chain}).start()
    t0 = time.time()
    while time.time() - t0 < a.seconds + 30 and len(c.windows(a.chain, lag=1)) < need:
        time.sleep(1.0)
    wins = c.windows(a.chain, lag=1)[-need:]
    if len(wins) < 32:
        c.stop()
        sys.exit("only %d windows" % len(wins))

    prn = a.prn
    if prn is None:
        best, bamp = None, -1.0
        for p in sorted({q for w in wins[-16:] for f in c.frame_set(a.chain, w).values()
                         for q in f.prns()}):
            s, _ = fleet_series(c, a.chain, wins[-16:], p)
            if s:
                m = statistics.mean([abs(v) for _, v in s])
                if m > bamp:
                    best, bamp = p, m
        prn = best

    # ONE reference instance for the command: REC_DOPPLER / REC_CPHASE are per-record model
    # quantities, identical across instances by construction (one broker seed), so averaging
    # them over instances would hide a disagreement rather than reveal one. Take one, and
    # report the cross-instance spread as a check that they really are identical.
    cmd = {}
    spread = []
    for w in wins:
        fs = c.frame_set(a.chain, w)
        for r in range(4):
            vals_d, vals_c = [], []
            for inst in sorted(fs):
                f = fs[inst]
                if r >= f.n_rec or not f.has_record(r):
                    continue
                row = f.row(r, prn)
                if row is None or row[telem.REC_P_ENERGY] <= 0:
                    continue
                vals_d.append(row[telem.REC_DOPPLER])
                vals_c.append(row[telem.REC_CPHASE])
            if vals_d:
                h = list(fs.values())[0].hop(r)
                cmd[h] = (vals_d[0], vals_c[0])
                if len(vals_d) > 1:
                    spread.append(max(vals_d) - min(vals_d))

    series, per = fleet_series(c, a.chain, wins, prn)
    c.stop()
    if len(series) < 4 * a.sub:
        sys.exit("PRN %s: only %d records" % (prn, len(series)))
    hops = [h for h, _ in series]
    vals = [v for _, v in series]
    dt = (hops[1] - hops[0]) / HOPS_PER_SEC

    print("chain %s  PRN %d  %d instances  %d records  span %.1f s"
          % (a.chain, prn, len(per), len(vals), len(vals) * dt))
    if spread:
        print("commanded Doppler spread ACROSS instances: max %.6f Hz (should be ~0 -- one seed)"
              % max(spread))
    print()

    # -- term 1: what the COMMAND did -------------------------------------------------------
    t_cmd = [(h - hops[0]) / HOPS_PER_SEC for h in hops if h in cmd]
    d_cmd = [cmd[h][0] for h in hops if h in cmd]
    # REC_CPHASE is the commanded phase INCREMENT in cycles -> /dt is the commanded frequency,
    # independent of whatever slot 1 reports. Two views of the same command; if they disagree,
    # that is itself the finding.
    f_from_phase = [cmd[h][1] / dt for h in hops if h in cmd]
    if len(t_cmd) < 8:
        sys.exit("not enough commanded samples")
    slope_cmd = lsq_slope(t_cmd, d_cmd)
    slope_phase = lsq_slope(t_cmd, f_from_phase)
    print("COMMAND (slot 1, reported Doppler): %.3f -> %.3f Hz, slope %+.4f Hz/s"
          % (d_cmd[0], d_cmd[-1], slope_cmd))
    print("COMMAND (slot 15, phase increment): %.3f -> %.3f Hz, slope %+.4f Hz/s"
          % (f_from_phase[0], f_from_phase[-1], slope_phase))
    print()

    # -- term 2: what the RESIDUAL did ------------------------------------------------------
    step = max(1, a.sub // 2)
    ts, rates = [], []
    for k in range(0, len(vals) - a.sub + 1, step):
        r, _, dero = _fit_rate(vals[k:k + a.sub], dt)
        if ladder(dero, [a.sub]).get(a.sub, 0.0) >= a.min_coh:
            ts.append(k * dt)
            rates.append(r)
    if len(rates) < 8:
        sys.exit("only %d coherent sub-windows" % len(rates))
    slope_res = lsq_slope(ts, rates)
    print("RESIDUAL (prompt phase rate): %+.3f -> %+.3f Hz over %.1f s, slope %+.4f Hz/s"
          % (rates[0], rates[-1], ts[-1] - ts[0], slope_res))
    print("  (%d coherent sub-windows of %.2f s)" % (len(rates), a.sub * dt))
    print()

    print("DECOMPOSITION   d(f_true)/dt = d(f_cmd)/dt + d(residual)/dt")
    print("                             = %+.4f      + %+.4f      = %+.4f Hz/s"
          % (slope_cmd, slope_res, slope_cmd + slope_res))
    print()
    if abs(slope_cmd) < 0.25 * abs(slope_res):
        print("=> THE COMMAND IS ESSENTIALLY STATIC while the residual carries the whole")
        print("   Doppler rate. We are despreading against a frequency that does not follow")
        print("   the satellite, and the deep fold then has to absorb the entire ramp.")
        print("   Check WHY: is the broker posting a rate at all, and does the tracker apply")
        print("   one between seeds? (gnssRecord.hpp slot 1 is a per-record model value, so a")
        print("   static slot 1 means the model itself is not advancing.)")
    elif abs(slope_cmd + slope_res) < 0.25 * abs(slope_cmd):
        print("=> The command IS following the satellite and the residual cancels it: the two")
        print("   slopes are equal and opposite, so the ramp is a COMMAND/TRUTH mismatch in")
        print("   the OFFSET, not a missing rate.")
    else:
        print("=> Both terms are significant. Neither 'the command is static' nor 'the command")
        print("   is right' explains it; get the true dop_rate from BRDC and compare all three.")


if __name__ == "__main__":
    sys.exit(main())
