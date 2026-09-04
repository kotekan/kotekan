#!/usr/bin/env python3
"""THE SUB-SECOND CODE EXCURSION, at the resolution it actually happens (task #51 / G23).

On 2026-08-15 G23 showed prompt power holding at 25-80x the fleet noise median while E and L
swapped dominance by 12-35x -- the code error crossing the full +-0.5-chip E/L spacing, twice,
inside six seconds, sometimes between consecutive 1 s samples. Every instrument pointed at it so
far is too slow to see its shape: the fleet loop averages 2-4 windows (84-168 ms) and the REST
polls sampled it at 1 Hz. This reads the telemetry stream itself, where the answer has been all
along -- 4 records per frame, 23.84 frames/s, so 95.4 records/s per instance, each carrying
E/P/L PER CHANNEL.

⚠️ RUN IT ON THE GATHER HOST. GnssTelemGather serves 127.0.0.1 and broadcasts to every
connected client, so this attaches alongside the broker without disturbing it.

THE THREE QUESTIONS, in the order that narrows fastest:

  1. WHAT IS THE TIMESCALE? Per RECORD (10.5 ms), per FRAME (41.9 ms), or slower? A step that
     is exactly one record wide is a different animal from a 200 ms swing.

  2. IS IT COMMON ACROSS INSTANCES? Instances are `freq_id mod 8` routing of ONE signal path --
     one PFB, one set of raw samples -- so a real code-phase move must appear in ALL of them at
     the SAME record. If it does not, the fault is downstream of the split and the word
     "per-instance" is hiding a per-CHANNEL effect. ⚠️ Never conclude "instance N is bad";
     conclude "the channels instance N holds are bad", and print them.

  3. IS IT COMMON ACROSS CHANNELS? A true delay is a phase RAMP across frequency, not a
     wholesale collapse. If E/L swap on every channel together, the code phase moved. If it is
     confined to part of the band, it is not a delay at all.

    ./excursion.py --seconds 20 --prn 23
"""
import argparse
import collections
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker import telem  # noqa: E402


def per_record(frames, prn):
    """[(win, rec, inst, e, p, l, nch, {fid: (e,p,l)})] -- ONE record, no averaging anywhere.

    Same expression as combdll.instance_taps, with the record loop left UNCOLLAPSED: that mean
    over records is exactly what hides a 10 ms step.
    """
    out = []
    for (win, inst), f in sorted(frames.items()):
        for r in range(f.n_rec):
            if not f.has_record(r):
                continue
            cmb = f.comb_epl(r, prn)
            if not cmb:
                continue
            gE = gP = gL = 0j
            wE = wP = wL = 0.0
            per_ch = {}
            for fid, E, P, L, (eE, eP, eL) in cmb:
                gE += E * eE
                gP += P * eP
                gL += L * eL
                wE += eE
                wP += eP
                wL += eL
                per_ch[fid] = (abs(E * eE) ** 2, abs(P * eP) ** 2, abs(L * eL) ** 2)
            if wP <= 0.0:
                continue
            out.append((win, r, inst,
                        (abs(gE) / wE) ** 2 if wE > 0 else 0.0,
                        (abs(gP) / wP) ** 2,
                        (abs(gL) / wL) ** 2 if wL > 0 else 0.0,
                        len(cmb), per_ch))
    return out


def disc_q(e, p, l):
    s = e + l
    return ((e - l) / s if s > 0 else 0.0), (2.0 * p / s if s > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--prn", type=int, default=23)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--show", type=int, default=60, help="records to print around the worst step")
    a = ap.parse_args()

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    time.sleep(a.seconds)
    wins = cl.windows(a.chain, lag=1)
    frames = {}
    for w in wins:
        for inst, f in cl.frame_set(a.chain, w).items():
            frames[(w, inst)] = f
    cl.stop()
    if not frames:
        raise SystemExit("no frames for chain %s -- is the gather serving on %s?" % (a.chain, a.gather))

    rows = per_record(frames, a.prn)
    if not rows:
        raise SystemExit("PRN %d has no live comb in %d frames" % (a.prn, len(frames)))
    insts = sorted({r[2] for r in rows})
    keys = sorted({(r[0], r[1]) for r in rows})
    print("%d windows x %d instances, %d record-samples, PRN %d, %d instances: %s"
          % (len(wins), len(insts), len(rows), a.prn, len(insts), ", ".join(insts)))
    print("record cadence 10.486 ms (95.4/s); frame = 4 records = 41.94 ms\n")

    # ---- (1) THE FLEET SERIES, PER RECORD -------------------------------------------------
    by_key = collections.defaultdict(list)
    for w, r, inst, e, p, l, nch, ch in rows:
        by_key[(w, r)].append((inst, e, p, l, ch))
    series = []
    for k in keys:
        v = by_key[k]
        E = sum(x[1] for x in v)
        P = sum(x[2] for x in v)
        L = sum(x[3] for x in v)
        d, q = disc_q(E, P, L)
        series.append((k, d, q, P, len(v)))

    # the worst single-record jump in disc: the thing that defines the timescale
    jumps = [(abs(series[i + 1][1] - series[i][1]), i) for i in range(len(series) - 1)]
    jumps.sort(reverse=True)
    big, at = jumps[0]
    print("BIGGEST ONE-RECORD disc STEP: %.3f, between record %d and %d (10.5 ms apart)"
          % (big, at, at + 1))
    dd = sorted(j for j, _ in jumps)
    print("  one-record |ddisc|: median %.3f  p90 %.3f  max %.3f" %
          (dd[len(dd) // 2], dd[int(0.9 * len(dd))], dd[-1]))
    print("  fleet disc range over the capture: %+.3f .. %+.3f\n"
          % (min(s[1] for s in series), max(s[1] for s in series)))

    lo = max(0, at - a.show // 2)
    hi = min(len(series), at + a.show // 2)
    print("  rec   win/slot      disc      q      p_pow    n_inst")
    for i in range(lo, hi):
        (w, r), d, q, P, n = series[i]
        mark = "  <<<" if i in (at, at + 1) else ""
        print("  %4d  %d/%d  %+8.3f %7.2f  %9.3e  %2d%s" % (i, w % 100000, r, d, q, P, n, mark))

    # ---- (2) IS THE STEP COMMON ACROSS INSTANCES? -----------------------------------------
    print("\nAT THE STEP -- every instance's own disc, same record:")
    for idx in (at, at + 1):
        (w, r) = series[idx][0]
        print("  record %d (win %d slot %d):" % (idx, w % 100000, r))
        for inst, e, p, l, ch in sorted(by_key[(w, r)]):
            d, q = disc_q(e, p, l)
            print("    %-9s disc %+7.3f  q %6.2f  e %9.3e p %9.3e l %9.3e" % (inst, d, q, e, p, l))

    # ---- (3) IS IT COMMON ACROSS CHANNELS? ------------------------------------------------
    print("\nAT THE STEP -- per CHANNEL, summed over instances (a delay is a RAMP, not a collapse):")
    for idx in (at, at + 1):
        (w, r) = series[idx][0]
        acc = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
        for inst, e, p, l, ch in by_key[(w, r)]:
            for fid, (ce, cp, cl_) in ch.items():
                acc[fid][0] += ce
                acc[fid][1] += cp
                acc[fid][2] += cl_
        fids = sorted(acc)
        print("  record %d: %d channels %d..%d" % (idx, len(fids), fids[0], fids[-1]))
        line = "    "
        for fid in fids:
            d, _ = disc_q(*acc[fid])
            line += "%+.2f " % d
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
