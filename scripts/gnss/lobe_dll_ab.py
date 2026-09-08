#!/usr/bin/env python3
"""INSTANCE-COHERENT vs LOBE-COHERENT fleet DLL, on the SAME gathered frames.

The A/B that motivated the lobe combine now shipped in gnssFleetDll.hpp / combdll.py (B is
what those form; A is what they formed before). Kept as the on-sky instrument for the same
question: XCOH and P(B)/P(A)*n_inst say whether the senders' phi0 puts them on one reference.

    scripts/gnss/lobe_dll_ab.py [--chain gps_l5] [--windows 32] [--probes 6,11,21]

The fleet discriminator is formed from the comb -- Early/Prompt/Late per CHANNEL, per record,
per sender. Two reductions of those numbers to one (E, P, L) per PRN are compared here:

  A  INSTANCE-COHERENT (what shipped before the lobe combine): per record, each sender sums its own channels coherently,
     |SUM_c G_c|^2 / (SUM_c E_c)^2, and the fleet adds those POWERS across senders. A sender is
     an arbitrary `freq_id mod 8` grouping of channels, so this makes the coherence unit a
     transport artefact: 7 channels at 3.125 MHz stride -> a 3.27-chip grating-lobe comb in
     the correlation response, and a noise floor 12x higher than the band can give.

  B  LOBE-COHERENT (what ships): per record, ONE complex sum over EVERY channel of the lobe, each sender's
     columns first rotated by exp(+i*phi0) (REC_PHI0 -- the per-sender NCO accumulator that
     the assembler applied, whose zero is arbitrary per sender; gnssRecord.hpp), then the
     power. No sender appears anywhere in the arithmetic.

  B0 B without the phi0 rotation -- the CONTROL. If B0 ~ B the rotation is a no-op and the
     senders are already on one reference; if B0 << B the rotation is load-bearing.

Read-only: a second consumer of the gather's broker stream. Changes nothing.

READING IT. Same records, same PRNs, so the comparison is paired. On a tracked satellite B's q
should be >= A's (same mainlobe width, no grating lobes); on a probe both must sit at q ~ 1.
The per-record disc scatter (sd) is the noise the code loop integrates; at the C/N0 the fleet
sees it is NOT reliably lower for B (the E and L taps carry the same sky noise either way,
and the grating comb A removes is a bias, not a scatter) -- read sd as a like-for-like check,
not as B's selling point. XCOH is the mean pairwise coherence of the senders' DEROTATED
prompts on shared records -- the direct test of whether phi0 puts them on one reference.
"""
import argparse
import cmath
import collections
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python", "scripts", "gnss"))

from gnss_broker import telem  # noqa: E402


def collect(host, port, chain, windows, timeout_s):
    c = telem.TelemClient(host=host, port=port, depth=max(64, windows + 8), retry_s=1.0).start()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if len(c.windows(chain, lag=1)) >= windows:
            break
        time.sleep(0.5)
    c.stop()
    return c


def _pow(g, w):
    return (abs(g) / w) ** 2 if w > 0.0 else 0.0


def reduce_frames(client, chain, wins):
    """{prn: {"A": [(e,p,l)], "B": [...], "B0": [...], "n_inst": [...], "n_chan": [...],
              "xcoh": [...]}} -- one entry per record (window, slot) that had a live comb."""
    out = {}
    for w in wins:
        fs = client.frame_set(chain, w)
        # per (slot, prn): the per-sender partial sums for this record
        per = collections.defaultdict(dict)   # (r, prn) -> inst -> dict
        for inst, f in fs.items():
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                for prn in f.prns():
                    cmb = f.comb_epl(r, prn)
                    if not cmb:
                        continue
                    row = f.row(r, prn)
                    phi0 = float(row[telem.REC_PHI0]) if row is not None else 0.0
                    gE = gP = gL = 0j
                    wE = wP = wL = 0.0
                    for _fid, E, P, L, (eE, eP, eL) in cmb:
                        gE += E * eE
                        gP += P * eP
                        gL += L * eL
                        wE += eE
                        wP += eP
                        wL += eL
                    if wP <= 0.0:
                        continue
                    per[(r, prn)][inst] = dict(gE=gE, gP=gP, gL=gL, wE=wE, wP=wP, wL=wL,
                                               rot=cmath.exp(1j * phi0), n_chan=len(cmb))
        for (r, prn), insts in per.items():
            d = out.setdefault(prn, {"A": [], "B": [], "B0": [], "n_inst": [], "n_chan": [],
                                     "xcoh": []})
            # A: power per sender, summed
            eA = sum(_pow(v["gE"], v["wE"]) for v in insts.values())
            pA = sum(_pow(v["gP"], v["wP"]) for v in insts.values())
            lA = sum(_pow(v["gL"], v["wL"]) for v in insts.values())
            # B: one coherent sum over every channel, senders derotated by phi0
            GE = sum(v["gE"] * v["rot"] for v in insts.values())
            GP = sum(v["gP"] * v["rot"] for v in insts.values())
            GL = sum(v["gL"] * v["rot"] for v in insts.values())
            WE = sum(v["wE"] for v in insts.values())
            WP = sum(v["wP"] for v in insts.values())
            WL = sum(v["wL"] for v in insts.values())
            # B0: the same without the rotation
            GE0 = sum(v["gE"] for v in insts.values())
            GP0 = sum(v["gP"] for v in insts.values())
            GL0 = sum(v["gL"] for v in insts.values())
            d["A"].append((eA, pA, lA))
            d["B"].append((_pow(GE, WE), _pow(GP, WP), _pow(GL, WL)))
            d["B0"].append((_pow(GE0, WE), _pow(GP0, WP), _pow(GL0, WL)))
            d["n_inst"].append(len(insts))
            d["n_chan"].append(sum(v["n_chan"] for v in insts.values()))
            # cross-sender coherence of the derotated, energy-normalised prompts this record
            ap = [v["gP"] / v["wP"] * v["rot"] for v in insts.values()]
            if len(ap) >= 2:
                num = abs(sum(ap)) ** 2 - sum(abs(a) ** 2 for a in ap)
                den = sum(abs(a) ** 2 for a in ap) * (len(ap) - 1)
                d["xcoh"].append(num / den if den > 0 else 0.0)
    return out


def summarise(rows):
    """(E, P, L) meaned over records -> disc, q; plus per-record disc sd."""
    if not rows:
        return None
    E = statistics.mean(r[0] for r in rows)
    P = statistics.mean(r[1] for r in rows)
    L = statistics.mean(r[2] for r in rows)
    if E + L <= 0.0:
        return None
    discs = [(r[0] - r[2]) / (r[0] + r[2]) for r in rows if r[0] + r[2] > 0.0]
    sd = statistics.pstdev(discs) if len(discs) > 1 else float("nan")
    return dict(disc=(E - L) / (E + L), q=2.0 * P / (E + L), p=P, sd=sd)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11061)
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--probes", default="", help="comma-separated probe PRNs (marked)")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()
    probes = {int(x) for x in a.probes.split(",") if x.strip()}

    c = collect(a.host, a.port, a.chain, a.windows, a.timeout)
    wins = c.windows(a.chain, lag=1)[-a.windows:]
    if not wins:
        sys.exit("no windows for chain %r -- is the gather up and is that chain sending?" % a.chain)
    insts = sorted({i for w in wins for i in c.frame_set(a.chain, w)})
    print("chain %s: %d windows [%d..%d], %d senders" % (a.chain, len(wins), wins[0], wins[-1],
                                                          len(insts)))
    res = reduce_frames(c, a.chain, wins)
    print("%-5s %-4s %-5s %-4s | %-7s %-6s %-7s | %-7s %-6s %-7s | %-7s %-6s | %-6s %s"
          % ("PRN", "nrec", "ninst", "nch", "A.disc", "A.q", "A.sd", "B.disc", "B.q", "B.sd",
             "B0.disc", "B0.q", "XCOH", "P(B)/P(A)"))
    for prn in sorted(res):
        d = res[prn]
        A, B, B0 = summarise(d["A"]), summarise(d["B"]), summarise(d["B0"])
        if not (A and B and B0):
            continue
        ninst = statistics.median(d["n_inst"])
        nch = statistics.median(d["n_chan"])
        xc = statistics.median(d["xcoh"]) if d["xcoh"] else float("nan")
        tag = "  probe" if prn in probes else ""
        print("%-5d %-4d %-5.0f %-4.0f | %+7.3f %6.2f %7.3f | %+7.3f %6.2f %7.3f | %+7.3f %6.2f | "
              "%6.3f %.3g%s"
              % (prn, len(d["A"]), ninst, nch, A["disc"], A["q"], A["sd"], B["disc"], B["q"],
                 B["sd"], B0["disc"], B0["q"], xc,
                 B["p"] / A["p"] * ninst if A["p"] > 0 else float("nan"), tag))
    print()
    print("P(B)/P(A)*n_inst: 1.0 = the senders' prompts add coherently after phi0 (signal, one")
    print("reference); 1/n_inst = they add as noise (a probe, or a bug in the reference).")
    print("XCOH is the same question asked per record: ~1 coherent, ~0 no common phase.")


if __name__ == "__main__":
    sys.exit(main())
