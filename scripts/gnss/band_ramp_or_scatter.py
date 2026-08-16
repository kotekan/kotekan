#!/usr/bin/env python3
"""IS THE "SCATTERED" ACROSS-BAND PHASE A STEEP WRAPPING RAMP, OR PER-INSTANCE CONSTANTS?

Both look identical in a phase-vs-frequency scatter plot, and I have now claimed each of them
on different nights. They are separable, because the channel comb has TWO scales:

  SAME-INSTANCE pairs   stride 16 channels = 3.1250 MHz   -> delay ALIASES every 320.0 ns
                        carry NO per-instance constant (same instance on both ends)
  CROSS-INSTANCE pairs  1, 2 or 4 channels = 0.195..0.781 MHz -> unambiguous to 1.28..5.12 us
                        but DO carry the per-instance constants

So each pair type is blind to exactly what the other measures:

    same-instance  ->  the fine delay, MODULO 320 ns, uncontaminated
    cross-instance ->  everything, including whichever 320 ns lobe we are on

⚠️ AND THE SHIPPING REDUCTION USES ONLY SAME-INSTANCE COHERENCE. That is why a per-instance
constant can sit here indefinitely without hurting anything we currently measure, and why the
320 ns lobe ambiguity has never been resolved: the combine never forms a cross-instance pair.

THE TEST. Per satellite:
  1. Fit a delay using ONLY same-instance pairs, scanned over the unambiguous +-160 ns. No
     unwrapping anywhere -- unwrapping a noisy phase random-walks into a spurious slope, and
     the probes below show that costs +-200 ns of fake delay.
  2. Remove it and measure the leftover scatter WITHIN each instance.
        tiny  -> inside an instance the phase is a clean line; the scatter is NOT per-channel
        large -> the phase really is rough across frequency
  3. Measure each instance's leftover CONSTANT, and check whether the constants AGREE ACROSS
     SATELLITES. Agreement = instrumental, one number per instance, removable. Disagreement =
     not an instance property at all.

Nulls throughout: the noise probes, and a rephase control that keeps every amplitude and
destroys only phase.

    ./band_ramp_or_scatter.py --chain gps_l5            # ON cf06
"""
import argparse
import cmath
import json
import math
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker import telem  # noqa: E402
sys.path.insert(0, HERE)
from band_phase_ramp import collect, eigen_phase, CHAN_HZ, CHIP_HZ, FID_REF  # noqa: E402


def pair_cov(fids, recs):
    """{(j,k): C_jk} for j<k, plus the per-channel power. The covariance is what carries the
    cross-channel phase; the per-record common phase cancels in it exactly."""
    m = len(fids)
    cols = [[rec.get(f, 0j) for rec in recs] for f in fids]
    n = len(recs)
    C, P = {}, []
    for j in range(m):
        P.append(sum(abs(x) ** 2 for x in cols[j]) / n)
    for j in range(m):
        for k in range(j + 1, m):
            s = 0j
            cj, ck = cols[j], cols[k]
            for r in range(n):
                a, b = cj[r], ck[r]
                if a != 0j and b != 0j:
                    s += a * b.conjugate()
            C[(j, k)] = s / n
    return C, P


def scan_delay(fids, C, keys, lo_ns, hi_ns, step_ns=0.25):
    """(tau_s, peak, floor) -- matched filter over the pairs in `keys`. NO UNWRAP.

    score(tau) = |sum_pairs C_jk exp(+2i pi df_jk tau)|, which is the coherent sum of every
    pair's measured phase against the delay hypothesis. `floor` is the median over the scan, so
    a probe's peak/floor says what pure noise scores.
    """
    if not keys:
        return None, 0.0, 0.0
    df = [( (fids[j] - fids[k]) * CHAN_HZ, C[(j, k)] ) for (j, k) in keys]
    n = int((hi_ns - lo_ns) / step_ns)
    best, bt, vals = -1.0, 0.0, []
    for i in range(n + 1):
        tau = (lo_ns + i * step_ns) * 1e-9
        s = 0j
        for d, c in df:
            s += c * cmath.exp(2j * math.pi * d * tau)
        v = abs(s)
        vals.append(v)
        if v > best:
            best, bt = v, tau
    vals.sort()
    return bt, best, vals[len(vals) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--broker", default="http://127.0.0.1:12060")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--min-duty", type=float, default=0.5)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--blocks", type=int, default=1,
                    help="cut the capture into N consecutive blocks and report each\n"
                         "instance's constant per block: the TIMESCALE on which it\n"
                         "holds, rather than two captures that happen to disagree.")
    a = ap.parse_args()

    with urllib.request.urlopen("%s/%s/get_status" % (a.broker.rstrip("/"), a.chain),
                                timeout=10) as r:
        rows = json.loads(r.read().decode())
    probes = {int(x["prn"]) for x in rows if x.get("noise_probe")}
    held = {int(x["prn"]) for x in rows
            if not x.get("noise_probe") and x.get("cn0_prompt_db") is not None
            and (x.get("cn0_prompt_duty") or 0) >= a.min_duty}
    cn0 = {int(x["prn"]): x.get("cn0_prompt_db") for x in rows}
    disc = {int(x["prn"]): x.get("dll_disc") for x in rows}

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    time.sleep(a.seconds)
    wins = cl.windows(a.chain, lag=1)[-a.windows:]
    ser, owner = collect(cl, a.chain, wins, held | probes)
    cl.stop()
    if not ser:
        raise SystemExit("no records -- run this ON cf06 (--gather is HOST:PORT)")

    alias_ns = 1e9 / (16 * CHAN_HZ)
    print("\nchannel step %.4f MHz;  same-instance stride 16 = %.4f MHz  ->  delay aliases "
          "every %.1f ns" % (CHAN_HZ / 1e6, 16 * CHAN_HZ / 1e6, alias_ns))
    print("cross-instance pairs are 1/2/4 channels apart -> unambiguous to %.0f ns\n"
          % (1e9 / (4 * CHAN_HZ)))

    rng = random.Random(7)
    print("  %-5s %-7s %-6s | SAME-INSTANCE (no per-inst term, aliases %.0f ns)      "
          "| CROSS-INSTANCE" % ("prn", "cn0", "n_ch", alias_ns))
    print("  %-5s %-7s %-6s | %-9s %-9s %-9s %-9s | %-9s %-9s"
          % ("", "", "", "tau_ns", "=chips", "pk/floor", "resid", "tau_ns", "pk/floor"))
    out = {}
    for prn in sorted(ser):
        fids, recs = ser[prn]
        if len(recs) < 40:
            continue
        C, P = pair_cov(fids, recs)
        same = [(j, k) for (j, k) in C
                if owner.get(fids[j]) == owner.get(fids[k])]
        cross = [(j, k) for (j, k) in C
                 if owner.get(fids[j]) != owner.get(fids[k])
                 and abs(fids[j] - fids[k]) <= 4]
        t_s, pk_s, fl_s = scan_delay(fids, C, same, -alias_ns / 2, alias_ns / 2)
        t_c, pk_c, fl_c = scan_delay(fids, C, cross, -alias_ns / 2, alias_ns / 2)
        # leftover scatter WITHIN each instance once that delay is removed: the coherence of
        # each same-instance pair against the fitted delay. 1 = a perfect line, 0 = rough.
        num, den = 0j, 0.0
        for (j, k) in same:
            d = (fids[j] - fids[k]) * CHAN_HZ
            num += C[(j, k)] * cmath.exp(2j * math.pi * d * t_s)
            den += abs(C[(j, k)])
        coh_in = abs(num) / den if den > 0 else float("nan")
        resid = math.sqrt(max(0.0, -2.0 * math.log(coh_in))) if 0 < coh_in <= 1 else float("nan")
        out[prn] = (fids, C, same, cross, t_s, t_c, coh_in)
        print("  %-5d %-7s %-6d | %-9.1f %-9.3f %-9.1f %-9.3f | %-9.1f %-9.1f %s"
              % (prn, ("%.1f" % cn0[prn]) if cn0.get(prn) is not None else "--", len(fids),
                 t_s * 1e9, t_s * CHIP_HZ, pk_s / fl_s if fl_s > 0 else float("nan"), resid,
                 t_c * 1e9, pk_c / fl_c if fl_c > 0 else float("nan"),
                 "probe" if prn in probes else ""))
    print("\n  resid = rms phase left INSIDE an instance after removing that instance-scale "
          "delay,\n          from the pair coherence (sqrt(-2 ln |coh|)). SMALL => within an "
          "instance the\n          phase across the full 20.3 MHz is a CLEAN LINE, so the "
          "scatter is not per-channel.")

    # ---- the per-instance constants, and whether they agree across satellites
    sats = [p for p in out if p not in probes]
    if sats:
        insts = sorted({owner[f] for p in sats for f in out[p][0] if f in owner})
        print("\n  PER-INSTANCE CONSTANT (same-instance delay removed), rad:")
        print("    %-5s %s" % ("prn", " ".join("%-7s" % i for i in insts)))
        const = {}
        for prn in sorted(sats):
            fids, C, same, cross, t_s, t_c, _ci = out[prn]
            recs = ser[prn][1]
            v, _l, _lo = eigen_phase(fids, recs)
            per = {}
            for f, x in zip(fids, v):
                if x == 0j:
                    continue
                ph = cmath.phase(x) + 2 * math.pi * (f - FID_REF) * CHAN_HZ * t_s
                per.setdefault(owner.get(f), []).append(abs(x) * cmath.exp(1j * ph))
            const[prn] = {i: cmath.phase(sum(z)) for i, z in per.items() if sum(z) != 0j}
            print("    %-5d %s" % (prn, " ".join(
                ("%+7.2f" % const[prn][i]) if i in const[prn] else "%7s" % "--"
                for i in insts)))
        # agreement across satellites, referenced to each PRN's own first instance so a
        # per-satellite global phase cannot masquerade as disagreement
        print("\n    same, each row re-referenced to %s (removes each PRN's global phase):"
              % insts[0])
        rel = {}
        for prn in sorted(sats):
            if insts[0] not in const[prn]:
                continue
            r0 = const[prn][insts[0]]
            rel[prn] = {i: (const[prn][i] - r0 + math.pi) % (2 * math.pi) - math.pi
                        for i in const[prn]}
            print("    %-5d %s" % (prn, " ".join(
                ("%+7.2f" % rel[prn][i]) if i in rel[prn] else "%7s" % "--" for i in insts)))
        if len(rel) > 1:
            print("\n    %-9s %-9s %-9s" % ("instance", "circ_mean", "|R| (1=agree, 0=random)"))
            agree = []
            for i in insts:
                zs = [cmath.exp(1j * rel[p][i]) for p in rel if i in rel[p]]
                if len(zs) < 2:
                    continue
                R = abs(sum(zs)) / len(zs)
                agree.append(R)
                print("    %-9s %+9.2f %-9.3f" % (i, cmath.phase(sum(zs)), R))
            # null: the same |R| statistic on random phases, same count
            nulls = []
            for _ in range(400):
                zs = [cmath.exp(2j * math.pi * rng.random()) for _ in range(len(rel))]
                nulls.append(abs(sum(zs)) / len(zs))
            nulls.sort()
            p95 = nulls[int(0.95 * len(nulls))]
            mean_R = sum(agree) / len(agree) if agree else float("nan")
            print("\n    mean |R| = %.3f   vs random-phase 95%% bound %.3f over %d satellites"
                  % (mean_R, p95, len(rel)))
            print("    => %s" % ("INSTRUMENTAL: one constant per instance, shared by every "
                                 "satellite. Measurable and\n       removable, and a BUG by "
                                 "the lockstep rule." if mean_R > p95 else
                                 "NOT a shared per-instance constant -- the offsets differ "
                                 "per satellite, so they are\n       not an instance property "
                                 "and cannot be calibrated out as one."))

    if a.blocks > 1 and sats:
        # ⚠️ TWO CAPTURES MINUTES APART DISAGREEING IS NOT A TIMESCALE. Cut ONE capture into
        # consecutive blocks and watch each instance's constant walk, so the answer is a rate
        # rather than two samples ([[one-observation-is-not-a-verdict]]).
        print("\n  PER-INSTANCE CONSTANT vs TIME -- one capture cut into %d blocks" % a.blocks)
        for prn in sorted(sats):
            fids, C, same, cross, t_s, t_c, ci = out[prn]
            recs = ser[prn][1]
            b = len(recs) // a.blocks
            if b < 24:
                continue
            insts = sorted({owner[f] for f in fids if f in owner})
            rows = []
            for bi in range(a.blocks):
                v, _l, _lo = eigen_phase(fids, recs[bi * b:(bi + 1) * b])
                per = {}
                for f, x in zip(fids, v):
                    if x == 0j:
                        continue
                    ph = cmath.phase(x) + 2 * math.pi * (f - FID_REF) * CHAN_HZ * t_s
                    per.setdefault(owner.get(f), []).append(abs(x) * cmath.exp(1j * ph))
                c0 = {i: sum(z) for i, z in per.items() if sum(z) != 0j}
                ref = c0.get(insts[0])
                rows.append({i: cmath.phase(z / ref) for i, z in c0.items()} if ref else {})
            print("    PRN %d (block = %.1f s of records)" % (prn, 0.0419 * b))
            print("      %-6s %s %s" % ("inst", " ".join("%-7s" % ("b%d" % i)
                                                         for i in range(a.blocks)), "|R|"))
            Rs = []
            for i in insts:
                zs = [cmath.exp(1j * r[i]) for r in rows if i in r]
                R = abs(sum(zs)) / len(zs) if zs else float("nan")
                if i != insts[0]:
                    Rs.append(R)
                print("      %-6s %s %.3f" % (i, " ".join(
                    ("%+7.2f" % r[i]) if i in r else "%7s" % "--" for r in rows), R))
            if Rs:
                print("      mean |R| over the other instances: %.3f  (1 = a fixed constant, "
                      "0 = re-rolls)" % (sum(Rs) / len(Rs)))

    if a.plot and sats:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                                       # noqa: BLE001
            print("  (no plot: %s)" % e)
            return 0
        fig, axes = plt.subplots(len(sats), 1, figsize=(11, 3.0 * len(sats)), squeeze=False)
        for ax, prn in zip(axes[:, 0], sorted(sats)):
            fids, C, same, cross, t_s, t_c, ci = out[prn]
            recs = ser[prn][1]
            v, _l, _lo = eigen_phase(fids, recs)
            byi = {}
            for f, x in zip(fids, v):
                if x != 0j:
                    byi.setdefault(owner.get(f), []).append((f, x))
            for i, pts in sorted(byi.items()):
                fr = [f * CHAN_HZ / 1e6 for f, _x in pts]
                ph = [cmath.phase(x) for _f, x in pts]
                ax.plot(fr, ph, "o-", ms=4, lw=0.8, label=i)
            ax.set_title("%s PRN %d -- phase per channel, JOINED WITHIN EACH INSTANCE. "
                         "same-inst tau %+.1f ns, in-instance coherence %.2f"
                         % (a.chain, prn, t_s * 1e9, ci), fontsize=9)
            ax.set_ylabel("phase (rad, wrapped)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, ncol=6, loc="upper right")
        axes[-1, 0].set_xlabel("RF frequency (MHz)")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)
        print("\n  wrote %s" % a.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
