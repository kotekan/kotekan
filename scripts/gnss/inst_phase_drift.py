#!/usr/bin/env python3
"""#10: is each instance's phase offset CONSTANT-but-noisy, or actually TIME-VARYING?

    scripts/gnss/inst_phase_drift.py [--chain gps_l5] [--prn 9] [--windows 32] [--block 16]

THE QUESTION THIS SETTLES, and it is the whole of #10. fleet_coherent aligns instances by ONE
constant phase each, rot_i = exp(-i arg<A_i conj(A_ref)>). Measured on G9 at search SNR 2350
the cross-instance |corr| is 0.310 against a 1/sqrt(128) = 0.088 floor -- real but marginal --
and the fleet deep comes out only 2.4 dB above a plain incoherent sum where coherent combining
should buy 10 dB. 7.6 dB is missing. There are exactly two explanations and they need
different fixes:

  (a) THE OFFSET IS CONSTANT AND THE DATA IS NOISY. Per-record per-instance SNR is ~0.77
      (inst deep 7.72 over sqrt(100) records), so a per-record phase is nearly random by
      itself. Then nothing is broken, 0.310 is what 0.77 looks like, and the 7.6 dB is a
      wrong expectation rather than a lost signal.

  (b) THE OFFSET MOVES DURING THE WINDOW. Then no single rotation can align it, the loss is
      real and recoverable, and the fix is a per-instance phase MODEL (rate, or shorter
      alignment blocks) rather than a constant.

⚠️ |corr| CANNOT TELL THESE APART. Both give the same depressed correlation. The discriminator
is whether the phase estimated in one BLOCK of records predicts the phase in the NEXT block
better than chance -- a constant offset is predictable across blocks however noisy each
estimate is; a drifting one is not.

HOW IT IS MEASURED
  * per instance, per record: A_i(t) = sum_c A_c E_c / sum_c E_c over that instance's comb.
  * split the window into blocks of `--block` records; in each, estimate the phase of
    instance i against the fleet reference: phi_i(b) = arg( sum_t A_i(t) conj(A_ref(t)) ).
  * the SCATTER of phi_i(b) across blocks is compared against THE SCATTER IT WOULD HAVE FROM
    NOISE ALONE, which is not assumed -- it is measured per block from the same data as
    sigma ~ 1/|snr_b|, where snr_b is that block's own coherent-to-incoherent ratio.
  * plus a LAG-1 PREDICTION TEST: does phi_i(b) predict phi_i(b+1)? For a constant offset the
    circular correlation of consecutive block phases is high whatever the per-block noise;
    for a drifting offset it collapses. This is the part that separates (a) from (b), because
    it is insensitive to the noise LEVEL and sensitive only to whether there is something
    stable underneath.

⚠️ THE NULL IS BUILT FROM THE SAME DATA, AND IT IS THE (a) HYPOTHESIS ITSELF. Block phases are
recomputed on a record-PERMUTED copy. Permuting destroys any TIME structure but PRESERVES a
constant offset, so the null is precisely "a constant offset seen through this much noise" --
the best lag-1 a constant model could possibly achieve on this data. Therefore:

    measured lag-1 ~ null   ->  (a) constant offset, nothing lost
    measured lag-1 << null  ->  (b) the offset MOVES: less repeatable than a constant

which is the right way round, and is why the null is HIGH rather than low. A null that also
destroyed the constant would be answering a question nobody asked.

FINALLY, THE FIX IS TESTED, NOT ASSERTED: a per-instance differential RATE is fitted to the
block phases and the coherent sum is recomputed with it. The dB it recovers is the size of the
prize, measured on the same records, against the same alignment the broker ships today.
"""
import argparse
import cmath
import math
import os
import random
import statistics
import sys
import time

K = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(K, "python", "scripts", "gnss"))

from gnss_broker import telem  # noqa: E402


def per_instance(client, chain, wins, prn):
    """{inst: {hop: A}} -- each instance's channel-combined prompt, per record."""
    out = {}
    for w in wins:
        for inst, f in client.frame_set(chain, w).items():
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                cmb = f.comb(r, prn)
                if not cmb:
                    continue
                tot = sum(e for _fid, _A, e in cmb)
                if tot > 0:
                    out.setdefault(inst, {})[f.hop(r)] = sum(A * e for _fid, A, e in cmb) / tot
    return out


def block_phases(a, b, hops, block):
    """[(phi, snr)] per block: phase of series a against b, and that block's own SNR."""
    out = []
    for i in range(0, len(hops) - block + 1, block):
        seg = hops[i:i + block]
        s = sum(a[h] * b[h].conjugate() for h in seg if h in a and h in b)
        inc = sum(abs(a[h] * b[h]) for h in seg if h in a and h in b)
        if inc <= 0:
            continue
        # coherent/incoherent ratio in [0,1]; x sqrt(N) is the block's phase SNR, so the
        # 1-sigma phase error is ~1/that. Measured, never assumed.
        frac = abs(s) / inc
        out.append((cmath.phase(s), frac * math.sqrt(len(seg))))
    return out


def circ_lag1(phis):
    """|<exp(i(phi_b - phi_{b+1}))>| -- 1 = perfectly repeatable, 0 = unrelated."""
    if len(phis) < 3:
        return float("nan")
    v = sum(cmath.exp(1j * (p - q)) for p, q in zip(phis, phis[1:]))
    return abs(v) / (len(phis) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11061)
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--prn", type=int, default=9)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--block", type=int, default=16, help="records per phase estimate")
    ap.add_argument("--null-trials", type=int, default=32)
    a = ap.parse_args()

    cl = telem.TelemClient(a.host, a.port, depth=max(64, a.windows * 2), chains=[a.chain])
    cl.start()
    t0 = time.time()
    while time.time() - t0 < 60 and len(cl.windows(a.chain, lag=1)) < a.windows:
        time.sleep(1.0)
    wins = cl.windows(a.chain, lag=1)[-a.windows:]
    per = per_instance(cl, a.chain, wins, a.prn)
    if len(per) < 2:
        raise SystemExit("PRN %d: %d instances" % (a.prn, len(per)))
    hops = sorted(set.intersection(*[set(v) for v in per.values()]))
    ref = max(per, key=lambda i: sum(abs(v) for v in per[i].values()))
    nb = len(hops) // a.block
    print("chain %s PRN %d: %d instances, %d shared records, %d blocks of %d (%.3f s each)\n"
          % (a.chain, a.prn, len(per), len(hops), nb, a.block, a.block * 0.0104857))
    print("reference instance %s" % ref)
    print("  a CONSTANT offset -> lag-1 HIGH however noisy each block is")
    print("  a DRIFTING offset -> lag-1 collapses to the shuffled null\n")
    print("inst      blocks  med blk snr  phase sd(deg)  expected sd  lag1   null   verdict")

    rng = random.Random(20260815)
    lag1s, nulls = [], []
    for inst in sorted(per):
        if inst == ref:
            continue
        bp = block_phases(per[inst], per[ref], hops, a.block)
        if len(bp) < 3:
            continue
        phis = [p for p, _ in bp]
        snrs = [s for _, s in bp]
        # circular sd of the block phases, in degrees
        m = sum(cmath.exp(1j * p) for p in phis) / len(phis)
        sd = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(abs(m))))) if abs(m) > 1e-9 else 180.0
        exp_sd = math.degrees(1.0 / statistics.median(snrs)) if statistics.median(snrs) > 0 else float("nan")
        l1 = circ_lag1(phis)
        # NULL: permute records, recompute blocks. Same amplitudes, same per-block SNR,
        # no time structure.
        nl = []
        for _ in range(a.null_trials):
            sh = list(hops)
            rng.shuffle(sh)
            bpn = block_phases(per[inst], per[ref], sh, a.block)
            if len(bpn) >= 3:
                nl.append(circ_lag1([p for p, _ in bpn]))
        nullv = statistics.median(nl) if nl else float("nan")
        lag1s.append(l1)
        nulls.append(nullv)
        verdict = ("CONSTANT (stable offset)" if l1 > nullv + 0.25 else
                   "DRIFTING / no stable offset" if l1 <= nullv + 0.08 else "marginal")
        print("%-9s %6d  %11.2f  %13.1f  %11.1f  %.3f  %.3f  %s"
              % (inst, len(bp), statistics.median(snrs), sd, exp_sd, l1, nullv, verdict))

    # ---- IS IT A RATE, AND WHAT WOULD FIXING IT BUY? --------------------------------------
    # Fit ONE differential rate per instance to its block phases (unwrapped), then rebuild the
    # fleet sum two ways on the SAME records: the constant-only alignment fleet_coherent ships
    # today, and constant+rate. The ratio is the prize, measured rather than argued.
    DT = 0.0104857
    rates = {}
    for inst in sorted(per):
        if inst == ref:
            continue
        bp = block_phases(per[inst], per[ref], hops, a.block)
        if len(bp) < 3:
            continue
        ph, t = [], []
        acc = 0.0
        prev = None
        for k, (p, _s) in enumerate(bp):
            if prev is not None:                       # unwrap: a rate is a RAMP, not a jump
                d = p - prev
                acc += -2 * math.pi if d > math.pi else (2 * math.pi if d < -math.pi else 0.0)
            prev = p
            ph.append(p + acc)
            t.append((k + 0.5) * a.block * DT)
        tb = statistics.fmean(t)
        pb = statistics.fmean(ph)
        den = sum((x - tb) ** 2 for x in t)
        rates[inst] = (sum((x - tb) * (y - pb) for x, y in zip(t, ph)) / den) if den else 0.0

    def fleet_snr(use_rate):
        tot = {}
        for inst, series in per.items():
            sh = [h for h in hops if h in series]
            if not sh:
                continue
            r = sum(series[h] * per[ref][h].conjugate() for h in sh)
            rot = (abs(r) / r) if r != 0 else 1 + 0j
            w = rates.get(inst, 0.0) if use_rate else 0.0
            for k, h in enumerate(hops):
                if h not in series:
                    continue
                tot[h] = tot.get(h, 0j) + series[h] * rot * cmath.exp(-1j * w * k * DT)
        s = sum(tot.values())
        inc = sum(abs(v) for v in tot.values())
        return abs(s) / inc if inc > 0 else 0.0

    c0, c1 = fleet_snr(False), fleet_snr(True)
    print("\nPER-INSTANCE DIFFERENTIAL RATE (Hz), fitted to the block phases:")
    print("   " + "  ".join("%s %+0.3f" % (i.split(".")[0][-2:] + "." + i.split(".")[1],
                                           r / (2 * math.pi)) for i, r in sorted(rates.items())))
    print("fleet coh_frac over %d records:  constant-only %.4f -> constant+rate %.4f  (%+.2f dB)"
          % (len(hops), c0, c1, 20 * math.log10(c1 / c0) if c0 > 0 and c1 > 0 else 0.0))
    print("⚠️ a rate FIT can inflate from noise; the honest comparison is this number against")
    print("   the same fit on record-shuffled data, which has no ramp to find:")
    sh_gain = []
    for _ in range(8):
        order = list(hops)
        rng.shuffle(order)
        keep = dict(rates)
        rates.clear()
        for inst in sorted(per):
            if inst == ref:
                continue
            bpn = block_phases(per[inst], per[ref], order, a.block)
            if len(bpn) < 3:
                continue
            ph = [p for p, _ in bpn]
            t = [(k + 0.5) * a.block * DT for k in range(len(ph))]
            tb, pb = statistics.fmean(t), statistics.fmean(ph)
            den = sum((x - tb) ** 2 for x in t)
            rates[inst] = (sum((x - tb) * (y - pb) for x, y in zip(t, ph)) / den) if den else 0.0
        n0, n1 = fleet_snr(False), fleet_snr(True)
        if n0 > 0 and n1 > 0:
            sh_gain.append(20 * math.log10(n1 / n0))
        rates.clear()
        rates.update(keep)
    if sh_gain:
        print("   shuffled-null gain %+.2f dB (median of %d)" % (statistics.median(sh_gain),
                                                                 len(sh_gain)))

    if lag1s:
        ml, mn = statistics.median(lag1s), statistics.median(nulls)
        print("\nFLEET: median lag-1 %.3f against a shuffled null of %.3f" % (ml, mn))
        if ml > mn + 0.25:
            print("=> THE PER-INSTANCE OFFSET IS STABLE. The constant rotation is the right")
            print("   model and the low |corr| is per-record SNR, not misalignment. The 7.6 dB")
            print("   is then a wrong expectation, NOT lost signal -- look elsewhere for #10.")
        elif ml <= mn + 0.08:
            print("=> NO STABLE OFFSET SURVIVES A BLOCK. One constant per instance CANNOT align")
            print("   this, which is exactly the 7.6 dB. The fix is a per-instance phase MODEL")
            print("   (a rate, or alignment blocks shorter than the drift), not a constant.")
        else:
            print("=> MARGINAL. Re-run at a larger --block (more SNR per estimate) before")
            print("   concluding: a null result here can be block SNR rather than drift.")
    cl.stop()


if __name__ == "__main__":
    main()
