#!/usr/bin/env python3
"""#10 on the CHANNEL axis: where does the coherent sum lose its dB?

    scripts/gnss/chan_phase.py [--chain gps_l5] [--prn 9] [--windows 64]

⚠️ READ THIS FIRST -- WHY THIS IS NOT A PER-INSTANCE TOOL. An "instance" is a bag of ~7
channels selected by `freq_id mod 8`, i.e. a ROUTING decision taken after the signal path.
Nothing physical can localise to one. Any apparent per-instance effect is a per-CHANNEL effect
averaged inside the bag, and chasing it at the bag level hides the variable that actually
matters (KV, twice). `inst_phase_drift.py` measured a 0.42 Hz "per-instance rate spread"; that
number is real but it is a SHADOW of whatever this tool measures per channel.

THE MODEL, and it has TWO parameters for the whole array, not one per group:

    arg A_c(t) = (sky, common to all channels) - 2*pi*f_c*( tau + taudot*t )

  * `tau`    a delay: a CONSTANT phase tilt across frequency. A code-phase offset, a cable
             length, an uncorrected group delay -- all look like this.
  * `taudot` the tilt DRIFTING: the code phase error changing with time. This is the one that
             a constant-per-anything alignment can never remove, and the one that turns into
             an apparent per-instance rate once you average channels into bags.

Two numbers describe the whole array. If they explain the loss, the fix is two numbers in
`fleet_coherent` -- not 10 free constants, and certainly not 65.

HOW IT IS MEASURED
  * per channel c, per record t: A_c(t) from the comb (all instances pooled -- a channel lives
    on exactly one, so pooling is a merge, not a sum).
  * S(t) = the all-channel sum per record: the common sky term, whatever it is doing.
  * R_c = sum_t A_c(t) conj(S(t)).  arg(R_c) is channel c's CONSTANT phase against the array.
    Fit arg(R_c) vs f_c -> tau.
  * split the records in half, redo -> arg(R_c) per half; the difference over the half-span is
    channel c's phase RATE. Fit vs f_c -> taudot.
  * then REBUILD the coherent sum with each correction in turn and report the dB.

⚠️ EVERY GAIN IS SCORED AGAINST A CHANNEL-PERMUTED NULL. Fitting a slope across 65 noisy
points recovers amplitude from nothing; the same fit on permuted channel LABELS has the same
noise and no real ramp. Only the excess counts. (This is the discipline that produced -- and
then survived -- the retracted-then-restored x38 rate claim on the same night.)
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

CHAN_HZ = 195312.5      # PFB bin spacing; freq_id * CHAN_HZ is the sky frequency
DT = 0.0104857          # one record


def per_channel(client, chain, wins, prn):
    """{freq_id: {hop: A}} pooled over instances -- a channel lives on exactly one."""
    out = {}
    for w in wins:
        for _inst, f in client.frame_set(chain, w).items():
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                for fid, A, e in f.comb(r, prn):
                    if e > 0:
                        out.setdefault(fid, {})[f.hop(r)] = A
    return out


def fit_slope(fids, phases, weights):
    """Weighted least-squares slope of phase vs freq_id, on UNWRAPPED phases (rad per Hz)."""
    xs = [fid * CHAN_HZ for fid in fids]
    W = sum(weights)
    if W <= 0 or len(xs) < 3:
        return 0.0, float("nan")
    xb = sum(w * x for w, x in zip(weights, xs)) / W
    yb = sum(w * y for w, y in zip(weights, phases)) / W
    den = sum(w * (x - xb) ** 2 for w, x in zip(weights, xs))
    if den <= 0:
        return 0.0, float("nan")
    m = sum(w * (x - xb) * (y - yb) for w, x, y in zip(weights, xs, phases)) / den
    resid = math.sqrt(sum(w * (y - yb - m * (x - xb)) ** 2
                          for w, x, y in zip(weights, xs, phases)) / W)
    return m, resid


def channel_phases(chans, hops):
    """[(fid, arg R_c, |R_c| weight)] against the LEAVE-ONE-OUT per-record sum.

    ⚠️ LEAVE-ONE-OUT IS NOT OPTIONAL. Correlating channel c against a reference that CONTAINS
    channel c adds a |A_c|^2 self-term, which is real and positive and drags every arg(R_c)
    toward 0 -- so the channels look aligned because each one was compared with itself. With
    65 channels the self-term is only ~1/65 of the reference, but it is exactly the systematic
    that makes a null look like a detection, and it is the same trap fleet_coherent's own
    leave-one-out exists to avoid.
    """
    S = {h: sum(c[h] for c in chans.values() if h in c) for h in hops}
    out = []
    for fid, series in sorted(chans.items()):
        R = sum(series[h] * (S[h] - series[h]).conjugate() for h in hops if h in series)
        if R != 0:
            out.append((fid, cmath.phase(R), abs(R)))
    return out


def coh(chans, hops, tau, taudot):
    """coh_frac of the array sum with channels derotated by the 2-parameter model."""
    tot = {}
    for fid, series in chans.items():
        w = 2 * math.pi * fid * CHAN_HZ
        for k, h in enumerate(hops):
            if h in series:
                tot[h] = tot.get(h, 0j) + series[h] * cmath.exp(1j * w * (tau + taudot * k * DT))
    s = abs(sum(tot.values()))
    inc = sum(abs(v) for v in tot.values())
    return s / inc if inc > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11061)
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--prn", type=int, default=9)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--null-trials", type=int, default=16)
    a = ap.parse_args()

    cl = telem.TelemClient(a.host, a.port, depth=max(64, a.windows * 2), chains=[a.chain])
    cl.start()
    t0 = time.time()
    while time.time() - t0 < 90 and len(cl.windows(a.chain, lag=1)) < a.windows:
        time.sleep(1.0)
    wins = cl.windows(a.chain, lag=1)[-a.windows:]
    chans = per_channel(cl, a.chain, wins, a.prn)
    if len(chans) < 8:
        raise SystemExit("only %d channels" % len(chans))
    hops = sorted(set.union(*[set(c) for c in chans.values()]))
    print("chain %s PRN %d: %d channels, %d records (%.2f s), freq_id %d..%d\n"
          % (a.chain, a.prn, len(chans), len(hops), len(hops) * DT,
             min(chans), max(chans)))

    # ---- 1. CONSTANT tilt across frequency = a delay -------------------------------------
    cp = channel_phases(chans, hops)
    fids = [f for f, _p, _w in cp]
    ph = [p for _f, p, _w in cp]
    wt = [w for _f, _p, w in cp]
    m, resid = fit_slope(fids, ph, wt)
    tau = -m / (2 * math.pi)
    print("[1] CONSTANT phase tilt across the comb")
    print("    fitted delay tau = %+.1f ns   (residual scatter %.2f rad)" % (tau * 1e9, resid))
    if resid > 1.2:
        print("    ⚠️ residual > 1.2 rad: the per-channel phases are NOT on a line, so a single")
        print("       delay does not describe them and this tau is not meaningful.")

    # ---- 2. Does the tilt DRIFT? ----------------------------------------------------------
    half = len(hops) // 2
    cp0 = dict((f, p) for f, p, _w in channel_phases(chans, hops[:half]))
    cp1 = dict((f, p) for f, p, _w in channel_phases(chans, hops[half:]))
    shared = sorted(set(cp0) & set(cp1))
    dphi, dw = [], []
    for f in shared:
        d = cp1[f] - cp0[f]
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        dphi.append(d)
        dw.append(1.0)
    span = half * DT
    m2, resid2 = fit_slope(shared, dphi, dw)
    taudot = -m2 / (2 * math.pi) / span
    print("\n[2] Does that tilt DRIFT? (first half vs second, %.2f s apart)" % span)
    print("    fitted taudot = %+.3f ns/s  = %+.4f chips/s   (residual %.2f rad)"
          % (taudot * 1e9, taudot * 10.23e6, resid2))
    print("    median |dphi| across channels %.2f rad -- the COMMON part is sky/carrier and"
          % statistics.median([abs(x) for x in dphi]))
    print("    is removed by the fit; only the FREQUENCY-DEPENDENT part is taudot.")

    # ---- 3. What does each correction actually buy? --------------------------------------
    base = coh(chans, hops, 0.0, 0.0)
    g_tau = coh(chans, hops, tau, 0.0)
    g_both = coh(chans, hops, tau, taudot)
    def db(x, y):
        return 20 * math.log10(x / y) if x > 0 and y > 0 else float("nan")
    print("\n[3] coh_frac of the whole-array sum over %d records" % len(hops))
    print("    blind (what ships today)      %.4f" % base)
    print("    + delay tau                   %.4f   (%+.2f dB)" % (g_tau, db(g_tau, base)))
    print("    + delay AND drift             %.4f   (%+.2f dB)" % (g_both, db(g_both, base)))

    # ---- 4. the null: same fit, channel LABELS permuted -----------------------------------
    rng = random.Random(20260815)
    nulls = []
    for _ in range(a.null_trials):
        ids = list(chans)
        vals = [chans[i] for i in ids]
        rng.shuffle(vals)
        sh = dict(zip(ids, vals))
        cps = channel_phases(sh, hops)
        mm, _r = fit_slope([f for f, _p, _w in cps], [p for _f, p, _w in cps],
                           [w for _f, _p, w in cps])
        t_n = -mm / (2 * math.pi)
        b_n = coh(sh, hops, 0.0, 0.0)
        g_n = coh(sh, hops, t_n, 0.0)
        if b_n > 0 and g_n > 0:
            nulls.append(db(g_n, b_n))
    if nulls:
        mn = statistics.median(nulls)
        print("    channel-permuted NULL for the delay fit: %+.2f dB (median of %d)"
              % (mn, len(nulls)))
        print("    => EXCESS over the null: %+.2f dB" % (db(g_tau, base) - mn))
    print("\n⚠️ A tilt fit across noisy channels recovers amplitude from nothing. Only the")
    print("   EXCESS over the channel-permuted null is a real, per-channel structure.")

    # ---- 5. IS THE PER-CHANNEL SCATTER REAL AND RECOVERABLE? ------------------------------
    # [1]'s residual is the number that matters: the channel phases are not on a line, so they
    # are not a delay -- but are they a STABLE per-channel constant (recoverable, and worth
    # ~exp(-sigma^2/2) of amplitude) or just noise?
    #
    # ⚠️ SPLIT-HALF, because fitting a phase per channel on the same records you then score is
    # self-reference and ALWAYS gains -- 65 free parameters will align pure noise. Fit on the
    # first half, APPLY TO THE SECOND, score there. A per-channel constant that is real
    # transfers; one that is noise does not.
    fitH, useH = hops[:half], hops[half:]
    fit_ph = dict((f, p) for f, p, _w in channel_phases(chans, fitH))
    sigma = None
    cps_all = channel_phases(chans, hops)
    if cps_all:
        mvec = sum(cmath.exp(1j * (p - (m * fid * CHAN_HZ)))
                   for fid, p, _w in cps_all) / len(cps_all)
        sigma = math.sqrt(max(0.0, -2.0 * math.log(abs(mvec)))) if abs(mvec) > 1e-9 else None

    def coh_derot(cc, hh, table):
        tot = {}
        for fid, series in cc.items():
            rot = cmath.exp(-1j * table.get(fid, 0.0))
            for h in hh:
                if h in series:
                    tot[h] = tot.get(h, 0j) + series[h] * rot
        s = abs(sum(tot.values()))
        inc = sum(abs(v) for v in tot.values())
        return s / inc if inc > 0 else 0.0

    b2 = coh_derot(chans, useH, {})
    g2 = coh_derot(chans, useH, fit_ph)
    nn = []
    for _ in range(a.null_trials):
        ids = list(fit_ph)
        vals = [fit_ph[i] for i in ids]
        rng.shuffle(vals)
        nn.append(db(coh_derot(chans, useH, dict(zip(ids, vals))), b2))
    print("\n[5] IS THE PER-CHANNEL PHASE A REAL, STABLE CONSTANT? (split-half, fit on the")
    print("    first %d records, applied to the LAST %d -- never scored on its own fit)"
          % (len(fitH), len(useH)))
    if sigma is not None:
        print("    per-channel phase scatter about the delay line: %.2f rad" % sigma)
        print("      -> costs exp(-sigma^2/2) = %.3f in amplitude = %.2f dB if it is REAL"
              % (math.exp(-sigma * sigma / 2), 20 * math.log10(math.exp(-sigma * sigma / 2))))
        print("      -> noise alone would give ~%.2f rad at this per-channel SNR" % (1.0 / 4.6))
    print("    held-out coh_frac  blind %.4f -> per-channel derotated %.4f  (%+.2f dB)"
          % (b2, g2, db(g2, b2)))
    if nn:
        print("    permuted-label null %+.2f dB  =>  EXCESS %+.2f dB"
              % (statistics.median(nn), db(g2, b2) - statistics.median(nn)))
        print("    A REAL per-channel constant transfers across the split and beats this null.")
        print("    If it does not, the scatter is noise and the missing dB is NOT here.")
    cl.stop()


if __name__ == "__main__":
    main()
