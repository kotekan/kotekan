#!/usr/bin/env python3
"""PER-CHANNEL PHASE ACROSS THE BAND, WITH NO PHASE REFERENCE TO GET WRONG.

WHY THIS EXISTS. `band_coherence_gate.py --per-channel` referenced each channel against the
record's own FULL-BAND SUM. Under a ramp that sum CANCELS, so the reference is noise, and the
number duly swung 0.919 -> 0.220 between runs minutes apart. Self-reference failing exactly
where the effect lives ([[gnss-phase-from-a-reference-not-an-argument]]).

THE FIX IS TO REFERENCE NOTHING. Build the record-averaged channel covariance

    C_jk = < g_j conj(g_k) >_records          (j != k; the DIAGONAL IS DELETED)

and take its leading eigenvector. A per-record common phase theta_r enters as
e^{i theta_r} g_j * conj(e^{i theta_r} g_k) = g_j conj(g_k) -- it CANCELS EXACTLY, for any
theta_r, however it re-rolls. So the sky phase, the carrier NCO, and the frame-boundary jump
all drop out and what is left is purely what differs BETWEEN channels. That is the quantity
in question.

⚠️ THE DIAGONAL MUST GO. C_kk = <|g_k|^2> is noise power plus signal power and is positive for
pure noise, so a covariance WITH its diagonal always has a "dominant" eigenvector -- on noise
it just picks the loudest channel. Deleting the diagonal makes the estimator see only
CROSS-channel structure, which is the only thing that can make a coherent sum work.

HONESTY (the estimator is fitted from the data, so it must not be scored on it):
 * SPLIT-HALF. The per-channel phases are derived on the FIRST half of the records and scored
   on the SECOND. A self-scored alignment is guaranteed to look good.
 * SHUFFLE NULL. Each channel's record series is independently permuted, which preserves every
   channel's amplitude distribution and weight and destroys ONLY the cross-channel phase
   relationship -- the axis under test ([[gnss-a-null-must-preserve-the-weights]]).
 * PROBES. Noise probes go through the identical path; whatever they score is the floor.
 * --self-test. A planted ramp plus noise, recovered; and the null checked to collapse.

WHAT THE OUTPUT MEANS
    resid_rms   scatter of the per-channel phase about the best-fit LINE. Small = the phase
                across the band is a clean RAMP (a delay), which is removable and expected.
    tau         that line's slope as a delay (ns and chips).
    eta_align   coherent/power ratio on HELD-OUT records after removing the measured
                per-channel phases. This is what #72's switch would buy. Probes and the
                shuffle null say what it scores on nothing.

    ./band_phase_ramp.py --chain gps_l5 --prn 11 --plot /tmp/g11.png      # ON cf06
"""
import argparse
import cmath
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker import telem  # noqa: E402

CHAN_HZ = 3200.0e6 / 2 / 8192      # 0.195312 MHz -- see band_coherence_gate for why N=8192
CHIP_HZ = 10.23e6
FID_REF = 6024                     # L5 centre, 1176.45 MHz -> fid 6023.4


def collect(client, chain, wins, prns, derotate_phi0=False):
    """{prn: (fids, [ {fid: complex g} per record ])} -- the raw per-channel prompt series.

    g is the energy-weighted prompt exactly as the shipping reduction forms it (P * wP), so
    nothing here is a different quantity from what the combine sums.
    """
    series, owner = {}, {}
    for w in wins:
        fs = client.frame_set(chain, w)
        if not fs:
            continue
        n_rec = max((f.n_rec for f in fs.values()), default=0)
        for r in range(n_rec):
            per = {}
            for inst, f in fs.items():
                if r >= f.n_rec or not f.has_record(r):
                    continue
                for prn in f.prns():
                    if prn not in prns:
                        continue
                    # #72: the comb was exported rotated by exp(-i*phi0) with a PER-INSTANCE
                    # arbitrary origin. Undo it and every instance lands on one common
                    # reference (ang0, measured bit-identical fleet-wide).
                    und = 1.0
                    if derotate_phi0:
                        row = f.row(r, prn)
                        if row is None:
                            continue
                        und = cmath.exp(1j * row[telem.REC_PHI0])
                    for fid, _E, P, _L, (_wE, wP, _wL) in f.comb_epl(r, prn):
                        if wP > 0.0:
                            per.setdefault(prn, {})[fid] = P * wP * und
                            owner[fid] = inst
            for prn, chans in per.items():
                if len(chans) >= 8:
                    series.setdefault(prn, []).append(chans)
    out = {}
    for prn, recs in series.items():
        # keep channels present in most records, so the covariance is not built from a
        # ragged support that changes which pairs each entry averages over
        cnt = {}
        for rec in recs:
            for fid in rec:
                cnt[fid] = cnt.get(fid, 0) + 1
        keep = sorted(f for f, c in cnt.items() if c >= 0.8 * len(recs))
        if len(keep) >= 8:
            out[prn] = (keep, recs)
    return out, owner


def instance_split(fids, v, owner, tau):
    """(between, within, per_inst, n_inst) -- is the leftover phase a PER-INSTANCE CONSTANT?

    ⚠️ THIS IS THE TEST THAT MATTERS FOR #72. An instance is `freq_id mod 8` and the channels
    inside one are DECIMATED across the whole band (stride 16, not a contiguous block), so a
    constant phase offset per instance shows up in a phase-vs-frequency plot as pure SCATTER --
    indistinguishable by eye from a rough bandpass. It is also exactly the fault the shipping
    reduction is blind to: summing coherently WITHIN an instance and in power ACROSS one leaves
    a per-instance constant completely harmless, which is why it could sit here unnoticed.

    By the lockstep rule ([[chord-nothing-is-per-node]]) a per-instance constant is a BUG, not
    a calibration: one PFB, one set of samples, nothing physical at that boundary.

    Removes the fitted ramp first, then compares the scatter BETWEEN instance means against the
    scatter WITHIN instances. between >> within  =>  the leftover is a per-instance constant.
    """
    res = {}
    for fid, x in zip(fids, v):
        if x == 0j:
            continue
        ramp = -2 * math.pi * (fid - FID_REF) * CHAN_HZ * tau
        res[fid] = abs(x) * cmath.exp(1j * (cmath.phase(x) - ramp))
    by = {}
    for fid, z in res.items():
        by.setdefault(owner.get(fid, "?"), []).append(z)
    per = {}
    for inst, zs in by.items():
        s = sum(zs)
        if s != 0j:
            per[inst] = (cmath.phase(s), abs(s) / sum(abs(z) for z in zs), len(zs))
    # within: each channel against ITS OWN instance mean; between: instance means against the
    # global mean. Both circular, both weighted by |v| so a dead channel cannot dominate.
    def circ_rms(pairs):
        num = sum(w * (1.0 - math.cos(d)) for d, w in pairs)
        den = sum(w for _d, w in pairs) or 1.0
        return math.sqrt(2.0 * num / den)          # -> rms angle for small angles
    win, btw = [], []
    gm = cmath.phase(sum(res.values())) if res else 0.0
    for inst, zs in by.items():
        if inst not in per:
            continue
        mi = per[inst][0]
        for z in zs:
            win.append((cmath.phase(z) - mi, abs(z)))
        btw.append((mi - gm, sum(abs(z) for z in zs)))
    return circ_rms(btw), circ_rms(win), per, len(by)


def eigen_phase(fids, recs, rephase=False, rng=None):
    """(v, lam1, lam_off) -- leading eigenvector of the DIAGONAL-DELETED channel covariance.

    `rephase` is THE NULL: give every (record, channel) an independent random unit phasor.
    Amplitudes, weights and the channel support are untouched; the ONLY thing destroyed is the
    phase relationship, which is the axis under test ([[gnss-a-null-must-preserve-the-weights]]).

    ⚠️ A RECORD-ORDER SHUFFLE IS NOT A NULL HERE, and was my first attempt. Permuting each
    channel's record series randomises only the per-record COMMON phase theta_r; the
    deterministic per-channel term (the ramp) survives untouched, so the estimator recovers it
    from the shuffled data and the "null" scores well above the floor. It tests cross-record
    consistency, not cross-channel structure.
    """
    m = len(fids)
    cols = []
    for fid in fids:
        col = [rec.get(fid, 0j) for rec in recs]
        if rephase:
            col = [c * cmath.exp(2j * math.pi * rng.random()) if c != 0j else 0j for c in col]
        cols.append(col)
    n = len(recs)
    C = [[0j] * m for _ in range(m)]
    for j in range(m):
        cj = cols[j]
        for k in range(j + 1, m):
            ck = cols[k]
            s = 0j
            for r in range(n):
                a, b = cj[r], ck[r]
                if a != 0j and b != 0j:
                    s += a * b.conjugate()
            C[j][k] = s / n
            C[k][j] = (s / n).conjugate()
    # power iteration on a Hermitian matrix with a zero diagonal. Its eigenvalues straddle
    # zero (trace = 0), so iterate on C + sI with s = the row-sum bound to keep the wanted
    # end dominant; the eigenvectors are unchanged by the shift.
    shift = max(sum(abs(C[j][k]) for k in range(m)) for j in range(m)) or 1.0
    v = [complex(1.0, 0.0)] * m
    lam = 0.0
    for _ in range(200):
        w = []
        for j in range(m):
            s = v[j] * shift
            row = C[j]
            for k in range(m):
                if k != j:
                    s += row[k] * v[k]
            w.append(s)
        nrm = math.sqrt(sum(abs(x) ** 2 for x in w)) or 1.0
        w = [x / nrm for x in w]
        if sum(abs(w[i] - v[i]) ** 2 for i in range(m)) < 1e-18:
            v = w
            break
        v = w
    # Rayleigh quotient on the UNSHIFTED matrix = the real eigenvalue we care about
    lam = 0.0
    for j in range(m):
        row = C[j]
        for k in range(m):
            if k != j:
                lam += (v[j].conjugate() * row[k] * v[k]).real
    lam_off = math.sqrt(sum(abs(C[j][k]) ** 2 for j in range(m) for k in range(m) if j != k))
    # fix the global phase so channel 0 is real: the eigenvector is defined only up to one
    ph0 = cmath.phase(v[0]) if v[0] != 0j else 0.0
    v = [x * cmath.exp(-1j * ph0) for x in v]
    return v, lam, lam_off


def fit_ramp(fids, v):
    """(tau_s, resid_rms_rad, phases, unwrapped) -- weighted straight line through the phase.

    Weighted by |v_k| so a channel the eigenvector barely supports cannot steer the slope.
    Unwrapped in fid order, which is safe only because neighbouring channels are 0.195 MHz
    apart: a full chip of delay is 0.019 cycles between neighbours, so no real delay can alias.
    """
    ph = [cmath.phase(x) for x in v]
    wt = [abs(x) for x in v]
    un, off = [], 0.0
    for i, p in enumerate(ph):
        if i:
            d = (p + off) - un[-1]
            off -= 2 * math.pi * round(d / (2 * math.pi))
        un.append(p + off)
    x = [(f - FID_REF) * CHAN_HZ for f in fids]
    sw = sum(wt) or 1.0
    mx = sum(w * xi for w, xi in zip(wt, x)) / sw
    my = sum(w * yi for w, yi in zip(wt, un)) / sw
    sxx = sum(w * (xi - mx) ** 2 for w, xi in zip(wt, x))
    sxy = sum(w * (xi - mx) * (yi - my) for w, xi, yi in zip(wt, x, un))
    slope = sxy / sxx if sxx > 0 else 0.0
    res = [yi - (my + slope * (xi - mx)) for xi, yi in zip(x, un)]
    rms = math.sqrt(sum(w * r * r for w, r in zip(wt, res)) / sw)
    return -slope / (2 * math.pi), rms, ph, un


def band_terms(fids, recs, v):
    """(D, X_align, X_raw, X_max, n) -- the coherent sum decomposed so PHASE separates from SNR.

    ⚠️ THIS IS THE POINT OF THE WHOLE TOOL. eta = <|sum_k g_k|^2> / (M <sum_k |g_k|^2>) has a
    CEILING SET BY PER-CHANNEL SNR, not by phase alignment:

        eta = (D + X) / (M D)      D = sum_k <|g_k|^2>        (power, signal + noise)
                                   X = sum_{j!=k} <g_j conj(g_k)>   (the coherent cross term)

    D contains the noise; X does not, because independent noise averages away over records. So
    a PERFECTLY aligned band whose per-channel SNR is low still reads eta ~ 1/M -- exactly the
    value that "the band does not cohere" was concluded from. eta cannot tell a misaligned band
    from a weak one, and no per-record ratio can.

    What CAN tell them apart is X_align / X_max, where X_max = sum_{j!=k} |<g_j conj(g_k)>| is
    the same cross term with every pair rotated into alignment. That ratio is 1 for a band that
    coheres however faint it is, and 0 for one that does not -- SNR divides out. X_raw is the
    cross term as the band sits today (what the shipping sum would get).

    ⚠️ X_max IS BIASED UP BY NOISE: |<n_j conj(n_k)>| > 0 for finite records, ~ 1/sqrt(n_rec).
    The rephase null measures that bias on the same data, so subtract it before believing a
    ratio near 1.
    """
    m = len(fids)
    cols = [[rec.get(fid, 0j) for rec in recs] for fid in fids]
    n = len(recs)
    D = 0.0
    for c in cols:
        s = sum(abs(x) ** 2 for x in c if x != 0j)
        D += s / n
    corr = [cmath.exp(-1j * cmath.phase(x)) if x != 0j else 1.0 for x in v]
    xa, xr, xm = 0.0, 0.0, 0.0
    for j in range(m):
        cj = cols[j]
        for k in range(m):
            if k == j:
                continue
            ck = cols[k]
            s = 0j
            for r in range(n):
                a, b = cj[r], ck[r]
                if a != 0j and b != 0j:
                    s += a * b.conjugate()
            s /= n
            xr += s.real
            xa += (s * corr[j] * corr[k].conjugate()).real
            xm += abs(s)
    return D, xa, xr, xm, n


def eta_holdout(fids, recs, v):
    """(eta_align, eta_raw, n) on the records given, using per-channel phases `v` from ELSEWHERE.

    eta = |sum_k g_k|^2 / (M * mean_k |g_k|^2), i.e. 1 when the channels are aligned and 1/M
    when they are unrelated -- the same normalisation as band_coherence_gate's eta_band, so
    the two numbers are directly comparable.
    """
    corr = [cmath.exp(-1j * cmath.phase(x)) if x != 0j else 1.0 for x in v]
    ea, er, n = 0.0, 0.0, 0
    for rec in recs:
        ga, gr, p, m = 0j, 0j, 0.0, 0
        for i, fid in enumerate(fids):
            g = rec.get(fid)
            if g is None or g == 0j:
                continue
            ga += g * corr[i]
            gr += g
            p += abs(g) ** 2
            m += 1
        if m < 8 or p <= 0.0:
            continue
        ea += abs(ga) ** 2 / (m * p / m) / m
        er += abs(gr) ** 2 / (m * p / m) / m
        n += 1
    return (ea / n if n else float("nan"), er / n if n else float("nan"), n)


def self_test():
    """Plant a ramp in noise, recover it; and check the shuffle null collapses."""
    rng = random.Random(20260816)
    fids = [5972 + k for k in range(104)]
    tau = 0.12 / CHIP_HZ                       # 0.12 chips of delay
    fail = 0

    def chk(ok, what, got, want):
        nonlocal fail
        if not ok:
            fail += 1
            print("  FAIL %-50s got %-12s want %s" % (what, got, want))
        else:
            print("  ok   %-50s %s" % (what, got))

    m = len(fids)
    for snr, dchip in ((4.0, 0.12), (0.5, 0.12), (0.5, 0.50)):
        tau = dchip / CHIP_HZ
        recs = []
        for _ in range(400):
            theta = rng.uniform(-math.pi, math.pi)      # per-record common phase, re-rolling
            rec = {}
            for f in fids:
                ramp = 2 * math.pi * (f - FID_REF) * CHAN_HZ * tau
                s = snr * cmath.exp(1j * (theta - ramp))
                rec[f] = s + complex(rng.gauss(0, 0.707), rng.gauss(0, 0.707))
            recs.append(rec)
        half = len(recs) // 2
        v, lam, _loff = eigen_phase(fids, recs[:half])
        t_hat, rms, _ph, _un = fit_ramp(fids, v)
        ea, er, _n = eta_holdout(fids, recs[half:], v)
        vs, lams, _los = eigen_phase(fids, recs[:half], rephase=True, rng=rng)
        eas, _e2, _n2 = eta_holdout(fids, recs[half:], vs)
        D, xa, xr, xm, _nn = band_terms(fids, recs[half:], v)
        # ⚠️ ANALYTIC BARS, NOT GUESSED ONES. My first pass asserted eta_raw < 0.1 and
        # eta_align > 0.6 and both "failed" against correct code: at 0.12 chips the ramp is
        # only 1.50 rad across the band (sinc^2 = 0.83, not 0), and at per-channel SNR 0.5 the
        # eta CEILING is 0.21 however perfect the alignment. Predict the number instead.
        S, N = snr ** 2, 1.0
        ceil = (S + N / m) / (S + N)                     # perfect alignment, this SNR
        phi = 2 * math.pi * (fids[-1] - fids[0]) * CHAN_HZ * tau
        sinc2 = (math.sin(phi / 2) / (phi / 2)) ** 2     # coherence a linear ramp leaves
        pred_raw = (S * sinc2 + N / m) / (S + N)
        print("  --- per-channel SNR %.1f, planted delay %.2f chip (%.2f rad across band) ---"
              % (snr, dchip, phi))
        chk(abs(t_hat - tau) * CHIP_HZ < 0.02,
            "recovers the planted delay (chips)", "%.4f" % (t_hat * CHIP_HZ),
            "%.4f" % (tau * CHIP_HZ))
        chk(rms < 0.35, "residual about the fitted line (rad)", "%.3f" % rms, "< 0.35")
        chk(abs(ea - ceil) < 0.15 * ceil + 0.02,
            "eta_align matches the SNR CEILING", "%.3f" % ea, "%.3f" % ceil)
        chk(abs(er - pred_raw) < 0.15 * ceil + 0.02,
            "eta_raw matches the ramp's predicted loss", "%.3f" % er, "%.3f" % pred_raw)
        chk(eas < 3.0 / m, "REPHASE NULL eta_align falls to the 1/M floor",
            "%.4f" % eas, "< %.4f" % (3.0 / m))
        chk(lam > 3 * lams, "eigenvalue beats its null", "%.3g vs %.3g" % (lam, lams), "3x")
        # the SNR-free statistic: does the band cohere, whatever its strength?
        cf = xa / xm if xm > 0 else float("nan")
        chk(cf > 0.9, "coherent fraction X_align/X_max (SNR-free)", "%.3f" % cf, "> 0.9")
        chk(xr / xm < sinc2 + 0.15 if phi > 1 else True,
            "raw cross term shows the ramp's loss", "%.3f" % (xr / xm), "~%.3f" % sinc2)
    # and a band with NO ramp must be left alone
    recs = []
    for _ in range(200):
        theta = rng.uniform(-math.pi, math.pi)
        recs.append({f: 4.0 * cmath.exp(1j * theta)
                     + complex(rng.gauss(0, .707), rng.gauss(0, .707)) for f in fids})
    v, _l, _lo = eigen_phase(fids, recs[:100])
    t0, _rms0, _p, _u = fit_ramp(fids, v)
    chk(abs(t0 * CHIP_HZ) < 0.01, "flat band -> zero delay", "%.4f" % (t0 * CHIP_HZ), "~0")
    _ea, er0, _n = eta_holdout(fids, recs[100:], v)
    chk(er0 > 0.9 * (16 + 1.0 / len(fids)) / 17, "flat band was already coherent RAW",
        "%.3f" % er0, "~%.3f" % ((16 + 1.0 / len(fids)) / 17))
    # PURE NOISE: no signal at all. The delay must not be "found", and the coherent fraction
    # must sit at its noise bias rather than near 1 -- the check that this tool cannot do to
    # noise what the deep fold did (41 dB-Hz on nothing).
    recs = [{f: complex(rng.gauss(0, .707), rng.gauss(0, .707)) for f in fids}
            for _ in range(400)]
    v, lam, _lo = eigen_phase(fids, recs[:200])
    _t, rmsn, _p, _u = fit_ramp(fids, v)
    ean, ern, _n = eta_holdout(fids, recs[200:], v)
    Dn, xan, _xrn, xmn, _nn = band_terms(fids, recs[200:], v)
    chk(rmsn > 1.0, "pure noise: phase does NOT look like a line", "%.3f" % rmsn, "> 1 rad")
    chk(ean < 4.0 / len(fids), "pure noise: eta_align stays at the 1/M floor",
        "%.4f" % ean, "< %.4f" % (4.0 / len(fids)))
    chk(abs(xan) / Dn < 0.5, "pure noise: no coherent cross term to speak of",
        "%.3f" % (abs(xan) / Dn), "<< M")
    print("    (noise reference: X_max/D = %.3f is the finite-record bias to subtract)"
          % (xmn / Dn))
    print("\nband_phase_ramp self-test: %s" % ("FAILED" if fail else "PASS"))
    return 1 if fail else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--broker", default="http://127.0.0.1:12060")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--prn", type=int, action="append", default=[],
                    help="PRN to measure (repeatable). Default: every held satellite.")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--min-duty", type=float, default=0.5)
    ap.add_argument("--plot", default=None, help="write phase-vs-frequency to this PNG")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    with urllib.request.urlopen("%s/%s/get_status" % (a.broker.rstrip("/"), a.chain),
                                timeout=10) as r:
        rows = json.loads(r.read().decode())
    probes = {int(x["prn"]) for x in rows if x.get("noise_probe")}
    held = {int(x["prn"]) for x in rows
            if not x.get("noise_probe") and x.get("cn0_prompt_db") is not None
            and (x.get("cn0_prompt_duty") or 0) >= a.min_duty}
    if a.prn:
        held = set(a.prn)
    cn0 = {int(x["prn"]): x.get("cn0_prompt_db") for x in rows}
    disc = {int(x["prn"]): x.get("dll_disc") for x in rows}
    if not held:
        raise SystemExit("INCONCLUSIVE: nothing held at duty >= %.2f on %s" % (a.min_duty,
                                                                              a.chain))

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    time.sleep(a.seconds)
    wins = cl.windows(a.chain, lag=1)[-a.windows:]
    ser, owner = collect(cl, a.chain, wins, held | probes)
    cl.stop()
    if not ser:
        raise SystemExit("no records -- --gather takes HOST:PORT and the gather serves "
                         "127.0.0.1 only, so run this ON cf06")

    rng = random.Random(1)
    print("chain %s: %d windows\n" % (a.chain, len(wins)))
    print("  %-5s %-7s %-6s %-6s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %s"
          % ("prn", "cn0", "n_rec", "n_ch", "tau(ns)", "= chips", "resid", "eta_raw",
             "eta_algn", "null", "coh_frac", ""))
    results = {}
    for prn in sorted(ser):
        fids, recs = ser[prn]
        if len(recs) < 40:
            continue
        half = len(recs) // 2
        v, lam, _lo = eigen_phase(fids, recs[:half])
        tau, rms, ph, un = fit_ramp(fids, v)
        ea, er, nn = eta_holdout(fids, recs[half:], v)
        vs, lams, _los = eigen_phase(fids, recs[:half], rephase=True, rng=rng)
        eas, _e2, _n2 = eta_holdout(fids, recs[half:], vs)
        D, xa, xr, xm, _nn = band_terms(fids, recs[half:], v)
        _Dn, _xan, _xrn, xmn, _n3 = band_terms(fids, recs[half:], vs)
        results[prn] = (fids, v, tau, rms, ph, un, ea, er, eas, lam, lams, nn,
                        D, xa, xr, xm, xmn)
        # ⚠️ NOT DEBIASED BY SUBTRACTION. |C_jk| is a magnitude, so its noise bias adds in
        # QUADRATURE, not linearly, and xm - xmn went negative (a NaN) the first time out on
        # sky. Report the raw ratio with the null's beside it and let the pair speak.
        cf = xa / xm if xm > 0 else float("nan")
        cfn = xmn / xm if xm > 0 else float("nan")
        print("  %-5d %-7s %-6d %-6d %-9.1f %-9.3f %-9.3f %-9.3f %-9.3f %-9.3f %-4.2f/%-4.2f %s"
              % (prn, ("%.1f" % cn0[prn]) if cn0.get(prn) is not None else "--",
                 len(recs), len(fids), tau * 1e9, tau * CHIP_HZ, rms, er, ea, eas, cf, cfn,
                 "probe" if prn in probes else ""))
    print("\n  resid    = rms scatter of per-channel phase about the fitted LINE (rad). "
          "Small => a clean RAMP.")
    print("  eta_raw  = held-out coherent/power with the band summed AS IT IS "
          "(1 = aligned, 1/M = unrelated).")
    print("  eta_algn = the same after removing the measured per-channel phase. "
          "null = the rephase control.")
    print("  coh_frac = X_align/(X_max - null): the cross term achieved vs achievable, with "
          "the finite-record\n             noise bias removed. ⚠️ THIS IS THE ONE THAT ANSWERS "
          "#72 -- eta has a CEILING set by\n             per-channel SNR, so a coherent but "
          "faint band still reads eta ~ 1/M.")
    sats = [p for p in results if p not in probes]
    if sats:
        print("\n  IS THE LEFTOVER PHASE A PER-INSTANCE CONSTANT? (ramp removed first)")
        print("    An instance is freq_id mod 8, DECIMATED across the band at stride 16, so a")
        print("    per-instance constant looks like pure scatter against frequency -- and the")
        print("    shipping reduction (coherent WITHIN, power ACROSS) is exactly blind to it.")
        print("    %-5s %-10s %-10s %-8s %s"
              % ("prn", "between", "within", "ratio", "verdict"))
        for prn in sorted(sats):
            fids, v = results[prn][0], results[prn][1]
            tau = results[prn][2]
            btw, win, per, ni = instance_split(fids, v, owner, tau)
            rat = btw / win if win > 0 else float("nan")
            print("    %-5d %-10.3f %-10.3f %-8.2f %s"
                  % (prn, btw, win, rat,
                     "PER-INSTANCE" if rat > 1.5 else
                     ("within-instance" if rat < 0.67 else "mixed")))
        print("\n    per-instance mean phase (rad), ramp removed -- compare ACROSS satellites:")
        insts = sorted({i for prn in sats
                        for i in instance_split(results[prn][0], results[prn][1],
                                                owner, results[prn][2])[2]})
        print("    %-5s %s" % ("prn", " ".join("%-8s" % i for i in insts)))
        for prn in sorted(sats):
            per = instance_split(results[prn][0], results[prn][1], owner, results[prn][2])[2]
            print("    %-5d %s" % (prn, " ".join(
                ("%+8.2f" % per[i][0]) if i in per else "%8s" % "--" for i in insts)))
        print("    ⚠️ AGREEMENT ACROSS SATELLITES is what makes it INSTRUMENTAL. A column that "
              "reads the\n       same for every PRN is one instance's own constant -- a BUG by "
              "the lockstep rule, and\n       removable. A column that scatters per PRN is not "
              "an instance property at all.")
    if sats and len(sats) > 1:
        ts = [results[p][2] * 1e9 for p in sats]
        print("\n  satellite tau spread %.1f ns over %d sats -- a shared INSTRUMENTAL delay "
              "would agree, a per-sat code error would not." % (max(ts) - min(ts), len(sats)))
        for p in sats:
            print("    prn %-4d tau %+8.1f ns   dll_disc %s" %
                  (p, results[p][2] * 1e9,
                   ("%+.3f" % disc[p]) if disc.get(p) is not None else "--"))

    if a.plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                                    # noqa: BLE001
            print("  (no plot: %s)" % e)
            return 0
        show = [p for p in sorted(results) if p not in probes] or sorted(results)
        fig, axes = plt.subplots(len(show), 1, figsize=(11, 3.0 * len(show)), squeeze=False)
        for ax, prn in zip(axes[:, 0], show):
            (fids, v, tau, rms, ph, un, ea, er, eas, lam, lams, nn,
             D, xa, xr, xm, xmn) = results[prn]
            fr = [(f * CHAN_HZ) / 1e6 for f in fids]
            amp = [abs(x) for x in v]
            mx = max(amp) or 1.0
            sc = ax.scatter(fr, un, c=[x / mx for x in amp], cmap="viridis",
                            vmin=0, vmax=1, s=26)
            xr = [(f - FID_REF) * CHAN_HZ for f in fids]
            sw = sum(amp) or 1.0
            mxx = sum(w * x for w, x in zip(amp, xr)) / sw
            myy = sum(w * y for w, y in zip(amp, un)) / sw
            ax.plot(fr, [myy + (-2 * math.pi * tau) * (x - mxx) for x in xr], "r-", lw=1.2,
                    label="fit: tau %+.1f ns (%+.3f chip), resid %.2f rad" %
                          (tau * 1e9, tau * CHIP_HZ, rms))
            ax.set_title("%s PRN %d  --  C/N0 %s dB-Hz, %d chan, %d held-out rec, "
                         "eta raw %.2f -> aligned %.2f (null %.2f)"
                         % (a.chain, prn,
                            ("%.1f" % cn0[prn]) if cn0.get(prn) is not None else "--",
                            len(fids), nn, er, ea, eas), fontsize=9)
            ax.set_ylabel("phase (rad, unwrapped)")
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=0.3)
            fig.colorbar(sc, ax=ax, label="|eigvec| (weight)")
        axes[-1, 0].set_xlabel("RF frequency (MHz)")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)
        print("\n  wrote %s" % a.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
