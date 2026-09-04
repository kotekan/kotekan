#!/usr/bin/env python3
"""THE VALIDATION BAR FOR THE KNOWN-RATE COHERENT C/N0 (task #57 step 3).

combdll.coh_cn0 folds the per-record prompt series at an INJECTED rate (previous cycle's
record-stream fit -- causal, never searched). Phase estimators fail in the VARIANCE
(gnss-phase-estimator-self-reference), so the bar is the full protocol:

  self-test (offline, synthetic -- run FIRST; an estimator that cannot pass on data it
  fully controls has no business being judged on sky):
    * SKY-series recovery at the exact rate lands on closed-form truth (<= 0.3 dB).
    * RAW-series loss under a white per-record common phase sigma=0.75 rad equals the
      predicted e^{-sigma^2} (~-2.4 dB) -- the loss is MODELLED, not discovered.
    * A rate error df folds down by sinc^2(df T) -- checked at df = 0.5/T.
    * GENIE: derotating by the TRUE total phase is the bound; the estimator must sit AT
      it on the sky series and BELOW it on raw. Beating a genie = self-reference.
    * SHUFFLED-NULL: the same fold on time-shuffled records (shuffle, never roll)
      collapses to the noise scale; probes sit inside their own null band.

  on sky:
    * AGREE   on satellites cn0_prompt holds at duty >= 0.9, cn0_kcoh - cn0_prompt is a
              CONSTANT (the modelled sky-phase loss for raw; the split-aperture
              convention for sky): the per-satellite SCATTER about the median offset
              must be <= 1.0 dB. The offset itself is measured and printed, per chain.
    * NULL    each satellite's real fold vs its own shuffled-null distribution: strong
              sats >= 10x their null; probes INSIDE theirs. The one leg that would have
              caught the deep fold's disease on day one.
    * PROBE   probe folds sit at sig ~ 1 against the probe floor (they ARE the floor
              population; this checks the clipped-mean did not run away).

    ./kcoh_gate.py --self-test
    ./kcoh_gate.py --chain gps_l5 --seconds 25     # on the gather host
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
from gnss_broker import combdll, telem  # noqa: E402

T_REC = 2048 * 5.12e-6


# ---------------------------------------------------------------------------------------
# Synthetic client: header rows only (coh_cn0 reads row()), controlled rate + sky phase.
# ---------------------------------------------------------------------------------------
class _Frame:
    # ⚠️ hops_per_record = 1 BECAUSE the fake's hop() counts RECORDS, not hops, and the
    # self-test passes hop_s = T_REC to match. coh_cn0 derives t_rec = hops_per_record *
    # hop_s and t_k = hop * hop_s from these two together; mixing conventions (2048 here
    # with record-indexed hops) made t_coh 2048x too long and cost -29 dB on the first
    # run of this very test. The real transport uses hop units with hop_s = 5.12e-6.
    def __init__(self, mk_row, win, prns, n_rec=4, hops_per_record=1):
        self.n_rec, self.hops_per_record = n_rec, hops_per_record
        self._mk, self._win, self._prns = mk_row, win, prns

    def has_record(self, r):
        return True

    def hop(self, r):
        return self._win * self.n_rec + r      # consecutive records, hop units of records

    def prns(self):
        return self._prns

    def row(self, r, prn):
        return self._mk(self._win, r, prn)


class _FakeClient:
    """Signal PRN 23 at amplitude s and residual rate f0_hz; probes 91-93 pure noise.

    The per-record COMMON sky phase phi_k ~ N(0, sky_sig^2), white in time, multiplies the
    RAW prompt only; the SKY slots carry the same series with it removed -- exactly the
    contract REC_SKY has on the real records. Noise is independent per (record, instance).
    """

    def __init__(self, seed=1, n_win=64, n_inst=3, s=1.0, noise=0.35,
                 f0_hz=3.7, sky_sig=0.75):
        rng = random.Random(seed)
        self._wins = list(range(2000, 2000 + n_win))
        self.f0, self.s = f0_hz, s
        self._rows = {}
        self.true_phase = {}          # (win, r) -> total signal phase (genie reference)
        prns = [23, 91, 92, 93]
        for w in self._wins:
            for r in range(4):
                hop = w * 4 + r
                t = hop * T_REC       # hop is in RECORD units here; hop_s param must match
                phi_sky = rng.gauss(0.0, sky_sig)
                self.true_phase[(w, r)] = 2 * math.pi * f0_hz * t + phi_sky
                for i in range(n_inst):
                    for prn in prns:
                        sig = (cmath.rect(s, self.true_phase[(w, r)])
                               if prn == 23 else 0j)
                        n1 = complex(rng.gauss(0, noise), rng.gauss(0, noise))
                        raw = sig + n1
                        # sky slot: common phase removed from the SIGNAL, noise its own
                        # (the split-aperture correction rotates signal+noise together;
                        # for the arithmetic check the signal term is what matters)
                        sky = (cmath.rect(s, 2 * math.pi * f0_hz * t) if prn == 23 else 0j) + n1
                        row = [0.0] * 26
                        row[0] = prn
                        row[5] = 1.0
                        row[3], row[4] = raw.real, raw.imag
                        row[24], row[25] = sky.real, sky.imag
                        self._rows[(w, r, prn, i)] = row
        self._n_inst, self._prns = n_inst, prns

    def windows(self, chain, lag=1):
        return list(self._wins)

    def frame_set(self, chain, win):
        out = {}
        for i in range(self._n_inst):
            out["cx%d.0" % i] = _Frame(
                lambda w, r, prn, i=i: self._rows.get((w, r, prn, i)), win, self._prns)
        return out


def _shuffle_series(series, rng):
    """Time-shuffled copy: amplitudes keep their values, times are permuted. SHUFFLE,
    never roll -- a roll preserves the phase ramp and nulls nothing."""
    out = {}
    for inst, rows in series.items():
        hops = [h for h, *_ in rows]
        rng.shuffle(hops)
        out[inst] = [(hops[i], rows[i][1], rows[i][2], rows[i][3], rows[i][4])
                     for i in range(len(rows))]
    return out


def _fold_series(series, f_hz, hop_s, pick_sky=False):
    """Mean over instances of |mean_k A_k e^{-2 pi i f t_k}|^2 -- coh_cn0's core, redone
    here so the null/genie legs can drive it with modified series."""
    pws = []
    for rows in series.values():
        sacc, n = 0j, 0
        for h, re_, im_, sre, sim in rows:
            if pick_sky:
                if sre is None:
                    continue
                v = complex(sre, sim)
            else:
                v = complex(re_, im_)
            sacc += v * cmath.exp(-2j * math.pi * f_hz * h * hop_s)
            n += 1
        if n:
            pws.append(abs(sacc / n) ** 2)
    return sum(pws) / len(pws) if pws else None


def self_test():
    s, noise, f0, sky_sig = 1.0, 0.35, 3.7, 0.75
    fc = _FakeClient(s=s, noise=noise, f0_hz=f0, sky_sig=sky_sig)
    got = combdll.coh_cn0(fc, "fake", rates={23: f0}, n_win=64,
                          probe_prns={91, 92, 93}, hop_s=T_REC, keep_series=True)
    if not got or 23 not in got:
        print("SELF-TEST: FAIL -- no fold produced")
        return 1
    v = got[23]
    n = v["n_rec"]
    rho_true = s * s / (2 * noise * noise)          # per-record SNR
    cn0_true = 10 * math.log10(rho_true / T_REC)
    fails = []
    # SKY series: exact rate, no sky phase -> closed form
    err_sky = (v["cn0_sky_db"] - cn0_true) if v["cn0_sky_db"] is not None else 99
    # RAW series: the modelled e^{-sigma^2} coherence loss
    loss_pred = 10 * math.log10(math.exp(-sky_sig ** 2))
    err_raw = (v["cn0_db"] - (cn0_true + loss_pred)) if v["cn0_db"] is not None else 99
    print("SELF-TEST: truth %.2f dB-Hz | sky %.2f (err %+.2f) | raw %.2f "
          "(pred loss %+.2f, err %+.2f) | eta %.0f/%d | sig %.0f"
          % (cn0_true, v["cn0_sky_db"], err_sky, v["cn0_db"], loss_pred, err_raw,
             v["eta"] or -1, n, v["sig"]))
    if abs(err_sky) > 0.3:
        fails.append("sky-series recovery off by %+.2f dB" % err_sky)
    if abs(err_raw) > 0.6:
        fails.append("raw-series loss off the e^-sigma^2 model by %+.2f dB" % err_raw)
    # GENIE: derotate by the TRUE total phase -- the bound the estimator must not beat.
    ser = v["series"]
    genie = {}
    for inst, rows in ser.items():
        g = []
        for h, re_, im_, sre, sim in rows:
            w, r = h // 4, h % 4
            tp = fc.true_phase[(w, r)]
            a = complex(re_, im_) * cmath.exp(-1j * tp)
            g.append((0, a.real, a.imag, None, None))   # phase fully removed; fold at f=0
        genie[inst] = g
    pw_genie = _fold_series(genie, 0.0, T_REC)
    pw_sky = _fold_series(ser, f0, T_REC, pick_sky=True)
    if pw_sky > pw_genie * 1.10:
        fails.append("sky fold BEATS the genie by %.2f dB -- self-reference"
                     % (10 * math.log10(pw_sky / pw_genie)))
    # RATE-ERROR leg: df = 0.5/T -> sinc^2 loss
    T = v["t_coh_s"]
    df = 0.5 / T
    pw_off = _fold_series(ser, f0 + df, T_REC, pick_sky=True)
    x = math.pi * df * T
    pred = 20 * math.log10(abs(math.sin(x) / x))
    meas = 10 * math.log10(pw_off / pw_sky)
    print("SELF-TEST rate-error: df=%.2f Hz -> %+.2f dB (sinc^2 predicts %+.2f)"
          % (df, meas, pred))
    if abs(meas - pred) > 1.0:
        fails.append("rate-error loss %+.2f vs predicted %+.2f" % (meas, pred))
    # SHUFFLED NULL: strong signal collapses; expectation |s|^2/n + sigma^2/n
    rng = random.Random(7)
    nulls = [_fold_series(_shuffle_series(ser, rng), f0, T_REC) for _ in range(16)]
    null_med = sorted(nulls)[len(nulls) // 2]
    pw_raw = _fold_series(ser, f0, T_REC)
    print("SELF-TEST null: fold %.3e vs shuffled median %.3e (x%.0f)"
          % (pw_raw, null_med, pw_raw / null_med))
    if pw_raw < 10 * null_med:
        fails.append("fold only %.1fx its shuffled null on a strong synthetic"
                     % (pw_raw / null_med))
    # Probes: judge the MEDIAN, not each draw -- one probe's fold is a mean of only
    # n_inst exponential draws (Gamma(3)/3 here), whose lower tail legitimately reaches
    # ~0.2 a few percent of the time. The median of the probe set is the stable statistic.
    psigs = sorted(got[p]["sig"] for p in (91, 92, 93) if p in got)
    if psigs and not (0.3 < psigs[len(psigs) // 2] < 3.0):
        fails.append("probe sig median %.2f, expected ~1 (%s)"
                     % (psigs[len(psigs) // 2],
                        ", ".join("%.2f" % x for x in psigs)))
    # #57 RECOVERY leg (2026-08-17): the live failure mode is an injected rate WRONG by
    # a few Hz (the +-8 Hz poll-to-poll wobble against a one-cycle-old fit) -- the fold
    # then duty-cycles 60 dB. With the residual-rate fit, coh_cn0 must RECOVER the fold
    # to truth and NAME the error in rate_resid_hz. A probe must NOT grow a significant
    # fold from the same fit (its noise gain lands in the shared floor).
    got_off = combdll.coh_cn0(fc, "fake", rates={23: f0 + 2.0}, n_win=64,
                              probe_prns={91, 92, 93}, hop_s=T_REC)
    vo = got_off.get(23, {})
    err_rec = ((vo.get("cn0_sky_db") - cn0_true)
               if vo.get("cn0_sky_db") is not None else 99.0)
    print("SELF-TEST #57 recovery: injected +2.00 Hz off -> sky err %+.2f dB, "
          "rate_resid %+.2f Hz (want -2.00)"
          % (err_rec, vo.get("rate_resid_hz", 99.0)))
    if abs(err_rec) > 0.5:
        fails.append("#57 off-rate recovery err %+.2f dB (fit failed to save the fold)"
                     % err_rec)
    if abs(vo.get("rate_resid_hz", 99.0) + 2.0) > 0.3:
        fails.append("#57 rate_resid %+.2f Hz, want -2.00"
                     % vo.get("rate_resid_hz", 99.0))
    # #57 WILD-injection leg: the live fcoh rate swings +-10 Hz cycle-to-cycle, so the
    # injection can be wrong by more than one fit's capture -- the ZERO-centered second
    # hypothesis must rescue the fold (truth f0=3.7 Hz is within +-20 Hz of zero).
    got_wild = combdll.coh_cn0(fc, "fake", rates={23: f0 + 9.0}, n_win=64,
                               probe_prns={91, 92, 93}, hop_s=T_REC)
    vw = got_wild.get(23, {})
    err_wild = ((vw.get("cn0_sky_db") - cn0_true)
                if vw.get("cn0_sky_db") is not None else 99.0)
    print("SELF-TEST #57 wild-injection: +9.00 Hz off -> sky err %+.2f dB, "
          "rate_resid %+.2f Hz (want -9.00)"
          % (err_wild, vw.get("rate_resid_hz", 99.0)))
    if abs(err_wild) > 0.5:
        fails.append("#57 wild-injection recovery err %+.2f dB" % err_wild)
    if fails:
        print("SELF-TEST: FAIL\n  " + "\n  ".join(fails))
        return 1
    print("SELF-TEST: PASS")
    return 0


def _broker_rows(broker, chain):
    with urllib.request.urlopen("%s/get_status?chain=%s" % (broker, chain), timeout=5.0) as h:
        return json.loads(h.read().decode())


def coherence_scan(a):
    """eta/n vs T_coh on a HELD satellite -- the carrier coherence time, measured.

    THE POINT: the deep fold re-searched a rate every integration, so it could report a
    detection whether or not the carrier stayed coherent -- it fitted the incoherence out.
    A known-rate fold cannot, so eta becomes a direct measurement of something the old
    estimator structurally could not show. Where eta/n falls through ~0.5 is where a
    coherent integration stops paying, which is the number that decides how deep into the
    sidelobes this instrument can reach.
    """
    rows = _broker_rows(a.broker, a.chain)
    probes = {int(r["prn"]) for r in rows if r.get("noise_probe")}
    rates = {int(r["prn"]): r["rec_rate_hz"] for r in rows
             if r.get("rec_rate_hz") is not None}
    held = [r for r in rows
            if not r.get("noise_probe") and r.get("cn0_prompt_db") is not None
            and (r.get("cn0_prompt_duty") or 0) >= a.min_duty]
    if not probes:
        raise SystemExit("no noise probes -- no floor, no scan")
    if not held:
        # AN EXPERIMENT THAT CANNOT SUCCEED IS NOT EVIDENCE. Say what was available.
        best = max(((r.get("cn0_prompt_duty") or 0), r["prn"]) for r in rows) if rows else (0, 0)
        raise SystemExit(
            "INCONCLUSIVE: no satellite on %s is held at duty >= %.2f (best %.2f on PRN %d).\n"
            "A coherence scan on an intermittently-tracked satellite measures the gaps, not "
            "the carrier. Re-run when the chain is holding." % (a.chain, a.min_duty, best[0],
                                                               best[1]))
    if a.prn is not None:
        sel = [r for r in held if int(r["prn"]) == a.prn]
        if not sel:
            raise SystemExit("PRN %d is not held at duty >= %.2f on %s" % (a.prn, a.min_duty,
                                                                          a.chain))
        tgt = sel[0]
    else:
        tgt = max(held, key=lambda r: r["cn0_prompt_db"])
    prn = int(tgt["prn"])
    # ⚠️ THE INJECTED RATE IS A HYPOTHESIS, SO IT HAS TO BE VARIABLE. The scan's whole
    # conclusion ("coherence dies at T") is conditional on the rate it derotates by, and if
    # that rate is itself noise the scan measures the NOISE's decoherence, not the carrier's.
    # --rate-hz overrides it (0 = no derotation at all), which turns "is the rate the cause?"
    # into a paired measurement instead of an inference from a number that looked wrong.
    broker_rate = rates.get(prn) or 0.0
    rate_used = broker_rate
    note = ""
    if a.rate_hz is not None:
        rate_used = a.rate_hz
        rates = dict(rates)
        rates[prn] = a.rate_hz
        note = "  [OVERRIDE -- broker said %+.3f Hz]" % broker_rate
    print("scan target PRN %d: cn0_inc %.1f dB-Hz, duty %.2f, rate %+.3f Hz%s"
          % (prn, tgt["cn0_prompt_db"], tgt["cn0_prompt_duty"], rate_used, note))

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    time.sleep(a.seconds)
    print("\n  T_coh(s)   n_rec   eta     eta/n    cn0_coh    cn0_coh_sky")
    prev = None
    t_half = None
    n_folded = 0
    for nw in (1, 2, 4, 8, 16, 32):
        g = combdll.coh_cn0(cl, a.chain, rates=rates, n_win=nw, probe_prns=probes)
        v = (g or {}).get(prn)
        if not v:
            print("  (%d windows: no fold)" % nw)
            continue
        n_folded += 1
        eff = (v["eta"] / v["n_rec"]) if v["eta"] else None
        print("  %-10.3f %-7d %-7s %-8s %-10s %s"
              % (v["t_coh_s"], v["n_rec"],
                 "%.1f" % v["eta"] if v["eta"] else "--",
                 "%.2f" % eff if eff else "--",
                 "%.1f" % v["cn0_db"] if v["cn0_db"] is not None else "--",
                 "%.1f" % v["cn0_sky_db"] if v["cn0_sky_db"] is not None else "--"))
        if eff is not None and prev is not None and prev[1] >= 0.5 > eff and t_half is None:
            t_half = (prev[0], v["t_coh_s"])
        if eff is not None:
            prev = (v["t_coh_s"], eff)
    cl.stop()
    print()
    # ⚠️ NO DATA IS NOT A MEASUREMENT. Every row can print "no fold" because the telemetry
    # client never connected -- --gather wants HOST:PORT and takes a URL without complaint,
    # and the gather serves 11061 on 127.0.0.1 only, so an off-box run connects to nothing.
    # Both mistakes were made in one sitting (2026-08-15) and this function answered each
    # with the confident "not adding coherently even over one frame" verdict below, which is
    # the strongest claim it can make, drawn from zero records. The real scan, run correctly
    # moments later, showed eta/n = 0.85 at one frame -- the opposite. An estimator that
    # cannot measure must say so ([[an estimator that cannot measure must serve NOTHING]]).
    if not n_folded:
        print("⚠️ NO FOLD AT ANY LENGTH -- and that is a measurement of NOTHING, not of "
              "incoherence. The scan got zero records, so no verdict is available. Check "
              "the telemetry connected at all (the line above should read 'telem: connected "
              "to gather'): --gather takes HOST:PORT, not a URL, and the gather's serve port "
              "is bound to 127.0.0.1 -- run this ON the gather host.")
        return 1
    if t_half:
        print("COHERENCE TIME: eta/n crosses 0.5 between %.3f s and %.3f s -- coherent "
              "integration stops paying beyond that." % t_half)
    elif prev and prev[1] >= 0.5:
        print("COHERENCE TIME: eta/n still >= 0.5 at %.3f s (the longest fold tried) -- "
              "the carrier holds at least that long; scan further with --windows." % prev[0])
    else:
        print("⚠️ eta/n IS BELOW 0.5 AT EVERY LENGTH INCLUDING THE SHORTEST. The records "
              "are not adding coherently even over one frame, so this is not a coherence "
              "TIME at all -- suspect the injected rate (a residual error of df decoheres "
              "in ~1/(2 df) s) or a per-record phase discontinuity. Check kcoh_rate_hz "
              "against the chain's carrier residual before reading anything into the C/N0.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--broker", default="http://127.0.0.1:12060")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--windows", type=int, default=32,
                    help="fold length in windows (32 = ~1.34 s, the production value)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--coherence-scan", action="store_true",
                    help="HOW LONG DO WE ACTUALLY HOLD CARRIER COHERENCE? Fold the same "
                         "records at 1,2,4...32 windows and report eta/n vs T_coh. "
                         "eta/n ~ 1 means every record added coherently; the T where it "
                         "falls through ~0.5 IS the coherence time. ⚠️ Runs only on a "
                         "satellite the incoherent estimator HOLDS (duty >= --min-duty): "
                         "on a weak or intermittently-tracked satellite eta is a ratio of "
                         "two small noisy numbers and means nothing, which is exactly the "
                         "kind of reading that has been mistaken for physics here before.")
    ap.add_argument("--rate-hz", type=float, default=None,
                    help="coherence-scan: derotate at THIS rate instead of the broker's "
                         "injected one. 0 = no derotation. The scan's conclusion is "
                         "conditional on the rate, so this is how the rate itself is "
                         "tested rather than assumed.")
    ap.add_argument("--prn", type=int, default=None,
                    help="coherence-scan: pin the target PRN (default: strongest held) "
                         "so a paired rate A/B lands on the SAME satellite.")
    ap.add_argument("--min-duty", type=float, default=0.8)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.coherence_scan:
        return coherence_scan(a)

    # Probes + rates + the per-record reference, all from the broker's served rows.
    try:
        with urllib.request.urlopen("%s/get_status?chain=%s" % (a.broker, a.chain),
                                    timeout=5.0) as h:
            rows = json.loads(h.read().decode())
    except Exception as e:
        raise SystemExit("broker unreachable (%s)" % e)
    probes = {int(r["prn"]) for r in rows if r.get("noise_probe")}
    rates = {int(r["prn"]): r["rec_rate_hz"] for r in rows
             if r.get("rec_rate_hz") is not None}
    prompt = {int(r["prn"]): (r.get("cn0_prompt_db"), r.get("cn0_prompt_duty"))
              for r in rows}
    if not probes:
        raise SystemExit("broker reports no noise probes -- no floor, no gate")
    print("probes %s; rates for %d PRNs" % (sorted(probes), len(rates)))

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    time.sleep(a.seconds)
    got = combdll.coh_cn0(cl, a.chain, rates=rates, n_win=a.windows,
                          probe_prns=probes, keep_series=True)
    cl.stop()
    if not got:
        raise SystemExit("no folds (no windows for %s, or no probe floor)" % a.chain)

    rng = random.Random(3)
    print("\n  PRN     kcoh_db   sky_db    sig     eta/n     null_x   prompt   d_raw   d_sky")
    agree_raw, agree_sky, fails = [], [], []
    for prn in sorted(got):
        v = got[prn]
        is_probe = prn in probes
        ser = v.get("series") or {}
        nulls = [_fold_series(_shuffle_series(ser, rng), v["rate_hz"], 5.12e-6)
                 for _ in range(8)]
        null_med = sorted(x for x in nulls if x)[len([x for x in nulls if x]) // 2] \
            if any(nulls) else None
        pw = _fold_series(ser, v["rate_hz"], 5.12e-6)
        null_x = (pw / null_med) if (pw and null_med) else None
        cp, duty = prompt.get(prn, (None, None))
        d_raw = (v["cn0_db"] - cp) if (v["cn0_db"] is not None and cp is not None) else None
        d_sky = (v["cn0_sky_db"] - cp) if (v["cn0_sky_db"] is not None
                                           and cp is not None) else None
        print("  %s%-4d %8s %8s %7s %6s/%-4d %8s %8s %7s %7s"
              % ("P" if is_probe else "G", prn,
                 "%.1f" % v["cn0_db"] if v["cn0_db"] is not None else "--",
                 "%.1f" % v["cn0_sky_db"] if v["cn0_sky_db"] is not None else "--",
                 "%.0f" % v["sig"],
                 "%.0f" % v["eta"] if v["eta"] is not None else "--", v["n_rec"],
                 "%.0fx" % null_x if null_x is not None else "--",
                 "%.1f" % cp if cp is not None else "--",
                 "%+.1f" % d_raw if d_raw is not None else "--",
                 "%+.1f" % d_sky if d_sky is not None else "--"))
        if is_probe:
            if null_x is not None and null_x > 5.0:
                fails.append("probe %d fold %.0fx its shuffled null" % (prn, null_x))
            continue
        strong = duty is not None and duty >= 0.9 and cp is not None
        if strong:
            if d_raw is not None:
                agree_raw.append(d_raw)
            if d_sky is not None:
                agree_sky.append(d_sky)
            if null_x is not None and null_x < 10.0:
                fails.append("G%d fold only %.1fx its null at prompt duty %.2f"
                             % (prn, null_x, duty))

    def _spread(ds, name):
        if len(ds) < 2:
            print("%s: INCONCLUSIVE (%d strong satellites)" % (name, len(ds)))
            return
        ds = sorted(ds)
        med = ds[len(ds) // 2]
        sc = max(abs(x - med) for x in ds)
        verdict = "PASS" if sc <= 1.0 else "FAIL"
        print("%s: offset %+.2f dB (the measured convention), scatter %.2f dB -> %s"
              % (name, med, sc, verdict))
        if verdict == "FAIL":
            fails.append("%s scatter %.2f dB" % (name, sc))

    print()
    _spread(agree_raw, "AGREE raw (kcoh - prompt)")
    _spread(agree_sky, "AGREE sky (kcoh_sky - prompt)")
    if fails:
        print("FAIL: " + "; ".join(fails))
        return 1
    print("NULL/PROBE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
