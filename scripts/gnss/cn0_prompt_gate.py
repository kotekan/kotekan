#!/usr/bin/env python3
"""THE VALIDATION BAR FOR THE SERVED C/N0 (task #57): combdll.prompt_cn0 on sky, three legs.

The estimator this judges replaces the deep fold's radiometry. The fold re-searched a
residual rate per integration -- a fit on something the tracking loop already fixed -- and
carried ~20 dB of its OWN scatter, measured PAIRED on the same records (>10x on 23% of
cycles). The bar below is written so that failure mode, and the ones found hunting it,
cannot ship again silently:

  AUC     Do the estimator's per-record inputs SEPARATE a satellite from the below-horizon
          probes? deep_snr did not (a PRN seeded 64 deg below the horizon read 11.2x its
          floor); bar >= 0.90 per served satellite. Each probe is also scored against the
          OTHER probes' pool -- a probe at AUC >~ 0.9 IS the fires-on-noise disease.

  SPLIT   Even/odd-record self-consistency of the served number, from the estimator itself
          (cn0_prompt_split_db). Bar: |split| <= 1.0 dB on a held satellite. This is the
          single-draw rule made mechanical: the value ships WITH its own paired witness.

  PAIRED  The same debiased-ratio estimator on the OTHER feed (/get_status's EMA'd
          e/p/l_pow, the polled arm) against the telemetry arm's UNGATED mean -- ungated
          on purpose: the EMA cannot be q-gated per record, so gate-vs-EMA differences are
          duty, not disagreement. Bar: <= 1.0 dB on a held satellite. The paired two-feed
          ratio is the sharpest instrument this codebase has (sky cancels); it is the
          measurement that convicted deep_snr, run here as a standing gate.

A leg with no qualifying satellite reports INCONCLUSIVE, never PASS -- an experiment that
cannot fail is not evidence (see chord-a-gate-that-cannot-fail).

    ./cn0_prompt_gate.py --self-test          # offline: known SNR in -> cn0 out, exact
    ./cn0_prompt_gate.py --seconds 25         # on the gather host, alongside the broker

Probes are auto-discovered from the broker's /get_status (`noise_probe` rows, served once
the #57 broker is running); --probes overrides for older brokers (the seed log line
"noise probe PRN %d seeded" names them).
"""
import argparse
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


def auc(pos, neg):
    """P(pos draw > neg draw), ties at 0.5 -- the Mann-Whitney AUC, no dependencies."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r, i, rank_sum = 0, 0, 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)          # 1-based average rank of the tie group
        rank_sum += avg_rank * sum(lab for _v, lab in allv[i:j])
        i = j
    u = rank_sum - 0.5 * len(pos) * (len(pos) + 1)
    return u / (len(pos) * len(neg))


def discover_probes(broker, chain):
    """PRNs flagged noise_probe in the broker's served rows, or None if it cannot say."""
    try:
        with urllib.request.urlopen("%s/get_status?chain=%s" % (broker, chain),
                                    timeout=5.0) as h:
            rows = json.loads(h.read().decode())
    except Exception as e:
        print("broker %s unreachable (%s)" % (broker, e))
        return None
    flagged = {int(r["prn"]) for r in rows if r.get("noise_probe")}
    if flagged:
        return flagged
    if any("noise_probe" in r for r in rows):
        return set()      # the broker knows the flag and flags nobody: probes are off
    return None           # pre-#57 broker: the rows cannot say


def _poll_once(endpoints, min_instances):
    """{prn: per-instance-mean p_pow} from one /get_status sweep of the fleet."""
    acc = {}
    for url in endpoints:
        try:
            with urllib.request.urlopen("%s/get_status" % url, timeout=5.0) as h:
                rows = json.loads(h.read().decode())
        except Exception:
            continue
        for r in rows or []:
            if int(r.get("pow_fft_len", 0)) <= 0:
                continue
            p = float(r.get("p_pow", 0.0))
            if p <= 0.0:
                continue
            d = acc.setdefault(int(r["prn"]), [0.0, 0])
            d[0] += p
            d[1] += 1
    return {prn: v[0] / v[1] for prn, v in acc.items() if v[1] >= min_instances}


def polled_cn0(polls, probes, t_rec):
    """The PAIRED arm: the identical debiased-ratio estimator on the /get_status feed.

    `polls` is a list of _poll_once() sweeps SPREAD ACROSS the telemetry capture. One sweep
    is not enough, for two measured reasons (gps_l5, 2026-08-15, +1.1..+1.9 dB common-mode
    gaps): the EMA behind p_pow covers ~1 s, so a single end-of-capture sweep pairs one
    second of sky against the telemetry arm's ~25 s -- a time mismatch, not a disagreement
    -- and the noise anchor from ONE sweep is a median of ~3 probe rows, carrying ~1 dB of
    its own draw. Averaging the sweeps per PRN and pooling every sweep's probe rows fixes
    both without touching either estimator.
    """
    acc = {}
    for m in polls:
        for prn, p in m.items():
            acc.setdefault(prn, []).append(p)
    means = {prn: sum(v) / len(v) for prn, v in acc.items()}
    pooled = [p for prn in probes for p in acc.get(prn, ())]
    if len(pooled) < 2:
        return {}, None, {}
    pooled.sort()
    med = pooled[len(pooled) // 2]
    kept = [x for x in pooled if x <= 8.0 * med]     # same clipped MEAN as the estimator
    s2 = sum(kept) / len(kept)
    if s2 <= 0.0:
        return {}, None, {}
    out, se_db = {}, {}
    for prn, p in means.items():
        rho = (p - s2) / s2
        out[prn] = 10.0 * math.log10(rho / t_rec) if rho > 0.0 else None
        # This arm's OWN precision, published with its value: the standard error of the
        # sweep means, in dB. A flickering satellite (measured: G23 at duty 0.64, sweeps
        # spanning ~4 dB) gives this arm a standard error above the pair bar itself, and
        # then no agreement claim is testable either way -- the caller must mark the pair
        # unscoreable rather than failed or passed. Judged on the arm's precision, never
        # on the outcome.
        v = acc[prn]
        if len(v) >= 3 and all(x > 0.0 for x in v):
            ldb = [10.0 * math.log10(x) for x in v]
            m = sum(ldb) / len(ldb)
            var = sum((x - m) ** 2 for x in ldb) / (len(ldb) - 1)
            se_db[prn] = (var / len(ldb)) ** 0.5
    return out, s2, se_db, means


# ---------------------------------------------------------------------------------------
# SELF-TEST: a fake client with a KNOWN SNR, through the shipped estimator untouched.
# rho_true = n_chan * |s|^2 / sigma_ch^2 (the channel mean divides the noise by n_chan and
# the probes measure exactly that reduced floor), so cn0_true is known in closed form and
# the check is absolute -- never a self-comparison (gnss-float-contraction rule).
# ---------------------------------------------------------------------------------------
class _FakeFrame(object):
    def __init__(self, rng, win, inst, chans, sats, sig_amp, noise_sig, n_rec=4):
        self.n_rec, self.hops_per_record = n_rec, 2048
        self._rng, self._chans, self._sats = rng, chans, sats
        self._sig, self._noise = sig_amp, noise_sig
        self._prns = sorted(set(p for p, _on in sats))

    def has_record(self, r):
        return True

    def prns(self):
        return self._prns

    def comb_epl(self, r, prn):
        on = dict(self._sats).get(prn, False)
        out = []
        for fid in self._chans:
            def n():
                return complex(self._rng.gauss(0.0, self._noise),
                               self._rng.gauss(0.0, self._noise))
            s = self._sig if on else 0.0
            # E = L = pure noise (a locked tap: the shoulders hold no power at 0.5-chip
            # spacing only approximately, but for the ARITHMETIC check E/L only feed q).
            out.append((fid, n(), s + n(), n(), (1.0, 1.0, 1.0)))
        return out


class _FakeClient(object):
    def __init__(self, seed=1, n_win=64, n_inst=3, chans_per_inst=6,
                 sig_amp=3.0, noise_sig=1.0, sats=None):
        self._rng = random.Random(seed)
        self._wins = list(range(1000, 1000 + n_win))
        self._frames = {}
        sats = sats or [(23, True), (7, True), (91, False), (92, False), (93, False)]
        for w in self._wins:
            for i in range(n_inst):
                chans = [i * chans_per_inst + c for c in range(chans_per_inst)]
                self._frames[(w, "cx%d.0" % i)] = _FakeFrame(
                    self._rng, w, i, chans, sats, sig_amp, noise_sig)
        self.n_chan_total = n_inst * chans_per_inst

    def windows(self, chain, lag=1):
        return list(self._wins)

    def frame_set(self, chain, win):
        return {inst: f for (w, inst), f in self._frames.items() if w == win}


def self_test():
    sig_amp, noise_sig, cpi = 3.0, 1.0, 6
    fc = _FakeClient(sig_amp=sig_amp, noise_sig=noise_sig, chans_per_inst=cpi)
    t_rec = 2048 * 5.12e-6
    got = combdll.prompt_cn0(fc, "fake", n_win=64, probe_prns={91, 92, 93},
                             keep_records=True)
    if not got:
        print("SELF-TEST: FAIL -- estimator returned nothing")
        return 1
    # per-channel complex noise variance 2*noise_sig^2; the cross-channel mean divides it
    # by n_chan per instance -- and the instance mean leaves it there (each instance is an
    # independent draw at the same level, signal and noise alike).
    rho_true = sig_amp ** 2 / (2.0 * noise_sig ** 2 / cpi)
    cn0_true = 10.0 * math.log10(rho_true / t_rec)
    fails = []
    for prn in (23, 7):
        v = got.get(prn)
        if not v or v["cn0_db"] is None:
            fails.append("PRN %d: no cn0 served" % prn)
            continue
        err = v["cn0_db"] - cn0_true
        print("SELF-TEST PRN %d: cn0 %.2f dB-Hz (truth %.2f, err %+.2f) duty %.2f "
              "split %s" % (prn, v["cn0_db"], cn0_true, err, v["duty"],
                            "%+.2f" % v["split_db"] if v["split_db"] is not None else "--"))
        # 0.3 dB: the mean-vs-median debias bias this test caught was +0.7 dB and a
        # loosened bar is how it would sneak back
        if abs(err) > 0.3:
            fails.append("PRN %d: cn0 off truth by %+.2f dB" % (prn, err))
        if v["duty"] < 0.9:
            fails.append("PRN %d: duty %.2f on a clean strong signal" % (prn, v["duty"]))
    for prn in (91, 92, 93):
        v = got.get(prn)
        if v and v["cn0_db"] is not None and v["duty"] > 0.2:
            fails.append("probe %d SERVED a cn0 at duty %.2f -- fires on noise"
                         % (prn, v["duty"]))
    # the AUC leg on the synthetic records: satellite vs pooled probe rho
    probe_rho = [x[2] for p in (91, 92, 93) for x in got[p]["recs"]]
    a = auc([x[2] for x in got[23]["recs"]], probe_rho)
    print("SELF-TEST AUC (PRN 23 vs probes): %.3f" % a)
    if a < 0.99:
        fails.append("AUC %.3f on a strong synthetic signal" % a)
    if fails:
        print("SELF-TEST: FAIL\n  " + "\n  ".join(fails))
        return 1
    print("SELF-TEST: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--broker", default="http://127.0.0.1:12060",
                    help="publisher, for probe discovery (config publish-port)")
    ap.add_argument("--probes", default="",
                    help="CSV probe PRNs; overrides broker discovery")
    ap.add_argument("--nodes", default="cx19,cx27,cx42,cx43,cx44,cx51")
    ap.add_argument("--min-duty", type=float, default=0.5,
                    help="a PRN below this gate duty is not judged (its cn0 is declinable)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    if a.probes:
        probes = {int(x) for x in a.probes.split(",")}
    else:
        probes = discover_probes(a.broker, a.chain)
        if probes is None:
            raise SystemExit("broker rows carry no noise_probe flag (pre-#57 broker?) -- "
                             "pass --probes; the seed log names them "
                             "('noise probe PRN %d seeded')")
        if not probes:
            raise SystemExit("broker reports no noise probes seeded -- the estimator has "
                             "no noise anchor; check `noise-probes:` in the chain yaml")
    print("probes: %s" % sorted(probes))

    # ⚠️ COMBINER NAMES ARE PER CHAIN (gnss{0,1}[_<band>]_n2combine). The first run of this
    # tool polled gps_l5's combiners while judging gal_e5b -- no probe rows matched, no pair
    # was computable, and the leg printed PASS on zero data: a gate that cannot fail. The
    # suffix map below and the per-leg judged counts are both consequences of that run.
    _sfx = {"gps_l5": "", "gal_e5a": "_e5a", "bds_b2a": "_b2a",
            "gal_e5b": "_e5b", "bds_b2b": "_b2b"}.get(a.chain)
    if _sfx is None:
        raise SystemExit("unknown chain %r -- no combiner suffix known" % a.chain)
    eps = ["http://%s:12049/gnss%d%s_n2combine" % (n, g, _sfx)
           for n in a.nodes.split(",") for g in (0, 1)]

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    # The polled sweeps ride INSIDE the capture window so the two arms describe the same
    # sky -- and inside its LAST `PAIR_S` seconds specifically, because the telemetry ring
    # (depth 4096 frames / ~12 instances) retains only ~14 s: sweeps spread over a longer
    # capture average an EARLIER span than the ring keeps, and on a rising satellite that
    # alone read as a +1.1..+1.6 dB "disagreement" on every strong PRN (gps_l5,
    # 2026-08-15 15:02). The pair's telemetry mean is computed over the same last-PAIR_S
    # slice below.
    PAIR_S = 12.0
    polls, n_sweeps = [], 6
    t0 = time.time()
    _pt0 = t0 + max(2.0, a.seconds - PAIR_S)
    _pspan = t0 + a.seconds - _pt0
    for i in range(n_sweeps):
        time.sleep(max(0.0, _pt0 + (i + 0.5) * _pspan / n_sweeps - time.time()))
        polls.append(_poll_once(eps, min_instances=2))
    time.sleep(max(0.0, t0 + a.seconds - time.time()))
    got = combdll.prompt_cn0(cl, a.chain, n_win=a.windows, probe_prns=probes,
                             keep_records=True)
    cl.stop()
    if not got:
        raise SystemExit("no estimate: no windows for %s, or < 16 probe records "
                         "(gather down? probes not despread?)" % a.chain)
    t_rec = next(iter(got.values()))["t_rec_s"]
    poll, poll_s2, poll_se, poll_p = polled_cn0(polls, probes, t_rec)

    probe_rho = [x[2] for p in probes for x in (got.get(p, {}).get("recs") or [])]
    sats = sorted(p for p in got if p not in probes)
    v0 = got[sats[0]] if sats else next(iter(got.values()))
    print("\n%d PRNs, %d probe records, sigma2 %.3e, q_gate %.2f, t_rec %.4f s"
          % (len(got), v0["n_probe_rec"], v0["sigma2"], v0["q_gate"], t_rec))
    print("\n  PRN     cn0_db   duty  n_used  split_dB    AUC   polled_dB  pair_dB")
    legs = {"auc": [], "split": [], "pair": []}      # failures per leg
    judged = {"auc": [], "split": [], "pair": []}    # PRNs each leg could actually score
    decomp = []                                      # (prn, signal ratio dB, pair gap dB)
    # THE ANCHOR CONVENTION OFFSET, measured per run from the probes themselves. The two
    # feeds' noise anchors sit a constant apart (gps_l5 2026-08-15: -1.43 dB, the comb
    # BELOW /get_status's p_pow) while their per-record powers agree to 0.07 dB on the
    # same samples -- probe_anchor_ab.py is the instrument that separated those, and it
    # exonerated #62. A constant convention inside one serving layer is not an estimator
    # disagreement (the coh_source_ab 6.34e9 precedent), so the pair verdict is judged on
    # the CONVENTION-CORRECTED gap; both raw and anchor are printed so the correction is
    # never silent.
    anch_db = None
    if poll_s2:
        anch_db = 10.0 * math.log10(next(iter(got.values()))["sigma2"] / poll_s2)
    for prn in sats + sorted(p for p in got if p in probes):
        v = got[prn]
        is_probe = prn in probes
        rho_rec = [x[2] for x in v["recs"]]
        # probes are scored against the OTHER probes' pool -- self-inclusion would drag
        # every probe's AUC toward 0.5 and hide a hot one
        neg = ([x for p in probes if p != prn
                for x in (got.get(p, {}).get("recs") or [])] if is_probe else None)
        A = auc(rho_rec, [x[2] for x in neg] if is_probe else probe_rho)
        # the paired leg compares UNGATED means over the SAME last-PAIR_S span the sweeps
        # covered (see the sweep-schedule comment in main)
        _k = max(8, int(PAIR_S / t_rec))
        rho_pair = rho_rec[-_k:]
        rho_all = sum(rho_pair) / len(rho_pair) if rho_pair else None
        tele_ug = (10.0 * math.log10(rho_all / t_rec)
                   if rho_all and rho_all > 0.0 else None)
        pol = poll.get(prn)
        pair = (tele_ug - pol) if (tele_ug is not None and pol is not None) else None
        # DON'T FIT ACROSS A TRANSIENT: first-half vs second-half of the ungated series,
        # measured from the arm itself. During lock churn (measured on gps_l5 mid-pull-in,
        # 2026-08-15: G10 duty 0.97 -> 0.00 between runs; pair gaps +-1.6 dB BOTH signs)
        # the two arms weight time differently and a gap is expected, not a disagreement --
        # so a PRN whose own level moved > 1 dB within the capture is marked '~' and the
        # pair leg reports it as unscoreable rather than failed OR passed.
        drift = None
        if len(rho_pair) >= 8:
            h1 = sum(rho_pair[:len(rho_pair) // 2]) / (len(rho_pair) // 2)
            h2 = sum(rho_pair[len(rho_pair) // 2:]) / (len(rho_pair) - len(rho_pair) // 2)
            if h1 > 0.0 and h2 > 0.0:
                drift = 10.0 * math.log10(h2 / h1)
        # Scoreable = the sky held still (drift) AND the polled arm can resolve the bar
        # (its own standard error across sweeps under 0.5 dB).
        _se = poll_se.get(prn)
        stationary = (drift is not None and abs(drift) <= 1.0
                      and _se is not None and _se <= 0.5)
        print("  %s%-4d %8s  %5.2f  %6d  %8s  %5s  %9s  %7s%s"
              % ("P" if is_probe else "G", prn,
                 "%.1f" % v["cn0_db"] if v["cn0_db"] is not None else "--",
                 v["duty"], v["n_used"],
                 "%+.2f" % v["split_db"] if v["split_db"] is not None else "--",
                 "%.3f" % A if A is not None else "--",
                 "%.1f" % pol if pol is not None else "--",
                 "%+.2f" % pair if pair is not None else "--",
                 "" if (pair is None or stationary)
                 else "~ (drift %s, poll se %s)"
                 % ("%+.1f dB" % drift if drift is not None else "--",
                    "%.2f dB" % _se if _se is not None else "--")))
        if is_probe:
            if A is not None and A >= 0.9:
                legs["auc"].append("probe %d AUC %.3f vs its peers -- FIRES ON NOISE"
                                   % (prn, A))
            if v["cn0_db"] is not None and v["duty"] > 0.2:
                legs["auc"].append("probe %d served cn0 %.1f at duty %.2f"
                                   % (prn, v["cn0_db"], v["duty"]))
            continue
        if v["cn0_db"] is None or v["duty"] < a.min_duty:
            continue      # declinable by its own published duty; not judged
        # A leg only judges a PRN it could actually SCORE -- a missing input must surface
        # as INCONCLUSIVE, never ride a non-empty overall set into PASS.
        if A is not None:
            judged["auc"].append(prn)
            if A < 0.90:
                legs["auc"].append("PRN %d AUC %.3f < 0.90" % (prn, A))
        if v["split_db"] is not None:
            judged["split"].append(prn)
            if abs(v["split_db"]) > 1.0:
                legs["split"].append("PRN %d split %+.2f dB" % (prn, v["split_db"]))
        if pair is not None and stationary and anch_db is not None:
            judged["pair"].append(prn)
            if abs(pair + anch_db) > 1.0:
                legs["pair"].append("PRN %d corrected gap %+.2f dB (raw %+.2f)"
                                    % (prn, pair + anch_db, pair))
        # DECOMPOSE the pair gap for this PRN: gap = (numerator ratio) - (anchor ratio).
        # A uniform gap across satellites with the numerators agreeing convicts the
        # ANCHORS; numerators moving convicts the record paths themselves (#62 territory).
        if (pair is not None and rho_all is not None and rho_all > 0.0
                and prn in poll_p and poll_s2):
            _s_tele = (rho_all + 1.0) * v["sigma2"]
            decomp.append((prn, 10.0 * math.log10(_s_tele / poll_p[prn]), pair))

    if decomp and anch_db is not None:
        print("\nDECOMPOSITION: anchor ratio (tele sigma2 / polled s2) %+.2f dB "
              "(a serving-layer convention -- pair verdicts are corrected by it); "
              "signal ratios (tele/polled): %s"
              % (anch_db, "  ".join("%s%d %+.2f" % ("P" if p in probes else "G", p, r)
                                    for p, r, _g in decomp)))
    print()
    rc = 0
    for name, bar in (("auc", "AUC >= 0.90 (+ no probe fires)"),
                      ("split", "|split| <= 1.0 dB"), ("pair", "two-feed <= 1.0 dB")):
        if legs[name]:
            print("%-6s FAIL (%s): %s" % (name.upper(), bar, "; ".join(legs[name])))
            rc = 1
        elif not judged[name]:
            print("%-6s INCONCLUSIVE: no satellite it could score at duty >= %.2f"
                  % (name.upper(), a.min_duty))
        else:
            print("%-6s PASS (%s) on %s" % (name.upper(), bar,
                                            ", ".join("G%d" % p for p in judged[name])))
    if poll_s2 is None:
        print("PAIRED note: polled arm had < 2 probe rows -- pair column empty")
    return rc


if __name__ == "__main__":
    sys.exit(main())
