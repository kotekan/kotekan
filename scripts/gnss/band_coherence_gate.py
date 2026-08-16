#!/usr/bin/env python3
"""DO THE FREQUENCY CHANNELS COHERE ACROSS THE WHOLE BAND? -- measurement only, nothing changes.

THE REDUCTION WE SHIP TODAY collapses the phase one step too early. Per record, per PRN, the
despread hands up a COMPLEX prompt PER FREQUENCY CHANNEL. Both arms -- combdll.prompt_cn0 /
coh_cn0 in the broker, and gnss::FleetDll in the gather -- then do:

    within one instance:   gP += P * wP            (coherent over that instance's channels)
    across instances:      t.p += |gP/wP|^2        (POWER, phase discarded)

⚠️ AN INSTANCE IS AN ARBITRARY GROUP OF FREQUENCY CHANNELS -- `freq_id mod 8`, applied AFTER
the signal path (one PFB, one set of raw samples). There is nothing physical at that boundary
and channels inside one instance do not cohere any better than channels across two. So the
split point of that coherent/incoherent switch is meaningless, and the phase thrown away at it
is real, usable phase: #32 fits the carrier phase ACROSS THE BAND, which is exactly a
cross-instance phase measurement.

WHAT THIS MEASURES. Both reductions on the SAME records:

    P_pow  = sum_inst |sum_{ch in inst} g_ch|^2 / (...)      <- what ships
    P_coh  = |sum_{all ch, all inst} g_ch|^2 / (...)         <- the whole band, one sum
    eta_band = P_coh / P_pow

Both are normalised by their own total weight, so they estimate the SAME normalised power:
aligned phases give eta_band -> 1, and completely unrelated phases give eta_band -> 1/M
(summing M random complex vectors gains M in power, not M^2). ⚠️ NOT "-> M": that was the
first version of this note and it is wrong -- coherent combining does not raise the normalised
signal power, it lowers the NOISE. A gate expecting M would call a healthy band broken.

⚠️ THE VALUE OF THE C/N0 DOES NOT MOVE EITHER WAY, and a gate expecting it to would fail
against a correct fix. Writing A_i = s + n_i: the incoherent mean is |s|^2 + sigma^2 and the
coherent one is |s|^2 + sigma^2/M, so AFTER the probe debias both estimate |s|^2. What
improves is the VARIANCE -- the residual noise falls by M. This is a sensitivity change, not
a calibration change.

⚠️ AND THE INCOHERENT AXIS IS TIME, NEVER INSTANCE. The probe floor has to be formed with the
IDENTICAL reduction as the signal it debiases, then averaged over RECORDS, which are genuinely
independent draws. Averaging |.|^2 over instances instead is not a variance estimate of
anything -- it is the arbitrary grouping leaking into the statistic.

NULL: the noise probes. They have no carrier, so their eta_band must sit at ~1 however the
channels are grouped. A satellite that beats its probes is the evidence; a satellite that does
not means the band is NOT coherent and the reduction cannot simply be switched.

    ./band_coherence_gate.py --chain gps_l5        # on the gather host (cf06)
"""
import argparse
import cmath
import json
import math
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker import telem  # noqa: E402

# freq_id -> RF frequency. CHORD samples at 3200 MHz into N=8192 channels, so a channel is
# 3200/2/8192 = 0.195312 MHz. N is PINNED, not assumed: of 2048/4096/8192/16384 only 8192
# puts L5 (1176.45 MHz) inside the freq_ids this signal actually occupies (5972..6076) --
# fid 6023.4, dead centre. The others land at 1506 / 3012 / 12047 and are excluded.
CHAN_HZ = 3200.0e6 / 2 / 8192
CHIP_HZ = 10.23e6


def reductions(client, chain, wins, prns, tau_s=None):
    """As below; `tau_s` = {prn: residual delay in seconds} derotates each channel by
    exp(-2i pi f_k tau) BEFORE summing.

    ⚠️ THE RAMP IS A DELAY, AND WE ALREADY MEASURE THE DELAY. A residual code error tau puts a
    LINEAR PHASE RAMP 2*pi*f_k*tau across the channels, and nothing removes it today: the
    per-channel prompt is NCO-derotated (carrier gone) but carries the delay phase, and #32's
    band phase-slope fit is labelled "measurement-only" in the chain config. Over the 20.3 MHz
    this signal occupies, 0.1 chip of delay error is 1.2 rad of ramp end-to-end and 0.5 chip is
    6.1 rad -- so a coherent sum cannot survive until it is taken out.
    ⚠️ DERIVE IT, NEVER FIT IT. tau comes from the tracker's own discriminator. Fitting a free
    slope per record would be the combine fitting what it should derive -- the failure already
    on record in this codebase. The scan below exists ONLY as a diagnostic, with a null.
    ⚠️ AND IT BITES THE CURRENT REDUCTION TOO. `freq_id mod 8` DECIMATES the band across
    instances rather than splitting it into contiguous blocks, so one instance's channels span
    nearly the same 20.3 MHz as the whole fleet's. The ramp therefore damages today's
    per-instance coherent sum about as much as it would damage a full-band one.
    """
    """{prn: (P_pow, P_coh, n_rec, n_chan_total, n_inst)} summed over records.

    Both reductions are formed from the SAME per-channel complex values in the same pass, so
    nothing but the grouping differs between them.
    """
    out = {}
    for w in wins:
        fs = client.frame_set(chain, w)
        if not fs:
            continue
        n_rec = max((f.n_rec for f in fs.values()), default=0)
        for r in range(n_rec):
            per_prn = {}
            for inst, f in fs.items():
                if r >= f.n_rec or not f.has_record(r):
                    continue
                for prn in f.prns():
                    if prn not in prns:
                        continue
                    cmb = f.comb_epl(r, prn)
                    if not cmb:
                        continue
                    gi, wi = 0j, 0.0
                    t = (tau_s or {}).get(prn)
                    for _fid, _E, P, _L, (_wE, wP, _wL) in cmb:
                        v = P
                        if t:
                            # phase relative to an arbitrary reference fid: a CONSTANT phase
                            # does not change |sum|, only the slope matters.
                            v = v * cmath.exp(-2j * math.pi * (_fid - 6024) * CHAN_HZ * t)
                        gi += v * wP
                        wi += wP
                    if wi <= 0.0:
                        continue
                    d = per_prn.setdefault(prn, {"g": 0j, "w": 0.0, "pow": 0.0,
                                                 "nch": 0, "ninst": 0})
                    d["g"] += gi                      # keep it complex: the WHOLE band
                    d["w"] += wi
                    d["pow"] += abs(gi / wi) ** 2     # what ships: per-instance, then power
                    d["nch"] += len(cmb)
                    d["ninst"] += 1
            for prn, d in per_prn.items():
                if d["ninst"] < 2 or d["w"] <= 0.0:
                    continue    # a single instance makes the two reductions identical
                o = out.setdefault(prn, [0.0, 0.0, 0, 0, 0])
                o[0] += d["pow"] / d["ninst"]              # mean over instances
                o[1] += abs(d["g"] / d["w"]) ** 2         # one coherent sum, whole band
                o[2] += 1
                o[3] += d["nch"]
                o[4] = max(o[4], d["ninst"])
    return out


def per_instance_phase(client, chain, wins, prns):
    """{prn: {inst: (mean_phase_rad, |vector mean|, n)}} -- each instance's prompt phase
    RELATIVE TO the same record's full-band sum, vector-averaged over records.

    Referencing against the record's own total removes the sky/carrier phase that is common to
    every instance at that instant, leaving only what differs BETWEEN instances -- which by the
    lockstep rule should be nothing. |vector mean| ~ 1 means a stable per-instance constant
    (measurable, removable); ~0 means it re-rolls per record and there is nothing to calibrate.
    """
    import cmath
    acc = {}
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
                    cmb = f.comb_epl(r, prn)
                    if not cmb:
                        continue
                    gi = 0j
                    for _fid, _E, P, _L, (_wE, wP, _wL) in cmb:
                        gi += P * wP
                    if gi != 0j:
                        per.setdefault(prn, {})[inst] = gi
            for prn, gs in per.items():
                if len(gs) < 2:
                    continue
                tot = sum(gs.values())
                if tot == 0j:
                    continue
                for inst, gi in gs.items():
                    # unit phasor of (this instance) vs (the whole band this record)
                    z = gi * tot.conjugate()
                    if z == 0j:
                        continue
                    d = acc.setdefault(prn, {}).setdefault(inst, [0j, 0])
                    d[0] += z / abs(z)
                    d[1] += 1
    out = {}
    for prn, per in acc.items():
        for inst, (v, n) in per.items():
            if n:
                out.setdefault(prn, {})[inst] = (cmath.phase(v / n), abs(v / n), n)
    return out


def self_test():
    """A known ramp, planted and removed. No sky, no fleet, no excuses."""
    fids = [5972 + 8 * k for k in range(14)]          # one instance: DECIMATED across the band
    fail = 0

    def band(tau_true, tau_removed):
        """|sum g|^2 / mean|g|^2 for unit channels carrying a tau_true ramp, derotated by
        tau_removed. 1.0 = perfectly aligned, ~0 = ramp intact."""
        g = 0j
        for fid in fids:
            ph = 2 * math.pi * (fid - 6024) * CHAN_HZ * (tau_true - tau_removed)
            g += cmath.exp(1j * ph)
        return abs(g / len(fids)) ** 2

    def chk(ok, what, got, want):
        nonlocal fail
        if not ok:
            fail += 1
            print("  FAIL %-52s got %.4f want %s" % (what, got, want))
        else:
            print("  ok   %-52s %.4f" % (what, got))

    for dchip in (0.1, 0.3, 0.5):
        tau = dchip / CHIP_HZ
        raw = band(tau, 0.0)
        fixed = band(tau, tau)
        wrong = band(tau, -tau)          # sign flipped: doubles the ramp
        chk(fixed > 0.999, "%.1f chip: derotated by the TRUE tau -> aligned" % dchip, fixed, "~1")
        chk(raw < fixed, "%.1f chip: raw ramp costs coherence" % dchip, raw, "< fixed")
        # ⚠️ THE SIGN CHECK ONLY BINDS WHILE THE RESPONSE IS MONOTONIC. Across this band one
        # chip of delay is 1.99 cycles, so 0.5 chip already sits in the first null: raw (~1
        # cycle) and wrong-sign (~2 cycles) are BOTH ~0.004 and comparing them is comparing
        # two nulls. My first bar asserted wrong < raw everywhere and duly failed at 0.5 chip
        # -- the BAR was wrong, not the code (the same trap as carrier_nco_gate's leg 0).
        # What binds at every delay is that the wrong sign fails to ALIGN.
        # The invariant that binds at EVERY delay: the TRUE tau is the maximum, so any other
        # tau -- including the sign-flipped one -- scores strictly less. Anything stronger
        # (a fixed fraction) is delay-dependent and produced two wrong bars already.
        chk(wrong < fixed - 1e-6, "%.1f chip: WRONG SIGN scores below the true tau" % dchip,
            wrong, "< fixed")
        if dchip <= 0.125:   # monotonic while 2*tau*span < ~0.5 cycle
            chk(wrong < raw, "%.1f chip: wrong sign worse than raw (monotonic regime)" % dchip,
                wrong, "< raw")
    chk(band(0.0, 0.0) > 0.999, "zero delay is left alone", band(0.0, 0.0), "~1")
    # the 20.3 MHz span means a chip of error is catastrophic -- state it as a number
    print("\n  ramp across the used band for 1 chip of delay error: %.2f cycles"
          % ((6076 - 5972) * CHAN_HZ / CHIP_HZ))
    print("\nband_coherence_gate self-test: %s" % ("FAILED" if fail else "PASS"))
    return 1 if fail else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--broker", default="http://127.0.0.1:12060")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--min-duty", type=float, default=0.5,
                    help="duty bar for 'held'. Lower it to inspect a fleet that is acquiring "
                         "but not holding -- the per-instance PHASE check only needs signal, "
                         "not a sustained lock, though eta_band still wants a real hold.")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic: plant a KNOWN delay ramp across the channels and check "
                         "the derotation removes it, that the WRONG SIGN makes it worse, and "
                         "that a zero-delay band is left alone. Catches the sign and scale "
                         "errors that would otherwise be discovered by a wasted sky run.")
    ap.add_argument("--derotate", action="store_true",
                    help="remove the across-band phase ramp before summing, using the delay "
                         "DERIVED from the tracker's own discriminator (tau = dll_tau(disc)). "
                         "This is step 2 of #72: the coherent combine cannot work until the "
                         "ramp is out, and the ramp is a delay we already measure.")
    ap.add_argument("--scan-tau", action="store_true",
                    help="DIAGNOSTIC ONLY, never an estimator: also scan tau for the value "
                         "that maximises the coherent sum, and report it beside the DERIVED "
                         "one with the probes as a null. A scan always finds a maximum (that "
                         "is how the deep fold read 41 dB-Hz on noise), so it is here to "
                         "check whether the derived tau is near the optimum -- not to replace "
                         "it.")
    ap.add_argument("--per-instance", action="store_true",
                    help="also report each instance's PHASE relative to a reference instance, "
                         "and its stability over records. This is what separates a FIXABLE "
                         "per-instance constant (a calibration/timing offset -- measure it and "
                         "remove it, then the band coheres) from a per-record random phase "
                         "(nothing to calibrate; the fault is upstream). Vector-averaged, so "
                         "|mean| near 1 = stable, near 0 = random.")
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
    if not held:
        raise SystemExit("INCONCLUSIVE: nothing held at duty >= %.2f on %s."
                         % (a.min_duty, a.chain))
    if not probes:
        raise SystemExit("no noise probes -- no null, no gate")
    cn0 = {int(x["prn"]): x.get("cn0_prompt_db") for x in rows}

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    time.sleep(a.seconds)
    wins = cl.windows(a.chain, lag=1)[-a.windows:]
    # tau DERIVED from the tracker's discriminator: dll_tau(disc, spacing) in chips -> seconds.
    # Same expression as combdll.dll_tau / gnss::dll_tau (|tau| <= 0.25 chips by construction).
    disc = {int(x["prn"]): (x.get("dll_disc") or 0.0) for x in rows}
    spacing = 0.5
    tau_s = None
    if a.derotate:
        tau_s = {p: (-max(-1.0, min(1.0, d)) / 4.0 * (spacing / 0.5)) / CHIP_HZ
                 for p, d in disc.items()}
    got = reductions(cl, a.chain, wins, held | probes, tau_s=tau_s)
    base = reductions(cl, a.chain, wins, held | probes) if a.derotate else None
    scanned = {}
    if a.scan_tau:
        grid = [i * 0.02 / CHIP_HZ for i in range(-50, 51)]   # +-1.0 chip, 0.02 chip steps
        for t in grid:
            g = reductions(cl, a.chain, wins, held | probes, tau_s={p: t for p in held | probes})
            for prn, v in g.items():
                if v[2] >= 8 and v[0] > 0:
                    cur = scanned.get(prn)
                    val = v[1] / v[0]
                    if cur is None or val > cur[1]:
                        scanned[prn] = (t, val)
    inst_ph = per_instance_phase(cl, a.chain, wins, held) if a.per_instance else {}
    cl.stop()

    if not got:
        print("⚠️ NO RECORDS with 2+ instances -- nothing to compare, and that is NOT a "
              "measurement of incoherence. Check the telemetry connected (--gather takes "
              "HOST:PORT, and the gather serves 127.0.0.1 only -- run this ON cf06).")
        return 1

    print("chain %s: %d windows, probes %s%s\n"
          % (a.chain, len(wins), sorted(probes),
             "   [DEROTATED by the derived tau]" if a.derotate else ""))
    print("  %-5s %-7s %-8s %-11s %-11s %-8s %s"
          % ("prn", "cn0", "n_rec", "P_pow", "P_coh", "eta_band", "n_inst"))
    sat_eta, probe_eta, n_inst_seen = [], [], 0
    for prn in sorted(got):
        p_pow, p_coh, n_rec, nch, ninst = got[prn]
        if n_rec < 8 or p_pow <= 0.0:
            continue
        eta = p_coh / p_pow
        n_inst_seen = max(n_inst_seen, ninst)
        tag = "probe" if prn in probes else ""
        print("  %-5d %-7s %-8d %-11.3e %-11.3e %-8.2f %d %s"
              % (prn, ("%.1f" % cn0[prn]) if cn0.get(prn) is not None else "--",
                 n_rec, p_pow / n_rec, p_coh / n_rec, eta, ninst, tag))
        (probe_eta if prn in probes else sat_eta).append(eta)

    # ⚠️ REFUSE ON A NON-UNIFORM FLEET. If some instances carry a stable phase and others are
    # random, the instances are NOT running the same code, and every number above is a
    # measurement of that split rather than of the sky. This happened on the first run
    # (2026-08-16): half the instances were still serving different phases pending a tracker
    # restart, and the headline read "the band does not cohere" -- a confident wrong answer
    # drawn from self-inflicted contamination. Bimodal per-instance stability is the tell, and
    # it is detectable from the data itself, so the tool checks rather than trusting the
    # operator to remember.
    if base:
        print("\n  DEROTATION EFFECT (derived tau from dll_disc), eta_band:")
        print("    %-5s %-9s %-11s %-11s %-9s %s"
              % ("prn", "disc", "eta_raw", "eta_derot", "gain", ""))
        for prn in sorted(got):
            if prn not in base or base[prn][2] < 8 or base[prn][0] <= 0 or got[prn][0] <= 0:
                continue
            er = base[prn][1] / base[prn][0]
            ed = got[prn][1] / got[prn][0]
            print("    %-5d %+9.3f %-11.3f %-11.3f %-9.2fx %s"
                  % (prn, disc.get(prn, 0.0), er, ed, (ed / er) if er > 0 else float("nan"),
                     "probe" if prn in probes else ""))
    if scanned:
        print("\n  TAU SCAN (diagnostic, not an estimator) -- best tau vs the DERIVED one:")
        print("    %-5s %-12s %-12s %-10s %s" % ("prn", "tau_best(chip)", "tau_derived",
                                                 "eta_best", ""))
        for prn in sorted(scanned):
            tb, vb = scanned[prn]
            td = (tau_s or {}).get(prn, 0.0)
            print("    %-5d %-12.3f %-12.3f %-10.3f %s"
                  % (prn, tb * CHIP_HZ, td * CHIP_HZ, vb, "probe" if prn in probes else ""))
        print("    ⚠️ a probe whose eta_best rivals the satellites' means the scan is finding "
              "noise, and no tau read off it means anything.")
    if inst_ph:
        print("\n  PER-INSTANCE PHASE vs the record's full-band sum (vector-averaged)")
        print("    %-5s %-10s %-9s %-7s %s" % ("prn", "inst", "phase(rad)", "|mean|", "n"))
        stab = []
        for prn in sorted(inst_ph):
            for inst in sorted(inst_ph[prn]):
                ph, mag, n = inst_ph[prn][inst]
                print("    %-5d %-10s %+9.3f %7.3f %d" % (prn, inst, ph, mag, n))
                stab.append(mag)
            print()
        if stab:
            lo = [x for x in stab if x < 0.4]
            hi = [x for x in stab if x > 0.7]
            if lo and hi and len(lo) >= 2 and len(hi) >= 2:
                print("\n⚠️ NON-UNIFORM FLEET -- NO CONCLUSION AVAILABLE. %d instance(s) carry a "
                      "STABLE phase (|mean| > 0.7) and %d are RANDOM (< 0.4). Instances run in "
                      "lockstep, so that split means they are not running the same code -- a "
                      "pending restart, a mixed arm, a half-deployed change. Every eta_band "
                      "above is measuring THAT, not the sky. Make the fleet uniform and re-run."
                      % (len(hi), len(lo)))
                return 1
            med = statistics.median(stab)
            print("    median |vector mean| = %.3f -- %s" % (
                med,
                "STABLE: a per-instance CONSTANT. Measure it once and remove it, and the band "
                "coheres." if med > 0.5 else
                "RANDOM per record: there is no constant to calibrate away. The phase is being "
                "destroyed upstream of the combine, so fix that before touching the reduction."))

    print()
    if not sat_eta or not probe_eta:
        print("INCONCLUSIVE: need at least one held satellite AND one probe.")
        return 1
    ms, mp = statistics.median(sat_eta), statistics.median(probe_eta)
    print("  median eta_band: satellites %.2f, probes %.2f (null)   n_inst = %d"
          % (ms, mp, n_inst_seen))
    print("  aligned phases read eta_band ~ 1; completely unrelated phases read ~ 1/n_inst "
          "= %.3f" % (1.0 / max(n_inst_seen, 1)))
    print()
    if ms < 1.5 * max(mp, 1e-9):  # the satellite must beat its own null, whatever the level
        print("⚠️ THE BAND DOES NOT COHERE: satellites (%.2f) do not beat their own probe null "
              "(%.2f). Switching the reduction to a full-band coherent sum would DESTROY "
              "signal, not recover it -- the per-instance power sum is accidentally robust to "
              "exactly this. Find the cross-channel phase error first; by the lockstep rule it "
              "is a BUG, not a property." % (ms, mp))
        return 1
    frac = ms
    print("✅ THE BAND COHERES: eta_band %.2f against a %.2f null (ideal 1.0, random %.3f). "
          "Combining coherently across the whole band is valid; expect the C/N0 VALUE to be "
          "unchanged and its VARIANCE to fall by up to ~%dx."
          % (ms, mp, 1.0 / max(n_inst_seen, 1), n_inst_seen))
    if frac < 0.7:
        print("⚠️ but %.2f is short of the ideal 1.0 -- a residual cross-channel phase error "
              "costs the rest. #32's band phase-slope fit is what should remove it." % frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
