"""THE FLEET DLL, BUILT FROM THE COMB -- no tracker-side cross-channel sum anywhere in it.

Step 2 of task #63 (KV, 2026-08-14: "purge the idea of summing across channels in each
instance... all derived quantities should happen in the broker, never a tracker"). The
discriminator the broker closes its code loop with is the LAST thing that needed the tracker's
summed slots; everything here is formed from `comb_epl()` -- Early, Prompt and Late per
CHANNEL, per record, as the transport ships them.

WHAT THIS REPRODUCES, EXACTLY. GnssCoherentCombiner forms, per record:

    A   = (SUM_c G_c) / (SUM_c E_c)                       ... the cross-channel coherent sum
    p2  = |A|^2 ,   e2 = |SUM_c G^E_c|^2 / (SUM_c E^E_c)^2 ,  l2 likewise

then averages over its integration window and publishes e_pow/p_pow/l_pow, which fleet_dll
sums over instances. `comb_epl()` gives A_c = G_c/E_c and E_c per channel, so SUM_c G_c =
SUM_c A_c*E_c and the same three numbers fall out here. The arithmetic is not new; WHERE it
happens is the whole point.

WHY THE NUMBERS WILL NOT MATCH TO THE LAST DIGIT, and which differences are legitimate:
  * AVERAGING. The combiner runs a rolling EMA of length 100 records (~1.05 s); this takes a
    plain mean over the windows held in the ring (32 windows = 128 records = 1.34 s). Two
    different low-pass filters on the same series, so they agree in the mean and differ by
    filter noise -- compare disc/q over many cycles, never one.
  * TASK #62. The comb is element-combined with weights updated by the very record it weights,
    the header slots with the previous record's. Measured 1-11% on the PROMPT. It hits all
    three taps through the same weights, so disc and q are far less affected than the absolute
    powers -- which is why the A/B is judged on the RATIOS and reports the power offset
    separately rather than calling it an error.

WHAT THE COMB BUYS THAT THE SUMMED SLOTS COULD NOT:
  * A PER-CHANNEL DISCRIMINATOR (`chan`). Each channel lives on exactly one instance
    (freq_id mod 8 routing), so the per-channel powers here are already fleet-wide -- ~56
    numbers across the lobe where there was one. That is the frequency axis the sum destroyed.
  * EXACT COLLATION. fleet_dll admits instances within `hop_window` of the newest because REST
    polls arrive whenever they arrive. Frames here are keyed on an ABSOLUTE window index, so
    "the same sky" is an integer equality with no tolerance and no staleness policy.

⚠️ THE COMB CARRIES NO DEEP STATISTICS. deep_snr / deep_floor / coherence_s come from the
combiner's own fold, so `coh_rows` must be supplied by the caller (fleet_coherent, or the
polled fleet_dll during migration) for #49's deep gate to have anything to gate on. Without
it the deep gate simply does not fire -- it never guesses.
"""
import math

from .fleet import apply_presence


def dll_tau(disc, spacing):
    """The code discriminator -> delay estimate, chips.

    ⚠️ |tau| <= 0.25 chips BY CONSTRUCTION, whatever `disc` is: the clamp is on the
    discriminator and it is then divided by four. THIS IS THE WHOLE OF #51 -- one update can
    never slew further than that at any gain, so the UPDATE RATE is the only lever. (Cutting
    the gain to "compensate" for a faster rate hands the entire win straight back: same gain,
    faster rate.)

    ⚠️ ONE CONVENTION, ONE PLACE. The C++ fleet loop calls gnss::dll_tau in
    lib/stages/gnss/gnssFleetDll.hpp, which is this expression character for character, and
    scripts/gnss/fleetdll_gate.py compares the two on identical bytes. This existed as two
    inline copies in the broker (the policy cycle and the fast thread) whose own comment
    claimed they were "one convention, one place it can be wrong" -- they were two.
    """
    return -max(-1.0, min(1.0, disc)) / 4.0 * (spacing / 0.5)


def dll_integrate(trim, disc, gain, leak, clamp, spacing):
    """One leaky-integrator update. Twin of gnss::dll_integrate.

    ⚠️ `leak` IS PER UPDATE, SO LOOP BANDWIDTH SCALES WITH RATE. Continuous form:
    dT/dt = -leak*f*T + gain*f*tau. The steady state (gain*tau/leak) does NOT move with f, but
    the closed-loop and noise bandwidths both scale with it -- 3.1 -> 23.8 Hz is ~8x the
    bandwidth at unchanged constants.

    ⚠️ AND THAT STEADY STATE IS A CEILING NO RATE CAN LIFT. Under a railed discriminator the
    trim converges to gain*0.25/leak = 1.25 chips at the shipped defaults (0.25, 0.05), which
    is below the residuals seen on sky and far below the +-3.0 clamp -- so the clamp is
    unreachable by construction. Measured on sky 2026-08-15: max |trim| 1.140 chips over 5174
    updates in 8 hours, never once past 1.25. A loop pushing at a railed discriminator without
    arriving is hitting THIS, and a faster loop will not fix it. See --dll-leak-present.
    """
    t = (1.0 - leak) * trim + gain * dll_tau(disc, spacing)
    return max(-clamp, min(clamp, t))


def instance_taps(client, chain, wins, prns=None, per_channel=True):
    """{prn: {inst: {e, p, l, hop, n_chan, n_rec, chan}}} -- per-record powers, meaned.

    One entry per (PRN, instance) that had at least one record with a live comb in `wins`.
    `chan` is {freq_id: [e, p, l, n_rec]}: the same three powers formed from ONE channel, kept
    unsummed. Records with no comb (PRN not despread that record, or an instance running
    without chan_export) contribute nothing rather than zeros -- a zeroed record is not a
    measurement of no signal, it is the absence of one, and averaging it in dilutes the power
    exactly the way the deep fold's zero-padding did.
    """
    want = None if prns is None else set(int(p) for p in prns)
    acc = {}
    for w in wins:
        for inst, f in client.frame_set(chain, w).items():
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                hop = f.hop(r)
                for prn in f.prns():
                    if want is not None and prn not in want:
                        continue
                    cmb = f.comb_epl(r, prn)
                    if not cmb:
                        continue
                    gE = gP = gL = 0j
                    eE = eP = eL = 0.0
                    d = acc.setdefault(prn, {}).setdefault(
                        inst, {"e": 0.0, "p": 0.0, "l": 0.0, "hop": -1, "n_chan": 0.0,
                               "n_rec": 0, "chan": {}})
                    for fid, E, P, L, (wE, wP, wL) in cmb:
                        gE += E * wE
                        gP += P * wP
                        gL += L * wL
                        eE += wE
                        eP += wP
                        eL += wL
                        if per_channel:
                            # ONE channel's own three powers, formed by the identical
                            # expression -- |G|^2/E^2 with each tap on its own replica energy.
                            c = d["chan"].setdefault(fid, [0.0, 0.0, 0.0, 0])
                            c[0] += abs(E) ** 2
                            c[1] += abs(P) ** 2
                            c[2] += abs(L) ** 2
                            c[3] += 1
                    if eP <= 0.0:
                        continue
                    d["e"] += (abs(gE) / eE) ** 2 if eE > 0.0 else 0.0
                    d["p"] += (abs(gP) / eP) ** 2
                    d["l"] += (abs(gL) / eL) ** 2 if eL > 0.0 else 0.0
                    d["n_chan"] += len(cmb)
                    d["n_rec"] += 1
                    d["hop"] = max(d["hop"], hop)
    for _prn, per_inst in acc.items():
        for _inst, d in per_inst.items():
            n = float(d["n_rec"]) or 1.0
            d["e"] /= n
            d["p"] /= n
            d["l"] /= n
            d["n_chan"] /= n
            for c in d["chan"].values():
                m = float(c[3]) or 1.0
                c[0] /= m
                c[1] /= m
                c[2] /= m
    return acc


#: The keys carried across from the polled arm untouched. All three are products of the
#: combiner's DEEP FOLD, which is not in the comb -- see the module header.
COH_KEYS = ("coh_row", "coh_src", "coh_quad")


def fleet_dll_comb(client, chain, n_win=32, lag=1, min_instances=2, k_sigma=3.0,
                   q_fallback=2.2, prns=None, probe_prns=None, deep_gate_prns=None,
                   deep_gate_margin=3.0, coh_from=None, per_channel=True):
    """fleet_dll's dict, from the comb. {prn: {disc, q, p_pow, hop, n_src, n_chan, ...}}.

    Same keys, same meanings, same presence policy (apply_presence, shared with fleet_dll so
    the two paths cannot drift apart in their verdicts) -- the difference is confined to where
    the three powers came from. Extra keys: `src` = "comb", `n_rec`, `chan` (per-channel
    powers and discriminators), `per_inst` (each instance's own three powers).

    `coh_from`: a fleet_dll-shaped dict whose COH_KEYS are copied across verbatim. The deep
    gate (#49) and the publisher's quadrature fallback both read them, and BOTH must be
    carried, not just the row: dropping coh_quad silently reverts the published deep_snr to
    the argmax, which measured 4.9 dB below the fleet value and made the served series step
    5-8 dB whenever the fleet gate flickered (docs 11.31). A missing key here would look like
    a display quirk and be a real regression in the C/N0 record.
    """
    wins = client.windows(chain, lag=lag)
    if not wins:
        return {}
    wins = wins[-int(n_win):]
    per_prn = instance_taps(client, chain, wins, prns=prns, per_channel=per_channel)
    coh_from = coh_from or {}
    out = {}
    for prn, per_inst in per_prn.items():
        use = {i: d for i, d in per_inst.items() if d["n_rec"] > 0}
        if len(use) < min_instances:
            continue
        E = sum(d["e"] for d in use.values())
        P = sum(d["p"] for d in use.values())
        L = sum(d["l"] for d in use.values())
        if E + L <= 0.0:
            continue
        src = coh_from.get(prn) or {}
        chan, dup = {}, []
        if per_channel:
            for _inst, d in use.items():
                for fid, c in d["chan"].items():
                    # A channel reaches exactly ONE instance (freq_id mod 8 routing), so this
                    # is a merge across instances, never a sum over duplicates. If two ever
                    # claim one freq_id something upstream is misconfigured: name it and DROP
                    # the channel rather than adding the two together, which would invent a
                    # number that is neither instance's measurement. The full-band powers are
                    # formed from the per-instance sums and are unaffected either way, so this
                    # never costs the loop its discriminator.
                    if fid in chan:
                        dup.append(fid)
                        continue
                    ce, cp, cl = c[0], c[1], c[2]
                    chan[fid] = {"e": ce, "p": cp, "l": cl, "n_rec": c[3],
                                 "disc": (ce - cl) / (ce + cl) if ce + cl > 0.0 else 0.0,
                                 "q": 2.0 * cp / (ce + cl) if ce + cl > 0.0 else 0.0}
            for fid in dup:
                chan.pop(fid, None)
        out[prn] = {"disc": (E - L) / (E + L),
                    "q": 2.0 * P / (E + L),
                    "p_pow": P,
                    "e_pow": E,
                    "l_pow": L,
                    "hop": max(d["hop"] for d in use.values()),
                    "n_src": len(use),
                    "n_chan": sum(d["n_chan"] for d in use.values()),
                    "n_rec": max(d["n_rec"] for d in use.values()),
                    "src": "comb",
                    "chan": chan,
                    "chan_dup": sorted(set(dup)),
                    "per_inst": {i: (d["e"], d["p"], d["l"], d["n_rec"])
                                 for i, d in use.items()}}
        # The deep statistics: carried across, never invented (module header).
        for k in COH_KEYS:
            out[prn][k] = src.get(k)
    return apply_presence(out, k_sigma, q_fallback, probe_prns=probe_prns,
                          deep_gate_prns=deep_gate_prns, deep_gate_margin=deep_gate_margin)


def prompt_cn0(client, chain, n_win=32, lag=1, prns=None, probe_prns=None,
               k_sigma=3.0, min_instances=2, hop_s=5.12e-6, keep_records=False):
    """THE SERVED C/N0 (task #57): per-record prompt power, q-gated, probe-debiased.

    Replaces the deep fold as the radiometry. The fold RE-SEARCHES a residual rate per
    integration -- a fit on something the tracking loop already fixed -- and that re-search
    gives it ~20 dB of its own scatter, measured PAIRED on the same records (>10x on 23% of
    cycles, 7912 samples, 2026-08-15). Nothing derived from it is a measurement of the
    satellite. This estimator fits NOTHING: the rate is the tracker's (already applied to the
    despread), the tap is the loop's, and the only arithmetic is a debiased power ratio.

    THE THREE INGREDIENTS, each load-bearing:

      * PER-RECORD prompt power, fleet-summed then divided by the record's instance count.
        The per-instance mean (not the raw sum) is what makes a record with a dropped frame
        comparable to a full one -- signal and noise both scale with n_inst, so the ratio to
        the probe floor is unaffected in the common case and robust in the degraded one.

      * q-GATED. C/N0 is radiometry CONDITIONAL ON LOCK: a record where the tap sat off the
        peak measures the tap, not the satellite, and averaging it in is the incoherent
        estimator's bias (#24). The gate bar comes from the PROBES' own per-record q
        population (med + k*MAD), i.e. the same statistic on noise-by-construction rows.
        ⚠️ This on-peak bias is CORRECT here and would be the #49 latch in a trim gate --
        an estimator publishes its duty and lets the consumer decline; a gate starves the
        loop. Do not transplant this bar into presence logic.
        ⚠️ SELECTION BIAS AT THE MARGIN: q and P share noise, so a satellite passing the
        gate only on upward fluctuations reads high. `duty` is published precisely so a
        low-duty cn0 can be declined; a duty near 1 has no selection to bias it.

      * PROBE-DEBIASED. E[|P|^2] = |s|^2 + sigma^2, and sigma^2 is MEASURED as the median
        per-record prompt power of the below-horizon probes -- the only rows that are noise
        by construction ([[chord-deep-snr-fires-on-noise]]). NO PEER FALLBACK: without
        probes this returns {} rather than quietly rebuilding the #49 peer competition
        (the tracked population's median is a SIGNAL level; see apply_presence).

    C/N0 = 10*log10(rho / T_rec) with rho = (P - sigma^2)/sigma^2 meaned over gated records
    and T_rec the record's coherent span (hops_per_record * hop_s, read from the frames).
    Every normalisation upstream of P -- element cal, channel weights, the 4+4b scale --
    cancels in the ratio because the probes ride the identical pipeline.

    Returns {prn: {cn0_db, rho, duty, n_used, n_rec, split_db, sigma2, q_gate, t_rec_s,
    n_probe_rec, probe}} for every PRN seen (probes included, flagged -- their cn0 SHOULD
    be None/negative, which is what the AUC leg of the validation bar checks).
    `split_db` is the even/odd-record self-consistency in dB: free, and it is the
    split-half witness of the validation bar. keep_records=True adds `recs`
    [(win, slot, rho, q, gated)] for the offline tools; the broker must not carry it.
    """
    probe_prns = set(int(p) for p in (probe_prns or ()))
    want = None
    if prns is not None:
        want = set(int(p) for p in prns) | probe_prns
    wins = client.windows(chain, lag=lag)
    if not wins:
        return {}
    wins = wins[-int(n_win):]
    t_rec = None
    recs = {}   # (win, slot) -> {prn: [e_sum, p_sum, l_sum, n_inst]}
    for w in wins:
        for inst, f in client.frame_set(chain, w).items():
            if t_rec is None and getattr(f, "hops_per_record", 0) > 0:
                t_rec = f.hops_per_record * hop_s
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                for prn in f.prns():
                    if want is not None and prn not in want:
                        continue
                    cmb = f.comb_epl(r, prn)
                    if not cmb:
                        continue
                    gE = gP = gL = 0j
                    eE = eP = eL = 0.0
                    for _fid, E, P, L, (wE, wP, wL) in cmb:
                        gE += E * wE
                        gP += P * wP
                        gL += L * wL
                        eE += wE
                        eP += wP
                        eL += wL
                    if eP <= 0.0:
                        continue
                    d = recs.setdefault((w, r), {}).setdefault(prn, [0.0, 0.0, 0.0, 0])
                    d[0] += (abs(gE) / eE) ** 2 if eE > 0.0 else 0.0
                    d[1] += (abs(gP) / eP) ** 2
                    d[2] += (abs(gL) / eL) ** 2 if eL > 0.0 else 0.0
                    d[3] += 1
    if not recs or t_rec is None:
        return {}

    # Per-PRN time-ordered per-record series: (win, slot, p_mean, q).
    series = {}
    for key in sorted(recs):
        for prn, (e, p, l, n) in recs[key].items():
            if n < min_instances:
                continue
            q = 2.0 * p / (e + l) if (e + l) > 0.0 else 0.0
            series.setdefault(prn, []).append((key[0], key[1], p / n, q))

    # THE NOISE ANCHOR. Pooled over the whole capture rather than per record: the per-record
    # median of 3 probes carries ~20% scatter, the pooled one ~1/sqrt(N); the #56 power
    # swings are ~hourly against this window's ~1.3 s, so pooling loses nothing they move.
    probe_p, probe_q = [], []
    for prn in probe_prns:
        for _w, _r, p, q in series.get(prn, ()):
            probe_p.append(p)
            probe_q.append(q)
    if len(probe_p) < 16:
        return {}   # no anchor, no estimate -- never a peer fallback
    probe_p.sort()
    probe_q.sort()
    # ⚠️ THE DEBIAS NEEDS THE MEAN, NOT THE MEDIAN. E[|P|^2] = |s|^2 + sigma^2 with
    # sigma^2 the noise power's EXPECTATION; the per-record noise power is Gamma-ish
    # (a few complex-Gaussian dof per instance, ~n_inst of them meaned), whose median
    # sits BELOW its mean -- the self-test caught a +0.7 dB high bias from exactly this
    # (Gamma(3): median/mean = 0.89 -> +0.5 dB; ~+0.13 dB at a healthy 11-instance
    # fleet, still a bias, not noise). The median stays as the CLIP reference only:
    # mean over records <= 8x median keeps a single contaminated probe record from
    # dragging the anchor, and clips essentially nothing of a genuine Gamma tail.
    _med = probe_p[len(probe_p) // 2]
    if _med <= 0.0:
        return {}
    _kept = [x for x in probe_p if x <= 8.0 * _med]
    sigma2 = sum(_kept) / len(_kept)
    if sigma2 <= 0.0:
        return {}
    q_med = probe_q[len(probe_q) // 2]
    _mad = sorted(abs(x - q_med) for x in probe_q)[len(probe_q) // 2]
    q_gate = q_med + max(k_sigma * 1.4826 * _mad, 0.05)

    out = {}
    for prn, rows in series.items():
        rho_gated, rec_rows = [], []
        for w, r, p, q in rows:
            rho = (p - sigma2) / sigma2
            gated = q >= q_gate
            if gated:
                rho_gated.append(rho)
            if keep_records:
                rec_rows.append((w, r, rho, q, gated))
        n_used, n_tot = len(rho_gated), len(rows)
        rho_mean = sum(rho_gated) / n_used if n_used else None
        cn0 = (10.0 * math.log10(rho_mean / t_rec)
               if rho_mean is not None and rho_mean > 0.0 else None)
        # Even/odd split of the GATED records: the self-consistency of the number served.
        split_db = None
        if n_used >= 8:
            re_ = sum(rho_gated[0::2]) / len(rho_gated[0::2])
            ro_ = sum(rho_gated[1::2]) / len(rho_gated[1::2])
            if re_ > 0.0 and ro_ > 0.0:
                split_db = 10.0 * math.log10(re_ / ro_)
        out[prn] = {"cn0_db": cn0,
                    "rho": rho_mean,
                    "duty": n_used / float(n_tot) if n_tot else 0.0,
                    "n_used": n_used, "n_rec": n_tot,
                    "split_db": split_db,
                    "sigma2": sigma2, "q_gate": q_gate, "t_rec_s": t_rec,
                    "n_probe_rec": len(probe_p),
                    "probe": prn in probe_prns}
        if keep_records:
            out[prn]["recs"] = rec_rows
    return out


def chan_profile(row):
    """[(freq_id, q, disc)] sorted by frequency -- the lobe shape one PRN's comb sees.

    The thing the tracker's sum made unknowable. A channel sitting at q ~ 1 while its
    neighbours sit at 3 is either interference or a dead subband, and either way it was being
    summed straight into the discriminator before this.
    """
    return [(fid, c["q"], c["disc"]) for fid, c in sorted((row.get("chan") or {}).items())]


def db(x, ref):
    """10*log10(x/ref), or None -- for reporting power offsets without inventing zeros."""
    if x is None or ref is None or x <= 0.0 or ref <= 0.0:
        return None
    return 10.0 * math.log10(x / ref)
