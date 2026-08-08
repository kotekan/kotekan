"""Fleet combines: the shared code discriminator and the cross-node coherent sum.

Extracted verbatim from gps_distributed_broker.py (task #27 M1). Both take an endpoint
list and return a per-PRN dict; both are per-CHAIN operations in the unified broker (each
signal has its own combiners), but neither holds state between calls.

See docs/CHORD_GNSS_SHARED_DLL.md for fleet_dll and gnss_gpu_search.md 11.15 for
fleet_coherent's one-way split and shuffled-null floor.
"""
import math
import random
import statistics

from .transport import _get, _log_rl


def fleet_dll(endpoints, hop_window, min_instances, k_sigma, q_fallback):
    """Sum the fleet's raw Early/Prompt/Late powers per PRN -> one full-bandwidth discriminator.

    THE PROBLEM THIS SOLVES. On CHORD the F-engine comb spreads L5 across all eight nodes and
    each GPU holds seven 195.3 kHz channels, so one tracker instance correlates 1.37 MHz of a
    20.46 MHz lobe -- 6.7%, -11.8 dB, and no instance can ever do better because the bandwidth
    is not on the machine. That gap is why the code lock is episodic; it is NOT a signal deficit
    (the full-band-equivalent C/N0 is 45.5 dB-Hz and a commercial receiver has tracked CHORD's
    deep sidelobes) and NOT the quality gate (measured correctly placed above its noise floor).

    WHY SUMMING IS LEGITIMATE, AND CHEAP. The DLL discriminator is NON-COHERENT -- (E-L)/(E+L)
    is built from POWERS. Powers add. So this needs no phase alignment between nodes, no
    sample-level sync, and no coherent machinery: only the same integration window, which
    pow_hop identifies EXACTLY (an absolute F-engine hop index shared by every node, so equality
    means the same sky -- no tolerance, unlike a capture-UTC float). Six numbers per PRN per
    window is ~6 kbit/s per instance; it rides the REST polling that already exists. Shipping
    record streams between hosts, or building a combiner hierarchy, is the wrong shape.

    WHY RAW POWERS AND NOT dll_disc. Ratios do not sum: (SUM E - SUM L)/(SUM E + SUM L) is not
    any function of the per-instance (E-L)/(E+L). Publishing e_pow/l_pow/p_pow was the single
    enabling change on the combiner side.

    WHAT THE SUM ACTUALLY BUYS, because it is NOT what it looks like. Summing K instances does
    NOT raise q = 2P/(E+L): every tap's mean scales by K, so the RATIO is untouched -- a fleet
    of 8 reports the same 1.5 a single node does. What collapses is the VARIANCE. Each summed
    power has K times the mean and K times the variance, so q's spread falls as 1/sqrt(K), and
    a signal-free PRN's q tightens onto 1.0 from a measured single-instance tail of 1.87 to
    ~1.3 at K=8. The gain is therefore a LOWER BAR, not a higher statistic -- which is why the
    caller must gate on a MEASURED noise floor and never on a fixed q. It also settles what
    looked like a contradiction on 2026-08-03: 1.3 is 1.5 sigma INTO the noise for one instance
    and ~3 sigma CLEAR of it for eight. Same number, opposite verdict, different population.

    Returns {prn: {disc, q, hop, n_src, n_chan, q_floor, q_med, q_sigma}} for PRNs with >=
    min_instances agreeing instances; PRNs below that are ABSENT, so the caller falls back to
    the single-combiner discriminator and a partly-down fleet degrades rather than stalls.
    """
    if not endpoints:
        return {}
    # prn -> list of (hop, e, p, l, n_chan). Unreachable instances are skipped, not fatal:
    # a node down for maintenance must cost sensitivity, never the loop.
    rows = {}
    best_coh = {}   # prn -> ((deep_snr, amp_snr), row, url): strongest instance's COHERENT view
    for url in endpoints:
        try:
            got = _get("%s/get_status" % url)
        except Exception as e:
            _log_rl("fleet-dll-%s" % url, "fleet DLL: %s unreachable (%s)" % (url, e))
            continue
        for r in got:
            hop = int(r.get("pow_hop", -1))
            e = float(r.get("e_pow", 0.0))
            l = float(r.get("l_pow", 0.0))
            if hop < 0 or (e + l) <= 0.0:
                continue  # no metadata on the record frames, or this PRN is not being despread
            # CURRENCY CHECK. A combiner configured without fft_len publishes pow_hop as a raw
            # SAMPLE index; its neighbours publish HOPS, 16384x smaller. Grouping those together
            # matches nothing, and the failure would look like "the fleet never agrees" rather
            # than "one config is stale" -- which is exactly why pow_fft_len is published. Drop
            # the odd instance out loudly instead of silently summing a fleet of one.
            if int(r.get("pow_fft_len", 0)) <= 0:
                _log_rl("fleet-dll-cur-%s" % url,
                        "fleet DLL: %s has no fft_len (pow_hop is a SAMPLE index) -- excluded; "
                        "regenerate that node's config" % url)
                continue
            prn = int(r["prn"])
            rows.setdefault(prn, []).append(
                (hop, e, float(r.get("p_pow", 0.0)), l, float(r.get("n_chan", 0.0))))
            # BEST-OF for the COHERENT statistics. deep_amplitude / deep_snr / coherence_s come
            # from each instance's own deep integration and CANNOT be merged here: this combine
            # sums powers, which is exactly what makes it phase-blind and cheap. So carry the
            # strongest instance's coherent row instead of pretending to a fleet number that
            # would need cross-node phase alignment. Ranked on deep_snr, falling back to
            # amp_snr so a chain that never certifies a deep still reports its best view.
            key = (float(r.get("deep_snr", 0.0)), float(r.get("amp_snr", 0.0)))
            if prn not in best_coh or key > best_coh[prn][0]:
                best_coh[prn] = (key, r, url)
    out = {}
    for prn, rs in rows.items():
        # Instances free-run, so their emit phases differ by up to one emit period: take the
        # newest window and admit everything within hop_window of it. INTEGER arithmetic on the
        # F-engine's own counter -- an instance a full second stale is 0.003 chips of code
        # motion at the measured 0.0033 chips/s, orders below what the discriminator resolves,
        # and one that is further behind than that is a fault worth excluding.
        newest = max(r[0] for r in rs)
        use = [r for r in rs if newest - r[0] <= hop_window]
        if len(use) < min_instances:
            continue
        E = sum(r[1] for r in use)
        P = sum(r[2] for r in use)
        L = sum(r[3] for r in use)
        if E + L <= 0.0:
            continue
        bc = best_coh.get(prn)
        out[prn] = {"disc": (E - L) / (E + L),
                    # the strongest instance's coherent row + which node it came from
                    "coh_row": bc[1] if bc else None,
                    "coh_src": bc[2] if bc else None,
                    # q = 2P/(E+L): 1.0 with no peak (all three taps equal noise power), 4.0 at
                    # a clean lock with 0.5-chip spacing. The three powers are built identically
                    # by the combiner (|sum of subband correlations|^2 / energy^2), which is what
                    # makes this comparable across instances and meaningful once summed.
                    "q": 2.0 * P / (E + L),
                    # kept raw: the gate is built on the summed PROMPT POWER (see below), and a
                    # ratio cannot answer "is there signal here" -- only "am I on the peak".
                    "p_pow": P,
                    "hop": newest,
                    "n_src": len(use),
                    "n_chan": sum(r[4] for r in use)}

    # LIVE NOISE FLOOR for q, measured every cycle instead of assumed. The floor moves with the
    # number of contributing instances and with the EMA length, so ANY constant is wrong for
    # some fleet size -- and a constant set for K=1 (2.2, correct there) rejects every real
    # signal once the sum has tightened the distribution around 1.0. Most tracked PRNs are
    # signal-free at any moment, so the MEDIAN of the q population IS the no-peak value and the
    # MAD is its spread; both are outlier-immune to the handful of sats that do have a peak.
    # This is the same discipline as the search's Gamma ceiling: derive the bar from the
    # population you are thresholding, and log it so it is never a silent constant.
    # GATE ON SIGNAL PRESENT, NOT ON q. This was wrong in the first version and the sky showed
    # it (2026-08-03): q = 2P/(E+L) is a peak-SHARPNESS metric, high only when the tracker is
    # ALREADY on the peak, so gating the trim on it says "only correct the code once it is
    # already correct" -- the loop can never pull in from the shoulder, which is the entire
    # pull-in region a DLL exists for. Measured that evening: PRN 10 at q 0.68 / disc -0.30 and
    # PRN 1 at q 0.77 / disc -0.54 were sitting on real, correctly-signed shoulders with the
    # fleet floor at 1.07, so the gate rejected exactly the satellites it existed to rescue.
    #
    # The right precondition is "is there signal here at all", which is independent of WHERE on
    # the correlation function we sit -- the summed PROMPT POWER against the noise population of
    # prompt powers. Same self-calibrating median/MAD as before (most tracked PRNs are
    # signal-free at any moment, so the median IS the no-signal level), applied to the statistic
    # that actually answers the question. q stays, reported, as the lock DIAGNOSTIC it is.
    def _floor(vals, k, lo_margin):
        s = sorted(vals)
        if len(s) < 8:  # too few rows to characterise a population
            return None, None, None
        m = s[len(s) // 2]
        mad = sorted(abs(x - m) for x in s)[len(s) // 2]
        sg = 1.4826 * mad
        # If MAD collapses (a degenerate population -- every instance reporting the same window
        # of zeros) the bar would collapse with it, so keep a small absolute margin too.
        return m, sg, max(m + k * sg, m + lo_margin)

    q_med, q_sigma, q_floor = _floor([v["q"] for v in out.values()], k_sigma, 0.05)
    # Prompt power spans orders of magnitude between satellites, so the bar is multiplicative:
    # median = the noise level, and a PRN must exceed it by k sigma to count as present.
    p_med, p_sigma, p_floor = _floor([v["p_pow"] for v in out.values()], k_sigma, 0.0)
    for v in out.values():
        v["q_med"], v["q_sigma"] = q_med, q_sigma
        v["q_floor"] = q_fallback if q_med is None else q_floor
        v["p_med"], v["p_floor"] = p_med, p_floor
        # No population to characterise -> fall back to the q bar rather than gating on nothing.
        v["present"] = (v["q"] >= v["q_floor"] if p_floor is None
                        else v["p_pow"] >= p_floor)
    return out


def _coherent_sum(a):
    """Coherent sum SNR / amplitude / coherence fraction of a per-record complex series.

    Line-for-line the same statistic as gnss::coherent_sum (gnssChannelizedDespread.cpp): rotate
    the sum real, take the ORTHOGONAL component as the noise estimate. Kept identical on purpose
    -- the broker's fleet number and a combiner's local number are compared against each other
    and against the same floors, and two SNR conventions differing by a factor nobody wrote down
    is how a floor stops meaning anything.

    Returns (snr, amplitude, coh_frac) where coh_frac = |sum|/sum|.| -- the chopping-independent
    coherence measure (STATE 8.20.24); snr scales with sqrt(N) and coh_frac does not.
    """
    n = len(a)
    if n < 2:
        return 0.0, 0.0, 0.0
    sr = sum(x.real for x in a)
    si = sum(x.imag for x in a)
    mag = math.hypot(sr, si)
    if mag <= 0.0:
        return 0.0, 0.0, 0.0
    cr, ci = sr / mag, -si / mag          # e^{-i arg(sum)}
    noise2 = 0.0
    for x in a:
        im = x.real * ci + x.imag * cr    # Im(x * rot)
        noise2 += im * im
    denom = math.sqrt(noise2)
    tot_abs = sum(abs(x) for x in a)
    # DEGENERATE-RESIDUAL GUARD -- mirrors gnss::residual_snr in gnssChannelizedDespread.cpp,
    # and MUST stay mirrored (this docstring promises the two statistics are identical).
    #
    # `denom > 0.0` catches only an exactly-zero residual, which is not the case that occurs.
    # A series aligned to within DOUBLE PRECISION gives ~1e-17 and sails through, returning
    # mag/1e-17 ~ 1e17. This path is MORE exposed than the C++ one, not less: both callers
    # above build their series by derotating with a unit-modulus reference
    # (s * conj(r)/|r|, and x * conj(rest)/|rest| per instance), so the products are nearly
    # REAL BY CONSTRUCTION and the orthogonal component is the very thing that derotation
    # removed. Seen live 2026-08-07: the viewer, which reads this fleet merge on :12060,
    # showed 6.93e16 sigma with coh_frac 0.13 and kept doing so after the C++ guard shipped --
    # because the number on screen comes from HERE, not from a node.
    #
    # Floor far above machine epsilon, far below anything physical: a 60 dB record has an
    # orthogonal residual ~1e-3 of mag. Fail closed (0.0), matching n < 2.
    degen = mag * math.sqrt(n) * 1e-12
    return (mag / denom if denom > degen else 0.0), mag / n, (mag / tot_abs if tot_abs > 0 else 0.0)


def fleet_coherent(endpoints, min_instances, min_records, prns=None, log=None,
                   null_trials=1, floor_margin=3.0, seed=0):
    """CROSS-NODE COHERENT COMBINE: the per-record sky phase, and the deep folds it unlocks.

    WHAT THIS FIXES, measured on sky 2026-08-05. Every instance's deep fold sat at ~14 sigma
    against a THERMAL ceiling of ~101 (per-record SNR 9.25, 120 records). The gap is a common
    per-record phase error of ~0.75 rad which is:
        * 0.984 COHERENT ACROSS INSTANCES (same sky, independent thermal noise), and
        * only ~0.57 autocorrelated record-to-record (lag 1) -- i.e. nearly WHITE IN TIME.
    So a within-node TEMPORAL estimator cannot touch it (a record's neighbours do not predict
    it; the tracker built on that premise measured 14.5 -> 13.2, a loss), while the other
    INSTANCES measure it instantly. Correcting each instance against the others took its deep
    fold from ~14 to 65-105, i.e. to the thermal bound.

    WHY IT MUST LIVE HERE AND NOT ON A NODE. Each instance has ample SNR to MEASURE the phase
    (9.25/record -> 0.11 rad). What it cannot do is use its OWN estimate: derotating a record by
    a phase computed from that same record subtracts its own noise, rectifies it, and inflates
    every statistic downstream. It needs an INDEPENDENT estimate of a COMMON quantity, and the
    only independent view of the same sky lives on the other nodes. (The other independent axis
    is the ELEMENT axis -- see gnssElemCal, deferred -- which is what makes this local again at
    full CHORD, where each node holds fewer channels but far more elements.)

    THE TWO ESTIMATORS, both leave-one-out by construction:
      * PER INSTANCE: derotate instance i's records by the phase of the sum of the OTHERS. The
        rotation is then statistically independent of instance i's noise, so noise cannot align
        with itself. This is the honest per-node number.
      * THE FLEET TOTAL: ONE-WAY SPLIT. Sum the instances into two disjoint halves S and R, and
        derotate S by R's phase -- S alone is the integrated signal, R is only the reference.
        ⚠️ NOT the symmetric S*e^{-i arg R} + R*e^{-i arg S}, and not a derotation by the fleet's
        own sum. Both are unbiased in the MEAN and both destroy the VARIANCE estimate, which is
        what every significance number is built on:
          - fleet-sum derotation collapses each record onto the positive real axis outright;
          - the symmetric pair expands to (r_S + r_R) cos d + i (r_S - r_R) sin d, so with
            balanced halves (r_S ~ r_R) the imaginary part vanishes BY CONSTRUCTION.
        Either way coherent_sum finds no orthogonal component, and the SNR explodes: measured
        38526 with coh_frac exactly 1.000 on live data, and 47.3 against a GENIE of 17.7 on
        synthetic -- "beating" perfect knowledge, the signature of a self-referential estimator.
        One-way keeps arg(S) - arg(R) random under noise, so the statistic keeps its meaning.
        The price is honest and small: only half the aperture is integrated (the other half is
        spent measuring the phase), i.e. ~sqrt(2) below a hypothetical full-aperture coherent
        sum -- while still ~sqrt(K/2) above any single instance.

    NO RATE SEARCH IS NEEDED, which is what makes this cheap enough for the poll loop. Every
    instance despreads against the SAME broker seed, so their residual carrier rates are
    identical (measured -0.20996 cyc/record on all ten) -- the alignment cross-product cancels
    it, and the leave-one-out derotation then removes the common ramp along with the sky phase.
    Cost is O(records x instances) per PRN with no transform anywhere: pure stdlib, ~1 ms.

    ALIGNMENT. Instances have arbitrary constant phase offsets (different combs, different NCO
    history), so each is rotated onto the strongest instance by arg(<A_i conj(A_ref)>) before
    anything is summed. Skipping this is not subtle -- an unaligned "fleet average" reads as a
    13 rad phase excursion of pure garbage (I made exactly that mistake while diagnosing this).
    Instances are weighted by replica ENERGY, which is the MRC weight: A = G/E has variance
    ~1/E, so summing E*A is summing the raw correlations G.

    THE FLOOR IS MEASURED, NOT ASSUMED (same principle as fleet_dll's q_floor). Every cycle the
    identical math is re-run on a NULL built by SHUFFLING each instance's records independently:
    real amplitudes, real energies, real instance count -- but the common per-record phase, which
    is the entire thing this estimator locks onto, destroyed. Whatever the null reads is what the
    estimator manufactures from nothing at this fleet size and window length, so a detection must
    clear floor_margin x that. Characterised over 6 polls / 726 null samples: median 1.35, 99th
    percentile 4.15, max 5.73, against a weakest REAL coherent detection of 22.2 -- a clean gap,
    and non-coherent PRNs sat in the null population (median 1.93) exactly as they should.
    A shuffle (not a roll) is required: a roll shifts the shared linear carrier ramp by a
    CONSTANT, leaving it fully intact, so a rolled null measures the rate-search floor (~18-25)
    rather than this estimator's own.

    Returns {prn: {deep_snr, deep_amplitude, coh_frac, n_src, n_rec, align, per_inst, floor,
    present}} for PRNs seen by >= min_instances instances over >= min_records common hops.
    `present` is the floor-cleared flag -- the caller publishes coherent numbers only for those
    and keeps the single-instance value otherwise, so a partly-down fleet degrades rather than
    stalls, and a noise-only PRN can never manufacture a fleet detection.
    """
    if not endpoints:
        return {}
    # url -> prn -> {hop: (A, energy)}. Unreachable instances are skipped, never fatal.
    got = {}
    for url in endpoints:
        try:
            recs = _get("%s/get_records" % url)
        except Exception as e:
            _log_rl("fleet-coh-%s" % url, "fleet coherent: %s unreachable (%s)" % (url, e))
            continue
        per = {}
        for r in recs or []:
            prn = int(r.get("prn", -1))
            if prn <= 0 or (prns is not None and prn not in prns):
                continue
            d = {}
            for x in r.get("records") or []:
                try:
                    hop, re_, im_, en = int(x[0]), float(x[1]), float(x[2]), float(x[3])
                except (TypeError, ValueError, IndexError):
                    continue
                if en > 0.0:
                    d[hop] = (complex(re_, im_), en)
            if d:
                per[prn] = d
        if per:
            got[url] = per

    def _solve(store):
      out = {}
      all_prns = set()
      for per in store.values():
        all_prns.update(per)
      for prn in sorted(all_prns):
        src = {u: per[prn] for u, per in store.items() if prn in per}
        if len(src) < min_instances:
            continue
        hops = None
        for d in src.values():
            hops = set(d) if hops is None else (hops & set(d))
        hops = sorted(hops or ())
        if len(hops) < min_records:
            continue
        # Reference instance = the most energetic view; every other is rotated onto it.
        ref_u = max(src, key=lambda u: sum(src[u][h][1] for h in hops))
        ref = [src[ref_u][h][0] for h in hops]
        aligned, align_q = {}, []
        for u, d in src.items():
            a = [d[h][0] for h in hops]
            w = sum(d[h][1] for h in hops) / float(len(hops))    # MRC weight = mean energy
            cr = sum(a[k] * ref[k].conjugate() for k in range(len(hops)))
            if abs(cr) <= 0.0:
                continue
            rot = complex(cr.real, -cr.imag) / abs(cr)           # e^{-i arg(<A conj(A_ref)>)}
            aligned[u] = [x * rot * w for x in a]
            # Alignment coherence: |<A conj(A_ref)>| / (sum|A||A_ref|). Near 1 means these two
            # instances really are seeing one sky with one carrier model -- the assumption the
            # whole combine rests on, so it is REPORTED rather than assumed.
            den = sum(abs(a[k]) * abs(ref[k]) for k in range(len(hops)))
            if den > 0.0 and u != ref_u:
                align_q.append(abs(cr) / den)
        if len(aligned) < min_instances:
            continue
        urls = sorted(aligned)
        nrec = len(hops)
        tot = [sum(aligned[u][k] for u in urls) for k in range(nrec)]
        # ---- per instance: leave THAT instance out of its own reference ----
        per_inst = {}
        for u in urls:
            corr = []
            for k in range(nrec):
                rest = tot[k] - aligned[u][k]
                m = abs(rest)
                corr.append(aligned[u][k] * complex(rest.real, -rest.imag) / m if m > 0.0
                            else aligned[u][k])
            per_inst[u] = _coherent_sum(corr)[0]
        # ---- fleet total: ONE-WAY split (S integrated, R only referenced) -- see the note ----
        half = {u: (i % 2) for i, u in enumerate(sorted(urls, key=lambda u: -sum(
            abs(x) for x in aligned[u])))}                        # interleave: balanced halves
        fleet = []
        for k in range(nrec):
            s = sum(aligned[u][k] for u in urls if half[u] == 0)  # SIGNAL half
            r = sum(aligned[u][k] for u in urls if half[u] == 1)  # REFERENCE half
            m = abs(r)
            fleet.append(s * complex(r.real, -r.imag) / m if m > 0.0 else 0j)
        snr, amp, cf = _coherent_sum(fleet)
        out[prn] = {
            "deep_snr": snr, "deep_amplitude": amp, "coh_frac": cf,
            "n_src": len(urls), "n_rec": nrec,
            "align": (sum(align_q) / len(align_q)) if align_q else 0.0,
            "per_inst": per_inst,
            "best_inst": max(per_inst, key=per_inst.get) if per_inst else None,
            "best_inst_snr": max(per_inst.values()) if per_inst else 0.0,
        }
      return out

    out = _solve(got)
    if not out:
        return {}
    # ---- MEASURED NULL FLOOR (see the docstring): same math, common phase destroyed ----
    # The shuffle is per instance and per PRN, so each instance's records keep their amplitudes
    # and energies but no longer line up in time with anyone else's. The floor is the MAX over
    # every null statistic produced this cycle -- fleet totals and per-instance values alike,
    # since both are published and both must be defensible.
    rng = random.Random(seed)
    floor = 0.0
    for _ in range(max(1, null_trials)):
        shuf = {}
        for u, per in got.items():
            sp = {}
            for prn, d in per.items():
                hops = list(d)
                vals = [d[h] for h in hops]
                rng.shuffle(vals)
                sp[prn] = {hops[i]: vals[i] for i in range(len(hops))}
            shuf[u] = sp
        nres = _solve(shuf)
        for v in nres.values():
            floor = max(floor, v["deep_snr"], *(v["per_inst"].values() or [0.0]))
    for v in out.values():
        v["floor"] = floor
        v["present"] = v["deep_snr"] >= floor_margin * floor
    if log:
        best = max(out, key=lambda p: out[p]["deep_snr"])
        b = out[best]
        log("fleet coherent: %d PRN (%d clear the floor), best PRN %d deep %.1f vs floor %.1f "
            "(best single instance %.1f, %dx over %d records, align %.3f, coh_frac %.3f)"
            % (len(out), sum(1 for v in out.values() if v["present"]), best, b["deep_snr"],
               floor, b["best_inst_snr"], b["n_src"], b["n_rec"], b["align"], b["coh_frac"]))
    return out
