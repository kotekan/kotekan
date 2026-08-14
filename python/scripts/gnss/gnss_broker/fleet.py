"""Fleet combines: the shared code discriminator and the cross-node coherent sum.

Extracted verbatim from gps_distributed_broker.py (task #27 M1). Both take an endpoint
list and return a per-PRN dict; both are per-CHAIN operations in the unified broker (each
signal has its own combiners), but neither holds state between calls.

See docs/CHORD_GNSS_SHARED_DLL.md for fleet_dll and gnss_gpu_search.md 11.15 for
fleet_coherent's one-way split and shuffled-null floor.
"""
import cmath
import collections
import math
import random
import statistics

from .transport import _get, _log_rl


def fleet_dll(endpoints, hop_window, min_instances, k_sigma, q_fallback,
              deep_gate_prns=None, deep_gate_margin=3.0, probe_prns=None):
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
    coh_cleared = {}  # prn -> floor-cleared per-instance deep_snrs, for the quadrature fallback
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
            # QUADRATURE FALLBACK input (2026-08-10). When fleet_coherent does not engage,
            # the served deep_snr used to be this argmax -- which sits a measured 4.9 dB
            # BELOW the fleet value, so the published series stepped 5-8 dB every time the
            # fleet gate flickered, and at the observed 20-70% duty that mixture alone
            # contributed ~3.5 dB of scatter (the C/N0 regression, docs 11.31).
            # sqrt(sum snr_i^2) over floor-cleared instances is the significance of the
            # incoherent power sum, and it lands at the fleet level (measured -0.7..+2.0 dB
            # over 5 strong PRNs) -- CONTINUITY is what it buys; it is NOT quieter than the
            # argmax within-state (5.0 vs 3.5 dB -- no scalar beats coherent combining, the
            # engagement rate is the real fix, task #10).
            # THE FLOOR-CLEARED GATE IS LOAD-BEARING: a floored deep reads ~7 on pure
            # noise, so an ungated quadrature over 12 instances would read sqrt(12)*7 ~ 24
            # and manufacture a detection from nothing. coherence_s > 0 is the same rule
            # sig_of() applies, and with it the noise PRNs (0-1 cleared instances) reduce
            # to the argmax exactly (verified on 113 polls, fixtures/fleetcap_20260810).
            if (float(r.get("coherence_s", 0.0) or 0.0) > 0.0
                    and float(r.get("deep_snr", 0.0)) > 0.0):
                coh_cleared.setdefault(prn, []).append(float(r["deep_snr"]))
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
        _cl = coh_cleared.get(prn) or []
        out[prn] = {"disc": (E - L) / (E + L),
                    # the strongest instance's coherent row + which node it came from
                    "coh_row": bc[1] if bc else None,
                    "coh_src": bc[2] if bc else None,
                    # quadrature over floor-cleared instances (see the comment at the
                    # accumulator); None when < 2 instances cleared -- then the argmax IS
                    # the honest answer and the publisher keeps it.
                    "coh_quad": (sum(x * x for x in _cl) ** 0.5, len(_cl))
                                if len(_cl) >= 2 else None,
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

    # Q FLOOR FROM THE PROBES TOO (2026-08-14). Same peer-competition trap as the prompt
    # bar below: computed over out.values() this reads ~1.0 only because most satellites
    # happen to be OFF-peak right now. On a good day with eight healthy rows the median
    # would be ~2.5 and the bar would lock out everything that was working. Probes are
    # E=P=L by construction, so their q is the noise value by definition -- measured 0.88,
    # 0.97, 1.06 against real satellites at 2.70 and 3.28.
    _probe_q = [v["q"] for k, v in out.items() if probe_prns and k in probe_prns]
    if len(_probe_q) >= 3:
        q_med, q_sigma, q_floor = _floor(_probe_q, k_sigma, 0.05)
        if q_med is None:                      # _floor needs 8; do it directly for a few
            _s = sorted(_probe_q)
            q_med = _s[len(_s) // 2]
            _mad = sorted(abs(x - q_med) for x in _probe_q)[len(_probe_q) // 2]
            q_sigma = 1.4826 * _mad
            q_floor = max(q_med + k_sigma * q_sigma, q_med + 0.05)
    else:
        q_med, q_sigma, q_floor = _floor([v["q"] for v in out.values()], k_sigma, 0.05)
    # Prompt power spans orders of magnitude between satellites, so the bar is multiplicative:
    # median = the noise level, and a PRN must exceed it by k sigma to count as present.
    # ---- THE PRESENCE FLOOR NEEDS A NOISE REFERENCE, NOT A PEER COMPARISON ------------
    # The line below (kept as the fallback) builds the bar from `out.values()` -- EVERY
    # TRACKED SATELLITE -- and then treats that population's median as the noise level.
    # That premise holds only when most rows are noise. On the airspy prototype it did,
    # because --noise-probes seeded below-horizon PRNs and put genuine signal-free records
    # into exactly this population. CHORD has never set --noise-probes (default 0), so the
    # median is a SIGNAL level and the bar lands at the top of the signal distribution:
    # measured 2026-08-14, gps_l5 2/9 present, gal_e5a 1/12, bds_b2a 1/9. Presence stopped
    # meaning "is this discriminator informative" and became "are you in the upper tail of
    # today's peers" -- a competition a satellite loses the moment it starts drifting,
    # which is precisely when it needs the loop (#49's latch).
    #
    # There is no fixing this from the tracked population alone: the measured p_pow spans
    # one decade with NO bimodality, so nothing in it locates the noise. (A log-domain bar
    # off the weakest quartile was tried on the same data: 2/9 and 4/12. Also guessing.)
    # The probes are not a refinement, they are the missing anchor.
    #
    # With probes present, the bar is what the comment below always claimed it was:
    # MULTIPLICATIVE against a measured noise level. Probes are pure noise by construction
    # (deepest below-horizon PRNs), so their median IS N, and k_sigma becomes "this many
    # times the noise floor" rather than sigmas of a mixed population -- which is also
    # robust to there being only a handful of them, where a MAD would not be.
    _probe_p = [v["p_pow"] for k, v in out.items()
                if probe_prns and k in probe_prns and v.get("p_pow")]
    if len(_probe_p) >= 2:
        _pm = sorted(_probe_p)[len(_probe_p) // 2]
        p_med, p_sigma, p_floor = _pm, None, _pm * max(k_sigma, 1.0)
        for v in out.values():
            v["p_floor_src"] = "probes:%d" % len(_probe_p)
    else:
        p_med, p_sigma, p_floor = _floor([v["p_pow"] for v in out.values()], k_sigma, 0.0)
        for v in out.values():
            v["p_floor_src"] = "peers:%d" % len(out)
    for v in out.values():
        v["q_med"], v["q_sigma"] = q_med, q_sigma
        v["q_floor"] = q_fallback if q_med is None else q_floor
        v["p_med"], v["p_floor"] = p_med, p_floor
        # No population to characterise -> fall back to the q bar rather than gating on nothing.
        # PREFER q WHEN ITS NOISE VALUE IS MEASURED. p_pow is an ABSOLUTE power, so the
        # prompt gate asks "are you bright" -- and brightness varies 25x between
        # satellites for reasons unrelated to whether the tap is on the peak (measured:
        # G10 and G23 at 7.7e-08 and 4.4e-08 against G8 at 9.0e-10, all three tracking).
        # q = 2P/(E+L) is a RATIO: it asks "is there a peak under the tap", which is the
        # question presence exists to answer, and it is scale-free so a faint satellite
        # with a clean peak passes while a bright one whose tap has slid does not.
        if len(_probe_q) >= 3:
            v["present"] = v["q"] >= v["q_floor"]
            v["present_gate"] = "q:probes"
        else:
            v["present"] = (v["q"] >= v["q_floor"] if p_floor is None
                            else v["p_pow"] >= p_floor)
            v["present_gate"] = "prompt"

        # ---- DEEP GATE (task #49, opt-in per PRN) --------------------------------------
        # THE PROMPT GATE ABOVE IS ON-PEAK-BIASED, WHICH MAKES IT A LATCH. Prompt power is
        # suppressed precisely WHEN THE TAP IS OFF-PEAK, so: off-peak -> low prompt -> fails
        # the gate -> never trimmed -> stays off-peak. The loop cannot pull in from the
        # shoulder, which is the entire region a DLL exists for.
        #
        # ⚠️ THIS IS THE SAME DEFECT THIS GATE WAS WRITTEN TO FIX. The comment above records
        # gating on q having exactly this property ("only correct the code once it is already
        # correct") and the remedy was to move to prompt power -- which is on-peak-biased in
        # the same way. One on-peak statistic was swapped for another, so the sky never
        # showed the fix failing. Do not "fix" this a third time with another tap ratio.
        #
        # MEASURED 2026-08-12 across all five chains: 36 satellites well-detected
        # (deep_snr > 3x deep_floor), only 10 passed the prompt gate, 26 (72%) excluded --
        # and the EXCLUDED ones carried the LARGER errors (e5a |disc| 0.850 vs 0.342 present;
        # e5b 0.742 vs 0.291). Witness case E33 on gal_e5a at el 69: deep_snr 35 (13x floor),
        # all 12 instances agreeing E:P:L = 0.11 : 1.0 : 2.0, and its trim FROZEN while the
        # same satellite tracked normally on gal_e5b.
        #
        # deep_snr is the right statistic BECAUSE OF the property that makes it wrong for a
        # C/N0 (task #47): the deep fold RE-SEARCHES rate and phase, so it detects the
        # satellite wherever the tap sits. Detection and tap-placement are exactly what a DLL
        # presence gate must separate.
        #
        # OPT-IN, one PRN at a time, because switching the whole fleet at once would make 72%
        # of it newly trimmable in one step -- against a slew cap already railing 67-100% of
        # the time. That trades a latch for an oscillation, and the A/B would be
        # uninterpretable. deep_gate_prns=True enables it fleet-wide once the pull-in has been
        # measured on the opt-in set.
        if deep_gate_prns:
            for prn, v in out.items():
                if deep_gate_prns is not True and prn not in deep_gate_prns:
                    continue
                c = v.get("coh_row") or {}
                ds = float(c.get("deep_snr", 0.0) or 0.0)
                fl = float(c.get("deep_floor", 0.0) or 0.0)
                # Needs a REAL floor to compare against. Without one there is no bar, and
                # defaulting to "present" would trim on noise -- strictly worse than the
                # latch. Fall through to the prompt verdict instead.
                if fl <= 0.0 or ds <= 0.0:
                    continue
                v["present"] = ds >= deep_gate_margin * fl
                v["present_gate"] = "deep"
                v["deep_gate_snr"], v["deep_gate_floor"] = ds, fl
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
                   null_trials=1, floor_margin=3.0, seed=0, max_age_hops=1 << 20):
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
    # THE FLEET'S CLOCK, taken across EVERY satellite an instance serves -- including the
    # ones this call was not asked about. Deriving it only from the requested PRNs would
    # make "now" depend on at least one of THOSE being current, which is exactly the
    # assumption that fails when a whole set of satellites goes down together.
    fleet_now_all = 0
    for url in endpoints:
        try:
            recs = _get("%s/get_records" % url)
        except Exception as e:
            _log_rl("fleet-coh-%s" % url, "fleet coherent: %s unreachable (%s)" % (url, e))
            continue
        per = {}
        for r in recs or []:
            prn = int(r.get("prn", -1))
            if prn <= 0:
                continue
            for _x in (r.get("records") or []):
                try:
                    fleet_now_all = max(fleet_now_all, int(_x[0]))
                except (TypeError, ValueError, IndexError):
                    pass
            if prns is not None and prn not in prns:
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
      # the fleet's own "now": the newest record anywhere, over every satellite
      fleet_now = max(fleet_now_all,
                      max((max(d) for per in store.values() for d in per.values() if d),
                          default=0))
      stale_prns = {}
      for prn in sorted(all_prns):
        src = {u: per[prn] for u, per in store.items() if prn in per}
        if len(src) < min_instances:
            continue
        # ---- ABSOLUTE STALENESS: a SET satellite must not keep reporting ----------------
        # Combiner buffers never expire a PRN's records: once a satellite stops being
        # tracked its last window sits there indefinitely (measured 2026-08-12: up to 2.9 h).
        # The relative test below cannot see that, because when EVERY instance is equally
        # stale they agree perfectly and the combine proceeds -- publishing an hour-old
        # detection as current. Observed the same day: gal_e5a PRN 27 combined at deep 35
        # over 12 instances from records 98 minutes old, and PRN 15 kept "reporting" for
        # 18 minutes after it set below the horizon at 15:48 UTC. That is the whole fleet
        # agreeing on a fossil.
        #
        # `fleet_now` is the newest record ANY instance holds for ANY satellite -- the
        # fleet's own clock, needing no wall time and no F-engine epoch. A PRN whose newest
        # record trails it by more than max_age_hops is not being tracked; drop it rather
        # than report it.
        if max_age_hops and fleet_now:
            prn_now = max(max(d) for d in src.values())
            if fleet_now - prn_now > max_age_hops:
                stale_prns[prn] = fleet_now - prn_now
                continue
        # ---- ANCHOR ON THE FRESHEST WINDOW, NOT ON UNANIMITY (2026-08-12) ------------
        # This used to intersect the hop sets of EVERY contributor. One instance outside the
        # others' window then emptied the common set for every satellite on every cycle, and
        # the estimator returned {} -- indistinguishable from an empty sky, with no log line
        # anywhere. Measured that day: cx19's GPU-0 pipeline froze (all five of its chains
        # stopped at the same hop, an hour stale) while the other seven agreed to +-0.10 s,
        # and the cross-node coherent combine was silently dead for the duration. An
        # estimator that gets MORE fragile as the fleet grows is backwards.
        #
        # STALENESS IS ABOUT TIME, so the newest data anchors -- not the largest group. A
        # first cut took the most-populous window and it was wrong in a way the fixture
        # caught: several instances frozen TOGETHER outvote the live ones and the fleet
        # happily combines hour-old records into a confident current number. Records are
        # stamped by hop, so "freshest" is knowable: anchor on the instance reaching the
        # newest hop, keep everyone who overlaps it by min_records, and refuse to fall back
        # onto an anchor that is itself more than one window behind the fleet.
        fresh_max = max(max(d) for d in src.values())
        keep, hops, dropped = None, set(), []
        for anchor in sorted(src, key=lambda u: max(src[u]), reverse=True):
            span = max(src[anchor]) - min(src[anchor])
            if max(src[anchor]) < fresh_max - span:
                break          # this anchor and every later one are stale: decline
            aset = set(src[anchor])
            cohort = {u: d for u, d in src.items() if len(set(d) & aset) >= min_records}
            if len(cohort) < min_instances:
                continue
            h = set.intersection(*(set(d) & aset for d in cohort.values()))
            while len(h) < min_records and len(cohort) > min_instances:
                worst = min(cohort, key=lambda u: len(set(cohort[u]) & aset))
                del cohort[worst]
                h = set.intersection(*(set(d) & aset for d in cohort.values()))
            if len(h) >= min_records and len(cohort) >= min_instances:
                keep, hops = cohort, h
                dropped = [(u, len(set(src[u]) & aset)) for u in src if u not in cohort]
                break
        if keep is None:
            continue
        src = keep
        hops = sorted(hops)
        drop_note = dropped
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
        # ---- THE S/R SPLIT: balanced in sum|w|^2, and STABLE across polls (task #6) ----
        #
        # WAS: sort by descending sum|A| and interleave. Two defects, and the second is the one
        # that cost us.
        #
        #   1. Interleaving balances the COUNT, not the ENERGY. The estimator integrates S and
        #      spends R on the phase reference, so its variance is set by how sum|w|^2 divides:
        #      too much in S and the reference is noisy (derotation loses coherence), too much
        #      in R and S under-integrates. Rank-interleave leaves that to luck, and with an
        #      odd instance count or one dominant node it is systematically off.
        #
        #   2. THE SORT KEY FLUCTUATES, so the MEMBERSHIP did. sum|A| is a noisy per-poll
        #      quantity; two instances of similar strength swap rank between polls, the halves
        #      reshuffle, and deep_snr steps because a DIFFERENT APERTURE is being integrated.
        #      That is a real change in the estimator, not in the sky -- and it is per-chain and
        #      independent between chains, which is exactly the fast term isolated on
        #      2026-08-09: the same satellite's two coherent sidebands share their SLOW
        #      variation (median corr +0.49, geometry through the beam) but NOT their fast
        #      variation (+0.11), so the fast part is generated downstream of the antennas,
        #      per chain. A split that reshuffles on noise is precisely such a generator.
        #
        # NOW: iterate in a STABLE order (sorted URL -- a constant, not a measurement) and
        # greedily assign each instance to whichever half currently holds less sum|w|^2. The
        # iteration order can no longer move, so membership changes only when the energies
        # genuinely change, and the greedy pass is the standard near-optimal answer to the
        # number-partitioning this is. `aligned` already carries the MRC weight, so sum|x|^2 IS
        # sum|w|^2 up to a common scale.
        # THE SPLIT IS A PURE FUNCTION OF THE URL SET -- no measurement enters it at all.
        #
        # Measured over 400 synthetic polls with 18% per-poll amplitude jitter:
        #     rank-interleave (was)          imbalance 0.062, membership changed 399/399 polls
        #     greedy on per-poll sum|w|^2    imbalance 0.045, membership changed 395/399
        #     fixed alternation by url       imbalance 0.084, membership changed     0
        # Greedy balances best and is STILL unstable, because any rule that reads a noisy
        # weight re-decides on noise. The trade is not close: an 8% energy imbalance is
        # second-order on the estimator's variance (the one-way split already concedes
        # sqrt(2)), whereas re-drawing WHICH APERTURE is integrated every poll is first-order
        # on the published number -- it steps deep_snr for reasons that have nothing to do
        # with the sky. Stability is the property worth buying.
        #
        # It is also physically sensible rather than arbitrary: sorted URLs alternate
        # cx19/gnss0, cx19/gnss1, cx27/gnss0, ... so each half receives one GPU from every
        # node -- equal channel count, equal hardware, balanced in expectation. The instances
        # are nominally identical (7 channels each off the same comb), so sorting them by
        # measured amplitude -- what the old code did -- was largely sorting them by noise.
        #
        # `split_imbalance` is published so a GENUINE asymmetry (a node with fewer channels, a
        # merged --combine-gpus instance) shows up as a persistently large value instead of
        # hiding. If that ever appears, the fix is a stable weight carried across polls, not a
        # return to deciding on the current one.
        half = {u: (i % 2) for i, u in enumerate(sorted(urls))}
        _tot = [sum(sum(abs(x) ** 2 for x in aligned[u]) for u in urls if half[u] == j)
                for j in (0, 1)]
        _imb = (abs(_tot[0] - _tot[1]) / (_tot[0] + _tot[1])) if (_tot[0] + _tot[1]) > 0 else 0.0
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
            # Published so the split can be judged rather than assumed: 0 = perfectly balanced
            # sum|w|^2, 1 = everything in one half. A value that WANDERS between polls means
            # membership is still moving and deep_snr will step with it.
            "split_imbalance": _imb,
            "split_s": sorted(u for u in urls if half[u] == 0),
            # Instances excluded from THIS PRN's combine and why (url, hops it had inside
            # the candidate window). Non-empty means the fleet degraded rather than failed;
            # the caller logs it, because the old code's silent {} was the actual defect.
            "dropped": drop_note,
        }
      if stale_prns and log:
          log("fleet coherent: %d PRN(s) excluded as STALE (their newest record trails "
              "the fleet by more than %d hops): %s"
              % (len(stale_prns), max_age_hops,
                 ", ".join("PRN %d (%.0f s)" % (p, a / 195312.5)
                           for p, a in sorted(stale_prns.items()))))
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


# --- fleet phase-slope delay fit (task #32, docs/CHORD_JOINT_TRACKING.md P1) --------------
#
# A delay is a phase ramp across frequency: dphi/df = -2*pi*tau. The per-channel prompts
# (GnssGpuRecordAssemble /get_spectrum -- NCO-derotated, element-combined, window-summed)
# contain the full-band delay at matched-filter precision, sigma_tau ~ 1/(2*pi*beta_rms*SNR).
# E/P/L keeps three lag samples of this spectrum; the E5a disc rails because per narrow
# channel E ~ P ~ L -- a slope fit is immune to that entirely.
#
# THE SPARSE-COMB HAZARD, and why this is a FLEET fit: one instance's 7 channels at
# 3.125 MHz stride alias at 3.27 chips (the measured tracker grating lobes). The defence is
# the union of combs -- 79 unique channels with different per-node offsets -- which is
# exactly how the search beat the same lobes. Two consequences baked into the code below:
#   * instances carry unknown constant phases phi_i (different NCO histories), solved
#     JOINTLY with tau by alternation: given tau, phi_i = arg sum_{c in i} E_c A_c
#     e^{+i 2 pi f_c tau}; given phi_i, one coherent fold over ALL channels. A few rounds
#     from tau = 0 converge, because a locked seed is already within ~1 chip.
#   * ⚠️ magnitude-per-instance summing does NOT work: |.| erases exactly the
#     cross-instance phase that discriminates the true peak from the comb lobes.
#
# THE FLOOR IS MEASURED, NEVER ASSUMED (fleet_coherent's rule): the identical fold runs on
# a null where each instance's channel->value assignment is shuffled -- real amplitudes,
# real comb, the frequency-phase association (the entire thing the fit locks onto)
# destroyed. A tau is reported only against that floor, and the caller sees both.

def fit_spectrum_delay(points, chip_rate_hz, chan_width_hz, span_chips=2.0,
                       coarse_step_chips=0.02, rounds=4, rng=None):
    """Joint (tau, phi_i) fit. `points` = [(freq_id, complex A, energy, inst_key)].

    Returns (tau_chips, peak, floor, n_pts, n_inst) -- tau of the correlation peak RELATIVE
    to the replica placement (positive = the sky's code arrives LATER than the replica
    models), the fold magnitude at the peak, the shuffled-null floor from the same points,
    and the sizes. Pure function: deterministic given `rng` (a random.Random), stdlib only.
    """
    import cmath
    import random as _random

    if len(points) < 3:
        return None
    insts = {}
    for fid, a, en, key in points:
        if en > 0.0 and (a.real != 0.0 or a.imag != 0.0):
            insts.setdefault(key, []).append((fid, a, en))
    insts = {k: v for k, v in insts.items() if len(v) >= 2}
    if not insts or sum(len(v) for v in insts.values()) < 3:
        return None
    # Weights: energy is the ML combining weight (A = G/E has variance ~1/E), same
    # convention as fleet_coherent. Precompute E*A per point; phases fold around f*tau.
    f0 = min(fid for v in insts.values() for fid, _, _ in v)

    def scan(store, taus, phi):
        best_tau, best_mag = None, -1.0
        for tau_s in taus:
            total = 0.0j
            for k, v in store.items():
                s = 0.0j
                for fid, a, en in v:
                    s += en * a * cmath.exp(2j * cmath.pi * (fid - f0) * chan_width_hz
                                            * tau_s)
                total += s * cmath.exp(-1j * phi.get(k, 0.0))
            m = abs(total)
            if m > best_mag:
                best_mag, best_tau = m, tau_s
        return best_tau, best_mag

    def solve_phi(store, tau_s):
        phi = {}
        for k, v in store.items():
            s = 0.0j
            for fid, a, en in v:
                s += en * a * cmath.exp(2j * cmath.pi * (fid - f0) * chan_width_hz * tau_s)
            if abs(s) > 0.0:
                phi[k] = cmath.phase(s)
        return phi

    def run(store):
        span_s = span_chips / chip_rate_hz
        step_s = coarse_step_chips / chip_rate_hz
        n = max(3, int(round(2.0 * span_s / step_s)) + 1)
        taus = [-span_s + i * (2.0 * span_s / (n - 1)) for i in range(n)]
        tau, phi = 0.0, {}
        for _ in range(rounds):
            phi = solve_phi(store, tau)
            tau, mag = scan(store, taus, phi)
            # zoom: next round scans +-3 coarse steps around the peak at 1/10 step
            taus = [tau + (i - 15) * step_s / 5.0 for i in range(31)]
        return tau, mag

    tau_s, peak = run(insts)
    # Shuffled null: same instances, same freqs, same ENERGY WEIGHTS -- only arg(A) is
    # permuted. Moving (A, en) together moves the weights and the null then folds as well as
    # the data whenever one channel dominates; see delay_combine for what that cost.
    rng = rng or _random.Random(0xC0DE)
    null_store = {}
    for k, v in insts.items():
        ph = [cmath.phase(a) for _, a, _ in v]
        rng.shuffle(ph)
        null_store[k] = [(fid, abs(a) * cmath.exp(1j * ph[i]), en)
                         for i, (fid, a, en) in enumerate(v)]
    _, floor = run(null_store)
    return (tau_s * chip_rate_hz, peak, floor, sum(len(v) for v in insts.values()),
            len(insts))


def fleet_spectrum_aligned(endpoints, prns=None, log=None, window=None):
    """Poll every instance for the SAME hop-quantised window (task #53) and gather the
    per-channel points across the whole fleet.

    Returns (points_by_prn, meta) where meta = {window, w0, w1, served, dropped, degraded}.
    `points_by_prn` has the same shape fleet_spectrum returns, so fit_spectrum_delay consumes
    it unchanged -- but every point now comes from the SAME RECORDS, which is what lets the
    caller derotate by a derived phase instead of fitting one free phase per instance.

    HOW THE COMMON INDEX IS CHOSEN. Ask each instance what it has (every reply carries
    available:[lo,hi], success or refusal), then take **min(hi)** -- the newest window EVERY
    instance has finished. Instances lag each other by a few records (~0.15 s, task #46), so
    the fastest one is typically a window ahead; taking its index would make the laggards
    answer `not_yet` forever.

    ⚠️ NEVER BLOCK ON UNANIMITY. An instance that cannot serve the common window is DROPPED BY
    NAME and the rest combine without it. Requiring all of them is exactly the failure of
    [[chord-fleet-combine-fragility]], where one frozen GPU silently emptied the common window
    for every satellite and the combine returned {} -- indistinguishable from an empty sky.

    `degraded` lists instances whose reply is NOT addressable (a node still running a config
    without spectrum_window_samples). Those cannot be aligned with anyone, so they are excluded
    rather than mixed in: a single unaligned member would put the free-phase problem straight
    back into the gathered set.
    """
    avail, served, dropped, degraded, reanchored = {}, {}, [], [], []
    for url in endpoints:
        try:
            r = _get("%s/get_spectrum" % url)
        except Exception as e:
            _log_rl("fleet-spec-%s" % url, "fleet spectrum: %s unreachable (%s)" % (url, e))
            dropped.append((url, "unreachable"))
            continue
        if not r.get("addressable"):
            degraded.append(url)
            continue
        a = r.get("available") or [-1, -1]
        if a[1] is None or a[1] < 0:
            dropped.append((url, "no complete window yet"))
            continue
        avail[url] = (int(a[0]), int(a[1]))
    if not avail:
        if degraded and log:
            log("fleet spectrum: %d instance(s) NOT addressable (pre-#53 config): %s"
                % (len(degraded), ", ".join(degraded)))
        return {}, {"window": None, "served": {}, "dropped": dropped, "degraded": degraded}

    idx = int(window) if window is not None else min(hi for _, hi in avail.values())
    out, w0s, w1s = {}, set(), set()
    for url, (lo, hi) in avail.items():
        if idx < lo or idx > hi:
            # Refused by its own bookkeeping before we even ask -- name it and move on.
            dropped.append((url, "window %d outside [%d,%d]" % (idx, lo, hi)))
            continue
        try:
            r = _get("%s/get_spectrum?window=%d" % (url, idx))
        except Exception as e:
            dropped.append((url, "unreachable on second poll (%s)" % e))
            continue
        if r.get("status") != "ok":
            dropped.append((url, r.get("status") or "no status"))
            continue
        w0s.add(r.get("wstart0"))
        w1s.add(r.get("wstart1"))
        served[url] = r
        fids = r.get("freq_ids") or []
        for row in r.get("prns") or []:
            prn = int(row.get("prn", -1))
            if prn <= 0 or (prns is not None and prn not in prns):
                continue
            # PHASE CURRENCY (task #52). The export applied exp(-i*phi0) to this window as a
            # whole, and phi0 is an accumulator the broker's own re-pins move every ~2 min.
            # Rotating it out here puts every window -- and every instance -- on ONE reference,
            # which is what makes windows comparable at all. n_reanchor > 0 means the
            # accumulator was reset or stepped MID-window: no single constant can undo that,
            # so drop the PRN for this window rather than fold a discontinuity into the sum.
            if int(row.get("n_reanchor", 0) or 0) > 0:
                reanchored.append((url, prn))
                continue
            derot = cmath.exp(1j * float(row.get("phi0", 0.0) or 0.0))
            ch = row.get("chan") or []
            pts = out.setdefault(prn, [])
            for i, fid in enumerate(fids):
                if i >= len(ch):
                    break
                re_, im_, en = ch[i]
                if en > 0.0:
                    pts.append((int(fid), complex(re_, im_) * derot, float(en), url))
    # THE INVARIANT, ASSERTED RATHER THAN ASSUMED. Same index must mean the same samples. If
    # two instances disagree here the quantisation is broken (mismatched window_samples in a
    # node's config is the way that happens), and every phase downstream is wrong -- so say so
    # loudly instead of combining anyway.
    if len(w0s) > 1 or len(w1s) > 1:
        if log:
            log("⚠️ fleet spectrum: window %d has DIFFERENT sample spans across instances "
                "(wstart0 %s, wstart1 %s) -- the quantisation is broken, check that every node "
                "config has the SAME spectrum_window_samples" % (idx, sorted(w0s), sorted(w1s)))
        return {}, {"window": idx, "served": {}, "dropped": dropped, "degraded": degraded,
                    "misaligned": True}
    if degraded and log:
        _log_rl("fleet-spec-degraded",
                "fleet spectrum: excluding %d NOT-addressable instance(s) (pre-#53 config): %s"
                % (len(degraded), ", ".join(degraded)))
    return out, {"window": idx, "w0": next(iter(w0s), None), "w1": next(iter(w1s), None),
                 "served": served, "dropped": dropped, "degraded": degraded,
                 "reanchored": reanchored}


def fleet_spectrum(endpoints, prns=None):
    """Poll every instance's /get_spectrum; return {prn: [(freq_id, A, energy, inst_key)]}.

    ⚠️ LEGACY (pre-#53): takes whatever each instance accumulated since ITS last poll, so the
    instances are NOT summing the same records and the per-instance phi_i in the fit is what
    absorbs the offset. Use fleet_spectrum_aligned instead -- that free phase per instance is
    the defect of task #52, not a modelling convenience. Kept for a fleet that has not been
    restarted onto the addressable endpoint yet.
    """
    out = {}
    for url in endpoints:
        try:
            r = _get("%s/get_spectrum" % url)
        except Exception as e:
            _log_rl("fleet-spec-%s" % url, "fleet spectrum: %s unreachable (%s)" % (url, e))
            continue
        fids = r.get("freq_ids") or []
        for row in r.get("prns") or []:
            prn = int(row.get("prn", -1))
            if prn <= 0 or (prns is not None and prn not in prns):
                continue
            ch = row.get("chan") or []
            pts = out.setdefault(prn, [])
            for i, fid in enumerate(fids):
                if i >= len(ch):
                    break
                re_, im_, en = ch[i]
                if en > 0.0:
                    pts.append((int(fid), complex(re_, im_), float(en), url))
    return out


def delay_combine(points, chip_rate_hz, chan_width_hz, tau_chips, rng=None, n_null=8):
    """Coherently sum per-channel points after derotating each by its OWN DERIVED phase.

    THIS IS TASK #52. The old combine gave every INSTANCE a free phase, fitted by
    cross-correlating it against the most energetic instance and then summing with that
    rotation -- twelve free parameters estimated from the data they are summed into, which
    aligns noise (a self-reference: [[gnss-phase-estimator-self-reference]]) and, when the fit
    fails, drops the whole chain to the quadrature fallback (gps_l5 measured align 0.143 with
    9/12 satellites on `quad`).

    Physics gives ONE parameter, not twelve. A delay is a phase ramp across frequency,
    phi(f) = -2*pi*f*tau, so once tau is known every channel's rotation FOLLOWS. And it must be
    applied PER CHANNEL, not per instance: each instance's 7 channels are a comb spanning
    ~18.75 MHz (stride 16), so at tau = 1 chip the ramp WITHIN one instance is ~11.5 rad --
    1.8 wraps. No single constant per instance can represent that, which is why summing an
    instance's channels without removing the ramp already destroys its coherence before any
    cross-instance step happens.

    Returns {snr, amplitude, coh_frac, n_pts, n_inst, floor, present} where `floor` is a
    SHUFFLED NULL over the same points. That gate is only meaningful now: with a free phase per
    instance the fit could always find some alignment, so a null could never fail. Here the
    rotation is derived from tau alone and a shuffled set must sum to nothing.

    Requires ALIGNED windows (fleet_spectrum_aligned). Feeding it unaligned points silently
    reintroduces the per-instance offset this exists to remove -- the caller must not mix.
    """
    import cmath
    import random as _random

    pts = [(fid, a, en, key) for fid, a, en, key in points
           if en > 0.0 and (a.real != 0.0 or a.imag != 0.0)]
    if len(pts) < 3:
        return None
    f0 = min(fid for fid, _, _, _ in pts)
    tau_s = tau_chips / chip_rate_hz

    def combine(seq):
        # MRC: energy is the ML weight (A = G/E has variance ~1/E), same convention as
        # _coherent_sum and fleet_dll -- two weighting conventions in one broker is how a floor
        # stops meaning anything.
        tot = 0.0j
        abs_sum = 0.0
        for fid, a, en, _ in seq:
            rot = cmath.exp(2j * cmath.pi * (fid - f0) * chan_width_hz * tau_s)
            tot += en * a * rot
            abs_sum += en * abs(a)
        return tot, abs_sum

    tot, abs_sum = combine(pts)
    mag = abs(tot)
    # SHUFFLED NULL. Reassign the AMPLITUDES across the frequency slots, keeping the
    # frequencies and the derotation identical -- that destroys the frequency<->phase pairing
    # (the delay ramp) while preserving every marginal. Shuffle, never roll: a roll IS a delay
    # and would leave the very structure being tested for.
    rng = rng or _random.Random(1234567)
    floor = 0.0
    # ⚠️ PERMUTE THE PHASE ONLY, NOT THE (AMPLITUDE, ENERGY) PAIR. Moving the pair moves the
    # ENERGY WEIGHTS with it, and this comb's weights are far from uniform (the edge channels
    # roll off). When one channel dominates the weight, a shuffled set folds just as well as
    # the real one -- |sum|/sum|.| approaches 1 for BOTH -- so the gate reads ~1.0 whatever the
    # truth is. That is exactly how it read on 2026-08-13 while the real coherence was 0.68-0.97,
    # and the wrong verdict survived a retraction, a night, and a morning of chasing it.
    # Keeping (f, |A|, E) pinned and permuting only arg(A) destroys the frequency<->phase
    # pairing the delay lives in and NOTHING else, which is the whole point of the null.
    phases = [cmath.phase(a) for _, a, _, _ in pts]
    for _ in range(max(1, n_null)):
        sh = list(phases)
        rng.shuffle(sh)
        null_pts = [(pts[i][0], abs(pts[i][1]) * cmath.exp(1j * sh[i]), pts[i][2], pts[i][3])
                    for i in range(len(pts))]
        floor = max(floor, abs(combine(null_pts)[0]))
    n_inst = len({k for _, _, _, k in pts})
    # EFFECTIVE POINT COUNT of the fold's own weights, w = E*|A|. The coherence statistic is
    # degenerate when the weight concentrates: at n_eff -> 1 a single point carries the sum and
    # |sum|/sum|.| -> 1 for ANY phases, so neither the value nor the null means anything. Live
    # sky runs n_eff ~48 of 79 (2026-08-13), which is healthy -- but publish it, because the
    # failure is invisible in the coherence alone and it cost a retracted correct result.
    _w = [en * abs(a) for _, a, en, _ in pts]
    _sw = sum(_w)
    n_eff = (_sw * _sw / sum(x * x for x in _w)) if _sw > 0.0 else 0.0
    return {"snr": (mag / floor) if floor > 0.0 else 0.0,
            "n_eff": n_eff,
            "amplitude": mag / len(pts),
            "coh_frac": (mag / abs_sum) if abs_sum > 0.0 else 0.0,
            "n_pts": len(pts), "n_inst": n_inst, "floor": floor,
            "tau_chips": tau_chips,
            # The null is a MAX over shuffles, so clearing it by a margin is a real detection
            # in the same sense the search's Gamma ceiling is (chord-gamma-noise-ceiling).
            "present": mag > 0.0 and floor > 0.0 and mag >= 1.5 * floor}


def fit_tau_coherent(points, chip_rate_hz, chan_width_hz, span_chips=2.0,
                     coarse_step_chips=0.02, refine=3, rng=None, n_null=16):
    """Fit ONE tau by maximising the coherence of the gathered points. No nuisance phases.

    WHY THIS REPLACES fit_spectrum_delay's SOLVE. That fit solves a free phase PER INSTANCE at
    every trial tau, and those parameters absorb the very ramp it is measuring: 12 free
    parameters flatten a 1-parameter objective, so it never leaves its start. Measured on sky
    2026-08-12, the four brightest gal_e5a satellites, shipped fit vs the coherence optimum:
        PRN 29  fit -0.000 (coh 0.020)  vs BEST -1.020 (coh 0.103)
        PRN 10  fit -0.100 (coh 0.050)  vs BEST +2.000 (coh 0.250)
        PRN 21  fit +0.056 (coh 0.094)  vs BEST -0.320 (coh 0.153)
        PRN 11  fit +0.068 (coh 0.067)  vs BEST +0.860 (coh 0.471)
    It clusters at ZERO at 2-7x worse coherence. That is not SNR inflation, it is loss of tau
    OBSERVABILITY -- and it explains spec_tau's whole record (11.3% clearing its null, a
    far-regime correlation of -0.375, and #50's re-seed firing 74 times while tau never shrank).

    ⚠️ REQUIRES ALIGNED POINTS (fleet_spectrum_aligned). The coherence being maximised is across
    the fleet at ONE instant, which is legitimate only when every instance summed the same
    records -- and that is exactly what task #53 established. Verified on sky first: 79 channels,
    one window, ONE ramp, coherence 0.794 against a 0.109 random baseline.

    Returns (tau_chips, coherence, floor, n_pts, n_inst) -- same shape as fit_spectrum_delay so
    it drops in -- where `floor` is a SHUFFLED-NULL coherence over the same points. The null
    permutes the PHASES across the frequency slots, destroying the frequency<->phase pairing
    while preserving every marginal INCLUDING THE ENERGY WEIGHTS; shuffle, never roll (a roll
    IS a delay). ⚠️ It shuffled g = E*A until 2026-08-13, which moved the weights too and made
    the gate unfalsifiable whenever one channel dominated -- see the note at the shuffle.
    """
    import cmath
    import random as _random

    pts = [(fid, en * a) for fid, a, en, _ in points
           if en > 0.0 and (a.real != 0.0 or a.imag != 0.0)]
    if len(pts) < 8:
        return None
    f0 = min(fid for fid, _ in pts)
    wsum = sum(abs(g) for _, g in pts)
    if wsum <= 0.0:
        return None

    def coh_at(tau_chips, seq):
        ts = tau_chips / chip_rate_hz
        tot = sum(g * cmath.exp(2j * cmath.pi * (fid - f0) * chan_width_hz * ts)
                  for fid, g in seq)
        return abs(tot) / wsum

    def best_over(seq, lo, hi, step):
        # Coarse scan then successive refinement. The peak half-width is 1.1-3.3 chips on sky,
        # so a 0.02-chip grid is far finer than the feature -- the refinement is for the
        # sub-grid vertex, not to find the peak.
        b, bc = lo, -1.0
        t = lo
        while t <= hi:
            c = coh_at(t, seq)
            if c > bc:
                b, bc = t, c
            t += step
        for _ in range(max(0, refine)):
            step *= 0.25
            for t in (b - 2 * step, b - step, b + step, b + 2 * step):
                c = coh_at(t, seq)
                if c > bc:
                    b, bc = t, c
        return b, bc

    tau, coh = best_over(pts, -span_chips, span_chips, coarse_step_chips)
    rng = rng or _random.Random(20260812)
    # PHASE-ONLY PERMUTATION. `pts` holds g = E*A, so shuffling g moves the WEIGHT as well as
    # the phase -- and with this comb's weights (edge channels roll off, one can dominate) the
    # null then folds as well as the data and the gate is unfalsifiable. Measured 2026-08-13:
    # this null read 0.94 against a true coherence of 0.97 on PRN 28, i.e. "no detection" on a
    # deep_snr 233 satellite. Pin |g| to its own frequency, permute only arg(g).
    phases = [cmath.phase(g) for _, g in pts]
    floor = 0.0
    for _ in range(max(1, n_null)):
        sh = list(phases)
        rng.shuffle(sh)
        _, c = best_over([(pts[i][0], abs(pts[i][1]) * cmath.exp(1j * sh[i]))
                          for i in range(len(pts))],
                         -span_chips, span_chips, coarse_step_chips)
        floor = max(floor, c)
    return (tau, coh, floor, len(pts), len({k for _, _, _, k in points}))
