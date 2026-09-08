"""THE FLEET DLL, BUILT FROM THE COMB -- no tracker-side cross-channel sum anywhere in it.

Step 2 of task #63 (KV, 2026-08-14: "purge the idea of summing across channels in each
instance... all derived quantities should happen in the broker, never a tracker"). The
discriminator the broker closes its code loop with is the LAST thing that needed the tracker's
summed slots; everything here is formed from `comb_epl()` -- Early, Prompt and Late per
CHANNEL, per record, as the transport ships them.

WHAT THIS FORMS. Per record, per PRN, over EVERY channel c of the lobe -- all senders, each
sender's columns first multiplied by exp(+i*phi0) (REC_PHI0) so they sit on one reference:

    p2 = |SUM_c G_c|^2 / (SUM_c E_c)^2 ,   e2 and l2 likewise, each tap on its own energy

then the mean over the records in the ring, and disc = (E-L)/(E+L), q = 2P/(E+L).
`comb_epl()` gives A_c = G_c/E_c and E_c per channel, so SUM_c G_c = SUM_c A_c*E_c.

⚠️ THE COHERENCE UNIT IS THE LOBE, NEVER THE SENDER. A sender is a `freq_id mod 8` grouping
of channels -- a transport artefact. Summing each sender's channels coherently and then adding
POWERS across senders (what GnssCoherentCombiner does per instance, and what this module did
before the lobe combine) puts that grouping into the discriminator: seven channels at 3.125 MHz
stride are a 3.27-chip grating comb in the correlation response, and the noise floor is
n_sender times what the band can give. No sender count is an operand anywhere here; the only
use of one is COMPLETENESS -- a record too few senders reached is left out, not reweighted.

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
import cmath
import math

# epl_decompose is re-exported: it is DLL algebra and belongs beside dll_tau, but combdll
# imports fleet for apply_presence, so the definition lives there to keep imports acyclic.
from .fleet import apply_presence
from .fleet import epl_decompose
from .telem import REC_PHI0

__all__ = ["epl_decompose"]  # the re-export, stated


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


def _lobe_fold(client, chain, wins, want, per_channel):
    """The one walk over the frames both lobe_taps and lobe_records reduce from.

    Returns (acc, chan_w): acc is {(win, slot): {prn: [gE, gP, gL, wE, wP, wL, n_chan,
    n_inst, hop]}} -- the rotated complex partial sums, NOT yet powers -- and chan_w is
    {win: ({prn: {fid: [e, p, l, n_rec]}}, {fid: owner}, {dup fids})}.
    """
    # (win, slot) -> {prn: [gE, gP, gL, wE, wP, wL, n_chan, n_inst, hop]}
    acc = {}
    # per window: {prn: {fid: [e, p, l, n_rec]}}, {fid: owner}, {dup fids}
    chan_w = {}
    for w in wins:
        cw = chan_w.setdefault(w, ({}, {}, set()))
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
                    for _fid, E, P, L, (wE, wP, wL) in cmb:
                        gE += E * wE
                        gP += P * wP
                        gL += L * wL
                        eE += wE
                        eP += wP
                        eL += wL
                    if eP <= 0.0:
                        continue
                    row = f.row(r, prn)
                    rot = cmath.exp(1j * float(row[REC_PHI0])) if row is not None else 1.0
                    d = acc.setdefault((w, r), {}).setdefault(
                        prn, [0j, 0j, 0j, 0.0, 0.0, 0.0, 0, 0, -1])
                    d[0] += gE * rot
                    d[1] += gP * rot
                    d[2] += gL * rot
                    d[3] += eE
                    d[4] += eP
                    d[5] += eL
                    d[6] += len(cmb)
                    d[7] += 1
                    d[8] = max(d[8], hop)
                    if per_channel:
                        per_fid = cw[0].setdefault(prn, {})
                        for fid, E, P, L, _w in cmb:
                            own = cw[1].setdefault(fid, inst)
                            if own != inst:
                                cw[2].add(fid)
                            # ONE channel's own three powers, formed by the identical
                            # expression -- |G|^2/E^2 with each tap on its own replica energy.
                            c = per_fid.setdefault(fid, [0.0, 0.0, 0.0, 0])
                            c[0] += abs(E) ** 2
                            c[1] += abs(P) ** 2
                            c[2] += abs(L) ** 2
                            c[3] += 1
    return acc, chan_w


def lobe_taps(client, chain, wins, prns=None, per_channel=True, min_instances=1):
    """{prn: {e, p, l, hop, n_chan, n_rec, n_inst, chan, chan_dup}} -- lobe-coherent per-record
    powers, meaned over records.

    THE COHERENCE UNIT IS THE LOBE. Per record, ONE complex sum over every channel of every
    sender, each sender's columns first multiplied by exp(+i*phi0) (REC_PHI0 -- the per-sender
    NCO accumulator the assembler applied, whose zero is arbitrary per sender), then the power
    |SUM_c G_c|^2 / (SUM_c E_c)^2 with each tap on its own replica energy. A sender is a
    `freq_id mod 8` grouping of channels -- a transport artefact -- and appears nowhere in the
    arithmetic: not as a partial-power term, not as a divisor, not as a weight.

    `n_inst` is the COMPLETENESS of a record (how many senders reached it) and is used for one
    thing only: a record fewer than `min_instances` senders reached is not averaged in. A
    partial record carries the same normalised signal power and more noise; whether it is
    still worth having is the caller's call, and the caller states it here.

    `chan` is {freq_id: [e, p, l, n_rec]}: the same three powers formed from ONE channel, kept
    unsummed, each meaned over ITS OWN live records. A freq_id two senders both carried in one
    window is a routing fault upstream: it is listed in `chan_dup` and DROPPED from `chan`
    rather than reported as a number that is neither sender's measurement (the lobe sum keeps
    it -- a duplicated channel is still that channel's measurement, twice).

    Records with no comb (PRN not despread that record, or a sender running without
    chan_export) contribute nothing rather than zeros -- a zeroed record is not a measurement
    of no signal, it is the absence of one, and averaging it in dilutes the power exactly the
    way the deep fold's zero-padding did.

    ⚠️ THE ORDER OF OPERATIONS IS THE C++ ARM'S (gnss::FleetDll::fold), on purpose: per
    sender the raw complex is summed over its channels FIRST, that partial sum is rotated,
    and the rotated partials are added. scripts/gnss/fleetdll_gate.py compares the two arms
    at 1e-9 on identical bytes, and a different association of the same sum is a different
    float.
    """
    want = None if prns is None else set(int(p) for p in prns)
    acc, chan_w = _lobe_fold(client, chain, wins, want, per_channel)
    out = {}
    complete = set()   # (win, prn) with at least one record that passed the gate
    for (w, _r), per_prn in acc.items():
        for prn, d in per_prn.items():
            if d[7] < min_instances:
                continue
            complete.add((w, prn))
            t = out.setdefault(prn, {"e": 0.0, "p": 0.0, "l": 0.0, "hop": -1, "n_chan": 0.0,
                                     "n_rec": 0, "n_inst": 0, "chan": {}, "chan_dup": set()})
            t["e"] += (abs(d[0]) / d[3]) ** 2 if d[3] > 0.0 else 0.0
            t["p"] += (abs(d[1]) / d[4]) ** 2
            t["l"] += (abs(d[2]) / d[5]) ** 2 if d[5] > 0.0 else 0.0
            t["n_chan"] += d[6]
            t["n_rec"] += 1
            t["n_inst"] = max(t["n_inst"], d[7])
            t["hop"] = max(t["hop"], d[8])
    if per_channel:
        for w, (per_prn, _own, dup) in chan_w.items():
            for prn, per_fid in per_prn.items():
                # A PRN whose every record THIS window was incomplete contributes no lobe
                # tap from it, and its channels must not appear without one: the two tables
                # describe the same records, window by window.
                if (w, prn) not in complete:
                    continue
                t = out[prn]
                t["chan_dup"] |= dup & set(per_fid)
                for fid, c in per_fid.items():
                    if fid in dup or c[3] <= 0:
                        continue
                    a = t["chan"].setdefault(fid, [0.0, 0.0, 0.0, 0])
                    a[0] += c[0]
                    a[1] += c[1]
                    a[2] += c[2]
                    a[3] += c[3]
    for t in out.values():
        n = float(t["n_rec"]) or 1.0
        t["e"] /= n
        t["p"] /= n
        t["l"] /= n
        t["n_chan"] /= n
        t["chan_dup"] = sorted(t["chan_dup"])
        for c in t["chan"].values():
            m = float(c[3]) or 1.0
            c[0] /= m
            c[1] /= m
            c[2] /= m
    return out


def lobe_records(client, chain, wins, prns=None):
    """{(win, slot): {prn: [e, p, l, n_inst, n_chan]}} -- the SAME lobe-coherent powers,
    per record, un-averaged. prompt_cn0's input; the twin of gnss::FleetDll::rec_series.

    Every record with a live comb is here, however partial: `n_inst`/`n_chan` say how
    complete it was and the consumer declines one. No power in it depends on a sender.
    """
    want = None if prns is None else set(int(p) for p in prns)
    acc, _cw = _lobe_fold(client, chain, wins, want, False)
    recs = {}
    for key, per_prn in acc.items():
        for prn, d in per_prn.items():
            recs.setdefault(key, {})[prn] = [
                (abs(d[0]) / d[3]) ** 2 if d[3] > 0.0 else 0.0,
                (abs(d[1]) / d[4]) ** 2,
                (abs(d[2]) / d[5]) ** 2 if d[5] > 0.0 else 0.0,
                d[7], d[6]]
    return recs


#: The keys carried across from the polled arm untouched. All three are products of the
#: combiner's DEEP FOLD, which is not in the comb -- see the module header.
COH_KEYS = ("coh_row", "coh_src", "coh_quad")


def taps_from_rest(get, url, chain, prns=None, timeout=5.0):
    """`lobe_taps`' object, fetched from the gather's C++ reduction instead of rebuilt.

    ⚠️ THIS IS THE SAME ARITHMETIC, NOT AN APPROXIMATION OF IT. gnss::FleetDll forms these taps
    from the same frames, and scripts/gnss/fleetdll_gate.py hands both arms IDENTICAL BYTES and
    requires e/p/l/n_chan to agree to 1e-9 with n_rec/n_inst/hop exact, per PRN AND per channel.
    That gate is the only thing standing behind this call: broker_equiv replays only what goes
    through gnss_broker.transport, and the gather is a raw socket, so a replay runs with no
    telemetry at all and quietly falls back to the polled discriminator.

    WHY IT EXISTS. Rebuilding this in Python walks every (window, sender, record, PRN, channel)
    of the gathered stream -- ~140k channel-tuples per chain per cycle, ~700k across the fleet,
    each allocating Python complex objects, ~18% of chain CPU. The broker is pinned at 100% of
    ONE core by the GIL, where cycle time IS the sum of the chains' Python CPU, so that 18% is
    ~2 s of every cycle. The reduction is ~1k numbers.

    The C++ arm drops duplicated freq_ids at window close and counts them in its stats; it
    does not list them per PRN, so `chan_dup` is [] here.

    `get` is the caller's HTTP getter (transport._get), passed in so this module keeps its
    no-transport-import property and so a replay records the fetch like any other.
    """
    d = get("%s/get_taps" % url.rstrip("/"), timeout=timeout) or {}
    want = None if prns is None else set(int(p) for p in prns)
    out = {}
    for prn, v in (d.get(chain) or {}).items():
        p = int(prn)
        if want is not None and p not in want:
            continue
        if int(v["n_rec"]) <= 0:
            continue
        out[p] = {"e": float(v["e"]), "p": float(v["p"]), "l": float(v["l"]),
                  "n_chan": float(v["n_chan"]), "n_rec": int(v["n_rec"]),
                  "n_inst": int(v["n_inst"]), "hop": int(v["hop"]),
                  "chan": {int(f): [float(c[0]), float(c[1]), float(c[2]), float(c[3])]
                           for f, c in (v.get("chan") or {}).items()},
                  "chan_dup": []}
    return out


def fleet_dll_comb(client, chain, n_win=32, lag=1, min_instances=2, k_sigma=3.0,
                   q_fallback=2.2, prns=None, probe_prns=None, deep_gate_prns=None,
                   deep_gate_margin=3.0, coh_from=None, per_channel=True, taps_src=None,
                   admit_displaced=None):
    """fleet_dll's dict, from the comb. {prn: {disc, q, p_pow, hop, n_src, n_chan, ...}}.

    Same keys, same meanings, same presence policy (apply_presence, shared with fleet_dll so
    the two paths cannot drift apart in their verdicts) -- the difference is confined to where
    the three powers came from. Extra keys: `src` = "comb", `n_rec`, `chan` (per-channel
    powers and discriminators), `chan_dup`.

    ⚠️ THE POWERS ARE LOBE-COHERENT AND NORMALISED (lobe_taps): p_pow is |SUM_c G_c|^2 over
    EVERY channel of the lobe divided by the summed energy squared, so for a signal it does
    not scale with how many senders or channels are up. `n_src` is the sender count behind
    the most complete record -- completeness, reported, never an operand.

    `coh_from`: a fleet_dll-shaped dict whose COH_KEYS are copied across verbatim. The deep
    gate (#49) and the publisher's quadrature fallback both read them, and BOTH must be
    carried, not just the row: dropping coh_quad silently reverts the published deep_snr to
    the argmax, which measured 4.9 dB below the fleet value and made the served series step
    5-8 dB whenever the fleet gate flickered (docs 11.31). A missing key here would look like
    a display quirk and be a real regression in the C/N0 record.
    """
    if taps_src is not None:
        # THE C++ ARM. No window selection here: the depth is the gather's `taps_win`, set to
        # match this broker's telem-windows. Asking for a different one silently would be the
        # 2-vs-32 window mismatch the split-ring gate exists to catch. The completeness gate
        # is applied in the C++ (its own min_instances), identically.
        per_prn = taps_src(chain, prns)
        if not per_prn:
            return {}
    else:
        wins = client.windows(chain, lag=lag)
        if not wins:
            return {}
        wins = wins[-int(n_win):]
        per_prn = lobe_taps(client, chain, wins, prns=prns, per_channel=per_channel,
                            min_instances=min_instances)
    coh_from = coh_from or {}
    out = {}
    for prn, t in per_prn.items():
        if t["n_rec"] <= 0:
            continue
        E, P, L = t["e"], t["p"], t["l"]
        if E + L <= 0.0:
            continue
        src = coh_from.get(prn) or {}
        chan = {}
        if per_channel:
            for fid, c in t["chan"].items():
                ce, cp, cl = c[0], c[1], c[2]
                chan[fid] = {"e": ce, "p": cp, "l": cl, "n_rec": c[3],
                             "disc": (ce - cl) / (ce + cl) if ce + cl > 0.0 else 0.0,
                             "q": 2.0 * cp / (ce + cl) if ce + cl > 0.0 else 0.0}
        out[prn] = {"disc": (E - L) / (E + L),
                    "q": 2.0 * P / (E + L),
                    "p_pow": P,
                    "e_pow": E,
                    "l_pow": L,
                    "hop": t["hop"],
                    "n_src": t["n_inst"],
                    "n_chan": t["n_chan"],
                    "n_rec": t["n_rec"],
                    "src": "comb",
                    "chan": chan,
                    "chan_dup": list(t.get("chan_dup") or [])}
        # The deep statistics: carried across, never invented (module header).
        for k in COH_KEYS:
            out[prn][k] = src.get(k)
    return apply_presence(out, k_sigma, q_fallback, probe_prns=probe_prns,
                          deep_gate_prns=deep_gate_prns, deep_gate_margin=deep_gate_margin,
                          admit_displaced=admit_displaced)


def recs_from_rest(get, url, chain, prns=None, timeout=5.0):
    """`prompt_cn0`'s per-record series, fetched from the gather's C++ reduction.

    Returns ({(win, slot): {prn: [e, p, l, n_inst, n_chan]}}, hops_per_record).

    ⚠️ SAME ARITHMETIC, GATED ON IDENTICAL BYTES: fleetdll_gate.py's REC-SERIES leg requires
    e/p/l to 1e-9 and n_inst/n_chan exactly, per (window, slot, PRN). That gate is the only thing
    behind this call -- broker_equiv replays only what goes through transport, and the gather
    is a raw socket, so a replay carries no telemetry and this path is never exercised.

    WHY. prompt_cn0 walks the gathered frames a SECOND time, after the comb DLL has already
    walked them for its own reduction: the same ~140k channel-tuples per chain per cycle,
    twice over, on a process pinned at 100% of one core by the GIL.
    """
    d = get("%s/get_rec_taps" % url.rstrip("/"), timeout=timeout) or {}
    want = None if prns is None else set(int(p) for p in prns)
    recs = {}
    for row in (d.get(chain) or []):
        win, slot, prn, n_inst, n_chan, e, p, l = row
        prn = int(prn)
        if want is not None and prn not in want:
            continue
        recs.setdefault((int(win), int(slot)), {})[prn] = [float(e), float(p), float(l),
                                                           int(n_inst), int(n_chan)]
    return recs, int((d.get("hops_per_record") or {}).get(chain) or 0)


def prompt_cn0(client, chain, n_win=32, lag=1, prns=None, probe_prns=None,
               min_sig=5.0, min_instances=2, hop_s=5.12e-6, keep_records=False,
               min_used=8, recs_src=None):
    """THE SERVED C/N0 (task #57): per-record prompt power, q-gated, probe-debiased.

    Replaces the deep fold as the radiometry. The fold RE-SEARCHES a residual rate per
    integration -- a fit on something the tracking loop already fixed -- and that re-search
    gives it ~20 dB of its own scatter, measured PAIRED on the same records (>10x on 23% of
    cycles, 7912 samples, 2026-08-15). Nothing derived from it is a measurement of the
    satellite. This estimator fits NOTHING: the rate is the tracker's (already applied to the
    despread), the tap is the loop's, and the only arithmetic is a debiased power ratio.

    THE THREE INGREDIENTS, each load-bearing:

      * PER-RECORD prompt power, LOBE-COHERENT: |SUM_c G_c|^2 / (SUM_c E_c)^2 over every
        channel of every sender that reached the record (lobe_taps' record, un-averaged).
        Normalised, so a signal reads the same however many senders are up, and the noise
        floor -- measured from the probes through the identical reduction -- falls with the
        channel count. A record fewer than `min_instances` senders reached is excluded
        rather than reweighted: no per-sender arithmetic survives anywhere in this path.

      * PRESENT, BY A SELECTION-FREE STATISTIC. A PRN is served only if its mean debiased
        prompt power over EVERY record (no gate anywhere upstream of it) is `min_sig`
        standard errors above zero -- `sig_inc`, the incoherent detection significance,
        with the scatter measured from the same records. Under the null the per-record
        prompt power is ONE complex dof in the lobe currency (Exp: sd = mean), so a probe's
        t is N(0,1) and a bar of 5 admits nothing that is not there; the weakest satellite
        this can serve at 384 records is ~14 dB-Hz, below anything the loop holds. This is
        the gate that keeps noise off the served rows -- NOT the lock conditional below,
        whose per-record statistic has a tail that no bar can be set on: in the lobe
        currency 9% of probe RECORDS clear a median+3*MAD q bar, and at min_used 8 that
        served below-horizon probes and untracked PRNs at 18-21 dB-Hz, duty 0.08-0.11.

      * LOCK-CONDITIONAL, PER WINDOW, WITHOUT A CONSTANT. C/N0 is radiometry CONDITIONAL ON
        LOCK: a record where the tap sat off the peak measures the tap, not the satellite,
        and averaging it in is the incoherent estimator's bias (#24). A window's records
        are kept when THE PROMPT IS THE TALLEST TAP on the window's summed E, P, L: true on
        the peak for every correlation shape the taps straddle, false once the tap is off
        by more than half the E/L spacing -- so it needs neither the spacing nor the
        chain's on-peak q, which differ per chain (4.4 on L5 at 0.5 chip, 6.7 on L2C at
        2.0) and would make any q bar a STRENGTH criterion: on-peak q falls with C/N0
        because E and L carry noise power, and the probe-derived bar was dropping half the
        records of a locked 31 dB-Hz satellite (duty 0.47, +0.9 dB selection bias). Noise
        passes this 1/3 of the time, which is why presence must never rest on it.
        ⚠️ This on-peak bias is CORRECT here and would be the #49 latch in a trim gate --
        an estimator publishes its duty and lets the consumer decline; a gate starves the
        loop. Do not transplant this bar into presence logic.
        ⚠️ SELECTION BIAS AT THE MARGIN: P and the conditional share noise, so a satellite
        passing only on upward fluctuations reads high. Judged per RECORD that is +0.7 dB at
        25 dB-Hz and +2 dB at 21 (duty 0.86 / 0.69); judged on the WINDOW's summed taps --
        the shortest span an off-peak episode can have, the loop acts on 30 s -- it is
        +0.1 / +0.6 dB with duty 0.99 / 0.92, and +1.5 dB at 19 dB-Hz where duty is 0.78.
        `duty` is published precisely so a low-duty cn0 can be declined; near 1 it is
        unbiased.

      * PROBE-DEBIASED. E[|P|^2] = |s|^2 + sigma^2, and sigma^2 is MEASURED as the median
        per-record prompt power of the below-horizon probes -- the only rows that are noise
        by construction ([[chord-deep-snr-fires-on-noise]]). NO PEER FALLBACK: without
        probes this returns {} rather than quietly rebuilding the #49 peer competition
        (the tracked population's median is a SIGNAL level; see apply_presence).

    C/N0 = 10*log10(rho / T_rec) with rho = (P - sigma^2)/sigma^2 meaned over gated records
    and T_rec the record's coherent span (hops_per_record * hop_s, read from the frames).
    Every normalisation upstream of P -- element cal, channel weights, the 4+4b scale --
    cancels in the ratio because the probes ride the identical pipeline.

    Returns {prn: {cn0_db, rho, sig_inc, duty, n_used, n_rec, split_db, sigma2, q_noise,
    min_sig, t_rec_s, n_probe_rec, probe}} for every PRN seen (probes included, flagged --
    their cn0 MUST be None, which is what the self-test and the AUC leg of the validation
    bar check). `q_noise` is the probes' median per-record q, the no-peak value, served as
    a diagnostic only.
    `split_db` is the even/odd-record self-consistency in dB: free, and it is the
    split-half witness of the validation bar. keep_records=True adds `recs`
    [(win, slot, rho, q, gated)] for the offline tools; the broker must not carry it.
    """
    probe_prns = set(int(p) for p in (probe_prns or ()))
    want = None
    if prns is not None:
        want = set(int(p) for p in prns) | probe_prns
    if recs_src is not None:
        # THE C++ ARM. Depth is the gather's taps_win, matched to this broker's telem-windows.
        recs, _hpr = recs_src(chain, want)
        t_rec = (_hpr * hop_s) if _hpr > 0 else None
        wins = []
    else:
        wins = client.windows(chain, lag=lag)
        if not wins:
            return {}
        wins = wins[-int(n_win):]
        t_rec = None
        recs = {}   # (win, slot) -> {prn: [e, p, l, n_inst, n_chan]}
    for w in wins:
        for _inst, f in client.frame_set(chain, w).items():
            if t_rec is None and getattr(f, "hops_per_record", 0) > 0:
                t_rec = f.hops_per_record * hop_s
    if wins:
        recs = lobe_records(client, chain, wins, prns=want)
    if not recs or t_rec is None:
        return {}

    # Per-PRN time-ordered per-record series: (win, slot, p, q, on_peak), on_peak judged on
    # the WINDOW's summed taps (see the docstring: the conditional's noise is what biases
    # the served number, and a window is the shortest span an off-peak episode can have).
    series = {}
    by_win = {}   # (prn, win) -> [E, P, L] summed over the window's records
    for key in sorted(recs):
        for prn, (e, p, l, n, _nch) in recs[key].items():
            if n < min_instances:
                continue
            q = 2.0 * p / (e + l) if (e + l) > 0.0 else 0.0
            series.setdefault(prn, []).append((key[0], key[1], p, q))
            t = by_win.setdefault((prn, key[0]), [0.0, 0.0, 0.0])
            t[0] += e
            t[1] += p
            t[2] += l
    for prn, rows in series.items():
        series[prn] = [(w, r, p, q, by_win[(prn, w)][1] > by_win[(prn, w)][0]
                        and by_win[(prn, w)][1] > by_win[(prn, w)][2]) for w, r, p, q in rows]

    # THE NOISE ANCHOR. Pooled over the whole capture rather than per record: the per-record
    # median of 3 probes carries ~20% scatter, the pooled one ~1/sqrt(N); the #56 power
    # swings are ~hourly against this window's ~1.3 s, so pooling loses nothing they move.
    probe_p, probe_q = [], []
    for prn in probe_prns:
        for _w, _r, p, q, _on in series.get(prn, ()):
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
    # mean over records <= 20x median keeps a single contaminated probe record from
    # dragging the anchor by more than ~2% of it. ⚠️ THE CLIP MUST CLEAR THE TAIL: in the
    # lobe currency the probe power is ONE complex dof (Exp), and a clip at 8x median
    # (5.5 sigma^2) cuts 0.4% of the records but 2.7% of the mean, which reads as +0.12 dB
    # on every served satellite (measured on the self-test, 6 seeds); at 20x the tail
    # above the clip is 1e-5 of the mean.
    _med = probe_p[len(probe_p) // 2]
    if _med <= 0.0:
        return {}
    _kept = [x for x in probe_p if x <= 20.0 * _med]
    sigma2 = sum(_kept) / len(_kept)
    if sigma2 <= 0.0:
        return {}
    q_noise = probe_q[len(probe_q) // 2]

    out = {}
    for prn, rows in series.items():
        rho_all, rho_gated, rec_rows = [], [], []
        for w, r, p, q, gated in rows:
            rho = (p - sigma2) / sigma2
            rho_all.append(rho)
            if gated:
                rho_gated.append(rho)
            if keep_records:
                rec_rows.append((w, r, rho, q, gated))
        n_used, n_tot = len(rho_gated), len(rows)
        # THE INCOHERENT DETECTION SIGNIFICANCE, on EVERY record: mean / (std/sqrt(n)) of the
        # ungated per-record rho. No distributional assumption -- the scatter is MEASURED
        # from the same records the mean is, so scintillation and any residual
        # non-whiteness make it conservative rather than wrong -- and no selection: the
        # gated records are the upward fluctuations by construction, and a t on them reads
        # a healthy significance off pure noise. This is the presence verdict below.
        #
        # ⚠️ WHY THIS EXISTS: `sig` was the deep fold's number, which read single digits
        # while the SEARCH saw the same satellites at hundreds of sigma (KV, 2026-08-15).
        # Replacing it with the COHERENT fold's significance is right only where the fold
        # actually cohered; where it did not, this is the honest detection statement, and
        # it is large for exactly the satellites that are obviously there.
        sig_inc = None
        if n_tot >= 8:
            _m = sum(rho_all) / n_tot
            _v = sum((x - _m) ** 2 for x in rho_all) / (n_tot - 1)
            _se = (_v / n_tot) ** 0.5
            if _se > 0.0:
                sig_inc = _m / _se
        present = sig_inc is not None and sig_inc >= min_sig
        # ⚠️ NOT PRESENT, OR TOO FEW ON-PEAK RECORDS, IS NOT A MEASUREMENT -- SERVE NOTHING.
        # The lock conditional and the signal share noise, so at low duty the records that
        # PASS are precisely the upward fluctuations: the mean of a handful of them is
        # biased high and scatters enormously (E19 transiting boresight on gal_e5b at q
        # 0.36, duty 0.01, n_used 1, served 27.8 dB-Hz off ONE record; the viewer plotted a
        # decade of that as a C/N0 history). `duty` was published so a consumer could
        # decline the number, but publishing a declinable number and hoping is not a gate
        # -- this is (see also the no-peer-fallback rule for the floor). duty/n_used/sig_inc
        # are still served, so "tracked but not measurable" stays visible and is
        # distinguishable from "never seen".
        rho_mean = ((sum(rho_gated) / n_used)
                    if present and n_used >= min_used else None)
        cn0 = (10.0 * math.log10(rho_mean / t_rec)
               if rho_mean is not None and rho_mean > 0.0 else None)
        # Even/odd split of the GATED records: the self-consistency of the number served.
        split_db = None
        if present and n_used >= 8:
            re_ = sum(rho_gated[0::2]) / len(rho_gated[0::2])
            ro_ = sum(rho_gated[1::2]) / len(rho_gated[1::2])
            if re_ > 0.0 and ro_ > 0.0:
                split_db = 10.0 * math.log10(re_ / ro_)
        out[prn] = {"cn0_db": cn0,
                    "rho": rho_mean,
                    "sig_inc": sig_inc,
                    "duty": n_used / float(n_tot) if n_tot else 0.0,
                    "n_used": n_used, "n_rec": n_tot,
                    "split_db": split_db,
                    "sigma2": sigma2, "q_noise": q_noise, "min_sig": min_sig,
                    "t_rec_s": t_rec,
                    "n_probe_rec": len(probe_p),
                    "probe": prn in probe_prns}
        if keep_records:
            out[prn]["recs"] = rec_rows
    return out


def coh_cn0(client, chain, rates=None, n_win=32, lag=1, prns=None, probe_prns=None,
            min_instances=2, hop_s=5.12e-6, keep_series=False, resid_s=0.025):
    """THE KNOWN-RATE COHERENT C/N0 (task #57 step 3): the ~1 s fold, with NO fit in it.

    ⚠️ #57 AMENDMENT (2026-08-17): "no fit in it" is no longer literally true -- see THE
    RESIDUAL-RATE FIT below. The injected rate is one cycle old and the true residual
    wobbles +-8 Hz poll-to-poll against a ~1 s fold, so the headline duty-cycled 60 dB on
    whether the injection happened to match (eta 0<->68 on G9 while code q sat at 3.0-3.5
    -- the fleet-wide "sig oscillation" of 08-17, [[chord-carrier-is-the-gate]]). The fix
    is a ONE-PARAMETER within-integration correction: derotate at a center rate, bin each
    instance's series into `resid_s` segments, and take the single-lag phase step
    f_res = arg(sum_k,inst S_{k+1} conj(S_k)) / (2 pi resid_s) -- per-instance phase
    offsets cancel exactly in the product, no unwrap, Nyquist +-1/(2 resid_s) = +-20 Hz at
    the default -- then a lag-8 refine. TWO CENTERS are tried, injected and ZERO, picked
    on stage-1 |R|: 12 min on sky showed the injected rate itself swings +-10 Hz
    cycle-to-cycle (the fcoh fit's own noise) and a single fit centered there aliases,
    while physics bounds the TRUE residual near zero ([[chord-deep-rate-alias]]).
    Everything then folds at the picked center + f_res.
    THE CALIBRATION BURDEN MOVES TO THE PROBES: a fitted rate always gains a little on
    noise, so the probes ride the IDENTICAL fit and the floor absorbs that gain -- `sig`
    stays probe-calibrated by construction, and `floor_white` still audits whiteness.
    f_res is served per PRN (`rate_resid_hz`, with `rate_pairs` lag-pair support): it is
    the per-satellite carrier-rate innovation the #83 Phase 3 controller will consume,
    measured here first. resid_s=0 restores the pure no-fit fold.

    The deep fold's defect was never coherence -- it was the per-integration rate RE-SEARCH,
    which re-finds a satellite wherever the tap sits and carries ~20 dB of its own paired
    scatter doing so ([[chord-deep-snr-fires-on-noise]]). This fold keeps the coherent gain
    (~10log10(n) over per-record -- the deep-sidelobe sensitivity) and removes the search:
    the residual carrier rate is INJECTED by the caller (`rates`, per PRN), taken from the
    PREVIOUS cycle's record-stream fit -- split-half sigma 37-46 mHz, i.e. ~0.25 rad over a
    1 s fold -- or 0.0 for a dead-reckoned satellite whose NCO is the model. Nothing about
    this integration is chosen by looking at this integration.

    ⚠️ AND THAT IS WHY IT CANNOT FIRE ON NOISE. A fold at a FIXED rate over noise is noise:
    E[|mean|^2] = sigma^2/n, exactly what the probes -- folded IDENTICALLY, each at its own
    (zero) rate entry -- measure as the floor. The re-search was the mechanism that turned
    noise into "detections"; with it gone, no q gate is needed here, and none is applied:
    the deep-sidelobe satellites this exists for are precisely the ones a per-record gate
    can never pass.

    TWO SERIES, BOTH FOLDED, BOTH SERVED -- their offsets are measured, not assumed:
      raw  the record-header prompt (REC_P). Absolutely calibrated against the probes, but
           the per-record COMMON sky phase (~0.75 rad, white in time -- gnssElemCal.hpp)
           decoheres the cross-record sum by ~e^{-sigma^2} (~-2.4 dB expected).
      sky  the split-aperture sky-corrected prompt (REC_SKY), which removes that phase per
           record -- at the one-way split's aperture cost (half the array integrates, half
           references), a CONVENTION offset the probes' own sky fold carries identically.
    Strong satellites tracked by cn0_prompt are the calibration: each fold's offset from
    the per-record estimator is measured on sky by scripts/gnss/kcoh_gate.py, never argued.

    Per (PRN, instance): Abar = (1/n) sum_k A_k exp(-2 pi i f t_k), t_k = hop_k * hop_s
    (absolute hops -- exact, shared across the fleet). Fleet: mean over instances of
    |Abar|^2. Debias against the probes' clipped-mean folded power (the Gamma-median
    lesson, [[chord-cn0-prompt-estimator]]).

    ⚠️ THE POWER MEAN OVER INSTANCES IS A KNOWN LIMITATION, NOT A DESIGN. This used to
    carry a comment calling cross-instance coherence "the delay-alignment problem [that]
    belongs to the combine, not the radiometry". THAT WAS FALSE and it has misled work
    repeatedly. An instance is an ARBITRARY GROUP OF FREQUENCY CHANNELS -- `freq_id mod 8`
    routing applied AFTER the signal path, one PFB, one set of raw samples. Channels inside
    one instance do not cohere any better than channels in different ones; there is nothing
    physical at the boundary. The instances run in lockstep, and ANY mismatch between them
    is a BUG, full stop -- never a property to design around. We already fit the carrier
    phase ACROSS THE BAND (#32), which is exactly a cross-instance phase measurement.

    So summing |Abar|^2 instead of coherently combining Abar throws away real sensitivity
    (~10log10(n_inst) at 12 instances) and, worse, makes the estimator BLIND to the very
    cross-instance phase errors that would be bugs. Fixing it is its own change with its
    own gate; until then this is a floor to be lifted, and no reader should take the shape
    of this reduction as evidence that instances are independent.
    C/N0 = 10log10(rho / T_coh), T_coh = n_rec_mean * t_rec; identical currency to
    cn0_prompt by construction, so on a strong stationary satellite THE TWO MUST AGREE --
    that standing cross-check is the point of serving both.

    Returns {prn: {cn0_db, cn0_sky_db, sig, sig_sky, rho, n_rec, n_src, eta, eta_sky,
    rate_hz, rate_src, t_coh_s, sigma2, sigma2_sky, n_probe, probe}}. `eta` is the
    coherence efficiency |Abar|^2 * n / mean|A|^2 -- n for a perfectly coherent series, ~1
    for noise -- the live diagnostic of whether coherence actually accrued (a re-pin
    discontinuity or a wrong rate shows up HERE, not as a silent low number).
    keep_series adds per-instance {inst: [(hop, re, im, sky_re, sky_im)]} for the offline
    tools (shuffled-null and genie legs); the broker must not carry it.
    """
    rates = rates or {}
    probe_prns = set(int(p) for p in (probe_prns or ()))
    want = None
    if prns is not None:
        want = set(int(p) for p in prns) | probe_prns
    wins = client.windows(chain, lag=lag)
    if not wins:
        return {}
    wins = wins[-int(n_win):]
    t_rec = None
    ser = {}   # (prn, inst) -> [(hop, A_raw, A_sky|None)]
    for w in wins:
        for inst, f in client.frame_set(chain, w).items():
            if t_rec is None and getattr(f, "hops_per_record", 0) > 0:
                t_rec = f.hops_per_record * hop_s
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                hop = f.hop(r)
                for prn in f.prns():
                    if want is not None and prn not in want:
                        continue
                    row = f.row(r, prn)
                    if row is None:
                        continue
                    en = row[5]                       # REC_P_ENERGY
                    if en <= 0.0:
                        continue
                    a = complex(row[3], row[4]) / en  # REC_P_RE/IM
                    skr, ski = row[24], row[25]       # REC_SKY_RE/IM (0 = absent)
                    sky = (complex(skr, ski) / en) if (skr != 0.0 or ski != 0.0) else None
                    ser.setdefault((prn, inst), []).append((hop, a, sky))
    if not ser or t_rec is None:
        return {}

    def _fold(rows, f_hz, pick):
        """(|mean|^2, n, mean|A|^2) of the derotated series; pick selects raw/sky."""
        s = 0j
        pw = 0.0
        n = 0
        for hop, a, sky in rows:
            v = a if pick == 0 else sky
            if v is None:
                continue
            s += v * cmath.exp(-2j * math.pi * f_hz * hop * hop_s)
            pw += abs(v) ** 2
            n += 1
        if n == 0:
            return None, 0, 0.0
        return abs(s / n) ** 2, n, pw / n

    # Per-PRN fleet reduction: per-instance folds, then the POWER mean over instances.
    # ⚠️ See the docstring: instances are an arbitrary freq_id grouping, they must cohere,
    # and this power mean is a limitation to lift -- not a statement about the sky.
    # ⚠️ RATE SANITY CLAMP. Physics bounds the residual at ~±10 Hz (decohered sats,
    # [[chord-deep-rate-alias]]); a rate fit taken on a re-acquiring fleet hands back
    # ±30-45 Hz of pure noise (measured 2026-08-15 16:52, minutes after a node restart),
    # and folding a REAL signal at a noise rate destroys it. Beyond the bound the honest
    # statement is "rate unknown": fold at 0 and say so in rate_src, so eta carries the
    # cost visibly instead of the C/N0 silently eating it.
    RATE_MAX_HZ = 15.0
    # ---- #57 THE RESIDUAL-RATE FIT (see the docstring amendment) ----------------------
    # Grouped per PRN first: the residual is a property of the SATELLITE, common across
    # instances, and the single-lag product S_{k+1}*conj(S_k) cancels each instance's
    # constant phase offset exactly -- so all instances pool into one estimate instead of
    # twelve noisy ones. Probes go through this same path: the fit's noise gain lands in
    # the floor, where it belongs.
    by_prn = {}
    for (prn, inst), rows in ser.items():
        by_prn.setdefault(prn, {})[inst] = rows
    acc = {}
    clamped = set()
    resid = {}   # prn -> (f_res_hz, n_lag_pairs)
    for prn, insts in by_prn.items():
        f_hz = float(rates.get(prn) or 0.0)
        if abs(f_hz) > RATE_MAX_HZ:
            f_hz = 0.0
            clamped.add(prn)
        def _fit_about(f_c):
            """Two-stage residual about center rate f_c: (f_res, n_pair, |R|_lag1).

            Stage 1 is the single-lag segment product summed over instances --
            per-instance phase offsets cancel exactly in S_{k+1}conj(S_k), Nyquist
            +-1/(2 resid_s). Stage 2 (lag M) exists because the self-test FAILED the
            single stage: ~0.1 Hz of lag-1 noise times a multi-second fold measured
            -1.2 dB on a clean series at the exact rate; the long lag divides the
            phase-to-frequency lever by M, and its narrow Nyquist is safe because
            stage 1 already brought the residual inside a fraction of a Hz.
            |R| at lag 1 is returned as the coherence score for the hypothesis pick.
            """
            segs = []
            for rows in insts.values():
                seg = {}
                for hop, a, sky in rows:
                    t = hop * hop_s
                    seg.setdefault(int(t / resid_s), []).append(
                        a * cmath.exp(-2j * math.pi * f_c * t))
                segs.append({k: sum(v) for k, v in seg.items()})
            R = 0j
            n1 = 0
            for ph in segs:
                for k, s1 in ph.items():
                    s2 = ph.get(k + 1)
                    if s2 is not None:
                        R += s2 * s1.conjugate()
                        n1 += 1
            # >= 4 adjacent-segment pairs before believing the step: a 1-2-pair "fit"
            # is a coin flip, and folding at a coin flip is the deep fold's disease.
            if n1 < 4 or abs(R) == 0.0:
                return 0.0, n1, 0.0
            fr = cmath.phase(R) / (2.0 * math.pi * resid_s)
            M = 8
            R2 = 0j
            n2 = 0
            for ph in segs:
                for k, s1 in ph.items():
                    s2 = ph.get(k + M)
                    if s2 is not None:
                        R2 += s2 * s1.conjugate()
                        n2 += 1
            if n2 >= 4 and abs(R2) > 0.0:
                rot = cmath.exp(-2j * math.pi * fr * M * resid_s)
                fr += cmath.phase(R2 * rot) / (2.0 * math.pi * M * resid_s)
            return fr, n1, abs(R)

        f_res, n_pair = 0.0, 0
        f_center = f_hz
        if resid_s and resid_s > 0.0:
            # TWO HYPOTHESES, because 12 min on sky said one is not enough: the INJECTED
            # rate itself swings +-10 Hz cycle-to-cycle (the fcoh fit's own noise), which
            # lands outside a single fit's capture and aliases -- gal_e5a PRN 6 read sig
            # 8869 -> 316 -> 20424 in 70 s. Physics bounds the TRUE residual at ~+-10 Hz
            # ([[chord-deep-rate-alias]]), so a second fit centered at ZERO reaches
            # everything a wild injection loses. Picked on stage-1 |R| (coherence of the
            # segment stream, magnitude only); probes ride the identical two-way pick, so
            # its noise gain prices itself into the floor. This is a 2-hypothesis choice,
            # not a rate search -- the deep fold's disease was a GRID argmax.
            f_res, n_pair, score = _fit_about(f_hz)
            if f_hz != 0.0:
                fr0, n0, score0 = _fit_about(0.0)
                if score0 > score:
                    f_center, f_res, n_pair = 0.0, fr0, n0
        resid[prn] = (f_center + f_res - f_hz, n_pair)
        for inst, rows in insts.items():
            pr, nr, pinc = _fold(rows, f_center + f_res, 0)
            ps, ns, pinc_s = _fold(rows, f_center + f_res, 1)
            if pr is None:
                continue
            d = acc.setdefault(prn, {"raw": [], "sky": [], "inc": [], "inc_sky": [],
                                     "n": [], "insts": 0, "series": {}})
            d["raw"].append(pr)
            d["inc"].append(pinc)
            d["n"].append(nr)
            if ps is not None and ns >= max(4, nr // 2):
                d["sky"].append(ps)
                d["inc_sky"].append(pinc_s)
            d["insts"] += 1
            if keep_series:
                d["series"][inst] = [(h, a.real, a.imag,
                                      s.real if s is not None else None,
                                      s.imag if s is not None else None)
                                     for h, a, s in rows]

    def _mean(v):
        return sum(v) / len(v) if v else None

    def _floor(vals):
        """Clipped MEAN of the probe population -- the median is Gamma-biased low."""
        if len(vals) < 2:
            return None
        s = sorted(vals)
        med = s[len(s) // 2]
        if med <= 0.0:
            return None
        kept = [x for x in s if x <= 8.0 * med]
        return sum(kept) / len(kept)

    # ---- THE FOLD FLOOR: sigma^2/n FROM THE PER-RECORD ANCHOR, NOT FROM THE FOLDS -----
    # White noise folds down as exactly sigma^2/n, and sigma^2 is measured by the probes'
    # PER-RECORD incoherent power -- thousands of draws (~1.5%), the same anchor
    # cn0_prompt uses. Anchoring on the probes' FOLDED powers instead would rest the whole
    # C/N0 on n_probes x n_inst (~36 in production, 9 in the self-test) exponential draws:
    # ~0.7 dB of floor noise per cycle, and the self-test caught a +3.4 dB error from
    # exactly that population being small. The whiteness assumption is LOAD-BEARING, so it
    # is served, not assumed: `floor_white` is the ratio of the directly-measured probe
    # fold floor to sigma^2/n -- ~1 when the noise is white record-to-record; a sustained
    # excess means correlated noise (RFI) and the C/N0 must not be believed at face value.
    probe_raw = [x for p in probe_prns for x in acc.get(p, {}).get("raw", ())]
    probe_sky = [x for p in probe_prns for x in acc.get(p, {}).get("sky", ())]
    s2_inc = _floor([x for p in probe_prns for x in acc.get(p, {}).get("inc", ())])
    s2_inc_sky = _floor([x for p in probe_prns for x in acc.get(p, {}).get("inc_sky", ())])
    if s2_inc is None or s2_inc <= 0.0:
        return {}   # no anchor, no estimate -- never a peer fallback
    probe_n = [x for p in probe_prns for x in acc.get(p, {}).get("n", ())]
    n_pr = _mean(probe_n) or 0.0
    fw_raw = (_floor(probe_raw) / (s2_inc / n_pr)
              if probe_raw and n_pr > 0 else None)
    fw_sky = (_floor(probe_sky) / (s2_inc_sky / n_pr)
              if probe_sky and s2_inc_sky and n_pr > 0 else None)

    out = {}
    for prn, d in acc.items():
        if d["insts"] < min_instances:
            continue
        pw = _mean(d["raw"])
        pinc = _mean(d["inc"])
        n_rec = _mean(d["n"]) or 0.0
        if pw is None or n_rec < 4:
            continue
        t_coh = n_rec * t_rec
        fl = s2_inc / n_rec                     # the white-noise fold floor for THIS n
        rho = (pw - fl) / fl
        row = {"cn0_db": (10.0 * math.log10(rho / t_coh) if rho > 0.0 else None),
               "sig": pw / fl,
               "rho": rho,
               "n_rec": int(round(n_rec)), "n_src": d["insts"],
               # coherence efficiency: n for coherent, ~1 for noise. Uses the DEBIASED
               # numerator against the debiased per-record power so a strong satellite's
               # eta is not diluted by the noise term.
               "eta": ((pw - fl) * n_rec / (pinc - s2_inc)
                       if (pinc is not None and pinc > s2_inc and rho > 0.0) else None),
               "rate_hz": 0.0 if prn in clamped else float(rates.get(prn) or 0.0),
               # #57: the within-integration residual (see the docstring amendment) --
               # the fold above ran at rate_hz + rate_resid_hz. This is the per-satellite
               # carrier-rate innovation the #83 Phase 3 controller will consume.
               "rate_resid_hz": resid.get(prn, (0.0, 0))[0],
               "rate_pairs": resid.get(prn, (0.0, 0))[1],
               "rate_src": ("clamped" if prn in clamped
                            else "rate" if rates.get(prn) else "zero"),
               "t_coh_s": t_coh,
               "sigma2": fl, "n_probe": len(probe_raw),
               "floor_white": fw_raw, "floor_white_sky": fw_sky,
               "probe": prn in probe_prns,
               "cn0_sky_db": None, "sig_sky": None, "eta_sky": None}
        pws = _mean(d["sky"])
        if pws is not None and s2_inc_sky is not None and s2_inc_sky > 0.0:
            fls = s2_inc_sky / n_rec
            rho_s = (pws - fls) / fls
            row["cn0_sky_db"] = (10.0 * math.log10(rho_s / t_coh) if rho_s > 0.0 else None)
            row["sig_sky"] = pws / fls
        if keep_series:
            row["series"] = d["series"]
        out[prn] = row
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
