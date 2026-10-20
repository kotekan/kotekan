"""The fleet DLL formed in the broker from the per-channel comb.

Everything here is built from `comb_epl()`: Early, Prompt and Late per channel, per record, as
the transport ships them; no tracker-side cross-channel sum enters. gnss::FleetDll
(lib/stages/gnss/gnssFleetDll.hpp) is the C++ twin, and scripts/gnss/fleetdll_gate.py compares
the two arms on identical bytes.

Per record, per PRN, over every channel c of the lobe (all senders, each sender's columns first
multiplied by exp(+i*phi0), REC_PHI0, so they sit on one reference):

    p2 = |SUM_c G_c|^2 / (SUM_c E_c)^2 ,   e2 and l2 likewise, each tap on its own energy

then the mean over the records in the ring, and disc = (E-L)/(E+L), q = 2P/(E+L).
`comb_epl()` gives A_c = G_c/E_c and E_c per channel, so SUM_c G_c = SUM_c A_c*E_c.

The coherence unit is the lobe, never the sender. A sender is a `freq_id mod 16` grouping of
channels (one GPU's comb at 3.125 MHz stride), a transport artefact. Summing each sender's
channels coherently and then adding powers across senders would put that grouping into the
discriminator as a 320 ns grating comb in the correlation response (3.27 chips at 10.23
Mcps), with a noise floor n_sender times what the band gives. No sender count is an operand
here; it is used only as completeness: a record too few senders reached is left out, not
reweighted.

The polled discriminator (fleet_dll, compared on the COMB-DLL log line) adds the instances'
powers, so it differs from this one by the coherence unit above and by two smaller effects:
  * Averaging. Each instance's GnssCoherentCombiner runs an EMA over records; this takes a
    plain mean over the windows in the ring. They agree in the mean and differ by filter
    noise, so compare disc and q over many cycles, never one.
  * Element weights. The comb is element-combined with weights already updated by the
    record they weight, the header slots with the previous record's (telem.comb_epl). That
    moves the absolute powers by up to ~10% on the prompt, but it reaches all three taps
    through the same weights, so disc and q move far less.

What the comb gives that summed slots cannot:
  * Per-channel powers (lobe_taps' `chan`). Each channel lives on exactly one sender, so
    the per-channel powers are already fleet-wide: the frequency axis a sum destroys.
  * Exact collation. fleet_dll admits instances within `hop_window` of the newest because
    REST polls arrive whenever they arrive; frames here are keyed on an absolute window
    index, so "the same sky" is an integer equality with no tolerance.

The comb carries no deep statistics. deep_snr, deep_floor and coherence_s come from the
combiner's own fold, so fleet_dll_comb copies them from `coh_from` (the polled fleet_dll's
dict); without them the deep gate does not fire.
"""
import cmath
import math

from .fleet import apply_presence
from .telem import REC_PHI0, REC_PROJ_COST

#: The clip on an exponential probe population, in multiples of its median: records above it
#: are dropped from the clipped mean. It must clear the tail: at 8x the median the clip drops
#: ~2.6% of an exponential's mean and reads the floor ~2% low; at 20x the tail above it is
#: ~1e-5 of the mean.
PROBE_CLIP_X = 20.0


def _floor(vals, clip_x):
    """Clipped mean (at `clip_x` times the median) of a probe population; a median is biased
    low. None for fewer than two values or a non-positive median."""
    if len(vals) < 2:
        return None
    s = sorted(vals)
    med = s[len(s) // 2]
    if med <= 0.0:
        return None
    kept = [x for x in s if x <= clip_x * med]
    return sum(kept) / len(kept)


def dll_tau(disc, spacing):
    """The code discriminator -> delay estimate, chips.

    |tau| <= spacing/2 (0.25 chips at 0.5-chip spacing) whatever `disc` is, because the clamp
    is on the discriminator. The discriminator therefore moves the trim at most
    gain*spacing/2 chips per update however large the error, so the update rate is the only
    lever on slew rate; cutting the gain to compensate for a faster rate gives that slew back.

    gnss::dll_tau (lib/stages/gnss/gnssFleetDll.hpp) is this expression character for
    character, and scripts/gnss/fleetdll_gate.py compares the two on identical bytes.
    """
    return -max(-1.0, min(1.0, disc)) / 4.0 * (spacing / 0.5)


def dll_integrate(trim, disc, gain, leak, clamp, spacing):
    """One leaky-integrator update. Twin of gnss::dll_integrate.

    `leak` is per update, so the loop bandwidth scales with the update rate. In continuous
    form dT/dt = -leak*f*T + gain*f*tau: the steady state gain*tau/leak does not move with f,
    but the closed-loop and noise bandwidths both scale with it.

    That steady state is a ceiling no rate can lift. Under a railed discriminator the trim
    converges to gain*(spacing/2)/leak, 1.25 chips at gain 0.25, leak 0.05 and 0.5-chip
    spacing, so integration alone never reaches the 3-chip clamp the broker passes. A loop
    pushing at a railed discriminator without arriving is hitting this, and a faster loop will
    not fix it; the leak is dll_leak / dll_leak_present in cli._FROZEN.
    """
    t = (1.0 - leak) * trim + gain * dll_tau(disc, spacing)
    return max(-clamp, min(clamp, t))


def _lobe_fold(client, chain, wins, want, per_channel):
    """The one walk over the frames both lobe_taps and lobe_records reduce from.

    Returns (acc, chan_w): acc is {(win, slot): {prn: [gE, gP, gL, wE, wP, wL, n_chan,
    n_inst, hop, aP, aP_sq]}}, the rotated complex partial sums (not yet powers), and chan_w
    is {win: ({prn: {fid: [e, p, l, n_rec]}}, {fid: owner}, {dup fids})}.

    aP and aP_sq check the rotation: SUM_i a_i and SUM_i |a_i|^2 over the senders'
    energy-normalised, derotated prompts a_i = gP_i / eP_i * rot_i. Their cross-sender
    coherence (|SUM a|^2 - SUM |a|^2) / (SUM |a|^2 (n-1)) is ~1 when phi0 put the senders on
    one reference and ~0 when the lobe sum is adding unrelated phasors. A diagnostic beside the
    sum, never an operand in it.
    """
    # (win, slot) -> {prn: [gE, gP, gL, wE, wP, wL, n_chan, n_inst, hop, aP, aP_sq]}
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
                    rot = (
                        cmath.exp(1j * float(row[REC_PHI0])) if row is not None else 1.0
                    )
                    # Projection-cost weight (REC_PROJ_COST): this sender kept (1 - b) of the
                    # satellite's amplitude, so its partial enters with weight (1 - b) on the
                    # numerator and the energies alike: maximal-ratio across senders, and the
                    # unweighted sum exactly when every sender has the same b. -1 weighs 1;
                    # b = 1 is an absent measurement. Mirrors gnss::FleetDll::fold.
                    pc = float(row[REC_PROJ_COST]) if row is not None else -1.0
                    pw = 1.0 - min(1.0, pc) if pc >= 0.0 else 1.0
                    if pw <= 0.0:
                        continue
                    d = acc.setdefault((w, r), {}).setdefault(
                        prn, [0j, 0j, 0j, 0.0, 0.0, 0.0, 0, 0, -1, 0j, 0.0]
                    )
                    d[0] += gE * rot * pw
                    d[1] += gP * rot * pw
                    d[2] += gL * rot * pw
                    d[3] += eE * pw
                    d[4] += eP * pw
                    d[5] += eL * pw
                    d[6] += len(cmb)
                    d[7] += 1
                    d[8] = max(d[8], hop)
                    a = (
                        gP / eP * rot
                    )  # normalise first, then rotate -- the C++ arm's order
                    d[9] += a
                    d[10] += a.real * a.real + a.imag * a.imag
                    if per_channel:
                        per_fid = cw[0].setdefault(prn, {})
                        for fid, E, P, L, _w in cmb:
                            own = cw[1].setdefault(fid, inst)
                            if own != inst:
                                cw[2].add(fid)
                            # One channel's own three powers, by the same expression:
                            # |G|^2/E^2 with each tap on its own replica energy.
                            c = per_fid.setdefault(fid, [0.0, 0.0, 0.0, 0])
                            c[0] += abs(E) ** 2
                            c[1] += abs(P) ** 2
                            c[2] += abs(L) ** 2
                            c[3] += 1
    return acc, chan_w


def _rec_xcoh(d):
    """One record's cross-sender coherence from a _lobe_fold accumulator; None below two
    senders (the statistic needs a pair)."""
    if d[7] < 2 or d[10] <= 0.0:
        return None
    return (abs(d[9]) ** 2 - d[10]) / (d[10] * (d[7] - 1))


def lobe_taps(client, chain, wins, prns=None, per_channel=True, min_instances=1):
    """{prn: {e, p, l, hop, n_chan, n_rec, n_inst, xcoh, n_xcoh, chan, chan_dup}}:
    lobe-coherent per-record powers, meaned over records.

    Per record, one complex sum over every channel of every sender, each sender's columns
    first multiplied by exp(+i*phi0) (REC_PHI0: the per-sender NCO accumulator the assembler
    applied, whose zero is arbitrary per sender), then the power |SUM_c G_c|^2 / (SUM_c E_c)^2
    with each tap on its own replica energy. The sender appears nowhere in the arithmetic: not
    as a partial-power term, not as a divisor, not as a weight.

    `n_inst` is the completeness of a record (how many senders reached it) and is used for one
    thing only: a record fewer than `min_instances` senders reached is not averaged in. A
    partial record carries the same normalised signal power and more noise; whether it is
    still worth having is the caller's call.

    `chan` is {freq_id: [e, p, l, n_rec]}: the same three powers formed from one channel, kept
    unsummed, each meaned over its own live records. A freq_id two senders both carried in one
    window is a routing fault upstream: it is listed in `chan_dup` and dropped from `chan`
    rather than reported as a number that is neither sender's measurement (the lobe sum keeps
    it: a duplicated channel is still that channel's measurement, twice).

    `xcoh` is the mean over the `n_xcoh` records two or more senders reached of the
    cross-sender coherence of their derotated prompts (_lobe_fold); None when none did. ~1 says
    phi0 put the senders on one reference; ~0 says the lobe sum is adding unrelated phasors
    and the prompt above is below what per-sender summing would give. It is reported beside
    the taps and enters none of them.

    Records with no comb (PRN not despread that record, or a sender running without
    chan_export) contribute nothing rather than zeros: a zeroed record is not a measurement of
    no signal but the absence of one, and averaging it in would dilute the power.

    The order of operations is the C++ arm's (gnss::FleetDll::fold): per sender the raw
    complex is summed over its channels first, that partial sum is rotated, and the rotated
    partials are added. scripts/gnss/fleetdll_gate.py compares the two arms at 1e-9 on
    identical bytes, and a different association of the same sum is a different float.
    """
    want = None if prns is None else set(int(p) for p in prns)
    acc, chan_w = _lobe_fold(client, chain, wins, want, per_channel)
    out = {}
    complete = set()  # (win, prn) with at least one record that passed the gate
    for (w, _r), per_prn in acc.items():
        for prn, d in per_prn.items():
            if d[7] < min_instances:
                continue
            complete.add((w, prn))
            t = out.setdefault(
                prn,
                {
                    "e": 0.0,
                    "p": 0.0,
                    "l": 0.0,
                    "hop": -1,
                    "n_chan": 0.0,
                    "n_rec": 0,
                    "n_inst": 0,
                    "xcoh": None,
                    "n_xcoh": 0,
                    "chan": {},
                    "chan_dup": set(),
                },
            )
            t["e"] += (abs(d[0]) / d[3]) ** 2 if d[3] > 0.0 else 0.0
            t["p"] += (abs(d[1]) / d[4]) ** 2
            t["l"] += (abs(d[2]) / d[5]) ** 2 if d[5] > 0.0 else 0.0
            t["n_chan"] += d[6]
            t["n_rec"] += 1
            t["n_inst"] = max(t["n_inst"], d[7])
            t["hop"] = max(t["hop"], d[8])
            xc = _rec_xcoh(d)
            if xc is not None:
                t["xcoh"] = (t["xcoh"] or 0.0) + xc
                t["n_xcoh"] += 1
    if per_channel:
        for w, (per_prn, _own, dup) in chan_w.items():
            for prn, per_fid in per_prn.items():
                # A PRN whose every record in this window was incomplete contributes no lobe
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
        if t["n_xcoh"]:
            t["xcoh"] /= float(t["n_xcoh"])
        t["chan_dup"] = sorted(t["chan_dup"])
        for c in t["chan"].values():
            m = float(c[3]) or 1.0
            c[0] /= m
            c[1] /= m
            c[2] /= m
    return out


def lobe_records(client, chain, wins, prns=None):
    """{(win, slot): {prn: [e, p, l, n_inst, n_chan]}}: the same lobe-coherent powers, per
    record, un-averaged. prompt_cn0's input; the twin of gnss::FleetDll::rec_series.

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
                d[7],
                d[6],
            ]
    return recs


#: The keys carried across from the polled arm untouched. All three are products of the
#: combiner's deep fold, which is not in the comb (module docstring).
COH_KEYS = ("coh_row", "coh_src", "coh_quad")


def _check_taps_win(d, want_win, what):
    """Refuse a gather reply whose `taps_win` is not this broker's window depth.

    The gather's depth is FLEET_TRIM_TAPS_WINDOWS (config/gen_chord_gnss_config.py) and the
    Python arm's is telem-windows; nothing else ties the two at runtime, so a reply at another
    depth would silently average a different span than the arm it replaces. An empty reply (a
    failed GET) is left to the caller's own no-data path."""
    if want_win and d and int(d.get("taps_win") or 0) != int(want_win):
        raise ValueError(
            "%s depth %s != telem-windows %d" % (what, d.get("taps_win"), int(want_win))
        )


def taps_from_rest(get, url, chain, prns=None, timeout=5.0, want_win=None):
    """`lobe_taps`' per-PRN object (without the per-channel table), fetched from the gather's
    C++ reduction (<url>/get_taps) instead of rebuilt.

    The same arithmetic, not an approximation of it: gnss::FleetDll forms these taps from the
    same frames, and scripts/gnss/fleetdll_gate.py hands both arms identical bytes and requires
    e/p/l/n_chan to agree to 1e-9 with n_rec/n_inst/hop exact, per PRN and per channel. That
    gate is the only check on this call: the comb arm needs the gather's telemetry, a raw socket
    outside gnss_broker.transport, so a broker_equiv replay has none and never reaches it.

    Rebuilding this in Python walks every (window, sender, record, PRN, channel) of the
    gathered stream and allocates a Python complex object for each; the gather already holds
    the frames in C++, and the reduction is a few numbers per PRN.

    `get` is the caller's HTTP getter (transport._get), passed in so this module keeps its
    no-transport-import property and so a recording transcribes the fetch like any other.

    `want_win`, when given, is the broker's window depth; a reply served at another depth
    raises ValueError (_check_taps_win), so the caller keeps the polled discriminator.
    """
    d = get("%s/get_taps" % url.rstrip("/"), timeout=timeout) or {}
    _check_taps_win(d, want_win, "taps")
    want = None if prns is None else set(int(p) for p in prns)
    out = {}
    for prn, v in (d.get(chain) or {}).items():
        p = int(prn)
        if want is not None and p not in want:
            continue
        if int(v["n_rec"]) <= 0:
            continue
        out[p] = {
            "e": float(v["e"]),
            "p": float(v["p"]),
            "l": float(v["l"]),
            "n_chan": float(v["n_chan"]),
            "n_rec": int(v["n_rec"]),
            "n_inst": int(v["n_inst"]),
            "hop": int(v["hop"]),
            "xcoh": None if v.get("xcoh") is None else float(v["xcoh"]),
            "n_xcoh": int(v.get("n_xcoh") or 0),
        }
    return out


def fleet_dll_comb(
    client,
    chain,
    n_win=32,
    lag=1,
    min_instances=2,
    k_sigma=3.0,
    q_fallback=2.2,
    prns=None,
    probe_prns=None,
    deep_gate_prns=None,
    deep_gate_margin=3.0,
    coh_from=None,
    taps_src=None,
    admit_displaced=None,
):
    """fleet_dll's dict, from the comb. {prn: {disc, q, p_pow, hop, n_src, n_chan, ...}}.

    Same keys, same meanings, same presence policy (apply_presence, shared with fleet_dll so
    the two paths cannot drift apart in their verdicts); only the source of the three powers
    differs. Extra keys: `src` = "comb", `e_pow`/`l_pow`, `n_rec`, `xcoh` (cross-sender
    coherence of the derotated prompts, lobe_taps: the live check that the senders sit on one
    phase reference).

    The powers are lobe-coherent and normalised (lobe_taps): p_pow is |SUM_c G_c|^2 over every
    channel of the lobe divided by the summed energy squared, so for a signal it does not
    scale with how many senders or channels are up. `n_src` is the sender count behind the
    most complete record: completeness, reported, never an operand.

    `taps_src(chain, prns)`, when given, supplies the per-PRN taps (taps_from_rest) in place of
    the Python walk; `n_win`, `lag` and `min_instances` then do not apply.

    `coh_from`: a fleet_dll-shaped dict whose COH_KEYS are copied across verbatim. The deep
    gate and the displaced admission (apply_presence) read coh_row, and the publisher's
    quadrature fallback reads coh_quad; without coh_quad the published deep_snr falls back to
    the single best instance's, several dB below the fleet value, and the served series steps
    whenever the fleet gate flickers.
    """
    if taps_src is not None:
        # The C++ arm. No window selection here: the depth is the gather's `taps_win`
        # (FLEET_TRIM_TAPS_WINDOWS in config/gen_chord_gnss_config.py), which must match this
        # broker's telem-windows (taps_from_rest's want_win refuses a reply that does not);
        # fleetdll_gate.py's split-ring leg checks that the served taps take that depth rather
        # than the fast loop's. The completeness gate is applied in the C++ (its own
        # min_instances), identically.
        per_prn = taps_src(chain, prns)
        if not per_prn:
            return {}
    else:
        wins = client.windows(chain, lag=lag)
        if not wins:
            return {}
        wins = wins[-int(n_win) :]
        per_prn = lobe_taps(
            client,
            chain,
            wins,
            prns=prns,
            per_channel=False,
            min_instances=min_instances,
        )
    coh_from = coh_from or {}
    out = {}
    for prn, t in per_prn.items():
        if t["n_rec"] <= 0:
            continue
        E, P, L = t["e"], t["p"], t["l"]
        if E + L <= 0.0:
            continue
        src = coh_from.get(prn) or {}
        out[prn] = {
            "disc": (E - L) / (E + L),
            "q": 2.0 * P / (E + L),
            "p_pow": P,
            "e_pow": E,
            "l_pow": L,
            "hop": t["hop"],
            "n_src": t["n_inst"],
            "n_chan": t["n_chan"],
            "n_rec": t["n_rec"],
            "xcoh": t.get("xcoh"),
            "n_xcoh": t.get("n_xcoh", 0),
            "src": "comb",
        }
        # The deep statistics: carried across, never invented (module docstring).
        for k in COH_KEYS:
            out[prn][k] = src.get(k)
    return apply_presence(
        out,
        k_sigma,
        q_fallback,
        probe_prns=probe_prns,
        deep_gate_prns=deep_gate_prns,
        deep_gate_margin=deep_gate_margin,
        admit_displaced=admit_displaced,
    )


def recs_from_rest(get, url, chain, prns=None, timeout=5.0, want_win=None):
    """`prompt_cn0`'s per-record series, fetched from the gather's C++ reduction
    (<url>/get_rec_taps).

    Returns ({(win, slot): {prn: [e, p, l, n_inst, n_chan]}}, hops_per_record).

    The same arithmetic, gated on identical bytes: fleetdll_gate.py's REC-SERIES leg requires
    e/p/l to 1e-9 and n_inst/n_chan exactly, per (window, slot, PRN). That gate is the only
    check on this call; a broker_equiv replay never reaches it (taps_from_rest). Without this
    call prompt_cn0 walks the gathered frames a second time, after the comb DLL has walked them
    for its own reduction. `want_win` as in taps_from_rest.
    """
    d = get("%s/get_rec_taps" % url.rstrip("/"), timeout=timeout) or {}
    _check_taps_win(d, want_win, "rec-taps")
    want = None if prns is None else set(int(p) for p in prns)
    recs = {}
    for row in d.get(chain) or []:
        win, slot, prn, n_inst, n_chan, e, p, l = row
        prn = int(prn)
        if want is not None and prn not in want:
            continue
        recs.setdefault((int(win), int(slot)), {})[prn] = [
            float(e),
            float(p),
            float(l),
            int(n_inst),
            int(n_chan),
        ]
    return recs, int((d.get("hops_per_record") or {}).get(chain) or 0)


def prompt_cn0(
    client,
    chain,
    n_win=32,
    lag=1,
    prns=None,
    probe_prns=None,
    min_sig=5.0,
    min_instances=2,
    hop_s=5.12e-6,
    keep_records=False,
    min_used=8,
    recs_src=None,
):
    """The served C/N0: per-record prompt power, gated on presence and lock, probe-debiased.

    Nothing is fitted: the carrier rate is the tracker's (already applied to the despread), the
    tap is the loop's, and the only arithmetic is a debiased power ratio. A fold that
    re-searches a residual rate per integration measures its own fit as much as the satellite.

    The ingredients, each load-bearing:

      * Per-record prompt power, lobe-coherent: |SUM_c G_c|^2 / (SUM_c E_c)^2 over every
        channel of every sender that reached the record (lobe_records). Normalised, so a
        signal reads the same however many senders are up, and the noise floor, measured from
        the probes through the identical reduction, falls with the channel count. A record
        fewer than `min_instances` senders reached is excluded rather than reweighted. With
        `recs_src` (recs_from_rest) the C++ reduction supplies the records instead.

      * Presence, by a selection-free statistic. A PRN is served only if its mean debiased
        prompt power over every record (no gate upstream of it) is `min_sig` standard errors
        above zero: `sig_inc`, the incoherent detection significance, with the scatter
        measured from the same records. Under the null the per-record prompt power is one
        complex dof in the lobe currency (exponential: sd = mean), so a probe's t is N(0,1)
        and a bar of 5 admits nothing that is not there; at 384 records the weakest
        servable satellite is ~14 dB-Hz. This is the gate that keeps noise off the served
        rows, not the lock conditional below, whose per-record statistic has a tail no bar
        can be set on.

      * Lock-conditional, per window, without a constant. C/N0 is radiometry conditional on
        lock: a record where the tap sat off the peak measures the tap, not the satellite.
        A window's records are kept when the prompt is the tallest tap on the window's summed
        E, P, L: true on the peak for every correlation shape the taps straddle, false once
        the tap is off by more than half the E/L spacing. It needs neither the spacing nor the
        chain's on-peak q, which differ per chain, and a q bar would be a strength criterion,
        because on-peak q falls with C/N0 as E and L carry noise power. Noise passes this 1/3
        of the time, which is why presence must never rest on it.
        This on-peak bias is right for an estimator, which publishes its duty and lets the
        consumer decline; in a trim or presence gate it would starve the loop of the off-peak
        satellites it must pull in. Do not reuse this conditional in presence logic.
        Selection bias at the margin: P and the conditional share noise, so a satellite
        passing only on upward fluctuations reads high. Judging on the window's summed taps
        rather than per record keeps that small: about +0.1 dB at duty 0.99, +0.6 dB at 0.92
        and +1.5 dB at 0.78. `duty` is published so a low-duty cn0 can be declined.

      * Probe-debiased. E[|P|^2] = |s|^2 + sigma^2, and sigma^2 is measured as the clipped
        mean per-record prompt power of the below-horizon probes, the only rows that are
        noise by construction. No peer fallback: without probes this returns {}, because the
        tracked population's median is a signal level (see apply_presence).

    C/N0 = 10*log10(rho / T_rec) with rho = (P - sigma^2)/sigma^2 meaned over gated records
    and T_rec the record's coherent span (hops_per_record * hop_s, read from the frames).
    Every normalisation upstream of P (element cal, channel weights, the 4+4b scale) cancels
    in the ratio because the probes ride the identical pipeline.

    Returns {prn: {cn0_db, rho, sig_inc, duty, n_used, n_rec, split_db, sigma2, q_noise,
    min_sig, t_rec_s, n_probe_rec, probe}} for every PRN seen, probes included and flagged
    (a probe's cn0 should be None). `q_noise` is the probes' median per-record q, the no-peak
    value, served as a diagnostic only. `split_db` is the even/odd self-consistency of the
    gated records in dB. keep_records=True adds `recs` [(win, slot, rho, q, gated)] for
    offline use; the broker must not carry it.
    """
    probe_prns = set(int(p) for p in (probe_prns or ()))
    want = None
    if prns is not None:
        want = set(int(p) for p in prns) | probe_prns
    if recs_src is not None:
        # The C++ arm. Depth is the gather's taps_win, matched to this broker's telem-windows.
        recs, _hpr = recs_src(chain, want)
        t_rec = (_hpr * hop_s) if _hpr > 0 else None
        wins = []
    else:
        wins = client.windows(chain, lag=lag)
        if not wins:
            return {}
        wins = wins[-int(n_win) :]
        t_rec = None
        recs = {}  # (win, slot) -> {prn: [e, p, l, n_inst, n_chan]}
    for w in wins:
        for _inst, f in client.frame_set(chain, w).items():
            if t_rec is None and getattr(f, "hops_per_record", 0) > 0:
                t_rec = f.hops_per_record * hop_s
    if wins:
        recs = lobe_records(client, chain, wins, prns=want)
    if not recs or t_rec is None:
        return {}

    # Per-PRN time-ordered per-record series: (win, slot, p, q, on_peak), on_peak judged on
    # the window's summed taps (docstring: the conditional's noise is what biases the served
    # number, and a window is the shortest span an off-peak episode can have).
    series = {}
    by_win = {}  # (prn, win) -> [E, P, L] summed over the window's records
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
        series[prn] = [
            (
                w,
                r,
                p,
                q,
                by_win[(prn, w)][1] > by_win[(prn, w)][0]
                and by_win[(prn, w)][1] > by_win[(prn, w)][2],
            )
            for w, r, p, q in rows
        ]

    # The noise anchor, pooled over every probe record in the ring rather than per record: a
    # per-record median of a few probes scatters ~20%, the pooled mean ~1/sqrt(N), and the
    # noise level moves slowly against the ring's span.
    probe_p, probe_q = [], []
    for prn in probe_prns:
        for _w, _r, p, q, _on in series.get(prn, ()):
            probe_p.append(p)
            probe_q.append(q)
    if len(probe_p) < 16:
        return {}  # no anchor, no estimate -- never a peer fallback
    probe_p.sort()
    probe_q.sort()
    # The debias needs the mean, not the median: E[|P|^2] = |s|^2 + sigma^2 with sigma^2 the
    # noise power's expectation, and the per-record noise power is exponential (one complex
    # dof in the lobe currency), whose median is ln 2 of its mean. The median is only the
    # clip reference: records above PROBE_CLIP_X times it are dropped, which bounds what one
    # contaminated probe record can do to the anchor. The clip must clear the tail: at 8x the
    # median it would drop ~2.6% of the mean and read +0.1 dB on every served satellite.
    _med = probe_p[len(probe_p) // 2]
    if _med <= 0.0:
        return {}
    _kept = [x for x in probe_p if x <= PROBE_CLIP_X * _med]
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
        # The incoherent detection significance, on every record: mean / (std/sqrt(n)) of the
        # ungated per-record rho. No distributional assumption (the scatter is measured from
        # the same records as the mean, so scintillation and residual non-whiteness make it
        # conservative rather than wrong) and no selection: the gated records are the upward
        # fluctuations by construction, and a t on them reads a healthy significance off pure
        # noise. This is the presence verdict below.
        sig_inc = None
        if n_tot >= 8:
            _m = sum(rho_all) / n_tot
            _v = sum((x - _m) ** 2 for x in rho_all) / (n_tot - 1)
            _se = (_v / n_tot) ** 0.5
            if _se > 0.0:
                sig_inc = _m / _se
        present = sig_inc is not None and sig_inc >= min_sig
        # Not present, or fewer than min_used on-peak records, is not a measurement: serve no
        # cn0. The lock conditional and the signal share noise, so at low duty the records
        # that pass are the upward fluctuations, and the mean of a handful of them is biased
        # high and scatters by tens of dB. duty, n_used and sig_inc are still served, so
        # "tracked but not measurable" stays distinguishable from "never seen".
        rho_mean = (sum(rho_gated) / n_used) if present and n_used >= min_used else None
        cn0 = (
            10.0 * math.log10(rho_mean / t_rec)
            if rho_mean is not None and rho_mean > 0.0
            else None
        )
        # Even/odd split of the gated records: the self-consistency of the number served.
        split_db = None
        if present and n_used >= 8:
            re_ = sum(rho_gated[0::2]) / len(rho_gated[0::2])
            ro_ = sum(rho_gated[1::2]) / len(rho_gated[1::2])
            if re_ > 0.0 and ro_ > 0.0:
                split_db = 10.0 * math.log10(re_ / ro_)
        out[prn] = {
            "cn0_db": cn0,
            "rho": rho_mean,
            "sig_inc": sig_inc,
            "duty": n_used / float(n_tot) if n_tot else 0.0,
            "n_used": n_used,
            "n_rec": n_tot,
            "split_db": split_db,
            "sigma2": sigma2,
            "q_noise": q_noise,
            "min_sig": min_sig,
            "t_rec_s": t_rec,
            "n_probe_rec": len(probe_p),
            "probe": prn in probe_prns,
        }
        if keep_records:
            out[prn]["recs"] = rec_rows
    return out


def coh_cn0(
    client,
    chain,
    rates=None,
    n_win=32,
    lag=1,
    prns=None,
    probe_prns=None,
    min_instances=2,
    hop_s=5.12e-6,
    keep_series=False,
    resid_s=0.025,
):
    """The known-rate coherent C/N0: the ~1 s fold over the ring, at an injected rate.

    The rate is not searched. The caller injects the residual carrier rate per PRN (`rates`),
    from the previous cycle's record-stream fit (fleet_coherent's rate_hz), or 0.0 for a PRN
    with none, so the fold's centre is never chosen by looking at this integration. That is
    why it cannot fire on noise: a fold at a fixed rate over noise is noise, E[|mean|^2] =
    sigma^2/n, which the probes, folded identically, measure as the floor. No q gate is
    applied, so it reaches the deep-sidelobe satellites a per-record gate never passes; the
    coherent gain over the per-record estimator is ~10log10(n) in sensitivity.

    Residual-rate fit. The injected rate is one cycle old and the true residual wobbles by Hz
    against a ~1 s fold, so a one-parameter within-integration correction is made: derotate at
    a centre rate, bin each instance's series into `resid_s` segments, and take the single-lag
    phase step f_res = arg(sum_k,inst S_{k+1} conj(S_k)) / (2 pi resid_s). Per-instance phase
    offsets cancel exactly in the product, with no unwrap; Nyquist is +-1/(2 resid_s) = +-20 Hz
    at the default. A lag-8 refine follows. Two centres are tried, the injected rate and zero,
    picked on stage-1 |R|: the injected rate itself swings by up to ~10 Hz between cycles (the
    record-stream fit's noise) and a single fit centred there aliases, while the true residual
    stays near zero. Everything then folds at the picked centre + f_res. A fitted rate always
    gains a little on noise, so the probes ride the identical fit and the floor absorbs that
    gain: `sig` stays probe-calibrated, and `floor_white` audits whiteness. f_res is served
    per PRN (`rate_resid_hz`, with `rate_pairs` lag-pair support); the joint rrate state takes
    rate_hz + rate_resid_hz under --rrate-kcoh-feed. resid_s=0 gives the pure fixed-rate fold.

    Two series, both folded and served, with offsets measured on sky rather than assumed:
      raw  the record-header prompt (REC_P). Calibrated against the probes, but the
           per-record common sky phase (close to white in time, gnssElemCal.hpp) decoheres
           the cross-record sum by ~e^{-sigma^2}.
      sky  the split-aperture sky-corrected prompt (REC_SKY), which removes that phase per
           record at the split's aperture cost (half the array integrates, half
           references), a convention offset the probes' own sky fold carries identically.

    Per (PRN, instance): Abar = (1/n) sum_k A_k exp(-2 pi i f t_k), t_k = hop_k * hop_s
    (absolute hops: exact and shared across the fleet). Fleet: mean over instances of
    |Abar|^2, debiased against the white-noise fold floor sigma^2/n, with sigma^2 the probes'
    clipped-mean per-record power.

    The power mean over instances is a known limitation, not a design. An instance is an
    arbitrary group of frequency channels (`freq_id mod 16` routing applied after the signal
    path: one PFB, one set of raw samples), and nothing physical sits at its boundary; the
    instances run in lockstep, and any mismatch between them is a bug. Summing |Abar|^2
    instead of combining Abar coherently throws away ~10log10(n_inst) of sensitivity and makes
    the estimator blind to the cross-instance phase errors that would be bugs. Do not read
    this reduction as evidence that instances are independent.

    C/N0 = 10log10(rho / T_coh), T_coh = n_rec_mean * t_rec: the same definition as
    prompt_cn0, and both are served so their difference on strong satellites is a standing
    cross-check. prompt_cn0 sums every sender coherently while this is a power mean over
    instances, so that difference includes this estimator's combine loss (up to
    ~10log10(n_inst), less when the lobe sum is not fully coherent).

    Returns {prn: {cn0_db, cn0_sky_db, sig, sig_sky, rho, n_rec, n_src, eta, eta_sky,
    rate_hz, rate_resid_hz, rate_pairs, rate_src, t_coh_s, sigma2, n_probe, floor_white,
    floor_white_sky, probe}}; `sigma2` is the fold floor sigma^2/n, and eta_sky is always
    None. `eta` is the coherence efficiency (|Abar|^2 - floor) * n / (mean|A|^2 - sigma^2):
    n for a perfectly coherent series, the live diagnostic of whether coherence accrued (a
    re-pin discontinuity or a wrong rate shows up here, not as a silent low number); None
    when either debiased power is not positive. keep_series adds per-instance
    {inst: [(hop, re, im, sky_re, sky_im)]} for offline use; the broker must not carry it.
    """
    rates = rates or {}
    probe_prns = set(int(p) for p in (probe_prns or ()))
    want = None
    if prns is not None:
        want = set(int(p) for p in prns) | probe_prns
    wins = client.windows(chain, lag=lag)
    if not wins:
        return {}
    wins = wins[-int(n_win) :]
    t_rec = None
    ser = {}  # (prn, inst) -> [(hop, A_raw, A_sky|None)]
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
                    en = row[5]  # REC_P_ENERGY
                    if en <= 0.0:
                        continue
                    a = complex(row[3], row[4]) / en  # REC_P_RE/IM
                    skr, ski = row[24], row[25]  # REC_SKY_RE/IM (0 = absent)
                    sky = (
                        (complex(skr, ski) / en) if (skr != 0.0 or ski != 0.0) else None
                    )
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

    # Per-PRN fleet reduction: per-instance folds, then the power mean over instances (a
    # limitation to lift, not a statement about the sky; see the docstring).
    # Rate sanity clamp. The true residual stays within ~+-10 Hz, while a rate fit taken on a
    # re-acquiring fleet can return tens of Hz of noise, and folding a real signal at a noise
    # rate destroys it. Beyond the bound the rate is unknown: fold at 0 and say so in
    # rate_src, so eta shows the cost instead of the C/N0 silently absorbing it.
    RATE_MAX_HZ = 15.0
    # ---- The residual-rate fit (docstring) -------------------------------------------
    # Grouped per PRN first: the residual is a property of the satellite, common across
    # instances, and the single-lag product S_{k+1}*conj(S_k) cancels each instance's
    # constant phase offset exactly, so all instances pool into one estimate. Probes go
    # through this same path, so the fit's noise gain lands in the floor.
    by_prn = {}
    for (prn, inst), rows in ser.items():
        by_prn.setdefault(prn, {})[inst] = rows
    acc = {}
    clamped = set()
    resid = {}  # prn -> (f_res_hz, n_lag_pairs)
    for prn, insts in by_prn.items():
        f_hz = float(rates.get(prn) or 0.0)
        if abs(f_hz) > RATE_MAX_HZ:
            f_hz = 0.0
            clamped.add(prn)

        def _fit_about(f_c):
            """Two-stage residual about center rate f_c: (f_res, n_pair, |R|_lag1).

            Stage 1 is the single-lag segment product summed over instances:
            per-instance phase offsets cancel exactly in S_{k+1}conj(S_k), Nyquist
            +-1/(2 resid_s). Stage 2 (lag M) refines it: ~0.1 Hz of lag-1 noise costs
            about a dB of coherence over a multi-second fold, and the long lag divides
            the phase-to-frequency lever by M. Its narrow Nyquist is safe because stage 1
            already brought the residual inside a fraction of a Hz.
            |R| at lag 1 is returned as the coherence score for the hypothesis pick.
            """
            segs = []
            for rows in insts.values():
                seg = {}
                for hop, a, sky in rows:
                    t = hop * hop_s
                    seg.setdefault(int(t / resid_s), []).append(
                        a * cmath.exp(-2j * math.pi * f_c * t)
                    )
                segs.append({k: sum(v) for k, v in seg.items()})
            R = 0j
            n1 = 0
            for ph in segs:
                for k, s1 in ph.items():
                    s2 = ph.get(k + 1)
                    if s2 is not None:
                        R += s2 * s1.conjugate()
                        n1 += 1
            # >= 4 adjacent-segment pairs before believing the step: a fit on one or two
            # pairs is a coin flip, and folding at it would decohere the signal.
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
            # Two hypotheses: the injected rate itself can swing ~10 Hz between cycles (the
            # record-stream fit's noise), outside a single fit's capture, where it aliases.
            # The true residual stays near zero, so a second fit centred at zero reaches
            # everything a wild injection loses. Picked on stage-1 |R| (coherence of the
            # segment stream, magnitude only); probes ride the identical two-way pick, so
            # its noise gain prices itself into the floor. A choice between two
            # hypotheses, not a grid search over rates.
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
            d = acc.setdefault(
                prn,
                {
                    "raw": [],
                    "sky": [],
                    "inc": [],
                    "inc_sky": [],
                    "n": [],
                    "insts": 0,
                    "series": {},
                },
            )
            d["raw"].append(pr)
            d["inc"].append(pinc)
            d["n"].append(nr)
            if ps is not None and ns >= max(4, nr // 2):
                d["sky"].append(ps)
                d["inc_sky"].append(pinc_s)
            d["insts"] += 1
            if keep_series:
                d["series"][inst] = [
                    (
                        h,
                        a.real,
                        a.imag,
                        s.real if s is not None else None,
                        s.imag if s is not None else None,
                    )
                    for h, a, s in rows
                ]

    def _mean(v):
        return sum(v) / len(v) if v else None

    # ---- The fold floor: sigma^2/n from the per-record anchor, not from the folds -------
    # White noise folds down as exactly sigma^2/n, and sigma^2 is measured by the probes'
    # per-record incoherent power: thousands of draws. The probes' folded powers are only
    # n_probes x n_inst exponential draws, too few to anchor the C/N0 on (most of a dB of
    # floor noise per cycle). Whiteness is therefore load-bearing, so it is served, not
    # assumed: `floor_white` is the directly measured probe fold floor over sigma^2/n, ~1
    # when the noise is white record to record; a sustained excess means correlated noise
    # (RFI), and the C/N0 must not be believed at face value.
    # The two populations take different clips. The anchor's are per-instance means over n
    # records, far narrower than an exponential, so 8x the median never cuts noise and only
    # bounds a contaminated probe instance. The folded powers are single exponential draws,
    # whose tail an 8x clip would cut, reading floor_white low (PROBE_CLIP_X).
    probe_raw = [x for p in probe_prns for x in acc.get(p, {}).get("raw", ())]
    probe_sky = [x for p in probe_prns for x in acc.get(p, {}).get("sky", ())]
    s2_inc = _floor([x for p in probe_prns for x in acc.get(p, {}).get("inc", ())], 8.0)
    s2_inc_sky = _floor(
        [x for p in probe_prns for x in acc.get(p, {}).get("inc_sky", ())], 8.0
    )
    if s2_inc is None or s2_inc <= 0.0:
        return {}  # no anchor, no estimate -- never a peer fallback
    probe_n = [x for p in probe_prns for x in acc.get(p, {}).get("n", ())]
    n_pr = _mean(probe_n) or 0.0
    fw_raw = (
        _floor(probe_raw, PROBE_CLIP_X) / (s2_inc / n_pr)
        if probe_raw and n_pr > 0
        else None
    )
    fw_sky = (
        _floor(probe_sky, PROBE_CLIP_X) / (s2_inc_sky / n_pr)
        if probe_sky and s2_inc_sky and n_pr > 0
        else None
    )

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
        fl = s2_inc / n_rec  # the white-noise fold floor for this n
        rho = (pw - fl) / fl
        row = {
            "cn0_db": (10.0 * math.log10(rho / t_coh) if rho > 0.0 else None),
            "sig": pw / fl,
            "rho": rho,
            "n_rec": int(round(n_rec)),
            "n_src": d["insts"],
            # Coherence efficiency, n for a coherent series: the debiased numerator over
            # the debiased per-record power, so a strong satellite's eta is not diluted by
            # the noise term.
            "eta": (
                (pw - fl) * n_rec / (pinc - s2_inc)
                if (pinc is not None and pinc > s2_inc and rho > 0.0)
                else None
            ),
            "rate_hz": 0.0 if prn in clamped else float(rates.get(prn) or 0.0),
            # The within-integration residual: the fold above ran at
            # rate_hz + rate_resid_hz, which the joint rrate state takes under
            # --rrate-kcoh-feed.
            "rate_resid_hz": resid.get(prn, (0.0, 0))[0],
            "rate_pairs": resid.get(prn, (0.0, 0))[1],
            "rate_src": (
                "clamped" if prn in clamped else "rate" if rates.get(prn) else "zero"
            ),
            "t_coh_s": t_coh,
            "sigma2": fl,
            "n_probe": len(probe_raw),
            "floor_white": fw_raw,
            "floor_white_sky": fw_sky,
            "probe": prn in probe_prns,
            "cn0_sky_db": None,
            "sig_sky": None,
            "eta_sky": None,
        }
        pws = _mean(d["sky"])
        if pws is not None and s2_inc_sky is not None and s2_inc_sky > 0.0:
            fls = s2_inc_sky / n_rec
            rho_s = (pws - fls) / fls
            row["cn0_sky_db"] = (
                10.0 * math.log10(rho_s / t_coh) if rho_s > 0.0 else None
            )
            row["sig_sky"] = pws / fls
        if keep_series:
            row["series"] = d["series"]
        out[prn] = row
    return out
