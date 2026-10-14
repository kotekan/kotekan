"""Broker fits and pure decisions: rate fits, the code-clock bias, seed-phase currency, RF lobes
and the stall verdicts.

Functions of their arguments, with no globals and no I/O of their own (rate_residuals logs through
an optional callback; rate_residuals and split_erratic_offsets update caller-owned state passed
in), so each is testable in isolation.
"""
from gnss_broker import signals


def fit_cp_rate(hist, code_len):
    """Least-squares fit of cp0 against capture hop -> (rate chips/hop, hop_ref, cp_ref), or None.

    Each search cp0 is anchored to its capture hop and drifts roughly linearly with it (the
    residual code-rate error); the slope lets the tracker extrapolate to its own window hop and
    sit on the peak. cp0 is unwrapped to the nearest wrap (period code_len) along the sequence
    before fitting. Hops are referenced to the latest one, which keeps the large absolute index
    from costing precision, and the result is anchored there (cp_ref is mod code_len). Returns
    None with fewer than 3 points or no spread in hop.
    """
    if len(hist) < 3:
        return None
    hops = [h for h, _ in hist]
    cps = [c for _, c in hist]
    unw = [cps[0]]
    for c in cps[1:]:
        d = c - (unw[-1] % code_len)
        d -= code_len * round(d / code_len)  # nearest wrap
        unw.append(unw[-1] + d)
    h0 = hops[-1]
    dh = [float(h - h0) for h in hops]
    n = len(dh)
    sh, sc = sum(dh), sum(unw)
    shh = sum(x * x for x in dh)
    shc = sum(x * y for x, y in zip(dh, unw))
    den = n * shh - sh * sh
    if den == 0.0:
        return None
    rate = (n * shc - sh * sc) / den  # chips per hop
    cp_ref = (sc - rate * sh) / n  # fitted cp0 at h0
    return rate, h0, cp_ref % code_len


def fit_dop_rate(hist, hops_per_sec, min_pts, min_span_s, max_rate):
    """Least-squares slope of measured Doppler against time -> doppler_rate_hz_s, or None.

    The model's rate (BRDC range rate differenced over a 4 s epoch pair) is a numerical
    derivative, weakest near zenith, where the curvature peaks and the strong passes are. The
    detections' own Doppler track measures the slope directly.

    Returns None unless there are min_pts points spanning min_span_s, because a slope fitted to
    a short baseline fits detection noise: with per-point scatter sigma the slope error is
    ~sigma/(T*sqrt(N/12)), so at ~1.5 Hz scatter 4 points over 44 s give ~0.06 Hz/s and 8 over
    88 s ~0.02 Hz/s. Also None when |slope| exceeds max_rate.
    """
    if len(hist) < max(2, min_pts):
        return None
    t = [h / hops_per_sec for h, _ in hist]
    f = [d for _, d in hist]
    span = t[-1] - t[0]
    if span < min_span_s:
        return None
    n = len(t)
    mt, mf = sum(t) / n, sum(f) / n
    den = sum((x - mt) ** 2 for x in t)
    if den <= 0.0:
        return None
    rate = sum((x - mt) * (y - mf) for x, y in zip(t, f)) / den
    # max_rate is a physical bound, not a tuned one: GPS Doppler acceleration peaks near
    # 0.94 Hz/s at L1 (~0.70 Hz/s at L5, scaled by 1176.45/1575.42), so a larger fit is detection
    # noise and seeding it would add curvature error. Reject rather than clamp: the caller then
    # keeps the model's rate.
    return rate if abs(rate) <= max_rate else None


def code_clock_bias_sample(rate_chips_per_hop, hops_per_sec, chip_hz):
    """One satellite's estimate of the receiver LO-vs-ADC clock offset (l - a), dimensionless.

    The fitted slope is the drift of the search's sample-0-anchored cp0, a residual rate: the
    replica and the search already apply the geometric code Doppler (chip_per_sample scales with
    the seeded Doppler, and the search references cp back to sample 0 with the matching term).
    What remains in the slope is f_chip * (l - a), plus a small Doppler-quantization residue, so
        l - a  =  slope / f_chip
    with no carrier_frac subtraction: v/c never appears in a residual slope. Subtracting
    doppler/carrier is right only for a raw code drift measured without the feed-forward; applied
    to a residual slope it contaminates l - a by each satellite's carrier_frac.
    """
    return rate_chips_per_hop * hops_per_sec / chip_hz


def rate_residuals(
    status,
    min_q,
    clip_hz,
    log=None,
    prev_hop=None,
    max_gap=2,
    rec_hops=2048,
    prev_val=None,
    max_step=3.0,
    unit_hop=None,
    rate_field="deep_rate_hz",
    q_field="deep_rate_q",
):
    """Per-PRN carrier-rate residual (Hz) from the combiner's phase-rate search.

    Returns (residuals, consensus): residuals is {prn: residual_hz} for the PRNs that pass every
    gate and the clip, consensus their amp_snr-weighted mean (the median when the clip leaves
    none); ({}, None) when nothing passes the gates.

    A weak satellite does not merely scatter: the search lands on the wrong spectral bin and is
    off by tens of Hz. An average alone cannot survive that, and a hard gate alone throws away
    usable satellites, so both:

      1. A hard gate on the search's own peak/median (q_field >= min_q): signal reads ~18-22 and
         noise ~3-6, so the bar is a property of the measurement, not of SNR.
      2. A clip about the median across PRNs (clip_hz), then the amp_snr-weighted consensus.
         The median comes first because a wrong-bin outlier is arbitrarily far and would drag
         any mean; weighting then lets the strongest satellites dominate what survives.

    A PRN that passes keeps its own residual: the per-satellite differences are real (they are
    what a per-PRN trim exists for). A PRN that fails is left out; the consensus is returned for
    a caller that wants the common-mode value instead.

    With prev_hop (and optionally prev_val and unit_hop) the caller supplies per-stream
    continuity state, which this updates; see the continuity and slew gates below.
    """
    cand = {}
    gapped, stepped = [], []
    for prn, rec in (status or {}).items():
        # rate_field/q_field: deep_rate_hz is the fold's pick, capped at the combiner's
        # DEEP_RATE_MAX_HZ (25 Hz), and past the cap it reports a noise bin; the carrier loop
        # reads the uncapped deep_rate_full_hz. The gates are the same either way, but each
        # field set needs its own prev_hop/prev_val state.
        q = float(rec.get(q_field) or 0.0)
        f = rec.get(rate_field)
        if f is None or q < min_q:
            continue
        # Continuity gate. Across a gap in a PRN's sequence the tracker has re-anchored, so the
        # step is a reference change, not a measurement, and integrating it walks the trim to
        # the rail. Never transport a quantity across a discontinuity: after a gap, record the
        # hop and skip the sample (re-baseline) rather than guess.
        h = rec.get("pow_hop")
        if prev_hop is not None:
            if h is None or h < 0:
                continue
            ph = prev_hop.get(int(prn))
            prev_hop[int(prn)] = h
            if ph is None or h == ph:
                continue  # first sight, or the same window again
            # The gap is measured in emits, not records: successive observations are one
            # combiner emit apart (~2 s), a cadence the broker does not configure. So the unit
            # is derived from the stream: emits share one pow_hop across all PRNs, and unit_hop
            # holds the smallest step seen, so a PRN whose first pair straddles a gap
            # self-corrects once a contiguous pair arrives. Without unit_hop the unit is
            # rec_hops.
            step = h - ph
            if unit_hop:
                unit_hop[0] = min(unit_hop[0], step) if unit_hop[0] else step
            u = (unit_hop[0] if unit_hop and unit_hop[0] else float(rec_hops)) or 1.0
            if step > max_gap * u * 1.5:  # 1.5: jitter margin on the derived unit
                gapped.append(int(prn))
                continue  # re-anchored: re-baselined above, skip
            # Slew gate. The gap test catches only re-anchors that coincide with a dropped
            # window. The tracker's f_ref follows the seed, so a new seed with a different
            # Doppler steps it mid-track: the record assembler keeps the phase continuous across
            # that re-pin, but the frequency steps, with no gap to mark it. It is separable by
            # size: contiguous emits step under ~1 Hz, a re-anchor several Hz, and max_step sits
            # between.
            pf = prev_val.get(int(prn)) if prev_val is not None else None
            if prev_val is not None:
                prev_val[int(prn)] = float(f)
                if pf is not None and abs(float(f) - pf) > max_step:
                    stepped.append(int(prn))
                    continue  # f_ref re-pinned: re-baselined, do not integrate
        w = float(rec.get("amp_snr") or 0.0)
        if w > 0.0:
            cand[int(prn)] = (float(f), w)
    if log and gapped:
        log(
            "carrier-rate: %d PRN(s) skipped across a window gap (re-anchor, not a "
            "measurement): %s" % (len(gapped), sorted(gapped))
        )
    if log and stepped:
        log(
            "carrier-rate: %d PRN(s) skipped on a slew step >%.1f Hz (f_ref re-pin, not a "
            "measurement): %s" % (len(stepped), max_step, sorted(stepped))
        )
    if not cand:
        return (
            {},
            None,
        )  # (residuals, consensus): always a 2-tuple, the caller unpacks it
    vals = sorted(v[0] for v in cand.values())
    med = vals[len(vals) // 2]
    # Clip about the median, then weight. clip_hz <= 0 disables the clip.
    keep = {
        p: (f, w)
        for p, (f, w) in cand.items()
        if clip_hz <= 0.0 or abs(f - med) <= clip_hz
    }
    if keep:
        sw = sum(w for _, w in keep.values())
        consensus = sum(f * w for f, w in keep.values()) / sw if sw > 0 else med
    else:
        consensus = med
    out = {p: f for p, (f, _) in keep.items()}
    dropped = [p for p in cand if p not in keep]
    if log and dropped:
        log(
            "carrier-rate: %d PRN(s) clipped as outliers (>%.1f Hz from median %+.2f): %s"
            % (len(dropped), clip_hz, med, sorted(dropped))
        )
    return out, consensus


def cp_rate_from_code_bias(code_bias, hops_per_sec, chip_hz):
    """Seed the cp0 slope (chips/hop) for a satellite with no fit yet, from the calibrated (l - a).

    Residual convention, as in code_clock_bias_sample: cp0 is anchored at sample 0 and the
    replica applies the geometric code Doppler itself, so the seeded slope is the clock residual
    alone,
        cp0_rate = f_chip * (l - a)
    Adding doppler/f_carrier (the full code rate) would apply the code Doppler twice, and an
    unfitted satellite would slide off the +-1-chip peak at the code-Doppler rate (~30 chips/s
    at L5, in under a second)."""
    return code_bias * chip_hz / hops_per_sec


def adr_fine_rate(rec, prev, rec_dt, wall_dt=None):
    """PLL fine observable: the residual carrier rate from the ADR's residual half.

    res_cycles is the combiner's running sum of measured per-record residual phase increments
    (dres) on the current arc: memoryless per record and fold-proof at converged residuals
    (~3e-3 cycles per record against a +-0.25 window). Its difference over a poll span, divided
    by the span, is a carrier-rate measurement with precision (per-record phase noise)/span,
    ~5 mHz over 20 s, which the rate search's spectrum cannot reach.

    Gates, all structural: same arc (a break means unobserved whole cycles); the record counter
    advanced (a frozen combiner must read as absent, never as 0 Hz); both endpoints present;
    and, when the caller supplies wall_dt, the record-implied span within 25% of it.

    The wall_dt gate enforces accumulator identity. res_cycles and trim_cycles are private to one
    combiner process, each with its own arbitrary arc zero, so a difference is defined only
    between two snapshots of the same accumulator. Never difference across accumulators; to
    combine instances, self-difference each and fuse the rates, weighted by each accumulator's
    own noise. That holds however many subbands an instance carries. A restart, a
    re-configuration or a switch of serving instance between snapshots changes the identity, and
    the span/wall mismatch is how it shows; it invalidates the whole difference, not just the
    applied reference.

    Returns (rate_cycles_per_s, n_records, applied_hz), or None, in the combiner's internal
    (r2c-flipped) sign convention: the caller applies the calibrated sign, measured on sky
    against deep_rate_full_hz, never assumed.
    """
    if not isinstance(rec, dict) or not isinstance(prev, dict):
        return None
    if rec.get("adr_arc") != prev.get("adr_arc"):
        return None
    n1 = rec.get("adr_records") or 0
    n0 = prev.get("adr_records") or 0
    r1 = rec.get("res_cycles")
    r0 = prev.get("res_cycles")
    if r1 is None or r0 is None or n1 <= n0:
        return None
    span = (n1 - n0) * rec_dt
    if wall_dt is not None and (wall_dt <= 0.0 or abs(span - wall_dt) > 0.25 * wall_dt):
        return None
    # The measured applied command. trim_cycles integrates the commanded-trim increments the
    # tracker actually applied, on the same arc as res_cycles, so its difference over the span
    # is the applied carrier command over exactly the measured span. The broker's own record
    # (RateFeedState.cmd_applied) is only what it posted (posting is not applying, and lags a
    # poll), and a slowly wrong applied reference makes the rate integrator marginal. None when
    # a record lacks the field; the caller then uses the posted command's span mean.
    t1 = rec.get("trim_cycles")
    t0v = prev.get("trim_cycles")
    applied = ((t1 - t0v) / span) if (t1 is not None and t0v is not None) else None
    return (r1 - r0) / span, n1 - n0, applied


# --- dead-reckon seed currency ----------------------------------------------------------
#
# The forward and inverse of one mapping, kept adjacent so they cannot drift apart. They mirror
# gnss::propagate_seed's phase model (gnssSeedTransport.cpp): the sample-0 back-reference scales
# the nominal advance by the code Doppler over the whole time since sample 0, cp_rate is the
# residual clock slope in chips per hop (see cp_rate_from_code_bias above), and the quadratic
# term is 0.5*(f_chip/f_carrier)*dop_rate*dt^2 from the seed's reference hop.


def dr_cp0(phys_chips, t_abs, doppler_hz, chip_hz, carrier_hz, code_doppler_sign, mod):
    """Physical code phase (incl. receiver clock) at t_abs -> the sample-0 seed currency.

    The dead-reckon seeder's birth formula: subtract the Doppler-scaled nominal advance since
    sample 0. `t_abs` is seconds since sample 0 and must be the same instant the seed's ref_hop
    encodes (ref_hop = round(t_abs * hps)); the sub-hop rounding cancels because the physical
    phase and the back-reference advance at the same rate to first order."""
    return (
        phys_chips
        - t_abs * chip_hz * (1.0 + code_doppler_sign * doppler_hz / carrier_hz)
    ) % mod


def dr_seed_phys(seed, h1, hops_per_sec, chip_hz, carrier_hz, code_doppler_sign, mod):
    """Physical code phase (chips, mod `mod`) the tracker's propagation implies at hop h1.

    The exact inverse of dr_cp0 plus the tracker's own extrapolation terms: the back-reference
    undone at the seed's Doppler (the mapping is defined by the Doppler the seed carries, not the
    one the sky has), the residual clock slope, and the dop_rate quadratic. It models the cp0
    argument only; for a seed carrying code_phase_at_ref_chips, which propagate_seed prefers,
    tracker_phase_at is the model. Used by the slewed refresh to measure how far the live model
    has walked from what a held seed is despreading at."""
    t1 = h1 / hops_per_sec
    dt = (h1 - seed["ref_hop"]) / hops_per_sec
    return (
        seed["code_phase_chips"]
        + t1 * chip_hz * (1.0 + code_doppler_sign * seed["doppler_hz"] / carrier_hz)
        + seed.get("code_phase_rate", 0.0) * (h1 - seed["ref_hop"])
        + 0.5 * (chip_hz / carrier_hz) * seed.get("doppler_rate_hz_s", 0.0) * dt * dt
    ) % mod


def retag_seed_doppler(
    cp_chips, old_dop, new_dop, t_eval_s, chip_hz, carrier_hz, code_doppler_sign, mod
):
    """Re-express a sample-0 cp in a new Doppler's currency, preserving the physical
    phase at t_eval_s.

    The only correct t_eval_s is now: the tracker rebuilds the phase as
    cp0 + t*f_chip*(1 + k*dop), with t elapsed since sample 0 and k = code_doppler_sign/f_carrier,
    so changing dop by ddop moves the physical phase at time t by t*f_chip*k*ddop, and the
    translation must cancel that where the despread is running. Translating at the seed's anchor
    epoch instead preserves the phase at the anchor and steps it now by anchor_age*f_chip*k*ddop
    per forecast update: over a coast at Doppler rate r, ~(f_chip/f_carrier)*r*age^2/2 chips of
    silent walk-off. Kept beside dr_cp0 and dr_seed_phys so the three transport directions cannot
    drift apart.
    """
    return (
        cp_chips
        - t_eval_s * chip_hz * code_doppler_sign * (new_dop - old_dop) / carrier_hz
    ) % mod


def seed_phase_at_ref(
    phys_chips,
    doppler_hz,
    chip_hz,
    hops_per_sec,
    carrier_hz,
    code_doppler_sign,
    mod,
    fft_len=None,
):
    """Broker-held physical phase -> `code_phase_at_ref_chips`, the field the tracker
    prefers over the sample-0 argument.

    The conversion is required. The broker's dr_cp0/dr_seed_phys pair references a hop at its
    first sample; ChannelizedReplicaBank, and therefore phase_from_arg, propagate_seed and every
    replica, references it at the last (window_start + fft_len - 1). That is one hop apart,
    ~52.4 chips at 10.23 Mcps, and shipping the broker's phase unconverted steps the commanded
    code by exactly that. test_track_vs_fit.py checks the conversion against the argument path's
    reconstruction.

    The phase is shipped because cp0 back-references through t_abs (days), so the pair
    (cp0, dop) means something only together: a producer that updates one without re-projecting
    the other injects t_abs*f_chip/f_carrier chips per Hz (~1700 at L5 after 2.3 days). A phase
    at its own epoch has no partner to fall out of step with.

    `fft_len` supplies the exact "-1 sample" term; omitted, the offset is one whole hop and the
    error is one sample (0.0064 chips at 10.23 Mcps), a constant far below the DLL's pull-in
    that lands in the clock rather than in tracking.
    """
    per_hop = (
        chip_hz / hops_per_sec * (1.0 + code_doppler_sign * doppler_hz / carrier_hz)
    )
    off = per_hop * (1.0 - 1.0 / fft_len) if fft_len else per_hop
    return (phys_chips + off) % mod


def tracker_phase_at(
    seed, h1, hops_per_sec, chip_hz, carrier_hz, code_doppler_sign, mod, fft_len=None
):
    """The phase the tracker will command at hop h1 for this seed, mirroring
    gnss::propagate_seed, including which code reference it uses.

    propagate_seed prefers `code_phase_at_ref_chips` over `code_phase_chips` whenever the former
    is >= 0, and the broker computes the two on different paths (cp0 from cp_long, the phase
    from the search's cp_at_ref), so they are not interchangeable. An audit that models cp0
    while the tracker reads the phase measures a stream nobody consumes.

    Returns the phase in the C++ last-sample convention (see seed_phase_at_ref), so it is
    directly comparable between consecutive seeds, which is what the seed audit and the
    innovation need.
    """
    per_hop = (
        chip_hz
        / hops_per_sec
        * (1.0 + code_doppler_sign * seed["doppler_hz"] / carrier_hz)
    )
    hop_off = per_hop * (1.0 - 1.0 / fft_len) if fft_len else per_hop
    ph_ref = seed.get("code_phase_at_ref_chips", -1.0)
    if ph_ref is None or ph_ref < 0.0:
        # the argument branch: undo the sample-0 back-reference, then move to the
        # tracker's reference point
        ph_ref = (
            dr_seed_phys(
                {k: v for k, v in seed.items() if k != "code_phase_at_ref_chips"},
                seed["ref_hop"],
                hops_per_sec,
                chip_hz,
                carrier_hz,
                code_doppler_sign,
                mod,
            )
            + hop_off
        )
    dh = h1 - seed["ref_hop"]
    dt = dh / hops_per_sec
    return (
        ph_ref
        + (per_hop + seed.get("code_phase_rate", 0.0) or 0.0) * dh
        + 0.5
        * (chip_hz / carrier_hz)
        * (seed.get("doppler_rate_hz_s", 0.0) or 0.0)
        * dt
        * dt
    ) % mod


def rf_lobes(chans, power, clip_lo, clip_hi, freq_ids=None, carriers=None):
    """Group one tap's per-channel RF stats into bands. Pure; see test_rf_lobes.py.

    The tap serves a flat channel list, the union of every chain's covering set on that GPU, so
    it arrives as contiguous runs. Contiguity alone does not identify bands: instances differ in
    which bands they hold (only some own an L2C channel), and abutting bands form one run
    (bds_b3i at 1268.52 MHz, whose BPSK(10) mainlobe edge lands on gal_e6's 1278.75 carrier).
    So `band` is the identity: it comes from the absolute freq_id the tap serves alongside
    `chans` (GnssChordVoltageTap reads it from the frame's chord metadata, never from config),
    and means the same on every instance whatever local subset that node holds. `lobe` is only
    an index within this tap's list and is not comparable across instances.

    Runs split on a channel gap or a band change. Without freq_ids (a node that does not serve
    them, or a tap that has not yet measured a chord-metadata frame) or without `carriers`, rows
    carry no band name and the grouping is contiguity alone: an unnamed row, never a wrongly
    named one.

    Power is meaned and clip is maxed: power is a level, so the band average is the summary,
    while clip is damage, and one railing channel averaged across a dozen quiet neighbours
    would hide exactly the narrowband case this exists to catch.
    """
    n = min(len(chans), len(power), len(clip_lo), len(clip_hi))
    if n == 0:
        return []
    # freq_ids is optional and must be exactly parallel to chans to be usable: a misaligned band
    # name is worse than none, so a short or ragged array falls back to unnamed.
    fids = list(freq_ids) if freq_ids else []
    if len(fids) != len(chans):
        fids = []
    # `carriers` is the declared band set (broker --rf-bands). The full RF_BAND_BY_CARRIER table
    # would name bands CHORD does not fly (E5b's lower shoulder reads as GLONASS L3), so with no
    # declaration nothing is named.
    named = [
        signals.band_of_freq_id(fids[i], carriers)
        if (fids and carriers)
        else (None, None)
        for i in range(n)
    ]
    band = [b for b, _ in named]
    carrier = [c for _, c in named]

    order = sorted(range(n), key=lambda i: chans[i])
    out, run = [], []

    def flush(run):
        if not run:
            return
        lo = max(range(len(run)), key=lambda k: clip_lo[run[k]])
        hi = max(range(len(run)), key=lambda k: clip_hi[run[k]])
        rec = {
            "lobe": len(out),
            "chan0": chans[run[0]],
            "chan1": chans[run[-1]],
            "n_chan": len(run),
            "power": sum(power[i] for i in run) / float(len(run)),
            "power_max": max(power[i] for i in run),
            "clip_lo": clip_lo[run[lo]],
            "clip_lo_chan": chans[run[lo]],
            "clip_hi": clip_hi[run[hi]],
            "clip_hi_chan": chans[run[hi]],
        }
        if fids:
            rec["band"] = band[run[0]]
            rec["carrier_mhz"] = carrier[run[0]]
            rec["freq_id0"] = fids[run[0]]
            rec["freq_id1"] = fids[run[-1]]
            rec["mhz0"] = round(fids[run[0]] * signals.FREQ_ID_MHZ, 3)
            rec["mhz1"] = round(fids[run[-1]] * signals.FREQ_ID_MHZ, 3)
        out.append(rec)

    for i in order:
        # Split on a channel gap or a band change. The band test is what separates the
        # abutting bds_b3i and gal_e6 runs, which no gap ever will.
        if run and (chans[i] != chans[run[-1]] + 1 or band[i] != band[run[-1]]):
            flush(run)
            run = []
        run.append(i)
    flush(run)
    return out


def instance_stall_verdict(prev, cur, now, min_stall_s, min_frac_advancing=0.5):
    """Which instances are serving but not advancing. Pure; see test_instance_stall.py.

    `prev` is {url: (hop, first_seen_at_this_hop)} carried across cycles, `cur` is {url: hop}
    from this cycle. Returns (new_prev, stalled) where `stalled` is a list of
    (url, hop, seconds_stuck) for instances whose newest hop has not moved in >= min_stall_s
    while the fleet around them advanced.

    fe_axis_stale_s cannot see this: it watches the hop of the one --combiner instance, which
    answers "has the time base frozen?" but not whether any other instance has wedged. This
    check is per instance.

    It keys on the hop counter, never on reachability, because a wedged instance still answers:
    with its capture window frozen and its whole stream dropped, it serves plausible,
    well-formed rows. An instance missing from `cur` is unreachable, a different fault already
    visible as n_src falling; it is dropped from the state here, never accused.

    `min_frac_advancing` is the control clause: if most of the fleet is also standing still,
    the cause is global (a paused F-engine, a clock step, a replay), not a per-instance stall,
    and accusing an instance would point the diagnosis the wrong way. Returns no accusations
    in that case.
    """
    new = {}
    for url, hop in cur.items():
        old = prev.get(url)
        new[url] = (hop, old[1] if (old is not None and old[0] == hop) else now)
    if not new:
        return new, []
    advancing = sum(1 for url, (hop, t0) in new.items() if t0 >= now)
    if advancing < max(1, int(round(min_frac_advancing * len(new)))):
        return new, []  # fleet-wide, not per-instance -- say nothing
    stalled = []
    for url, (hop, t0) in sorted(new.items()):
        stuck = now - t0
        if stuck >= min_stall_s:
            stalled.append((url, hop, stuck))
    return new, stalled


def q_stall_verdict(hist, now, window_s, frac, min_best, best, min_samples=10):
    """The q stall verdict: has this chain been quietly degraded?

    `hist` is [(t, duty)] appended once per broker cycle, `best` the highest windowed median
    duty seen so far this session (None before the first). Returns (new_best, verdict_or_None);
    the verdict is (current, best, current/best).

    A pure function so a constructed collapse can test it (test_q_stall.py): the on-sky replay
    fixtures are too short for the duty to fall, so a replay cannot tell "did not fire" from
    "cannot fire".

    The baseline is the chain's own best, never a fleet-common bar, because chains differ ~4x
    in duty by construction. It only rises, so a degrading chain cannot redefine normal
    downward. min_best exempts a chain with no headroom to fall from (bds_b2b sits near 0.2 for
    structural nav-bit reasons).
    """
    cut = now - window_s
    recent = [d for t, d in hist if t >= cut]
    if len(recent) < min_samples:
        return best, None
    cur = sorted(recent)[len(recent) // 2]
    if best is None or cur > best:
        return cur, None
    if best >= min_best and cur < best * frac:
        return best, (cur, best, (cur / best) if best else 0.0)
    return best, None


def split_erratic_offsets(offs, hist, now_w, bound_chips, max_age_s, code_len):
    """Split clock-solve offsets into (keep, drop) on per-satellite continuity of the raw d_i.

    `offs` is [(prn, d_i)], d_i = clk + b_i: both are stable, so a real satellite's d_i moves a
    few chips between cycles at most. A satellite whose d_i moved more than bound_chips (wrapped
    to half a code period) since its value at most max_age_s ago is dropped.

    Test the raw d_i; do not subtract a Doppler lever (t_i*chip_rate*sign*dop/carrier). The
    search's published cp0 embeds that term with the same Doppler and ref_hop, and cp_loc adds
    it back (gnssSeedTransport.cpp detection_phase), so raw d_i never carries it. Subtracting it
    reintroduces ~1700 chips per Hz of Doppler re-estimate, every satellite trips the guard, and
    the caller's min-sats floor keeps them all. The call site logs WHAT MOVED (d cp_loc, dt_i,
    ddop) for each dropped satellite.

    `hist` is {prn: (t, d_i)} from the previous cycle and is mutated here: every satellite's
    current value is recorded, dropped ones included, so a track that settles can rejoin.

    A satellite with no fresh history is always kept: the test needs two cycles, and refusing
    first sightings would stall the bootstrap.
    """
    keep, drop = [], []
    for prn, d in offs:
        prev = hist.get(prn)
        hist[prn] = (now_w, d)
        if prev is not None and now_w - prev[0] <= max_age_s:
            delta = d - prev[1]
            jump = abs(((delta + code_len / 2.0) % code_len) - code_len / 2.0)
            if jump > bound_chips:
                drop.append((prn, jump))
                continue
        keep.append((prn, d))
    return keep, drop
