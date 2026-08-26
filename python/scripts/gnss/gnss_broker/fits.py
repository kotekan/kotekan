"""Broker fits: code-phase and Doppler rate estimation, and the code-clock bias.

Extracted verbatim from gps_distributed_broker.py (task #27 M1). Pure functions of their
arguments -- no globals, no I/O -- so they are testable in isolation for the first time.
"""
import math
import statistics


def fit_cp_rate(hist, code_len):
    """Least-squares fit cp0 vs capture hop -> (rate chips/hop, hop_ref, cp_ref).

    The search anchors each cp0 to its snapshot hop (capture time, shared with the
    tracker via fpga_seq), but cp0 drifts ~linearly with that hop (residual code-rate
    error). Fitting the slope lets the tracker extrapolate to its own window hop and
    sit on the peak -- a first-order code model that removes the seed-staleness bias.
    Returns None if there isn't enough spread to fit. cp0 is unwrapped (period
    code_len) along the sequence before fitting; the result is anchored at the latest
    hop. hop is centred to keep the (large) absolute index from losing precision.
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
    rate = (n * shc - sh * sc) / den          # chips per hop
    cp_ref = (sc - rate * sh) / n             # fitted cp0 at h0
    return rate, h0, cp_ref % code_len


def fit_dop_rate(hist, hops_per_sec, min_pts, min_span_s, max_rate):
    """Least-squares slope of MEASURED Doppler vs time -> doppler_rate_hz_s, or None.

    Replaces BRDC's range-rate differencing over a 4 s epoch pair, which is a numerical
    derivative evaluated where the curvature is largest (near zenith, exactly where the strong
    passes are). Measured 2026-08-04: BRDC gave PRN 3 -0.4699 Hz/s against -0.578 from the
    Doppler track, and that 0.108 Hz/s error is ~0.7 rad of phase curvature inside the 1.05 s
    deep window -- part of the 2.55 rad that costs the coherent sum 29 dB.

    Returns None unless there are min_pts points spanning min_span_s, because a slope fitted to
    a short baseline is fitted to detection noise: with ~1.5 Hz per-point scatter the slope
    error is ~sigma/(T*sqrt(N/12)), so 4 points over 44 s give ~0.06 Hz/s (already better than
    BRDC) and 8 over 88 s give ~0.02 Hz/s. Below that the model's rate is the better bet.
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
    # PHYSICAL BOUND, not a tuned one: GPS Doppler acceleration peaks at ~0.94 Hz/s at L1, so
    # ~0.70 Hz/s at L5 (scaled by 1176.45/1575.42). A fit beyond that is detection noise, not
    # sky -- observed immediately on deploy, PRN 20 fitted at -1.16 Hz/s -- and seeding it would
    # ADD curvature error rather than remove it. Reject rather than clamp: a rejected fit falls
    # back to the almanac's rate, which is at least physical.
    return rate if abs(rate) <= max_rate else None


def code_clock_bias_sample(rate_chips_per_hop, doppler_hz, hops_per_sec, chip_hz, carrier_hz):
    """One satellite's estimate of the receiver LO-vs-ADC clock offset (l - a), dimensionless.

    CONVENTION (the 2026-07-04 L5 finding): the fitted slope here is the drift of the SEARCH's
    absolute-anchored cp0, which is a RESIDUAL rate -- the replica/search already apply the
    geometric code Doppler internally (chip_per_sample scales with the seeded Doppler; the
    search references cp back to sample 0 with the matching term). What remains in the cp0
    slope is f_chip * (l - a) (+ the small Doppler-quantization residue), so
        l - a  =  slope / f_chip
    with NO carrier_frac subtraction: v/c never appears in a residual slope. (The original
    formula subtracted doppler/carrier, valid for the OFFLINE raw-drift tools like
    l1_code_drift.py where the code drift is measured without a feed-forward -- but applied to
    residual-convention slopes it contaminated l-a by the per-sat carrier_frac: the estimates
    disagreed band-to-band (+0.25 ppm L1 / -0.63 ppm L5) where the residual reading agrees
    (+0.03 / -0.04 ppm, both near the GPSDO's measured +0.06).)
    doppler_hz/carrier_hz are kept in the signature for call-site stability; unused.
    """
    del doppler_hz, carrier_hz  # residual convention: geometry is already fed forward
    return rate_chips_per_hop * hops_per_sec / chip_hz


def rate_residuals(status, min_q, clip_hz, log=None, prev_hop=None, max_gap=2, fft_len=16384,
                   rec_hops=2048, prev_val=None, max_step=3.0, unit_hop=None,
                   rate_field="deep_rate_hz", q_field="deep_rate_q"):
    """Per-PRN carrier residual (Hz) from the combiner's phase-rate search.

    TWO FAILURE MODES, TWO DEFENCES. Measured on sky 2026-08-04 by splitting each PRN's records
    into independent halves and fitting each:

        amp_snr 83.7 -> halves agree to 0.000 Hz      amp_snr 9.3 -> 0.093 Hz
        amp_snr 44.2 -> 0.186                         amp_snr 9.5 -> 41.7 Hz  <-- WRONG BIN
        amp_snr 43.1 -> 0.466

    So a strong sat is pinned to ~0.2 Hz (against the <1 Hz a 1.05 s window needs), while a weak
    one does not merely scatter -- it lands on the wrong spectral bin and is off by tens of Hz.
    An average alone cannot survive that, and a hard gate alone throws away sats that are
    perfectly usable. Hence both:

      1. HARD GATE on the search's own peak/median (deep_rate_q): 17.9-22.0 on signal vs 2.8-6.1
         on noise, so min_q sits in the gap and is a property of the measurement, not of SNR.
      2. ROBUST WEIGHTED CONSENSUS across PRNs, weight = amp_snr, after clipping about the
         MEDIAN. The median comes first precisely because a wrong-bin outlier is arbitrarily far
         and would drag any mean; weighting then lets the 83-sigma sat dominate the 9-sigma ones
         inside the surviving set, which is the right thing when precision tracks SNR.

    A PRN that passes the gate keeps its OWN residual -- the per-satellite differences are real
    (they are what a per-PRN trim exists for). One that fails INHERITS the consensus, which is
    far better than either a wrong bin or no correction at all, because the dominant term is
    common-mode (every node reports the same rate for a given PRN to within one search bin).

    Returns {prn: residual_hz}; empty when nothing clears the gate.
    """
    cand = {}
    gapped, stepped = [], []
    for prn, rec in (status or {}).items():
        # rate_field/q_field (#40): the FOLD's pick (deep_rate_hz) is clamped to
        # deep_rate_max_hz and degrades to a noise bin past the cap; the carrier loop must
        # read deep_rate_full_hz instead. Same gates either way -- a caller choosing the
        # full fields brings its OWN prev_hop/prev_val state, never shares this one.
        q = float(rec.get(q_field) or 0.0)
        f = rec.get(rate_field)
        if f is None or q < min_q:
            continue
        # CONTINUITY GATE (2026-08-04). Measured per emit window: when the sequence is
        # CONTIGUOUS the rate walks -0.24..-0.72 Hz per emit (~-0.36 Hz/s), smooth and
        # trackable. Every jump of tens of Hz lands on a GAP, where the PRN was dropped and
        # the tracker re-anchored -- so the step across a gap is a reference change, not a
        # measurement, and integrating it walks the trim to the rail. Same rule the nh anchor
        # and code_phase_chips already taught us: never transport a quantity across a
        # discontinuity. After a gap we RE-BASELINE (record the hop, skip this sample) rather
        # than guess.
        h = rec.get("pow_hop")
        if prev_hop is not None:
            if h is None or h < 0:
                continue
            ph = prev_hop.get(int(prn))
            prev_hop[int(prn)] = h
            if ph is None or h == ph:
                continue                       # first sight, or the same window again
            # SPACING IS ONE EMIT, NOT ONE RECORD. This compared (h - ph) against
            # max_gap * rec_hops = 4096 hops, i.e. two RECORDS -- but successive observations
            # are one EMIT apart, measured at 389120 hops = 190 records = 1.99 s. So the test
            # overshot by ~95x and gated out EVERY sample: 118 log lines, zero measurements
            # admitted, the carrier loop starved while appearing to run. Derive the unit from
            # the stream instead of naming it: emits share one pow_hop across all PRNs, and
            # the cadence is a property of the combiner's window, not of anything the broker
            # knows. `unit` is the running minimum positive step, so a PRN whose first sighting
            # straddles a gap self-corrects as soon as one contiguous pair arrives.
            step = h - ph
            if unit_hop:
                unit_hop[0] = min(unit_hop[0], step) if unit_hop[0] else step
            u = (unit_hop[0] if unit_hop and unit_hop[0] else float(rec_hops)) or 1.0
            if step > max_gap * u * 1.5:       # 1.5: jitter margin on the derived unit
                gapped.append(int(prn))
                continue                       # re-anchored: re-baselined above, skip
            # SLEW GATE. The hop gap above only catches re-anchors that COINCIDE with a dropped
            # window, and the tracker's f_ref fence does not: it fires mid-tracking whenever
            # |f_ref - dop| exceeds fll_reacq_hz, adopting the new seed wholesale. The
            # phase-continuity fold keeps Phi_cmd smooth across that, but the FREQUENCY steps --
            # so a fence re-anchor is a reference change with no gap to mark it.
            #
            # It is separable by size. Measured per emit: contiguous tracking steps -0.238 to
            # -0.715 Hz (the residual drifting at the Doppler rate against a FIXED f_ref), while
            # the smallest observed re-anchor jump was 7.6 Hz. max_step sits in that gap.
            pf = prev_val.get(int(prn)) if prev_val is not None else None
            if prev_val is not None:
                prev_val[int(prn)] = float(f)
                if pf is not None and abs(float(f) - pf) > max_step:
                    stepped.append(int(prn))
                    continue                   # f_ref re-pinned: re-baselined, do not integrate
        w = float(rec.get("amp_snr") or 0.0)
        if w > 0.0:
            cand[int(prn)] = (float(f), w)
    if log and gapped:
        log("carrier-rate: %d PRN(s) skipped across a window gap (re-anchor, not a "
            "measurement): %s" % (len(gapped), sorted(gapped)))
    if log and stepped:
        log("carrier-rate: %d PRN(s) skipped on a slew step >%.1f Hz (f_ref re-pin, not a "
            "measurement): %s" % (len(stepped), max_step, sorted(stepped)))
    if not cand:
        return {}, None   # (residuals, consensus) -- ALWAYS a 2-tuple; a bare {} here killed
                          # the broker the first time every PRN was gated out at once
    vals = sorted(v[0] for v in cand.values())
    med = vals[len(vals) // 2]
    # Clip about the median, then weight. clip_hz <= 0 disables the clip (keep everything gated).
    keep = {p: (f, w) for p, (f, w) in cand.items()
            if clip_hz <= 0.0 or abs(f - med) <= clip_hz}
    if keep:
        sw = sum(w for _, w in keep.values())
        consensus = sum(f * w for f, w in keep.values()) / sw if sw > 0 else med
    else:
        consensus = med
    out = {p: f for p, (f, _) in keep.items()}
    dropped = [p for p in cand if p not in keep]
    if log and dropped:
        log("carrier-rate: %d PRN(s) clipped as outliers (>%.1f Hz from median %+.2f): %s"
            % (len(dropped), clip_hz, med, sorted(dropped)))
    return out, consensus


def cp_rate_from_code_bias(doppler_hz, code_bias, hops_per_sec, chip_hz, carrier_hz):
    """Seed the cp0 slope (chips/hop) for a not-yet-fittable sat from the calibrated (l - a).

    RESIDUAL convention (2026-07-04 L5 finding): cp0 is absolute-anchored and the replica applies
    the geometric code Doppler itself, so the correct seeded slope is ONLY the clock residual
        cp0_rate = f_chip * (l - a)
    The original formula added doppler/f_carrier -- the FULL physics code rate -- which the
    replica then applied AGAIN: unfitted sats slid off-peak at the code-Doppler rate. Fatal at
    L5 (+-30 chips/s -> off the +-1 chip peak in <1 s; the replay smoking gun: seeded cp 6690,
    displayed 6477 after 65 s = -3.28 chips/s = exactly dop/f * f_chip), historically masked at
    L1 (+-3 chips/s) by fast fits + the old per-record pull-in re-centering.
    doppler_hz/carrier_hz kept in the signature for call-site stability; unused."""
    del doppler_hz, carrier_hz  # residual convention: geometry is already fed forward
    return code_bias * chip_hz / hops_per_sec


def adr_fine_rate(rec, prev, rec_dt, wall_dt=None):
    """#33 PLL fine observable: the residual carrier rate from the ADR's residual half.

    res_cycles is the combiner's running sum of MEASURED per-record residual phase
    increments (dres) on the current arc -- memoryless per record and fold-proof at
    converged residuals (~3e-3 cyc/record against a +-0.25 window). Its difference over a
    poll span, divided by the record count, is a carrier-rate measurement whose precision
    is (per-record phase noise)/span: ~5 mHz over 20 s -- the budget class the rate
    search's spectrum cannot reach.

    GATES, all structural: same arc (a break means unobserved whole cycles -- no
    measurement); the record counter advanced (a frozen combiner must read as absent,
    never as 0 Hz); both endpoints present; and, when the caller supplies wall_dt, the
    record-implied span must agree with the wall-clock span.

    THE ACCUMULATOR-IDENTITY RULE (wall_dt's real job, restated 2026-08-24): res_cycles
    and trim_cycles are accumulators private to ONE combiner process, each with its own
    arbitrary arc zero -- so a difference is only defined between two snapshots of the
    SAME accumulator. Never difference across accumulators; when the observable must
    combine instances, self-difference each accumulator and FUSE the resulting rates
    (they measure one physical quantity; the fusion weight is each accumulator's own
    noise). That rule is instance-count-agnostic: it holds unchanged when an instance
    carries 16 subbands or one. The wall_dt gate is the tripwire for a changed
    accumulator identity slipping between snapshots (a restart, a re-config, or --
    historically -- a serving path that picked best-of-instance per poll and so churned
    the identity; measured 2026-08-14 on a held +5.00 Hz trim: same-accumulator pairs
    read 5.0000 exactly, cross-accumulator pairs 0.23-16.7 Hz. The current status path
    polls one configured combiner, so churn is gone, but the gate stays -- it is what
    makes the pairing an invariant rather than a deployment accident). A span/wall
    mismatch invalidates the whole difference, not just the applied reference. Returns
    (rate_cycles_per_s, n_records, applied_hz) in the
    combiner's INTERNAL (r2c-flipped) sign convention -- the caller applies the calibrated
    sign, which is measured on sky against deep_rate_full_hz, never assumed.
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
    # THE MEASURED APPLIED COMMAND (2026-08-14 02:xx, the last assumption removed).
    # trim_cycles integrates the commanded-trim increments the tracker ACTUALLY applied,
    # on the same arc as res_cycles -- so d(trim)/d(span) is the applied carrier command
    # over EXACTLY the measured span, from the same data stream. The broker's own
    # rr_cmd_applied is an assumption (posted != applied, one-poll-lag class), and a
    # slowly wrong applied-reference is a marginal integrator: arm 8's fleet-common
    # +0.3-0.7 Hz/min ramp. None when the field is absent (old binary): the caller then
    # falls back to its assumption, loudly.
    t1 = rec.get("trim_cycles")
    t0v = prev.get("trim_cycles")
    applied = ((t1 - t0v) / span) if (t1 is not None and t0v is not None) else None
    return (r1 - r0) / span, n1 - n0, applied


# (An `unalias` helper lived here for ~40 minutes on 2026-08-13: it unwrapped the rrate
#  feed's measurements toward the filter's own prediction, on the theory that deep_rate_hz
#  wrapped mod 9.537 Hz. The theory was WRONG -- built from 1-3 noisy samples per sat --
#  and the construction was SELF-REFERENTIAL: a walking row dragged its unwrap anchor with
#  it and commands ran away past +-10 Hz. The real mechanism was the deep_rate_max_hz CAP
#  turning out-of-window residuals into in-cap noise picks; the fix is the combiner's
#  deep_rate_full_* fields (#40), consumed via rate_field/q_field above.)

# --- dead-reckon seed currency (task #30) -----------------------------------------------
#
# The two directions of the SAME mapping, kept adjacent so they cannot drift apart. The
# broker's dead-reckon seeder built cp0 inline (the dr_cp0 formula below) while nothing ever
# computed the inverse -- so when #30 needed "where does the TRACKER think the code is right
# now", there was no function to ask. These mirror gnss::propagate_seed's phase model
# line-for-line (gnssSeedTransport.cpp): the sample-0 back-reference scales the nominal
# advance by the code Doppler over the WHOLE time since sample 0, cp_rate is the residual
# clock slope in chips/HOP (see cp_rate_from_code_bias above), and the quadratic term is
# 0.5*(f_chip/f_carrier)*dop_rate*dt^2 from the seed's reference hop.

def dr_cp0(phys_chips, t_abs, doppler_hz, chip_hz, carrier_hz, code_doppler_sign, mod):
    """Physical code phase (incl. receiver clock) at t_abs -> the sample-0 seed currency.

    This IS the dead-reckon seeder's birth formula, extracted verbatim: subtract the
    Doppler-scaled nominal advance since sample 0. `t_abs` is seconds since sample 0 and
    must be the same instant the seed's ref_hop encodes (ref_hop = round(t_abs * hps));
    the sub-hop rounding cancels because the physical phase and the back-reference advance
    at the same rate to first order."""
    return (phys_chips
            - t_abs * chip_hz * (1.0 + code_doppler_sign * doppler_hz / carrier_hz)) % mod


def dr_seed_phys(seed, h1, hops_per_sec, chip_hz, carrier_hz, code_doppler_sign, mod):
    """Physical code phase (chips, mod `mod`) the TRACKER's propagation implies at hop h1.

    The exact inverse of dr_cp0 plus the tracker's own extrapolation terms, i.e. what
    gnss::propagate_seed will hand the despread at h1 for this seed: the back-reference
    undone at the SEED's Doppler (the mapping is defined by the doppler the seed carries,
    not the doppler the sky has -- the cp-currency rule), the residual clock slope, and the
    dop_rate quadratic. Used by the slewed refresh to measure how far the live model has
    walked away from what a held seed is actually despreading at."""
    t1 = h1 / hops_per_sec
    dt = (h1 - seed["ref_hop"]) / hops_per_sec
    return (seed["code_phase_chips"]
            + t1 * chip_hz * (1.0 + code_doppler_sign * seed["doppler_hz"] / carrier_hz)
            + seed.get("code_phase_rate", 0.0) * (h1 - seed["ref_hop"])
            + 0.5 * (chip_hz / carrier_hz)
              * seed.get("doppler_rate_hz_s", 0.0) * dt * dt) % mod


def track_vs_fit_chips(held_seed, det_cp_loc, det_ref_hop, dll_trim_chips,
                       hops_per_sec, chip_hz, carrier_hz, code_doppler_sign, code_len):
    """Track-vs-search residual, both sides physical, at the DETECTION's epoch
    (#42 / #45 step 1).

    Replaces the sample-0 currency comparison cp_err used until 2026-08-12. That formula
    translated between two SEED-dop currencies with the full-run-age lever (~1700 chips/Hz
    at 2.3 days), so any motion of the seed-vs-detection dop bias -- the carrier clock_bias
    EMA, a quantity DESIGNED to move -- read as -1700*dBias chips of track error: 145
    specimens in 7 minutes, seven false ESCAPES in one evening against tracks healthy at
    40 dB-Hz.

    Both inputs here are physical code phases at det_ref_hop, in one convention:
      * det_cp_loc: the detection's own (cp0, dop) pair undone -- cp0 + t*f_chip*(1+k*dop),
        the SAME pair the search published together, so no translation happens and a
        pair-inconsistent candidate cannot inject anything. Measured continuity on sky:
        0.27 chips median over 1423 consecutive detections.
      * dr_seed_phys(held): where the tracker's propagation puts the despread, from the
        held tuple's own labels used self-consistently.

    An earlier draft used the payload's cp_at_ref instead. It is better conditioned in
    principle, but it is referenced at the hop's LAST sample and carries the replica
    anchor's Doppler term (+52.3711 + 1.39e-4*dop chips vs this quantity, measured on
    26,815 banked detections) -- so consuming it means importing the search's fft_len AND
    anchor geometry into every consumer, which is the coupling this pass exists to remove,
    for a conditioning gain the sky says is not needed. The C++ side keeps cp_at_ref in its
    own last-sample convention, which is a contract there (gnssSeedTransport).

    Returns the wrapped residual (fresh - held - trim), or None if det_cp_loc is missing.
    """
    if det_cp_loc is None or det_cp_loc < 0.0:
        return None
    held = dr_seed_phys(held_seed, det_ref_hop, hops_per_sec, chip_hz, carrier_hz,
                        code_doppler_sign, code_len)
    return ((det_cp_loc - held - dll_trim_chips + code_len / 2.0) % code_len
            ) - code_len / 2.0


def retag_seed_doppler(cp_chips, old_dop, new_dop, t_eval_s, chip_hz, carrier_hz,
                       code_doppler_sign, mod):
    """Re-express a sample-0 cp in a new Doppler's currency, preserving the physical
    phase AT t_eval_s (#44 / #45 step 4).

    The ONLY correct t_eval_s is NOW: the tracker rebuilds the phase as
    cp0 + t*f_chip*(1 + k*dop) with t elapsed since sample 0, so changing dop by ddop
    moves the physical phase at time t by t*f_chip*k*ddop -- the translation must cancel
    it where the despread is actually running, which is the present. The coast path
    translated at the seed's ANCHOR epoch instead: that preserves the phase at the
    anchor and steps the phase NOW by anchor_age * k_c * ddop per forecast update --
    integrated over a coast at dop rate r, ~(f_chip/f_car)*r*age^2/2 chips of silent
    walk-off (the residual half of "long coasts silently lost the code peak"). The hold
    path always used the current epoch; now both go through this one function, kept
    beside dr_cp0/dr_seed_phys so the three transport directions cannot drift apart.
    """
    return ((cp_chips
             - t_eval_s * chip_hz * code_doppler_sign * (new_dop - old_dop) / carrier_hz)
            % mod)





def seed_phase_at_ref(phys_chips, doppler_hz, chip_hz, hops_per_sec, carrier_hz,
                      code_doppler_sign, mod, fft_len=None):
    """Broker-held physical phase -> `code_phase_at_ref_chips`, the field the tracker
    prefers over the sample-0 argument (#45 step 6).

    THE CONVERSION IS NOT OPTIONAL AND IT IS NOT A DETAIL. The broker's dr_cp0/dr_seed_phys
    pair references a hop at its FIRST sample; ChannelizedReplicaBank -- and therefore
    phase_from_arg, propagate_seed and every replica -- references it at the LAST
    (window_start + fft_len - 1). One hop apart: 52.3713 chips at CHORD. Shipping the
    broker's phase unconverted steps the commanded code by exactly that.

    MEASURED, not asserted (scripts/gnss/e2e_phase_transport.py, real propagate_seed + real
    GPU despread, PRN 3):
        arg    (production, cp0 only)          worst |err| 0.788 chips
        phase  (unconverted)                   worst |err| 51.999   <- the whole hop
        phase+ (this function)                 worst |err| 0.785    <- agrees to 0.003
    Re-run that after touching this, and note the middle row: it is what a "harmless
    refactor" of the seed transport looks like when the convention is dropped.

    Why ship the phase at all: cp0 back-references through t_abs (~2.3 days), so the pair
    (cp0, dop) only means anything TOGETHER -- every producer that updates one without
    re-projecting the other injects ~1700 chips/Hz (#42's writer, #44's coast). A phase at
    its own epoch has no partner to fall out of step with.

    `fft_len` supplies the exact "-1 sample" term; omitted, the offset is one whole hop and
    the residual is 0.0064 chips (two orders below the DLL's pull-in, and a constant, so it
    lands in the clock rather than in tracking).
    """
    per_hop = chip_hz / hops_per_sec * (1.0 + code_doppler_sign * doppler_hz / carrier_hz)
    off = per_hop * (1.0 - 1.0 / fft_len) if fft_len else per_hop
    return (phys_chips + off) % mod


def tracker_phase_at(seed, h1, hops_per_sec, chip_hz, carrier_hz, code_doppler_sign, mod,
                     fft_len=None):
    """The phase the TRACKER will command at hop h1 for this seed -- mirroring
    gnss::propagate_seed, including WHICH code reference it actually uses (#45 step 7).

    propagate_seed prefers `code_phase_at_ref_chips` over `code_phase_chips` whenever the
    former is >= 0 ("transported as a phase: no back-reference at all"), and the two are
    computed by DIFFERENT broker paths -- cp0 from cp_long, the phase from the search's
    cp_at_ref -- so they are not interchangeable. Any audit that models cp0 while the
    tracker reads the phase is measuring a stream nobody consumes: that is #43, whose
    symptom was +-90,000-chip "steps" on gps_l5 while the same satellites tracked at
    40 dB-Hz with live DLL trim.

    Returns the phase in the C++ last-sample convention (see seed_phase_at_ref), so it is
    directly comparable between consecutive seeds -- which is the seed audit's whole job.
    """
    per_hop = chip_hz / hops_per_sec * (1.0 + code_doppler_sign * seed["doppler_hz"]
                                        / carrier_hz)
    hop_off = per_hop * (1.0 - 1.0 / fft_len) if fft_len else per_hop
    ph_ref = seed.get("code_phase_at_ref_chips", -1.0)
    if ph_ref is None or ph_ref < 0.0:
        # the argument branch: undo the sample-0 back-reference, then move to the
        # tracker's reference point
        ph_ref = (dr_seed_phys({k: v for k, v in seed.items()
                                if k != "code_phase_at_ref_chips"},
                               seed["ref_hop"], hops_per_sec, chip_hz, carrier_hz,
                               code_doppler_sign, mod) + hop_off)
    dh = h1 - seed["ref_hop"]
    dt = dh / hops_per_sec
    return (ph_ref
            + (per_hop + seed.get("code_phase_rate", 0.0) or 0.0) * dh
            + 0.5 * (chip_hz / carrier_hz)
              * (seed.get("doppler_rate_hz_s", 0.0) or 0.0) * dt * dt) % mod


def rf_lobes(chans, power, clip_lo, clip_hi):
    """Group one tap's per-channel RF stats into contiguous LOBES. Pure; see test_rf_lobes.py.

    The tap serves a flat channel list because that is what it walks. But the list is the
    UNION of every chain's covering set on that GPU, so it arrives as one contiguous run per
    BAND -- 277..283 (1176.45 MHz) and 287..293 (1207.14) on cx19/gnss0. Contiguity is
    therefore the band grouping, and it needs no extra configuration and no second place that
    has to be kept in step with the generator.

    ⚠️ LOBES ARE NUMBERED, NOT NAMED. Naming them by frequency needs the node's GLOBAL channel
    map and the tap serves LOCAL comb indices -- the same two axes that put freq_id 5972 into
    a 384-channel frame while this was being built. Rather than guess a mapping here, the
    channel range is published and the consumer labels it. Numbering is ordered by channel, so
    lobe 0 is always the lower band.

    POWER IS MEANED, CLIP IS MAXED, and the asymmetry is deliberate: power is a LEVEL and the
    band's average is the honest summary, while clip is DAMAGE -- one railing channel corrupts
    what passes through it, and averaging it across thirteen quiet neighbours hides exactly the
    narrowband case this exists to catch (#56, and the 08-18 event was one band of two).
    """
    n = min(len(chans), len(power), len(clip_lo), len(clip_hi))
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: chans[i])
    out, run = [], []

    def flush(run):
        if not run:
            return
        lo = max(range(len(run)), key=lambda k: clip_lo[run[k]])
        hi = max(range(len(run)), key=lambda k: clip_hi[run[k]])
        out.append({
            "lobe": len(out),
            "chan0": chans[run[0]], "chan1": chans[run[-1]], "n_chan": len(run),
            "power": sum(power[i] for i in run) / float(len(run)),
            "power_max": max(power[i] for i in run),
            "clip_lo": clip_lo[run[lo]], "clip_lo_chan": chans[run[lo]],
            "clip_hi": clip_hi[run[hi]], "clip_hi_chan": chans[run[hi]],
        })

    for i in order:
        if run and chans[i] != chans[run[-1]] + 1:
            flush(run)
            run = []
        run.append(i)
    flush(run)
    return out


def instance_stall_verdict(prev, cur, now, min_stall_s, min_frac_advancing=0.5):
    """Which instances are SERVING but not ADVANCING. Pure; see test_instance_stall.py.

    `prev` is {url: (hop, first_seen_at_this_hop)} carried across cycles, `cur` is {url: hop}
    from this cycle. Returns (new_prev, stalled) where `stalled` is a list of
    (url, hop, seconds_stuck) for instances whose newest hop has not moved in >= min_stall_s
    while the fleet around them advanced.

    ⚠️ WHY THIS EXISTS WHEN --fe-axis-stale-s ALREADY WATCHES THE HOP. That guard watches the
    MAXIMUM hop across instances, which answers "has the whole time base frozen?" -- a real
    question, and it caught the cx19 collapse. It cannot see ONE instance of twelve wedged,
    because the eleven healthy ones keep the maximum climbing. On 2026-08-18 four instances
    wedged at once and that guard stayed correctly silent the entire time. This one is
    per-instance, which is the axis the other cannot resolve.

    ⚠️ AND IT KEYS ON THE COUNTER, NEVER ON REACHABILITY, because the whole trap is that a
    wedged instance ANSWERS. cx42/port 0 served plausible, well-formed rows while its capture
    window was frozen and the entire 195,313 pkt/s stream was being dropped; "all 12 respond"
    said nothing. An instance missing from `cur` is UNREACHABLE, a different fault that is
    already visible as n_src falling -- it is dropped from the state here, never accused.

    `min_frac_advancing` is the control clause: if most of the fleet is also standing still,
    this is not a per-instance stall but something global (a paused F-engine, a clock step,
    a replay), and calling it an instance fault would point the next hour in the wrong
    direction. Returns no accusations in that case -- deliberately silent rather than wrong.
    """
    new = {}
    for url, hop in cur.items():
        old = prev.get(url)
        new[url] = (hop, old[1] if (old is not None and old[0] == hop) else now)
    if not new:
        return new, []
    advancing = sum(1 for url, (hop, t0) in new.items() if t0 >= now)
    if advancing < max(1, int(round(min_frac_advancing * len(new)))):
        return new, []          # fleet-wide, not per-instance -- say nothing
    stalled = []
    for url, (hop, t0) in sorted(new.items()):
        stuck = now - t0
        if stuck >= min_stall_s:
            stalled.append((url, hop, stuck))
    return new, stalled


def q_stall_verdict(hist, now, window_s, frac, min_best, best, min_samples=10):
    """#70/#87 THE q STALL VERDICT: has this chain been quietly degraded?

    `hist` is [(t, duty)] appended once per broker cycle, `best` the highest windowed
    duty seen so far this session (None before any verdict). Returns
    (new_best, verdict_or_None); the verdict is (current, best, frac_of_best).

    WHY A PURE FUNCTION, AND WHY IT LIVES HERE: the first version of this lived inline in
    main() and could not be tested -- the on-sky fixtures run ~11 cycles at a duty that
    never falls, so a forced-trip replay could not make it fire, and "it did not fire"
    was indistinguishable from "it cannot fire". A guard nobody can trip is not a guard
    (chord-a-gate-that-cannot-fail), so the decision is separated from the plumbing and
    tested against a constructed collapse.

    SELF-REFERENTIAL BY DESIGN: chains differ ~4x in duty by construction (#49), so the
    baseline is the chain's own best, never a fleet-common bar. The baseline only RISES:
    a degrading chain must not be allowed to redefine "normal" downward, which is exactly
    how #87 ran for 3.5 h. min_best exempts a chain with no headroom to fall from
    (bds_b2b lives near 0.2 for structural nav-bit reasons, #31).
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
    """Split clock-solve offsets into (keep, drop) on per-satellite CONTINUITY of raw d_i.

    `offs` is [(prn, d_i)], d_i = clk + b_i: both stable, so a real satellite's d_i moves
    a few chips between cycles at most.

    ⚠️ HISTORY, because this function has now been wrong in BOTH directions and the second
    time was worse (2026-08-11):

    * The 2026-08-11 afternoon "Doppler lever" version subtracted
      lever = t_i*chip_rate*sign*dop/carrier from d_i before differencing, on the theory
      that d_i carried the detection Doppler times the 2.24-day sample-0 age (1696
      chips/Hz). The arithmetic of the lever is real -- but it CANCELS EXACTLY inside
      cp_loc, because the search's published cp0 embeds -t*chip_rate*sign*dop/carrier
      with the SAME dop and the SAME ref_hop (gnssSeedTransport.cpp detection_phase);
      cp_loc adds it back. Raw d_i never carried the lever at all. Subtracting it
      RE-INTRODUCED the term, so (d - lev) stepped 1696*ddop for any Doppler re-estimate
      > 0.059 Hz -- i.e. every satellite, every cycle (detection Doppler jitters +-60 Hz
      pass to pass). The guard then flagged all sats, hit the min-sats floor, and kept
      everything: a no-op that also invalidated the A/B that "measured" the lever fix.
      MEASURED live 2026-08-11 21:1x: "6 PRN(s) jumped ... keeping all" every cycle;
      and 1423 consecutive live detections show cp_loc continuous to median 0.27 chips
      (p99 2.37, none > 5), so raw d_i is the right quantity to test.

    * The still-open question this guard exists for: the 2026-08-10 PRN 2 incident
      (a non-L5 PRN reading noise, +-3000-chip swings dragging the median) and the
      2026-08-11 19:24-19:38 burst of genuine raw-d_i jumps (era-dependent, absent from
      the live stream at 21:xx). When it fires again, the WHAT MOVED log at the call
      site now decomposes d(cp_loc) -- the cancelled, physical quantity -- so the next
      reading of it does not have to guess (raw dcp is uniform mod L by construction
      and means nothing without the embed removed).

    `hist` is {prn: (t, d_i)} from the previous cycle and IS MUTATED here (every
    satellite's current value is recorded, including dropped ones -- a track that
    settles down must be able to rejoin).

    A satellite with no fresh history is always kept: the test needs two cycles, and
    refusing first sightings would stall the bootstrap.
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
