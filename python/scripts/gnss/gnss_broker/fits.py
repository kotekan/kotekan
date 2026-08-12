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
                   rec_hops=2048, prev_val=None, max_step=3.0, unit_hop=None):
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
        q = float(rec.get("deep_rate_q") or 0.0)
        f = rec.get("deep_rate_hz")
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
