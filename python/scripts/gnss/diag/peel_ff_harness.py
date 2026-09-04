#!/usr/bin/env python3
"""Peel v2 Doppler-rate feed-forward harness (design proof, offline, synthetic gains).

Models the EXACT GnssVoltagePeel v2 gain-smoothing math (GnssVoltagePeel.cpp:322-364) in gain
space -- the despread itself is fine, the bug lives in the cross-record gain smoothing -- and asks:
how does the achievable peel depth degrade with the true Doppler RATE, and does the common-reference
feed-forward restore it?

Signal model (gain space, per code-period record r, absolute time t_r = r*dt):
  * true dish gain a0 (constant, slowly varying is the easy case; the FF is about carrier phase).
  * true carrier trajectory f(t) = dop0 + drate*t  (a pure ramp; dop0 is the anchor f_ref).
  * the REPLICA for record r is regenerated at fcar_r (broker seed), carrier ANCHORED to sample 0
    (gnssChannelizedReplica.cpp:229), so it removes phase 2*pi*fcar_r*t_r. The broker refreshes fcar_r
    to the current best Doppler every T_broker; between refreshes it is HELD (the staleness the memory
    flagged). => despread gain phase = 2*pi*[ (dop0-fcar_r)*t_r + 0.5*drate*t_r^2 ].
  * thermal noise n_r ~ CN(0, sigma^2) added to the per-record gain.

peel depth = |mean_r a_true| / |mean_r (a_true - a_sub)|   (coherent, over the span)
where a_sub is the gain each method subtracts. a_true here is the clean gain a0*bit_r (bit=+1: pilot).

Methods:
  V1            : subtract the raw per-record gain (no smoothing)               -> ~ per-record SNR floor
  V2_CURRENT    : causal FLL derotate + EMA, exactly as the C++ does today       -> lags the ramp
  V2_FIXED      : common-reference derotation (remove the KNOWN model phase to a fixed f_ref, incl.
                  the drate ramp and the fcar-refresh jump) + small residual FLL + EMA
  IDEAL         : derotate by the exact true phase (upper bound; = prototype's poly track)
"""
import numpy as np

TWO_PI = 2.0 * np.pi


def run(drate_hz_s, dt=10e-3, span_s=30.0, per_rec_snr_db=17.0, t_broker=1.0,
        fll_gain=0.05, alpha=0.15, seed=0):
    rng = np.random.default_rng(seed)
    nrec = int(round(span_s / dt))
    t = np.arange(nrec) * dt                       # absolute time per record
    dop0 = 0.0                                     # anchor; only drate matters for the residual

    # broker fcar_r: refreshed to the current TRUE freq every t_broker, held in between.
    fcar = np.empty(nrec)
    last = 0.0
    for r in range(nrec):
        if (t[r] - last) >= t_broker or r == 0:
            last = t[r]
        fcar[r] = dop0 + drate_hz_s * last         # held at the value at the last refresh

    # despread gain phase relative to the dop0 anchor (see model above)
    gain_phase = TWO_PI * ((dop0 - fcar) * t + 0.5 * drate_hz_s * t**2)

    a0 = 1.0                                        # clean dish gain (pilot, bit=+1)
    sigma = a0 / (10.0 ** (per_rec_snr_db / 20.0))  # per-record complex-gain noise
    noise = (rng.standard_normal(nrec) + 1j * rng.standard_normal(nrec)) * (sigma / np.sqrt(2))
    A = (a0 + noise) * np.exp(1j * gain_phase)      # the per-record despread gain the stage sees

    a_true = a0 * np.exp(1j * gain_phase)           # the CLEAN gain the peel should subtract (bit=+1)

    def depth(a_sub):
        # PHYSICAL metric: the peel leaves residual noise along R = (a_sub - a_true) per record.
        # Depth = signal / rms(subtracted-gain error). v1 (a_sub = raw A) -> rms = sigma = per-rec
        # SNR floor; smoothing drops the rms ~sqrt(N_eff); phase lag adds a systematic bias floor.
        err = a_sub - a_true
        rms = np.sqrt(np.mean(np.abs(err) ** 2))
        return 20.0 * np.log10(a0 / max(rms, 1e-12))

    # ---- V1: raw per-record gain (no smoothing) ----
    d_v1 = depth(A)

    # ---- V2_CURRENT: causal FLL + EMA (faithful to GnssVoltagePeel.cpp:322-364) ----
    # mode: 'none' = today (no FF); 'full' = common-reference (remove ALL known model phase, incl.
    # the absolute fcar-refresh jump); 'ramp' = remove only the within-poll ramp incrementally +
    # re-anchor (reset FLL/phi) at each dop refresh, accepting the per-poll seam (the cheap option).
    def v2(mode):
        phi = 0.0
        f_track = 0.0
        a_prev = 0.0 + 0j
        prev_ok = False
        gain_ema = 0.0 + 0j
        a_sub = np.empty(nrec, dtype=complex)
        t_poll = 0.0
        for r in range(nrec):
            # 'full': remove ALL known model phase to a fixed dop0 anchor (analytic, no lag, bridges
            # the fcar-refresh jump). 'ramp': remove only the within-poll ramp; re-anchor at each dop
            # refresh (the absolute jump is left for the FLL to re-converge over ~1/alpha records).
            if r > 0 and fcar[r] != fcar[r - 1]:       # broker dop refresh this window
                if mode == 'ramp':
                    t_poll = t[r]
                    phi = 0.0
                    prev_ok = False                    # re-anchor the FLL (seam)
            if mode == 'full':
                corr = TWO_PI * ((dop0 * t[r] + 0.5 * drate_hz_s * t[r]**2) - fcar[r] * t[r])
            elif mode == 'ramp':
                corr = TWO_PI * (0.5 * drate_hz_s * (t[r]**2 - t_poll**2)
                                 - (fcar[r] - dop0) * (t[r] - t_poll))  # ramp since last refresh
            else:
                corr = 0.0
            a_raw = A[r] * np.exp(-1j * corr)          # -> common reference frame (FIX) or raw (current)
            dt_r = dt if prev_ok else 0.0
            if prev_ok and dt_r > 0.0:
                phi += TWO_PI * f_track * dt_r
                phi = np.remainder(phi + np.pi, TWO_PI) - np.pi
            a_corr = a_raw * np.exp(-1j * phi)         # derotate by the causal FLL track
            if prev_ok and dt_r > 0.0 and abs(a_corr) > 0 and abs(a_prev) > 0:
                prod = a_corr * np.conj(a_prev)
                f_track += fll_gain * np.angle(prod * prod) / 2.0 / (TWO_PI * dt_r)
            a_prev = a_corr
            prev_ok = True
            # decision-directed EMA (pilot -> sign +1 always here; kept for faithfulness)
            sign = 1.0 if (gain_ema == 0 or np.real(a_corr * np.conj(gain_ema)) >= 0) else -1.0
            g_deb = a_corr * sign
            gain_ema = g_deb if gain_ema == 0 else (1 - alpha) * gain_ema + alpha * g_deb
            a_smooth = gain_ema * sign
            # back to the model frame (undo derotation + the reference correction) to match the replica
            a_sub[r] = a_smooth * np.exp(1j * (phi + corr))
        return depth(a_sub)

    d_cur = v2('none')
    d_ramp = v2('ramp')
    d_fix = v2('full')
    return d_v1, d_cur, d_ramp, d_fix


def seg_run(drate_hz_s=0.5, dt=10e-3, span_s=30.0, per_rec_snr_db=17.0, bf=0.45,
            alpha=0.15, seed=0):
    """Gap 2: every real window straddles the code-period boundary -> 2 segments with OPPOSITE
    overlay signs. Today (v1-segmented) subtracts a RAW per-segment gain each record (each estimated
    from ~half the record -> ~2 dof/window self-noise, no cross-record averaging). FIXED: common-ref
    derotate each segment, decision-directed sign-normalize (folds the overlay flip), feed BOTH into
    one per-PRN EMA -> the gain estimate averages over ~2x the records -> deeper peel.

    Compares the coherent residual power (physical: mean over ALL segment samples)."""
    rng = np.random.default_rng(seed)
    nrec = int(round(span_s / dt))
    t = np.arange(nrec) * dt
    gain_phase = TWO_PI * (0.5 * drate_hz_s * t**2)          # dop0=fcar anchor (isolate the ramp)
    a0 = 1.0
    sigma = a0 / (10.0 ** (per_rec_snr_db / 20.0))
    # per-segment overlay sign: head=+1, tail=-1 (the boundary flip); each segment ~half the record
    # so its gain-estimate noise is ~sqrt(2) larger than a full record.
    seg_sigma = sigma * np.sqrt(2.0)

    def seg_gain(ov):  # one segment's noisy despread gain (clean gain a0*overlay, phase gain_phase)
        n = (rng.standard_normal(nrec) + 1j * rng.standard_normal(nrec)) * (seg_sigma / np.sqrt(2))
        return (a0 * ov + n) * np.exp(1j * gain_phase)

    Ah, At = seg_gain(+1.0), seg_gain(-1.0)                  # head, tail per-record segment gains
    true_h = a0 * (+1.0) * np.exp(1j * gain_phase)           # clean head gain (overlay +1)
    true_t = a0 * (-1.0) * np.exp(1j * gain_phase)           # clean tail gain (overlay -1)

    def resid_power(sub_h, sub_t):
        # PHYSICAL: rms error of the subtracted per-segment gain (residual noise injected along R).
        err = np.concatenate([sub_h - true_h, sub_t - true_t])
        rms = np.sqrt(np.mean(np.abs(err) ** 2))
        return 20.0 * np.log10(a0 / max(rms, 1e-12))

    # CURRENT (v1 segmented): subtract the raw per-segment gain each record (no smoothing)
    d_cur = resid_power(Ah, At)

    # FIXED: common-ref derotate (here gain_phase is known-model -> remove it), sign-normalize both
    # segments, ONE EMA fed by both -> re-apply sign + phase to subtract.
    ema = 0.0 + 0j
    sub_h = np.empty(nrec, complex)
    sub_t = np.empty(nrec, complex)
    for r in range(nrec):
        ph = np.exp(-1j * gain_phase[r])
        for seg, A_seg, out in ((0, Ah[r], sub_h), (1, At[r], sub_t)):
            a_ref = A_seg * ph                               # common-reference (model phase removed)
            sign = 1.0 if (ema == 0 or np.real(a_ref * np.conj(ema)) >= 0) else -1.0
            g_deb = a_ref * sign
            ema = g_deb if ema == 0 else (1 - alpha) * ema + alpha * g_deb
            out[r] = (ema * sign) * np.exp(1j * gain_phase[r])   # back to model frame + overlay sign
    d_fix = resid_power(sub_h, sub_t)
    return d_cur, d_fix


if __name__ == "__main__":
    # PHASE-ONLY (noise off): isolate the carrier-phase tracking. Residual is then PURELY the
    # smoothing-lag bias -- V1/IDEAL/V2_FIXED cancel to machine precision (inf dB); V2_CURRENT
    # shows the FLL's ramp-lag floor. This is the clean, confound-free design proof.
    print("PHASE-ONLY (no thermal noise) -- residual is pure carrier-phase-lag bias")
    print("  B1C: dt=10ms, span=30s, broker refresh=1s, fll_gain=0.05, alpha=0.15")
    print("  drate(Hz/s)   V2_CURRENT   V2_RAMP-only   V2_FULL   (inf = perfect)")
    for drate in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        _, d_cur, d_ramp, d_fix = run(drate, per_rec_snr_db=400.0)
        f = lambda d: " inf " if d > 150 else "%5.1f" % d
        print("  %8.2f      %s        %s         %s" % (drate, f(d_cur), f(d_ramp), f(d_fix)))

    # WITH thermal noise: the physical floor is sqrt(N)/sigma; the FF restores V2_CURRENT to it.
    # (V1 omitted -- per-record self-cancellation is a single-R artifact, not a real floor;
    #  V2_FIXED can read slightly above IDEAL for the same correlated-noise reason -- read the
    #  CURRENT-vs-FIXED gap, not the absolute dB.)
    print("\nWITH thermal noise (per-rec SNR=17 dB, 8-seed mean); V1 floor = 17.0 dB")
    print("  drate(Hz/s)   V1     V2_CURRENT   V2_RAMP-only   V2_FULL")
    for drate in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        res = np.mean([run(drate, seed=s) for s in range(8)], axis=0)
        d_v1, d_cur, d_ramp, d_fix = res
        print("  %8.2f    %5.1f     %6.1f       %6.1f       %6.1f"
              % (drate, d_v1, d_cur, d_ramp, d_fix))

    # Gap 2: segmented-window smoothing (every real window straddles the boundary)
    print("\nGap 2 -- SEGMENTED windows (drate=0.5 Hz/s, per-rec SNR=17 dB, 16-seed mean):")
    seg = np.mean([seg_run(seed=s) for s in range(16)], axis=0)
    print("  v1-segmented (raw per-segment) = %5.1f dB   ->   per-segment EMA (FIXED) = %5.1f dB"
          "   (+%.1f dB)" % (seg[0], seg[1], seg[1] - seg[0]))
