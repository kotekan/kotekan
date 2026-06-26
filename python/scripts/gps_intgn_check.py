#!/usr/bin/env python3
"""Coherent-vs-incoherent integration check from a live_intgn.yaml capture.

The combiner (integration_length: 1) records every tracker record raw, so each
row is one ~1 ms despread's full-band complex amplitude A = rec[4] + i*rec[5]
(|A| = rec[3], UTC double in slots 9..10), per PRN. This reconstructs, for any
integration length K, the two ways to combine K consecutive records:

  coherent   |<A>|        -- average the complex A, then take |.|  (SNR ~ K in power)
  incoherent sqrt<|A|^2>  -- average the power, then sqrt          (SNR ~ sqrt(K))

A genuine code+carrier lock holds coherent ~= incoherent until the phase
decorrelates -- the 20 ms GPS nav-bit edge (sign flip) is the hard wall, plus
any residual Doppler. A noise-only PRN's coherent estimate instead falls as
1/sqrt(K) from K=1 (random phases average to zero). The ratio of the two is the
processing gain that digs into the dish sidelobes.

Usage:  python3 gps_intgn_check.py [/tmp/gpsintgn] [--record-floats 11]
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

# Combiner record slot layout (see GnssCoherentCombiner.cpp).
S_PRN, S_IAMP, S_ARE, S_AIM, S_CAMP = 0, 3, 4, 5, 6
UTC_SLOT = 9


def load_records(base_dir, record_floats, n_prn):
    """Parse rawFileWrite framing: each frame is [uint32 metadata_size][metadata]
    [n_prn*record_floats float32 of data]. out_buf is metadata-less (size 0), but
    that 4-byte header is ALWAYS written, so a flat float reshape is misaligned --
    skip the header per frame."""
    files = sorted(glob.glob(os.path.join(base_dir, "*.raw")))
    if not files:
        sys.exit("no *.raw under %s -- run live_intgn.yaml first" % base_dir)
    frame_floats = n_prn * record_floats
    frame_bytes = frame_floats * 4
    frames = []
    for fn in files:
        buf = open(fn, "rb").read()
        pos, n = 0, len(buf)
        while pos + 4 <= n:
            (msize,) = struct.unpack_from("<I", buf, pos)
            pos += 4 + msize
            if pos + frame_bytes > n:
                break
            frames.append(np.frombuffer(buf, np.float32, frame_floats, pos))
            pos += frame_bytes
    if not frames:
        sys.exit("no complete %d-PRN frames found (wrong --n-prn? expected n_prn*%d floats)"
                 % (n_prn, record_floats))
    flat = np.stack(frames).reshape(-1, record_floats)  # (n_frames*n_prn, record_floats)
    by_prn = {}
    for prn in np.unique(np.rint(flat[:, S_PRN]).astype(int)):
        if prn <= 0:
            continue
        rows = flat[np.rint(flat[:, S_PRN]).astype(int) == prn]
        A = rows[:, S_ARE].astype(np.float64) + 1j * rows[:, S_AIM].astype(np.float64)
        utc = np.ascontiguousarray(rows[:, UTC_SLOT:UTC_SLOT + 2]).view(np.float64).ravel()
        by_prn[int(prn)] = (A, utc)
    return by_prn


def fit_residual_doppler(A, t, max_gap_s, min_pairs=50):
    """Residual carrier (Hz) from the phase walk between adjacent records. The product
    A_{i+1} conj A_i carries 2pi*df*dt PLUS a possible pi flip when a nav-bit edge falls
    between the records; squaring it (angle(p^2)/2) cancels the +-1 data so the fit isn't
    fooled by bit flips (valid for |df*dt| < 1/4, i.e. |df| < 250 Hz here). Only adjacent
    (non-gap) pairs. Needs CAPTURE-time t (set capture_utc0); emit-time t would scramble it."""
    dt = np.diff(t)
    prod = A[1:] * np.conj(A[:-1])
    dphi = np.angle(prod * prod) / 2.0  # squared -> removes the +-1 nav-bit pi flips
    ok = (dt > 0) & (dt < max_gap_s) & (np.abs(A[1:]) > 0) & (np.abs(A[:-1]) > 0)
    if ok.sum() < min_pairs:
        return 0.0, 0
    return float(np.median(dphi[ok] / dt[ok]) / (2.0 * np.pi)), int(ok.sum())


def derotate_wander(A, t, win, max_gap_s):
    """Remove the slow carrier WANDER (not just a global offset): fit a local residual
    per win-record window, integrate it into a continuous phase, and derotate. This is
    a perfect-FLL oracle -- if coherence still caps after this, the wall is the nav bit
    (or phase noise faster than `win`), not carrier tracking."""
    dfs = np.zeros(len(A))
    for s in range(0, len(A), win):
        e = min(s + win, len(A))
        if e - s >= 20:
            f, n = fit_residual_doppler(A[s:e], t[s:e], max_gap_s, min_pairs=15)
            dfs[s:e] = f if n > (e - s) // 2 else 0.0
    dt = np.diff(t, prepend=t[0])
    phi = np.cumsum(2.0 * np.pi * dfs * dt)  # continuous phase (keeps cross-window coherence)
    return A * np.exp(-1j * phi)


def navbit_wipe(A, t, bit_s=0.020, nphase=20):
    """Estimated GPS nav-bit (50 bps -> 20 ms) data wipe-off. Bit-sync by the 20 ms
    phase that maximises per-bit coherence, resolve the bit sign per epoch via squaring
    (theta0 = 1/2 arg sum s^2, removing the +-1 ambiguity), then multiply it back out.
    Removing the data modulation lets coherent integration run past the 20 ms nav bit.
    Estimated here (bright sat); for deep sidelobes the bits are DECODED from a strong
    reference / predicted from ephemeris (same bits for every beam). Pass FLL-derotated
    A so the carrier doesn't smear the per-epoch sums."""
    t0 = t[0]
    best_off, best_g = 0.0, -1.0
    for k in range(nphase):
        off = k * bit_s / nphase
        ep = np.floor((t - t0 - off) / bit_s).astype(np.int64)
        ep -= ep.min()
        cnt = np.bincount(ep)
        sr = np.bincount(ep, weights=A.real)
        si = np.bincount(ep, weights=A.imag)
        v = cnt >= 2
        g = float(np.mean(np.hypot(sr[v], si[v]))) if v.any() else 0.0
        if g > best_g:
            best_g, best_off = g, off
    ep = np.floor((t - t0 - best_off) / bit_s).astype(np.int64)
    ep -= ep.min()
    s = np.bincount(ep, weights=A.real) + 1j * np.bincount(ep, weights=A.imag)
    # Track the carrier phase PER EPOCH (squaring + unwrap removes the +-1 data), then
    # the bit is the sign along that phase. This follows a slow carrier far better than a
    # single global phase; a fast wander still aliases the unwrap -> the live FLL is what
    # ultimately delivers clean bits, but this decodes a well-tracked carrier cleanly.
    phi = 0.5 * np.unwrap(np.angle(s * s))
    bit = np.sign(np.real(s * np.exp(-1j * phi)))     # +-1 per 20 ms epoch
    bit[bit == 0] = 1.0
    return A * bit[ep], best_off, int((bit != bit[0]).sum()), bit


def windowed_doppler(A, t, win, max_gap_s):
    """Residual carrier fit in consecutive win-record windows -> array of df (Hz).
    A constant residual gives a tight cluster (one derotation fixes it); a wandering
    seed (broker re-solving a coarse-grid clock bias each cycle) gives a wide spread."""
    out = []
    for s in range(0, len(A) - win, win):
        f, n = fit_residual_doppler(A[s:s + win], t[s:s + win], max_gap_s)
        if n > win // 2:
            out.append(f)
    return np.array(out)


def integrate(A, Ks):
    """Per K, over non-overlapping K-record blocks: coherent |<A>| and incoherent
    sqrt<|A|^2>, as mean +/- std across blocks. Returns {K: dict(coh, coh_std,
    incoh, incoh_std, nb)}."""
    N = len(A)
    out = {}
    for K in Ks:
        nb = N // K
        if nb < 1:
            continue
        blk = A[:nb * K].reshape(nb, K)
        coh = np.abs(blk.mean(axis=1))                  # |<A>| per block
        incoh = np.sqrt((np.abs(blk) ** 2).mean(axis=1))  # sqrt<|A|^2> per block
        out[K] = dict(coh=coh.mean(), coh_std=coh.std(),
                      incoh=incoh.mean(), incoh_std=incoh.std(), nb=nb)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", nargs="?", default="/tmp/gpsintgn")
    ap.add_argument("--record-floats", type=int, default=11)
    ap.add_argument("--n-prn", type=int, default=9, help="records per frame (config n_prn)")
    ap.add_argument("--dt-ms", type=float, default=1.0, help="record period (ms) for the time axis")
    ap.add_argument("--png", default=None, help="write a coh/incoh-vs-time plot here")
    args = ap.parse_args(argv)

    by_prn = load_records(args.base_dir, args.record_floats, args.n_prn)
    # Unseeded PRNs are emitted as EXACT zeros (tracker memsets inactive records) --
    # drop them (std ~0) so they can't be mistaken for a noise reference.
    tracked = {p: v for p, v in by_prn.items()
               if len(v[0]) > 8 and float(np.abs(v[0]).std()) > 1e-9}
    if not tracked:
        sys.exit("no PRN was actively tracked (all records zero -- broker never seeded?)")
    strength = {p: float(np.abs(A).mean()) for p, (A, _) in tracked.items()}
    ranked = sorted(strength, key=strength.get, reverse=True)
    by_prn = tracked
    print("per-record mean |A| by PRN (lock = high, noise = low):")
    for p in ranked:
        print("  PRN %2d  |A|=%.3f  (%d records)" % (p, strength[p], len(by_prn[p][0])))

    locked = [p for p in ranked if strength[p] > 2.0 * strength[ranked[-1]]][:3] or ranked[:1]
    noise_prn = ranked[-1]
    if strength[noise_prn] > 0.5 * strength[ranked[0]]:
        print("\nNOTE: weakest tracked PRN %d (|A|=%.3f) still has signal -- no clean noise\n"
              "reference, so the noise floor below is an OVER-estimate (SNR_coh a lower bound)."
              % (noise_prn, strength[noise_prn]))
    Ks = [1, 2, 3, 5, 8, 12, 20, 32, 50, 80, 128, 200]

    # Contiguity: large UTC gaps mean K records span > K*dt ms (valve drops).
    A0, utc0 = by_prn[ranked[0]]
    dutc = np.diff(utc0) * 1e3  # ms
    med = float(np.median(dutc)) if len(dutc) else args.dt_ms
    gaps = int(np.sum(dutc > 3 * med)) if len(dutc) else 0
    print("\nmedian inter-record dt %.2f ms; %d gaps >3x median (of %d). Using dt=%.2f ms."
          % (med, gaps, len(dutc), args.dt_ms))

    # The noise PRN gives the NOISE FLOOR of each method: coherent |<A>| averages the
    # random-phase noise toward zero (falls ~1/sqrt(K)); incoherent sqrt<|A|^2> squares
    # away the phase so the noise can't cancel (flat pedestal ~sigma). That contrast --
    # NOT the signal estimate, which converges to |a| for BOTH -- is the whole argument.
    nz = integrate(by_prn[noise_prn][0], Ks)
    print("\n=== NOISE FLOOR (PRN %d, no signal) ===" % noise_prn)
    print("   K   t(ms)   coh floor   incoh floor    (coh should fall ~1/sqrt(K); incoh ~flat)")
    base = nz[1]["coh"] if 1 in nz else float("nan")
    for K in Ks:
        if K not in nz:
            continue
        ideal = base / np.sqrt(K)  # 1/sqrt(K) reference from K=1
        print("  %3d  %6.1f   %8.4f     %8.4f      (1/sqrtK ref %.4f)"
              % (K, K * args.dt_ms, nz[K]["coh"], nz[K]["incoh"], ideal))

    nominal_s = med * 1e-3 if med > 0 else args.dt_ms * 1e-3
    for p in locked:
        A, t = by_prn[p]
        cur = integrate(A, Ks)
        # Global residual (one number) for reference, then a perfect-FLL oracle that
        # removes the slow carrier WANDER per window. If deRot still caps, the wall is
        # the nav bit / fast phase noise -- not carrier tracking (no FLL would beat it).
        df, npair = fit_residual_doppler(A, t, max_gap_s=1.5 * nominal_s)
        fll_win = max(50, int(round(0.05 / nominal_s)))  # ~50 ms windows track the wander
        deA = derotate_wander(A, t, fll_win, 1.5 * nominal_s)
        der = integrate(deA, Ks)
        print("\n=== PRN %d ===  global residual %+0.1f Hz (%d pairs); deRot = per-window "
              "carrier removed (perfect-FLL oracle)" % (p, df, npair))
        print("   K   t(ms) | coh|<A>|  coh/incoh  SNR_coh |  deRot coh  deRot c/i  deRot SNR")
        for K in Ks:
            if K not in cur or K not in nz:
                continue
            coh, incoh = cur[K]["coh"], cur[K]["incoh"]
            dcoh, dincoh = der[K]["coh"], der[K]["incoh"]
            f = nz[K]["coh"]
            snr = coh / f if f else float("nan")
            dsnr = dcoh / f if f else float("nan")
            tag = "  <- ~20 ms nav bit" if (K * med >= 20 and (K - 1) * med < 20) else ""
            print("  %3d  %6.1f | %7.3f    %5.2f   %6.1f | %8.3f    %5.2f   %6.1f%s"
                  % (K, K * med, coh, coh / incoh if incoh else 0, snr,
                     dcoh, dcoh / dincoh if dincoh else 0, dsnr, tag))

    # Nav-bit wipe on the brightest sat: FLL-derotate, estimate + remove the 20 ms data
    # bits, then integrate well past 20 ms. This is the path to seconds of coherent
    # integration (deep-sidelobe SNR) -- here the bits are estimated from the bright sat;
    # the live science decodes/predicts them from a strong reference.
    pb = locked[0]
    Ab, tb = by_prn[pb]
    fll_win = max(50, int(round(0.05 / nominal_s)))
    # Estimate + wipe the bits on the RAW signal (per-epoch carrier-tracked), THEN remove
    # the carrier from the bit-free stream (the FLL discriminator is meaningless across a
    # bit edge, so it must run after the wipe). no-wipe reference = carrier removed only.
    wiped_raw, woff, nflip, bitseq = navbit_wipe(Ab, tb)
    deAb = derotate_wander(Ab, tb, fll_win, 1.5 * nominal_s)
    wiped = derotate_wander(wiped_raw, tb, fll_win, 1.5 * nominal_s)
    # Decode the actual GPS nav message from the estimated bits (parity-validated):
    # real subframes with the TLM preamble + a marching TOW prove these are the true
    # data bits, so the wipe is exact, not a guess.
    try:
        from gps_nav_decode import frame_sync
        bits01 = (bitseq < 0).astype(np.int8)            # +-1 epoch signs -> 0/1 (polarity-free)
        subs = frame_sync(bits01)
        if subs:
            tows = [tow for _, tow, _ in subs]
            print("\nNAV DECODE (PRN %d): %d parity-valid subframes; subframe IDs %s; TOW %s"
                  % (pb, len(subs), [s for _, _, s in subs][:6],
                     "marching" if len(tows) > 1 and all(
                         tows[i + 1] - tows[i] == 1 for i in range(len(tows) - 1)) else tows[:6]))
        else:
            print("\nNAV DECODE (PRN %d): no parity-valid subframe yet (bit errors from gaps/SNR "
                  "or too short) -- estimated wipe still applies." % pb)
    except Exception as e:
        print("\nNAV DECODE skipped: %s" % e)
    Kw = [20, 50, 100, 200, 500, 1000, 2000]
    rawc = integrate(deAb, Kw)
    wipc = integrate(wiped, Kw)
    nzc = integrate(by_prn[noise_prn][0], Kw)
    print("\n=== NAV-BIT WIPE (PRN %d) ===  bit-sync offset %.1f ms, %d bit flips; "
          "FLL-derotated then data-wiped" % (pb, woff * 1e3, nflip))
    print("   K   t(ms) |  no-wipe coh  c/i |  WIPED coh   c/i   SNR_coh   (past 20 ms)")
    for K in Kw:
        if K not in wipc:
            continue
        r, w = rawc.get(K), wipc[K]
        fl = nzc[K]["coh"] if K in nzc and nzc[K]["coh"] else float("nan")
        print("  %4d  %6.0f |  %8.3f  %4.2f |  %8.3f  %4.2f  %7.1f"
              % (K, K * med, r["coh"] if r else float("nan"),
                 (r["coh"] / r["incoh"] if r and r["incoh"] else 0),
                 w["coh"], w["coh"] / w["incoh"] if w["incoh"] else 0,
                 w["coh"] / fl if fl == fl and fl else float("nan")))
    print("  >> no-wipe coh dies at the 20 ms nav bit; WIPED coh keeps growing -> SNR climbs")
    print("     past it. Seconds of coherent integration = the deep-sidelobe sensitivity.")

    # Is the residual carrier a CONSTANT (one derotation fixes it) or WANDERING (the
    # broker re-seeds Doppler each ~0.2 s cycle from a coarse grid -> a staircase no
    # single df can track)? Fit df in per-cycle windows and report the spread.
    p0 = locked[0]
    A0, t0 = by_prn[p0]
    win = max(50, int(round(0.2 / nominal_s)))
    wf = windowed_doppler(A0, t0, win, 1.5 * nominal_s)
    if len(wf) > 2:
        print("\nCarrier stability (PRN %d, %d-record ~%.0f ms windows, %d of them):" %
              (p0, win, win * nominal_s * 1e3, len(wf)))
        print("  residual df  min %+.0f / med %+.0f / max %+.0f Hz, spread (std) %.0f Hz" %
              (wf.min(), np.median(wf), wf.max(), wf.std()))
        print("  >> spread >> a few Hz means the SEED Doppler is wandering cycle-to-cycle")
        print("     (coarse search grid -> jittery clock-bias re-solve); one global")
        print("     derotation can't track it. A continuous carrier loop (FLL) is the fix.")
        print("  >> spread of only a few Hz but coherence still short -> oscillator phase noise.")

    print("\nRead:")
    print(" * coh|<A>| and incoh sqrt<|A|^2> both converge to |a| -- equal estimates, not SNRs.")
    print("   The K-vs-sqrt(K) is in the FLOOR: coherent's falls ~1/sqrt(K), incoherent's is")
    print("   a fixed sigma pedestal. So SNR_coh climbs ~sqrt(K) WHILE phase-coherent.")
    print(" * RAW coh/incoh rolling off from K=1 = a continuous phase drift (residual carrier),")
    print("   NOT the nav bit (which would hold ~1 to 20 ms then crash). The fitted Hz is that")
    print("   drift. deRot = after removing it: if deRot coh/incoh now holds to ~20 ms, the")
    print("   live limit is Doppler precision (-> tracker FLL/Doppler-refine); if it then")
    print("   crashes right at 20 ms, THAT is the nav bit (-> nav-bit wipe / L5 pilot).")

    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            for p in locked:
                cur = integrate(by_prn[p][0], Ks)
                ks = [k for k in Ks if k in cur]
                t = [k * args.dt_ms for k in ks]
                ax.plot(t, [cur[k]["coh"] for k in ks], "-o", label="PRN%d coherent |<A>|" % p)
                ax.plot(t, [cur[k]["incoh"] for k in ks], "--s",
                        label="PRN%d incoherent" % p)
            ks = [k for k in Ks if k in nz]
            t = [k * args.dt_ms for k in ks]
            ax.plot(t, [nz[k]["coh"] for k in ks], ":x", color="gray",
                    label="PRN%d coherent FLOOR (1/sqrtK)" % noise_prn)
            ax.plot(t, [nz[k]["incoh"] for k in ks], ":+", color="black",
                    label="PRN%d incoherent FLOOR (flat)" % noise_prn)
            ax.axvline(20.0, color="r", ls=":", lw=1, label="20 ms nav bit")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("integration time (ms)"); ax.set_ylabel("|A|")
            ax.set_title("Coherent vs incoherent integration"); ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout(); fig.savefig(args.png, dpi=110)
            print("\nwrote %s" % args.png)
        except Exception as e:
            print("\n(plot skipped: %s)" % e)


if __name__ == "__main__":
    main()
