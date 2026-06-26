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
import sys

import numpy as np

# Combiner record slot layout (see GnssCoherentCombiner.cpp).
S_PRN, S_IAMP, S_ARE, S_AIM, S_CAMP = 0, 3, 4, 5, 6
UTC_SLOT = 9


def load_records(base_dir, record_floats):
    files = sorted(glob.glob(os.path.join(base_dir, "*.raw")))
    if not files:
        sys.exit("no *.raw under %s -- run live_intgn.yaml first" % base_dir)
    raw = np.concatenate([np.fromfile(f, dtype=np.float32) for f in files])
    nrec_floats = raw.size - (raw.size % record_floats)
    flat = raw[:nrec_floats].reshape(-1, record_floats)
    n_prn_guess = None
    # Frames hold n_prn records back-to-back; infer n_prn from the PRN-id column
    # repeating (first row's neighbours share the frame). Fall back to grouping by
    # the distinct PRN ids present.
    prn_ids = np.unique(np.rint(flat[:, S_PRN]).astype(int))
    prn_ids = prn_ids[prn_ids > 0]
    by_prn = {}
    for prn in prn_ids:
        sel = np.rint(flat[:, S_PRN]).astype(int) == prn
        rows = flat[sel]
        A = rows[:, S_ARE].astype(np.float64) + 1j * rows[:, S_AIM].astype(np.float64)
        utc = rows[:, UTC_SLOT:UTC_SLOT + 2].copy().view(np.float64).ravel()
        by_prn[int(prn)] = (A, utc)
    _ = n_prn_guess
    return by_prn


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
    ap.add_argument("--dt-ms", type=float, default=1.0, help="record period (ms) for the time axis")
    ap.add_argument("--png", default=None, help="write a coh/incoh-vs-time plot here")
    args = ap.parse_args(argv)

    by_prn = load_records(args.base_dir, args.record_floats)
    # Mean per-record |A| ranks locked sats (high) vs noise PRNs (low).
    strength = {p: float(np.abs(A).mean()) for p, (A, _) in by_prn.items() if len(A) > 8}
    if not strength:
        sys.exit("no PRN had enough records to integrate")
    ranked = sorted(strength, key=strength.get, reverse=True)
    print("per-record mean |A| by PRN (lock = high, noise = low):")
    for p in ranked:
        n = len(by_prn[p][0])
        print("  PRN %2d  |A|=%.3f  (%d records)" % (p, strength[p], n))

    locked = [p for p in ranked if strength[p] > 2.0 * strength[ranked[-1]]][:3] or ranked[:1]
    noise_prn = ranked[-1]
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

    for p in locked:
        cur = integrate(by_prn[p][0], Ks)
        print("\n=== PRN %d ===  signal estimate; SNR_coh = coh|<A>| / coherent floor" % p)
        print("   K   t(ms)   coh|<A>|  incoh sqrt<|A|^2>   coh/incoh   SNR_coh(~sqrtK?)")
        for K in Ks:
            if K not in cur or K not in nz:
                continue
            coh, incoh = cur[K]["coh"], cur[K]["incoh"]
            # Signal over the collapsing coherent floor -> amplitude SNR ~sqrt(K)
            # (= K in power) WHILE phase-coherent; rolls off past the coherence limit.
            snr_coh = coh / nz[K]["coh"] if nz[K]["coh"] else float("nan")
            tag = ""
            if K * args.dt_ms >= 20 and (K - 1) * args.dt_ms < 20:
                tag = "  <- ~20 ms nav-bit edge"
            print("  %3d  %6.1f   %7.3f     %8.3f       %5.2f       %6.1f%s"
                  % (K, K * args.dt_ms, coh, incoh, coh / incoh if incoh else 0,
                     snr_coh, tag))

    print("\nRead:")
    print(" * coh|<A>| and incoh sqrt<|A|^2> BOTH converge to |a| (the signal amplitude) --")
    print("   that's why they look equal; a mean amplitude is not an SNR.")
    print(" * The difference is the FLOOR (table above): coherent's collapses ~1/sqrt(K),")
    print("   incoherent's is a fixed sigma pedestal. So the coherent amplitude SNR climbs")
    print("   ~sqrt(K) (= K in power); incoherent only gains ~sqrt(K) in POWER (slower), via")
    print("   shrinking its floor's block-to-block scatter -- it never lowers the floor.")
    print(" * coh/incoh ~1 then rolling off, and SNR_coh peaking, mark the coherence limit")
    print("   (20 ms nav bit, or earlier if residual Doppler) -- that sets the next step:")
    print("   nav-bit wipe / L5 pilot to keep integrating coherently past it.")

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
