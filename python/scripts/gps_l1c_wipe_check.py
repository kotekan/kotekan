#!/usr/bin/env python3
"""L1C-P overlay-wipe / coherence check from a live_intgn_l1c.yaml capture.

The combiner (integration_length: 1) records every tracker record raw, so each row
is one 10 ms L1C-P despread's full-band complex amplitude A = rec[4] + i*rec[5], per
PRN, with capture UTC. This works out -- OFFLINE, before wiring it live -- whether the
known per-PRN L1CO overlay wipe actually digs the dataless pilot out, and what the
limit is:

  * Run the wipe (search all 1800 overlay phases for the coherent-sum max) per PRN and
    show its noise floor (~sqrt(2 ln 1800) ~ 3.9 sigma -- the 1800-way max biases H0 up).
  * On the brightest GPS III sat (the only ones that broadcast L1C: PRN 04/11/14/18/23/28),
    compare integration vs K: incoherent sqrt<|A|^2>, coherent-NO-wipe |<A>| (dies as the
    overlay flips sign across primary periods), coherent-WITH-wipe |<overlay*A>| (should
    build past the 10 ms primary period IF the carrier stays coherent and the alignment is
    right). A perfect-FLL oracle (derotate the slow carrier) separates the two limits:
      - wiped-coherent caps early, derotate RESCUES it  -> carrier/FLL is the wall (OCXO).
      - wiped-coherent builds for seconds already        -> the wall live is the 1800-phase
        search degeneracy -> SEED the phase from GPS time + a short window (no OCXO needed).
  * Report the recovered overlay phase + cpi[0] so the live GPS-time phase-seed can be wired.

Usage:  python3 gps_l1c_wipe_check.py [/tmp/gpsintgn_l1c] [--png out.png]
"""
import argparse
import os
import sys

import numpy as np

# Reuse the capture loader + integration/FLL machinery from the L1/L2C harness (same
# record layout: A in slots 4/5, |A| slot 3, UTC double in slots 9..10, record_floats 11).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gps_intgn_check import load_records, integrate, fit_residual_doppler, derotate_wander  # noqa: E402

# GPS III space vehicles broadcast L1C; older blocks (IIR/IIR-M/IIF) do NOT. PRNs as of 2026
# (SV01-06 = PRN 04/18/23/14/11/28); update as IIIF launches. Used only to TAG the output --
# a "lock" on any other PRN here is cross-correlation/noise, not real L1C.
GPS_III_PRNS = {4, 11, 14, 18, 23, 28}

# --- L1CO overlay generator (port of gps::generate_l1co_code; PocketSDR sec_code_L1CP, S1 only,
# exact for GPS PRNs 1..63). 11-stage Galois LFSR, per-PRN feedback poly + init (octal). -------
_L1CO_S1_POLY = (
    0o5111, 0o5421, 0o5501, 0o5403, 0o6417, 0o6141, 0o6351, 0o6501, 0o6205, 0o6235, 0o7751,
    0o6623, 0o6733, 0o7627, 0o5667, 0o5051, 0o7665, 0o6325, 0o4365, 0o4745, 0o7633, 0o6747,
    0o4475, 0o4225, 0o7063, 0o4423, 0o6651, 0o4161, 0o7237, 0o4473, 0o5477, 0o6163)
_L1CO_S1_INIT = (
    0o3266, 0o2040, 0o1527, 0o3307, 0o3756, 0o3026, 0o0562, 0o0420, 0o3415, 0o0337, 0o0265,
    0o1230, 0o2204, 0o1440, 0o2412, 0o3516, 0o2761, 0o3750, 0o2701, 0o1206, 0o1544, 0o1774,
    0o0546, 0o2213, 0o3707, 0o2051, 0o3650, 0o1777, 0o3203, 0o1762, 0o2100, 0o0571)
L1CO_LEN = 1800


def _rev11(r):
    rr = 0
    for i in range(11):
        rr = (rr << 1) | ((r >> i) & 1)
    return rr


def generate_l1co_code(prn):
    """Bipolar (+-1) L1C-P overlay for PRN 1..32, length 1800 (matches the C++ generator)."""
    if not 1 <= prn <= 32:
        raise ValueError("PRN must be 1..32")
    tap = _rev11(_L1CO_S1_POLY[prn - 1] >> 1)
    R = _rev11(_L1CO_S1_INIT[prn - 1])
    code = np.empty(L1CO_LEN, dtype=np.int8)
    for i in range(L1CO_LEN):
        code[i] = -1 if (R & 1) else 1                    # CHIP: bit0 -> +1, bit1 -> -1
        fb = bin(R & tap).count("1") & 1
        R = (fb << 10) | (R >> 1)
    return code


def cpi_from_utc(utc):
    """Absolute primary-period index per record from capture UTC (median step = no-drop period);
    a valve drop just skips an index, so the overlay stays aligned. Mirrors overlay_wipe()."""
    d = np.diff(utc)
    rec_dt = float(np.median(d)) if len(d) else 0.0
    if not rec_dt > 0:
        return None, 0.0
    return np.rint((utc - utc[0]) / rec_dt).astype(np.int64), rec_dt


def overlay_wipe(A, cpi, overlay):
    """Port of gnss::overlay_wipe: search the L overlay alignments for the coherent-sum max.
    Returns (best_phase, deep_amplitude=|sum|/nrec, deep_snr=|sum|/orthogonal-noise)."""
    L = len(overlay)
    O = overlay.astype(np.float64)
    m = (cpi % L).astype(np.int64)
    W = np.zeros(L, dtype=np.complex128)              # per-residue complex weight
    np.add.at(W, m, A)
    # s(p) = sum_m W[m]*O[(m+p)%L] = circular cross-correlation, via FFT (validated by the
    # unit-tested C++; here L=1800 so even the direct form is cheap).
    S = np.fft.ifft(np.conj(np.fft.fft(W)) * np.fft.fft(O))
    p = int(np.argmax(np.abs(S)))
    best = S[p]
    mag = abs(best)
    # Significance: component of each overlay-corrected record orthogonal to the coherent sum.
    rot = np.exp(-1j * np.angle(best)) if mag > 0 else 1.0
    vr = A * O[(m + p) % L] * rot
    noise = float(np.sqrt(np.sum(vr.imag ** 2)))
    return p, mag / len(A), (mag / noise if noise > 0 else 0.0)


def wipe_stream(A, cpi, overlay, phase):
    """Multiply each record by its overlay chip at the given alignment -> a (pilot) stream whose
    primary periods are in-phase, so coherent integration can run past one 10 ms period."""
    L = len(overlay)
    return A * overlay[((cpi % L) + phase) % L].astype(np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", nargs="?", default="/tmp/gpsintgn_l1c")
    ap.add_argument("--record-floats", type=int, default=11)
    ap.add_argument("--n-prn", type=int, default=0, help="records/frame; 0 = auto-detect")
    ap.add_argument("--png", default=None, help="write a wiped-vs-unwiped integration plot here")
    args = ap.parse_args(argv)

    by_prn = load_records(args.base_dir, args.record_floats, args.n_prn)
    tracked = {p: v for p, v in by_prn.items()
               if len(v[0]) > 8 and float(np.abs(v[0]).std()) > 1e-9}
    if not tracked:
        sys.exit("no PRN was actively tracked (all records zero -- broker never seeded?)")
    strength = {p: float(np.abs(A).mean()) for p, (A, _) in tracked.items()}
    ranked = sorted(strength, key=strength.get, reverse=True)

    print("per-record mean |A| by PRN (L1C = GPS III only; * = broadcasts L1C):")
    for p in ranked:
        tag = " *GPS III" if p in GPS_III_PRNS else "  (no L1C -> cross-corr/noise)"
        print("  PRN %2d  |A|=%.4f  (%d records)%s" % (p, strength[p], len(tracked[p][0]), tag))

    # Full-capture overlay wipe per PRN: the blind 1800-phase search the live combiner runs.
    # The noise floor (max of 1800 Rayleigh phases) is ~3.9 sigma -- read deep_snr against THAT.
    print("\n=== blind 1800-phase overlay wipe (whole capture) ===")
    print("  PRN   deep|A|   deep_snr   phase     (floor ~3.9 sigma; * = GPS III)")
    wipe_res = {}
    for p in ranked:
        A, utc = tracked[p]
        cpi, rec_dt = cpi_from_utc(utc)
        if cpi is None:
            continue
        ov = generate_l1co_code(p)
        ph, amp, snr = overlay_wipe(A, cpi, ov)
        wipe_res[p] = (ph, amp, snr, cpi, rec_dt, ov)
        flag = " *" if p in GPS_III_PRNS else "  "
        print("  %3d   %7.4f   %7.2f   %5d   %s" % (p, amp, snr, ph, flag))

    giii = [p for p in ranked if p in GPS_III_PRNS]
    if not giii:
        print("\nNO GPS III satellite was tracked -- nothing here can be real L1C. Re-capture on a\n"
              "pass with PRN 04/11/14/18/23/28 high in the sky.")
        return
    target = giii[0]  # brightest GPS III sat = the one that can actually validate the wipe
    noise_prn = ranked[-1]
    print("\nanalysis target: PRN %d (brightest GPS III); noise ref: PRN %d" % (target, noise_prn))

    A, utc = tracked[target]
    ph_raw, amp_raw, snr_raw, cpi, rec_dt, ov = wipe_res[target]
    nominal_s = rec_dt if rec_dt > 0 else 10e-3

    # Perfect-FLL oracle: remove the slow carrier WANDER, THEN search the overlay phase on the
    # now-coherent stream. This is the OCXO-vs-software-seed fork. We must find the phase on the
    # DEROTATED A -- on the raw A a wandered carrier scrambles the coherent sum, so the blind
    # search picks a noise phase (that is the live failure mode, captured by ph_raw above).
    df, npair = fit_residual_doppler(A, utc, max_gap_s=1.5 * nominal_s)
    fll_win = max(50, int(round(0.5 / nominal_s)))      # ~0.5 s windows (10 ms records)
    A_der = derotate_wander(A, utc, fll_win, 1.5 * nominal_s)
    ph_der, amp_der, snr_der = overlay_wipe(A_der, cpi, ov)

    wiped = wipe_stream(A, cpi, ov, ph_raw)             # actual: live carrier, blind-search phase
    wiped_der = wipe_stream(A_der, cpi, ov, ph_der)     # ceiling: perfect FLL + its phase

    Ks = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 1800, 3600]
    raw_i = integrate(A, Ks)              # no wipe: coherent dies, incoherent = the pedestal
    wip_i = integrate(wiped, Ks)          # wiped (live carrier): builds only while raw-coherent
    der_i = integrate(wiped_der, Ks)      # wiped + perfect-FLL: the carrier-free ceiling
    nz_i = integrate(tracked[noise_prn][0], Ks)

    print("\n=== overlay phase: raw-search %d (snr %.1f)  vs  FLL-search %d (snr %.1f);  residual "
          "%+0.1f Hz (%d pairs) ===" % (ph_raw, snr_raw, ph_der, snr_der, df, npair))
    print("  (raw-search >> floor: live wipe already works. FLL-search >> raw-search: carrier is "
          "the wall -> OCXO/feed-forward.)")
    print("\n=== integration on PRN %d ===" % target)
    print("    K    t(s) | incoh   coh(noWipe) | WIPED coh   c/i | +FLL coh   SNR_coh   (floor)")
    for K in Ks:
        if K not in wip_i:
            continue
        t = K * nominal_s
        inc = raw_i[K]["incoh"]
        cnw = raw_i[K]["coh"]
        wc, wi = wip_i[K]["coh"], wip_i[K]["incoh"]
        dc = der_i[K]["coh"]
        fl = nz_i[K]["coh"] if K in nz_i else float("nan")
        tag = "  <- 10 ms period" if (K * nominal_s >= 0.01 and (K - 1) * nominal_s < 0.01) else ""
        print("  %4d  %5.2f | %6.3f   %6.3f     | %7.3f  %4.2f | %7.3f  %7.1f%s"
              % (K, t, inc, cnw, wc, (wc / wi if wi else 0), dc,
                 (dc / fl if fl == fl and fl else float("nan")), tag))
    print("  >> coh(noWipe) dies past the 10 ms period (overlay sign-flips cancel it); WIPED coh")
    print("     tracks incoh while phase-coherent. If WIPED caps but +FLL keeps climbing, the wall")
    print("     is carrier/clock (OCXO + feed-forward). If +FLL also builds for seconds, the live")
    print("     wall is just the blind 1800-phase search -> seed the phase from GPS time + short win.")

    # Seeded-vs-blind over a SHORT window: with the phase KNOWN (from GPS time), a short
    # coherence-matched window wipes cleanly; the blind 1800-search over the same short window is
    # degenerate (fewer records than phases). Use the FLL-search phase (the trustworthy one) and
    # the derotated stream so this isolates the search-vs-seed question from the carrier.
    short_K = max(50, int(round(5.0 / nominal_s)))      # ~5 s
    if len(A) >= short_K:
        seg, segc = A_der[:short_K], cpi[:short_K]
        seeded = abs(np.mean(wipe_stream(seg, segc, ov, ph_der)))  # known phase, no search
        bph, bamp, bsnr = overlay_wipe(seg, segc, ov)              # blind search, same short window
        print("\nshort %d-record (~%.0f s) window: seeded(known-phase) coh|A|=%.4f vs blind-search "
              "deep|A|=%.4f @phase %d (snr %.1f)" % (short_K, short_K * nominal_s, seeded,
                                                     bamp, bph, bsnr))
        print("  >> seeded >= blind (and the blind phase != %d) means: pin the overlay phase from "
              "GPS time, don't search it." % ph_der)

    # Epoch calibration for the live GPS-time seed: the overlay symbol index of record 0 is
    # (-phase) mod L; with the capture UTC of record 0 that fixes UTC -> overlay-index for the
    # pipeline. Use the FLL-search phase. The SAME record0 symbol (mod the small per-sat code
    # start) must reproduce across GPS III sats -- that cross-check validates the epoch.
    print("\nepoch calibration (for the live GPS-time seed), from the FLL-search phase:")
    print("  PRN %d phase %d  ->  record0 is overlay symbol %d  @ capture UTC %.6f"
          % (target, ph_der, (-ph_der) % L1CO_LEN, utc[0]))
    for p in giii[1:]:
        pA, putc = tracked[p]
        pcpi, _ = cpi_from_utc(putc)
        if pcpi is None:
            continue
        pA_der = derotate_wander(pA, putc, fll_win, 1.5 * nominal_s)
        pph, pamp, psnr = overlay_wipe(pA_der, pcpi, generate_l1co_code(p))
        print("    PRN %2d  FLL-phase %4d  deep_snr %5.1f  -> record0 symbol %4d"
              % (p, pph, psnr, (-pph) % L1CO_LEN))

    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ks = [k for k in Ks if k in wip_i]
            t = [k * nominal_s for k in ks]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(t, [raw_i[k]["incoh"] for k in ks], "--s", label="incoherent sqrt<|A|^2>")
            ax.plot(t, [raw_i[k]["coh"] for k in ks], ":x", label="coherent NO wipe")
            ax.plot(t, [wip_i[k]["coh"] for k in ks], "-o", label="WIPED coherent")
            ax.plot(t, [der_i[k]["coh"] for k in ks], "-^", label="WIPED + perfect-FLL")
            ax.plot(t, [nz_i[k]["coh"] for k in ks if k in nz_i], ":+", color="black",
                    label="noise floor (PRN %d)" % noise_prn)
            ax.axvline(0.010, color="r", ls=":", lw=1, label="10 ms primary period")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("integration time (s)"); ax.set_ylabel("|A|")
            ax.set_title("L1C-P PRN %d: overlay-wiped coherent integration" % target)
            ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout(); fig.savefig(args.png, dpi=110)
            print("\nwrote %s" % args.png)
        except Exception as e:
            print("\n(plot skipped: %s)" % e)


if __name__ == "__main__":
    main()
