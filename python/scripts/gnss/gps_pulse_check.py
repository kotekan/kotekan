#!/usr/bin/env python3
"""Parse an airspy_rx raw capture and decide, in the TIME domain, whether the elevated L2
floor is PULSED interference (radar -> pulse-blanking helps), CONTINUOUS/CW or broadband
(-> a front-end filter is the fix, blanking won't help), or just the receiver noise floor.
Also answers "is the rms floor analog or digitization noise?" from the raw ADC counts.

The autocorr VIEWER shows the integrated SPECTRUM, where a 1-2 us pulse spreads thin across
frequency and barely lifts the floor -- so pulses are ~invisible there even when present.
This looks at the raw samples in time, where a pulse is a short high-amplitude burst.

Capture (one real U16 grab serves both checks; matches kotekan's RAW front end best):
  airspy_rx -r /tmp/l2.u16 -f 1227.6 -a 2500000 -t 4 -b 1 -l 14 -m 15 -v 15 -n 50000000
  python3 python/scripts/gps_pulse_check.py /tmp/l2.u16            # L2
  # A/B: recapture at -f 1575.42 (L1, the clean band) and compare.
airspy real modes output ~2x the -a rate, so -a 2500000 -t 4 -> ~5 MSPS real (--fs 5e6, default).

--type matches airspy_rx -t: u16_real(4, default) int16_real(3) int16_iq(2) float32_iq(0)
float32_real(1). For the digitization/ENOB question use a *real* raw type (u16_real best).
"""
import argparse
import sys

import numpy as np

TYPES = {  # name -> (numpy dtype, is_complex, nominal DC offset)
    "u16_real": (np.uint16, False, 2048.0),
    "int16_real": (np.int16, False, 0.0),
    "int16_iq": (np.int16, True, 0.0),
    "float32_iq": (np.float32, True, 0.0),
    "float32_real": (np.float32, False, 0.0),
}


def excess_kurtosis(x):
    """Pearson excess kurtosis: 0 for Gaussian, > 0 for impulsive/peaky."""
    x = x - x.mean()
    v = np.mean(x * x)
    return float(np.mean(x ** 4) / (v * v) - 3.0) if v > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="airspy_rx capture file")
    ap.add_argument("--type", default="u16_real", choices=list(TYPES),
                    help="sample format (matches airspy_rx -t; default u16_real = -t 4)")
    ap.add_argument("--fs", type=float, default=5e6,
                    help="sample rate Hz (real modes ~= 2x the airspy -a rate; default 5e6)")
    ap.add_argument("--win", type=int, default=16,
                    help="short-time power window in samples (~pulse width; 16 = 3.2 us @ 5 MSPS)")
    ap.add_argument("--thresh", type=float, default=6.0,
                    help="pulse threshold in multiples of the median short-time power")
    ap.add_argument("--max-samples", type=int, default=60_000_000,
                    help="cap samples read (memory guard)")
    args = ap.parse_args()

    dtype, is_complex, dc = TYPES[args.type]
    raw = np.fromfile(args.file, dtype=dtype, count=args.max_samples)
    if raw.size == 0:
        sys.exit("empty / unreadable file")

    # Build a real-valued series for envelope work, plus keep raw for the ADC-level check.
    if is_complex:
        z = raw.astype(np.float64).view(np.complex128) if dtype == np.float64 else \
            raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
        env_inst = np.abs(z)                # instantaneous IQ envelope
        x = env_inst                        # treat |z| as the "amplitude" series
        nsamp = z.size
    else:
        x = raw.astype(np.float64) - dc     # DC-removed real samples (counts for u16)
        nsamp = x.size

    dur = nsamp / args.fs
    print("file: %s  (%s, %d samples, %.2f s @ %.1f MSPS)"
          % (args.file, args.type, nsamp, dur, args.fs / 1e6))

    # ---- ADC level / digitization check (most meaningful on raw real counts) ---------------
    if not is_complex:
        rms = float(np.sqrt(np.mean(x * x)))
        peak = float(np.max(np.abs(x)))
        q_lsb = 1.0 / np.sqrt(12.0)          # ideal 1-LSB quantization-noise rms
        rail_lo = float(np.mean(raw <= (1 if dtype == np.uint16 else -2047)))
        rail_hi = float(np.mean(raw >= (4094 if dtype == np.uint16 else 2046)))
        print("\nADC level (raw counts):")
        print("  rms %.2f  peak %.0f  crest %.1f  DC(raw mean) %.1f"
              % (rms, peak, peak / rms if rms else 0, float(raw.mean())))
        print("  rms = %.0fx the 1-LSB quantization floor (%.2f cts) -> %s"
              % (rms / q_lsb, q_lsb,
                 "ANALOG-noise limited (lots of headroom)" if rms / q_lsb > 4
                 else "approaching DIGITIZATION-limited"))
        print("  noise spans ~%.1f bits; rail fraction lo %.4f hi %.4f%s"
              % (np.log2(max(rms, 1e-9)) + 1, rail_lo, rail_hi,
                 "  <-- CLIPPING" if rail_lo + rail_hi > 1e-3 else ""))
        # tiny ASCII histogram of sample values
        lo, hi = np.percentile(x, [0.1, 99.9])
        h, edges = np.histogram(x[(x > lo) & (x < hi)], bins=21)
        hm = h.max() or 1
        print("  value histogram (%.0f..%.0f cts):" % (lo, hi))
        for c, e in zip(h, edges[:-1]):
            print("    %+7.0f |%s" % (e, "#" * int(round(40 * c / hm))))

    # ---- rms-vs-time timeline: is the floor STEADY or does it swing (rotating/intermittent
    #      radar)? The swing's low side is the "usable" window for opportunistic L2. ----------
    nchunk = min(120, max(10, int(dur / 0.1)))  # ~100 ms chunks
    clen = (x.size // nchunk)
    if clen > 16:
        cr = np.sqrt((x[:clen * nchunk] ** 2).reshape(nchunk, clen).mean(axis=1))
        rmn, rmd, rmx = float(cr.min()), float(np.median(cr)), float(cr.max())
        usable = float(np.mean(cr < 2.0 * rmn))   # within 3 dB of the quietest = "radar quiet"
        print("\nrms timeline (%d x %.0f ms chunks):  min %.0f  median %.0f  max %.0f  (swing %.1f dB)"
              % (nchunk, 1e3 * clen / args.fs, rmn, rmd, rmx, 20 * np.log10(rmx / max(rmn, 1e-9))))
        print("  %.0f%% of the time within 3 dB of the quietest (the usable L2 window)" % (100 * usable))
        cmax = cr.max() or 1
        glyph = " .:-=+*#%@"
        line = "".join(glyph[min(len(glyph) - 1, int(len(glyph) * v / cmax))] for v in cr)
        print("  |" + line + "|  (each col %.0f ms)" % (1e3 * clen / args.fs))
        if rmx / max(rmn, 1e-9) > 3.0:
            print("  -> floor SWINGS: intermittent/rotating radar. L2 is usable in the dips --")
            print("     gate acquisition on a low rms, and lean on rolling integration there.")

    # ---- short-time power envelope + pulse detection ---------------------------------------
    W = max(1, args.win)
    ntrim = (x.size // W) * W
    p = (x[:ntrim] ** 2).reshape(-1, W).mean(axis=1)   # power per W-sample window
    env_fs = args.fs / W
    med = float(np.median(p))
    pk = float(np.max(p))
    duty = float(np.mean(p > args.thresh * med))
    kx = excess_kurtosis(x[:ntrim])                    # impulsiveness of the samples
    kp = excess_kurtosis(p)                             # impulsiveness of the power envelope
    print("\nshort-time power (window %d samp = %.2f us):" % (W, 1e6 * W / args.fs))
    print("  median %.3g  peak %.3g  peak/median %.0fx" % (med, pk, pk / med if med else 0))
    print("  duty cycle above %.0fx median: %.3f%%   (= blanking loss if pulsed)"
          % (args.thresh, 100 * duty))
    print("  excess kurtosis: samples %.2f  envelope %.2f  (0 = Gaussian; >>0 = impulsive)"
          % (kx, kp))

    # ---- PRF: autocorrelate the (median-subtracted) envelope, look for 50-2000 Hz periodicity
    e = p - med
    e = e[: 1 << int(np.floor(np.log2(e.size)))]       # pow2 for a fast FFT
    if e.size >= 1024:
        f = np.fft.rfft(e * np.hanning(e.size))
        ac = np.fft.irfft(np.abs(f) ** 2)
        ac = ac[: ac.size // 2]
        lo_lag = max(1, int(env_fs / 2000.0))          # 2000 Hz
        hi_lag = min(ac.size - 1, int(env_fs / 50.0))  # 50 Hz
        if hi_lag > lo_lag + 2 and ac[0] > 0:
            seg = ac[lo_lag:hi_lag]
            k = int(np.argmax(seg)) + lo_lag
            prf = env_fs / k
            strength = float(ac[k] / ac[0])            # 0..1, periodicity contrast
            print("\nstrongest periodicity 50-2000 Hz: PRF ~ %.0f Hz  (contrast %.2f)%s"
                  % (prf, strength,
                     "  <-- looks PULSED at a radar PRF" if strength > 0.05 else
                     "  (weak -> not strongly periodic)"))

    # ---- in-channel spectrum shape: is the floor a few NARROW carriers (-> notch), flat
    #      BROADBAND (-> co-channel, nothing within 2.5 MHz helps), or EDGE-heavy (-> adjacent
    #      leakage a sharper front-end filter would cut)? This picks the L2 mitigation. --------
    NF = 4096
    nseg = min(400, (nsamp if is_complex else x.size) // NF)
    if nseg >= 4:
        win = np.hanning(NF)
        acc = np.zeros((NF if is_complex else NF // 2 + 1), dtype=np.float64)
        src = z if is_complex else x
        for s in range(nseg):
            seg = src[s * NF:(s + 1) * NF]
            sp = np.fft.fftshift(np.fft.fft(seg * win)) if is_complex else np.fft.rfft(seg * win)
            acc += np.abs(sp) ** 2
        psd = acc / nseg
        psd_db = 10 * np.log10(psd / psd.max() + 1e-12)
        med = float(np.median(psd))
        # spectral flatness: geomean/mean (1 = white/broadband, <<1 = peaky/tonal)
        flat = float(np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12))
        ncar = int(np.sum(psd > 10 * med))         # bins >10x median = discrete carriers
        nbin = psd.size
        # power split: how much sits in those carrier bins vs the broadband floor
        carrier_frac = float(psd[psd > 10 * med].sum() / psd.sum())
        # edge vs centre (real: 0..Fs/2; iq: -Fs/2..Fs/2) -- leakage piles up off-centre
        q = nbin // 4
        edge = float((psd[:q].mean() + psd[-q:].mean()) / 2)
        cen = float(psd[q:-q].mean())
        print("\nin-channel spectrum (%d-pt, %d avgs):" % (NF, nseg))
        print("  spectral flatness %.3f (1=broadband/noise-like, <0.2=tonal); "
              "%d carrier bins (>10x med) holding %.1f%% of power" % (flat, ncar, 100 * carrier_frac))
        edge_db = 10 * np.log10(edge / cen + 1e-12)
        edge_desc = ("EDGE-heavy -> adjacent-band leakage; a sharper front-end filter would help"
                     if edge > 2 * cen else
                     "CENTRE-filled (in-band block, edges are the airspy channel filter)"
                     if edge < 0.5 * cen else "fairly uniform across the band")
        print("  edge/centre power %.1f dB (%s)" % (edge_db, edge_desc))
        # ASCII spectrum
        cols = 68
        step = max(1, nbin // cols)
        prof = psd_db[: (nbin // step) * step].reshape(-1, step).max(axis=1)
        lo = max(-40.0, float(prof.min()))
        for i in range(0, len(prof), max(1, len(prof) // 16)):
            v = prof[i]
            bar = int(round((v - lo) / (0.0 - lo) * 40)) if v > lo else 0
            fghz = (i * step) / nbin * (args.fs if is_complex else args.fs / 2) / 1e6
            print("    %+5.1f MHz %6.1f dB |%s" % (fghz - (args.fs / 2e6 if is_complex else 0),
                                                   v, "#" * bar))
        # Mitigation decision tree (keyed to the recoverable failure modes):
        if carrier_frac > 0.3:
            print("  -> TONAL: discrete carriers hold %.0f%% of power; a DSP NOTCH recovers most."
                  % (100 * carrier_frac))
        elif edge > 2 * cen:
            print("  -> EDGE/LEAKAGE: power piles up off-centre = strong adjacent-band signals")
            print("     leaking through the soft decimation; a sharper front-end SAW would help.")
        else:
            print("  -> BROADBAND CO-CHANNEL: no lines to notch, power fills the in-band passband.")
            print("     Nothing within the GPS band (notch/filter) recovers it -- it overlaps the")
            print("     signal. Only spatial nulling, a quieter time/site, or another band helps.")

    # ---- ASCII envelope strip over the first ~20 ms so pulses are visible -------------------
    span = min(p.size, int(0.02 * env_fs))
    if span > 40:
        strip = p[:span]
        cols = 70
        chunk = max(1, span // cols)
        bars = strip[: (span // chunk) * chunk].reshape(-1, chunk).max(axis=1)
        bmax = bars.max() or 1
        glyph = " .:-=+*#%@"
        line = "".join(glyph[min(len(glyph) - 1, int(len(glyph) * b / bmax))] for b in bars)
        print("\nenvelope, first %.0f ms (each col %.0f us, peak-held):" % (1e3 * span / env_fs,
                                                                            1e6 * chunk / env_fs))
        print("  |" + line + "|")

    # ---- verdict ---------------------------------------------------------------------------
    print("\nverdict:")
    pulsed = (kx > 0.7 or kp > 3.0) and duty < 0.30 and pk / max(med, 1e-30) > 8
    if pulsed:
        print("  PULSED interference (impulsive, low duty) -> pulse-blanking should help; it")
        print("  would excise ~%.2f%% of samples. Blank above ~%.0fx median power." % (100 * duty,
                                                                                       args.thresh))
    elif kx < 0.3:
        print("  GAUSSIAN / continuous -- no pulses. The elevated power (if any) is CW or")
        print("  broadband; pulse-blanking won't help. A front-end band filter is the fix")
        print("  (or, if the level just tracks the band, it's benign tuner gain).")
    else:
        print("  MIXED / mildly impulsive -- inspect the envelope strip and PRF above before")
        print("  committing to blanking.")


if __name__ == "__main__":
    main()
